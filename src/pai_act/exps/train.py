"""学習ループ（全ポリシーで共通）。PyTorch Lightning で進行・ログ・保存を管理する。

使い方（リポジトリ直下で実行）:
    bash scripts/train.sh                          # 既定の設定で学習
    bash scripts/train.sh --steps 5000 --name test # 設定を上書き
    bash scripts/train.sh --resume runs/baseline/lightning/last.ckpt   # 中断した学習を再開
    uv run python -m pai_act.exps.train --help     # 設定項目の一覧と既定値

流れ:
    1. データセットを読み、エピソード単位で学習用 / 検証用に分ける（AlohaDataModule）
       （フレーム単位で分けると、同じデモの隣のフレームが両方に入って検証にならない）
    2. 学習用エピソードから正規化の統計量を求め、ポリシーに設定する
    3. ミニバッチを取り出す → policy.compute_loss → 逆伝播 → パラメータ更新 を steps 回繰り返す
       （PolicyModule.training_step。逆伝播・更新・混合精度・勾配クリップは Lightning の Trainer が行う）
    4. 定期的に、損失を wandb に記録・検証損失を計算・チェックポイントを保存する

Lightning の役割分担:
    ポリシー（policies/）は素の PyTorch の nn.Module のまま。PolicyModule はそれを包んで
    「1ステップで何を計算するか」と「optimizer の作り方」だけを Lightning に教える薄い殻で、
    ポリシーの中身は Lightning に依存しない（評価・ROS2 では Lightning なしで読み込める）。

出力（runs/<name>/）:
    config.json          … 学習設定
    wandb/               … wandb のローカルファイル（学習曲線は wandb の Web 画面で見る）
    ckpt_XXXXXX.pt       … 途中のチェックポイント（ポリシーの重みと設定のみ。評価に使う）
    ckpt_last.pt         … 最終ステップのチェックポイント
    lightning/last.ckpt  … 学習再開用（optimizer の状態なども含む。--resume に渡す）

検証損失はあくまで目安。模倣学習では「損失が下がる = シミュレータで成功する」とは限らないので
（小さな誤差が積み重なって軌道が外れる compounding error があるため）、最終的な判断は
exps/evaluate.py のシミュレータ上の成功率で行う。
"""

import json
from dataclasses import asdict

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset

from pai_act.data.chunk_dataset import ChunkDataset
from pai_act.data.dataset import AlohaDataset
from pai_act.exps.checkpoint import build_policy, save_checkpoint
from pai_act.exps.config import TrainConfig, parse_config
from pai_act.policies.base import Policy

DATASETS = {"transfer_cube": "lerobot/aloha_sim_transfer_cube_human"}


def split_episodes(n: int, num_val: int, seed: int) -> tuple[list[int], list[int]]:
    """エピソード番号 0..n-1 をシャッフルし、(学習用, 検証用) に分ける。"""
    perm = np.random.default_rng(seed).permutation(n).tolist()
    return sorted(perm[num_val:]), sorted(perm[:num_val])


def make_optimizer(policy: Policy, cfg: TrainConfig) -> torch.optim.Optimizer:
    """バックボーンとそれ以外で学習率を分けた AdamW を作る。"""
    backbone, others = [], []
    for name, p in policy.named_parameters():
        if p.requires_grad:
            (backbone if name.startswith("backbone.") else others).append(p)
    return torch.optim.AdamW(
        [{"params": others, "lr": cfg.lr}, {"params": backbone, "lr": cfg.lr_backbone}],
        weight_decay=cfg.weight_decay,
    )


class AlohaDataModule(L.LightningDataModule):
    """データセットの読み込み、エピソード単位の分割、DataLoader の作成をまとめる。"""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.source: AlohaDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        # Trainer の fit / validate のたびに呼ばれるが、読み込みと分割は1回で済ませる
        if self.source is not None:
            return
        cfg = self.cfg
        self.source = AlohaDataset(DATASETS[cfg.task])
        self.train_eps, self.val_eps = split_episodes(
            len(self.source), cfg.num_val_episodes, cfg.seed
        )
        print(f"episodes: train {len(self.train_eps)} / val {self.val_eps}")
        self.train_set = ChunkDataset(self.source, self.train_eps, cfg.chunk_size)
        val_set = ChunkDataset(self.source, self.val_eps, cfg.chunk_size)
        self.val_set = Subset(val_set, range(0, len(val_set), cfg.val_stride))

    def train_frames(self) -> np.ndarray:
        """学習用エピソードに含まれる全フレームの番号（正規化の統計量を求めるのに使う）。"""
        assert self.source is not None, "setup() を先に呼ぶ"
        return np.concatenate([np.arange(*self.source.ranges[ep]) for ep in self.train_eps])

    def train_dataloader(self) -> DataLoader:
        # エポック（データ1周）ごとにシャッフルし直して回り続け、Trainer の max_steps で止まる
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.cfg.num_workers > 0,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )


class PolicyModule(L.LightningModule):
    """Policy を Lightning の学習ループに載せるための薄い殻。

    Lightning が代わりにやってくれること（素の PyTorch では自分で書いていたもの）:
        - batch の GPU への転送、policy.train() / policy.eval() の切り替え
        - zero_grad → backward → optimizer.step
        - 混合精度（precision="bf16-mixed" で、区間を torch.autocast で囲むのと同じ）
        - 検証時の torch.no_grad と、バッチの大きさで重み付けした検証損失の平均
    """

    def __init__(self, policy: Policy, cfg: TrainConfig):
        super().__init__()
        # 属性名は policy。パラメータ名は "policy.backbone...." になるが、make_optimizer と
        # 保存（PolicyCheckpoint）には self.policy を渡すので、ポリシー側の約束はそのまま使える
        self.policy = policy
        self.cfg = cfg
        # 学習再開用の .ckpt と wandb の config に学習設定を残す
        self.save_hyperparameters(asdict(cfg))

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self.policy.compute_loss(batch)
        bs = len(batch["state"])
        # on_step: 毎ステップ記録する（実際に wandb に送るのは Trainer の log_every_n_steps ごと）
        self.log_dict({f"train/{k}": v for k, v in out.items()}, on_step=True, batch_size=bs)
        self.log("loss", out["loss"], prog_bar=True, logger=False, batch_size=bs)
        # dict のまま返す: Lightning は "loss" で逆伝播し、dict 全体を on_train_batch_end の
        # outputs に渡す（LossPrinter がログ用の値も区間平均できるように）
        return out

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.policy.compute_loss(batch)
        # on_epoch: 検証データ全体で、バッチの大きさで重み付けした平均を記録する
        self.log_dict(
            {f"val/{k}": v for k, v in out.items()}, on_epoch=True, batch_size=len(batch["state"])
        )

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:  # 学習前の動作確認（数バッチだけ）の結果は表示しない
            return
        val = {
            k: v.item() for k, v in self.trainer.callback_metrics.items() if k.startswith("val/")
        }
        self.print(
            f"step {self.global_step}: val " + " ".join(f"{k}={v:.4f}" for k, v in val.items())
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return make_optimizer(self.policy, self.cfg)

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ) -> None:
        """勾配が一時的に大きくなっても更新幅が暴れないよう、勾配全体のノルムを上限で抑える。

        Trainer の gradient_clip_val でも同じことができるが、それだとクリップ前のノルムが
        記録されないので、自分でクリップしてノルムをログに残す。
        """
        if self.cfg.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip)
            self.log("train/grad_norm", grad_norm, on_step=True)


class PolicyCheckpoint(Callback):
    """評価用のチェックポイント（exps/checkpoint.py の形式）を save_every ステップごとと最後に保存する。

    Lightning の .ckpt は学習再開用で optimizer の状態まで含み、読み込みに Lightning が要る。
    評価（exps/evaluate.py）はポリシーの重みと設定だけを持つ .pt を使う。
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step  # 実行済みの optimizer.step の回数（1始まりのステップ番号）
        if step % self.cfg.save_every == 0:
            path = self.cfg.out_dir / f"ckpt_{step:06d}.pt"
            save_checkpoint(path, pl_module.policy, self.cfg, step)

    def on_train_end(self, trainer, pl_module) -> None:
        path = self.cfg.out_dir / "ckpt_last.pt"
        save_checkpoint(path, pl_module.policy, self.cfg, trainer.global_step)
        print(f"saved -> {path}")


class LossPrinter(Callback):
    """学習損失の区間平均を print_every ステップごとに端末へ表示する。

    進捗バーの loss は最新の1ステップの値で、学習後には最後の値しか残らない。1ステップの損失は
    ミニバッチ（Diffusion ではさらに拡散の時刻 t）の選び方でばらつくので、区間で平均して並べないと
    下がっているかが分からない。wandb を使わない試走（--wandb-mode disabled）でも確かめられるようにする。
    """

    def __init__(self, cfg: TrainConfig):
        # 0 なら学習全体を約 10 区間に分ける（試走 300 ステップ → 30 ごと、本番 80k → val_every ごと）
        self.every = cfg.print_every or max(1, min(cfg.val_every, cfg.steps // 10))
        self.sums: dict[str, float] = {}
        self.count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        for k, v in outputs.items():
            self.sums[k] = self.sums.get(k, 0.0) + v.item()
        self.count += 1
        if trainer.global_step % self.every == 0:
            avg = " ".join(f"{k}={s / self.count:.4f}" for k, s in self.sums.items())
            pl_module.print(
                f"step {trainer.global_step}: train {avg}（直近 {self.count} ステップの平均）"
            )
            self.sums, self.count = {}, 0


class StepProgressBar(TQDMProgressBar):
    """学習全体（global_step / max_steps）の進み具合を1本のバーで表示する。

    既定のバーはエポック（データ1周、約2千ステップ）ごとに作り直されて、学習全体の残り時間が出ない。
    """

    def on_train_start(self, *args) -> None:
        # 親クラスの引数の形 (*_) に合わせ、trainer は self.trainer から取る
        super().on_train_start(*args)
        bar = self.train_progress_bar
        bar.reset(total=self.trainer.max_steps)
        # 再開時は途中のステップから数え始める（initial を揃えると it/s も正しく出る）
        bar.initial = bar.n = self.trainer.global_step
        bar.set_description("train")

    def on_train_epoch_start(self, trainer, *_) -> None:
        pass  # エポックの区切りでバーを作り直さない

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        bar = self.train_progress_bar
        if trainer.global_step % self.refresh_rate == 0:
            bar.set_postfix(self.get_metrics(trainer, pl_module), refresh=False)
        bar.update(trainer.global_step - bar.n)


def train(cfg: TrainConfig) -> None:
    L.seed_everything(cfg.seed)
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    # --- 1. データ ---
    data = AlohaDataModule(cfg)
    data.setup()
    assert data.source is not None

    # --- 2. ポリシーと正規化 ---
    # 再開時は .ckpt の値で上書きされるが、同じ学習用エピソードから求めるので値は変わらない
    policy = build_policy(cfg)
    train_frames = data.train_frames()
    policy.state_norm.fit(data.source.state[train_frames])
    policy.action_norm.fit(data.source.action[train_frames])
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"policy: {cfg.policy} ({n_params / 1e6:.1f}M params)")
    module = PolicyModule(policy, cfg)

    # API キーは環境変数 WANDB_API_KEY から読まれる（.devcontainer/devcontainer.json の remoteEnv）
    logger = WandbLogger(
        project=cfg.wandb_project, name=cfg.name, save_dir=out_dir, mode=cfg.wandb_mode
    )
    logger.experiment.summary["n_params"] = n_params

    # --- 3. 学習ループ / 4. ログ・検証・保存 ---
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        # エポック数ではなくステップ数で学習を区切る
        max_steps=cfg.steps,
        max_epochs=-1,
        # autocast: 行列積・畳み込みを bfloat16 で計算する（速く省メモリ。勾配は float32 で持つ）
        precision="bf16-mixed" if cfg.amp else "32-true",
        # 入力サイズが毎回同じなので、cuDNN に最速の畳み込みアルゴリズムを選ばせる
        benchmark=True,
        log_every_n_steps=cfg.log_every,
        # check_val_every_n_epoch=None: エポックの区切りと関係なく、通算 val_every ステップごとに検証
        val_check_interval=cfg.val_every,
        check_val_every_n_epoch=None,
        logger=logger,
        callbacks=[
            StepProgressBar(),
            LossPrinter(cfg),
            PolicyCheckpoint(cfg),
            ModelCheckpoint(
                dirpath=out_dir / "lightning",
                every_n_train_steps=cfg.save_every,
                save_top_k=0,
                save_last=True,
            ),
        ],
        default_root_dir=out_dir,
    )
    trainer.fit(module, datamodule=data, ckpt_path=cfg.resume)
    # 検証は val_every の倍数のステップでしか走らないので、最終ステップの検証損失を補う
    if cfg.steps % cfg.val_every != 0:
        trainer.validate(module, datamodule=data, verbose=False)
    logger.experiment.finish()


if __name__ == "__main__":
    train(parse_config(TrainConfig))
