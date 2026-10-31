"""ハイパーパラメータと実行設定。

設定は dataclass で1か所にまとめ、コマンドライン引数はフィールドから自動で作る（parse_config）。
フィールド名の `_` は引数では `-` になる（例: chunk_size → --chunk-size）。
`uv run python -m pai_act.exps.train --help` で一覧と既定値が見られる。

学習の設定（TrainConfig）はチェックポイントに丸ごと保存し、評価時にそこからポリシーを組み立て直す。
"""

import argparse
import typing
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def arg(default=MISSING, help: str = ""):
    """dataclass のフィールドに --help 用の説明を付けるための短縮形。default 省略で必須引数になる。"""
    return field(default=default, metadata={"help": help})


@dataclass
class TrainConfig:
    """学習の設定。既定値は ACT 原著の sim 実験（transfer cube）に合わせられるものは合わせている。"""

    name: str = arg("baseline", "実行名。出力は runs/<name>/ に保存される")
    policy: str = arg("baseline", "ポリシーの種類（exps/checkpoint.py の build_policy を参照）")
    task: str = arg("transfer_cube", "タスク名（exps/sim.py の ENV_IDS のキー）")

    # --- モデル ---
    chunk_size: int = arg(100, "行動チャンクの長さ k（1回の推論で何ステップ先まで出すか）")
    hidden_dim: int = arg(512, "baseline: MLP の中間層の幅")
    d_model: int = arg(512, "act / diffusion / flow_matching: Transformer のトークンの次元数")
    num_heads: int = arg(8, "act / diffusion / flow_matching: attention の head 数")
    dim_feedforward: int = arg(3200, "act / diffusion / flow_matching: FFN の中間層の幅")
    num_encoder_layers: int = arg(4, "act / diffusion / flow_matching: 観測側の Encoder の層数")
    num_decoder_layers: int = arg(
        1,
        "act / diffusion / flow_matching: Decoder の層数（ACT 公式実装は実質1層。policies/act.py 参照）",
    )
    num_vae_encoder_layers: int = arg(4, "act: CVAE エンコーダの層数")
    latent_dim: int = arg(32, "act: 潜在変数 z の次元数")
    dropout: float = arg(0.1, "act / diffusion / flow_matching: dropout の確率")
    kl_weight: float = arg(10.0, "act: 損失の KL 項の重み β")
    learned_query_pos: bool = arg(
        True,
        "act: 行動クエリの位置ベクトルを学習させる（公式実装）。--no-learned-query-pos で"
        "論文どおりの固定 sinusoidal",
    )
    num_diffusion_steps: int = arg(100, "diffusion: 拡散のステップ数 T（学習時のノイズの段階数）")
    sampler: str = arg(
        "ddim", "diffusion: 推論のサンプラー（ddpm / ddim）。評価時に --sampler で上書きできる"
    )
    num_inference_steps: int = arg(
        10,
        "diffusion: DDIM の推論ステップ数 S（T の約数）/ flow_matching: ODE を解くステップ数 N"
        "（heun はデノイザを 2N 回呼ぶ）。評価時に上書きできる",
    )
    solver: str = arg(
        "euler",
        "flow_matching: 推論の ODE ソルバー（euler / heun）。評価時に --solver で上書きできる",
    )
    x0_clip: float = arg(
        6.0,
        "diffusion: サンプリング中の x̂_0 を [-この値, この値] にクリップする"
        "（正規化後の単位。0 でクリップしない）。"
        "評価時に上書きできる",
    )
    ddim_eta: float = arg(
        0.0,
        "diffusion: DDIM で毎ステップ足す乱数の大きさ η（0〜1。0 で決定的）。評価時に上書きできる",
    )

    # --- 最適化 ---
    steps: int = arg(20000, "学習ステップ数（パラメータ更新の回数）")
    batch_size: int = arg(8, "ミニバッチの大きさ")
    lr: float = arg(1e-4, "バックボーン以外の学習率")
    lr_backbone: float = arg(1e-5, "ResNet18 の学習率（事前学習済みなので小さくして壊さない）")
    weight_decay: float = arg(1e-4, "AdamW の重み減衰")
    grad_clip: float = arg(10.0, "勾配ノルムの上限（0 で無効）")
    amp: bool = arg(True, "bfloat16 の自動混合精度で計算を速くする")

    # --- データ ---
    num_val_episodes: int = arg(5, "検証用に取り分けるエピソード数（残りを学習に使う）")
    val_stride: int = arg(10, "検証では各エピソードの何フレームおきにサンプルを取るか")
    num_workers: int = arg(8, "DataLoader のワーカープロセス数")

    # --- ログ・保存 ---
    log_every: int = arg(50, "何ステップごとに学習損失を記録するか")
    print_every: int = arg(
        0,
        "何ステップごとに学習損失の区間平均を端末に表示するか（0: steps の 1/10、上限 val_every）",
    )
    val_every: int = arg(1000, "何ステップごとに検証損失を計算するか")
    save_every: int = arg(5000, "何ステップごとにチェックポイントを保存するか")
    seed: int = arg(0, "乱数シード（エピソード分割・初期化・サンプル順）")
    # Path ではなく str: 設定は config.json とチェックポイントにそのまま保存されるため
    # （Path は JSON にできず、torch.load(weights_only=True) でも読めない）
    resume: str | None = arg(
        None, "学習を再開する Lightning のチェックポイント（runs/<name>/lightning/last.ckpt）"
    )
    wandb_project: str = arg("pai-act", "wandb のプロジェクト名")
    wandb_mode: str = arg(
        "online",
        "wandb の動作: online（送信する）/ offline（手元に保存のみ。後で wandb sync で送れる）"
        " / disabled（記録しない）",
    )

    @property
    def out_dir(self) -> Path:
        return Path("runs") / self.name


@dataclass
class EvalConfig:
    """評価（シミュレータでのロールアウト）の設定。"""

    ckpt: Path = arg(help="評価するチェックポイント（runs/<name>/ckpt_*.pt）")
    num_episodes: int = arg(50, "ロールアウトするエピソード数")
    seed: int = arg(
        0, "最初のエピソードのシード。エピソード i はシード seed+i（キューブ位置が決まる）"
    )
    max_steps: int = arg(400, "1エピソードの最大ステップ数（デモと同じ 400 = 8秒）")
    temporal_ensemble: bool = arg(
        True, "毎ステップ推論し、重なった予測を重み付き平均する（ACT の temporal ensembling）"
    )
    ensemble_coef: float = arg(0.01, "temporal ensembling の重み exp(-m*i) の m")
    query_every: int = arg(0, "ensembling 無効時、何ステップごとに推論するか（0 でチャンク長 k）")
    num_videos: int = arg(5, "先頭から何エピソード分の動画を保存するか")
    save_failed_videos: bool = arg(
        True, "失敗したエピソードは num_videos に関わらず全て動画を保存する（失敗の仕方を見るため）"
    )
    sampler: str | None = arg(
        None, "diffusion: サンプラーを上書きする（ddpm / ddim。省略時は学習時の設定）"
    )
    num_inference_steps: int | None = arg(
        None,
        "diffusion / flow_matching: 推論ステップ数を上書きする（省略時は学習時の設定。"
        "diffusion は DDIM のみに効く）",
    )
    solver: str | None = arg(
        None, "flow_matching: ODE ソルバーを上書きする（euler / heun。省略時は学習時の設定）"
    )
    x0_clip: float | None = arg(
        None,
        "diffusion: x̂_0 のクリップの値を上書きする（0 でクリップしない。省略時は学習時の設定）",
    )
    ddim_eta: float | None = arg(
        None,
        "diffusion: DDIM の η を上書きする（0〜1。省略時は学習時の設定。DDIM のみに効く）",
    )
    out_dir: Path | None = arg(None, "出力先（省略時は <ckptのフォルダ>/eval_<ckpt名>/）")


def parse_config[T: DataclassInstance](cls: type[T], argv: list[str] | None = None) -> T:
    """dataclass の各フィールドからコマンドライン引数を作り、解析結果で cls のインスタンスを作る。

    対応する型: int / float / str / Path / bool（--flag / --no-flag）/ 上記の X | None
    """
    parser = argparse.ArgumentParser(
        description=cls.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for f in fields(cls):
        typ = f.type
        if typing.get_origin(typ) is not None:  # X | None → X
            typ = next(t for t in typing.get_args(typ) if t is not type(None))
        flag = "--" + f.name.replace("_", "-")
        kwargs: dict[str, Any] = {"dest": f.name, "help": f.metadata.get("help", "")}
        if f.default is MISSING:
            kwargs["required"] = True
        else:
            kwargs["default"] = f.default
        if typ is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, **kwargs)
        else:
            parser.add_argument(flag, type=typ, **kwargs)
    return cls(**vars(parser.parse_args(argv)))
