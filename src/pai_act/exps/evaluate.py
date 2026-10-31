"""評価ハーネス: シミュレータ上でのロールアウト、成功率の算出、動画保存（全ポリシーで共通）。

使い方（リポジトリ直下で実行）:
    bash scripts/evaluate.sh                                         # runs/baseline/ckpt_last.pt を評価
    bash scripts/evaluate.sh --ckpt runs/xxx/ckpt_last.pt --num-episodes 10 --no-temporal-ensemble
    uv run python -m pai_act.exps.evaluate --help                    # 設定項目の一覧と既定値
出力は既定で runs/<name>/eval_<ckpt名>/ に保存される（results.json と videos/*.mp4）。
results.json の各項目・報酬・失敗の型の定義は docs/EVALUATION.md にまとめてある。

ロールアウト（rollout）とは、学習済みポリシーを閉ループで動かすこと:
    観測 → ポリシーが行動チャンクを予測 → ChunkExecutor が今の行動を選ぶ → シミュレータを1ステップ進める → 観測 → ...
学習時は常に「デモの正しい観測」を入力にしていたが、ロールアウトでは自分の行動の結果が次の入力になる。
小さな誤差で軌道がデモから外れると、見たことのない観測に対して行動することになり、誤差が積み重なる。
そのため、学習の損失だけでなく、この閉ループでの成功率で性能を測る。

transfer cube の報酬（gym-aloha の定義。毎ステップその瞬間の接触から決まり、下の条件のうち満たす最も大きい値になる。
エピソード中の最大値で「どこまで進んだか」が分かる）:
    0: 右グリッパーの指がキューブに触れていない
    1: 右グリッパーの指がキューブに触れている
    2: 右グリッパーの指が触れていて、キューブがテーブルに触れていない（持ち上げた）
    3: 左グリッパーの指がキューブに触れている（受け渡しを試みた）
    4: 左グリッパーの指が触れていて、キューブがテーブルに触れていない = 成功（ここでエピソードを終了する）

評価条件: エピソード i はシード seed+i でリセットする（キューブの初期位置がシードで決まる）。
同じシード列を使えば、異なるポリシーを同じ初期配置の集合で比較できる。
"""

import json
import time
from functools import partial
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from pai_act.exps.checkpoint import load_checkpoint
from pai_act.exps.config import EvalConfig, parse_config
from pai_act.exps.inference import ChunkExecutor
from pai_act.exps.sim import make_env
from pai_act.policies.base import Policy

SUCCESS_REWARD = 4
REWARD_LABELS = ["none", "touched", "lifted", "transfer attempted", "success"]
# 失敗したエピソードの型。最大報酬 r の次の段階 r+1 に進めなかった、という意味で名付ける
FAILURE_TYPES = ["no_touch", "not_lifted", "not_transferred", "transfer_failed"]
# diffusion / flow_matching の推論の設定。評価時に上書きでき、results.json に実際に使った値を残す
# （ポリシーが持っているものだけ。diffusion は sampler / x0_clip、flow_matching は solver を持つ）
SAMPLER_OPTIONS = ("sampler", "solver", "num_inference_steps", "x0_clip", "ddim_eta")


@torch.no_grad()
def predict(
    policy: Policy, obs: dict, device: torch.device, timings: list[float] | None = None
) -> np.ndarray:
    """gym-aloha の観測1つからチャンク (k, 14) を予測する。

    シミュレータの観測を、学習バッチと同じ形・型（バッチ次元1、画像は (3, H, W) uint8、状態は float32）に揃えて渡す。
    timings を渡すと、predict_chunk 1回にかかった秒数を追記する（制御周期 50Hz = 20ms に収まるかを見るため）。
    """
    image = torch.from_numpy(obs["pixels"]["top"]).permute(2, 0, 1)[None].to(device)
    state = torch.from_numpy(obs["agent_pos"]).float()[None].to(device)
    start = time.perf_counter()
    chunk = policy.predict_chunk({"image": image, "state": state})
    if timings is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # GPU の計算は非同期なので、終わるのを待ってから測る
        timings.append(time.perf_counter() - start)
    return chunk[0].float().cpu().numpy()


def annotate(frame: np.ndarray, text: str) -> np.ndarray:
    """動画のフレーム左上に文字を書き込む（どのエピソードの何ステップ目かを動画で分かるように）。"""
    im = Image.fromarray(frame)
    ImageDraw.Draw(im).text(
        (8, 8), text, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0)
    )
    return np.asarray(im)


def rollout(
    env,
    policy: Policy,
    executor: ChunkExecutor,
    seed: int,
    device: torch.device,
    record: bool,
    timings: list[float] | None = None,
) -> dict:
    """1エピソードを実行する。

    Returns:
        success: 成功したか / max_reward: エピソード中の最大報酬（0〜4）/ steps: 実行したステップ数
        frames: record=True のとき、各ステップのカメラ画像 (H, W, 3) の list（長さ T+1）
    """
    obs, _ = env.reset(seed=seed)
    executor.reset()
    frames, max_reward, steps = [], 0, 0
    while True:
        if record:
            frames.append(
                annotate(obs["pixels"]["top"], f"seed {seed}  t={steps}  reward={max_reward}")
            )
        # 推論が必要な時刻にだけ predict が呼ばれる（partial で現在の観測を渡しておく）
        action = executor.act(partial(predict, policy, obs, device, timings))
        obs, reward, terminated, truncated, _ = env.step(action.astype(np.float64))
        max_reward = max(max_reward, int(reward))
        steps += 1
        # terminated: 成功してエピソードが終わった / truncated: 最大ステップ数に達した
        if terminated or truncated:
            break
    if record:
        result = "SUCCESS" if max_reward == SUCCESS_REWARD else "FAIL"
        frames.append(annotate(obs["pixels"]["top"], f"seed {seed}  t={steps}  {result}"))
    return {
        "success": max_reward == SUCCESS_REWARD,
        "max_reward": max_reward,
        "steps": steps,
        # np.stack すると 1 エピソード約 370MB を丸ごと複製するので、list のまま書き出しに渡す
        "frames": frames if record else None,
    }


def evaluate(policy: Policy, cfg: EvalConfig, task: str, out_dir: Path) -> dict:
    """cfg.num_episodes 回ロールアウトし、成功率を計算して結果と動画を out_dir に保存する。

    出力:
        results.json                          … 設定・成功率・エピソードごとの結果
        videos/ep000_seed0_success.mp4 など    … 先頭 cfg.num_videos エピソードの動画
        videos/ep002_seed2_fail_not_lifted.mp4 など … 失敗したエピソードの動画
                                                （cfg.save_failed_videos のとき全て。末尾は失敗の型）
    """
    device = next(policy.parameters()).device
    env = make_env(task, max_episode_steps=cfg.max_steps)
    executor = ChunkExecutor(
        chunk_size=policy.chunk_size,
        query_every=cfg.query_every or policy.chunk_size,
        temporal_ensemble=cfg.temporal_ensemble,
        ensemble_coef=cfg.ensemble_coef,
    )
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)

    episodes, timings = [], []
    pbar = tqdm(range(cfg.num_episodes), desc="eval")
    for i in pbar:
        seed = cfg.seed + i
        # 失敗するかは終わるまで分からないので、失敗を全て残すときは全エピソードを記録しておく
        record = i < cfg.num_videos or cfg.save_failed_videos
        r = rollout(env, policy, executor, seed, device, record=record, timings=timings)
        failure_type = None if r["success"] else FAILURE_TYPES[r["max_reward"]]
        if r["frames"] is not None and (i < cfg.num_videos or failure_type is not None):
            tag = "success" if failure_type is None else f"fail_{failure_type}"
            path = out_dir / "videos" / f"ep{i:03d}_seed{seed}_{tag}.mp4"
            imageio.mimsave(path, r["frames"], fps=env.metadata["render_fps"])
        episodes.append(
            {
                "seed": seed,
                **{k: r[k] for k in ("success", "max_reward", "steps")},
                "failure_type": failure_type,
            }
        )
        pbar.set_postfix(success=f"{np.mean([e['success'] for e in episodes]):.0%}")
    env.close()

    max_rewards = np.array([e["max_reward"] for e in episodes])
    summary = {
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        # 各段階まで到達したエピソードの割合（どこで失敗しがちかを見るため）
        "reached": {REWARD_LABELS[r]: float(np.mean(max_rewards >= r)) for r in range(1, 5)},
        # 失敗したエピソードの本数を型ごとに数える（成功は含めない）
        "failure_types": {
            name: int(np.sum(max_rewards == r)) for r, name in enumerate(FAILURE_TYPES)
        },
        # predict_chunk 1回あたりの時間。最初の数回は GPU の準備（cudnn.benchmark など）で遅いので中央値も残す
        "predict_ms": {
            "mean": float(np.mean(timings) * 1000),
            "median": float(np.median(timings) * 1000),
            "calls": len(timings),
        },
    }
    # diffusion / flow_matching は同じチェックポイントをサンプラーを変えて評価するので、実際に使った設定を残す
    sampler = {name: getattr(policy, name) for name in SAMPLER_OPTIONS if hasattr(policy, name)}
    if sampler:
        summary["sampler"] = sampler
    results = {"config": {k: str(v) for k, v in vars(cfg).items()}, **summary, "episodes": episodes}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return summary


def apply_sampler_overrides(policy: Policy, cfg: EvalConfig) -> None:
    """評価の引数でサンプラー・ODE ソルバー・推論ステップ数・x̂_0 のクリップの値を上書きする。

    diffusion / flow_matching のみ。学習し直さずに比べるため。
    """
    for name in SAMPLER_OPTIONS:
        value = getattr(cfg, name)
        if value is None:
            continue
        if not hasattr(policy, name):
            raise ValueError(f"--{name.replace('_', '-')} は {type(policy).__name__} では使えない")
        setattr(policy, name, value)


def main() -> None:
    """コマンドライン引数で指定したチェックポイントを評価し、結果を表示する。"""
    cfg = parse_config(EvalConfig)
    policy, train_cfg = load_checkpoint(cfg.ckpt, device="cuda")
    apply_sampler_overrides(policy, cfg)
    # diffusion は同じチェックポイントをサンプラーを変えて何度も評価するので、出力先を分ける
    # （例: eval_ckpt_last_ddim10）。それ以外のポリシーは eval_ckpt_last のまま
    tag = ""
    if hasattr(policy, "sampler"):
        sampler = policy.sampler
        # DDPM は常に T ステップ全部を辿るので、推論ステップ数の代わりに T を付ける
        steps_attr = "num_diffusion_steps" if sampler == "ddpm" else "num_inference_steps"
        tag = f"_{sampler}{getattr(policy, steps_attr)}"
        # クリップの値を上書きしたときだけ付ける（例: eval_ckpt_last_ddim10_clip0 はクリップしない場合）
        if cfg.x0_clip is not None:
            tag += f"_clip{cfg.x0_clip:g}"
        # η も上書きしたときだけ付ける（例: eval_ckpt_last_ddim100_eta1）。DDPM には効かないので付けない
        if cfg.ddim_eta is not None and sampler == "ddim":
            tag += f"_eta{cfg.ddim_eta:g}"
    elif hasattr(policy, "solver"):
        # flow_matching も同じ理由で分ける（例: eval_ckpt_last_euler10、eval_ckpt_last_heun5）
        tag = f"_{policy.solver}{policy.num_inference_steps}"
    # temporal ensembling なしの評価は、ありの評価（既定）の結果を上書きしないよう出力先を分ける。
    # 末尾の数字は何ステップごとに推論したか（例: eval_ckpt_last_noens100、eval_ckpt_last_ddim10_noens50）
    if not cfg.temporal_ensemble:
        tag += f"_noens{cfg.query_every or policy.chunk_size}"
    # ensembling の重みの係数 m を既定から変えたときも分ける（例: eval_ckpt_last_coef0.1）
    elif cfg.ensemble_coef != EvalConfig.ensemble_coef:
        tag += f"_coef{cfg.ensemble_coef:g}"
    # 評価のシードの範囲を既定（シード 0 から 50 本）から変えたときも分ける
    # （例: eval_ckpt_last_s50n150 はシード 50〜199。既定の 50 本と合わせて 200 本として集計する）
    if cfg.seed != EvalConfig.seed or cfg.num_episodes != EvalConfig.num_episodes:
        tag += f"_s{cfg.seed}n{cfg.num_episodes}"
    out_dir = cfg.out_dir or cfg.ckpt.parent / f"eval_{cfg.ckpt.stem}{tag}"
    torch.backends.cudnn.benchmark = True

    summary = evaluate(policy, cfg, train_cfg.task, out_dir)
    print(f"success rate: {summary['success_rate']:.1%} ({cfg.num_episodes} episodes)")
    for label in REWARD_LABELS[1:]:
        print(f"  reached '{label}': {summary['reached'][label]:.1%}")
    print("failures: " + " / ".join(f"{k} {v}" for k, v in summary["failure_types"].items()))
    ms = summary["predict_ms"]
    print(
        f"predict_chunk: mean {ms['mean']:.1f} ms / median {ms['median']:.1f} ms ({ms['calls']} calls)"
    )
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
