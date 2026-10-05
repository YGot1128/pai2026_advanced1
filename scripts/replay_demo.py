"""デモの行動をシムで開ループ再生し、データとシムの対応を検証する。

確認したいこと:
- データセットの action をそのまま env.step() に渡してよいか（行動空間の一致）
- 再生したときにシムの関節状態がデータの observation.state を追従するか
- 成功判定（reward == 4）がどう出るか

キューブの初期位置はデータセットに記録されていないため、
初期フレームに写っている赤いキューブの画素重心からテーブル上の (x, y) を推定してシムに置く。

使い方: uv run python scripts/replay_demo.py [--episodes 0 1 2] [--video-episodes 0 1]
"""

import argparse
from pathlib import Path

import imageio
import numpy as np

from pai_act.dataset import AlohaDataset
from pai_act.sim import make_env, reset_with_box_pose

CUBE_Z = 0.05
CUBE_QUAT = [1.0, 0.0, 0.0, 0.0]


def red_centroid(image: np.ndarray) -> np.ndarray:
    im = image.astype(np.int16)
    # 影の中ではキューブの赤が暗くなるので、明るさではなく色の比率で判定する
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mask = (r > 40) & (r > 2 * g) & (r > 2 * b)
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0, "赤いキューブが画像に見つからない"
    return np.array([xs.mean(), ys.mean()])


def poly_features(uv: np.ndarray) -> np.ndarray:
    u, v = uv[:, 0] / 640, uv[:, 1] / 480
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)


def calibrate(env, n: int = 7) -> np.ndarray:
    """キューブをテーブル上の格子点に置いて描画し、画素重心 → (x, y) の2次多項式を最小二乗で当てる。"""
    # gym-aloha のサンプリング範囲（x∈[0, 0.2], y∈[0.4, 0.6]）を覆う
    xs, ys = np.meshgrid(np.linspace(0.0, 0.2, n), np.linspace(0.4, 0.6, n))
    xy = np.stack([xs.ravel(), ys.ravel()], axis=1)
    uv = np.array(
        [
            red_centroid(reset_with_box_pose(env, [x, y, CUBE_Z, *CUBE_QUAT])["pixels"]["top"])
            for x, y in xy
        ]
    )
    coef, *_ = np.linalg.lstsq(poly_features(uv), xy, rcond=None)
    err = np.abs(poly_features(uv) @ coef - xy).max()
    print(f"calibration: max fit error {err * 1000:.2f} mm over {len(xy)} points")
    return coef


def replay(env, episode, box_pose) -> dict:
    obs = reset_with_box_pose(env, box_pose)
    frames, states, rewards = [obs["pixels"]["top"]], [obs["agent_pos"]], []
    for action in episode.action:
        obs, reward, _, _, _ = env.step(action.astype(np.float64))
        frames.append(obs["pixels"]["top"])
        states.append(obs["agent_pos"])
        rewards.append(reward)
    states = np.array(states[:-1])  # states[t] は action[t] を実行する前の状態
    return {
        "frames": np.array(frames[:-1]),
        "state_err": np.abs(states - episode.state).mean(axis=0),
        "max_reward": max(rewards),
        "success": 4 in rewards,
        "first_success_step": rewards.index(4) if 4 in rewards else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--video-episodes", type=int, nargs="*", default=[0])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/replay"))
    args = parser.parse_args()

    ds = AlohaDataset()
    env = make_env(max_episode_steps=len(ds[0].action) + 1)
    coef = calibrate(env)
    episodes = args.episodes if args.episodes is not None else range(len(ds))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i in episodes:
        ep = ds[i]
        uv = red_centroid(np.asarray(ep.images[0]))
        x, y = (poly_features(uv[None]) @ coef)[0]
        r = replay(env, ep, [x, y, CUBE_Z, *CUBE_QUAT])
        # 推定したキューブ位置がデータの初期フレームと合っているか（画素重心のずれ）
        pix_err = np.linalg.norm(red_centroid(r["frames"][0]) - uv)
        results.append(r["success"])
        print(
            f"ep {i:2d}: cube=({x:.3f}, {y:.3f}) pix_err={pix_err:.1f}px "
            f"max_reward={r['max_reward']} success={r['success']} "
            f"(step {r['first_success_step']}) "
            f"arm_err={r['state_err'][[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].mean():.4f}rad "
            f"grip_err={r['state_err'][[6, 13]].mean():.4f}"
        )
        if i in args.video_episodes:
            side_by_side = np.concatenate([r["frames"], np.asarray(ep.images)], axis=2)
            path = args.out_dir / f"replay_ep{i:02d}.mp4"
            imageio.mimsave(path, side_by_side, fps=ds.fps)
            print(f"  saved {path} (左: シム再生 / 右: デモ動画)")

    print(f"success: {sum(results)}/{len(results)} = {np.mean(results):.1%}")


if __name__ == "__main__":
    main()
