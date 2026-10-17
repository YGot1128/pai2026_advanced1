"""デモの行動をシミュレータで開ループ再生し、データとシミュレータの対応を検証する。

確認したいこと:
- データセットの action をそのまま env.step() に渡してよいか（行動空間の一致）
- 再生したときにシミュレータの関節状態がデータの observation.state を追従するか
- 成功判定（reward == 4）がどう出るか

「開ループ」とは、観測を見ずに、記録された行動列を先頭から順に流し込むこと。
シミュレータとデータが一致していれば、デモと同じ初期配置から同じ行動を流せば同じ結果（成功）になるはず。
これが成り立たないと、学習したポリシーがいくら正確でも成功しないので、学習より先に確認しておく。

キューブの初期位置はデータセットに記録されていないため、
初期フレームに写っている赤いキューブの画素重心からテーブル上の (x, y) を推定してシミュレータに置く:
    1. calibrate: シミュレータでキューブを格子点に置いて描画し、「画素重心 (u, v) → 位置 (x, y)」の対応を集め、
       2次多項式を最小二乗法で当てはめる
    2. デモの初期フレームの画素重心を 1 の多項式に入れて (x, y) を推定する

使い方（リポジトリ直下で実行）:
    bash scripts/replay_demo.sh                                   # 全50本を再生して検証
    bash scripts/replay_demo.sh --episodes 0 1 2 --video-episodes 0 # 一部だけ再生し、動画も保存
出力: outputs/replay/replay_epXX.mp4（左: シミュレータでの再生 / 右: デモの動画）
"""

import argparse
from pathlib import Path

import imageio
import numpy as np

from pai_act.data.dataset import AlohaDataset
from pai_act.exps.sim import make_env, reset_with_box_pose

CUBE_Z = 0.05  # キューブの高さ（gym-aloha のサンプリングでも固定値）
CUBE_QUAT = [1.0, 0.0, 0.0, 0.0]  # 回転なし（gym-aloha も回転させない）


def red_centroid(image: np.ndarray) -> np.ndarray:
    """画像 (H, W, 3) の中の赤いキューブの画素重心 [u, v]（u: 横方向の列, v: 縦方向の行）を返す。"""
    im = image.astype(np.int16)  # uint8 のままだと 2 * g が 255 を超えたときにあふれる
    # 影の中ではキューブの赤が暗くなるので、明るさではなく色の比率で判定する
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mask = (r > 40) & (r > 2 * g) & (r > 2 * b)
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0, "赤いキューブが画像に見つからない"
    return np.array([xs.mean(), ys.mean()])


def poly_features(uv: np.ndarray) -> np.ndarray:
    """画素座標 (N, 2) を2次多項式の特徴 [1, u, v, u², uv, v²] (N, 6) に変換する。

    カメラは斜め上からテーブルを見ているので、画素座標とテーブル上の位置の関係は線形ではない
    （遠くほど小さく写る）。2次の項を入れてその歪みを近似する。
    u, v は画像サイズで割って 0〜1 程度にし、最小二乗の数値的な安定性を保つ。
    """
    u, v = uv[:, 0] / 640, uv[:, 1] / 480
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)


def calibrate(env, n: int = 7) -> np.ndarray:
    """キューブをテーブル上の格子点に置いて描画し、画素重心 → (x, y) の2次多項式を最小二乗で当てる。

    Returns:
        係数行列 coef (6, 2)。poly_features(uv) @ coef で (x, y) が得られる。
    """
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
    """キューブを box_pose に置いてリセットし、デモの行動を先頭から順に実行する。

    Returns:
        frames: シミュレータのカメラ画像 (T, H, W, 3)。デモの images と同じ時刻に揃えてある
        state_err: 次元ごとの関節角の平均絶対誤差 (14,)（シミュレータの状態 vs デモの state）
        max_reward / success / first_success_step: 報酬の最大値・成功したか・初めて成功したステップ
    """
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
    parser.add_argument(
        "--episodes", type=int, nargs="*", default=None, help="省略時は全エピソード"
    )
    parser.add_argument("--video-episodes", type=int, nargs="*", default=[0], help="動画を保存する")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/replay"))
    args = parser.parse_args()

    ds = AlohaDataset()
    # 成功判定で打ち切られずに全行動を流せるよう、ステップ上限はデモの長さより長くしておく
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
        # arm_err: 腕の12関節の誤差 [rad] / grip_err: グリッパー2つ（インデックス 6, 13）の誤差
        print(
            f"ep {i:2d}: cube=({x:.3f}, {y:.3f}) pix_err={pix_err:.1f}px "
            f"max_reward={r['max_reward']} success={r['success']} "
            f"(step {r['first_success_step']}) "
            f"arm_err={r['state_err'][[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].mean():.4f}rad "
            f"grip_err={r['state_err'][[6, 13]].mean():.4f}"
        )
        if i in args.video_episodes:
            # 横方向（axis=2 は幅）に並べて、シミュレータとデモを同時に見比べられる動画にする
            side_by_side = np.concatenate([r["frames"], np.asarray(ep.images)], axis=2)
            path = args.out_dir / f"replay_ep{i:02d}.mp4"
            imageio.mimsave(path, list(side_by_side), fps=ds.fps)
            print(f"  saved {path} (左: シミュレータでの再生 / 右: デモ動画)")

    print(f"success: {sum(results)}/{len(results)} = {np.mean(results):.1%}")


if __name__ == "__main__":
    main()
