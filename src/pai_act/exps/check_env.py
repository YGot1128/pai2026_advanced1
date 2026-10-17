"""環境の動作確認: GPU(PyTorch) と gym-aloha の描画・動画保存が通るかを見る。

確認する項目（上から順に実行し、どこかで失敗したらそこで止まる）:
    1. check_torch: PyTorch から CUDA（GPU）が使えるか
    2. check_gl:    MuJoCo の描画（OpenGL / EGL）が GPU で動いているか
    3. check_aloha: gym-aloha の環境を作って動かし、カメラ画像を動画ファイルに保存できるか

使い方（リポジトリ直下で実行）: bash scripts/check_env.sh
出力: outputs/check_env.mp4（ランダムな行動で 100 ステップ動かした動画）
"""

import os
from pathlib import Path

import gym_aloha  # noqa: F401  (gymnasium に環境を登録する)
import gymnasium as gym
import imageio
import torch


def check_torch() -> None:
    """CUDA が使えることを確かめ、GPU 上で行列積を1回計算する。"""
    print(f"torch {torch.__version__} / CUDA build {torch.version.cuda}")
    assert torch.cuda.is_available(), "CUDA が使えない（--gpus=all でコンテナを起動しているか確認）"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(1024, 1024, device="cuda")
    print(f"matmul OK: {(x @ x).sum().item():.3f}")


def check_gl() -> None:
    """MuJoCo と同じ EGL で OpenGL のコンテキストを作り、どのデバイスで描画されるかを表示する。

    MUJOCO_GL=egl はディスプレイなし（ヘッドレス）で描画するための設定（devcontainer.json で指定）。
    renderer に NVIDIA と出れば GPU 描画、llvmpipe と出れば CPU によるソフトウェア描画（遅い）。
    """
    from mujoco.egl import GLContext
    from OpenGL import GL

    ctx = GLContext(64, 64)
    ctx.make_current()
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} / renderer: {renderer}")
    if "NVIDIA" not in renderer:
        print("  WARNING: NVIDIA GPU で描画されていない（llvmpipe なら CPU 描画で遅い）")


def check_aloha(out_path: Path, steps: int = 100) -> None:
    """gym-aloha でランダムな行動を steps 回実行し、観測の形を表示して動画を out_path に保存する。"""
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=0)
    print(f"observation keys: {list(obs.keys())}")
    print(f"pixels: {obs['pixels']['top'].shape}, agent_pos: {obs['agent_pos'].shape}")
    print(f"action space: {env.action_space}")

    frames = [obs["pixels"]["top"]]
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        frames.append(obs["pixels"]["top"])
        if terminated or truncated:
            break
    env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=50)
    print(f"saved {len(frames)} frames -> {out_path}")


if __name__ == "__main__":
    check_torch()
    check_gl()
    check_aloha(Path("outputs/check_env.mp4"))
    print("ALL OK")
