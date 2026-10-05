"""環境の動作確認: GPU(PyTorch) と gym-aloha の描画・動画保存が通るかを見る。

使い方: uv run python scripts/check_env.py
"""

import os
from pathlib import Path

import gym_aloha  # noqa: F401  (gymnasium に環境を登録する)
import gymnasium as gym
import imageio
import numpy as np
import torch


def check_torch() -> None:
    print(f"torch {torch.__version__} / CUDA build {torch.version.cuda}")
    assert torch.cuda.is_available(), "CUDA が使えない（--gpus=all でコンテナを起動しているか確認）"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(1024, 1024, device="cuda")
    print(f"matmul OK: {(x @ x).sum().item():.3f}")


def check_gl() -> None:
    from mujoco.egl import GLContext
    from OpenGL import GL

    ctx = GLContext(64, 64)
    ctx.make_current()
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} / renderer: {renderer}")
    if "NVIDIA" not in renderer:
        print("  WARNING: NVIDIA GPU で描画されていない（llvmpipe なら CPU 描画で遅い）")


def check_aloha(out_path: Path, steps: int = 100) -> None:
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
    imageio.mimsave(out_path, np.stack(frames), fps=50)
    print(f"saved {len(frames)} frames -> {out_path}")


if __name__ == "__main__":
    check_torch()
    check_gl()
    check_aloha(Path("outputs/check_env.mp4"))
    print("ALL OK")
