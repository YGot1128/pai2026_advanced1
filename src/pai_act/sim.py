"""gym-aloha（借用部分: 物理シム）を扱うための薄いラッパ。"""

import gym_aloha  # noqa: F401  (gymnasium に環境を登録する)
import gymnasium as gym
import numpy as np
from gym_aloha.tasks.sim import BOX_POSE

ENV_IDS = {
    "transfer_cube": "gym_aloha/AlohaTransferCube-v0",
    "insertion": "gym_aloha/AlohaInsertion-v0",
}


def make_env(task: str = "transfer_cube", max_episode_steps: int = 400) -> gym.Env:
    # gym-aloha の既定は 300 ステップ（6秒）。デモは 400 フレームなので伸ばせるようにする
    return gym.make(
        ENV_IDS[task], obs_type="pixels_agent_pos", max_episode_steps=max_episode_steps
    )


def reset_with_box_pose(env: gym.Env, box_pose: np.ndarray) -> dict:
    """キューブの初期姿勢 [x, y, z, qw, qx, qy, qz] を指定してリセットする。

    gym-aloha の reset() はシードから姿勢を乱数で決めてしまうため、
    同じ処理（グローバル変数 BOX_POSE を設定 → 内部の dm_control 環境を reset）を直接行う。
    """
    env.reset(seed=0)  # TimeLimit のステップ数カウンタを戻すため
    base = env.unwrapped
    BOX_POSE[0] = np.asarray(box_pose, dtype=np.float64)
    raw_obs = base._env.reset()
    return base._format_raw_obs(raw_obs.observation)
