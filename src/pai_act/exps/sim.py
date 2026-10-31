"""gym-aloha（借用部分: 物理シミュレータ）を扱うための薄いラッパ。

gym-aloha は MuJoCo（dm_control）で ALOHA 双腕ロボットを再現し、Gymnasium の API で操作できるようにしたもの。

    env = make_env()
    obs, info = env.reset(seed=0)           # シードでキューブの初期位置が決まる
    obs, reward, terminated, truncated, info = env.step(action)

観測 obs（obs_type="pixels_agent_pos" のとき）:
    obs["pixels"]["top"] : 上方カメラの画像 (480, 640, 3) uint8
    obs["agent_pos"]     : 両腕の関節角 (14,) float64 … データセットの observation.state と同じ並び
行動 action: 両腕の目標関節角 (14,) … データセットの action と同じ並び・単位（exps/replay_demo.py で検証済み）
制御周期: 1ステップ = 0.02 秒（50Hz）。データセットの 1 フレームに対応する。
報酬: transfer cube では 0〜4 の段階（exps/evaluate.py の説明を参照）。4 で成功し terminated=True になる。
"""

from typing import cast

import gym_aloha  # noqa: F401  (gymnasium に環境を登録する)
import gymnasium as gym
import numpy as np
from gym_aloha.env import AlohaEnv
from gym_aloha.tasks.sim import BOX_POSE
from numpy.typing import ArrayLike

ENV_IDS = {
    "transfer_cube": "gym_aloha/AlohaTransferCube-v0",
    "insertion": "gym_aloha/AlohaInsertion-v0",
}


def make_env(task: str = "transfer_cube", max_episode_steps: int = 400) -> gym.Env:
    """タスク名から gym-aloha の環境を作る。

    Args:
        task: ENV_IDS のキー。
        max_episode_steps: このステップ数に達すると step() が truncated=True を返す（打ち切り）。
            gym-aloha の既定は 300 ステップ（6秒）だが、人間のデモには 300 ステップ以降に成功するものが
            多い（50本中19本）ので、デモと同じ 400 を既定にしている。
    """
    return gym.make(ENV_IDS[task], obs_type="pixels_agent_pos", max_episode_steps=max_episode_steps)


def reset_with_box_pose(env: gym.Env, box_pose: ArrayLike) -> dict:
    """キューブの初期姿勢 [x, y, z, qw, qx, qy, qz] を指定してリセットする。

    gym-aloha の reset() はシードから姿勢を乱数で決めてしまうため、
    同じ処理（グローバル変数 BOX_POSE を設定 → 内部の dm_control 環境を reset）を直接行う。
    デモの再生（exps/replay_demo.py）で、デモと同じ位置にキューブを置くために使う。

    Returns:
        リセット直後の観測（env.reset() の obs と同じ形）。
    """
    env.reset(seed=0)  # TimeLimit のステップ数カウンタを戻すため
    base = cast(AlohaEnv, env.unwrapped)  # TimeLimit などのラッパを外した gym-aloha 本体
    BOX_POSE[0] = np.asarray(box_pose, dtype=np.float64)
    raw_obs = base._env.reset()
    return base._format_raw_obs(raw_obs.observation)
