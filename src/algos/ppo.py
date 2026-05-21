"""PPO 빌더 — SB3 의 PPO 를 그대로 사용한다."""

from __future__ import annotations

import copy
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


def build_ppo(
    cfg: dict[str, Any],
    *,
    env: VecEnv,
    tensorboard_log: str | None,
    seed: int,
    device: str = "auto",
) -> PPO:
    algo_cfg = cfg["algo"]
    kwargs = copy.deepcopy(algo_cfg.get("kwargs", {}) or {})
    policy = kwargs.pop("policy", "CnnPolicy")

    return PPO(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        seed=seed,
        device=device,
        verbose=1,
        **kwargs,
    )
