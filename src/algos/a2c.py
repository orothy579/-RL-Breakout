"""A2C 빌더 — SB3 의 A2C 를 그대로 사용한다."""

from __future__ import annotations

import copy
from typing import Any

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecEnv


def build_a2c(
    cfg: dict[str, Any],
    *,
    env: VecEnv,
    tensorboard_log: str | None,
    seed: int,
    device: str = "auto",
) -> A2C:
    algo_cfg = cfg["algo"]
    kwargs = copy.deepcopy(algo_cfg.get("kwargs", {}) or {})
    policy = kwargs.pop("policy", "CnnPolicy")

    return A2C(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        seed=seed,
        device=device,
        verbose=0,
        **kwargs,
    )
