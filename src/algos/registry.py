"""알고리즘 이름 -> builder 매핑."""

from __future__ import annotations

from typing import Any, Callable

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from src.algos.a2c import build_a2c
from src.algos.dqn import build_dqn
from src.algos.ppo import build_ppo

BuilderFn = Callable[..., BaseAlgorithm]

REGISTRY: dict[str, BuilderFn] = {
    "dqn": build_dqn,
    "ppo": build_ppo,
    "a2c": build_a2c,
}


def list_algorithms() -> list[str]:
    return sorted(REGISTRY.keys())


def build_model(
    cfg: dict[str, Any],
    *,
    env: VecEnv,
    tensorboard_log: str | None,
    seed: int,
    device: str = "auto",
) -> BaseAlgorithm:
    name = str(cfg["algo"]["name"]).lower()
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {list_algorithms()}"
        )
    return REGISTRY[name](
        cfg, env=env, tensorboard_log=tensorboard_log, seed=seed, device=device
    )


def load_model(
    cfg: dict[str, Any],
    model_path: str,
    *,
    env: VecEnv | None = None,
    device: str = "auto",
) -> BaseAlgorithm:
    from stable_baselines3 import A2C, DQN, PPO

    from src.algos.dqn import DoubleDQN

    name = str(cfg["algo"]["name"]).lower()
    features = cfg["algo"].get("features", {}) or {}

    if name == "ppo":
        return PPO.load(model_path, env=env, device=device)
    if name == "a2c":
        return A2C.load(model_path, env=env, device=device)
    if name == "dqn":
        cls = DoubleDQN if features.get("double_q", False) else DQN
        return cls.load(model_path, env=env, device=device)
    raise KeyError(f"Unknown algorithm '{name}'.")
