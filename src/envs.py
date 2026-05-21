"""Breakout v5 환경 factory.

모든 알고리즘이 **동일한 baseline 환경**을 공유하도록 한 곳에서 정의한다.
Installation Manual §1.2 / Guide §B.4 의 가이드라인을 그대로 적용:

    env_kwargs = {
        "frameskip": 1,
        "repeat_action_probability": 0.0,
        "full_action_space": False,
    }
    AtariWrapper (NoopReset/MaxAndSkip(4)/EpisodicLife/FireReset/WarpFrame/ClipReward)
    + VecFrameStack(n_stack=4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ale_py
import gymnasium as gym

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecTransposeImage,
)

ENV_ID_DEFAULT = "ALE/Breakout-v5"


def _ensure_registered() -> None:
    gym.register_envs(ale_py)


def _coerce_env_kwargs(env_kwargs: dict[str, Any]) -> dict[str, Any]:
    """YAML 에서 들어온 dict 의 타입 보정 (bool/숫자)."""
    out: dict[str, Any] = {}
    for k, v in env_kwargs.items():
        if isinstance(v, str) and v.lower() in {"true", "false"}:
            out[k] = v.lower() == "true"
        else:
            out[k] = v
    return out


def build_train_env(
    env_cfg: dict[str, Any],
    *,
    seed: int,
    monitor_dir: str | Path | None = None,
) -> VecEnv:
    """학습용 VecEnv 구성."""
    _ensure_registered()

    env_id = env_cfg.get("env_id", ENV_ID_DEFAULT)
    n_envs = int(env_cfg.get("n_envs", 8))
    frame_stack = int(env_cfg.get("frame_stack", 4))
    env_kwargs = _coerce_env_kwargs(env_cfg.get("env_kwargs", {}))
    wrapper_kwargs = dict(env_cfg.get("wrapper_kwargs", {}))

    monitor_dir_str = str(monitor_dir) if monitor_dir is not None else None
    if monitor_dir_str is not None:
        Path(monitor_dir_str).mkdir(parents=True, exist_ok=True)

    venv = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir_str,
        env_kwargs=env_kwargs,
        wrapper_kwargs=wrapper_kwargs or None,
    )
    if frame_stack and frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)
    return venv


def build_eval_env(
    env_cfg: dict[str, Any],
    eval_env_cfg: dict[str, Any],
    *,
    seed: int,
) -> VecEnv:
    """평가용 VecEnv."""
    _ensure_registered()

    env_id = env_cfg.get("env_id", ENV_ID_DEFAULT)
    n_envs = int(eval_env_cfg.get("n_envs", 1))
    frame_stack = int(env_cfg.get("frame_stack", 4))
    env_kwargs = _coerce_env_kwargs(env_cfg.get("env_kwargs", {}))

    eval_wrapper = dict(env_cfg.get("wrapper_kwargs", {}))
    eval_wrapper.update(eval_env_cfg.get("wrapper_kwargs", {}))
    eval_wrapper.setdefault("terminal_on_life_loss", False)
    eval_wrapper.setdefault("clip_reward", False)

    venv: VecEnv = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        wrapper_kwargs=eval_wrapper,
    )
    if frame_stack and frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)
    return VecTransposeImage(venv)


def build_play_env(
    env_cfg: dict[str, Any],
    eval_env_cfg: dict[str, Any],
    *,
    seed: int,
    render_mode: str = "human",
) -> VecEnv:
    """대화형 시청용 단일 env."""
    _ensure_registered()
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    from stable_baselines3.common.monitor import Monitor

    env_id = env_cfg.get("env_id", ENV_ID_DEFAULT)
    frame_stack = int(env_cfg.get("frame_stack", 4))
    env_kwargs = _coerce_env_kwargs(env_cfg.get("env_kwargs", {}))

    eval_wrapper = dict(env_cfg.get("wrapper_kwargs", {}))
    eval_wrapper.update(eval_env_cfg.get("wrapper_kwargs", {}))
    eval_wrapper.setdefault("terminal_on_life_loss", False)
    eval_wrapper.setdefault("clip_reward", False)

    def _make() -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
        env = Monitor(env)
        env = AtariWrapper(env, **eval_wrapper)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    venv: VecEnv = DummyVecEnv([_make])
    if frame_stack and frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)
    return venv
