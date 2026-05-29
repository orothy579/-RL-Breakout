"""학습된 정책 시각화."""

from __future__ import annotations

import time
from typing import cast

import cv2
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, is_vecenv_wrapped


def play_native_window(
    model: BaseAlgorithm,
    env: VecEnv,
    *,
    n_episodes: int = 3,
    deterministic: bool = True,
) -> None:
    if not is_vecenv_wrapped(env, VecMonitor):
        env = VecMonitor(env)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward[0])
            env.render()
        print(f"[play] episode {ep + 1}/{n_episodes} reward={ep_reward:.1f}")


def play_opencv_scaled(
    model: BaseAlgorithm,
    env: VecEnv,
    *,
    n_episodes: int = 3,
    deterministic: bool = True,
    scale: float = 4,
    aspect_stretch: float = 1.6,
    render_fps: float = 30.0,
) -> list[float]:
    if not is_vecenv_wrapped(env, VecMonitor):
        env = VecMonitor(env)

    rewards: list[float] = []
    delay = 1.0 / render_fps if render_fps > 0 else 0.0

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward[0])

            frame = env.render(mode="rgb_array")
            if frame is None:
                continue
            arr = np.asarray(frame)
            if arr.ndim == 4:
                arr = arr[0]
            elif arr.ndim != 3:
                continue
            img = cast(np.ndarray, arr)
            if scale > 1 or aspect_stretch != 1.0:
                h, w = img.shape[:2]
                img = cv2.resize(
                    img,
                    (int(w * scale * aspect_stretch), int(h * scale)),
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow("Breakout Agent", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return rewards
            if delay > 0:
                time.sleep(delay)

        rewards.append(ep_reward)
        print(f"[play] episode {ep + 1}/{n_episodes} reward={ep_reward:.1f}")

    cv2.destroyAllWindows()
    return rewards


def play_with_stats(
    model: BaseAlgorithm,
    env: VecEnv,
    *,
    n_episodes: int = 3,
    deterministic: bool = True,
) -> list[float]:
    rewards, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_episodes,
        deterministic=deterministic,
        render=False,
        return_episode_rewards=True,
        warn=True,
    )
    return list(rewards)
