#!/usr/bin/env python3
"""Environment installation verification (ALE/Breakout-v5 baseline)."""

from __future__ import annotations

import importlib.metadata as _imd
import sys


def _forget_gymnasium() -> None:
    for k in list(sys.modules):
        if (
            k == "gymnasium"
            or k.startswith("gymnasium.")
            or k == "ale_py"
            or k.startswith("ale_py.")
        ):
            del sys.modules[k]


def main() -> None:
    total = 4

    def step(idx: int, label: str) -> None:
        print(f"[{idx}/{total}] {label} ... ", end="", flush=True)

    def ok(detail: str = "") -> None:
        print(f"OK  {detail}")

    def fail(reason: str, hint: str = "") -> None:
        print("FAIL")
        print(f"    Cause: {reason}")
        if hint:
            print(f"    Fix: {hint}")
        raise SystemExit(1)

    print(f"[diag] Python: {sys.executable}")

    step(1, "Gymnasium import")
    try:
        _forget_gymnasium()
        import gymnasium

        pip_ver = _imd.version("gymnasium")
        mod_ver = gymnasium.__version__
        ok(f"(gymnasium.__version__={mod_ver})")
        print(f"    [diag] gymnasium 로드 경로: {gymnasium.__file__}")
        print(f"    [diag] pip 설치 메타데이터 버전: {pip_ver}")
        if not mod_ver.startswith("1.3"):
            print("    [Warning] 권장: gymnasium==1.3.0 (과제 고정 버전)")
    except Exception as e:
        fail(f"{type(e).__name__}: {e}", 'pip install "gymnasium[atari]==1.3.0"')

    step(2, "ale-py import + ALE registration")
    try:
        import ale_py

        if hasattr(gymnasium, "register_envs"):
            gymnasium.register_envs(ale_py)
        ok(f"(version {ale_py.__version__})")
    except ImportError as e:
        fail(
            f"{e}",
            'pip install "ale-py==0.11.2" "autorom[accept-rom-license]==0.6.1" '
            "&& AutoROM --accept-license",
        )

    step(3, "ALE/Breakout-v5 environment creation")
    env = None
    try:
        env = gymnasium.make(
            "ALE/Breakout-v5",
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=False,
        )
        ok(f"(action_space={env.action_space})")
    except gymnasium.error.NamespaceNotFound:
        fail(
            "Failed to register ALE namespace",
            "Kernel 재시작 후 다시 실행하거나 AutoROM --accept-license",
        )
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")

    step(4, "Random play 100 steps")
    try:
        obs, info = env.reset(seed=0)
        total_reward = 0.0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                obs, info = env.reset()
        env.close()
        ok(f"(Cumulative reward={total_reward})")
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")

    print()
    print("✅ All checks passed. You are ready to start training!")


if __name__ == "__main__":
    main()
