#!/usr/bin/env python3
"""Breakout v5 — 학습된 정책을 화면으로 시연한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.algos.registry import load_model
from src.envs import build_play_env
from src.render import play_native_window, play_opencv_scaled
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a trained agent playing Breakout v5.")
    p.add_argument("--run", type=Path, default=None)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--window-scale", type=int, default=None, metavar="N")
    p.add_argument("--render-fps", type=float, default=30.0)
    return p.parse_args()


def _resolve(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.run is not None:
        run = args.run.resolve()
        cfg_path = run / "config.yaml"
        cand = [run / "best_model" / "best_model.zip", run / "final_model.zip"]
        model_path = next((p for p in cand if p.exists()), None)
        if model_path is None or not cfg_path.exists():
            raise SystemExit(f"{run} 안에 config.yaml 또는 모델 파일이 없습니다.")
        return model_path, cfg_path
    if args.model is None or args.config is None:
        raise SystemExit("--run 또는 --model + --config 둘 중 하나는 지정해야 합니다.")
    return args.model.resolve(), args.config.resolve()


def main() -> None:
    args = parse_args()
    model_path, cfg_path = _resolve(args)
    cfg = load_config(cfg_path)

    use_opencv = args.window_scale is not None
    render_mode = "rgb_array" if use_opencv else "human"

    env = build_play_env(
        cfg["env"],
        cfg.get("eval_env", {}),
        seed=args.seed,
        render_mode=render_mode,
    )
    model = load_model(cfg, str(model_path), env=env, device=args.device)

    print(f"[play] model = {model_path}")
    print(
        f"[play] mode  = {'opencv x'+str(args.window_scale) if use_opencv else 'human (Stella)'}"
    )

    try:
        if use_opencv:
            scale = args.window_scale if args.window_scale is not None else 1
            rewards = play_opencv_scaled(
                model,
                env,
                n_episodes=args.n_episodes,
                deterministic=args.deterministic,
                scale=scale,
                render_fps=args.render_fps,
            )
            if rewards:
                arr = np.asarray(rewards, dtype=np.float64)
                print(
                    f"[play] 에피소드 {len(rewards)}회 평균 보상: "
                    f"{arr.mean():.2f} ± {arr.std():.2f}"
                )
        else:
            play_native_window(
                model,
                env,
                n_episodes=args.n_episodes,
                deterministic=args.deterministic,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
