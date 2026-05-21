#!/usr/bin/env python3
"""Breakout v5 — 학습된 정책 평가."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algos.registry import load_model
from src.envs import build_eval_env
from src.eval import evaluate_model, format_summary
from src.utils.config import load_config
from src.utils.logging import write_episode_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained agent on ALE/Breakout-v5.")
    p.add_argument("--run", type=Path, default=None, help="실험 디렉토리")
    p.add_argument("--model", type=Path, default=None, help="모델 .zip")
    p.add_argument("--config", type=Path, default=None, help="config yaml")
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="평가 env 시드")
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="결과 저장 디렉토리 (기본: <run>/eval_runs/seed{eval_seed})",
    )
    return p.parse_args()


def _algo_slug(cfg: dict) -> str:
    name = str(cfg["algo"]["name"]).lower()
    feats = cfg["algo"].get("features", {}) or {}
    if name == "dqn":
        if feats.get("dueling"):
            return "dueling_dqn"
        if feats.get("double_q"):
            return "ddqn"
        return "dqn"
    return name


def _training_seed(run_dir: Path | None) -> int | str:
    if run_dir is None:
        return "unknown"
    for token in run_dir.name.split("_"):
        if token.startswith("seed"):
            try:
                return int(token.replace("seed", ""))
            except ValueError:
                return token
    return "unknown"


def _eval_artifact_stem(cfg: dict, *, run_dir: Path | None) -> str:
    algo = _algo_slug(cfg)
    seed = _training_seed(run_dir)
    timesteps = int(cfg.get("train", {}).get("total_timesteps", 0))
    return f"summary_{algo}_seed{seed}_{timesteps}"


def _resolve_run_artifacts(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None]:
    if args.run is not None:
        run = args.run.resolve()
        cfg_path = run / "config.yaml"
        cand = [run / "best_model" / "best_model.zip", run / "final_model.zip"]
        model_path = next((p for p in cand if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(f"No best/final model in {run}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.yaml in {run}")
        return model_path, cfg_path, run
    if args.model is None or args.config is None:
        raise SystemExit("--run 또는 --model + --config 둘 중 하나는 지정해야 합니다.")
    return args.model.resolve(), args.config.resolve(), None


def main() -> None:
    args = parse_args()
    model_path, cfg_path, run_dir = _resolve_run_artifacts(args)
    cfg = load_config(cfg_path)

    out_dir = args.output_dir
    if out_dir is None and args.run is not None:
        out_dir = args.run / "eval_runs" / f"seed{args.seed}"
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] model  = {model_path}")
    print(f"[eval] config = {cfg_path}")

    eval_env = build_eval_env(
        cfg["env"], cfg.get("eval_env", {}), seed=args.seed + 10_000
    )
    model = load_model(cfg, str(model_path), env=eval_env, device=args.device)

    summary = evaluate_model(
        model,
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    eval_env.close()

    print(format_summary(summary, name=cfg["algo"]["name"]))

    if out_dir is not None:
        stem = _eval_artifact_stem(cfg, run_dir=run_dir)
        json_path = out_dir / f"{stem}.json"
        csv_path = out_dir / f"{stem.replace('summary_', 'episodes_', 1)}.csv"
        payload = summary.to_dict(drop_lists=True)
        payload["meta"] = {
            "algo": _algo_slug(cfg),
            "training_seed": str(_training_seed(run_dir)),
            "total_timesteps": int(cfg.get("train", {}).get("total_timesteps", 0)),
            "eval_seed": args.seed,
            "n_eval_episodes": args.n_eval_episodes,
            "deterministic": args.deterministic,
            "model_path": str(model_path),
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        write_episode_csv(
            csv_path,
            rewards=summary.rewards,
            lengths=summary.lengths,
            meta={
                "algo": _algo_slug(cfg),
                "training_seed": str(_training_seed(run_dir)),
                "total_timesteps": str(cfg.get("train", {}).get("total_timesteps", "")),
                "eval_seed": str(args.seed),
                "deterministic": str(args.deterministic),
                "model_path": str(model_path),
            },
        )
        print(f"[eval] saved  = {json_path}")
        print(f"[eval] saved  = {csv_path}")


if __name__ == "__main__":
    main()
