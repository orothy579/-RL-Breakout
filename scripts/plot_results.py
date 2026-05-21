#!/usr/bin/env python3
"""experiments/* 디렉토리들을 훑어 학습/평가 결과를 한 장에 정리한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _read_run(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    eval_npz = run_dir / "eval" / "evaluations.npz"
    timesteps: np.ndarray | None = None
    results_mean: np.ndarray | None = None
    results_std: np.ndarray | None = None
    if eval_npz.exists():
        data = np.load(eval_npz)
        timesteps = np.asarray(data["timesteps"])
        results = np.asarray(data["results"])
        results_mean = results.mean(axis=1)
        results_std = results.std(axis=1)

    summary_files: list[Path] = []
    if (run_dir / "eval_runs").exists():
        for seed_dir in sorted((run_dir / "eval_runs").glob("seed*/")):
            named = sorted(
                p for p in seed_dir.glob("summary*.json") if p.name != "summary.json"
            )
            if named:
                summary_files.extend(named)
            elif (seed_dir / "summary.json").exists():
                summary_files.append(seed_dir / "summary.json")

    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in summary_files]

    return {
        "run_dir": run_dir,
        "name": run_dir.name,
        "algo": cfg.get("algo", {}).get("name", "?"),
        "features": cfg.get("algo", {}).get("features", {}) or {},
        "seed": _extract_seed(run_dir.name),
        "timesteps": timesteps,
        "ep_rew_mean": results_mean,
        "ep_rew_std": results_std,
        "summaries": summaries,
    }


def _extract_seed(name: str) -> int | None:
    for token in name.split("_"):
        if token.startswith("seed"):
            try:
                return int(token.replace("seed", ""))
            except ValueError:
                return None
    return None


def _group_label(run: dict[str, Any]) -> str:
    algo = run["algo"]
    feats = run["features"]
    if algo == "dqn":
        tags = []
        if feats.get("double_q"):
            tags.append("double")
        if feats.get("dueling"):
            tags.append("dueling")
        if tags:
            algo = "dqn[" + "+".join(tags) + "]"
    return algo


def plot_learning_curves(runs: list[dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        if r["timesteps"] is None:
            continue
        groups.setdefault(_group_label(r), []).append(r)

    if not groups:
        print("[plot] eval/evaluations.npz 가 어느 런에도 없습니다.")
        return

    for label, members in sorted(groups.items()):
        all_steps = sorted({int(t) for r in members for t in (r["timesteps"] or [])})
        grid = np.array(all_steps)
        if len(grid) == 0:
            continue
        stacked = []
        for r in members:
            xs = np.asarray(r["timesteps"], dtype=np.float64)
            ys = np.asarray(r["ep_rew_mean"], dtype=np.float64)
            stacked.append(np.interp(grid, xs, ys))
        stacked_arr = np.stack(stacked, axis=0)
        mean = stacked_arr.mean(axis=0)
        std = stacked_arr.std(axis=0)
        plt.plot(grid, mean, label=f"{label} (n={len(members)})")
        plt.fill_between(grid, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Environment steps")
    plt.ylabel("Eval episode reward (mean ± std over seeds)")
    plt.title("Breakout v5 — Learning Curves")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] saved {out_path}")


def plot_eval_distribution(runs: list[dict[str, Any]], out_path: Path) -> None:
    by_label: dict[str, list[float]] = {}
    for r in runs:
        label = _group_label(r)
        for s in r["summaries"]:
            by_label.setdefault(label, []).append(float(s["mean"]))

    if not by_label:
        print("[plot] eval_runs/*/summary*.json 이 없으면 분포 plot 을 건너뜁니다.")
        return

    labels = sorted(by_label.keys())
    data = [by_label[l] for l in labels]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel("Eval mean episode reward (per run)")
    plt.title("Final-policy evaluation — distribution across seeds/runs")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] saved {out_path}")


def write_runs_summary(runs: list[dict[str, Any]], out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for r in runs:
        timesteps = r["timesteps"]
        last_eval_mean = (
            float(r["ep_rew_mean"][-1]) if r["ep_rew_mean"] is not None else float("nan")
        )
        last_eval_std = (
            float(r["ep_rew_std"][-1]) if r["ep_rew_std"] is not None else float("nan")
        )
        last_step = int(timesteps[-1]) if timesteps is not None and len(timesteps) > 0 else None
        rows.append(
            {
                "run": r["name"],
                "algo": _group_label(r),
                "seed": r["seed"],
                "last_step": last_step,
                "last_eval_mean": last_eval_mean,
                "last_eval_std": last_eval_std,
                "n_final_eval_runs": len(r["summaries"]),
                "final_eval_mean_avg": (
                    float(np.mean([s["mean"] for s in r["summaries"]]))
                    if r["summaries"]
                    else float("nan")
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values(by=["algo", "seed", "run"], na_position="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[plot] saved {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate experiments/* into figures & tables.")
    p.add_argument("--experiments-root", type=Path, default=ROOT / "experiments")
    p.add_argument("--figures-dir", type=Path, default=ROOT / "reports" / "figures")
    p.add_argument("--tables-dir", type=Path, default=ROOT / "reports" / "tables")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.experiments_root.resolve()
    runs: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        rec = _read_run(child)
        if rec is not None:
            runs.append(rec)
        for sub in sorted(child.iterdir()):
            if sub.is_dir() and (sub / "config.yaml").exists():
                rec2 = _read_run(sub)
                if rec2 is not None:
                    runs.append(rec2)

    if not runs:
        print(f"[plot] no runs under {root}")
        return

    plot_learning_curves(runs, args.figures_dir / "learning_curves.png")
    plot_eval_distribution(runs, args.figures_dir / "eval_distribution.png")
    write_runs_summary(runs, args.tables_dir / "runs_summary.csv")


if __name__ == "__main__":
    main()
