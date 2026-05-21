"""학습/평가 결과 로깅 보조."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


def write_episode_csv(
    csv_path: str | Path,
    *,
    rewards: Iterable[float],
    lengths: Iterable[int] | None = None,
    meta: Mapping[str, str] | None = None,
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    lengths_list = list(lengths) if lengths is not None else None
    rewards_list = list(rewards)
    if lengths_list is not None and len(lengths_list) != len(rewards_list):
        raise ValueError("rewards/lengths length mismatch")

    meta = dict(meta or {})
    base_fields = ["episode", "reward"] + (["length"] if lengths_list is not None else [])
    meta_fields = list(meta.keys())

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(base_fields + meta_fields)
        for i, r in enumerate(rewards_list):
            row = [i, float(r)]
            if lengths_list is not None:
                row.append(int(lengths_list[i]))
            row.extend(str(meta[k]) for k in meta_fields)
            writer.writerow(row)
