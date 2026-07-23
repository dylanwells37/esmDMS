#!/usr/bin/env python3
"""Concatenate per-dataset analyze outputs into one aggregate result set.

Each dataset's ``esmdms analyze`` run writes ``baselines.csv``,
``prior_sweeps.csv``, and ``summary.csv`` into its own output directory. This
merges the five into a single aggregate directory the reporting notebook reads,
recomputing the summary from the combined sweeps so ``auc_gain_over_alpha0`` is
consistent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from esmdms.workflow import _summarize


def _concat(results_root: Path, datasets: list[str], name: str) -> pd.DataFrame:
    frames = []
    for dataset in datasets:
        path = results_root / dataset / name
        if path.is_file():
            frames.append(pd.read_csv(path))
        else:
            print(f"warning: missing {path}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--datasets-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--primary-cutoff", type=int, default=0)
    args = parser.parse_args()

    datasets = [
        line.strip()
        for line in args.datasets_file.read_text().splitlines()
        if line.strip()
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines = _concat(args.results_root, datasets, "baselines.csv")
    sweeps = _concat(args.results_root, datasets, "prior_sweeps.csv")
    baselines.to_csv(output_dir / "baselines.csv", index=False)
    sweeps.to_csv(output_dir / "prior_sweeps.csv", index=False)

    if not sweeps.empty:
        summary = _summarize(sweeps, args.primary_cutoff)
        summary.to_csv(output_dir / "summary.csv", index=False)
        print(f"Aggregated {len(datasets)} datasets -> {output_dir}")
    else:
        print("No sweep rows found; wrote empty aggregate.")


if __name__ == "__main__":
    main()
