#!/usr/bin/env python3
"""Generate per-dataset analyze configs and the job manifests for the experiment.

One JSON config per dataset drives ``esmdms analyze`` with the scale-matched
alpha axis and popDMS elbow gamma selection. Two TSV manifests enumerate the GPU
LLR jobs and the CPU analyze jobs so the controller script stays declarative.
All paths written into the configs and manifests are absolute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The five clinically annotated DMS datasets. BRCA2's C-terminal 2048-residue
# window (truncation 1371-3418) is applied automatically from dataset.json.
DATASETS = [
    "MV_BRCA1_Findlay_2018",
    "MV_VHL_Buckley_2024",
    "MV_TP53_Kotler_2018",
    "MV_MSH2_Jia_2020",
    "MV_BRCA2_Huang_2025",
]

# ESM-C model sizes -> Hugging Face ids, with GPU LLR resource requests.
MODELS = {
    "300M": {"id": "biohub/ESMC-300M", "mem": "32G", "time": "00:30:00"},
    "600M": {"id": "biohub/ESMC-600M", "mem": "48G", "time": "00:45:00"},
    "6B": {"id": "biohub/ESMC-6B", "mem": "64G", "time": "01:00:00"},
}

# MSH2's near-saturating ~17k-feature substitution basis dominates the CPU sweep,
# so it gets a bigger, longer analyze job than the others.
ANALYZE_RESOURCES = {
    "MV_MSH2_Jia_2020": {"mem": "64G", "time": "08:00:00"},
    "_default": {"mem": "32G", "time": "04:00:00"},
}

REVIEW_STAR_CUTOFFS = [0, 1, 2, 3]
ALPHA_SCALE_MULTIPLES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def build_config(dataset: str, repo_root: Path, results_root: Path) -> dict:
    priors = {
        f"ESM-C {size} LLR": str(repo_root / "artifacts" / dataset / f"{size}_llr.npz")
        for size in MODELS
    }
    return {
        "output_dir": str(results_root / dataset),
        "gammas": {"start": -5, "stop": 4, "num": 52},
        "review_star_cutoffs": REVIEW_STAR_CUTOFFS,
        # popDMS elbow gamma selection for the zero-prior baselines.
        "gamma_selection": "popdms_elbow",
        "corr_cutoff_pct": 0.10,
        # Scale-matched prior-strength axis plus the unscaled raw-LLR point.
        "alpha_mode": "matched",
        "alpha_scale_multiples": ALPHA_SCALE_MULTIPLES,
        "include_unscaled_llr": True,
        "datasets": [
            {"path": str(repo_root / "datasets" / dataset), "priors": priors}
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument(
        "--config-dir",
        required=True,
        type=Path,
        help="Directory to write per-dataset JSON configs and manifests.",
    )
    parser.add_argument(
        "--results-root",
        required=True,
        type=Path,
        help="Root under which each dataset's analyze output_dir is placed.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    config_dir = args.config_dir.resolve()
    results_root = args.results_root.resolve()
    config_dir.mkdir(parents=True, exist_ok=True)

    llr_lines = []
    analyze_lines = []
    for dataset in DATASETS:
        config_path = config_dir / f"{dataset}.json"
        config_path.write_text(
            json.dumps(build_config(dataset, repo_root, results_root), indent=2)
        )
        for size, spec in MODELS.items():
            output_npz = repo_root / "artifacts" / dataset / f"{size}_llr.npz"
            llr_lines.append(
                "\t".join([dataset, size, spec["id"], str(output_npz), spec["mem"], spec["time"]])
            )
        resources = ANALYZE_RESOURCES.get(dataset, ANALYZE_RESOURCES["_default"])
        analyze_lines.append(
            "\t".join([dataset, str(config_path), resources["mem"], resources["time"]])
        )

    (config_dir / "llr_jobs.tsv").write_text("\n".join(llr_lines) + "\n")
    (config_dir / "analyze_jobs.tsv").write_text("\n".join(analyze_lines) + "\n")
    (config_dir / "datasets.txt").write_text("\n".join(DATASETS) + "\n")
    print(f"Wrote {len(DATASETS)} configs and manifests to {config_dir}")


if __name__ == "__main__":
    main()
