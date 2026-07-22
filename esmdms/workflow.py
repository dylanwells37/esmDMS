"""Configuration-driven multi-dataset analysis used by the repository notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .inference import infer, prior_sweep, substitution_basis
from .metrics import enrichment_fitness, evaluate_fitness, functional_score_fitness
from .schema import Dataset, FeatureArtifact


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config = json.loads(path.read_text())
    config["_config_dir"] = str(path.resolve().parent)
    return config


def _resolve(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_config_dir"]) / path


def _grid(config: dict[str, Any], name: str, default: list[float]) -> np.ndarray:
    value = config.get(name, default)
    if isinstance(value, dict):
        return np.logspace(
            float(value["start"]), float(value["stop"]), int(value["num"])
        )
    return np.asarray(value, dtype=float)


def _metric_columns(
    dataset: Dataset, fitness: FeatureArtifact, cutoffs: list[int]
) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for cutoff in cutoffs:
        metrics = evaluate_fitness(dataset, fitness, min_review_stars=cutoff)
        for name, value in metrics.items():
            output[f"{name}_stars_{cutoff}"] = value
    return output


def _best_gamma_baseline(
    dataset: Dataset,
    features: FeatureArtifact,
    gammas: np.ndarray,
    cutoffs: list[int],
    label: str,
) -> tuple[pd.DataFrame, FeatureArtifact]:
    rows = []
    results = []
    for gamma in gammas:
        result = infer(dataset, features, gamma=float(gamma))
        fitness = result.fitness()
        rows.append(
            {
                "dataset": dataset.name,
                "method": label,
                "gamma": float(gamma),
                "cross_replicate_consistency": result.cross_replicate_consistency,
                **_metric_columns(dataset, fitness, cutoffs),
            }
        )
        results.append(fitness)
    table = pd.DataFrame(rows)
    consistency = table["cross_replicate_consistency"].to_numpy(dtype=float)
    best_index = int(np.nanargmax(consistency)) if np.isfinite(consistency).any() else 0
    return table, results[best_index]


def run_analysis(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Run all configured datasets and write tidy tables under one output directory."""
    output_dir = _resolve(config, config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    gammas = _grid(config, "gammas", list(np.logspace(-5, 4, 52)))
    alphas = _grid(config, "alphas", [0.0, 0.25, 0.5, 1.0, 2.0])
    cutoffs = [int(value) for value in config.get("review_star_cutoffs", [0, 1, 2, 3])]

    sweep_tables = []
    baseline_tables = []
    for dataset_spec in config["datasets"]:
        dataset = Dataset.load(_resolve(config, dataset_spec["path"]))
        basis = substitution_basis(dataset)

        baseline_rows = []
        enrichment = enrichment_fitness(dataset)
        baseline_rows.append(
            {
                "dataset": dataset.name,
                "method": "Enrichment ratio",
                **_metric_columns(dataset, enrichment, cutoffs),
            }
        )
        if (
            "functional_score" in dataset.variants
            and pd.to_numeric(dataset.variants["functional_score"], errors="coerce")
            .notna()
            .any()
        ):
            functional = functional_score_fitness(dataset)
            baseline_rows.append(
                {
                    "dataset": dataset.name,
                    "method": "DMS functional score",
                    **_metric_columns(dataset, functional, cutoffs),
                }
            )

        regular_table, regular_fitness = _best_gamma_baseline(
            dataset, basis, gammas, cutoffs, "Regular popDMS"
        )
        regular_table.to_csv(
            output_dir / f"{dataset.name}__regular_popdms_gamma.csv", index=False
        )
        baseline_rows.append(
            {
                "dataset": dataset.name,
                "method": "Regular popDMS",
                **_metric_columns(dataset, regular_fitness, cutoffs),
            }
        )

        for label, artifact_path in dataset_spec.get("features", {}).items():
            features = FeatureArtifact.load(_resolve(config, artifact_path))
            table, best_fitness = _best_gamma_baseline(
                dataset, features, gammas, cutoffs, label
            )
            table.to_csv(
                output_dir / f"{dataset.name}__{_safe_name(label)}__gamma.csv",
                index=False,
            )
            baseline_rows.append(
                {
                    "dataset": dataset.name,
                    "method": label,
                    **_metric_columns(dataset, best_fitness, cutoffs),
                }
            )

        for label, artifact_path in dataset_spec.get("priors", {}).items():
            raw_prior = FeatureArtifact.load(_resolve(config, artifact_path))
            raw_fitness = FeatureArtifact(
                raw_prior.sequence_ids,
                raw_prior.values,
                ("fitness",),
                "fitness",
                dataset.name,
                {"method": label},
            )
            raw_metrics: dict[str, float | int] = {}
            for cutoff in cutoffs:
                metrics = evaluate_fitness(
                    dataset, raw_fitness, min_review_stars=cutoff, pathogenic_high=False
                )
                for name, value in metrics.items():
                    raw_metrics[f"{name}_stars_{cutoff}"] = value
            baseline_rows.append(
                {"dataset": dataset.name, "method": f"Raw {label}", **raw_metrics}
            )

            sweep = prior_sweep(
                dataset,
                raw_prior,
                alphas=alphas,
                gammas=gammas,
                evaluate=lambda fitness, current=dataset: _metric_columns(
                    current, fitness, cutoffs
                ),
            )
            sweep.insert(0, "prior", label)
            sweep.insert(0, "dataset", dataset.name)
            sweep.to_csv(
                output_dir / f"{dataset.name}__{_safe_name(label)}__prior_sweep.csv",
                index=False,
            )
            sweep_tables.append(sweep)

        baseline_tables.append(pd.DataFrame(baseline_rows))

    sweeps = (
        pd.concat(sweep_tables, ignore_index=True) if sweep_tables else pd.DataFrame()
    )
    baselines = (
        pd.concat(baseline_tables, ignore_index=True)
        if baseline_tables
        else pd.DataFrame()
    )
    sweeps.to_csv(output_dir / "prior_sweeps.csv", index=False)
    baselines.to_csv(output_dir / "baselines.csv", index=False)

    summary = pd.DataFrame()
    if not sweeps.empty:
        primary_auc = f"auc_stars_{cutoffs[0]}"
        summary = (
            sweeps.sort_values(
                [primary_auc, "alpha", "gamma"], ascending=[False, True, True]
            )
            .groupby(["dataset", "prior"], as_index=False)
            .first()
        )
        summary.to_csv(output_dir / "summary.csv", index=False)
    return {"sweeps": sweeps, "baselines": baselines, "summary": summary}


def run_analysis_file(path: str | Path) -> dict[str, pd.DataFrame]:
    return run_analysis(load_config(path))


def _safe_name(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_" for character in value
    ).strip("_")
