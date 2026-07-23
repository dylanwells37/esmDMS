"""Configuration-driven multi-dataset analysis used by the repository notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .inference import (
    SubstitutionBasis,
    assay_oriented_scores,
    build_problem,
    prior_sweep,
    substitution_basis,
)
from .metrics import enrichment_fitness, evaluate_fitness, functional_score_fitness
from .schema import Dataset, FeatureArtifact

# Metrics that depend on the ClinVar review-star cutoff and are therefore
# reported once per configured cutoff.
CUTOFF_METRICS = ("auc", "n_benign", "n_pathogenic")
# Metrics that do not depend on the cutoff and are reported exactly once.
GLOBAL_METRICS = ("spearman_rho",)


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
    dataset: Dataset,
    fitness: FeatureArtifact,
    cutoffs: list[int],
    *,
    pathogenic_high: bool | None = None,
) -> dict[str, float | int]:
    """Return cutoff-suffixed ClinVar metrics plus the cutoff-free Spearman rho.

    ``spearman_rho`` compares inferred fitness to the assay's own functional
    score and never consults ClinVar review status, so it is emitted once rather
    than duplicated under a misleading ``_stars_N`` suffix.
    """
    output: dict[str, float | int] = {}
    for index, cutoff in enumerate(cutoffs):
        metrics = evaluate_fitness(
            dataset,
            fitness,
            min_review_stars=cutoff,
            pathogenic_high=pathogenic_high,
        )
        for name in CUTOFF_METRICS:
            output[f"{name}_stars_{cutoff}"] = metrics[name]
        if index == 0:
            for name in GLOBAL_METRICS:
                output[name] = metrics[name]
    return output


def _baseline_row(
    dataset: Dataset,
    method: str,
    metrics: dict[str, float | int],
    *,
    gamma: float = float("nan"),
    consistency: float = float("nan"),
) -> dict[str, Any]:
    """Build one baselines.csv row, always recording the gamma that produced it."""
    return {
        "dataset": dataset.name,
        "method": method,
        "gamma": gamma,
        "cross_replicate_consistency": consistency,
        **metrics,
    }


def _best_gamma_baseline(
    dataset: Dataset,
    features: FeatureArtifact | SubstitutionBasis,
    gammas: np.ndarray,
    cutoffs: list[int],
    label: str,
) -> tuple[pd.DataFrame, FeatureArtifact, float, float]:
    """Sweep gamma once and select by unsupervised cross-replicate consistency."""
    problem = build_problem(dataset, features)
    rows = []
    results = []
    for gamma in gammas:
        result = problem.solve(gamma=float(gamma))
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
    return (
        table,
        results[best_index],
        float(table["gamma"].iloc[best_index]),
        float(consistency[best_index]),
    )


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
            _baseline_row(
                dataset,
                "Enrichment ratio",
                _metric_columns(dataset, enrichment, cutoffs),
            )
        )
        if (
            "functional_score" in dataset.variants
            and pd.to_numeric(dataset.variants["functional_score"], errors="coerce")
            .notna()
            .any()
        ):
            functional = functional_score_fitness(dataset)
            baseline_rows.append(
                _baseline_row(
                    dataset,
                    "DMS functional score",
                    _metric_columns(dataset, functional, cutoffs),
                )
            )

        regular_table, regular_fitness, regular_gamma, regular_consistency = (
            _best_gamma_baseline(dataset, basis, gammas, cutoffs, "Regular popDMS")
        )
        regular_table.to_csv(
            output_dir / f"{dataset.name}__regular_popdms_gamma.csv", index=False
        )
        baseline_rows.append(
            _baseline_row(
                dataset,
                "Regular popDMS",
                _metric_columns(dataset, regular_fitness, cutoffs),
                gamma=regular_gamma,
                consistency=regular_consistency,
            )
        )

        for label, artifact_path in dataset_spec.get("features", {}).items():
            features = FeatureArtifact.load(_resolve(config, artifact_path))
            table, best_fitness, best_gamma, best_consistency = _best_gamma_baseline(
                dataset, features, gammas, cutoffs, label
            )
            table.to_csv(
                output_dir / f"{dataset.name}__{_safe_name(label)}__gamma.csv",
                index=False,
            )
            baseline_rows.append(
                _baseline_row(
                    dataset,
                    label,
                    _metric_columns(dataset, best_fitness, cutoffs),
                    gamma=best_gamma,
                    consistency=best_consistency,
                )
            )

        for label, artifact_path in dataset_spec.get("priors", {}).items():
            raw_prior = FeatureArtifact.load(_resolve(config, artifact_path))
            # Orient the raw prior exactly the way prior-guided inference does, so
            # the baseline AUC and its Spearman rho use one direction convention
            # that is comparable with every other row in the table.
            oriented = assay_oriented_scores(dataset, raw_prior)
            raw_fitness = FeatureArtifact(
                tuple(oriented.index.astype(str)),
                oriented.to_numpy(dtype=float)[:, None],
                ("fitness",),
                "fitness",
                dataset.name,
                {"method": label, "orientation": "assay_selection"},
            )
            baseline_rows.append(
                _baseline_row(
                    dataset,
                    f"Raw {label}",
                    _metric_columns(dataset, raw_fitness, cutoffs),
                )
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
        summary = _summarize(sweeps, cutoffs[0])
        summary.to_csv(output_dir / "summary.csv", index=False)
    return {"sweeps": sweeps, "baselines": baselines, "summary": summary}


def _summarize(sweeps: pd.DataFrame, primary_cutoff: int) -> pd.DataFrame:
    """Select the highest primary-cutoff AUC row per dataset and prior.

    ``head(1)`` keeps one intact row. ``GroupBy.first()`` must not be used here:
    it returns the first non-null value of each column independently, so a NaN
    in the winning row would be backfilled from a different alpha/gamma.
    """
    primary_auc = f"auc_stars_{primary_cutoff}"
    ranked = sweeps.sort_values(
        [primary_auc, "alpha", "gamma"], ascending=[False, True, True]
    )
    summary = (
        ranked.groupby(["dataset", "prior"], sort=False)
        .head(1)
        .sort_values(["dataset", "prior"])
        .reset_index(drop=True)
    )
    # The alpha = 0 rows are the regularized zero-mean popDMS control, so the
    # gain over them is what the LLR prior actually bought at this cutoff.
    control = (
        ranked[ranked["alpha"] == 0.0]
        .groupby(["dataset", "prior"], sort=False)[primary_auc]
        .max()
        .rename("auc_alpha0")
    )
    summary = summary.merge(control, on=["dataset", "prior"], how="left")
    summary["auc_gain_over_alpha0"] = summary[primary_auc] - summary["auc_alpha0"]
    summary["prior_used"] = summary["alpha"] > 0.0
    return summary


def run_analysis_file(path: str | Path) -> dict[str, pd.DataFrame]:
    return run_analysis(load_config(path))


def _safe_name(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_" for character in value
    ).strip("_")
