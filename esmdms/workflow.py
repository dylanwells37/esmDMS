"""Configuration-driven multi-dataset analysis used by the repository notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .inference import (
    InferenceProblem,
    SubstitutionBasis,
    assay_oriented_scores,
    build_problem,
    matched_alpha_grid,
    prior_sweep,
    substitution_basis,
)
from .metrics import enrichment_fitness, evaluate_fitness, functional_score_fitness
from .regularization import (
    DEFAULT_CORR_CUTOFF_PCT,
    get_best_regularization,
    popdms_gamma_grid,
)
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
    cutoffs: list[int],
    label: str,
    *,
    corr_cutoff_pct: float = DEFAULT_CORR_CUTOFF_PCT,
    problem: InferenceProblem | None = None,
) -> tuple[pd.DataFrame, FeatureArtifact, float, float]:
    """Sweep gamma on the popDMS grid and select by the correlation elbow.

    The gamma is chosen exactly as canonical popDMS does: over the ascending grid
    ``logspace(log10(1/max_reads), 4, 20)`` the cross-replicate correlation curve
    is fed to :func:`get_best_regularization`, which walks down from the peak to
    the elbow rather than taking the (over-regularized) argmax. ``problem`` may be
    passed to reuse precomputed moments.
    """
    if problem is None:
        problem = build_problem(dataset, features)
    gammas = popdms_gamma_grid(dataset)
    rows = []
    consistency = []
    for gamma in gammas:
        result = problem.solve(gamma=float(gamma))
        consistency.append(result.cross_replicate_consistency)
        rows.append(
            {
                "dataset": dataset.name,
                "method": label,
                "gamma": float(gamma),
                "cross_replicate_consistency": result.cross_replicate_consistency,
                **_metric_columns(dataset, result.fitness(), cutoffs),
            }
        )
    table = pd.DataFrame(rows)
    if len(problem.replicate_labels) < 2 or not np.isfinite(consistency).any():
        # popDMS fixes gamma = 1 when the replicate correlation is undefined.
        best_gamma = 1.0
    else:
        best_gamma = float(
            get_best_regularization(consistency, gammas, corr_cutoff_pct)
        )
    best_result = problem.solve(gamma=best_gamma)
    return (
        table,
        best_result.fitness(),
        best_gamma,
        float(best_result.cross_replicate_consistency),
    )


def run_analysis(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Run all configured datasets and write tidy tables under one output directory."""
    output_dir = _resolve(config, config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    gammas = _grid(config, "gammas", list(np.logspace(-5, 4, 52)))
    cutoffs = [int(value) for value in config.get("review_star_cutoffs", [0, 1, 2, 3])]

    # Gamma for the zero-prior baselines is chosen by the popDMS correlation
    # elbow (config default) rather than argmax; corr_cutoff_pct tunes the elbow.
    corr_cutoff_pct = float(config.get("corr_cutoff_pct", DEFAULT_CORR_CUTOFF_PCT))
    # The prior-strength (alpha) axis is scale-matched to the popDMS coefficients
    # by default; "fixed" reproduces the legacy raw-multiplier behaviour.
    alpha_mode = config.get("alpha_mode", "matched")
    fixed_alphas = _grid(config, "alphas", [0.0, 0.25, 0.5, 1.0, 2.0])
    scale_multiples = _grid(
        config, "alpha_scale_multiples", [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    )
    include_unscaled_llr = bool(config.get("include_unscaled_llr", True))

    sweep_tables = []
    baseline_tables = []
    for dataset_spec in config["datasets"]:
        dataset = Dataset.load(_resolve(config, dataset_spec["path"]))
        basis = substitution_basis(dataset)
        # Precompute the substitution-basis moments once; regular popDMS, the
        # matched-scale reference, and every prior sweep on this dataset reuse it.
        basis_problem = build_problem(dataset, basis)

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
            _best_gamma_baseline(
                dataset,
                basis,
                cutoffs,
                "Regular popDMS",
                corr_cutoff_pct=corr_cutoff_pct,
                problem=basis_problem,
            )
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
                dataset, features, cutoffs, label, corr_cutoff_pct=corr_cutoff_pct
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

            if alpha_mode == "matched":
                sweep_alphas, alpha_records, scale_meta = matched_alpha_grid(
                    dataset,
                    raw_prior,
                    reference_gamma=regular_gamma,
                    basis=basis,
                    problem=basis_problem,
                    scale_multiples=scale_multiples,
                    include_unscaled_llr=include_unscaled_llr,
                )
            else:
                sweep_alphas = list(fixed_alphas)
                alpha_records = [
                    {
                        "alpha": float(alpha),
                        "scale_multiple": float("nan"),
                        "unscaled_raw_llr": False,
                    }
                    for alpha in sweep_alphas
                ]
                scale_meta = {
                    "matched_scale": float("nan"),
                    "sigma_coeff": float("nan"),
                    "sigma_prior": float("nan"),
                    "reference_gamma": float(regular_gamma),
                }

            sweep = prior_sweep(
                dataset,
                raw_prior,
                alphas=sweep_alphas,
                gammas=gammas,
                evaluate=lambda fitness, current=dataset: _metric_columns(
                    current, fitness, cutoffs
                ),
                basis=basis,
                problem=basis_problem,
            )
            # Attach per-alpha semantics (multiple of the matched scale, and which
            # row is the unscaled raw-LLR point) and the scalar scale metadata.
            attributes = pd.DataFrame(alpha_records).drop_duplicates("alpha")
            sweep = sweep.merge(attributes, on="alpha", how="left")
            for key, value in scale_meta.items():
                sweep[key] = value
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
