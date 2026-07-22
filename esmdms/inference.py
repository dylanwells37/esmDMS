"""popDMS inference for any canonical feature artifact, with optional priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator, cg
from scipy.stats import pearsonr

from .schema import Dataset, FeatureArtifact, SEQUENCE_ID


@dataclass(frozen=True)
class _Moment:
    integration_weight: float
    values: np.ndarray
    probabilities: np.ndarray
    mean: np.ndarray


@dataclass(frozen=True)
class _ReplicateStatistics:
    dx: np.ndarray
    moments: tuple[_Moment, ...]

    def covariance_matvec(self, vector: np.ndarray) -> np.ndarray:
        result = np.zeros_like(vector, dtype=float)
        for moment in self.moments:
            second = moment.values.T @ (moment.probabilities * (moment.values @ vector))
            result += moment.integration_weight * (
                second - moment.mean * float(moment.mean @ vector)
            )
        return result


@dataclass(frozen=True)
class InferenceResult:
    """Per-replicate and joint selection coefficients for a feature basis."""

    dataset: str
    features: FeatureArtifact
    gamma: float
    replicate_labels: tuple[str, ...]
    replicate_coefficients: np.ndarray
    joint_coefficients: np.ndarray
    prior: np.ndarray

    @property
    def cross_replicate_consistency(self) -> float:
        correlations = [
            pearsonr(
                self.replicate_coefficients[left], self.replicate_coefficients[right]
            ).statistic
            for left in range(len(self.replicate_coefficients))
            for right in range(left + 1, len(self.replicate_coefficients))
        ]
        return float(np.nanmean(correlations)) if correlations else float("nan")

    def fitness(self, *, joint: bool = True, intercept: float = 1.0) -> FeatureArtifact:
        coefficients = (
            self.joint_coefficients
            if joint
            else self.replicate_coefficients.mean(axis=0)
        )
        values = intercept + self.features.values @ coefficients
        return FeatureArtifact(
            self.features.sequence_ids,
            values[:, None],
            ("fitness",),
            "fitness",
            self.dataset,
            {
                "gamma": self.gamma,
                "source_kind": self.features.kind,
                "source_provenance": self.features.provenance,
            },
        )


def substitution_basis(dataset: Dataset) -> FeatureArtifact:
    """Create a one-hot amino-acid substitution basis for popDMS priors."""
    variants = dataset.variants
    eligible = ~variants["is_synonymous"] & variants["mutant_aa"].astype(str).ne("*")
    feature_by_row = np.where(
        eligible,
        variants["position"].astype(str) + ":" + variants["mutant_aa"].astype(str),
        "",
    )
    feature_names = tuple(dict.fromkeys(value for value in feature_by_row if value))
    if not feature_names:
        raise ValueError("Dataset contains no nonsynonymous non-stop substitutions.")
    column_by_name = {name: index for index, name in enumerate(feature_names)}
    values = np.zeros((len(variants), len(feature_names)), dtype=np.float32)
    for row_index, feature_name in enumerate(feature_by_row):
        if feature_name:
            values[row_index, column_by_name[feature_name]] = 1.0
    return FeatureArtifact(
        dataset.sequence_ids,
        values,
        feature_names,
        "basis",
        dataset.name,
        {"basis": "amino_acid_substitution"},
    )


def _integration_weights(generations: np.ndarray) -> np.ndarray:
    if len(generations) < 2:
        raise ValueError("A replicate needs at least two generations.")
    weights = np.empty(len(generations), dtype=float)
    weights[0] = (generations[1] - generations[0]) / 2
    weights[-1] = (generations[-1] - generations[-2]) / 2
    if len(generations) > 2:
        weights[1:-1] = (generations[2:] - generations[:-2]) / 2
    if np.any(weights <= 0):
        raise ValueError(
            "Generations must be strictly increasing within each replicate."
        )
    return weights


def _statistics(
    dataset: Dataset, features: FeatureArtifact
) -> tuple[tuple[str, ...], list[_ReplicateStatistics]]:
    if features.dataset != dataset.name:
        raise ValueError("Feature artifact and dataset names do not match.")
    feature_by_id = {
        sequence_id: features.values[index]
        for index, sequence_id in enumerate(features.sequence_ids)
    }
    missing = sorted(
        set(dataset.trajectory[SEQUENCE_ID].astype(str)).difference(feature_by_id)
    )
    if missing:
        raise ValueError(f"Features are missing {len(missing)} trajectory sequences.")

    labels: list[str] = []
    statistics: list[_ReplicateStatistics] = []
    for replicate, replicate_frame in dataset.trajectory.groupby(
        "Replicate", sort=True
    ):
        generations = np.sort(replicate_frame["Generation"].unique().astype(float))
        weights = _integration_weights(generations)
        moments = []
        means = []
        for generation, integration_weight in zip(generations, weights):
            frame = replicate_frame[replicate_frame["Generation"].eq(generation)]
            frequencies = frame["Frequency"].to_numpy(dtype=float)
            total = float(frequencies.sum())
            if total <= 0:
                raise ValueError(
                    f"Replicate {replicate!r}, generation {generation:g} has zero total frequency."
                )
            probabilities = frequencies / total
            values = np.vstack(
                [feature_by_id[value] for value in frame[SEQUENCE_ID].astype(str)]
            )
            mean = probabilities @ values
            means.append(mean)
            moments.append(
                _Moment(float(integration_weight), values, probabilities, mean)
            )
        labels.append(str(replicate))
        statistics.append(_ReplicateStatistics(means[-1] - means[0], tuple(moments)))
    return tuple(labels), statistics


def _prior_vector(
    dataset: Dataset, basis: FeatureArtifact, prior: FeatureArtifact | None
) -> np.ndarray:
    if prior is None:
        return np.zeros(len(basis.feature_names), dtype=float)
    if prior.kind != "llr_prior":
        raise ValueError("popDMS coefficient priors must have kind='llr_prior'.")
    if prior.dataset != dataset.name:
        raise ValueError("Prior and dataset names do not match.")
    scores = prior.scalar_series()
    variants = dataset.variants.set_index(SEQUENCE_ID)
    values_by_feature: dict[str, list[float]] = {}
    for sequence_id, score in scores.items():
        if sequence_id not in variants.index:
            raise ValueError(f"Prior contains unknown SequenceIndex {sequence_id!r}.")
        row = variants.loc[sequence_id]
        feature_name = f"{int(row['position'])}:{row['mutant_aa']}"
        values_by_feature.setdefault(feature_name, []).append(float(score))

    missing = [name for name in basis.feature_names if name not in values_by_feature]
    if missing:
        raise ValueError(f"Prior is missing {len(missing)} substitution features.")
    output = []
    for name in basis.feature_names:
        feature_values = np.asarray(values_by_feature[name], dtype=float)
        if not np.allclose(feature_values, feature_values[0]):
            raise ValueError(f"Prior has inconsistent scores for substitution {name}.")
        output.append(float(feature_values[0]))
    values = np.asarray(output)
    orientation = prior.provenance.get("orientation", "selection")
    if orientation == "raw_llr":
        return (-1.0 if dataset.pathogenic_high_selection else 1.0) * values
    if orientation != "selection":
        raise ValueError(f"Unsupported LLR prior orientation {orientation!r}.")
    return values


def _solve(
    statistics: Iterable[_ReplicateStatistics], gamma: float, prior: np.ndarray
) -> np.ndarray:
    statistics = tuple(statistics)
    precision = float(gamma) * len(statistics)
    dimension = len(prior)

    def matvec(vector: np.ndarray) -> np.ndarray:
        return precision * vector + sum(
            (item.covariance_matvec(vector) for item in statistics),
            start=np.zeros(dimension, dtype=float),
        )

    operator = LinearOperator((dimension, dimension), matvec=matvec, dtype=float)
    rhs = (
        sum((item.dx for item in statistics), start=np.zeros(dimension, dtype=float))
        + precision * prior
    )
    try:
        selection, info = cg(
            operator, rhs, rtol=1e-8, atol=0.0, maxiter=max(2000, dimension * 10)
        )
    except TypeError:
        selection, info = cg(operator, rhs, tol=1e-8, maxiter=max(2000, dimension * 10))
    if info != 0:
        raise RuntimeError(f"Conjugate-gradient solve failed with status {info}.")
    return selection


def infer(
    dataset: Dataset,
    features: FeatureArtifact,
    *,
    gamma: float,
    prior: FeatureArtifact | None = None,
    prior_scale: float = 1.0,
) -> InferenceResult:
    """Infer popDMS selection coefficients on an arbitrary feature basis."""
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if prior is not None and features.kind != "basis":
        raise ValueError("LLR priors are defined on the substitution basis.")
    labels, statistics = _statistics(dataset, features)
    prior_values = prior_scale * _prior_vector(dataset, features, prior)
    replicate_coefficients = np.vstack(
        [_solve((item,), gamma, prior_values) for item in statistics]
    )
    joint_coefficients = _solve(statistics, gamma, prior_values)
    return InferenceResult(
        dataset.name,
        features,
        float(gamma),
        labels,
        replicate_coefficients,
        joint_coefficients,
        prior_values,
    )


def prior_sweep(
    dataset: Dataset,
    prior: FeatureArtifact,
    *,
    alphas: Iterable[float],
    gammas: Iterable[float],
    evaluate: Callable[[FeatureArtifact], dict[str, float | int]] | None = None,
) -> pd.DataFrame:
    """Evaluate an alpha-by-gamma LLR-prior grid on the substitution basis."""
    basis = substitution_basis(dataset)
    rows = []
    for alpha in alphas:
        for gamma in gammas:
            result = infer(
                dataset,
                basis,
                gamma=float(gamma),
                prior=prior,
                prior_scale=float(alpha),
            )
            row: dict[str, float | int] = {
                "alpha": float(alpha),
                "gamma": float(gamma),
                "cross_replicate_consistency": result.cross_replicate_consistency,
            }
            if evaluate is not None:
                row.update(evaluate(result.fitness()))
            rows.append(row)
    return pd.DataFrame(rows)
