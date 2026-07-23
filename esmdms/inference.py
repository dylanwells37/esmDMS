"""popDMS inference for any canonical feature artifact, with optional priors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator, cg
from scipy.stats import pearsonr

from .schema import Dataset, FeatureArtifact, SEQUENCE_ID


NO_SUBSTITUTION = -1
FeatureBasis = "FeatureArtifact | SubstitutionBasis"


@dataclass(frozen=True)
class SubstitutionBasis:
    """One-hot amino-acid substitution basis stored as column indices only.

    Every supported dataset carries one assayed variant per substitution, so the
    dense matrix would be an ``n x n`` permutation matrix. Only the column index
    of each row is kept; ``NO_SUBSTITUTION`` marks rows outside the basis, such
    as the wild-type reference row. This exposes the same read-only attributes
    as :class:`~esmdms.schema.FeatureArtifact` so inference can accept either.
    """

    sequence_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    columns: np.ndarray
    dataset: str
    provenance: dict[str, Any] = field(default_factory=dict)
    kind: str = "basis"

    def __post_init__(self) -> None:
        columns = np.asarray(self.columns, dtype=np.int64)
        if columns.shape != (len(self.sequence_ids),):
            raise ValueError("Exactly one column index is required per sequence.")
        if columns.size and int(columns.max()) >= len(self.feature_names):
            raise ValueError("Column index exceeds the number of basis features.")
        if columns.size and int(columns.min()) < NO_SUBSTITUTION:
            raise ValueError("Column indices must be >= -1.")
        object.__setattr__(self, "columns", columns)

    def project(self, coefficients: np.ndarray) -> np.ndarray:
        """Return ``Z @ coefficients`` without materializing ``Z``."""
        coefficients = np.asarray(coefficients, dtype=float)
        values = np.zeros(len(self.columns), dtype=float)
        present = self.columns >= 0
        values[present] = coefficients[self.columns[present]]
        return values

    def dense(self) -> np.ndarray:
        """Materialize the one-hot matrix. Intended for tests and small data."""
        values = np.zeros(
            (len(self.columns), len(self.feature_names)), dtype=np.float32
        )
        present = np.flatnonzero(self.columns >= 0)
        values[present, self.columns[present]] = 1.0
        return values

    def to_artifact(self) -> FeatureArtifact:
        """Return the equivalent dense artifact. Allocates the full matrix."""
        return FeatureArtifact(
            self.sequence_ids,
            self.dense(),
            self.feature_names,
            "basis",
            self.dataset,
            dict(self.provenance),
        )


@dataclass(frozen=True)
class _DenseRows:
    """Explicit feature rows for one time point."""

    values: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        return self.values @ vector

    def rmatvec(self, weights: np.ndarray) -> np.ndarray:
        return self.values.T @ weights


@dataclass(frozen=True)
class _OneHotRows:
    """One-hot feature rows for one time point, stored as column indices."""

    columns: np.ndarray
    n_features: int

    def __post_init__(self) -> None:
        columns = np.asarray(self.columns, dtype=np.int64)
        present = np.flatnonzero(columns >= 0)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "_present", present)
        object.__setattr__(self, "_present_columns", columns[present])

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        values = np.zeros(len(self.columns), dtype=float)
        values[self._present] = np.asarray(vector, dtype=float)[self._present_columns]
        return values

    def rmatvec(self, weights: np.ndarray) -> np.ndarray:
        return np.bincount(
            self._present_columns,
            weights=np.asarray(weights, dtype=float)[self._present],
            minlength=self.n_features,
        )


@dataclass(frozen=True)
class _Moment:
    integration_weight: float
    rows: _DenseRows | _OneHotRows
    probabilities: np.ndarray
    mean: np.ndarray


@dataclass(frozen=True)
class _ReplicateStatistics:
    dx: np.ndarray
    moments: tuple[_Moment, ...]

    def covariance_matvec(self, vector: np.ndarray) -> np.ndarray:
        result = np.zeros_like(vector, dtype=float)
        for moment in self.moments:
            projected = moment.rows.matvec(vector)
            second = moment.rows.rmatvec(moment.probabilities * projected)
            result += moment.integration_weight * (
                second - moment.mean * float(moment.mean @ vector)
            )
        return result


@dataclass(frozen=True)
class InferenceResult:
    """Per-replicate and joint selection coefficients for a feature basis."""

    dataset: str
    features: FeatureArtifact | SubstitutionBasis
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
        values = intercept + _project(self.features, coefficients)
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


def _project(
    features: FeatureArtifact | SubstitutionBasis, coefficients: np.ndarray
) -> np.ndarray:
    if isinstance(features, SubstitutionBasis):
        return features.project(coefficients)
    return features.values @ coefficients


def substitution_basis(dataset: Dataset) -> SubstitutionBasis:
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
    columns = np.fromiter(
        (column_by_name.get(name, NO_SUBSTITUTION) for name in feature_by_row),
        dtype=np.int64,
        count=len(feature_by_row),
    )
    return SubstitutionBasis(
        dataset.sequence_ids,
        feature_names,
        columns,
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
    dataset: Dataset, features: FeatureArtifact | SubstitutionBasis
) -> tuple[tuple[str, ...], list[_ReplicateStatistics]]:
    if features.dataset != dataset.name:
        raise ValueError("Feature artifact and dataset names do not match.")

    if isinstance(features, SubstitutionBasis):
        n_features = len(features.feature_names)
        row_by_id = {
            sequence_id: int(column)
            for sequence_id, column in zip(features.sequence_ids, features.columns)
        }

        def encode(sequence_ids: list[str]) -> _OneHotRows:
            return _OneHotRows(
                np.fromiter(
                    (row_by_id[value] for value in sequence_ids),
                    dtype=np.int64,
                    count=len(sequence_ids),
                ),
                n_features,
            )

    else:
        values = features.values
        row_by_id = {
            sequence_id: index
            for index, sequence_id in enumerate(features.sequence_ids)
        }

        def encode(sequence_ids: list[str]) -> _DenseRows:
            return _DenseRows(values[[row_by_id[value] for value in sequence_ids]])

    missing = sorted(
        set(dataset.trajectory[SEQUENCE_ID].astype(str)).difference(row_by_id)
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
            rows = encode(list(frame[SEQUENCE_ID].astype(str)))
            mean = rows.rmatvec(probabilities)
            means.append(mean)
            moments.append(
                _Moment(float(integration_weight), rows, probabilities, mean)
            )
        labels.append(str(replicate))
        statistics.append(_ReplicateStatistics(means[-1] - means[0], tuple(moments)))
    return tuple(labels), statistics


def assay_oriented_scores(dataset: Dataset, prior: FeatureArtifact) -> pd.Series:
    """Rotate raw prior scores so that higher always means higher assay selection.

    This is the single place the ``orientation`` provenance field and the
    dataset's selection direction are combined. Both prior-guided inference and
    the raw-prior baseline use it, so they can never disagree on direction.
    """
    if prior.kind != "llr_prior":
        raise ValueError("popDMS coefficient priors must have kind='llr_prior'.")
    if prior.dataset != dataset.name:
        raise ValueError("Prior and dataset names do not match.")
    scores = prior.scalar_series()
    orientation = prior.provenance.get("orientation", "selection")
    if orientation == "raw_llr":
        return scores * (-1.0 if dataset.pathogenic_high_selection else 1.0)
    if orientation != "selection":
        raise ValueError(f"Unsupported LLR prior orientation {orientation!r}.")
    return scores


def prior_vector(
    dataset: Dataset,
    basis: FeatureArtifact | SubstitutionBasis,
    prior: FeatureArtifact | None,
) -> np.ndarray:
    """Align an assay-oriented prior to substitution-basis column order."""
    if prior is None:
        return np.zeros(len(basis.feature_names), dtype=float)
    scores = assay_oriented_scores(dataset, prior)
    variants = dataset.variants.set_index(SEQUENCE_ID)
    unknown = scores.index.difference(variants.index)
    if len(unknown):
        raise ValueError(
            f"Prior contains unknown SequenceIndex {sorted(unknown)[0]!r}."
        )
    rows = variants.loc[scores.index]
    feature_names = (
        rows["position"].astype(int).astype(str)
        + ":"
        + rows["mutant_aa"].astype(str)
    )
    grouped = (
        pd.DataFrame(
            {
                "feature": feature_names.to_numpy(),
                "score": scores.to_numpy(dtype=float),
            }
        )
        .groupby("feature")["score"]
        .agg(["first", "min", "max"])
    )
    inconsistent = grouped.index[~np.isclose(grouped["min"], grouped["max"])]
    if len(inconsistent):
        raise ValueError(
            f"Prior has inconsistent scores for substitution {inconsistent[0]!r}."
        )
    aligned = grouped["first"].reindex(list(basis.feature_names))
    missing = int(aligned.isna().sum())
    if missing:
        raise ValueError(f"Prior is missing {missing} substitution features.")
    return aligned.to_numpy(dtype=float)


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
        selection, info = cg(
            operator, rhs, tol=1e-8, atol=0.0, maxiter=max(2000, dimension * 10)
        )
    if info != 0:
        raise RuntimeError(f"Conjugate-gradient solve failed with status {info}.")
    return selection


@dataclass(frozen=True)
class InferenceProblem:
    """Replicate moments for one dataset and feature basis, reusable across fits.

    The moments depend only on the trajectory and the feature basis, never on
    ``gamma`` or the prior, so a gamma or alpha-by-gamma sweep builds this once
    and calls :meth:`solve` per grid point.
    """

    dataset: str
    features: FeatureArtifact | SubstitutionBasis
    replicate_labels: tuple[str, ...]
    statistics: tuple[_ReplicateStatistics, ...]

    @property
    def n_features(self) -> int:
        return len(self.features.feature_names)

    def solve(
        self, *, gamma: float, prior_values: np.ndarray | None = None
    ) -> InferenceResult:
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if prior_values is None:
            prior_values = np.zeros(self.n_features, dtype=float)
        else:
            prior_values = np.asarray(prior_values, dtype=float)
            if prior_values.shape != (self.n_features,):
                raise ValueError(
                    f"Prior vector has shape {prior_values.shape}; expected "
                    f"({self.n_features},)."
                )
        replicate_coefficients = np.vstack(
            [_solve((item,), gamma, prior_values) for item in self.statistics]
        )
        joint_coefficients = _solve(self.statistics, gamma, prior_values)
        return InferenceResult(
            self.dataset,
            self.features,
            float(gamma),
            self.replicate_labels,
            replicate_coefficients,
            joint_coefficients,
            prior_values,
        )


def build_problem(
    dataset: Dataset, features: FeatureArtifact | SubstitutionBasis
) -> InferenceProblem:
    """Precompute the replicate moments shared by every gamma and alpha."""
    labels, statistics = _statistics(dataset, features)
    return InferenceProblem(dataset.name, features, labels, tuple(statistics))


def infer(
    dataset: Dataset,
    features: FeatureArtifact | SubstitutionBasis,
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
    prior_values = float(prior_scale) * prior_vector(dataset, features, prior)
    return build_problem(dataset, features).solve(
        gamma=gamma, prior_values=prior_values
    )


def prior_sweep(
    dataset: Dataset,
    prior: FeatureArtifact,
    *,
    alphas: Iterable[float],
    gammas: Iterable[float],
    evaluate: Callable[[FeatureArtifact], dict[str, float | int]] | None = None,
    basis: SubstitutionBasis | None = None,
    problem: InferenceProblem | None = None,
) -> pd.DataFrame:
    """Evaluate an alpha-by-gamma LLR-prior grid on the substitution basis.

    ``basis`` and ``problem`` may be supplied to reuse a substitution basis and
    its precomputed moments across several priors on the same dataset; the moments
    dominate the cost for large assays (MSH2), so rebuilding them per prior is
    wasteful. When omitted they are constructed here as before.
    """
    if basis is None:
        basis = substitution_basis(dataset)
    if problem is None:
        problem = build_problem(dataset, basis)
    base_prior = prior_vector(dataset, basis, prior)
    rows = []
    for alpha in alphas:
        prior_values = float(alpha) * base_prior
        for gamma in gammas:
            result = problem.solve(gamma=float(gamma), prior_values=prior_values)
            row: dict[str, float | int] = {
                "alpha": float(alpha),
                "gamma": float(gamma),
                "cross_replicate_consistency": result.cross_replicate_consistency,
            }
            if evaluate is not None:
                row.update(evaluate(result.fitness()))
            rows.append(row)
    return pd.DataFrame(rows)


def matched_alpha_grid(
    dataset: Dataset,
    prior: FeatureArtifact,
    *,
    reference_gamma: float,
    basis: SubstitutionBasis | None = None,
    problem: InferenceProblem | None = None,
    scale_multiples: Iterable[float] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    include_unscaled_llr: bool = True,
) -> tuple[list[float], list[dict[str, Any]], dict[str, float]]:
    """Build a scale-matched prior-strength (alpha) grid for the prior sweep.

    The LLR prior is scaled so its coefficient spread matches the regular-popDMS
    selection coefficients: with ``s* = std(c) / std(p)`` (``c`` the zero-prior
    joint coefficients at ``reference_gamma``, ``p`` the assay-oriented,
    basis-aligned LLR vector), ``std(s*·p) == std(c)``. The returned alphas are
    the zero-prior control, ``s*`` times each requested multiple, and optionally
    the unscaled raw-LLR magnitude (alpha = 1.0).

    Returns ``(alphas, records, meta)`` where ``records`` carries per-alpha
    semantics (``scale_multiple``, ``unscaled_raw_llr``) and ``meta`` carries the
    scalar ``matched_scale``/``sigma_coeff``/``sigma_prior``/``reference_gamma``.
    """
    if basis is None:
        basis = substitution_basis(dataset)
    if problem is None:
        problem = build_problem(dataset, basis)
    coefficients = problem.solve(gamma=float(reference_gamma)).joint_coefficients
    sigma_c = float(np.std(coefficients))
    prior_values = prior_vector(dataset, basis, prior)
    sigma_p = float(np.std(prior_values))
    s_star = sigma_c / sigma_p if sigma_p > 0 else 0.0

    records: list[dict[str, Any]] = [
        {"alpha": 0.0, "scale_multiple": 0.0, "unscaled_raw_llr": False}
    ]
    for multiple in scale_multiples:
        records.append(
            {
                "alpha": s_star * float(multiple),
                "scale_multiple": float(multiple),
                "unscaled_raw_llr": False,
            }
        )
    if include_unscaled_llr:
        records.append(
            {
                "alpha": 1.0,
                "scale_multiple": (1.0 / s_star) if s_star > 0 else float("nan"),
                "unscaled_raw_llr": True,
            }
        )
    alphas = [record["alpha"] for record in records]
    meta = {
        "matched_scale": s_star,
        "sigma_coeff": sigma_c,
        "sigma_prior": sigma_p,
        "reference_gamma": float(reference_gamma),
    }
    return alphas, records, meta
