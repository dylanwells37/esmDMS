"""Dataset-level fitness baselines and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .schema import Dataset, FeatureArtifact, SEQUENCE_ID


def enrichment_fitness(
    dataset: Dataset, *, pseudocount: float = 0.5
) -> FeatureArtifact:
    """Mean log2 endpoint enrichment across replicates."""
    observed = set(dataset.trajectory[SEQUENCE_ID].astype(str))
    sequence_ids = tuple(
        sequence_id for sequence_id in dataset.sequence_ids if sequence_id in observed
    )
    values_by_replicate = []
    for _, frame in dataset.trajectory.groupby("Replicate", sort=True):
        generations = np.sort(frame["Generation"].unique())
        endpoint = frame[frame["Generation"].isin((generations[0], generations[-1]))]
        counts = endpoint.pivot_table(
            index=SEQUENCE_ID,
            columns="Generation",
            values="Frequency",
            aggfunc="sum",
            fill_value=0.0,
        ).reindex(sequence_ids, fill_value=0.0)
        initial = counts[generations[0]].to_numpy(dtype=float)
        final = counts[generations[-1]].to_numpy(dtype=float)
        initial_probability = (initial + pseudocount) / (
            initial.sum() + pseudocount * len(initial)
        )
        final_probability = (final + pseudocount) / (
            final.sum() + pseudocount * len(final)
        )
        values_by_replicate.append(np.log2(final_probability / initial_probability))
    values = np.mean(values_by_replicate, axis=0)
    return FeatureArtifact(
        sequence_ids,
        values[:, None],
        ("fitness",),
        "fitness",
        dataset.name,
        {"method": "endpoint_enrichment", "pseudocount": pseudocount},
    )


def functional_score_fitness(dataset: Dataset) -> FeatureArtifact:
    if "functional_score" not in dataset.variants:
        raise ValueError("variants.csv does not contain functional_score.")
    values = pd.to_numeric(
        dataset.variants["functional_score"], errors="coerce"
    ).to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("functional_score contains no finite values.")
    sequence_ids = tuple(np.asarray(dataset.sequence_ids)[finite])
    return FeatureArtifact(
        sequence_ids,
        values[finite, None],
        ("fitness",),
        "fitness",
        dataset.name,
        {"method": "functional_score"},
    )


def evaluate_fitness(
    dataset: Dataset,
    fitness: FeatureArtifact,
    *,
    min_review_stars: int = 0,
    pathogenic_high: bool | None = None,
) -> dict[str, float | int]:
    """Return assay-oriented ClinVar AUC and functional-score Spearman rho."""
    if fitness.kind != "fitness" or fitness.dataset != dataset.name:
        raise ValueError("Expected a fitness artifact for this dataset.")
    variants = dataset.variants.set_index(SEQUENCE_ID)
    scores = fitness.scalar_series().rename("fitness").to_frame().join(variants)
    output: dict[str, float | int] = {
        "auc": float("nan"),
        "n_benign": 0,
        "n_pathogenic": 0,
        "spearman_rho": float("nan"),
    }

    if {"annotation", "review_stars"}.issubset(scores.columns):
        annotation = scores["annotation"].astype(str).str.lower()
        review_stars = pd.to_numeric(scores["review_stars"], errors="coerce")
        keep = annotation.isin(("benign", "pathogenic")) & review_stars.ge(
            min_review_stars
        )
        selected = scores[keep]
        labels = annotation[keep].eq("pathogenic").to_numpy()
        oriented = selected["fitness"].to_numpy(dtype=float)
        pathogenic_high = (
            dataset.pathogenic_high_selection
            if pathogenic_high is None
            else pathogenic_high
        )
        if not pathogenic_high:
            oriented = -oriented
        n_positive = int(labels.sum())
        n_negative = int((~labels).sum())
        output["n_pathogenic"] = n_positive
        output["n_benign"] = n_negative
        if n_positive and n_negative:
            ranks = rankdata(oriented, method="average")
            output["auc"] = float(
                (ranks[labels].sum() - n_positive * (n_positive + 1) / 2)
                / (n_positive * n_negative)
            )

    if "functional_score" in scores:
        functional = pd.to_numeric(
            scores["functional_score"], errors="coerce"
        ).to_numpy(dtype=float)
        inferred = scores["fitness"].to_numpy(dtype=float)
        finite = np.isfinite(functional) & np.isfinite(inferred)
        if finite.sum() >= 3:
            output["spearman_rho"] = float(
                spearmanr(inferred[finite], functional[finite]).statistic
            )
    return output
