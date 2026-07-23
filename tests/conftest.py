from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esmdms.schema import Dataset, FeatureArtifact


@pytest.fixture
def dataset() -> Dataset:
    return synthetic_dataset()


def synthetic_dataset() -> Dataset:
    variants = pd.DataFrame(
        {
            "SequenceIndex": ["wt", "v1", "v1b", "v2", "v3"],
            "protein_sequence": ["AAA", "CAA", "CAA", "ADA", "AAE"],
            "position": [1, 1, 1, 2, 3],
            "wt_aa": ["A"] * 5,
            "mutant_aa": ["A", "C", "C", "D", "E"],
            "is_synonymous": [True, False, False, False, False],
            "functional_score": [1.0, -1.0, -0.9, -0.3, 0.5],
            "annotation": [
                "benign",
                "pathogenic",
                "pathogenic",
                "pathogenic",
                "benign",
            ],
            "review_stars": [2, 2, 1, 2, 1],
        }
    )
    counts = {
        ("r1", 0): [70, 5, 5, 10, 10],
        ("r1", 1): [40, 15, 15, 20, 10],
        ("r2", 0): [60, 8, 7, 15, 10],
        ("r2", 1): [35, 12, 13, 25, 15],
    }
    trajectory = pd.DataFrame(
        [
            {
                "SequenceIndex": sequence_id,
                "Replicate": replicate,
                "Generation": generation,
                "Frequency": frequency,
            }
            for (replicate, generation), values in counts.items()
            for sequence_id, frequency in zip(variants["SequenceIndex"], values)
        ]
    )
    return Dataset("synthetic", "AAA", variants, trajectory)


def synthetic_prior(dataset: Dataset, **provenance) -> FeatureArtifact:
    return FeatureArtifact(
        ("v1", "v1b", "v2", "v3"),
        np.asarray([[-2.0], [-2.0], [-1.0], [-0.25]]),
        ("llr",),
        "llr_prior",
        dataset.name,
        provenance or {"model": "test"},
    )
