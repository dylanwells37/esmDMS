from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from esmdms.features import (
    SAEConfig,
    analysis_window,
    embed,
    embed_and_llr,
    masked_marginal_llr,
    train_sae,
)
from esmdms.inference import infer, prior_sweep, substitution_basis
from esmdms.import_data import process_imported_csv
from esmdms.metrics import (
    enrichment_fitness,
    evaluate_fitness,
    functional_score_fitness,
)
from esmdms.schema import Dataset, FeatureArtifact
from esmdms.workflow import run_analysis
from examples.brca1_100_demo import build_subset_dataset


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


def synthetic_prior(dataset: Dataset) -> FeatureArtifact:
    return FeatureArtifact(
        ("v1", "v1b", "v2", "v3"),
        np.asarray([[-2.0], [-2.0], [-1.0], [-0.25]]),
        ("llr",),
        "llr_prior",
        dataset.name,
        {"model": "test"},
    )


def test_dataset_and_artifact_round_trip(tmp_path):
    source = synthetic_dataset()
    dataset = Dataset(
        source.name,
        source.reference_sequence,
        source.variants,
        source.trajectory,
        truncation=(1, 3),
    )
    dataset.save(tmp_path / "dataset")
    loaded_dataset = Dataset.load(tmp_path / "dataset")
    assert loaded_dataset.sequence_ids == dataset.sequence_ids
    assert loaded_dataset.reference_sequence == "AAA"
    assert loaded_dataset.truncation == (1, 3)
    manifest = json.loads((tmp_path / "dataset" / "dataset.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["truncation"] == {"start": 1, "end": 3}

    artifact = synthetic_prior(dataset)
    artifact.save(tmp_path / "prior.npz")
    loaded_artifact = FeatureArtifact.load(tmp_path / "prior.npz")
    assert loaded_artifact.kind == "llr_prior"
    np.testing.assert_array_equal(loaded_artifact.values, artifact.values)


def test_brca1_demo_subset_keeps_reference_and_selected_trajectories(tmp_path):
    source = synthetic_dataset()
    subset = build_subset_dataset(source, tmp_path / "subset", variant_count=2)

    assert subset.name == "synthetic__first2"
    assert subset.sequence_ids == ("wt", "v1", "v1b")
    assert subset.metadata["selection"]["variant_count"] == 2
    assert set(subset.trajectory["SequenceIndex"]) == {"v1", "v1b"}
    assert not subset.trajectory["SequenceIndex"].eq("wt").any()
    Dataset.load(tmp_path / "subset")


def test_imported_csv_processor_builds_canonical_dataset(tmp_path):
    frame = pd.DataFrame(
        {
            "mutant": ["A1C", "A2D", "A3*"],
            "mutated_sequence": ["CAA", "ADA", "AA"],
            "has_clinvar": ["true", "false", "true"],
            "clinvar_significance": ["Pathogenic", "", "Pathogenic"],
            "clinvar_significance_normalized": ["pathogenic", "", "pathogenic"],
            "clinvar_review_status": [
                "reviewed by expert panel",
                "",
                "criteria provided, single submitter",
            ],
            "clinvar_conditions": ["condition", "", "condition"],
            "clinvar_variation_ids": ["1", "", "2"],
            "count__count_day11_rep1": [30, 20, 1],
            "count__count_day11_rep2": [31, 21, 1],
            "count__count_day5_rep1": [20, 15, 1],
            "count__count_day5_rep2": [21, 16, 1],
            "count__count_library": [10, 10, 1],
            "count__count_negative_control": [0, 0, 0],
            "count__count_rna_rep1": [1, 1, 0],
            "count__count_rna_rep2": [1, 1, 0],
            "functional_score": [-1.0, 0.5, -2.0],
        }
    )
    source = tmp_path / "MV_BRCA1_Findlay_2018.csv"
    frame.to_csv(source, index=False)
    dataset = process_imported_csv(source, tmp_path / "datasets")

    assert dataset.reference_sequence == "AAA"
    assert dataset.metadata["dropped_stop_rows"] == 1
    assert dataset.metadata["source_file"] == source.name
    assert len(dataset.metadata["source_file_sha256"]) == 64
    assert dataset.variants.iloc[0]["SequenceIndex"] == "__wildtype__"
    assert len(dataset.variants) == 3
    assert len(dataset.trajectory) == 12
    assert set(dataset.trajectory["Replicate"]) == {"rep1", "rep2"}
    assert set(dataset.trajectory["Generation"]) == {0.0, 5.0, 11.0}
    assert not dataset.trajectory["SequenceIndex"].eq("__wildtype__").any()
    assert sorted(
        path.name for path in (tmp_path / "datasets" / source.stem).iterdir()
    ) == [
        "dataset.json",
        "trajectory.csv",
        "variants.csv",
    ]


def test_partial_functional_scores_exclude_missing_rows():
    dataset = synthetic_dataset()
    variants = dataset.variants.copy()
    variants.loc[[0, 2], "functional_score"] = np.nan
    partial = Dataset(
        dataset.name,
        dataset.reference_sequence,
        variants,
        dataset.trajectory,
    )
    fitness = functional_score_fitness(partial)
    assert fitness.sequence_ids == ("v1", "v2", "v3")
    assert np.isfinite(fitness.values).all()


def test_fixed_analysis_window_covers_all_substitutions():
    variants = pd.DataFrame(
        {
            "SequenceIndex": ["wt", "v8", "v9"],
            "protein_sequence": ["AAAAAAAAAA", "AAAAAAACAA", "AAAAAAAADA"],
            "position": [1, 8, 9],
            "wt_aa": ["A", "A", "A"],
            "mutant_aa": ["A", "C", "D"],
            "is_synonymous": [True, False, False],
        }
    )
    trajectory = pd.DataFrame(
        {
            "SequenceIndex": ["v8", "v9", "v8", "v9"],
            "Replicate": ["r1"] * 4,
            "Generation": [0, 0, 1, 1],
            "Frequency": [10, 10, 5, 15],
        }
    )
    dataset = Dataset("window", "AAAAAAAAAA", variants, trajectory)
    start, end = analysis_window(dataset, 4)
    assert start <= 7 < end
    assert start <= 8 < end
    assert end - start == 4

    assert analysis_window(dataset, 4, truncate=(6, 9)) == (5, 9)
    configured = Dataset(
        dataset.name,
        dataset.reference_sequence,
        dataset.variants,
        dataset.trajectory,
        truncation=(7, 10),
    )
    assert analysis_window(configured, 4) == (6, 10)
    with pytest.raises(ValueError, match="does not cover every"):
        analysis_window(dataset, 4, truncate=(1, 4))
    with pytest.raises(ValueError, match="exceeding window_size"):
        analysis_window(dataset, 4, truncate=(5, 9))
    with pytest.raises(ValueError, match="valid 1-based inclusive"):
        analysis_window(dataset, 4, truncate=(0, 3))
    with pytest.raises(ValueError, match="must cover every"):
        Dataset(
            dataset.name,
            dataset.reference_sequence,
            dataset.variants,
            dataset.trajectory,
            truncation=(1, 4),
        )


def test_substitution_prior_inference_and_metrics():
    dataset = synthetic_dataset()
    basis = substitution_basis(dataset)
    assert basis.feature_names == ("1:C", "2:D", "3:E")
    result = infer(
        dataset, basis, gamma=0.1, prior=synthetic_prior(dataset), prior_scale=0.5
    )
    assert result.replicate_coefficients.shape == (2, 3)
    assert result.joint_coefficients.shape == (3,)
    assert np.isfinite(result.joint_coefficients).all()
    fitness = result.fitness()
    assert fitness.values.shape == (5, 1)
    metrics = evaluate_fitness(dataset, fitness)
    assert 0 <= metrics["auc"] <= 1
    assert metrics["n_pathogenic"] == 3


def test_raw_llr_orientation_follows_dataset_selection_direction():
    source = synthetic_dataset()
    dataset = Dataset(
        "synthetic-high",
        source.reference_sequence,
        source.variants,
        source.trajectory,
        pathogenic_high_selection=True,
    )
    raw_prior = FeatureArtifact(
        ("v1", "v1b", "v2", "v3"),
        np.asarray([[-2.0], [-2.0], [-1.0], [-0.25]]),
        ("llr",),
        "llr_prior",
        dataset.name,
        {"orientation": "raw_llr"},
    )
    result = infer(dataset, substitution_basis(dataset), gamma=1.0, prior=raw_prior)
    np.testing.assert_allclose(result.prior, [2.0, 1.0, 0.25])


def test_prior_sweep_and_enrichment():
    dataset = synthetic_dataset()
    sweep = prior_sweep(
        dataset,
        synthetic_prior(dataset),
        alphas=[0.0, 1.0],
        gammas=[0.01, 1.0],
        evaluate=lambda fitness: evaluate_fitness(dataset, fitness),
    )
    assert len(sweep) == 4
    assert set(("alpha", "gamma", "auc", "cross_replicate_consistency")).issubset(sweep)
    enrichment = enrichment_fitness(dataset)
    assert enrichment.values.shape == (5, 1)
    assert np.isfinite(enrichment.values).all()


def test_configured_workflow(tmp_path):
    dataset = synthetic_dataset()
    dataset.save(tmp_path / "dataset")
    synthetic_prior(dataset).save(tmp_path / "prior.npz")
    config = {
        "_config_dir": str(tmp_path),
        "output_dir": "results",
        "gammas": [0.01, 1.0],
        "alphas": [0.0, 1.0],
        "review_star_cutoffs": [0, 2],
        "datasets": [{"path": "dataset", "priors": {"Test LLR": "prior.npz"}}],
    }
    results = run_analysis(config)
    assert len(results["sweeps"]) == 4
    assert len(results["summary"]) == 1
    assert (tmp_path / "results" / "summary.csv").is_file()
    json.loads((tmp_path / "dataset" / "dataset.json").read_text())


def test_sae_returns_canonical_artifact():
    dataset = synthetic_dataset()
    rng = np.random.default_rng(7)
    embeddings = FeatureArtifact(
        dataset.sequence_ids,
        rng.normal(size=(len(dataset.sequence_ids), 8)).astype(np.float32),
        tuple(f"e{index}" for index in range(8)),
        "embedding",
        dataset.name,
    )
    sae = train_sae(
        dataset,
        embeddings,
        config=SAEConfig(
            n_features=8,
            epochs=4,
            batch_size=2,
            mode="topk",
            k=3,
            active_frequency=0.01,
        ),
        device="cpu",
    )
    assert sae.kind == "sae"
    assert sae.sequence_ids == dataset.sequence_ids
    assert 0 < sae.values.shape[1] <= 8
    assert np.isfinite(sae.values).all()


def test_embedding_and_llr_generation_share_model_path(monkeypatch):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    token_by_aa = {amino_acid: index + 1 for index, amino_acid in enumerate(alphabet)}

    class Tokenizer:
        mask_token_id = len(alphabet) + 3

        def __call__(
            self,
            sequence,
            *,
            return_tensors=None,
            add_special_tokens=True,
            return_special_tokens_mask=False,
        ):
            if not add_special_tokens:
                return {"input_ids": [token_by_aa[sequence]]}
            ids = [0, *(token_by_aa[value] for value in sequence), len(alphabet) + 1]
            payload = {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
            if return_special_tokens_mask:
                payload["special_tokens_mask"] = torch.tensor(
                    [[1, *([0] * len(sequence)), 1]], dtype=torch.long
                )
            return payload

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(len(alphabet) + 4, 4)

        @property
        def device(self):
            return self.embed.weight.device

        def forward(self, sequence_tokens, sequence_id):
            length = sequence_tokens.shape[1]
            positions = torch.arange(length, dtype=torch.float32).view(1, length, 1)
            base = positions + torch.arange(4, dtype=torch.float32).view(1, 1, 4)
            logits = torch.arange(len(alphabet) + 4, dtype=torch.float32).view(1, 1, -1)
            logits = logits.expand(1, length, -1)
            return SimpleNamespace(
                sequence_logits=logits,
                hidden_states=torch.stack([base], dim=0),
                embeddings=base + 1.0,
            )

    loads = []

    def load_model(*args, **kwargs):
        loads.append(args[0])
        return Tokenizer(), Model()

    monkeypatch.setattr("esmdms.features.load_language_model", load_model)
    dataset = synthetic_dataset()
    embeddings = embed(dataset, "test-model", layers=[0, 1], pooling="max")
    assert set(embeddings) == {0, 1}
    assert embeddings[0].values.shape == (5, 4)
    np.testing.assert_array_equal(embeddings[0].values[1], embeddings[0].values[2])

    prior = masked_marginal_llr(dataset, "test-model")
    assert prior.kind == "llr_prior"
    assert prior.provenance["orientation"] == "raw_llr"
    assert prior.values.shape == (4, 1)
    assert prior.values[0, 0] == prior.values[1, 0]

    loads.clear()
    joint_embeddings, joint_prior = embed_and_llr(
        dataset,
        "test-model",
        layers=[0, 1],
        pooling="max",
        truncate=(1, 3),
    )
    assert loads == ["test-model"]
    assert joint_embeddings[1].sequence_ids == dataset.sequence_ids
    assert joint_prior.sequence_ids == ("v1", "v1b", "v2", "v3")
    assert joint_embeddings[1].provenance["window_start"] == 1
    assert joint_embeddings[1].provenance["window_end"] == 3
    assert joint_embeddings[1].provenance["window_selection"] == "custom"
    assert joint_prior.provenance["window_start"] == 1
    assert joint_prior.provenance["window_end"] == 3
    assert joint_prior.provenance["window_selection"] == "custom"

    configured_dataset = Dataset(
        dataset.name,
        dataset.reference_sequence,
        dataset.variants,
        dataset.trajectory,
        truncation=(1, 3),
    )
    configured_embeddings, configured_prior = embed_and_llr(
        configured_dataset, "test-model", layers=[1]
    )
    assert configured_embeddings[1].provenance["window_selection"] == "dataset"
    assert configured_prior.provenance["window_selection"] == "dataset"

    loads.clear()
    shard_0 = embed_and_llr(
        dataset,
        "test-model",
        layers=[1],
        shard_index=0,
        num_shards=2,
    )
    shard_1 = embed_and_llr(
        dataset,
        "test-model",
        layers=[1],
        shard_index=1,
        num_shards=2,
    )
    assert loads == ["test-model", "test-model"]
    merged_embedding = FeatureArtifact.merge(
        [shard_0[0][1], shard_1[0][1]],
        expected_sequence_ids=dataset.sequence_ids,
    )
    merged_prior = FeatureArtifact.merge(
        [shard_0[1], shard_1[1]],
        expected_sequence_ids=("v1", "v1b", "v2", "v3"),
    )
    assert merged_embedding.sequence_ids == dataset.sequence_ids
    assert merged_embedding.provenance["merged_shards"] == 2
    assert merged_prior.sequence_ids == ("v1", "v1b", "v2", "v3")
