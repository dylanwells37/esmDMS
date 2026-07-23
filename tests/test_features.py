"""Model-facing tests. Skipped wholesale when torch is unavailable."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from esmdms.features import (  # noqa: E402
    SAEConfig,
    _checkpoint_files,
    analysis_window,
    embed,
    embed_and_llr,
    masked_marginal_llr,
    train_sae,
)
from esmdms.schema import Dataset, FeatureArtifact  # noqa: E402
from examples.brca1_100_demo import build_subset_dataset  # noqa: E402

from .conftest import synthetic_dataset  # noqa: E402


def test_checkpoint_files_supports_single_and_sharded_weights(tmp_path):
    single = tmp_path / "single"
    single.mkdir()
    (single / "model.safetensors").touch()
    assert _checkpoint_files(single) == (single / "model.safetensors",)

    sharded = tmp_path / "sharded"
    sharded.mkdir()
    shard_1 = sharded / "model-00001-of-00002.safetensors"
    shard_2 = sharded / "model-00002-of-00002.safetensors"
    shard_1.touch()
    shard_2.touch()
    (sharded / "model.safetensors.index.json").write_text(
        '{"weight_map": {"first": "model-00001-of-00002.safetensors", '
        '"second": "model-00002-of-00002.safetensors"}}'
    )
    assert _checkpoint_files(sharded) == (shard_1, shard_2)


def test_brca1_demo_subset_keeps_reference_and_selected_trajectories(tmp_path):
    source = synthetic_dataset()
    subset = build_subset_dataset(source, tmp_path / "subset", variant_count=2)

    assert subset.name == "synthetic__first2"
    assert subset.sequence_ids == ("wt", "v1", "v1b")
    assert subset.metadata["selection"]["variant_count"] == 2
    assert set(subset.trajectory["SequenceIndex"]) == {"v1", "v1b"}
    assert not subset.trajectory["SequenceIndex"].eq("wt").any()
    Dataset.load(tmp_path / "subset")


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
    # The held-out split is scored rather than silently discarded.
    assert sae.provenance["train_sequences"] >= 1
    assert sae.provenance["validation_sequences"] >= 1
    assert np.isfinite(sae.provenance["validation_reconstruction_mse"])


def test_batchtopk_activations_do_not_depend_on_the_evaluation_batch():
    dataset = synthetic_dataset()
    rng = np.random.default_rng(11)
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
            mode="batchtopk",
            k=3,
            active_frequency=0.01,
        ),
        device="cpu",
    )
    threshold = sae.provenance["activation_threshold"]
    assert threshold is not None and np.isfinite(threshold)

    # Encoding a single row alone must match its row inside the full matrix.
    from esmdms.features import SparseAutoencoder

    model = SparseAutoencoder(8, 8, "batchtopk", 3)
    state = torch.load(sae.provenance["model_path"], map_location="cpu") if (
        "model_path" in sae.provenance
    ) else None
    if state is not None:
        model.load_state_dict(state["state_dict"])
    else:
        model.activation_threshold.fill_(threshold)
    model.eval()
    values = torch.from_numpy(embeddings.values.astype(np.float32))
    with torch.no_grad():
        full = model.encode(values)
        alone = torch.vstack([model.encode(values[index : index + 1]) for index in range(len(values))])
    torch.testing.assert_close(full, alone)


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
    # Only the final index carries the model's output LayerNorm.
    assert embeddings[0].provenance["final_layer_norm_applied"] is False
    assert embeddings[1].provenance["final_layer_norm_applied"] is True

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
