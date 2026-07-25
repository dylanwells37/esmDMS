"""Protein language-model LLR tests. Skipped when torch is unavailable."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from esmdms.features import (  # noqa: E402
    _checkpoint_files,
    analysis_window,
    masked_marginal_llr,
)
from esmdms.schema import Dataset, FeatureArtifact  # noqa: E402

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
    assert start <= 7 < end and start <= 8 < end
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


def test_masked_marginal_llr_generation_and_sharding(monkeypatch):
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
            self.weight = torch.nn.Parameter(torch.zeros(1))

        @property
        def device(self):
            return self.weight.device

        def forward(self, sequence_tokens, sequence_id):
            length = sequence_tokens.shape[1]
            logits = torch.arange(len(alphabet) + 4, dtype=torch.float32)
            return SimpleNamespace(
                sequence_logits=logits.view(1, 1, -1).expand(1, length, -1)
            )

    monkeypatch.setattr(
        "esmdms.features.load_language_model",
        lambda *args, **kwargs: (Tokenizer(), Model()),
    )
    dataset = synthetic_dataset()
    prior = masked_marginal_llr(dataset, "test-model", truncate=(1, 3))
    assert prior.kind == "llr_prior"
    assert prior.provenance["orientation"] == "raw_llr"
    assert prior.provenance["window_selection"] == "custom"
    assert prior.values.shape == (4, 1)
    assert prior.values[0, 0] == prior.values[1, 0]

    shards = [
        masked_marginal_llr(
            dataset, "test-model", shard_index=index, num_shards=2
        )
        for index in range(2)
    ]
    merged = FeatureArtifact.merge(
        shards, expected_sequence_ids=("v1", "v1b", "v2", "v3")
    )
    assert merged.sequence_ids == ("v1", "v1b", "v2", "v3")
    assert merged.provenance["merged_shards"] == 2
