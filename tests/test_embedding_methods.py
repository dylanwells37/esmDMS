import numpy as np
import pandas as pd
import torch

from embedding_scripts.embed_sequences import (
    build_sequence_dataframe_mavedb,
    cls_sequence_representation,
    mutation_site_representation,
    pool_sequence_representation,
    _expand_unpooled_mutation_embeddings,
)
from mega_analysis import emb_df_to_inference_dfs
from esmdmsfunctions import load_inference_df


def test_pool_and_cls_exclude_special_tokens():
    token_reps = torch.arange(5 * 3, dtype=torch.float32).reshape(1, 5, 3)
    inputs = {
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        "special_tokens_mask": torch.tensor([[1, 0, 0, 0, 1]]),
    }

    pooled = pool_sequence_representation(token_reps, inputs)
    cls = cls_sequence_representation(token_reps)

    expected_pool = token_reps[:, 1:4, :].mean(dim=1).squeeze(0).numpy()
    expected_cls = token_reps[:, 0, :].squeeze(0).numpy()
    assert np.allclose(pooled, expected_pool)
    assert np.allclose(cls, expected_cls)


def test_mutation_site_representation_unpooled_and_pooled():
    token_reps = torch.arange(6 * 2, dtype=torch.float32).reshape(1, 6, 2)
    unpooled = mutation_site_representation(token_reps, [0, 2], pool_mutations=False)
    pooled = mutation_site_representation(token_reps, [0, 2], pool_mutations=True)

    expected = token_reps[:, [1, 3], :].squeeze(0).numpy()
    assert np.allclose(unpooled, expected)
    assert np.allclose(pooled, expected.mean(axis=0))


def test_mavedb_sequence_dataframe_tracks_multiple_aa_mutation_sites(tmp_path):
    csv_path = tmp_path / "counts.csv"
    csv_path.write_text(
        "hgvs_nt,repA_c_0,repA_c_1\n"
        "c.[1A>G;4A>G],10,20\n"
    )
    # AAA AAA -> KK; GAA GAA -> EE, two amino-acid substitutions at sites 0 and 1.
    df = build_sequence_dataframe_mavedb(str(csv_path), "AAAAAA", skip_stop_codons=False)
    assert df["ProteinSequence"].iloc[0] == "EE"
    assert df["MutationSites"].iloc[0] == [0, 1]
    assert set(df["Generation"]) == {0, 1}


def test_unpooled_mutation_embeddings_expand_to_one_row_per_site():
    seq_df = pd.DataFrame({
        "ProteinSequence": ["ACD"],
        "MutationSites": [[0, 2]],
        "Replicate": [1],
        "Generation": [0],
        "Frequency": [5],
    })
    seq_to_emb = {"ACD": np.ones((2, 4, 3))}
    out = _expand_unpooled_mutation_embeddings(seq_df, seq_to_emb)

    assert len(out) == 2
    assert out["MutationSite"].tolist() == [0, 2]
    assert out["MutationSiteIndex"].tolist() == [0, 1]
    assert out["Embedding"].iloc[0].shape == (4, 3)


def test_compact_inference_keeps_separate_mutation_site_entries(tmp_path):
    emb_a = np.stack([np.ones(3), np.ones(3) * 2])
    emb_b = np.stack([np.ones(3) * 3, np.ones(3) * 4])
    df = pd.DataFrame({
        "ProteinSequence": ["ACD", "ACD", "ACD", "ACD"],
        "MutationSites": [[0, 2], [0, 2], [0, 2], [0, 2]],
        "MutationSite": [0, 0, 2, 2],
        "MutationSiteIndex": [0, 0, 1, 1],
        "Replicate": [1, 1, 1, 1],
        "Generation": [0, 1, 0, 1],
        "Frequency": [10, 20, 10, 20],
        "Embedding": [emb_a, emb_a, emb_b, emb_b],
    })
    emb_df_to_inference_dfs(df, tmp_path)
    loaded = load_inference_df(0, str(tmp_path))

    assert loaded["MutationSiteIndex"].nunique() == 2
    assert len({tuple(x) for x in loaded["Embedding"]}) == 2
