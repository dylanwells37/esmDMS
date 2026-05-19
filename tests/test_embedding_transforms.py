import pickle

import numpy as np
import pandas as pd

from embedding_transforms import (
    ICATransform,
    IdentityTransform,
    PCATransform,
    SparseAutoencoderTransform,
    SparsePCATransform,
)
from mega_analysis import EmbeddingAnalysisDataset, run_modular_inference


def synthetic_embeddings(n=40, d=12, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    loadings = rng.normal(size=(3, d))
    return (latent @ loadings + 0.05 * rng.normal(size=(n, d))).astype(np.float32)


def synthetic_inference_df(embeddings):
    rng = np.random.default_rng(2)
    s = rng.normal(size=embeddings.shape[1])
    logits = embeddings @ s
    pre = np.ones(len(embeddings), dtype=float)
    post = np.exp(logits - logits.max())
    post = post / post.sum() * len(embeddings)
    rows = []
    for i, emb in enumerate(embeddings):
        rows.append({"seq_id": i, "Replicate": 1, "Generation": 0, "Frequency": pre[i]})
        rows.append({"seq_id": i, "Replicate": 1, "Generation": 1, "Frequency": post[i]})
    return pd.DataFrame(rows)


def test_each_transform_implements_fit_and_transform():
    transforms = [
        IdentityTransform(),
        PCATransform(variance_threshold=0.8),
        SparsePCATransform(n_components=4, max_iter=50),
        ICATransform(n_components=4, max_iter=200),
        SparseAutoencoderTransform(latent_dim=6, epochs=10, sparsity_weight=0.05),
    ]
    for transform in transforms:
        assert callable(transform.fit)
        assert callable(transform.transform)


def test_fit_transform_shapes_end_to_end():
    x = synthetic_embeddings()
    transforms = [
        IdentityTransform(),
        PCATransform(variance_threshold=0.8),
        SparsePCATransform(n_components=4, max_iter=50),
        ICATransform(n_components=4, max_iter=300),
        SparseAutoencoderTransform(latent_dim=6, epochs=15, sparsity_weight=0.05),
    ]
    for transform in transforms:
        z = transform.fit(x).transform(x)
        assert z.shape[0] == x.shape[0]
        assert z.ndim == 2
        assert np.all(np.isfinite(z))


def test_sae_activations_are_sparse():
    x = synthetic_embeddings(n=60, d=10)
    transform = SparseAutoencoderTransform(
        latent_dim=12,
        epochs=40,
        sparsity_weight=0.2,
        activation_threshold=0.05,
        random_state=3,
    )
    z = transform.fit(x).transform(x)
    zero_fraction = np.mean(np.abs(z) <= transform.activation_threshold)
    assert zero_fraction > 0.25


def test_pca_and_spca_reduce_dimensionality_as_configured():
    x = synthetic_embeddings(n=50, d=14)
    pca = PCATransform(variance_threshold=0.7).fit(x)
    z_pca = pca.transform(x)
    assert 1 <= z_pca.shape[1] < x.shape[1]

    spca = SparsePCATransform(n_components=5, max_iter=50).fit(x)
    z_spca = spca.transform(x)
    assert z_spca.shape == (x.shape[0], 5)


def test_spca_variance_threshold_picks_fewest_components():
    x = synthetic_embeddings(n=50, d=14)
    full = SparsePCATransform(n_components=10, max_iter=80).fit(x)
    thresh = SparsePCATransform(
        n_components=10, variance_threshold=0.6, max_components=10, max_iter=80,
    ).fit(x)
    assert thresh.transform(x).shape[1] <= full.transform(x).shape[1]
    cum = np.cumsum(thresh.explained_variance_ratio_)
    assert cum[-1] >= 0.6 - 1e-6 or len(cum) == 10


def test_full_modular_loop_produces_one_entry_per_transform(tmp_path):
    x = synthetic_embeddings(n=18, d=8)
    metadata = synthetic_inference_df(x)
    metadata.to_pickle(tmp_path / "inference_metadata.pkl")
    with open(tmp_path / "layer0_seq_to_emb.pkl", "wb") as f:
        pickle.dump(x, f)

    cfg = {
        "layers": [0],
        "embedding_transforms": ["identity", "pca", "spca", "sae", "ica"],
        "transform_config": {
            "pca": {"variance_threshold": 0.8},
            "spca": {"n_components": 3, "max_iter": 50},
            "sae": {
                "latent_dim": 4,
                "epochs": 15,
                "sparsity_weight": 0.1,
                "activation_threshold": 0.05,
            },
            "ica": {"n_components": 3, "max_iter": 300},
        },
    }
    results = run_modular_inference(str(tmp_path), cfg, save_results=False, force_recompute=True)
    assert set(results) == set(cfg["embedding_transforms"])
    for transform_name in cfg["embedding_transforms"]:
        run_result = results[transform_name]
        assert 0 in run_result.layers
        assert np.all(np.isfinite(run_result.layers[0].s_joint))


def test_class_api_runs_inference_on_transform_object(tmp_path):
    x = synthetic_embeddings(n=18, d=8)
    metadata = synthetic_inference_df(x)
    metadata.to_pickle(tmp_path / "inference_metadata.pkl")
    with open(tmp_path / "layer0_seq_to_emb.pkl", "wb") as f:
        pickle.dump(x, f)

    dataset = EmbeddingAnalysisDataset(
        "toy",
        str(tmp_path),
        inference_cfg={"layers": [0], "normalize": "none"},
    )
    result = dataset.run_inference(
        PCATransform(variance_threshold=0.8),
        save_results=False,
        force_recompute=True,
        cache=False,
    )
    assert result.transform_name == "pca"
    assert 0 in result.layers
    assert result.layers[0].feature_dim < x.shape[1]
    assert result.layers[0].s_joint.shape[0] == result.layers[0].feature_dim


def test_class_api_runs_simulation_on_transform_object(tmp_path):
    x = synthetic_embeddings(n=10, d=6)
    sim_df = pd.DataFrame({
        "Embedding": [x[i] for i in range(len(x))],
        "Rep1_PreNums": np.full(len(x), 10.0),
        "Rep2_PreNums": np.full(len(x), 10.0),
    })
    sim_df.to_pickle(tmp_path / "layer0_sim_df.pkl")

    dataset = EmbeddingAnalysisDataset(
        "toy",
        str(tmp_path),
        simulation_cfg={
            "layers": [0],
            "n_gens": 2,
            "save_every": 1,
            "sel_func": "zero",
            "fitness_fn": "exp",
            "method": "fullcov",
        },
    )
    result = dataset.run_simulation(PCATransform(variance_threshold=0.8), save_results=False)
    assert result.transform_name == "pca"
    assert 0 in result.layers
    assert 0 in result.layer_fits
    assert result.true_selection_coefficients[0].shape[0] == result.layers[0].feature_dim
