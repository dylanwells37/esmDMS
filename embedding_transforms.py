"""Swappable feature transforms for ESM2 embeddings used by popDMS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.decomposition import FastICA, PCA, SparsePCA


class EmbeddingTransform(ABC):
    """Base interface for embedding feature maps consumed by popDMS."""

    name = "base"

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> "EmbeddingTransform":
        """Learn transform parameters from a population embedding matrix."""

    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Return the feature matrix passed downstream to popDMS."""

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.fit(embeddings).transform(embeddings)

    def project_coefficients_to_embedding_space(self, coefficients: np.ndarray) -> Optional[np.ndarray]:
        """Map feature-basis coefficients back to raw ESM2 dimensions when linear."""
        return None

    @property
    def diagnostics(self) -> Dict[str, Any]:
        """Optional fit diagnostics for notebooks and comparison reports."""
        return {}


def _as_2d_float(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {arr.shape}")
    return arr


@dataclass
class IdentityTransform(EmbeddingTransform):
    """No-op baseline: selection coefficients are in raw ESM2 coordinates."""

    name: str = "identity"
    n_features_in_: Optional[int] = None

    def fit(self, embeddings: np.ndarray) -> "IdentityTransform":
        self.n_features_in_ = _as_2d_float(embeddings).shape[1]
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return _as_2d_float(embeddings)

    def project_coefficients_to_embedding_space(self, coefficients: np.ndarray) -> np.ndarray:
        return np.asarray(coefficients)


@dataclass
class PCATransform(EmbeddingTransform):
    """PCA basis retaining enough components to explain a target variance fraction."""

    variance_threshold: float = 0.95
    max_components: Optional[int] = None
    min_components: int = 2
    random_state: int = 0
    name: str = "pca"
    model: Optional[PCA] = None
    n_features_in_: Optional[int] = None

    def fit(self, embeddings: np.ndarray) -> "PCATransform":
        x = _as_2d_float(embeddings)
        self.n_features_in_ = x.shape[1]
        max_rank = min(x.shape)
        n_components = self.max_components or max_rank
        n_components = min(n_components, max_rank)
        full = PCA(n_components=n_components, random_state=self.random_state)
        full.fit(x)
        cumulative = np.cumsum(full.explained_variance_ratio_)
        k = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
        k = max(k, min(self.min_components, n_components))
        k = max(1, min(k, n_components))
        self.model = PCA(n_components=k, random_state=self.random_state)
        self.model.fit(x)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCATransform must be fit before transform().")
        return self.model.transform(_as_2d_float(embeddings)).astype(np.float32)

    def project_coefficients_to_embedding_space(self, coefficients: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCATransform must be fit before projecting coefficients.")
        return self.model.components_.T @ np.asarray(coefficients)

    @property
    def diagnostics(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        return {
            "explained_variance_ratio": self.model.explained_variance_ratio_.copy(),
            "cumulative_explained_variance": np.cumsum(self.model.explained_variance_ratio_),
            "n_components": self.model.n_components_,
        }


@dataclass
class SparsePCATransform(EmbeddingTransform):
    """Sparse PCA basis with loadings concentrated on fewer ESM2 dimensions.

    Set ``variance_threshold`` to keep the fewest components (ranked by
    transformed-feature variance) needed to reach a target explained-variance
    fraction. Otherwise, ``n_components`` fixes the basis size.
    """

    n_components: int = 32
    variance_threshold: Optional[float] = None
    max_components: Optional[int] = None
    alpha: float = 1.0
    ridge_alpha: float = 0.01
    random_state: int = 0
    max_iter: int = 500
    name: str = "spca"
    model: Optional[SparsePCA] = None
    mean_: Optional[np.ndarray] = None
    explained_variance_ratio_: Optional[np.ndarray] = None
    keep_indices_: Optional[np.ndarray] = None
    n_features_in_: Optional[int] = None

    def fit(self, embeddings: np.ndarray) -> "SparsePCATransform":
        x = _as_2d_float(embeddings)
        self.n_features_in_ = x.shape[1]
        max_rank = min(x.shape)
        if self.variance_threshold is not None:
            fit_k = self.max_components or max_rank
        else:
            fit_k = self.n_components
        fit_k = max(1, min(fit_k, max_rank))

        self.model = SparsePCA(
            n_components=fit_k,
            alpha=self.alpha,
            ridge_alpha=self.ridge_alpha,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.model.fit(x)
        self.mean_ = x.mean(axis=0)
        z = self.model.transform(x)
        total_var = float(np.var(x - self.mean_, axis=0).sum())
        comp_var = np.var(z, axis=0)
        order = np.argsort(comp_var)[::-1]

        if self.variance_threshold is not None and total_var > 0:
            cumulative = np.cumsum(comp_var[order]) / total_var
            k = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
            k = max(1, min(k, fit_k))
            self.keep_indices_ = order[:k]
        else:
            self.keep_indices_ = order

        kept_var = comp_var[self.keep_indices_]
        self.explained_variance_ratio_ = (
            kept_var / total_var if total_var > 0 else np.zeros_like(kept_var)
        )
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SparsePCATransform must be fit before transform().")
        z = self.model.transform(_as_2d_float(embeddings))
        return z[:, self.keep_indices_].astype(np.float32)

    def project_coefficients_to_embedding_space(self, coefficients: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SparsePCATransform must be fit before projecting coefficients.")
        return self.model.components_[self.keep_indices_].T @ np.asarray(coefficients)

    @property
    def diagnostics(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        kept = self.model.components_[self.keep_indices_]
        return {
            "explained_variance_ratio": self.explained_variance_ratio_.copy(),
            "cumulative_explained_variance": np.cumsum(self.explained_variance_ratio_),
            "n_components": int(len(self.keep_indices_)),
            "component_sparsity": np.mean(np.isclose(kept, 0.0), axis=1),
        }


@dataclass
class ICATransform(EmbeddingTransform):
    """Independent-component basis for non-Gaussian ESM2 variation.

    FastICA is useful here because fitness-linked embedding directions may be
    statistically independent and non-Gaussian rather than merely high-variance.
    Separating those latent sources can reduce isotropic embedding noise that
    PCA would keep if it explains variance but not selection-relevant change.
    """

    n_components: int = 32
    random_state: int = 0
    max_iter: int = 1000
    name: str = "ica"
    model: Optional[FastICA] = None
    n_features_in_: Optional[int] = None

    def fit(self, embeddings: np.ndarray) -> "ICATransform":
        x = _as_2d_float(embeddings)
        self.n_features_in_ = x.shape[1]
        n_components = min(self.n_components, min(x.shape))
        self.model = FastICA(
            n_components=n_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
            whiten="unit-variance",
        )
        self.model.fit(x)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ICATransform must be fit before transform().")
        return self.model.transform(_as_2d_float(embeddings)).astype(np.float32)

    def project_coefficients_to_embedding_space(self, coefficients: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ICATransform must be fit before projecting coefficients.")
        return self.model.components_.T @ np.asarray(coefficients)


@dataclass
class SparseAutoencoderTransform(EmbeddingTransform):
    """Sparse autoencoder whose ReLU latent activations form popDMS features."""

    latent_dim: int = 64
    sparsity_weight: float = 1e-3
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 256
    activation_threshold: float = 1e-3
    random_state: int = 0
    name: str = "sae"
    encoder_: Any = field(default=None, init=False, repr=False)
    decoder_: Any = field(default=None, init=False, repr=False)
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    loss_history_: list = field(default_factory=list)
    sparsity_history_: list = field(default_factory=list)
    n_features_in_: Optional[int] = None

    def fit(self, embeddings: np.ndarray) -> "SparseAutoencoderTransform":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        x = _as_2d_float(embeddings)
        self.n_features_in_ = x.shape[1]
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        x_scaled = ((x - self.mean_) / self.scale_).astype(np.float32)

        input_dim = x_scaled.shape[1]
        self.encoder_ = nn.Sequential(nn.Linear(input_dim, self.latent_dim), nn.ReLU())
        self.decoder_ = nn.Linear(self.latent_dim, input_dim)
        params = list(self.encoder_.parameters()) + list(self.decoder_.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x_scaled)),
            batch_size=min(self.batch_size, len(x_scaled)),
            shuffle=True,
        )

        self.loss_history_ = []
        self.sparsity_history_ = []
        for _ in range(self.epochs):
            epoch_loss = 0.0
            epoch_sparse = 0.0
            n_seen = 0
            for (batch,) in loader:
                optimizer.zero_grad()
                z = self.encoder_(batch)
                recon = self.decoder_(z)
                recon_loss = torch.mean((recon - batch) ** 2)
                sparse_loss = torch.mean(torch.abs(z))
                loss = recon_loss + self.sparsity_weight * sparse_loss
                loss.backward()
                optimizer.step()
                n = len(batch)
                epoch_loss += float(recon_loss.detach()) * n
                epoch_sparse += float((z <= self.activation_threshold).float().mean().detach()) * n
                n_seen += n
            self.loss_history_.append(epoch_loss / n_seen)
            self.sparsity_history_.append(epoch_sparse / n_seen)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.encoder_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("SparseAutoencoderTransform must be fit before transform().")
        import torch

        x = _as_2d_float(embeddings)
        x_scaled = ((x - self.mean_) / self.scale_).astype(np.float32)
        self.encoder_.eval()
        with torch.no_grad():
            z = self.encoder_(torch.from_numpy(x_scaled)).numpy()
        z[np.abs(z) <= self.activation_threshold] = 0.0
        return z.astype(np.float32)

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return {
            "reconstruction_loss": np.array(self.loss_history_, dtype=float),
            "activation_sparsity": np.array(self.sparsity_history_, dtype=float),
            "latent_dim": self.latent_dim,
            "sparsity_weight": self.sparsity_weight,
        }


def make_transform(name: str, config: Optional[Dict[str, Any]] = None) -> EmbeddingTransform:
    """Factory for registry/config driven construction."""
    config = dict(config or {})
    registry = {
        "identity": IdentityTransform,
        "pca": PCATransform,
        "spca": SparsePCATransform,
        "sae": SparseAutoencoderTransform,
        "ica": ICATransform,
    }
    if name not in registry:
        raise KeyError(f"Unknown embedding transform '{name}'. Available: {sorted(registry)}")
    return registry[name](**config)


def make_transforms(names: Iterable[str], config: Optional[Dict[str, Dict[str, Any]]] = None):
    config = config or {}
    return [make_transform(name, config.get(name, {})) for name in names]


def unique_embeddings_from_df(embedding_df) -> np.ndarray:
    """Extract one embedding row per unique variant from a repeated inference df."""
    x = np.vstack(embedding_df["Embedding"].values).astype(np.float32)
    _, idx = np.unique(x, axis=0, return_index=True)
    return x[np.sort(idx)]


def transform_embedding_df(embedding_df, transform: EmbeddingTransform):
    """Return a dataframe copy whose Embedding column is in transform feature space."""
    out = embedding_df.copy()
    x = np.vstack(out["Embedding"].values).astype(np.float32)
    z = transform.transform(x)
    out["Embedding"] = [z[i] for i in range(z.shape[0])]
    return out
