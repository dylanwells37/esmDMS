from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import pickle
import queue
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import time
import traceback

# Hugging Face access token for gated models (e.g. biohub/ESMC-6B).
# Prefer the HF_TOKEN env var; fall back to the constant below if you must
# paste the token directly. Do NOT commit a real token to the repo.
HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from embedding_scripts.embed_sequences import (
    CODON2AA,
    get_reference_nuc_sequence,
    build_sequence_dataframe_mavedb,
    get_reference_sequence_from_wildtypes,
    build_sequence_dataframe,
    all_residue_representation,
    parse_hgvs_nt,
    apply_substitutions,
    translate_nuc_sequence,
)

from popDMS import mini_infer_esm, infer_gamma_range, InferenceResult


## TYPE DEFINITIONS #################################


EmbeddingModel = Literal[
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D",
    "biohub/ESMC-300M",
    "biohub/ESMC-600M",
    "biohub/ESMC-6B",
]

EmbeddingType = Literal["per_residue", "mutation_pooled", "mean_pool", "max_pool"]
AbstractionMethod = Literal["none", "PCA", "SAE", "DeltaSAE", "SPCA"]

## CONFIGURATION AND INPUT CLASSES #################################

@dataclass(frozen=True)
class ESMDMSConfig:
    embedding_model: EmbeddingModel = "facebook/esm2_t33_650M_UR50D"
    embedding_type: EmbeddingType = "per_residue"
    embedding_method: str | None = None
    local_or_disk: Literal['local', 'disk', 'both'] = 'local'
    save_dir: str | None = None
    dataset_name: str | None = None

    def __post_init__(self):
        if self.local_or_disk not in {"local", "disk", "both"}:
            raise ValueError("local_or_disk must be one of 'local', 'disk', or 'both'.")
        if (self.local_or_disk == 'disk' or self.local_or_disk == 'both') and self.save_dir is None:
                raise ValueError("save_dir must be specified when local_or_disk is set to 'disk' or 'both'.")
        if self.embedding_method in {"cls", "cls_token"}:
            raise ValueError("CLS embeddings are no longer supported. Use embedding_type='per_residue', 'mutation_pooled', or 'mean_pool'.")
        if self.embedding_method is not None and self.embedding_method != self.embedding_type:
            object.__setattr__(self, "embedding_type", self.embedding_method)
        if self.embedding_type in {"mutation_site", "mutation_pool", "pooled"}:
            object.__setattr__(self, "embedding_type", "mutation_pooled")
        if self.embedding_type in {"mean_pooled", "mean"}:
            object.__setattr__(self, "embedding_type", "mean_pool")
        if self.embedding_type in {"max_pooled", "max"}:
            object.__setattr__(self, "embedding_type", "max_pool")
        if self.embedding_type not in {"per_residue", "mutation_pooled", "mean_pool", "max_pool"}:
            raise ValueError("embedding_type must be one of 'per_residue', 'mutation_pooled', 'mean_pool', or 'max_pool'.")
        if self.dataset_name is not None:
            if not self.dataset_name:
                raise ValueError("dataset_name must not be empty when specified.")
            if any(sep in self.dataset_name for sep in ("/", "\\")):
                raise ValueError("dataset_name must be a file name component and cannot contain path separators.")
        
    
@dataclass(frozen=True)
class CellularDMSInput:
    reference_nuc_path: Path
    mavedb_csv_path: Path
    scores_csv_path: Path | None = None
    use_replicates: list[str] | None = None
    reference_kind: Literal["auto", "nucleotide", "protein"] = "auto"
    primary_key: str = "hgvs_nt"
    wildtype_key: str = "__wildtype__"
    kind: Literal['cellular'] = 'cellular'

    def __post_init__(self):
        object.__setattr__(self, 'reference_nuc_path', Path(self.reference_nuc_path))
        object.__setattr__(self, 'mavedb_csv_path', Path(self.mavedb_csv_path))
        if self.scores_csv_path is not None:
            object.__setattr__(self, 'scores_csv_path', Path(self.scores_csv_path))
        if not self.reference_nuc_path.is_file():
            raise ValueError(f"Reference nucleotide sequence path {self.reference_nuc_path} does not exist or is not a file.")
        if not self.mavedb_csv_path.is_file():
            raise ValueError(f"MaveDB CSV path {self.mavedb_csv_path} does not exist or is not a file.")
        if self.scores_csv_path is not None and not self.scores_csv_path.is_file():
            raise ValueError(f"MaveDB scores CSV path {self.scores_csv_path} does not exist or is not a file.")
        if self.reference_kind not in {"auto", "nucleotide", "protein"}:
            raise ValueError("reference_kind must be one of 'auto', 'nucleotide', or 'protein'.")
        if not self.primary_key:
            raise ValueError("primary_key must not be empty.")


@dataclass(frozen=True)
class ViralDMSInput:
    pre_files: tuple[Path, ...]
    post_files: tuple[Path, ...]
    kind: Literal['viral'] = 'viral'

    def __post_init__(self):
        object.__setattr__(self, 'pre_files', tuple(Path(p) for p in self.pre_files))
        object.__setattr__(self, 'post_files', tuple(Path(p) for p in self.post_files))
        if len(self.pre_files) != len(self.post_files):
            raise ValueError(f"Number of pre-selection files ({len(self.pre_files)}) must match number of post-selection files ({len(self.post_files)}).")
        for p in self.pre_files:
            if not p.is_file():
                raise ValueError(f"Pre-selection file path {p} does not exist or is not a file.")
        for p in self.post_files:
            if not p.is_file():
                raise ValueError(f"Post-selection file path {p} does not exist or is not a file.")

DMSInput = CellularDMSInput | ViralDMSInput

## SPARSE AUTOENCODER ########################################################


SparsityMode = Literal["normal", "topk", "batchtopk"]


class SparseAutoencoder(torch.nn.Module):
    """
    Simple sparse autoencoder: Linear+ReLU encoder, linear decoder (no bias).

    Architecture:
        x  →  encoder (Linear + ReLU)  →  z  →  decoder (Linear)  →  x_hat

    The decoder columns are optionally kept at unit norm throughout training to
    prevent feature collapse (standard SAE practice).

    Sparsity is enforced via ``sparsity_mode``:
      - 'normal':    activations are unconstrained inside the module; sparsity
                     comes from an L1 penalty added to the loss outside.
      - 'topk':      per sample, keep only the top-k activations (others → 0).
      - 'batchtopk': across the whole batch, keep only the top k * batch_size
                     activations (others → 0).
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        normalize_decoder: bool = True,
        sparsity_mode: SparsityMode = "normal",
        k: int | None = None,
    ):
        super().__init__()
        if sparsity_mode not in {"normal", "topk", "batchtopk"}:
            raise ValueError(
                f"sparsity_mode must be one of 'normal', 'topk', 'batchtopk'; got {sparsity_mode!r}"
            )
        if sparsity_mode in {"topk", "batchtopk"} and (k is None or k <= 0):
            raise ValueError(
                f"sparsity_mode={sparsity_mode!r} requires a positive integer k."
            )
        self.encoder = torch.nn.Linear(input_dim, n_features)
        self.decoder = torch.nn.Linear(n_features, input_dim, bias=False)
        self.normalize_decoder = normalize_decoder
        self.sparsity_mode = sparsity_mode
        self.k = k
        self._init_weights()

    def _init_weights(self) -> None:
        torch.nn.init.kaiming_uniform_(self.encoder.weight)
        torch.nn.init.zeros_(self.encoder.bias)
        torch.nn.init.kaiming_uniform_(self.decoder.weight)
        if self.normalize_decoder:
            self._renorm_decoder()

    def _renorm_decoder(self) -> None:
        """Project decoder columns back to the unit sphere (in-place, no grad)."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.encoder(x))
        if self.sparsity_mode == "topk":
            k = min(self.k, z.shape[-1])
            _, topk_idx = z.topk(k, dim=-1)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, topk_idx, 1.0)
            z = z * mask
        elif self.sparsity_mode == "batchtopk":
            flat = z.reshape(-1)
            total_k = min(self.k * z.shape[0], flat.numel())
            _, topk_idx = flat.topk(total_k)
            mask = torch.zeros_like(flat)
            mask.scatter_(0, topk_idx, 1.0)
            z = (flat * mask).view_as(z)
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z


####### esmDMS CLASS #########################################################

class esmDMS:

    """
    Base class for ESM DMS analysis.

    The Goal of this class is to provide a structured framework for analyzing deep mutational scanning (DMS) data using ESM models. 
    It will 

    1. Take in DMS data from MabeDB format (cellular or viral data)
    2. Process the data to reconstruct a protein sequence / mutation site -> frequency time series
    3. Construct esm embeddings for the for each mutation / protein sequence
    4. (Optional) Perform an abstraction on top of the embeddings (PCA, SAE, SPCA, etc.)

    5. Analyze the results:

        Pure statistical analysis from real data
        6. Run the abstract 'features' through the popDMS framework to calculate selection coefficients on the feature space
            - Compare the cross-replicate consistency of inferred selection coefficients
            - Compare the cross-replucate consistency of inferred fitness
            - Compare the inferred selection coefficients and fitness across different models (linear, non-linear) and regularization schemes (L2, L1, ElasticNet)
            - Compare the inferred results across different abstracted feature spaces (Pure ESM, PCA, SAE, SPCA, etc.)
            - Compare the inferred results across sizes of ESM embeddings (ESM-2 8M, ESM-2 650M, etc.)
            - Investigate the sparsity of the inferred selection coefficients and fitness across different models and regularization schemes
            - Investigate how shuffling frequencies across time points affects the inferred selection coefficients and fitness (as a control)
            - Investigate how embedding type impacts all of this (mean-pooling vs per-residue)
            - Investigate how ALL of this changes across DMS datasets (BRCA1, BF520, BG505, Ube4b, TpoR, etc.)

        Statistical analysis of simulated data (reconstructions)
        7. Run simulations
            - Compare true vs inferred selection coefficients
            - Compare true vs inferred fitness
            - Compare the consistency of inferred selection coefficients across replicates
            - Compare the consistency of inferred fitness across replicates
            - Analyze how fitness functions and models are reconstructed
            - Investigate different models of selection (linear, non-linear) and how they are reconstructed
            - Investigate how the regularization of the model affects the reconstruction of the fitness function (L2 vs L1 vs ElasticNet)
    """

    reference_sequence: str | None
    sequence_dataframe: pd.DataFrame | None

    def __init__(self, input_data: DMSInput, config: ESMDMSConfig = ESMDMSConfig()):
        self.input_data = input_data
        self.config = config
        self.reference_sequence = None
        self.sequence_dataframe = None
        self.sequence_to_mutation_sites = None
        self.sequence_to_protein_sequence = None
        self.sequence_to_embeddings = {}
        self.sequence_to_features = {}
        self.inference_results = {}
        self.reference_kind = None
        self.sequence_metadata = None
        self.scores_dataframe = None

    def _use_memory(self) -> bool:
        return self.config.local_or_disk in {"local", "both"}

    def _use_disk(self) -> bool:
        return self.config.local_or_disk in {"disk", "both"}

    def _save_dir(self) -> Path:
        if self.config.save_dir is None:
            raise ValueError("save_dir must be specified for disk-backed caching.")
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    @staticmethod
    def _layer_label(layer: str | int) -> str:
        layer_str = str(layer)
        if layer_str.startswith("Layer_"):
            return layer_str
        return f"Layer_{int(layer_str)}"

    def _embedding_key(self, layer: str | int) -> str:
        return f"Embeddings_{self._layer_label(layer)}"

    def _feature_key(self, method: str, layer: str | int, embedding_type: str | None = None) -> str:
        return f"{self._embedding_type(embedding_type)}_{self._abstraction_type(method)}_{self._layer_label(layer)}"

    def _dataset_prefix(self) -> str:
        if self.config.dataset_name is None:
            return ""
        return f"{self.config.dataset_name}_"

    def _model_cache_label(self) -> str:
        return self.config.embedding_model.replace("/", "__")

    def _embedding_path(self, layer: str | int, embedding_type: str | None = None) -> Path:
        embedding_type = self._embedding_type(embedding_type)
        return self._feature_path("none", layer, embedding_type)

    def _feature_path(self, method: str, layer: str | int, embedding_type: str | None = None) -> Path:
        embedding_type = self._embedding_type(embedding_type)
        abstraction_type = self._abstraction_type(method)
        return self._save_dir() / (
            f"{self._dataset_prefix()}{self._model_cache_label()}_{embedding_type}_{abstraction_type}_"
            f"{self._layer_label(layer)}_seq_to_features.pkl"
        )

    def _inference_path(
        self,
        abstraction_method: str,
        layer: str | int,
        norm_scheme: str,
        embedding_type: str | None = None,
    ) -> Path:
        embedding_type = self._embedding_type(embedding_type)
        abstraction_type = self._abstraction_type(abstraction_method)
        return self._save_dir() / (
            f"{self._dataset_prefix()}{self._model_cache_label()}_{embedding_type}_{abstraction_type}_"
            f"{self._layer_label(layer)}_{norm_scheme}_inference_results.pkl"
        )

    def _inference_key(
        self,
        layer: str,
        abstraction_method: str,
        norm_scheme: str,
        embedding_type: str | None = None,
    ) -> str:
        return f"{self._embedding_type(embedding_type)}_{self._abstraction_type(abstraction_method)}_{self._layer_label(layer)}_{norm_scheme}_inference_results"

    @staticmethod
    def _safe_file_component(value: object) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
        return safe or "run"

    @classmethod
    def _sae_sweep_run_label(cls, idx: int, params: dict) -> str:
        run_label = params.get("run_label")
        if run_label:
            return cls._safe_file_component(run_label)
        parts = [f"run{idx:03d}"]
        for key in ("n_features", "sparsity_mode", "k", "sparsity_coeff", "lr", "epochs", "batch_size", "seed"):
            if key in params:
                parts.append(f"{key}-{params[key]}")
        return cls._safe_file_component("_".join(parts))

    def _sae_sweep_job_dir(
        self,
        layer: str | int,
        method: str,
        embedding_type: str | None = None,
        job_dir: str | Path | None = None,
    ) -> Path:
        if job_dir is not None:
            sweep_dir = Path(job_dir)
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            sweep_dir = (
                self._save_dir()
                / "sae_sweep_jobs"
                / (
                    f"{self._dataset_prefix()}{self._model_cache_label()}_"
                    f"{self._embedding_type(embedding_type)}_{self._abstraction_type(method)}_"
                    f"{self._layer_label(layer)}_{stamp}"
                )
            )
        sweep_dir.mkdir(parents=True, exist_ok=True)
        return sweep_dir

    @staticmethod
    def _abstraction_type(method: str) -> str:
        if method == "Embeddings":
            return "none"
        return method

    @staticmethod
    def _abstraction_base_method(method: str) -> str:
        method = esmDMS._abstraction_type(method)
        if method.startswith("PCA"):
            return "PCA"
        if method.startswith("DeltaSAE"):
            return "DeltaSAE"
        if method.startswith("SAE"):
            return "SAE"
        if method.startswith("SPCA"):
            return "SPCA"
        return method

    def _embedding_type(self, embedding_type: str | None = None) -> str:
        embedding_type = embedding_type or self.config.embedding_type
        if embedding_type in {"mutation_site", "mutation_pool", "pooled"}:
            return "mutation_pooled"
        if embedding_type in {"mean_pooled", "mean"}:
            return "mean_pool"
        if embedding_type in {"max_pooled", "max"}:
            return "max_pool"
        if embedding_type not in {"per_residue", "mutation_pooled", "mean_pool", "max_pool"}:
            raise ValueError("embedding_type must be one of 'per_residue', 'mutation_pooled', 'mean_pool', or 'max_pool'.")
        return embedding_type

    @staticmethod
    def _load_pickle(path: Path):
        with path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _save_pickle(value, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(value, f)

    def _batch_dir(self, job_dir: str | Path | None = None) -> Path:
        if job_dir is not None:
            batch_dir = Path(job_dir)
        else:
            batch_dir = self._save_dir() / "embedding_batches" / f"{self._dataset_prefix()}{self._model_cache_label()}_all_data"
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

    def _batch_payload_path(self, batch_dir: Path) -> Path:
        return batch_dir / f"{self._dataset_prefix()}embedding_batch_payload.pkl"

    def _batch_chunk_path(self, batch_dir: Path, chunk_idx: int) -> Path:
        return batch_dir / f"{self._dataset_prefix()}embeddings_chunk_{chunk_idx}.pkl"

    def _batch_feature_chunk_path(self, batch_dir: Path, chunk_idx: int, embedding_type: str) -> Path:
        return batch_dir / f"{self._dataset_prefix()}{embedding_type}_embeddings_chunk_{chunk_idx}.pkl"

    def _merged_embeddings_path(self, batch_dir: Path) -> Path:
        return batch_dir / f"{self._dataset_prefix()}merged_sequence_embeddings.pkl"

    def _inference_job_dir(
        self,
        layer: str | int,
        abstraction_method: str,
        norm_scheme: str,
        embedding_type: str | None = None,
        job_dir: str | Path | None = None,
    ) -> Path:
        if job_dir is not None:
            inference_job_dir = Path(job_dir)
        else:
            inference_job_dir = (
                self._save_dir()
                / "inference_jobs"
                / f"{self._dataset_prefix()}{self._model_cache_label()}_{self._embedding_type(embedding_type)}_{self._abstraction_type(abstraction_method)}_{self._layer_label(layer)}_{norm_scheme}"
            )
        inference_job_dir.mkdir(parents=True, exist_ok=True)
        return inference_job_dir

    def _inference_payload_path(self, inference_job_dir: Path) -> Path:
        return inference_job_dir / f"{self._dataset_prefix()}inference_payload.pkl"

    # ── SAE path helpers ──────────────────────────────────────────────────

    def _sae_model_dir(self) -> Path:
        return self._save_dir() / "sae_models"

    def _sae_tag(
        self,
        layer: str | int,
        n_features: int,
        sparsity_coeff: float,
        embedding_type: str | None = None,
        sparsity_mode: str = "normal",
        k: int | None = None,
        run_label: str | None = None,
    ) -> str:
        embedding_part = f"{self._embedding_type(embedding_type)}_" if embedding_type is not None else ""
        # Only suffix when non-default so existing 'normal'-mode caches keep their filenames.
        mode_part = f"_{sparsity_mode}_k{k}" if sparsity_mode != "normal" else ""
        run_part = f"_{run_label}" if run_label else ""
        return (
            f"{self._dataset_prefix()}{embedding_part}sae_"
            f"{self._layer_label(layer)}_{n_features}_{sparsity_coeff}{mode_part}{run_part}"
        )

    def _sae_model_path(
        self,
        layer: str | int,
        n_features: int,
        sparsity_coeff: float,
        embedding_type: str | None = None,
        sparsity_mode: str = "normal",
        k: int | None = None,
        run_label: str | None = None,
    ) -> Path:
        return self._sae_model_dir() / (
            f"{self._sae_tag(layer, n_features, sparsity_coeff, embedding_type, sparsity_mode, k, run_label)}_model.pt"
        )

    def _sae_viz_path(
        self,
        layer: str | int,
        n_features: int,
        sparsity_coeff: float,
        embedding_type: str | None = None,
        sparsity_mode: str = "normal",
        k: int | None = None,
        run_label: str | None = None,
    ) -> Path:
        return self._sae_model_dir() / (
            f"{self._sae_tag(layer, n_features, sparsity_coeff, embedding_type, sparsity_mode, k, run_label)}_viz_data.pkl"
        )

    # ── Embedding layer selection ─────────────────────────────────────────

    @staticmethod
    def _select_layer(
        embeddings: dict[str, np.ndarray],
        layer: str | int,
        allow_per_residue: bool = False,
    ) -> dict[str, np.ndarray]:
        layer_idx = int(str(layer).replace("Layer_", ""))
        selected = {}
        for seq_id, embedding in embeddings.items():
            if embedding is None and allow_per_residue:
                selected[seq_id] = None
                continue
            embedding = np.asarray(embedding)
            if embedding.ndim == 1:
                selected[seq_id] = embedding
            elif embedding.ndim == 2:
                selected[seq_id] = embedding[layer_idx]
            elif embedding.ndim == 3:
                layer_embedding = embedding[:, layer_idx, :]
                if not allow_per_residue:
                    raise ValueError(
                        "Unpooled per-residue embeddings are not valid feature vectors for inference. "
                        "Use embedding_type='mutation_pooled' or add a feature abstraction that flattens them intentionally."
                    )
                selected[seq_id] = layer_embedding
            else:
                raise ValueError(f"Unsupported embedding shape for sequence {seq_id}: {embedding.shape}")
        return selected

    def _derive_embedding_type(
        self,
        layer_embeddings: dict[str, np.ndarray],
        embedding_type: str,
    ) -> dict[str, np.ndarray]:
        if embedding_type not in {"per_residue", "mutation_pooled", "mean_pool", "max_pool"}:
            raise ValueError(f"Unsupported embedding_type: {embedding_type}")

        derived = {}
        for seq_id, embedding in layer_embeddings.items():
            if embedding is None:
                derived[seq_id] = None
                continue
            embedding = np.asarray(embedding)
            if embedding.ndim != 2:
                raise ValueError(
                    f"Cannot derive {embedding_type} from layer embedding shape {embedding.shape} "
                    f"for sequence {seq_id}."
                )

            if embedding_type == "mean_pool":
                derived[seq_id] = embedding.mean(axis=0)
                continue
            if embedding_type == "max_pool":
                derived[seq_id] = embedding.max(axis=0)
                continue

            if self.sequence_to_mutation_sites is None:
                raise ValueError(
                    f"Cannot derive {embedding_type} embeddings without sequence_to_mutation_sites. "
                    "Run process_raw_data() before deriving mutation-site features."
                )

            mutation_sites = self._lookup_mutation_sites(self.sequence_to_mutation_sites, seq_id)
            if mutation_sites is None:
                raise KeyError(f"No mutation-site mapping found for sequence {seq_id}.")
            if not mutation_sites:
                derived[seq_id] = None
                continue
            if max(mutation_sites) >= embedding.shape[0] or min(mutation_sites) < 0:
                raise ValueError(
                    f"Mutation sites {mutation_sites} are out of bounds for sequence {seq_id} "
                    f"with length {embedding.shape[0]}."
                )
            mutation_embeddings = embedding[mutation_sites]
            if embedding_type == "per_residue":
                derived[seq_id] = mutation_embeddings
            else:
                derived[seq_id] = mutation_embeddings.mean(axis=0)
        return derived

    @staticmethod
    def _lookup_mutation_sites(sequence_to_mutation_sites: dict, seq_id) -> list[int] | None:
        candidates = [seq_id]
        if isinstance(seq_id, np.integer):
            candidates.append(int(seq_id))
        else:
            try:
                candidates.append(int(seq_id))
            except (TypeError, ValueError):
                pass
        candidates.append(str(seq_id))
        for candidate in candidates:
            if candidate in sequence_to_mutation_sites:
                return sequence_to_mutation_sites[candidate]
        return None

    @classmethod
    def _derive_mean_pool_features(cls, layer_embeddings: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        mean_pool = {}
        for seq_id, embedding in layer_embeddings.items():
            if embedding is None:
                mean_pool[seq_id] = None
                continue
            embedding = np.asarray(embedding)
            if embedding.ndim != 2:
                raise ValueError(
                    f"Cannot mean-pool embedding for sequence {seq_id} with shape {embedding.shape}."
                )
            mean_pool[seq_id] = embedding.mean(axis=0)
        return mean_pool

    @classmethod
    def _derive_max_pool_features(cls, layer_embeddings: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        max_pool = {}
        for seq_id, embedding in layer_embeddings.items():
            if embedding is None:
                max_pool[seq_id] = None
                continue
            embedding = np.asarray(embedding)
            if embedding.ndim != 2:
                raise ValueError(
                    f"Cannot max-pool embedding for sequence {seq_id} with shape {embedding.shape}."
                )
            max_pool[seq_id] = embedding.max(axis=0)
        return max_pool

    @classmethod
    def _derive_per_residue_features(
        cls,
        layer_embeddings: dict[str, np.ndarray],
        sequence_to_mutation_sites: dict,
    ) -> dict[str, np.ndarray]:
        if sequence_to_mutation_sites is None:
            raise ValueError(
                "Cannot derive per_residue embeddings without sequence_to_mutation_sites. "
                "Run process_raw_data() before deriving mutation-site features."
            )
        per_residue = {}
        for seq_id, embedding in layer_embeddings.items():
            if embedding is None:
                per_residue[seq_id] = None
                continue
            embedding = np.asarray(embedding)
            if embedding.ndim != 2:
                raise ValueError(
                    f"Cannot select mutation-site residues for sequence {seq_id} with shape {embedding.shape}."
                )
            mutation_sites = cls._lookup_mutation_sites(sequence_to_mutation_sites, seq_id)
            if mutation_sites is None:
                raise KeyError(f"No mutation-site mapping found for sequence {seq_id}.")
            if not mutation_sites:
                per_residue[seq_id] = None
                continue
            if max(mutation_sites) >= embedding.shape[0] or min(mutation_sites) < 0:
                raise ValueError(
                    f"Mutation sites {mutation_sites} are out of bounds for sequence {seq_id} "
                    f"with length {embedding.shape[0]}."
                )
            per_residue[seq_id] = embedding[mutation_sites]
        return per_residue

    @staticmethod
    def _pool_per_residue_features(per_residue_embeddings: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        pooled = {}
        for seq_id, embedding in per_residue_embeddings.items():
            if embedding is None:
                pooled[seq_id] = None
                continue
            embedding = np.asarray(embedding)
            if embedding.ndim == 1:
                pooled[seq_id] = embedding
            elif embedding.ndim == 2:
                pooled[seq_id] = embedding.mean(axis=0)
            else:
                raise ValueError(
                    f"Cannot mutation-pool per-residue embedding for sequence {seq_id} "
                    f"with shape {embedding.shape}."
                )
        return pooled

    def _save_layer_feature_caches(self, layer_embeddings: dict[str, np.ndarray], layer: str | int) -> None:
        mean_pool = self._derive_mean_pool_features(layer_embeddings)
        max_pool = self._derive_max_pool_features(layer_embeddings)
        per_residue = self._derive_per_residue_features(layer_embeddings, self.sequence_to_mutation_sites)

        if self._use_memory():
            self.sequence_to_features[self._feature_key("none", layer, "mean_pool")] = mean_pool
            self.sequence_to_features[self._feature_key("none", layer, "max_pool")] = max_pool
            self.sequence_to_features[self._feature_key("none", layer, "per_residue")] = per_residue
        if self._use_disk():
            self._save_pickle(mean_pool, self._embedding_path(layer, "mean_pool"))
            self._save_pickle(max_pool, self._embedding_path(layer, "max_pool"))
            self._save_pickle(per_residue, self._embedding_path(layer, "per_residue"))

    @classmethod
    def _build_feature_chunks(
        cls,
        embeddings: dict[str, np.ndarray],
        layer: str | int,
        sequence_to_mutation_sites: dict,
    ) -> dict:
        first_embedding = next((np.asarray(embedding) for embedding in embeddings.values() if embedding is not None), None)
        if first_embedding is None:
            raise ValueError("No non-empty embeddings were created.")

        if layer == "all":
            layer_axis = 1 if first_embedding.ndim == 3 else 0
            layers = range(first_embedding.shape[layer_axis])
        else:
            layers = [layer]

        feature_chunks = {"mean_pool": {}, "max_pool": {}, "per_residue": {}}
        for layer_value in layers:
            layer_embeddings = cls._select_layer(embeddings, layer_value, allow_per_residue=True)
            layer_label = cls._layer_label(layer_value)
            feature_chunks["mean_pool"][layer_label] = cls._derive_mean_pool_features(layer_embeddings)
            feature_chunks["max_pool"][layer_label] = cls._derive_max_pool_features(layer_embeddings)
            feature_chunks["per_residue"][layer_label] = cls._derive_per_residue_features(
                layer_embeddings,
                sequence_to_mutation_sites,
            )

        return feature_chunks

    @staticmethod
    def _require_vector_features(seq_to_features: dict[str, np.ndarray], context: str) -> None:
        bad = [
            (seq_id, np.asarray(feature).shape)
            for seq_id, feature in seq_to_features.items()
            if feature is None or np.asarray(feature).ndim != 1
        ]
        if bad:
            seq_id, shape = bad[0]
            raise ValueError(
                f"{context} requires one vector per sequence, but sequence {seq_id} has feature shape {shape}. "
                "Use embedding_type='mutation_pooled' or embedding_type='mean_pool', or add an abstraction "
                "that explicitly handles per-residue features."
            )

    @staticmethod
    def _drop_missing_features(
        sequence_dataframe: pd.DataFrame | None,
        seq_to_features: dict[str, np.ndarray],
        context: str,
    ) -> tuple[pd.DataFrame | None, dict[str, np.ndarray]]:
        filtered_features = {
            seq_id: feature
            for seq_id, feature in seq_to_features.items()
            if feature is not None
        }
        dropped_ids = set(seq_to_features) - set(filtered_features)
        if dropped_ids:
            print(f"{context}: dropping {len(dropped_ids)} sequence(s) with no mutation-site features.")

        if sequence_dataframe is None:
            return None, filtered_features

        sequence_dataframe = sequence_dataframe[
            sequence_dataframe["SequenceIndex"].isin(filtered_features)
        ].copy()
        if sequence_dataframe.empty:
            raise ValueError(f"{context}: no sequence rows remain after filtering missing mutation-site features.")
        return sequence_dataframe, filtered_features

    @staticmethod
    def _per_residue_feature_id(seq_id, mutation_site: int) -> str:
        return f"{seq_id}__site_{mutation_site}"

    @staticmethod
    def _parse_per_residue_feature_id(feature_id) -> tuple[str, int] | None:
        feature_id = str(feature_id)
        for sep in ("__site_", "__residue_"):
            if sep not in feature_id:
                continue
            parent_id, site = feature_id.rsplit(sep, 1)
            try:
                return parent_id, int(site)
            except ValueError:
                return None
        return None

    @staticmethod
    def _seq_id_lookup_keys(seq_id) -> tuple:
        keys = [seq_id, str(seq_id)]
        if isinstance(seq_id, np.integer):
            keys.append(int(seq_id))
        else:
            try:
                keys.append(int(seq_id))
            except (TypeError, ValueError):
                pass
        deduped = []
        for key in keys:
            if key not in deduped:
                deduped.append(key)
        return tuple(deduped)

    @classmethod
    def _expand_per_residue_feature_vectors(
        cls,
        seq_to_features: dict,
        sequence_to_mutation_sites: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        expanded_features = {}
        parent_to_feature_ids = {}

        def register_parent(parent_id, feature_id):
            for key in cls._seq_id_lookup_keys(parent_id):
                parent_to_feature_ids.setdefault(key, []).append(feature_id)

        for seq_id, feature in seq_to_features.items():
            if feature is None:
                continue
            feature = np.asarray(feature)
            if feature.ndim == 1:
                parsed = cls._parse_per_residue_feature_id(seq_id)
                expanded_features[seq_id] = feature
                if parsed is None:
                    register_parent(seq_id, seq_id)
                else:
                    parent_id, _ = parsed
                    register_parent(parent_id, seq_id)
                continue
            if feature.ndim != 2:
                raise ValueError(
                    f"per_residue features require vectors or matrices, "
                    f"but sequence {seq_id} has feature shape {feature.shape}."
                )

            mutation_sites = None
            if sequence_to_mutation_sites is not None:
                mutation_sites = cls._lookup_mutation_sites(sequence_to_mutation_sites, seq_id)
            if mutation_sites is None:
                mutation_sites = list(range(feature.shape[0]))
            if len(mutation_sites) != feature.shape[0]:
                raise ValueError(
                    f"Sequence {seq_id} has {feature.shape[0]} per_residue vectors but "
                    f"{len(mutation_sites)} mutation-site labels."
                )

            for row_idx, residue_feature in enumerate(feature):
                feature_id = cls._per_residue_feature_id(seq_id, mutation_sites[row_idx])
                expanded_features[feature_id] = residue_feature
                register_parent(seq_id, feature_id)

        return expanded_features, parent_to_feature_ids

    @classmethod
    def _expand_per_residue_features_for_inference(
        cls,
        sequence_dataframe: pd.DataFrame,
        seq_to_features: dict[str, np.ndarray],
        sequence_to_mutation_sites: dict | None = None,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """
        Convert sequence-level rows with multiple mutation-site residue embeddings
        into residue-level pseudo-individuals for raw per_residue inference.
        """
        expanded_features, parent_to_feature_ids = cls._expand_per_residue_feature_vectors(
            seq_to_features,
            sequence_to_mutation_sites,
        )

        if not expanded_features:
            raise ValueError("Inference: no residue-level per_residue features are available.")

        expanded_rows = []
        for _, row in sequence_dataframe.iterrows():
            seq_id = row["SequenceIndex"]
            feature_ids = None
            for key in cls._seq_id_lookup_keys(seq_id):
                if key in parent_to_feature_ids:
                    feature_ids = parent_to_feature_ids[key]
                    break
            if feature_ids is None:
                continue
            for residue_idx, feature_id in enumerate(feature_ids):
                expanded_row = row.to_dict()
                expanded_row["ParentSequenceIndex"] = seq_id
                expanded_row["ResidueFeatureIndex"] = residue_idx
                parsed = cls._parse_per_residue_feature_id(feature_id)
                if parsed is not None:
                    expanded_row["MutationSite"] = parsed[1]
                expanded_row["SequenceIndex"] = feature_id
                expanded_rows.append(expanded_row)

        if not expanded_rows:
            raise ValueError("Inference: no sequence rows matched residue-level per_residue features.")
        return pd.DataFrame(expanded_rows).reset_index(drop=True), expanded_features

    # ── MaveDB parsing helpers ───────────────────────────────────────────

    @staticmethod
    def _read_sequence_file(path: str | Path) -> str:
        with Path(path).open("r") as f:
            return "".join(line.strip() for line in f if not line.startswith(">")).upper()

    @staticmethod
    def _detect_reference_kind(reference_sequence: str) -> Literal["nucleotide", "protein"]:
        letters = set(reference_sequence.upper())
        if letters and letters.issubset(set("ACGTUN")) and len(reference_sequence) % 3 == 0:
            return "nucleotide"
        return "protein"

    @staticmethod
    def _strip_hgvs_prefix(hgvs_nt: str) -> str:
        hgvs_nt = str(hgvs_nt).strip()
        if ":c." in hgvs_nt:
            return "c." + hgvs_nt.split(":c.", 1)[1]
        return hgvs_nt

    @staticmethod
    def _read_mavedb_csv(path: str | Path) -> pd.DataFrame:
        with Path(path).open() as f:
            content = "".join(line for line in f if not line.startswith("#"))
        return pd.read_csv(io.StringIO(content))

    @staticmethod
    def _infer_aa_sequence_from_protein_reference(
        reference_aa_sequence: str,
        substitutions: list[tuple[int, str, str]],
        skip_stop_codons: bool = True,
    ) -> tuple[str | None, list[int] | None, str | None]:
        aa_to_codons = {}
        for codon, aa in CODON2AA.items():
            aa_to_codons.setdefault(aa, []).append(codon)

        # Group substitutions by codon so that multi-nucleotide variants within
        # a single codon are resolved against the same wildtype codon candidate.
        by_codon: dict[int, dict[int, tuple[str, str]]] = {}
        for nuc_pos, ref_nuc, alt_nuc in substitutions:
            aa_idx = nuc_pos // 3
            codon_phase = nuc_pos % 3
            if aa_idx < 0 or aa_idx >= len(reference_aa_sequence):
                return None, None, "position_out_of_range"
            phase_map = by_codon.setdefault(aa_idx, {})
            existing = phase_map.get(codon_phase)
            if existing is not None and existing != (ref_nuc, alt_nuc):
                return None, None, "conflicting_substitutions_at_same_codon_position"
            phase_map[codon_phase] = (ref_nuc, alt_nuc)

        aa_list = list(reference_aa_sequence)
        mutation_sites = set()
        for aa_idx, phase_map in by_codon.items():
            wt_aa = reference_aa_sequence[aa_idx]
            alt_aas = set()
            for codon in aa_to_codons.get(wt_aa, []):
                if any(codon[phase] != ref_nuc for phase, (ref_nuc, _) in phase_map.items()):
                    continue
                mut_codon = list(codon)
                for phase, (_, alt_nuc) in phase_map.items():
                    mut_codon[phase] = alt_nuc
                alt_aas.add(CODON2AA.get("".join(mut_codon), "X"))

            if not alt_aas:
                return None, None, "reference_nucleotide_incompatible_with_reference_aa"
            if len(alt_aas) > 1:
                return None, None, "ambiguous_protein_effect_from_protein_reference"

            alt_aa = next(iter(alt_aas))
            if skip_stop_codons and alt_aa == "*":
                return None, None, "stop_codon"
            if alt_aa != wt_aa:
                aa_list[aa_idx] = alt_aa
                mutation_sites.add(aa_idx)

        aa_sequence = "".join(aa_list)
        return aa_sequence, sorted(mutation_sites), None

    @staticmethod
    def _count_column_specs(df: pd.DataFrame) -> list[dict]:
        day_rep_re = re.compile(r"^count_day(?P<day>\d+)_rep(?P<replicate>\d+)$")
        legacy_re = re.compile(r"^(.+)_c_(\d+)$")

        day_rep_matches = []
        for col in df.columns:
            match = day_rep_re.match(col)
            if match:
                day_rep_matches.append({
                    "column": col,
                    "replicate": int(match.group("replicate")),
                    "day": int(match.group("day")),
                })

        if day_rep_matches:
            # count_library is the pre-selection (generation 0) baseline, shared
            # across replicates. The remaining count_day{N}_rep{M} columns are
            # ordered by day and remapped to generations 1, 2, ..., N so that
            # the trapezoidal time integration uses the library as t=0.
            unique_days = sorted({entry["day"] for entry in day_rep_matches})
            day_to_generation = {day: idx + 1 for idx, day in enumerate(unique_days)}
            replicate_indices = sorted({entry["replicate"] for entry in day_rep_matches})

            specs = []
            if "count_library" in df.columns:
                for rep_idx in replicate_indices:
                    specs.append({
                        "column": "count_library",
                        "replicate": rep_idx,
                        "replicate_name": f"rep{rep_idx}",
                        "generation": 0,
                    })
            for entry in day_rep_matches:
                specs.append({
                    "column": entry["column"],
                    "replicate": entry["replicate"],
                    "replicate_name": f"rep{entry['replicate']}",
                    "generation": day_to_generation[entry["day"]],
                })
            return sorted(specs, key=lambda x: (x["replicate"], x["generation"], x["column"]))

        specs = []

        legacy_seen = {}
        for col in df.columns:
            match = legacy_re.match(col)
            if match:
                legacy_seen.setdefault(match.group(1), {})[int(match.group(2))] = col
        rep_name_to_idx = {name: idx + 1 for idx, name in enumerate(sorted(legacy_seen))}
        for rep_name, gen_dict in legacy_seen.items():
            for gen_idx, col in gen_dict.items():
                specs.append({
                    "column": col,
                    "replicate": rep_name_to_idx[rep_name],
                    "replicate_name": rep_name,
                    "generation": gen_idx,
                })
        return sorted(specs, key=lambda x: (x["replicate"], x["generation"], x["column"]))

    @staticmethod
    def _replicate_selected(spec: dict, use_replicates: list[str] | None) -> bool:
        if use_replicates is None:
            return True
        allowed = {str(rep) for rep in use_replicates}
        return str(spec["replicate"]) in allowed or spec["replicate_name"] in allowed

    def _build_sequence_dataframe_mavedb_primary_keyed(
        self,
        counts_csv_path: Path,
        reference_sequence: str,
        reference_kind: Literal["nucleotide", "protein"],
        use_replicates: list[str] | None = None,
        skip_stop_codons: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, str], dict[str, list[int]], pd.DataFrame]:
        counts_df = self._read_mavedb_csv(counts_csv_path)
        primary_key = self.input_data.primary_key
        if primary_key not in counts_df.columns:
            raise ValueError(f"MaveDB counts file must contain primary key column {primary_key!r}.")

        count_specs = self._count_column_specs(counts_df)
        if not count_specs:
            raise ValueError(
                "No supported count columns found. Expected count_day<day>_rep<rep> "
                "or legacy columns ending in '_c_<generation>'."
            )

        if reference_kind == "nucleotide":
            ref_nuc_seq = reference_sequence
            ref_aa_seq = translate_nuc_sequence(ref_nuc_seq)
        else:
            ref_nuc_seq = None
            ref_aa_seq = reference_sequence

        records = []
        sequence_to_protein_sequence = {self.input_data.wildtype_key: ref_aa_seq}
        sequence_to_mutation_sites = {self.input_data.wildtype_key: []}
        metadata_rows = [{
            "SequenceIndex": self.input_data.wildtype_key,
            primary_key: self.input_data.wildtype_key,
            "accession": None,
            "n_mutation_sites": 0,
            "mutation_sites": [],
            "parse_status": "wildtype_reference",
        }]

        skipped = {}
        for _, row in counts_df.iterrows():
            seq_id = str(row[primary_key])
            hgvs_for_parse = self._strip_hgvs_prefix(seq_id)

            if hgvs_for_parse == "_wt":
                aa_seq = ref_aa_seq
                mutation_sites = []
                parse_status = "wildtype_row"
            else:
                substitutions = parse_hgvs_nt(hgvs_for_parse)
                if not substitutions:
                    skipped["unsupported_hgvs"] = skipped.get("unsupported_hgvs", 0) + 1
                    continue

                if reference_kind == "nucleotide":
                    mut_nuc_seq = apply_substitutions(ref_nuc_seq, substitutions)
                    if mut_nuc_seq is None:
                        skipped["reference_nucleotide_mismatch"] = skipped.get("reference_nucleotide_mismatch", 0) + 1
                        continue
                    aa_seq = translate_nuc_sequence(mut_nuc_seq)
                    if skip_stop_codons and "*" in aa_seq:
                        skipped["stop_codon"] = skipped.get("stop_codon", 0) + 1
                        continue
                    mutation_sites = [
                        i
                        for i, (ref_aa, alt_aa) in enumerate(zip(ref_aa_seq, aa_seq))
                        if ref_aa != alt_aa
                    ]
                    parse_status = "ok"
                else:
                    aa_seq, mutation_sites, reason = self._infer_aa_sequence_from_protein_reference(
                        ref_aa_seq,
                        substitutions,
                        skip_stop_codons=skip_stop_codons,
                    )
                    if reason is not None:
                        skipped[reason] = skipped.get(reason, 0) + 1
                        continue
                    parse_status = "ok"

            sequence_to_protein_sequence[seq_id] = aa_seq
            sequence_to_mutation_sites[seq_id] = mutation_sites
            metadata_rows.append({
                "SequenceIndex": seq_id,
                primary_key: seq_id,
                "accession": row.get("accession"),
                "n_mutation_sites": len(mutation_sites),
                "mutation_sites": mutation_sites,
                "parse_status": parse_status,
            })

            for spec in count_specs:
                if not self._replicate_selected(spec, use_replicates):
                    continue
                count = row.get(spec["column"])
                frequency = float(count) if pd.notna(count) else 0.0
                records.append({
                    "SequenceIndex": seq_id,
                    primary_key: seq_id,
                    "Replicate": spec["replicate"],
                    "ReplicateName": spec["replicate_name"],
                    "Generation": spec["generation"],
                    "Frequency": frequency,
                    "CountColumn": spec["column"],
                })

        sequence_dataframe = pd.DataFrame(records)
        if sequence_dataframe.empty:
            raise ValueError("No MaveDB count rows remained after sequence reconstruction.")

        metadata = pd.DataFrame(metadata_rows)
        metadata.attrs["skipped_counts"] = skipped
        metadata.attrs["reference_kind"] = reference_kind
        return (
            sequence_dataframe.reset_index(drop=True),
            sequence_to_protein_sequence,
            sequence_to_mutation_sites,
            metadata,
        )

    def load_functional_scores(self) -> pd.DataFrame:
        if self.input_data.kind != "cellular" or self.input_data.scores_csv_path is None:
            raise ValueError("Functional score loading requires CellularDMSInput.scores_csv_path.")
        scores = self._read_mavedb_csv(self.input_data.scores_csv_path)
        primary_key = self.input_data.primary_key
        if primary_key not in scores.columns:
            raise ValueError(f"MaveDB scores file must contain primary key column {primary_key!r}.")
        scores = scores.copy()
        scores["SequenceIndex"] = scores[primary_key].astype(str)
        self.scores_dataframe = scores
        return scores

    def load_reference_sequence(self):
        """
        Load the reference sequence for the input DMS data.
        """
        if self.input_data.kind == 'cellular':
            self.reference_sequence = self._read_sequence_file(self.input_data.reference_nuc_path)
            reference_kind = self.input_data.reference_kind
            if reference_kind == "auto":
                reference_kind = self._detect_reference_kind(self.reference_sequence)
            self.reference_kind = reference_kind
        elif self.input_data.kind == 'viral':
            self.reference_sequence = get_reference_sequence_from_wildtypes(
                self.input_data.pre_files[0]
            )
            self.reference_kind = "protein"
        else:
            raise ValueError(f"Unsupported DMS data kind: {self.input_data.kind}")

    def process_raw_data(self, drop_stop_codons: bool = True) -> None:
        """
        Process the loaded DMS data to reconstruct protein 
        sequences and mutation site -> frequency time series.
        """
        if self.reference_sequence is None:
            self.load_reference_sequence()

        if self.input_data.kind == "cellular":
            (
                self.sequence_dataframe,
                self.sequence_to_protein_sequence,
                self.sequence_to_mutation_sites,
                self.sequence_metadata,
            ) = self._build_sequence_dataframe_mavedb_primary_keyed(
                self.input_data.mavedb_csv_path,
                self.reference_sequence,
                self.reference_kind,
                self.input_data.use_replicates,
                skip_stop_codons=drop_stop_codons,
            )
            if self.input_data.scores_csv_path is not None:
                self.load_functional_scores()

        elif self.input_data.kind == "viral":
            (
                self.sequence_dataframe,
                self.sequence_to_protein_sequence,
                self.sequence_to_mutation_sites,
            ) = build_sequence_dataframe(
                self.input_data.pre_files,
                self.input_data.post_files,
                self.reference_sequence,
            )
            self.sequence_metadata = pd.DataFrame({
                "SequenceIndex": list(self.sequence_to_protein_sequence),
                "mutation_sites": [
                    self.sequence_to_mutation_sites[idx]
                    for idx in self.sequence_to_protein_sequence
                ],
            })

        else:
            raise ValueError(f"Unsupported DMS data kind: {self.input_data.kind}")


    @staticmethod
    def _is_esmc_model(model_name: str) -> bool:
        return "ESMC" in model_name or "esmc" in model_name.lower()

    @staticmethod
    def _resolve_torch_dtype(name: str | None):
        if not name:
            return None
        key = str(name).lower()
        if key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if key in {"fp16", "float16", "half"}:
            return torch.float16
        if key in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unknown torch_dtype: {name!r}")

    @staticmethod
    def _load_embedding_model(model_name: str, torch_dtype: str | None = None):
        """Load tokenizer + HF model, picking the right class for ESMC vs ESM2.

        ESMC is published with an MLM head and is loaded via
        AutoModelForMaskedLM. ESM2 keeps the existing AutoModel path.

        device_map="auto" is used only when CUDA is available — accelerate
        leaves rotary inv_freq buffers on the meta device on MPS/CPU, which
        raises at first forward. On MPS/CPU we load normally and move with
        .to(device).

        torch_dtype (arg or ESMDMS_TORCH_DTYPE env var) controls precision.
        ESMC on CUDA defaults to bfloat16 so the 6B variant fits in ~12 GB.
        HF_TOKEN is forwarded to from_pretrained for gated checkpoints.
        """
        token = HF_TOKEN or None
        dtype_name = torch_dtype or os.environ.get("ESMDMS_TORCH_DTYPE")
        dtype = esmDMS._resolve_torch_dtype(dtype_name)
        is_esmc = esmDMS._is_esmc_model(model_name)
        if dtype is None and is_esmc and torch.cuda.is_available():
            dtype = torch.bfloat16

        load_kwargs: dict = {"token": token}
        if dtype is not None:
            load_kwargs["dtype"] = dtype

        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, token=token)
        if is_esmc:
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                # Multi-GPU: shard across devices. accelerate handles meta->real
                # initialization for all state-dict tensors.
                model = AutoModelForMaskedLM.from_pretrained(
                    model_name, device_map="auto", **load_kwargs
                )
            else:
                # Single-GPU (or CPU/MPS): load normally and move. Avoids
                # device_map="auto", which leaves ESMC's non-persistent rotary
                # buffer (inv_freq) on the meta device because it isn't in the
                # state dict for accelerate to materialize.
                model = AutoModelForMaskedLM.from_pretrained(model_name, **load_kwargs)
                if torch.cuda.is_available():
                    model = model.to("cuda")
                elif torch.backends.mps.is_available():
                    model = model.to("mps")
        else:
            model = AutoModel.from_pretrained(model_name, **load_kwargs)
        model.eval()
        return tokenizer, model

    @staticmethod
    def _embed_sequence(sequence: str, tokenizer, model) -> np.ndarray:
        """Embed one sequence, moving inputs to the model's device first.

        Returns (num_residues, num_layers, embedding_dim) hidden states, matching
        embedding_scripts.embed_sequences.embed_sequence but device-safe for ESMC.
        """
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        try:
            device = model.device
        except AttributeError:
            device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key != "special_tokens_mask"
        }
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True)
        # numpy has no bfloat16 dtype, so cast each layer to fp32 before exporting.
        layer_embeddings = [
            all_residue_representation(layer.float(), inputs)
            for layer in outputs.hidden_states
        ]
        return np.stack(layer_embeddings, axis=1)

    @staticmethod
    def _embed_sequence_feature_chunks(
        seq_id: str,
        sequence: str,
        tokenizer,
        model,
        mutation_sites: list[int],
        layer: str | int = "all",
    ) -> dict:
        """Derive ESMC layer features without materializing all hidden states."""
        if not (hasattr(model, "esmc") and hasattr(model.esmc, "embed")):
            embeddings = {seq_id: esmDMS._embed_sequence(sequence, tokenizer, model)}
            return esmDMS._build_feature_chunks(embeddings, layer, {seq_id: mutation_sites})

        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        try:
            device = model.device
        except AttributeError:
            device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        attention_mask = inputs.get("attention_mask")
        special_mask = inputs.get("special_tokens_mask")
        if attention_mask is None:
            residue_mask = torch.ones_like(inputs["input_ids"], dtype=torch.bool)
        else:
            residue_mask = attention_mask.bool()
        if special_mask is not None:
            residue_mask = residue_mask & ~special_mask.bool()
        residue_mask_1d = residue_mask.squeeze(0)

        n_residues = int(residue_mask_1d.sum().item())
        if n_residues == 0:
            raise ValueError(f"No residue tokens found for sequence {seq_id}.")
        if mutation_sites:
            if max(mutation_sites) >= n_residues or min(mutation_sites) < 0:
                raise ValueError(
                    f"Mutation sites {mutation_sites} are out of bounds for sequence {seq_id} "
                    f"with length {n_residues}."
                )
            mutation_index = torch.as_tensor(mutation_sites, device=device, dtype=torch.long)
        else:
            mutation_index = None

        n_layers = int(model.esmc.config.n_layers)
        if layer == "all":
            layers_to_save = set(range(n_layers + 1))
        else:
            layers_to_save = {int(str(layer).replace("Layer_", ""))}

        feature_chunks = {"mean_pool": {}, "max_pool": {}, "per_residue": {}}

        def save_layer_features(layer_idx: int, hidden_state: torch.Tensor) -> None:
            if layer_idx not in layers_to_save:
                return
            layer_residue_mask = residue_mask_1d.to(hidden_state.device)
            residue_state = hidden_state[0, layer_residue_mask, :].float()
            layer_label = esmDMS._layer_label(layer_idx)
            feature_chunks["mean_pool"][layer_label] = {
                seq_id: residue_state.mean(dim=0).detach().cpu().numpy()
            }
            feature_chunks["max_pool"][layer_label] = {
                seq_id: residue_state.max(dim=0).values.detach().cpu().numpy()
            }
            if mutation_index is None:
                per_residue = None
            else:
                per_residue = residue_state.index_select(
                    0,
                    mutation_index.to(residue_state.device),
                ).detach().cpu().numpy()
            feature_chunks["per_residue"][layer_label] = {seq_id: per_residue}

        with torch.no_grad():
            x = model.esmc.embed(inputs["input_ids"])
            for layer_idx, block in enumerate(model.esmc.transformer.blocks):
                save_layer_features(layer_idx, x)
                x, _ = block(x, None, output_attentions=False)
            save_layer_features(n_layers, model.esmc.transformer.norm(x))

        return feature_chunks

    def embed_sequences(self, seq_ids: list[str], out_path: str | None = None,
                        verbose: bool = False) -> dict[str, np.ndarray]:
        """
        Embed protein sequences using the specified ESM model and embedding method.
        This function will only embed the sequences in the parameter, I will
        then make a merging function that merges the embeddings into one big dictionary of seq_id -> embedding,
        this allows for the calculating of embeddings in batches and parallel jobs.
        Parameters:
        -----------
        seq_ids : list[str]
            A list of sequence indices to embed.
        out_path : str | None
            The path to save the embeddings.

        Returns:
        --------
        dict[str, np.ndarray]
            A dictionary mapping sequence indices to their corresponding embeddings.
        """
        esm_model = f"{self.config.embedding_model}"

        tokenizer, model = self._load_embedding_model(esm_model)

        seq_idx_to_embedding = {}

        current_time = time.time()
        for idx in seq_ids:
            prot_seq = self.sequence_to_protein_sequence[idx]
            layer_embeddings = self._embed_sequence(prot_seq, tokenizer, model)
            seq_idx_to_embedding[idx] = layer_embeddings
            if verbose:
                elapsed = time.time() - current_time
                print(f"Embedded sequence {idx} in {elapsed:.2f} seconds.")
                current_time = time.time()


        if self._use_memory():
            self.sequence_to_embeddings.update(seq_idx_to_embedding)

        # Save the embeddings to the specified path if it exists
        if out_path is not None:
            self._save_pickle(seq_idx_to_embedding, Path(out_path))
        return seq_idx_to_embedding
    

    def embed_all_sequences(self, layer: str = "all", test_num: int | None = None) -> None:
        """
        Embed all sequences and cache compact derived features by layer.

        Disk caches are written as separate mean_pool and mutation-site
        per_residue files. mutation_pooled is derived from per_residue on load.
        """
        if self.sequence_dataframe is None:
            raise ValueError("Sequence dataframe is not available. Please run process_raw_data() first.")
        if self.sequence_to_mutation_sites is None:
            raise ValueError("Mutation-site mapping is not available. Please run process_raw_data() first.")

        verbose = False
        seq_ids = sorted(self.sequence_to_protein_sequence, key=str)
        if test_num is not None:
            seq_ids = seq_ids[:test_num]
            verbose = True
        
        embeddings = self.embed_sequences(seq_ids, verbose=verbose)

        first_embedding = next((np.asarray(embedding) for embedding in embeddings.values() if embedding is not None), None)
        if first_embedding is None:
            raise ValueError("No non-empty embeddings were created.")
        n_layers = first_embedding.shape[1] if first_embedding.ndim == 3 else first_embedding.shape[0]
        if layer == 'all':
            for l in range(n_layers):
                layer_embeddings = self._select_layer(embeddings, l, allow_per_residue=True)
                self._save_layer_feature_caches(layer_embeddings, l)
        else:
            layer_embeddings = self._select_layer(embeddings, layer, allow_per_residue=True)
            self._save_layer_feature_caches(layer_embeddings, layer)

    def create_embedding_batch_job(
        self,
        job_dir: str | Path | None = None,
        n_chunks: int = 10,
        max_active_jobs: int | None = None,
        job_name: str = "esm_embed",
        partition: str = "dept_cpu",
        gres: str | None = None,
        constraint: str | None = None,
        cpus_per_task: int = 4,
        mem: str = "16G",
        time: str = "06:00:00",
        python_executable: str = "python3",
        scratch_root: str | Path = "/scr",
        hf_home: str | Path | None = None,
        torch_dtype: str | None = None,
        submit: bool = False,
    ) -> dict[str, Path | str]:
        """
        Create a Slurm array job that embeds this dataset's processed protein sequences.

        Run process_raw_data() first. Each array task writes separate
        mean_pool_embeddings_chunk_<idx>.pkl and
        per_residue_embeddings_chunk_<idx>.pkl files to scratch, then copies
        them back to job_dir. After the jobs finish, call
        merge_embedding_batch_outputs(job_dir).
        Set max_active_jobs to limit concurrently active Slurm array tasks.

        ESMC runs are forced onto CUDA jobs by default. If a caller leaves CPU
        defaults in place, the generated script is rewritten to use
        partition="dept_gpu", gres="gpu:1", constraint="C8", and
        torch_dtype="bfloat16". Use constraint="L40|A100" for larger ESMC
        checkpoints. Set hf_home on shared storage so array tasks share the
        downloaded weights.

        Note: this cluster encodes GPU model as a Slurm feature/constraint,
        not as a typed GRES — use gres="gpu:N" + constraint="L40", not
        gres="gpu:l40:N".
        """
        if not self._use_disk():
            raise ValueError("Embedding batch jobs require local_or_disk to be 'disk' or 'both'.")
        if self.sequence_dataframe is None or self.sequence_to_protein_sequence is None or self.sequence_to_mutation_sites is None:
            raise ValueError("Run process_raw_data() before creating an embedding batch job.")
        if n_chunks < 1:
            raise ValueError("n_chunks must be at least 1.")
        if max_active_jobs is not None and max_active_jobs < 1:
            raise ValueError("max_active_jobs must be at least 1 when specified.")

        is_esmc_job = self._is_esmc_model(str(self.config.embedding_model))
        if is_esmc_job:
            if partition in {"dept_cpu", "any_cpu", "big_memory"}:
                partition = "dept_gpu"
            if gres is None:
                gres = "gpu:1"
            if constraint is None:
                constraint = "C8"
            if torch_dtype is None:
                torch_dtype = "bfloat16"

        batch_dir = self._batch_dir(job_dir)
        logs_dir = batch_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        seq_ids = sorted(self.sequence_to_protein_sequence, key=str)
        if not seq_ids:
            raise ValueError("No sequences are available to embed.")
        payload = {
            "seq_ids": seq_ids,
            "sequence_to_protein_sequence": {
                seq_id: self.sequence_to_protein_sequence[seq_id] for seq_id in seq_ids
            },
            "sequence_to_mutation_sites": {
                seq_id: self.sequence_to_mutation_sites[seq_id] for seq_id in seq_ids
            },
            "embedding_model": self.config.embedding_model,
            "embedding_type": "derived_features",
            "dataset_name": self.config.dataset_name,
            "dataset_prefix": self._dataset_prefix(),
            "layer": "all",
            "n_chunks": n_chunks,
            "max_active_jobs": max_active_jobs,
            "batch_dir": str(batch_dir),
        }

        payload_path = self._batch_payload_path(batch_dir)
        self._save_pickle(payload, payload_path)

        script_path = batch_dir / "submit_embedding_array.sh"
        array_spec = f"0-{n_chunks - 1}"
        if max_active_jobs is not None:
            array_spec = f"{array_spec}%{max_active_jobs}"
        gres_line = f"#SBATCH --gres={gres}\n" if gres else ""
        constraint_line = f"#SBATCH --constraint={constraint}\n" if constraint else ""
        env_exports = []
        if hf_home is not None:
            env_exports.append(f"export HF_HOME={hf_home}")
        if torch_dtype is not None:
            env_exports.append(f"export ESMDMS_TORCH_DTYPE={torch_dtype}")
        env_block = ("\n".join(env_exports) + "\n") if env_exports else ""
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
{gres_line}{constraint_line}#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --array={array_spec}
#SBATCH --output={logs_dir}/slurm-%A_%a.out
#SBATCH --error={logs_dir}/slurm-%A_%a.err

set -euo pipefail
cd {Path.cwd()}

{env_block}SCRDIR={scratch_root}/${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
mkdir -p "$SCRDIR"
export TMPDIR="$SCRDIR"

{python_executable} -c "import sys, os; sys.path.insert(0, r'{Path.cwd()}'); import popDMS; from esmDMS import esmDMS; esmDMS.run_embedding_batch_chunk(r'{payload_path}', int(os.environ['SLURM_ARRAY_TASK_ID']), scratch_dir=os.environ['TMPDIR'])"
"""
        script_path.write_text(script)
        script_path.chmod(0o755)

        job_id = ""
        if submit:
            completed = subprocess.run(
                ["sbatch", str(script_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            job_id = completed.stdout.strip()

        return {
            "batch_dir": batch_dir,
            "payload_path": payload_path,
            "script_path": script_path,
            "job_id": job_id,
        }

    @staticmethod
    def run_embedding_batch_chunk(
        payload_path: str | Path,
        chunk_idx: int,
        scratch_dir: str | Path | None = None,
    ) -> dict[str, Path] | Path:
        """
        Worker entrypoint used by create_embedding_batch_job().
        """
        payload_path = Path(payload_path)
        with payload_path.open("rb") as f:
            payload = pickle.load(f)

        seq_ids = payload["seq_ids"]
        n_chunks = payload["n_chunks"]
        chunks = np.array_split(np.asarray(seq_ids), n_chunks)
        if chunk_idx < 0 or chunk_idx >= n_chunks:
            raise ValueError(f"chunk_idx must be between 0 and {n_chunks - 1}.")

        esm_model = f"{payload['embedding_model']}"
        if esmDMS._is_esmc_model(esm_model) and not torch.cuda.is_available():
            raise RuntimeError(
                "ESMC batch embedding requires a CUDA GPU. This worker has no CUDA device, "
                "so running would fall back to CPU and exhaust job memory. Recreate the "
                "embedding batch job with partition='dept_gpu', gres='gpu:1', and "
                "constraint='C8' (or 'L40|A100' for larger ESMC checkpoints)."
            )
        tokenizer, model = esmDMS._load_embedding_model(esm_model)

        if payload.get("sequence_to_mutation_sites") is not None:
            feature_chunks = {"mean_pool": {}, "max_pool": {}, "per_residue": {}}
            for seq_id in chunks[chunk_idx].tolist():
                prot_seq = payload["sequence_to_protein_sequence"][seq_id]
                seq_feature_chunks = esmDMS._embed_sequence_feature_chunks(
                    seq_id,
                    prot_seq,
                    tokenizer,
                    model,
                    payload["sequence_to_mutation_sites"][seq_id],
                    payload.get("layer", "all"),
                )
                for embedding_type, features_by_layer in seq_feature_chunks.items():
                    for layer_label, seq_to_features in features_by_layer.items():
                        feature_chunks[embedding_type].setdefault(layer_label, {})
                        feature_chunks[embedding_type][layer_label].update(seq_to_features)
                del seq_feature_chunks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            written_paths = {}
            for embedding_type, features_by_layer in feature_chunks.items():
                chunk_payload = {
                    "format": "esmDMS_embedding_feature_chunk_v2",
                    "embedding_type": embedding_type,
                    "features_by_layer": features_by_layer,
                }
                final_path = (
                    Path(payload["batch_dir"])
                    / f"{payload.get('dataset_prefix', '')}{embedding_type}_embeddings_chunk_{chunk_idx}.pkl"
                )
                if scratch_dir is None:
                    scratch_path = final_path
                else:
                    scratch_path = (
                        Path(scratch_dir)
                        / "esm_embed_saves"
                        / f"{payload.get('dataset_prefix', '')}{embedding_type}_embeddings_chunk_{chunk_idx}.pkl"
                    )
                    scratch_path.parent.mkdir(parents=True, exist_ok=True)
                with scratch_path.open("wb") as f:
                    pickle.dump(chunk_payload, f)
                if scratch_path != final_path:
                    shutil.copy2(scratch_path, final_path)
                written_paths[embedding_type] = final_path
            return written_paths

        out = {}
        for seq_id in chunks[chunk_idx].tolist():
            prot_seq = payload["sequence_to_protein_sequence"][seq_id]
            out[seq_id] = esmDMS._embed_sequence(prot_seq, tokenizer, model)

        dataset_prefix = payload.get("dataset_prefix", "")
        final_path = Path(payload["batch_dir"]) / f"{dataset_prefix}embeddings_chunk_{chunk_idx}.pkl"
        if scratch_dir is None:
            scratch_path = final_path
        else:
            scratch_path = Path(scratch_dir) / "esm_embed_saves" / f"{dataset_prefix}embeddings_chunk_{chunk_idx}.pkl"
            scratch_path.parent.mkdir(parents=True, exist_ok=True)

        with scratch_path.open("wb") as f:
            pickle.dump(out, f)
        if scratch_path != final_path:
            shutil.copy2(scratch_path, final_path)
        return final_path

    def create_embedding_batch_merge_job(
        self,
        job_dir: str | Path | None = None,
        layer: str | int = "all",
        n_chunks: int | None = None,
        save_layers: bool = True,
        job_name: str = "esm_embed_merge",
        partition: str = "dept_cpu",
        cpus_per_task: int = 1,
        mem: str = "16G",
        time: str = "01:00:00",
        python_executable: str = "python3",
        submit: bool = False,
    ) -> dict[str, Path | str]:
        """
        Create a Slurm job that merges completed embedding chunk outputs.
        """
        if not self._use_disk():
            raise ValueError("Embedding batch merge jobs require local_or_disk to be 'disk' or 'both'.")

        batch_dir = self._batch_dir(job_dir)
        logs_dir = batch_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        payload = {
            "job_dir": str(batch_dir),
            "layer": layer,
            "n_chunks": n_chunks,
            "save_layers": save_layers,
            "embedding_model": self.config.embedding_model,
            "embedding_type": self.config.embedding_type,
            "local_or_disk": "disk",
            "save_dir": self.config.save_dir,
            "dataset_name": self.config.dataset_name,
        }
        payload_path = batch_dir / f"{self._dataset_prefix()}embedding_merge_payload.pkl"
        self._save_pickle(payload, payload_path)

        script_path = batch_dir / "submit_embedding_merge.sh"
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output={logs_dir}/merge-%j.out
#SBATCH --error={logs_dir}/merge-%j.err

set -euo pipefail
cd {Path.cwd()}

{python_executable} -c "import sys; sys.path.insert(0, r'{Path.cwd()}'); import popDMS; from esmDMS import esmDMS; esmDMS.run_embedding_batch_merge(r'{payload_path}')"
"""
        script_path.write_text(script)
        script_path.chmod(0o755)

        job_id = ""
        if submit:
            completed = subprocess.run(
                ["sbatch", str(script_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            job_id = completed.stdout.strip()

        return {
            "batch_dir": batch_dir,
            "payload_path": payload_path,
            "script_path": script_path,
            "job_id": job_id,
        }

    @staticmethod
    def run_embedding_batch_merge(payload_path: str | Path) -> dict[str, np.ndarray]:
        """
        Worker entrypoint used by create_embedding_batch_merge_job().
        """
        payload = esmDMS._load_pickle(Path(payload_path))
        runner = object.__new__(esmDMS)
        runner.config = ESMDMSConfig(
            embedding_model=payload["embedding_model"],
            embedding_type=payload.get("embedding_type", "per_residue"),
            local_or_disk=payload["local_or_disk"],
            save_dir=payload["save_dir"],
            dataset_name=payload["dataset_name"],
        )
        runner.sequence_to_mutation_sites = None
        runner.sequence_to_embeddings = {}
        runner.sequence_to_features = {}
        return runner.merge_embedding_batch_outputs(
            job_dir=payload["job_dir"],
            layer=payload["layer"],
            n_chunks=payload["n_chunks"],
            save_layers=payload["save_layers"],
        )

    def merge_embedding_batch_outputs(
        self,
        job_dir: str | Path | None = None,
        layer: str | int = "all",
        n_chunks: int | None = None,
        save_layers: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Merge embedding chunk pickle files from create_embedding_batch_job().
        """
        batch_dir = self._batch_dir(job_dir)
        payload_path = self._batch_payload_path(batch_dir)
        payload = self._load_pickle(payload_path) if payload_path.is_file() else None
        if payload_path.is_file() and n_chunks is None:
            n_chunks = payload["n_chunks"]
        if payload is not None and payload.get("embedding_type") == "derived_features":
            if getattr(self, "sequence_to_mutation_sites", None) is None:
                self.sequence_to_mutation_sites = payload.get("sequence_to_mutation_sites")

            feature_types = ("mean_pool", "max_pool", "per_residue")
            if n_chunks is None:
                feature_chunk_files = {
                    embedding_type: sorted(
                        batch_dir.glob(f"{self._dataset_prefix()}{embedding_type}_embeddings_chunk_*.pkl")
                    )
                    for embedding_type in feature_types
                }
            else:
                feature_chunk_files = {
                    embedding_type: [
                        self._batch_feature_chunk_path(batch_dir, idx, embedding_type)
                        for idx in range(n_chunks)
                    ]
                    for embedding_type in feature_types
                }

            missing_feature_chunks = [
                path
                for chunk_paths in feature_chunk_files.values()
                for path in chunk_paths
                if not path.is_file()
            ]
            if missing_feature_chunks:
                raise FileNotFoundError(f"Missing embedding feature chunk files: {missing_feature_chunks}")

            merged_features = {}
            for embedding_type, chunk_paths in feature_chunk_files.items():
                if not chunk_paths:
                    raise FileNotFoundError(
                        f"No {embedding_type} embedding chunk files found in {batch_dir}."
                    )
                for path in chunk_paths:
                    chunk = self._load_pickle(path)
                    if not (
                        isinstance(chunk, dict)
                        and chunk.get("format") == "esmDMS_embedding_feature_chunk_v2"
                        and chunk.get("embedding_type") == embedding_type
                    ):
                        raise ValueError(f"Unexpected {embedding_type} feature chunk format in {path}.")
                    for layer_label, seq_to_features in chunk["features_by_layer"].items():
                        merged_features.setdefault(layer_label, {"mean_pool": {}, "max_pool": {}, "per_residue": {}})
                        merged_features[layer_label][embedding_type].update(seq_to_features)

            for layer_label, layer_features in merged_features.items():
                if self._use_memory():
                    self.sequence_to_features[self._feature_key("none", layer_label, "mean_pool")] = layer_features["mean_pool"]
                    self.sequence_to_features[self._feature_key("none", layer_label, "max_pool")] = layer_features["max_pool"]
                    self.sequence_to_features[self._feature_key("none", layer_label, "per_residue")] = layer_features["per_residue"]
                if save_layers and self._use_disk():
                    self._save_pickle(layer_features["mean_pool"], self._embedding_path(layer_label, "mean_pool"))
                    self._save_pickle(layer_features["max_pool"], self._embedding_path(layer_label, "max_pool"))
                    self._save_pickle(layer_features["per_residue"], self._embedding_path(layer_label, "per_residue"))
            return merged_features

        if n_chunks is None:
            chunk_files = sorted(batch_dir.glob(f"{self._dataset_prefix()}embeddings_chunk_*.pkl"))
        else:
            chunk_files = [self._batch_chunk_path(batch_dir, idx) for idx in range(n_chunks)]
        if not chunk_files:
            raise FileNotFoundError(f"No embedding chunk files found in {batch_dir}.")

        missing = [path for path in chunk_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing embedding chunk files: {missing}")

        if getattr(self, "sequence_to_mutation_sites", None) is None and payload is not None:
            self.sequence_to_mutation_sites = payload.get("sequence_to_mutation_sites")

        merged = {}
        for path in chunk_files:
            chunk = self._load_pickle(path)
            merged.update(chunk)

        if self._use_memory():
            self.sequence_to_embeddings.update(merged)

        return merged
        


    def load_embeddings(self, layer: str | int, embedding_type: EmbeddingType | None = None) -> dict[str, np.ndarray]:
        """
        Load embeddings from disk if they exist.

        Returns:
        --------
        dict[str, np.ndarray] | None
            A dictionary mapping sequence indices to their corresponding embeddings, or None if no saved embeddings are found.
        """

        embedding_type = self._embedding_type(embedding_type)
        key = self._feature_key("none", layer, embedding_type)
        if self._use_memory() and self.sequence_to_features.get(key) is not None:
            #print("Embeddings already exist in memory. Returning existing embeddings.")
            return self.sequence_to_features[key]

        per_residue_key = self._feature_key("none", layer, "per_residue")
        if embedding_type == "mutation_pooled":
            if self._use_memory() and self.sequence_to_features.get(per_residue_key) is not None:
                embeddings = self._pool_per_residue_features(self.sequence_to_features[per_residue_key])
                self.sequence_to_features[key] = embeddings
                return embeddings

            if self._use_disk():
                per_residue_path = self._embedding_path(layer, "per_residue")
                if per_residue_path.is_file():
                    #print(f"Loading per_residue embeddings from {per_residue_path}")
                    per_residue_embeddings = self._load_pickle(per_residue_path)
                    embeddings = self._pool_per_residue_features(per_residue_embeddings)
                    if self._use_memory():
                        self.sequence_to_features[per_residue_key] = per_residue_embeddings
                        self.sequence_to_features[key] = embeddings
                    return embeddings

        base_key = self._embedding_key(layer)
        if self._use_memory() and self.sequence_to_features.get(base_key) is not None:
            if embedding_type == "mutation_pooled":
                per_residue_embeddings = self._derive_embedding_type(self.sequence_to_features[base_key], "per_residue")
                embeddings = self._pool_per_residue_features(per_residue_embeddings)
                self.sequence_to_features[per_residue_key] = per_residue_embeddings
                if self._use_disk():
                    self._save_pickle(per_residue_embeddings, self._embedding_path(layer, "per_residue"))
            else:
                embeddings = self._derive_embedding_type(self.sequence_to_features[base_key], embedding_type)
                if self._use_disk():
                    self._save_pickle(embeddings, self._embedding_path(layer, embedding_type))
            self.sequence_to_features[key] = embeddings
            return embeddings

        if self._use_memory() and self.sequence_to_embeddings:
            all_data_embeddings = self._select_layer(self.sequence_to_embeddings, layer, allow_per_residue=True)
            if embedding_type == "mutation_pooled":
                per_residue_embeddings = self._derive_embedding_type(all_data_embeddings, "per_residue")
                layer_embeddings = self._pool_per_residue_features(per_residue_embeddings)
                self.sequence_to_features[per_residue_key] = per_residue_embeddings
                if self._use_disk():
                    self._save_pickle(per_residue_embeddings, self._embedding_path(layer, "per_residue"))
            else:
                layer_embeddings = self._derive_embedding_type(all_data_embeddings, embedding_type)
                if self._use_disk():
                    self._save_pickle(layer_embeddings, self._embedding_path(layer, embedding_type))
            self.sequence_to_features[key] = layer_embeddings
            return layer_embeddings

        if self._use_disk():
            save_path = self._embedding_path(layer, embedding_type)
            if embedding_type != "mutation_pooled" and save_path.is_file():
                #print(f"Loading {embedding_type} embeddings from {save_path}")
                embeddings = self._load_pickle(save_path)
                if self._use_memory():
                    self.sequence_to_features[key] = embeddings
                return embeddings
            if embedding_type == "mutation_pooled":
                raise FileNotFoundError(
                    f"No saved per_residue embeddings found at {self._embedding_path(layer, 'per_residue')}. "
                    f"Run embed_all_sequences({layer!r}) first."
                )
            raise FileNotFoundError(f"No saved {embedding_type} embeddings found at {save_path}. Run embed_all_sequences({layer!r}) first.")

        raise ValueError("No embeddings found in memory. Run embed_all_sequences() before inference.")


    def get_mutation_sites(self, indices: list[str]) -> list[list[int]]:
        """
        Get the mutation sites for the specified sequence indices.

        Parameters:
        -----------
        indices : list[str]
            A list of sequence indices.

        Returns:
        --------
        list[list[int]]
            A list of mutation sites for each sequence.
        """
        mutation_sites = []
        for idx in indices:
            if self.sequence_to_mutation_sites is None:
                raise ValueError("Sequence to mutation sites mapping is not available. Please run process_raw_data() first.")
            mutation_sites.append(self.sequence_to_mutation_sites[idx])
        return mutation_sites


    def create_feature_space(self, layer: str, 
                            method: AbstractionMethod = 'none',
                            method_params: dict | None = None,
                            embedding_type: EmbeddingType | None = None,
                            force_recompute: bool = False) -> dict[str, np.ndarray]:
        """
        Create an abstraction of the embeddings using the specified method.

        Parameters:
        -----------
        layer : str
            The layer from which to extract features.
        method : Literal['none', 'PCA', 'SAE', 'DeltaSAE', 'SPCA']
            The method to use for creating the abstraction.
        method_params : dict | None
            Parameters for the abstraction method.
        force_recompute : bool
            If True, bypass cached abstracted features and recompute them.

        Returns:
        --------
        dict[str, np.ndarray]
            A dictionary mapping sequence indices to their corresponding features.
        """
        _params = dict(method_params or {})
        force_recompute = bool(force_recompute or _params.pop("force_recompute", False))
        embedding_type = self._embedding_type(embedding_type)
        method = self._abstraction_type(method)
        if method == 'none':
            #print("No abstraction method specified. Using raw embeddings as features.")
            features = self.load_embeddings(layer, embedding_type)
            _, features = self._drop_missing_features(None, features, f"{embedding_type} features")
            if self._use_disk() and embedding_type != "mutation_pooled":
                self._save_pickle(features, self._feature_path(method, layer, embedding_type))
            return features

        embeddings = self.load_embeddings(layer, embedding_type)
        _, embeddings = self._drop_missing_features(None, embeddings, f"{method} abstraction")
        if embedding_type == "per_residue":
            embeddings, _ = self._expand_per_residue_feature_vectors(
                embeddings,
                self.sequence_to_mutation_sites,
            )
        self._require_vector_features(embeddings, f"{method} abstraction with embedding_type={embedding_type!r}")
        key = self._feature_key(method, layer, embedding_type)

        if not force_recompute and self._use_memory() and self.sequence_to_features.get(key) is not None:
            #print("Abstracted features already exist in memory. Returning existing features.")
            return self.sequence_to_features[key]

        if not force_recompute and self._use_disk():
            save_path = self._feature_path(method, layer, embedding_type)
            if save_path.is_file():
                #print(f"Abstracted already saved in {save_path}")
                abstracted_features = self._load_pickle(save_path)
                if self._use_memory():
                    self.sequence_to_features[key] = abstracted_features
                return abstracted_features

        #print("No abstracted features found. Creating new features.")
        _params.setdefault("_layer", layer)
        _params.setdefault("_embedding_type", embedding_type)
        abstracted_features = self._create_feature_space(embeddings, method, _params)
        if self._use_memory():
            self.sequence_to_features[key] = abstracted_features
        if self._use_disk():
            self._save_pickle(abstracted_features, self._feature_path(method, layer, embedding_type))
        return abstracted_features


    def _create_feature_space(self, embeddings: dict[str, np.ndarray], 
                        method: AbstractionMethod = 'none',
                        method_params: dict | None = None) -> dict[str, np.ndarray]:
        # Placeholder for abstraction implementation
        method = self._abstraction_type(method)
        base_method = self._abstraction_base_method(method)
        if base_method == 'none':
            return embeddings
        elif base_method == 'PCA':
            # Implement PCA abstraction here
            return self._pca_abstraction(embeddings, method_params)
        elif base_method == 'SAE':
            # Implement SAE abstraction here
            return self._sae_abstraction(embeddings, method_params)
        elif base_method == 'DeltaSAE':
            return self._delta_sae_abstraction(embeddings, method_params)
        elif base_method == 'SPCA':
            # Implement SPCA abstraction here
            return self._spca_abstraction(embeddings, method_params)
        else:
            raise ValueError(f"Unsupported abstraction method: {method}")


    def _pca_abstraction(self, embeddings: dict[str, np.ndarray], method_params: dict | None) -> dict[str, np.ndarray]:
        """
        Perform PCA abstraction on the embeddings.

        Parameters:
        -----------
        embeddings : dict[str, np.ndarray]
            A dictionary mapping sequence indices to their corresponding embeddings.
        method_params : dict | None
            Parameters for PCA, such as the number of components.

        Returns:
        --------
        dict[str, np.ndarray]
            A dictionary mapping sequence indices to their corresponding PCA-transformed features.
        """
        raise NotImplementedError("PCA abstraction is not implemented yet.")


    def _sae_abstraction(self, embeddings: dict[str, np.ndarray], method_params: dict | None) -> dict[str, np.ndarray]:
        """
        Perform SAE abstraction on the embeddings.

        Parameters:
        -----------
        embeddings : dict[str, np.ndarray]
            A dictionary mapping sequence indices to their corresponding embeddings.
        method_params : dict | None
            Parameters for SAE, such as the architecture and training parameters.

        Returns:
        --------
        dict[str, np.ndarray]             
        A dictionary mapping sequence indices to their corresponding SAE-transformed features.
        """
        params = method_params or {}
        layer = params.get("_layer", "unknown")
        embedding_type = params.get("_embedding_type")

        # ── Hyper-parameters ──────────────────────────────────────────────
        sparsity_coeff: float = params.get("sparsity_coeff", 1e-3)
        lr: float = params.get("lr", 1e-3)
        epochs: int = params.get("epochs", 200)
        batch_size: int = params.get("batch_size", 64)
        train_frac: float = params.get("train_frac", 0.8)
        normalize_decoder: bool = params.get("normalize_decoder", True)
        activity_threshold: float = params.get("activity_threshold", 1e-3)
        seed: int = params.get("seed", 42)
        sparsity_mode: SparsityMode = params.get("sparsity_mode", "normal")
        k: int | None = params.get("k")
        run_label: str | None = params.get("run_label")
        pretrained_model_path: str | Path | None = params.get("pretrained_model_path")
        if sparsity_mode in {"topk", "batchtopk"} and (k is None or k <= 0):
            raise ValueError(
                f"sparsity_mode={sparsity_mode!r} requires a positive integer 'k' in method_params."
            )

        # ── Build data matrix ─────────────────────────────────────────────
        seq_ids = list(embeddings.keys())
        X = np.asarray([embeddings[sid] for sid in seq_ids], dtype=np.float32)
        n_samples, input_dim = X.shape

        n_features: int = params.get("n_features", input_dim * 2)

        # ── Train / test split ────────────────────────────────────────────
        n_train = max(1, int(n_samples * train_frac))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_samples)
        train_idx, test_idx = perm[:n_train], perm[n_train:]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train = torch.from_numpy(X[train_idx]).to(device)
        X_test = torch.from_numpy(X[test_idx]).to(device) if len(test_idx) else None

        train_losses: list[float] = []
        test_losses: list[float] = []
        checkpoint_active_mask = None
        if pretrained_model_path is not None:
            checkpoint = torch.load(Path(pretrained_model_path), map_location=device)
            input_dim_ckpt = checkpoint.get("input_dim", input_dim)
            if input_dim_ckpt != input_dim:
                raise ValueError(
                    f"Pretrained SAE input_dim={input_dim_ckpt} does not match embedding input_dim={input_dim}."
                )
            n_features = checkpoint.get("n_features", n_features)
            normalize_decoder = checkpoint.get("normalize_decoder", normalize_decoder)
            sparsity_mode = checkpoint.get("sparsity_mode", sparsity_mode)
            k = checkpoint.get("k", k)
            model = SparseAutoencoder(
                input_dim,
                n_features,
                normalize_decoder,
                sparsity_mode=sparsity_mode,
                k=k,
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            checkpoint_active_mask = checkpoint.get("active_mask")
            print(f"Loaded pretrained SAE model from {pretrained_model_path}")
        else:
            # ── Model + optimizer ─────────────────────────────────────────
            model = SparseAutoencoder(
                input_dim,
                n_features,
                normalize_decoder,
                sparsity_mode=sparsity_mode,
                k=k,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # ── Training loop ─────────────────────────────────────────────
            for epoch in range(epochs):
                model.train()
                epoch_perm = torch.randperm(len(X_train), device=device)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, len(X_train), batch_size):
                    batch = X_train[epoch_perm[start : start + batch_size]]
                    optimizer.zero_grad()
                    x_hat, z = model(batch)
                    loss = torch.nn.functional.mse_loss(x_hat, batch)
                    # In topk / batchtopk modes the encoder masks activations directly,
                    # so the L1 penalty is redundant and is omitted (standard TopK-SAE recipe).
                    if sparsity_mode == "normal":
                        loss = loss + sparsity_coeff * z.abs().mean()
                    loss.backward()
                    optimizer.step()
                    if normalize_decoder:
                        model._renorm_decoder()
                    epoch_loss += loss.item()
                    n_batches += 1
                train_losses.append(epoch_loss / max(n_batches, 1))

                model.eval()
                with torch.no_grad():
                    if X_test is not None:
                        x_hat_t, _ = model(X_test)
                        test_losses.append(torch.nn.functional.mse_loss(x_hat_t, X_test).item())
                    else:
                        test_losses.append(float("nan"))

                if (epoch + 1) % 50 == 0:
                    print(
                        f"  SAE epoch {epoch + 1}/{epochs}  "
                        f"train_loss={train_losses[-1]:.5f}  "
                        f"test_loss={test_losses[-1]:.5f}"
                    )

        # ── Extract activations for all sequences ─────────────────────────
        model.eval()
        X_all_t = torch.from_numpy(X).to(device)
        with torch.no_grad():
            X_recon_t, Z_all_t = model(X_all_t)
        Z_all = Z_all_t.cpu().numpy()          # (n_samples, n_features)
        X_recon = X_recon_t.cpu().numpy()      # (n_samples, input_dim)

        # ── Identify active neurons ───────────────────────────────────────
        # Keep only neurons whose activation frequency is strictly in (0, 1):
        # exclude both fully-sparse (never fires) and fully-dense (always fires) features.
        act_freq = (Z_all > 0).mean(axis=0)
        if checkpoint_active_mask is not None:
            active_mask = np.asarray(checkpoint_active_mask, dtype=bool)
        else:
            active_mask = (act_freq > 0.0) & (act_freq < 1.0)
        if not active_mask.any():
            print(
                "Warning: no neurons had activation frequency strictly between 0 and 1. "
                "Falling back to the top 10% most active neurons."
            )
            mean_act = Z_all.mean(axis=0)
            active_mask = mean_act >= np.percentile(mean_act, 90)

        n_active = int(active_mask.sum())
        print(
            f"SAE: {n_active}/{n_features} neurons active "
            f"(0 < activation frequency < 1, layer={layer})"
        )
        Z_active = Z_all[:, active_mask]       # (n_samples, n_active)

        result = {sid: Z_active[i] for i, sid in enumerate(seq_ids)}

        # ── Persist model + viz data ──────────────────────────────────────
        if self._use_disk():
            sae_dir = self._sae_model_dir()
            sae_dir.mkdir(parents=True, exist_ok=True)

            model_path = self._sae_model_path(
                layer, n_features, sparsity_coeff, embedding_type, sparsity_mode, k, run_label
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "n_features": n_features,
                    "normalize_decoder": normalize_decoder,
                    "sparsity_mode": sparsity_mode,
                    "k": k,
                    "active_mask": active_mask,
                    "params": params,
                },
                model_path,
            )
            print(f"SAE model saved to {model_path}")

            viz_data = {
                "seq_ids": seq_ids,
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "Z_all": Z_all,
                "X_original": X,
                "X_reconstructed": X_recon,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "active_mask": active_mask,
            }
            self._save_pickle(
                viz_data,
                self._sae_viz_path(
                    layer,
                    n_features,
                    sparsity_coeff,
                    embedding_type,
                    sparsity_mode,
                    k,
                    run_label,
                ),
            )

        return result

    #TODO Add a regular autoencoder abstraction as well

    def _delta_sae_abstraction(self, embeddings: dict[str, np.ndarray], method_params: dict | None) -> dict[str, np.ndarray]:
        """
        Encode embeddings with an SAE and return each mutation's SAE activation
        vector relative to the wildtype activation vector.
        """
        params = dict(method_params or {})
        wildtype_key = params.get("wildtype_key")
        if wildtype_key is None and self.input_data.kind == "cellular":
            wildtype_key = self.input_data.wildtype_key
        if wildtype_key is None:
            raise ValueError("DeltaSAE requires a wildtype_key.")
        if wildtype_key not in embeddings:
            raise KeyError(
                f"Wildtype key {wildtype_key!r} is not present in embeddings. "
                "Run embeddings with the class-managed sequence_to_protein_sequence map."
            )

        sae_features = self._sae_abstraction(embeddings, params)
        if wildtype_key not in sae_features:
            raise KeyError(f"Wildtype key {wildtype_key!r} is not present in SAE features.")
        wt_vector = np.asarray(sae_features[wildtype_key])
        return {
            seq_id: np.asarray(feature) - wt_vector
            for seq_id, feature in sae_features.items()
            if seq_id != wildtype_key
        }

    def _spca_abstraction(self, embeddings: dict[str, np.ndarray], method_params: dict | None) -> dict[str, np.ndarray]:
        """
        Perform SPCA abstraction on the embeddings.

        Parameters:
        -----------
        embeddings : dict[str, np.ndarray]
            A dictionary mapping sequence indices to their corresponding embeddings.
        method_params : dict | None
            Parameters for SPCA, such as the number of components and sparsity penalty.

        Returns:
        --------
        dict[str, np.ndarray]
            A dictionary mapping sequence indices to their corresponding SPCA-transformed features.
        """
        raise NotImplementedError("SPCA abstraction is not implemented yet.")


    def visualize_sae_reconstructions(
        self,
        layer: str | int,
        method_params: dict | None = None,
        embedding_type: EmbeddingType | None = None,
        output_path: str | Path | None = None,
        save: bool = True,
    ) -> plt.Figure:
        """
        Visualise SAE reconstruction quality for a trained model.

        Four panels (2x2):
          - Per-sample MSE distribution (train / test)
          - Per-sample cosine similarity distribution (train / test)
          - Reconstruction accuracy: cumulative fraction of samples above each
            cosine-similarity threshold
          - Spread of hidden-neuron activation frequencies across the population

        Parameters
        ----------
        layer : str | int
            Layer for which the SAE was trained.
        method_params : dict | None
            Same params used when calling create_feature_space(method='SAE').
            Must provide n_features / sparsity_coeff so the right model can be
            located (defaults will match defaults used during training).
        output_path : str | Path | None
            Override where the figure is saved.  When None the figure is saved
            to the SAE model directory (disk mode) and / or shown interactively
            (local mode), matching the local_or_disk setting.
        save : bool
            When False, return the figure without saving it to disk.
        """
        params = dict(method_params or {})
        embedding_type = self._embedding_type(embedding_type or params.get("_embedding_type"))
        sparsity_coeff: float = params.get("sparsity_coeff", 1e-3)
        sparsity_mode: str = params.get("sparsity_mode", "normal")
        k: int | None = params.get("k")
        run_label: str | None = params.get("run_label")

        # Resolve n_features: need input_dim to compute the default.
        embeddings = self.load_embeddings(layer, embedding_type)
        if embedding_type == "per_residue":
            embeddings, _ = self._expand_per_residue_feature_vectors(
                embeddings,
                self.sequence_to_mutation_sites,
            )
        input_dim = next(iter(embeddings.values())).shape[0]
        n_features: int = params.get("n_features", input_dim * 2)

        viz_path = self._sae_viz_path(
            layer, n_features, sparsity_coeff, embedding_type, sparsity_mode, k, run_label
        )
        if not viz_path.is_file():
            raise FileNotFoundError(
                f"No SAE visualisation data found at {viz_path}. "
                "Run create_feature_space(..., method='SAE') first."
            )
        viz = self._load_pickle(viz_path)

        X_orig = viz["X_original"]           # (n, d)
        X_recon = viz["X_reconstructed"]     # (n, d)
        Z_all = viz["Z_all"]                 # (n, n_features)
        train_idx = np.asarray(viz["train_idx"])
        test_idx = np.asarray(viz["test_idx"])
        train_losses = viz["train_losses"]
        test_losses = viz["test_losses"]
        active_mask = viz["active_mask"]

        # ── Per-sample metrics ────────────────────────────────────────────
        mse_per_sample = np.mean((X_orig - X_recon) ** 2, axis=1)

        norm_orig = np.linalg.norm(X_orig, axis=1, keepdims=True).clip(min=1e-8)
        norm_recon = np.linalg.norm(X_recon, axis=1, keepdims=True).clip(min=1e-8)
        cos_sim = np.sum((X_orig / norm_orig) * (X_recon / norm_recon), axis=1)

        # Activation frequency per hidden neuron
        act_freq = (Z_all > 0).mean(axis=0)

        # ── Figure ────────────────────────────────────────────────────────
        sns.set_theme(style="darkgrid")
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        fig.suptitle(
            f"SAE Reconstruction Quality — {self._layer_label(layer)}, "
            f"n_features={n_features}, λ={sparsity_coeff}",
            fontsize=13,
        )

        splits = [("train", train_idx)]
        if len(test_idx):
            splits.append(("test", test_idx))

        # Panel 1: MSE
        ax = axes[0, 0]
        for name, idx in splits:
            ax.hist(mse_per_sample[idx], bins=50, alpha=0.65, density=True,
                    label=f"{name} (n={len(idx)})")
        ax.set_xlabel("MSE")
        ax.set_ylabel("Density")
        ax.set_title("Per-sample reconstruction MSE")
        ax.legend()

        # Panel 2: Cosine similarity
        ax = axes[0, 1]
        for name, idx in splits:
            ax.hist(cos_sim[idx], bins=50, alpha=0.65, density=True, label=name)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        ax.set_title("Per-sample cosine similarity (original vs reconstructed)")
        ax.legend()

        # Panel 3: Reconstruction accuracy (cumulative cosine-sim curve)
        ax = axes[1, 0]
        thresholds = np.linspace(0.0, 1.0, 300)
        for name, idx in splits:
            frac_above = np.array([(cos_sim[idx] >= t).mean() for t in thresholds])
            ax.plot(thresholds, frac_above, label=name)
        ax.set_xlabel("Cosine similarity threshold")
        ax.set_ylabel("Fraction of samples above threshold")
        ax.set_title("Reconstruction accuracy (cumulative)")
        ax.legend()

        # Panel 4: Activation frequency spread
        ax = axes[1, 1]
        ax.hist(act_freq, bins=50, color="steelblue", alpha=0.85)
        threshold_val = params.get("activity_threshold", 1e-3)
        n_active = int(active_mask.sum())
        ax.axvline(
            threshold_val,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"threshold → {n_active}/{len(active_mask)} active",
        )
        ax.set_xlabel("Activation frequency")
        ax.set_ylabel("Number of hidden neurons")
        ax.set_title("Hidden-neuron activation frequency spread")
        ax.set_yscale("log")
        ax.legend()

        fig.tight_layout()

        # ── Save / show ───────────────────────────────────────────────────
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
        elif save:
            if self._use_disk():
                default_path = (
                    self._sae_model_dir()
                    / f"{self._sae_tag(layer, n_features, sparsity_coeff, embedding_type, sparsity_mode, k, run_label)}_viz.png"
                )
                self._sae_model_dir().mkdir(parents=True, exist_ok=True)
                fig.savefig(default_path, bbox_inches="tight", dpi=150)
                print(f"SAE visualization saved to {default_path}")

        return fig


    def _load_abstracted_features(self, layer: str, 
                                    method: AbstractionMethod | str,
                                    embedding_type: EmbeddingType | None,
                                    method_params: dict | None) -> dict[str, np.ndarray]:
        """
        Load abstracted features from disk if they exist.

        Parameters:
        -----------
        layer : str
            The layer from which to extract features.
        method : Literal['none', 'PCA', 'SAE', 'DeltaSAE', 'SPCA']
            The method used for abstraction.
        method_params : dict | None
            Parameters for the abstraction method.

        Returns:
        --------
        dict[str, np.ndarray] | None
            A dictionary mapping sequence indices to their corresponding abstracted features, or None if no saved features are found.
        """
        params = dict(method_params or {})
        force_recompute = bool(params.pop("force_recompute", False))
        embedding_type = self._embedding_type(embedding_type)
        method = self._abstraction_type(method)
        if method == 'none':
            #print("No abstraction method specified. Using raw embeddings as features.")
            return self.load_embeddings(layer, embedding_type)

        key = self._feature_key(method, layer, embedding_type)
        if not force_recompute and self._use_memory() and self.sequence_to_features.get(key) is not None:
            #print("Abstracted features already exist in memory. Returning existing features.")
            return self.sequence_to_features[key]

        if not force_recompute and self._use_disk():
            save_path = self._feature_path(method, layer, embedding_type)
            if save_path.is_file():
                #print(f"Loading abstracted features from {save_path}")
                abstracted_features = self._load_pickle(save_path)
                if self._use_memory():
                    self.sequence_to_features[key] = abstracted_features
                return abstracted_features

        return self.create_feature_space(layer, method, params, embedding_type, force_recompute)

    @staticmethod
    def _normalize_features(features, norm_scheme):
        if norm_scheme == "cross_feature":
            return (features - np.mean(features)) / (np.std(features) + 1e-8)
        if norm_scheme == "per_feature":
            return (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        if norm_scheme == "none":
            return features
        raise ValueError(f"Unsupported normalization scheme: {norm_scheme}")

    def _feature_input_path(
        self,
        abstraction_method: AbstractionMethod | str,
        layer: str | int,
        embedding_type: EmbeddingType | None = None,
    ) -> Path:
        return self._feature_path(abstraction_method, layer, embedding_type)

    def create_inference_job(
        self,
        layer: str | int,
        abstraction_method: AbstractionMethod = 'none',
        abstraction_params: dict | None = None,
        embedding_type: EmbeddingType | None = None,
        job_dir: str | Path | None = None,
        job_name: str = "esm_infer",
        partition: str = "dept_cpu",
        cpus_per_task: int = 4,
        mem: str = "16G",
        time: str = "06:00:00",
        python_executable: str = "python3",
        scratch_root: str | Path = "/scr",
        submit: bool = False,
    ) -> dict[str, Path | str]:
        """
        Create a single Slurm job that runs feature inference for one layer.

        The worker reads saved features from disk, writes inference results into
        scratch, then copies the result to the standard inference cache path.
        """
        if not self._use_disk():
            raise ValueError("Inference jobs require local_or_disk to be 'disk' or 'both'.")
        if self.sequence_dataframe is None:
            raise ValueError("Run process_raw_data() before creating an inference job.")

        abstraction_params = abstraction_params or {}
        embedding_type = self._embedding_type(embedding_type)
        abstraction_method = self._abstraction_type(abstraction_method)
        norm_scheme = abstraction_params.get("norm_scheme", "none")
        if abstraction_method == "none" and embedding_type == "mutation_pooled":
            feature_path = self._embedding_path(layer, "per_residue")
            if not feature_path.is_file():
                self.load_embeddings(layer, embedding_type)
        else:
            feature_path = self._feature_input_path(abstraction_method, layer, embedding_type)
            if not feature_path.is_file():
                self.create_feature_space(layer, abstraction_method, abstraction_params, embedding_type)

        output_path = self._inference_path(abstraction_method, layer, norm_scheme, embedding_type)
        inference_job_dir = self._inference_job_dir(layer, abstraction_method, norm_scheme, embedding_type, job_dir)
        logs_dir = inference_job_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        payload = {
            "sequence_dataframe": self.sequence_dataframe,
            "feature_path": str(feature_path),
            "output_path": str(output_path),
            "embedding_type": embedding_type,
            "abstraction_method": abstraction_method,
            "norm_scheme": norm_scheme,
            "sequence_to_mutation_sites": self.sequence_to_mutation_sites,
        }
        payload_path = self._inference_payload_path(inference_job_dir)
        self._save_pickle(payload, payload_path)

        script_path = inference_job_dir / "submit_inference_job.sh"
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output={logs_dir}/slurm-%j.out
#SBATCH --error={logs_dir}/slurm-%j.err

set -euo pipefail
cd {Path.cwd()}

SCRDIR={scratch_root}/${{SLURM_JOB_ID}}
mkdir -p "$SCRDIR"
export TMPDIR="$SCRDIR"

{python_executable} -c "import sys, os; sys.path.insert(0, r'{Path.cwd()}'); import popDMS; from esmDMS import esmDMS; esmDMS.run_inference_job(r'{payload_path}', scratch_dir=os.environ['TMPDIR'])"
"""
        script_path.write_text(script)
        script_path.chmod(0o755)

        job_id = ""
        if submit:
            completed = subprocess.run(
                ["sbatch", str(script_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            job_id = completed.stdout.strip()

        return {
            "inference_job_dir": inference_job_dir,
            "payload_path": payload_path,
            "script_path": script_path,
            "output_path": output_path,
            "job_id": job_id,
        }

    def create_sae_sweep_job(
        self,
        layer: str | int,
        sweep_params: list[dict],
        method: AbstractionMethod | str = "DeltaSAE",
        embedding_type: EmbeddingType | None = None,
        job_dir: str | Path | None = None,
        job_name: str = "sae_sweep",
        partition: str = "any_gpu",
        gpus: int = 1,
        gres: str | None = None,
        constraint: str | None = None,
        cpus_per_task: int | None = None,
        mem: str = "16G",
        time: str = "06:00:00",
        python_executable: str = "python3",
        scratch_root: str | Path = "/scr",
        max_parallel_runs: int | None = None,
        run_inference: bool = True,
        force_recompute: bool = False,
        require_cuda: bool = True,
        submit: bool = False,
    ) -> dict[str, Path | str]:
        """
        Create one Slurm job that sweeps SAE/DeltaSAE hyperparameters on GPUs.

        The job requests up to ``gpus`` GPUs in a single allocation and runs one
        SAE training worker per visible GPU. Each worker consumes sweep configs
        from a shared queue, so the allocation is filled without submitting a
        separate Slurm job for every hyperparameter setting.
        """
        if not self._use_disk():
            raise ValueError("SAE sweep jobs require local_or_disk to be 'disk' or 'both'.")
        if not sweep_params:
            raise ValueError("sweep_params must contain at least one parameter dictionary.")
        if gpus < 1:
            raise ValueError("gpus must be at least 1.")
        if max_parallel_runs is not None and max_parallel_runs < 1:
            raise ValueError("max_parallel_runs must be at least 1 when specified.")
        method = self._abstraction_type(method)
        if self._abstraction_base_method(method) not in {"SAE", "DeltaSAE"}:
            raise ValueError("create_sae_sweep_job only supports SAE and DeltaSAE methods.")
        embedding_type = self._embedding_type(embedding_type)
        if run_inference and self.sequence_dataframe is None:
            raise ValueError("Run process_raw_data() before creating an SAE sweep job with run_inference=True.")

        embedding_path = self._embedding_path(layer, embedding_type)
        if not embedding_path.is_file():
            raise FileNotFoundError(
                f"No saved {embedding_type} embeddings found at {embedding_path}. "
                f"Run embed_all_sequences({layer!r}) first."
            )

        sweep_dir = self._sae_sweep_job_dir(layer, method, embedding_type, job_dir)
        logs_dir = sweep_dir / "logs"
        output_root = sweep_dir / "runs"
        logs_dir.mkdir(exist_ok=True)
        output_root.mkdir(exist_ok=True)

        configs = []
        seen_labels = set()
        for idx, params in enumerate(sweep_params):
            run_params = dict(params)
            run_label = self._sae_sweep_run_label(idx, run_params)
            if run_label in seen_labels:
                raise ValueError(f"Duplicate SAE sweep run_label after sanitization: {run_label!r}")
            seen_labels.add(run_label)
            run_params["run_label"] = run_label
            configs.append({"run_label": run_label, "params": run_params})

        cpus_per_task = cpus_per_task if cpus_per_task is not None else max(4, gpus * 2)
        if gres is None:
            gres = f"gpu:{gpus}"

        payload = {
            "input_data": self.input_data,
            "config": {
                "embedding_model": self.config.embedding_model,
                "embedding_type": embedding_type,
                "embedding_method": None,
                "local_or_disk": "both",
                "save_dir": str(self._save_dir()),
                "dataset_name": self.config.dataset_name,
            },
            "sequence_dataframe": self.sequence_dataframe,
            "sequence_to_mutation_sites": self.sequence_to_mutation_sites,
            "sequence_to_protein_sequence": self.sequence_to_protein_sequence,
            "sequence_metadata": self.sequence_metadata,
            "scores_dataframe": self.scores_dataframe,
            "layer": layer,
            "method": method,
            "embedding_type": embedding_type,
            "embedding_path": str(embedding_path),
            "output_root": str(output_root),
            "sweep_dir": str(sweep_dir),
            "configs": configs,
            "gpus": gpus,
            "max_parallel_runs": max_parallel_runs,
            "run_inference": run_inference,
            "force_recompute": force_recompute,
            "require_cuda": require_cuda,
        }
        payload_path = sweep_dir / f"{self._dataset_prefix()}sae_sweep_payload.pkl"
        self._save_pickle(payload, payload_path)

        runner_path = sweep_dir / "run_sae_sweep.py"
        runner_script = f"""from pathlib import Path
import os
import sys

sys.path.insert(0, r"{Path.cwd()}")

import popDMS  # noqa: F401
from esmDMS import esmDMS


if __name__ == "__main__":
    esmDMS.run_sae_sweep_job(
        Path(r"{payload_path}"),
        scratch_dir=os.environ.get("TMPDIR"),
    )
"""
        runner_path.write_text(runner_script)
        runner_path.chmod(0o755)

        script_path = sweep_dir / "submit_sae_sweep.sh"
        gres_line = f"#SBATCH --gres={gres}\n" if gres else ""
        constraint_line = f"#SBATCH --constraint={constraint}\n" if constraint else ""
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
{gres_line}{constraint_line}#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output={logs_dir}/slurm-%j.out
#SBATCH --error={logs_dir}/slurm-%j.err

set -euo pipefail
cd {Path.cwd()}

SCRDIR={scratch_root}/${{SLURM_JOB_ID}}_sae_sweep
mkdir -p "$SCRDIR"
export TMPDIR="$SCRDIR"
export MPLCONFIGDIR="$SCRDIR/mplconfig"
mkdir -p "$MPLCONFIGDIR"

{python_executable} {runner_path}
"""
        script_path.write_text(script)
        script_path.chmod(0o755)

        job_id = ""
        if submit:
            completed = subprocess.run(
                ["sbatch", str(script_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            job_id = completed.stdout.strip()

        return {
            "sweep_dir": sweep_dir,
            "payload_path": payload_path,
            "runner_path": runner_path,
            "script_path": script_path,
            "output_root": output_root,
            "job_id": job_id,
        }

    @staticmethod
    def _run_sae_sweep_config_safe(
        payload: dict,
        config_idx: int,
        gpu_idx: int | None,
        scratch_dir: str | Path | None = None,
    ) -> dict:
        try:
            return esmDMS._run_sae_sweep_config(payload, config_idx, gpu_idx, scratch_dir)
        except Exception as exc:
            config = payload["configs"][config_idx]
            return {
                "status": "failed",
                "config_idx": config_idx,
                "run_label": config.get("run_label"),
                "gpu_idx": gpu_idx,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    @staticmethod
    def _run_sae_sweep_config(
        payload: dict,
        config_idx: int,
        gpu_idx: int | None,
        scratch_dir: str | Path | None = None,
    ) -> dict:
        if gpu_idx is not None and torch.cuda.is_available():
            torch.cuda.set_device(gpu_idx)

        started = time.time()
        config_entry = payload["configs"][config_idx]
        run_label = config_entry["run_label"]
        params = dict(config_entry["params"])
        params["run_label"] = run_label
        layer = payload["layer"]
        method = payload["method"]
        embedding_type = payload["embedding_type"]
        norm_scheme = params.get("norm_scheme", "none")

        output_root = Path(payload["output_root"])
        final_run_dir = output_root / run_label
        if scratch_dir is None:
            work_run_dir = final_run_dir
        else:
            work_run_dir = Path(scratch_dir) / "sae_sweep_runs" / run_label
        work_run_dir.mkdir(parents=True, exist_ok=True)
        final_run_dir.mkdir(parents=True, exist_ok=True)

        base_config = dict(payload["config"])
        work_config = dict(base_config)
        work_config["save_dir"] = str(work_run_dir)
        final_config = dict(base_config)
        final_config["save_dir"] = str(final_run_dir)

        runner = esmDMS(payload["input_data"], ESMDMSConfig(**work_config))
        final_runner = esmDMS(payload["input_data"], ESMDMSConfig(**final_config))
        for target in (runner, final_runner):
            target.sequence_dataframe = payload.get("sequence_dataframe")
            target.sequence_to_mutation_sites = payload.get("sequence_to_mutation_sites")
            target.sequence_to_protein_sequence = payload.get("sequence_to_protein_sequence")
            target.sequence_metadata = payload.get("sequence_metadata")
            target.scores_dataframe = payload.get("scores_dataframe")

        final_feature_path = final_runner._feature_path(method, layer, embedding_type)
        final_inference_path = final_runner._inference_path(method, layer, norm_scheme, embedding_type)
        feature_status = "trained"
        force_recompute = bool(payload.get("force_recompute", False))

        if final_feature_path.is_file() and not force_recompute:
            abstracted_features = final_runner._load_pickle(final_feature_path)
            feature_status = "cached"
        else:
            embeddings = esmDMS._load_pickle(Path(payload["embedding_path"]))
            _, embeddings = esmDMS._drop_missing_features(None, embeddings, f"{method} sweep abstraction")
            if embedding_type == "per_residue":
                embeddings, _ = esmDMS._expand_per_residue_feature_vectors(
                    embeddings,
                    payload.get("sequence_to_mutation_sites"),
                )
            esmDMS._require_vector_features(embeddings, f"{method} sweep abstraction")
            params.setdefault("_layer", layer)
            params.setdefault("_embedding_type", embedding_type)
            abstracted_features = runner._create_feature_space(embeddings, method, params)
            runner._save_pickle(abstracted_features, runner._feature_path(method, layer, embedding_type))

            if work_run_dir != final_run_dir:
                shutil.copytree(work_run_dir, final_run_dir, dirs_exist_ok=True)

        inference_status = "not_requested"
        if payload.get("run_inference", True):
            if final_inference_path.is_file() and not force_recompute:
                inference_status = "cached"
            else:
                if force_recompute and final_inference_path.is_file():
                    final_inference_path.unlink()
                inference_runner = final_runner if feature_status == "cached" else runner
                inference_runner.run_feature_inference(
                    layer=layer,
                    abstraction_method=method,
                    abstraction_params=params,
                    embedding_type=embedding_type,
                )
                if work_run_dir != final_run_dir and inference_runner is runner:
                    shutil.copytree(work_run_dir, final_run_dir, dirs_exist_ok=True)
                inference_status = "ran"

        feature_dim = 0
        if abstracted_features:
            first_feature = next(iter(abstracted_features.values()))
            feature_dim = int(np.asarray(first_feature).shape[0])

        model_path = final_runner._sae_model_path(
            layer,
            params.get("n_features", "unknown"),
            params.get("sparsity_coeff", 1e-3),
            embedding_type,
            params.get("sparsity_mode", "normal"),
            params.get("k"),
            run_label,
        )
        viz_path = final_runner._sae_viz_path(
            layer,
            params.get("n_features", "unknown"),
            params.get("sparsity_coeff", 1e-3),
            embedding_type,
            params.get("sparsity_mode", "normal"),
            params.get("k"),
            run_label,
        )

        return {
            "status": "ok",
            "config_idx": config_idx,
            "run_label": run_label,
            "gpu_idx": gpu_idx,
            "feature_status": feature_status,
            "inference_status": inference_status,
            "elapsed_seconds": round(time.time() - started, 3),
            "method": method,
            "embedding_type": embedding_type,
            "layer": esmDMS._layer_label(layer),
            "n_features": params.get("n_features"),
            "sparsity_mode": params.get("sparsity_mode", "normal"),
            "k": params.get("k"),
            "sparsity_coeff": params.get("sparsity_coeff", 1e-3),
            "lr": params.get("lr", 1e-3),
            "epochs": params.get("epochs", 200),
            "batch_size": params.get("batch_size", 64),
            "seed": params.get("seed", 42),
            "feature_count": len(abstracted_features),
            "feature_dim": feature_dim,
            "run_dir": str(final_run_dir),
            "feature_path": str(final_feature_path),
            "inference_path": str(final_inference_path) if payload.get("run_inference", True) else "",
            "model_path": str(model_path),
            "viz_path": str(viz_path),
        }

    @staticmethod
    def _sae_sweep_gpu_worker(
        payload: dict,
        task_queue,
        result_queue,
        gpu_idx: int | None,
        scratch_dir: str | Path | None,
    ) -> None:
        while True:
            config_idx = task_queue.get()
            if config_idx is None:
                return
            result_queue.put(
                esmDMS._run_sae_sweep_config_safe(payload, config_idx, gpu_idx, scratch_dir)
            )

    @staticmethod
    def run_sae_sweep_job(
        payload_path: str | Path,
        scratch_dir: str | Path | None = None,
    ) -> list[dict]:
        """
        Worker entrypoint used by create_sae_sweep_job().
        """
        payload_path = Path(payload_path)
        payload = esmDMS._load_pickle(payload_path)
        configs = payload["configs"]
        n_configs = len(configs)
        requested_gpus = int(payload.get("gpus", 1))
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if payload.get("require_cuda", False) and available_gpus < 1:
            raise RuntimeError(
                "SAE sweep was configured with require_cuda=True, but no CUDA GPU is visible."
            )
        max_parallel = payload.get("max_parallel_runs")
        if max_parallel is None:
            max_parallel = requested_gpus if available_gpus else 1
        max_parallel = int(max_parallel)
        n_workers = min(n_configs, max_parallel)
        if available_gpus:
            n_workers = min(n_workers, available_gpus, requested_gpus)
            gpu_ids = list(range(n_workers))
        else:
            n_workers = 1
            gpu_ids = [None]

        print(
            f"Starting SAE sweep with {n_configs} config(s), "
            f"{available_gpus} visible CUDA GPU(s), {n_workers} worker(s)."
        )

        if n_workers == 1:
            results = [
                esmDMS._run_sae_sweep_config_safe(payload, idx, gpu_ids[0], scratch_dir)
                for idx in range(n_configs)
            ]
        else:
            ctx = mp.get_context("spawn")
            task_queue = ctx.Queue()
            result_queue = ctx.Queue()
            for idx in range(n_configs):
                task_queue.put(idx)
            for _ in range(n_workers):
                task_queue.put(None)

            processes = [
                ctx.Process(
                    target=esmDMS._sae_sweep_gpu_worker,
                    args=(payload, task_queue, result_queue, gpu_ids[worker_idx], scratch_dir),
                )
                for worker_idx in range(n_workers)
            ]
            for proc in processes:
                proc.start()

            results = []
            while len(results) < n_configs:
                try:
                    results.append(result_queue.get(timeout=5))
                except queue.Empty:
                    if all(not proc.is_alive() for proc in processes):
                        break

            for proc in processes:
                proc.join()
            if len(results) != n_configs:
                raise RuntimeError(
                    f"SAE sweep workers exited after returning {len(results)}/{n_configs} result(s)."
                )

        results = sorted(results, key=lambda row: row.get("config_idx", -1))
        sweep_dir = Path(payload["sweep_dir"])
        summary_path = sweep_dir / "sae_sweep_results.csv"
        json_path = sweep_dir / "sae_sweep_results.json"
        pd.DataFrame(results).to_csv(summary_path, index=False)
        with json_path.open("w") as f:
            json.dump(results, f, indent=2)

        failed = [row for row in results if row.get("status") != "ok"]
        if failed:
            for row in failed:
                print(f"SAE sweep failed for {row.get('run_label')}: {row.get('error')}")
                print(row.get("traceback", ""))
            raise RuntimeError(f"{len(failed)} SAE sweep run(s) failed. See {json_path}.")

        print(f"SAE sweep complete. Summary written to {summary_path}")
        return results

    @staticmethod
    def run_inference_job(
        payload_path: str | Path,
        scratch_dir: str | Path | None = None,
    ) -> Path:
        """
        Worker entrypoint used by create_inference_job().
        """
        payload_path = Path(payload_path)
        with payload_path.open("rb") as f:
            payload = pickle.load(f)

        sequence_dataframe = payload["sequence_dataframe"]
        feature_path = Path(payload["feature_path"])
        output_path = Path(payload["output_path"])
        embedding_type = payload.get("embedding_type")
        abstraction_method = payload.get("abstraction_method", "none")
        norm_scheme = payload["norm_scheme"]

        seq_to_features = esmDMS._load_pickle(feature_path)
        if abstraction_method == "none" and embedding_type == "mutation_pooled":
            seq_to_features = esmDMS._pool_per_residue_features(seq_to_features)
        if embedding_type == "per_residue":
            sequence_dataframe, seq_to_features = esmDMS._expand_per_residue_features_for_inference(
                sequence_dataframe,
                seq_to_features,
                payload.get("sequence_to_mutation_sites"),
            )
        sequence_dataframe, seq_to_features = esmDMS._drop_missing_features(
            sequence_dataframe,
            seq_to_features,
            "Inference",
        )
        esmDMS._require_vector_features(seq_to_features, "Inference")
        if norm_scheme is not None and norm_scheme != "none":
            seq_ids = list(seq_to_features)
            features = np.asarray([seq_to_features[seq_id] for seq_id in seq_ids])
            features = esmDMS._normalize_features(features, norm_scheme)
            seq_to_features = dict(zip(seq_ids, features))

        result = mini_infer_esm(sequence_dataframe, seq_to_features)
        if scratch_dir is None:
            scratch_path = output_path
        else:
            scratch_path = Path(scratch_dir) / "esm_inference_saves" / output_path.name
            scratch_path.parent.mkdir(parents=True, exist_ok=True)

        esmDMS._save_pickle(result, scratch_path)
        if scratch_path != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(scratch_path, output_path)
        return output_path

    #TODO: add parameters for model type (linear, non-linear) and regularization scheme (L2, L1, ElasticNet)
    def run_feature_inference(self, layer: str,
                              abstraction_method: AbstractionMethod = 'none',
                              abstraction_params: dict | None = None,
                              embedding_type: EmbeddingType | None = None) -> InferenceResult: 
        """
        Run the abstracted features through the popDMS framework to calculate selection coefficients and fitness.

        Parameters:
        -----------
        layer : str | None
            The layer from which to extract features.
        abstraction_method : Literal['none', 'PCA', 'SAE', 'SPCA']
            The method used for feature abstraction.
        abstraction_params : dict | None
            Parameters for the abstraction method.

        Returns:
        --------
        InferenceResult
            An InferenceResult object containing inferred selection coefficients and fitness values.
            class InferenceResult:
            dx: list
            icov: list
            s: np.ndarray
            s_joint: np.ndarray
            gamma_opt: float
            x_array: list
            error_bars: np.ndarray
            s_joint_error_bars: np.ndarray
        """

        if self.sequence_dataframe is None:
            raise ValueError("Sequence dataframe is not available. Please run process_raw_data() first.")

        abstraction_params = abstraction_params or {}
        embedding_type = self._embedding_type(embedding_type)
        abstraction_method = self._abstraction_type(abstraction_method)
        norm_scheme = abstraction_params.get('norm_scheme', 'none')
        if self._use_disk():
            # check if the features are already saved to disk
            save_path = self._inference_path(abstraction_method, layer, norm_scheme, embedding_type)
            if save_path.is_file():
                #print(f"Loading inference results from {save_path}")
                return self._load_pickle(save_path)
            else:
                #print(f"No saved inference results found at {save_path}. Running inference and saving results.")
                pass
        
        # load features, seq_to_features type = dict[str, np.ndarray]
        seq_to_features = self._load_abstracted_features(layer, abstraction_method, embedding_type, abstraction_params)
        if embedding_type == "per_residue":
            sequence_dataframe, seq_to_features = self._expand_per_residue_features_for_inference(
                self.sequence_dataframe,
                seq_to_features,
                self.sequence_to_mutation_sites,
            )
        else:
            sequence_dataframe = self.sequence_dataframe
        sequence_dataframe, seq_to_features = self._drop_missing_features(
            sequence_dataframe,
            seq_to_features,
            "Inference",
        )
        self._require_vector_features(seq_to_features, "Inference")
        if norm_scheme is not None and norm_scheme != "none":
            seq_ids = list(seq_to_features)
            features = np.asarray([seq_to_features[seq_id] for seq_id in seq_ids])
            features = self._normalize_features(features, norm_scheme)
            seq_to_features = dict(zip(seq_ids, features))

        inf_result = mini_infer_esm(sequence_dataframe, seq_to_features)
        if self._use_disk():
            self._save_inference_results(inf_result, layer, 
                                         abstraction_method, norm_scheme, embedding_type)
        if self._use_memory():
            self.inference_results[self._inference_key(layer, abstraction_method, norm_scheme, embedding_type)] = inf_result
        return inf_result


    def _save_inference_results(
        self,
        results: dict,
        layer: str,
        abstraction_method: str,
        norm_scheme: str,
        embedding_type: str | None = None,
    ) -> None:
        if self.config.save_dir is None:
            raise ValueError("save_dir must be specified in the configuration to save inference results to disk.")
        
        save_path = self._inference_path(abstraction_method, layer, norm_scheme, embedding_type)
        self._save_pickle(results, save_path)

    def load_inference_results(
        self,
        layer: str | int,
        abstraction_method: AbstractionMethod = 'none',
        norm_scheme: str = "none",
        embedding_type: EmbeddingType | None = None,
    ) -> InferenceResult:
        """
        Load inference results from memory or disk for a layer/method/norm tuple.
        """
        embedding_type = self._embedding_type(embedding_type)
        abstraction_method = self._abstraction_type(abstraction_method)
        key = self._inference_key(str(layer), abstraction_method, norm_scheme, embedding_type)
        if self._use_memory() and key in self.inference_results:
            return self.inference_results[key]

        if self._use_disk():
            path = self._inference_path(abstraction_method, layer, norm_scheme, embedding_type)
            if path.is_file():
                result = self._load_pickle(path)
                if self._use_memory():
                    self.inference_results[key] = result
                return result

        raise FileNotFoundError(
            f"No inference results found for layer={layer}, "
            f"embedding_type={embedding_type}, abstraction_method={abstraction_method}, norm_scheme={norm_scheme}."
        )

    def fitness_dataframe(
        self,
        layer: str | int,
        abstraction_method: AbstractionMethod | str,
        abstraction_params: dict | None = None,
        embedding_type: EmbeddingType | None = None,
        norm_scheme: str = "none",
        use_joint: bool = True,
        baseline: float = 1.0,
    ) -> pd.DataFrame:
        """
        Return inferred fitness for each mutation key.

        Fitness is calculated as baseline + selection_coefficients dot features.
        For DeltaSAE, the features are already mutant SAE activations minus the
        wildtype SAE activation vector.
        """
        result = self.load_inference_results(layer, abstraction_method, norm_scheme, embedding_type)
        seq_ids, features = self._features_for_inference(
            layer,
            abstraction_method,
            embedding_type,
            norm_scheme,
            abstraction_params,
        )

        rows = []
        if use_joint and getattr(result, "s_joint", None) is not None:
            fitness = baseline + features @ result.s_joint
            rows.extend({
                "SequenceIndex": seq_id,
                "fitness": value,
                "replicate": "joint",
            } for seq_id, value in zip(seq_ids, fitness))
        else:
            for rep_idx in range(result.s.shape[0]):
                fitness = baseline + features @ result.s[rep_idx]
                rows.extend({
                    "SequenceIndex": seq_id,
                    "fitness": value,
                    "replicate": rep_idx + 1,
                } for seq_id, value in zip(seq_ids, fitness))
        return pd.DataFrame(rows)

    def plot_functional_score_comparison(
        self,
        layer: str | int,
        abstraction_method: AbstractionMethod | str,
        abstraction_params: dict | None = None,
        embedding_type: EmbeddingType | None = None,
        norm_scheme: str = "none",
        score_col: str = "score",
        output_path: str | Path | None = None,
    ) -> tuple[plt.Figure, pd.DataFrame, dict]:
        """
        Compare inferred fitness against averaged MaveDB functional scores.
        """
        if self.scores_dataframe is None:
            self.load_functional_scores()
        if score_col not in self.scores_dataframe.columns:
            raise ValueError(f"Score column {score_col!r} is not present in scores_dataframe.")

        fitness_df = self.fitness_dataframe(
            layer,
            abstraction_method,
            abstraction_params,
            embedding_type,
            norm_scheme,
            use_joint=True,
            baseline=1.0,
        )
        score_df = self.scores_dataframe[["SequenceIndex", score_col]].copy()
        score_df = score_df.rename(columns={score_col: "functional_score"})
        comparison_df = fitness_df[fitness_df["replicate"] == "joint"].merge(
            score_df,
            on="SequenceIndex",
            how="inner",
        )
        comparison_df = comparison_df[
            np.isfinite(comparison_df["fitness"]) & np.isfinite(comparison_df["functional_score"])
        ].copy()
        rho = self._safe_corr(comparison_df["fitness"], comparison_df["functional_score"], spearmanr)

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(5.6, 5.0))
        sns.scatterplot(
            data=comparison_df,
            x="functional_score",
            y="fitness",
            s=18,
            alpha=0.6,
            edgecolor=None,
            ax=ax,
        )
        ax.set_xlabel(f"MaveDB functional score ({score_col})")
        ax.set_ylabel("Inferred fitness")
        ax.set_title(
            f"{self._abstraction_type(abstraction_method)} {self._embedding_type(embedding_type)} "
            f"{self._layer_label(layer)}\nSpearman rho={rho:.3f}, n={len(comparison_df)}"
        )
        fig.tight_layout()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight", dpi=150)

        stats = {
            "layer": layer,
            "embedding_type": self._embedding_type(embedding_type),
            "abstraction_method": self._abstraction_type(abstraction_method),
            "score_col": score_col,
            "spearman_rho": rho,
            "n": len(comparison_df),
        }
        return fig, comparison_df, stats

    def plot_regularization_curve(
        self,
        layer: str | int,
        abstraction_method: AbstractionMethod | str,
        abstraction_params: dict | None = None,
        embedding_type: EmbeddingType | None = None,
        norm_scheme: str = "none",
        gamma_values: np.ndarray | None = None,
        output_path: str | Path | None = None,
    ) -> tuple[plt.Figure, pd.DataFrame]:
        """
        Sweep gamma and plot replicate selection-coefficient consistency.
        """
        if self.sequence_dataframe is None:
            raise ValueError("Sequence dataframe is not available. Please run process_raw_data() first.")
        seq_to_features = self._load_abstracted_features(
            layer,
            abstraction_method,
            embedding_type,
            abstraction_params,
        )
        sequence_dataframe = self.sequence_dataframe
        sequence_dataframe, seq_to_features = self._drop_missing_features(
            sequence_dataframe,
            seq_to_features,
            "Regularization sweep",
        )
        self._require_vector_features(seq_to_features, "Regularization sweep")
        if norm_scheme is not None and norm_scheme != "none":
            seq_ids = list(seq_to_features)
            features = np.asarray([seq_to_features[seq_id] for seq_id in seq_ids])
            features = self._normalize_features(features, norm_scheme)
            seq_to_features = dict(zip(seq_ids, features))

        gamma_values, s_by_gamma, _ = infer_gamma_range(
            sequence_dataframe,
            seq_to_features,
            gamma_values=gamma_values,
        )
        rows = []
        for gamma_idx, gamma in enumerate(gamma_values):
            pair_corrs = []
            for rep_i, rep_j in self._rep_pairs(s_by_gamma.shape[1]):
                rho = self._safe_corr(s_by_gamma[gamma_idx, rep_i], s_by_gamma[gamma_idx, rep_j], pearsonr)
                pair_corrs.append(rho)
                rows.append({
                    "gamma": gamma,
                    "rep_i": rep_i + 1,
                    "rep_j": rep_j + 1,
                    "pearson_r": rho,
                })
            rows.append({
                "gamma": gamma,
                "rep_i": "mean",
                "rep_j": "mean",
                "pearson_r": np.nanmean(pair_corrs) if pair_corrs else np.nan,
            })
        reg_df = pd.DataFrame(rows)

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        mean_df = reg_df[reg_df["rep_i"] == "mean"]
        ax.plot(mean_df["gamma"], mean_df["pearson_r"], marker="o", label="Mean pairwise r")
        pair_df = reg_df[reg_df["rep_i"] != "mean"]
        for (rep_i, rep_j), pair_rows in pair_df.groupby(["rep_i", "rep_j"]):
            ax.plot(pair_rows["gamma"], pair_rows["pearson_r"], alpha=0.35, linewidth=1, label=f"Rep {rep_i} vs {rep_j}")
        ax.set_xscale("log")
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Regularization strength (gamma)")
        ax.set_ylabel("Selection coefficient Pearson r")
        ax.set_title(f"Regularization sweep, {self._layer_label(layer)}")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
        return fig, reg_df

    def plot_selection_coefficient_distribution(
        self,
        layer: str | int,
        abstraction_method: AbstractionMethod | str,
        embedding_type: EmbeddingType | None = None,
        norm_scheme: str = "none",
        joint: bool = True,
        output_path: str | Path | None = None,
    ) -> tuple[plt.Figure, pd.DataFrame]:
        """
        Plot the distribution of inferred selection coefficients.
        """
        result = self.load_inference_results(layer, abstraction_method, norm_scheme, embedding_type)
        rows = []
        if joint and getattr(result, "s_joint", None) is not None:
            rows = [{"coefficient": value, "replicate": "joint"} for value in result.s_joint]
        else:
            for rep_idx in range(result.s.shape[0]):
                rows.extend({
                    "coefficient": value,
                    "replicate": rep_idx + 1,
                } for value in result.s[rep_idx])
        coef_df = pd.DataFrame(rows)

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        sns.histplot(data=coef_df, x="coefficient", hue="replicate", bins=60, kde=True, ax=ax)
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xlabel("Selection coefficient")
        ax.set_ylabel("Feature count")
        ax.set_title(f"Selection coefficient distribution, {self._layer_label(layer)}")
        fig.tight_layout()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
        return fig, coef_df

    @staticmethod
    def _safe_corr(x, y, corr_fn):
        x = np.asarray(x)
        y = np.asarray(y)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return np.nan
        x = x[mask]
        y = y[mask]
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return corr_fn(x, y).statistic

    @staticmethod
    def _rep_pairs(n_reps: int) -> list[tuple[int, int]]:
        return [(i, j) for i in range(n_reps) for j in range(i + 1, n_reps)]

    def _features_for_inference(
        self,
        layer: str | int,
        abstraction_method: AbstractionMethod | str,
        embedding_type: EmbeddingType | None,
        norm_scheme: str,
        abstraction_params: dict | None = None,
    ) -> tuple[list, np.ndarray]:
        method_params = dict(abstraction_params or {})
        method_params.setdefault("norm_scheme", norm_scheme)
        embedding_type = self._embedding_type(embedding_type)
        abstraction_method = self._abstraction_type(abstraction_method)
        seq_to_features = self._load_abstracted_features(layer, abstraction_method, embedding_type, method_params)
        if embedding_type == "per_residue":
            _, seq_to_features = self._expand_per_residue_features_for_inference(
                self.sequence_dataframe,
                seq_to_features,
                self.sequence_to_mutation_sites,
            )
        _, seq_to_features = self._drop_missing_features(None, seq_to_features, "Inference plotting")
        self._require_vector_features(seq_to_features, "Inference plotting")
        seq_ids = list(seq_to_features)
        features = np.asarray([seq_to_features[seq_id] for seq_id in seq_ids])
        if norm_scheme is not None and norm_scheme != "none":
            features = self._normalize_features(features, norm_scheme)
        return seq_ids, features

    def _fitness_for_method(
        self,
        layer: str | int,
        abstraction_method: AbstractionMethod | str,
        embedding_type: EmbeddingType | None,
        norm_scheme: str,
        abstraction_params: dict | None = None,
    ) -> dict:
        result = self.load_inference_results(layer, abstraction_method, norm_scheme, embedding_type)
        seq_ids, features = self._features_for_inference(
            layer,
            abstraction_method,
            embedding_type,
            norm_scheme,
            abstraction_params,
        )
        if getattr(result, "s_joint", None) is not None:
            fitness = features @ result.s_joint
        else:
            fitness = np.asarray([features @ result.s[rep_idx] for rep_idx in range(result.s.shape[0])]).mean(axis=0)
        return dict(zip(seq_ids, fitness))

    @staticmethod
    def _paired_fitness_values(left_fitness: dict, right_fitness: dict) -> tuple[np.ndarray, np.ndarray, list]:
        common_ids = [seq_id for seq_id in left_fitness if seq_id in right_fitness]
        if not common_ids:
            raise ValueError("No shared sequence IDs found between the two fitness mappings.")
        left_values = np.asarray([left_fitness[seq_id] for seq_id in common_ids], dtype=float)
        right_values = np.asarray([right_fitness[seq_id] for seq_id in common_ids], dtype=float)
        return left_values, right_values, common_ids

    def plot_fitness_method_comparison(
        self,
        layer: str | int,
        left_abstraction_method: AbstractionMethod | str,
        right_abstraction_method: AbstractionMethod | str,
        left_abstraction_params: dict | None = None,
        right_abstraction_params: dict | None = None,
        embedding_type: EmbeddingType | None = None,
        norm_scheme: str = "none",
        left_label: str | None = None,
        right_label: str | None = None,
        output_path: str | Path | None = None,
    ) -> tuple[plt.Figure, dict]:
        """
        Scatter inferred sequence fitness from one feature method against another.
        """
        left_fitness = self._fitness_for_method(
            layer,
            left_abstraction_method,
            embedding_type,
            norm_scheme,
            left_abstraction_params,
        )
        right_fitness = self._fitness_for_method(
            layer,
            right_abstraction_method,
            embedding_type,
            norm_scheme,
            right_abstraction_params,
        )
        left_values, right_values, common_ids = self._paired_fitness_values(left_fitness, right_fitness)
        rho = self._safe_corr(left_values, right_values, spearmanr)

        left_label = left_label or self._abstraction_type(left_abstraction_method)
        right_label = right_label or self._abstraction_type(right_abstraction_method)

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        sns.scatterplot(x=left_values, y=right_values, ax=ax, s=18, alpha=0.6, edgecolor=None)
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        if finite.any():
            lo = min(np.min(left_values[finite]), np.min(right_values[finite]))
            hi = max(np.max(left_values[finite]), np.max(right_values[finite]))
            pad = (hi - lo) * 0.05 if hi > lo else 1.0
            lo -= pad
            hi += pad
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.7)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{left_label} inferred fitness")
        ax.set_ylabel(f"{right_label} inferred fitness")
        ax.set_title(f"{self._layer_label(layer)} fitness comparison\nSpearman rho={rho:.3f}, n={len(common_ids)}")
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, bbox_inches="tight", dpi=150)

        stats = {
            "layer": layer,
            "left_method": self._abstraction_type(left_abstraction_method),
            "right_method": self._abstraction_type(right_abstraction_method),
            "spearman_rho": rho,
            "n_sequences": len(common_ids),
        }
        return fig, stats

    def plot_fitness_method_correlation_by_layer(
        self,
        layers: list[str | int],
        left_abstraction_method: AbstractionMethod | str,
        right_abstraction_method: AbstractionMethod | str,
        left_abstraction_params: dict | None = None,
        right_abstraction_params: dict | None = None,
        embedding_type: EmbeddingType | None = None,
        norm_scheme: str = "none",
        left_label: str | None = None,
        right_label: str | None = None,
        output_path: str | Path | None = None,
    ) -> tuple[plt.Figure, pd.DataFrame]:
        """
        Plot Spearman rho between two methods' inferred fitness across layers.
        """
        rows = []
        for layer in layers:
            left_fitness = self._fitness_for_method(
                layer,
                left_abstraction_method,
                embedding_type,
                norm_scheme,
                left_abstraction_params,
            )
            right_fitness = self._fitness_for_method(
                layer,
                right_abstraction_method,
                embedding_type,
                norm_scheme,
                right_abstraction_params,
            )
            left_values, right_values, common_ids = self._paired_fitness_values(left_fitness, right_fitness)
            rows.append({
                "layer": layer,
                "spearman_rho": self._safe_corr(left_values, right_values, spearmanr),
                "n_sequences": len(common_ids),
            })

        corr_df = pd.DataFrame(rows)
        left_label = left_label or self._abstraction_type(left_abstraction_method)
        right_label = right_label or self._abstraction_type(right_abstraction_method)

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(max(6, len(corr_df) * 0.45), 4))
        ax.plot(corr_df["layer"], corr_df["spearman_rho"], marker="o")
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Spearman rho")
        ax.set_title(f"{left_label} vs {right_label} inferred fitness by layer")
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
        return fig, corr_df

    def analyze_method_fitness_consistency_by_layer(
        self,
        layers: list[str | int],
        method_specs: list[dict],
        norm_scheme: str = "none",
        ensure_inference: bool = False,
        output_path: str | Path | None = None,
    ) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
        """
        Compare methods by cross-replicate inferred-fitness Spearman rho.

        method_specs entries may include:
            label, abstraction_method, abstraction_params, embedding_type
        """
        detail_rows = []
        summary_rows = []

        for spec in method_specs:
            method = spec.get("abstraction_method", "none")
            params = dict(spec.get("abstraction_params") or {})
            params.setdefault("norm_scheme", norm_scheme)
            embedding_type = self._embedding_type(spec.get("embedding_type"))
            label = spec.get("label") or f"{embedding_type}:{self._abstraction_type(method)}"

            for layer in layers:
                if ensure_inference:
                    result = self.run_feature_inference(
                        layer=layer,
                        abstraction_method=method,
                        abstraction_params=params,
                        embedding_type=embedding_type,
                    )
                else:
                    result = self.load_inference_results(
                        layer=layer,
                        abstraction_method=method,
                        norm_scheme=norm_scheme,
                        embedding_type=embedding_type,
                    )

                _, features = self._features_for_inference(
                    layer,
                    method,
                    embedding_type,
                    norm_scheme,
                    params,
                )
                rep_fitness = np.asarray([features @ result.s[rep_idx] for rep_idx in range(result.s.shape[0])])
                pair_rows = []
                for rep_i, rep_j in self._rep_pairs(rep_fitness.shape[0]):
                    rho = self._safe_corr(rep_fitness[rep_i], rep_fitness[rep_j], spearmanr)
                    row = {
                        "label": label,
                        "layer": layer,
                        "rep_i": rep_i + 1,
                        "rep_j": rep_j + 1,
                        "spearman_rho": rho,
                        "n_sequences": rep_fitness.shape[1],
                        "embedding_type": embedding_type,
                        "abstraction_method": self._abstraction_type(method),
                    }
                    detail_rows.append(row)
                    pair_rows.append(row)

                pair_rhos = [row["spearman_rho"] for row in pair_rows]
                summary_rows.append({
                    "label": label,
                    "layer": layer,
                    "mean_spearman_rho": np.nanmean(pair_rhos) if pair_rhos else np.nan,
                    "median_spearman_rho": np.nanmedian(pair_rhos) if pair_rhos else np.nan,
                    "n_pairs": len(pair_rows),
                    "n_sequences": rep_fitness.shape[1],
                    "embedding_type": embedding_type,
                    "abstraction_method": self._abstraction_type(method),
                })

        summary_df = pd.DataFrame(summary_rows)
        detail_df = pd.DataFrame(detail_rows)
        if summary_df.empty:
            raise ValueError("No method consistency rows were created.")

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(max(7, len(layers) * 0.6), 4.8))
        sns.lineplot(
            data=summary_df,
            x="layer",
            y="mean_spearman_rho",
            hue="label",
            marker="o",
            ax=ax,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean pairwise fitness Spearman rho")
        ax.set_title("Cross-replicate inferred-fitness consistency by method")
        ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
        return fig, summary_df, detail_df

    def _plot_rep_scatter_grid(
        self,
        rep_values: np.ndarray,
        title: str,
        axis_label: str,
        output_path: str | Path | None = None,
        max_cols: int = 3,
    ):
        sns.set_theme(style="darkgrid")
        rep_values = np.asarray(rep_values)
        n_reps = rep_values.shape[0]
        pairs = self._rep_pairs(n_reps)
        if not pairs:
            raise ValueError("At least two replicates are required for replicate comparison plots.")

        n_cols = min(max_cols, len(pairs))
        n_rows = int(np.ceil(len(pairs) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
        fig.suptitle(title, fontsize=13)

        for ax_idx, (ri, rj) in enumerate(pairs):
            row, col = divmod(ax_idx, n_cols)
            ax = axes[row][col]
            x = rep_values[ri]
            y = rep_values[rj]
            pr = self._safe_corr(x, y, pearsonr)
            sr = self._safe_corr(x, y, spearmanr)
            sns.scatterplot(x=x, y=y, ax=ax, s=14, alpha=0.55, edgecolor=None)

            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                lo = min(np.min(x[finite]), np.min(y[finite]))
                hi = max(np.max(x[finite]), np.max(y[finite]))
                pad = (hi - lo) * 0.05 if hi > lo else 1.0
                lo -= pad
                hi += pad
                ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.7)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)

            ax.set_title(f"Rep {ri + 1} vs Rep {rj + 1}\nr={pr:.3f}, rho={sr:.3f}", fontsize=10)
            ax.set_xlabel(f"Rep {ri + 1} {axis_label}")
            ax.set_ylabel(f"Rep {rj + 1} {axis_label}")

        for ax_idx in range(len(pairs), n_rows * n_cols):
            row, col = divmod(ax_idx, n_cols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
        return fig

    def plot_rep_sel_comps(
        self,
        layer: str | int,
        abstraction_method: AbstractionMethod = 'none',
        embedding_type: EmbeddingType | None = None,
        norm_scheme: str = "none",
        label: str | None = None,
        output_path: str | Path | None = None,
        max_cols: int = 3,
    ):
        """
        Plot replicate-vs-replicate inferred selection coefficients for a saved result.
        """
        result = self.load_inference_results(layer, abstraction_method, norm_scheme, embedding_type)
        title_label = label or abstraction_method
        title = f"{title_label} selection coefficients, {self._layer_label(layer)}"
        return self._plot_rep_scatter_grid(result.s, title, "selection coefficient", output_path, max_cols)

    def plot_rep_fit_comps(
        self,
        layer: str | int,
        abstraction_method: AbstractionMethod = 'none',
        norm_scheme: str = "none",
        embedding_type: EmbeddingType | None = None,
        label: str | None = None,
        output_path: str | Path | None = None,
        max_cols: int = 3,
    ):
        """
        Plot replicate-vs-replicate inferred sequence fitness for a saved result.
        """
        result = self.load_inference_results(layer, abstraction_method, norm_scheme, embedding_type)
        _, features = self._features_for_inference(layer, abstraction_method, embedding_type, norm_scheme)
        rep_fitness = np.asarray([features @ result.s[rep_idx] for rep_idx in range(result.s.shape[0])])
        title_label = label or abstraction_method
        title = f"{title_label} inferred fitness, {self._layer_label(layer)}"
        return self._plot_rep_scatter_grid(rep_fitness, title, "fitness", output_path, max_cols)

    def plot_avg_rep_correlations_by_layer(
        self,
        layers: list[str | int],
        abstraction_method: AbstractionMethod = 'none',
        norm_scheme: str = "none",
        embedding_type: EmbeddingType | None = None,
        comparison: Literal["selection", "fitness"] = "selection",
        label: str | None = None,
        output_path: str | Path | None = None,
    ) -> tuple[plt.Figure, pd.DataFrame]:
        """
        Plot average pairwise Pearson and Spearman replicate correlations across layers.
        """
        sns.set_theme(style="darkgrid")
        rows = []
        for layer in layers:
            result = self.load_inference_results(layer, abstraction_method, norm_scheme, embedding_type)
            if comparison == "selection":
                rep_values = result.s
            elif comparison == "fitness":
                _, features = self._features_for_inference(layer, abstraction_method, embedding_type, norm_scheme)
                rep_values = np.asarray([features @ result.s[rep_idx] for rep_idx in range(result.s.shape[0])])
            else:
                raise ValueError("comparison must be either 'selection' or 'fitness'.")

            pearson_vals = []
            spearman_vals = []
            pairs = self._rep_pairs(rep_values.shape[0])
            for ri, rj in pairs:
                pearson_vals.append(self._safe_corr(rep_values[ri], rep_values[rj], pearsonr))
                spearman_vals.append(self._safe_corr(rep_values[ri], rep_values[rj], spearmanr))

            rows.append({
                "layer": layer,
                "pearson_r": np.nanmean(pearson_vals) if pearson_vals else np.nan,
                "spearman_r": np.nanmean(spearman_vals) if spearman_vals else np.nan,
                "n_pairs": len(pairs),
            })

        corr_df = pd.DataFrame(rows)
        if corr_df.empty:
            raise ValueError("At least one layer is required for the correlation summary plot.")

        fig, ax = plt.subplots(figsize=(max(6, len(corr_df) * 0.45), 4))
        ax.plot(corr_df["layer"], corr_df["pearson_r"], marker="o", label="Pearson r")
        ax.plot(corr_df["layer"], corr_df["spearman_r"], marker="s", label="Spearman rho")
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Average pairwise replicate correlation")
        title_label = label or abstraction_method
        ax.set_title(f"{title_label} {comparison} replicate correlations by layer")
        ax.legend()
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
        return fig, corr_df
