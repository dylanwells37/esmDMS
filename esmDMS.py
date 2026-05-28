from __future__ import annotations

import pickle
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModel, AutoTokenizer

from embedding_scripts.embed_sequences import (
    get_reference_nuc_sequence,
    build_sequence_dataframe_mavedb,
    get_reference_sequence_from_wildtypes,
    build_sequence_dataframe,
    embed_sequence
)

from popDMS import mini_infer_esm, InferenceResult


## TYPE DEFINITIONS #################################


EmbeddingModel = Literal[
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
]

EmbeddingType = Literal["per_residue", "mutation_pooled", "mean_pool"]
AbstractionMethod = Literal["none", "PCA", "SAE", "SPCA"]

## CONFIGURATION AND INPUT CLASSES #################################

@dataclass(frozen=True)
class ESMDMSConfig:
    embedding_model: EmbeddingModel = "esm2_t33_650M_UR50D"
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
        if self.embedding_type not in {"per_residue", "mutation_pooled", "mean_pool"}:
            raise ValueError("embedding_type must be one of 'per_residue', 'mutation_pooled', or 'mean_pool'.")
        if self.dataset_name is not None:
            if not self.dataset_name:
                raise ValueError("dataset_name must not be empty when specified.")
            if any(sep in self.dataset_name for sep in ("/", "\\")):
                raise ValueError("dataset_name must be a file name component and cannot contain path separators.")
        
    
@dataclass(frozen=True)
class CellularDMSInput:
    reference_nuc_path: Path
    mavedb_csv_path: Path
    use_replicates: list[str] | None = None
    kind: Literal['cellular'] = 'cellular'

    def __post_init__(self):
        object.__setattr__(self, 'reference_nuc_path', Path(self.reference_nuc_path))
        object.__setattr__(self, 'mavedb_csv_path', Path(self.mavedb_csv_path))
        if not self.reference_nuc_path.is_file():
            raise ValueError(f"Reference nucleotide sequence path {self.reference_nuc_path} does not exist or is not a file.")
        if not self.mavedb_csv_path.is_file():
            raise ValueError(f"MaveDB CSV path {self.mavedb_csv_path} does not exist or is not a file.")


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


class SparseAutoencoder(torch.nn.Module):
    """
    Simple sparse autoencoder: Linear+ReLU encoder, linear decoder (no bias).

    Architecture:
        x  →  encoder (Linear + ReLU)  →  z  →  decoder (Linear)  →  x_hat

    The decoder columns are optionally kept at unit norm throughout training to
    prevent feature collapse (standard SAE practice).

    Loss = MSE(x, x_hat) + sparsity_coeff * mean(|z|)
    """

    def __init__(self, input_dim: int, n_features: int, normalize_decoder: bool = True):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, n_features)
        self.decoder = torch.nn.Linear(n_features, input_dim, bias=False)
        self.normalize_decoder = normalize_decoder
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
        return torch.relu(self.encoder(x))

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

    def _base_embedding_path(self, layer: str | int) -> Path:
        return self._save_dir() / f"{self._dataset_prefix()}{self.config.embedding_model}_all_data_{self._layer_label(layer)}_embeddings.pkl"

    def _embedding_path(self, layer: str | int, embedding_type: str | None = None) -> Path:
        embedding_type = self._embedding_type(embedding_type)
        return self._feature_path("none", layer, embedding_type)

    def _feature_path(self, method: str, layer: str | int, embedding_type: str | None = None) -> Path:
        embedding_type = self._embedding_type(embedding_type)
        abstraction_type = self._abstraction_type(method)
        return self._save_dir() / (
            f"{self._dataset_prefix()}{self.config.embedding_model}_{embedding_type}_{abstraction_type}_"
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
            f"{self._dataset_prefix()}{self.config.embedding_model}_{embedding_type}_{abstraction_type}_"
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
    def _abstraction_type(method: str) -> str:
        if method == "Embeddings":
            return "none"
        return method

    @staticmethod
    def _abstraction_base_method(method: str) -> str:
        method = esmDMS._abstraction_type(method)
        if method.startswith("PCA"):
            return "PCA"
        if method.startswith("SAE"):
            return "SAE"
        if method.startswith("SPCA"):
            return "SPCA"
        return method

    def _embedding_type(self, embedding_type: str | None = None) -> str:
        embedding_type = embedding_type or self.config.embedding_type
        if embedding_type in {"mutation_site", "mutation_pool", "pooled"}:
            return "mutation_pooled"
        if embedding_type not in {"per_residue", "mutation_pooled", "mean_pool"}:
            raise ValueError("embedding_type must be one of 'per_residue', 'mutation_pooled', or 'mean_pool'.")
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
            batch_dir = self._save_dir() / "embedding_batches" / f"{self._dataset_prefix()}{self.config.embedding_model}_all_data"
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
                / f"{self._dataset_prefix()}{self.config.embedding_model}_{self._embedding_type(embedding_type)}_{self._abstraction_type(abstraction_method)}_{self._layer_label(layer)}_{norm_scheme}"
            )
        inference_job_dir.mkdir(parents=True, exist_ok=True)
        return inference_job_dir

    def _inference_payload_path(self, inference_job_dir: Path) -> Path:
        return inference_job_dir / f"{self._dataset_prefix()}inference_payload.pkl"

    # ── SAE path helpers ──────────────────────────────────────────────────

    def _sae_model_dir(self) -> Path:
        return self._save_dir() / "sae_models"

    def _sae_tag(self, layer: str | int, n_features: int, sparsity_coeff: float) -> str:
        return f"{self._dataset_prefix()}sae_{self._layer_label(layer)}_{n_features}_{sparsity_coeff}"

    def _sae_model_path(self, layer: str | int, n_features: int, sparsity_coeff: float) -> Path:
        return self._sae_model_dir() / f"{self._sae_tag(layer, n_features, sparsity_coeff)}_model.pt"

    def _sae_viz_path(self, layer: str | int, n_features: int, sparsity_coeff: float) -> Path:
        return self._sae_model_dir() / f"{self._sae_tag(layer, n_features, sparsity_coeff)}_viz_data.pkl"

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
        if embedding_type not in {"per_residue", "mutation_pooled", "mean_pool"}:
            raise ValueError(f"Unsupported embedding_type: {embedding_type}")

        derived = {}
        for seq_id, embedding in layer_embeddings.items():
            if embedding is None:
                derived[seq_id] = None
                continue
            embedding = np.asarray(embedding)
            if embedding.ndim != 2:
                raise ValueError(
                    f"Cannot derive {embedding_type} from all-data layer embedding shape {embedding.shape} "
                    f"for sequence {seq_id}."
                )

            if embedding_type == "mean_pool":
                derived[seq_id] = embedding.mean(axis=0)
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
        per_residue = self._derive_per_residue_features(layer_embeddings, self.sequence_to_mutation_sites)

        if self._use_memory():
            self.sequence_to_features[self._feature_key("none", layer, "mean_pool")] = mean_pool
            self.sequence_to_features[self._feature_key("none", layer, "per_residue")] = per_residue
        if self._use_disk():
            self._save_pickle(mean_pool, self._embedding_path(layer, "mean_pool"))
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

        feature_chunks = {"mean_pool": {}, "per_residue": {}}
        for layer_value in layers:
            layer_embeddings = cls._select_layer(embeddings, layer_value, allow_per_residue=True)
            layer_label = cls._layer_label(layer_value)
            feature_chunks["mean_pool"][layer_label] = cls._derive_mean_pool_features(layer_embeddings)
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

    def load_reference_sequence(self):
        """
        Load the reference sequence for the input DMS data.
        """
        if self.input_data.kind == 'cellular':
            self.reference_sequence = get_reference_nuc_sequence(
                self.input_data.reference_nuc_path
            )
        elif self.input_data.kind == 'viral':
            self.reference_sequence = get_reference_sequence_from_wildtypes(
                self.input_data.pre_files[0]
            )
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
            ) = build_sequence_dataframe_mavedb(
                self.input_data.mavedb_csv_path,
                self.reference_sequence,
                self.input_data.use_replicates,
                skip_stop_codons=drop_stop_codons
            )

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

        else:
            raise ValueError(f"Unsupported DMS data kind: {self.input_data.kind}")


    def embed_sequences(self, seq_ids: list[str], out_path: str | None = None) -> dict[str, np.ndarray]:
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
        esm_model = f"facebook/{self.config.embedding_model}"

        tokenizer = AutoTokenizer.from_pretrained(esm_model, do_lower_case=False)
        model = AutoModel.from_pretrained(esm_model)
        model.eval()

        seq_idx_to_embedding = {}

        for idx in seq_ids:
            prot_seq = self.sequence_to_protein_sequence[idx]
            layer_embeddings = embed_sequence(prot_seq, tokenizer, model)
            seq_idx_to_embedding[idx] = layer_embeddings

        if self._use_memory():
            self.sequence_to_embeddings.update(seq_idx_to_embedding)

        # Save the embeddings to the specified path if it exists
        if out_path is not None:
            self._save_pickle(seq_idx_to_embedding, Path(out_path))
        return seq_idx_to_embedding
    

    def embed_all_sequences(self, layer: str = "all") -> None:
        """
        Embed all sequences and cache compact derived features by layer.

        Disk caches are written as separate mean_pool and mutation-site
        per_residue files. mutation_pooled is derived from per_residue on load.
        """
        if self.sequence_dataframe is None:
            raise ValueError("Sequence dataframe is not available. Please run process_raw_data() first.")
        if self.sequence_to_mutation_sites is None:
            raise ValueError("Mutation-site mapping is not available. Please run process_raw_data() first.")

        seq_ids = sorted(self.sequence_dataframe['SequenceIndex'].unique())
        embeddings = self.embed_sequences(seq_ids)

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
        cpus_per_task: int = 4,
        mem: str = "16G",
        time: str = "06:00:00",
        python_executable: str = "python3",
        scratch_root: str | Path = "/scr",
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
        """
        if not self._use_disk():
            raise ValueError("Embedding batch jobs require local_or_disk to be 'disk' or 'both'.")
        if self.sequence_dataframe is None or self.sequence_to_protein_sequence is None or self.sequence_to_mutation_sites is None:
            raise ValueError("Run process_raw_data() before creating an embedding batch job.")
        if n_chunks < 1:
            raise ValueError("n_chunks must be at least 1.")
        if max_active_jobs is not None and max_active_jobs < 1:
            raise ValueError("max_active_jobs must be at least 1 when specified.")

        batch_dir = self._batch_dir(job_dir)
        logs_dir = batch_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        seq_ids = [int(seq_id) for seq_id in sorted(self.sequence_dataframe["SequenceIndex"].unique())]
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
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --array={array_spec}
#SBATCH --output={logs_dir}/slurm-%A_%a.out
#SBATCH --error={logs_dir}/slurm-%A_%a.err

set -euo pipefail
cd {Path.cwd()}

SCRDIR={scratch_root}/${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
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

        esm_model = f"facebook/{payload['embedding_model']}"
        tokenizer = AutoTokenizer.from_pretrained(esm_model, do_lower_case=False)
        model = AutoModel.from_pretrained(esm_model)
        model.eval()

        out = {}
        for seq_id in chunks[chunk_idx].tolist():
            prot_seq = payload["sequence_to_protein_sequence"][seq_id]
            out[seq_id] = embed_sequence(prot_seq, tokenizer, model)
        if payload.get("sequence_to_mutation_sites") is not None:
            feature_chunks = esmDMS._build_feature_chunks(
                out,
                payload.get("layer", "all"),
                payload["sequence_to_mutation_sites"],
            )
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

            feature_types = ("mean_pool", "per_residue")
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
            legacy_chunk_files = [
                self._batch_chunk_path(batch_dir, idx) for idx in range(n_chunks or 0)
            ]
            has_legacy_chunks = legacy_chunk_files and all(path.is_file() for path in legacy_chunk_files)
            if missing_feature_chunks and not has_legacy_chunks:
                raise FileNotFoundError(f"Missing embedding feature chunk files: {missing_feature_chunks}")

            if not missing_feature_chunks:
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
                            merged_features.setdefault(layer_label, {"mean_pool": {}, "per_residue": {}})
                            merged_features[layer_label][embedding_type].update(seq_to_features)

                for layer_label, layer_features in merged_features.items():
                    if self._use_memory():
                        self.sequence_to_features[self._feature_key("none", layer_label, "mean_pool")] = layer_features["mean_pool"]
                        self.sequence_to_features[self._feature_key("none", layer_label, "per_residue")] = layer_features["per_residue"]
                    if save_layers and self._use_disk():
                        self._save_pickle(layer_features["mean_pool"], self._embedding_path(layer_label, "mean_pool"))
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
        feature_chunks = []
        for path in chunk_files:
            chunk = self._load_pickle(path)
            if isinstance(chunk, dict) and chunk.get("format") == "esmDMS_embedding_features_v1":
                feature_chunks.append(chunk)
            else:
                merged.update(chunk)

        if feature_chunks:
            merged_features = {}
            for chunk in feature_chunks:
                for layer_label, layer_features in chunk["features_by_layer"].items():
                    merged_features.setdefault(layer_label, {"mean_pool": {}, "per_residue": {}})
                    merged_features[layer_label]["mean_pool"].update(layer_features["mean_pool"])
                    merged_features[layer_label]["per_residue"].update(layer_features["per_residue"])

            for layer_label, layer_features in merged_features.items():
                if self._use_memory():
                    self.sequence_to_features[self._feature_key("none", layer_label, "mean_pool")] = layer_features["mean_pool"]
                    self.sequence_to_features[self._feature_key("none", layer_label, "per_residue")] = layer_features["per_residue"]
                if save_layers and self._use_disk():
                    self._save_pickle(layer_features["mean_pool"], self._embedding_path(layer_label, "mean_pool"))
                    self._save_pickle(layer_features["per_residue"], self._embedding_path(layer_label, "per_residue"))

            if not merged:
                return merged_features

        if self._use_memory():
            self.sequence_to_embeddings.update(merged)

        first_embedding = next((np.asarray(embedding) for embedding in merged.values() if embedding is not None), None)
        if first_embedding is None:
            raise ValueError("No non-empty embeddings found in merged batch outputs.")
        if layer == "all":
            layer_axis = 1 if first_embedding.ndim == 3 else 0
            layers = range(first_embedding.shape[layer_axis])
        else:
            layers = [layer]
        for layer_value in layers:
            layer_embeddings = self._select_layer(merged, layer_value, allow_per_residue=True)
            if save_layers and self._use_disk():
                self._save_layer_feature_caches(layer_embeddings, layer_value)
            elif self._use_memory():
                mean_pool = self._derive_mean_pool_features(layer_embeddings)
                per_residue = self._derive_per_residue_features(layer_embeddings, self.sequence_to_mutation_sites)
                self.sequence_to_features[self._feature_key("none", layer_value, "mean_pool")] = mean_pool
                self.sequence_to_features[self._feature_key("none", layer_value, "per_residue")] = per_residue

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
            print("Embeddings already exist in memory. Returning existing embeddings.")
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
                    print(f"Loading per_residue embeddings from {per_residue_path}")
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
                print(f"Loading {embedding_type} embeddings from {save_path}")
                embeddings = self._load_pickle(save_path)
                if self._use_memory():
                    self.sequence_to_features[key] = embeddings
                return embeddings
            base_path = self._base_embedding_path(layer)
            if base_path.is_file():
                print(f"Loading all-data embeddings from {base_path}")
                all_data_embeddings = self._load_pickle(base_path)
                if embedding_type == "mutation_pooled":
                    per_residue_embeddings = self._derive_embedding_type(all_data_embeddings, "per_residue")
                    embeddings = self._pool_per_residue_features(per_residue_embeddings)
                    self._save_pickle(per_residue_embeddings, self._embedding_path(layer, "per_residue"))
                else:
                    embeddings = self._derive_embedding_type(all_data_embeddings, embedding_type)
                    self._save_pickle(embeddings, save_path)
                if self._use_memory():
                    self.sequence_to_features[base_key] = all_data_embeddings
                    if embedding_type == "mutation_pooled":
                        self.sequence_to_features[per_residue_key] = per_residue_embeddings
                    self.sequence_to_features[key] = embeddings
                return embeddings
            if embedding_type == "mutation_pooled":
                raise FileNotFoundError(
                    f"No saved per_residue embeddings found at {self._embedding_path(layer, 'per_residue')} "
                    f"and no legacy all-data embeddings found at {base_path}. Run embed_all_sequences({layer!r}) first."
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
                            embedding_type: EmbeddingType | None = None) -> dict[str, np.ndarray]:
        """
        Create an abstraction of the embeddings using the specified method.

        Parameters:
        -----------
        layer : str
            The layer from which to extract features.
        method : Literal['none', 'PCA', 'SAE', 'SPCA']
            The method to use for creating the abstraction.
        method_params : dict | None
            Parameters for the abstraction method.

        Returns:
        --------
        dict[str, np.ndarray]
            A dictionary mapping sequence indices to their corresponding features.
        """
        embedding_type = self._embedding_type(embedding_type)
        method = self._abstraction_type(method)
        if method == 'none':
            print("No abstraction method specified. Using raw embeddings as features.")
            features = self.load_embeddings(layer, embedding_type)
            _, features = self._drop_missing_features(None, features, f"{embedding_type} features")
            if self._use_disk() and embedding_type != "mutation_pooled":
                self._save_pickle(features, self._feature_path(method, layer, embedding_type))
            return features

        embeddings = self.load_embeddings(layer, embedding_type)
        _, embeddings = self._drop_missing_features(None, embeddings, f"{method} abstraction")
        self._require_vector_features(embeddings, f"{method} abstraction with embedding_type={embedding_type!r}")
        key = self._feature_key(method, layer, embedding_type)

        if self._use_memory() and self.sequence_to_features.get(key) is not None:
            print("Abstracted features already exist in memory. Returning existing features.")
            return self.sequence_to_features[key]

        if self._use_disk():
            save_path = self._feature_path(method, layer, embedding_type)
            if save_path.is_file():
                print(f"Abstracted already saved in {save_path}")
                abstracted_features = self._load_pickle(save_path)
                if self._use_memory():
                    self.sequence_to_features[key] = abstracted_features
                return abstracted_features

        print("No abstracted features found. Creating new features.")
        _params = dict(method_params or {})
        _params.setdefault("_layer", layer)
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

        # ── Hyper-parameters ──────────────────────────────────────────────
        sparsity_coeff: float = params.get("sparsity_coeff", 1e-3)
        lr: float = params.get("lr", 1e-3)
        epochs: int = params.get("epochs", 200)
        batch_size: int = params.get("batch_size", 64)
        train_frac: float = params.get("train_frac", 0.8)
        normalize_decoder: bool = params.get("normalize_decoder", True)
        activity_threshold: float = params.get("activity_threshold", 1e-3)
        seed: int = params.get("seed", 42)

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

        # ── Model + optimizer ─────────────────────────────────────────────
        model = SparseAutoencoder(input_dim, n_features, normalize_decoder).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # ── Training loop ─────────────────────────────────────────────────
        train_losses: list[float] = []
        test_losses: list[float] = []
        for epoch in range(epochs):
            model.train()
            epoch_perm = torch.randperm(len(X_train), device=device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(X_train), batch_size):
                batch = X_train[epoch_perm[start : start + batch_size]]
                optimizer.zero_grad()
                x_hat, z = model(batch)
                loss = torch.nn.functional.mse_loss(x_hat, batch) + sparsity_coeff * z.abs().mean()
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
        mean_act = Z_all.mean(axis=0)
        active_mask = mean_act > activity_threshold
        if not active_mask.any():
            print(
                "Warning: no neurons exceeded the activity threshold. "
                "Falling back to the top 10% most active neurons."
            )
            active_mask = mean_act >= np.percentile(mean_act, 90)

        n_active = int(active_mask.sum())
        print(
            f"SAE: {n_active}/{n_features} neurons active "
            f"(threshold={activity_threshold}, layer={layer})"
        )
        Z_active = Z_all[:, active_mask]       # (n_samples, n_active)

        result = {sid: Z_active[i] for i, sid in enumerate(seq_ids)}

        # ── Persist model + viz data ──────────────────────────────────────
        if self._use_disk():
            sae_dir = self._sae_model_dir()
            sae_dir.mkdir(parents=True, exist_ok=True)

            model_path = self._sae_model_path(layer, n_features, sparsity_coeff)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "n_features": n_features,
                    "normalize_decoder": normalize_decoder,
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
            self._save_pickle(viz_data, self._sae_viz_path(layer, n_features, sparsity_coeff))

        return result

    #TODO Add a regular autoencoder abstraction as well

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
        sparsity_coeff: float = params.get("sparsity_coeff", 1e-3)

        # Resolve n_features: need input_dim to compute the default.
        embeddings = self.load_embeddings(layer)
        input_dim = next(iter(embeddings.values())).shape[0]
        n_features: int = params.get("n_features", input_dim * 2)

        viz_path = self._sae_viz_path(layer, n_features, sparsity_coeff)
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
                    / f"{self._sae_tag(layer, n_features, sparsity_coeff)}_viz.png"
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
        method : Literal['none', 'PCA', 'SAE', 'SPCA']
            The method used for abstraction.
        method_params : dict | None
            Parameters for the abstraction method.

        Returns:
        --------
        dict[str, np.ndarray] | None
            A dictionary mapping sequence indices to their corresponding abstracted features, or None if no saved features are found.
        """
        embedding_type = self._embedding_type(embedding_type)
        method = self._abstraction_type(method)
        if method == 'none':
            print("No abstraction method specified. Using raw embeddings as features.")
            return self.load_embeddings(layer, embedding_type)

        key = self._feature_key(method, layer, embedding_type)
        if self._use_memory() and self.sequence_to_features.get(key) is not None:
            print("Abstracted features already exist in memory. Returning existing features.")
            return self.sequence_to_features[key]

        if self._use_disk():
            save_path = self._feature_path(method, layer, embedding_type)
            if save_path.is_file():
                print(f"Loading abstracted features from {save_path}")
                abstracted_features = self._load_pickle(save_path)
                if self._use_memory():
                    self.sequence_to_features[key] = abstracted_features
                return abstracted_features

        return self.create_feature_space(layer, method, method_params, embedding_type)

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
                print(f"Loading inference results from {save_path}")
                return self._load_pickle(save_path)
            else:
                print(f"No saved inference results found at {save_path}. Running inference and saving results.")
        
        # load features, seq_to_features type = dict[str, np.ndarray]
        seq_to_features = self._load_abstracted_features(layer, abstraction_method, embedding_type, abstraction_params)
        sequence_dataframe, seq_to_features = self._drop_missing_features(
            self.sequence_dataframe,
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
        seq_to_features = self._load_abstracted_features(layer, abstraction_method, embedding_type, method_params)
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
