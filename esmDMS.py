from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from embedding_scripts.embed_sequences import (
    get_reference_nuc_sequence,
    build_sequence_dataframe_mavedb,
    get_reference_sequence_from_wildtypes,
    build_sequence_dataframe,
    cls_sequence_representation,
    pool_sequence_representation,
    mutation_site_representation,
    embed_sequence
)

from popDMS import mini_infer_esm


## TYPE DEFINITIONS #################################


EmbeddingModel = Literal[
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
]

EmbeddingMethod = Literal["mean_pool", "per_residue", "cls_token"]

## CONFIGURATION AND INPUT CLASSES #################################

@dataclass(frozen=True)
class ESMDMSConfig:
    embedding_model: EmbeddingModel = "esm2_t33_650M_UR50D"
    embedding_method: EmbeddingMethod = "mean_pool"
    per_residue_mutation_pooling: bool = False
    local_or_disk: Literal['local', 'disk', 'both'] = 'local'
    save_dir: str | None = None

    def __post_init__(self):
        if self.local_or_disk not in {"local", "disk", "both"}:
            raise ValueError("local_or_disk must be one of 'local', 'disk', or 'both'.")
        if (self.local_or_disk == 'disk' or self.local_or_disk == 'both') and self.save_dir is None:
                raise ValueError("save_dir must be specified when local_or_disk is set to 'disk' or 'both'.")
        
    
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
            - Investigate how embedding method impacts all of this (mean-pooling vs per-residue vs cls token)
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

    def _feature_key(self, method: str, layer: str | int) -> str:
        return f"{method}_{self._layer_label(layer)}"

    def _embedding_path(self, layer: str | int) -> Path:
        return self._save_dir() / f"{self.config.embedding_model}_{self.config.embedding_method}_{self._layer_label(layer)}_embeddings.pkl"

    def _feature_path(self, method: str, layer: str | int) -> Path:
        return self._save_dir() / f"{method}_{self._layer_label(layer)}_abstracted_features.pkl"

    def _inference_path(self, abstraction_method: str, layer: str | int, norm_scheme: str) -> Path:
        return self._save_dir() / f"{abstraction_method}_{self._layer_label(layer)}_{norm_scheme}_inference_results.pkl"

    @staticmethod
    def _load_pickle(path: Path):
        with path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _save_pickle(value, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(value, f)

    @staticmethod
    def _select_layer(embeddings: dict[str, np.ndarray], layer: str | int) -> dict[str, np.ndarray]:
        layer_idx = int(str(layer).replace("Layer_", ""))
        selected = {}
        for seq_id, embedding in embeddings.items():
            embedding = np.asarray(embedding)
            if embedding.ndim == 1:
                selected[seq_id] = embedding
            elif embedding.ndim == 2:
                selected[seq_id] = embedding[layer_idx]
            elif embedding.ndim == 3:
                layer_embedding = embedding[:, layer_idx, :]
                if layer_embedding.shape[0] != 1:
                    raise ValueError(
                        "Unpooled per-residue embeddings are not valid feature vectors for inference. "
                        "Set per_residue_mutation_pooling=True or add a feature abstraction that flattens them intentionally."
                    )
                selected[seq_id] = layer_embedding[0]
            else:
                raise ValueError(f"Unsupported embedding shape for sequence {seq_id}: {embedding.shape}")
        return selected

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
        embedding_method = self.config.embedding_method

        tokenizer = AutoTokenizer.from_pretrained(esm_model, do_lower_case=False)
        model = AutoModel.from_pretrained(esm_model)
        model.eval()

        mutation_sites = self.get_mutation_sites(seq_ids)
        seq_idx_to_embedding = {}

        for idx, idx_mutation_sites in zip(seq_ids, mutation_sites):
            prot_seq = self.sequence_to_protein_sequence[idx]
            layer_embeddings = embed_sequence(prot_seq, tokenizer, model,
                                             embedding_method=embedding_method,
                                             mutation_sites=idx_mutation_sites,
                                             pool_mutations=self.config.per_residue_mutation_pooling)
            seq_idx_to_embedding[idx] = layer_embeddings

        if self._use_memory():
            self.sequence_to_embeddings.update(seq_idx_to_embedding)

        # Save the embeddings to the specified path if it exists
        if out_path is not None:
            self._save_pickle(seq_idx_to_embedding, Path(out_path))
        return seq_idx_to_embedding
    

    def embed_all_sequences(self, layer: str = "all") -> None:
        """
        Embed all sequences in the sequence dataframe and save the embeddings to disk if specified in the configuration.
        """
        if self.sequence_dataframe is None:
            raise ValueError("Sequence dataframe is not available. Please run process_raw_data() first.")

        seq_ids = sorted(self.sequence_dataframe['SequenceIndex'].unique())
        embeddings = self.embed_sequences(seq_ids)

        n_layers = embeddings[seq_ids[0]].shape[0]
        if layer == 'all':
            for l in range(n_layers):
                layer_embeddings = self._select_layer(embeddings, l)
                if self._use_memory():
                    self.sequence_to_features[self._embedding_key(l)] = layer_embeddings
                if self._use_disk():
                    save_path = self._embedding_path(l)
                    self._save_pickle(layer_embeddings, save_path)
        else:
            layer_embeddings = self._select_layer(embeddings, layer)
            if self._use_memory():
                self.sequence_to_features[self._embedding_key(layer)] = layer_embeddings
            if self._use_disk():
                save_path = self._embedding_path(layer)
                self._save_pickle(layer_embeddings, save_path)
        


    def load_embeddings(self, layer: str) -> dict[str, np.ndarray]:
        """
        Load embeddings from disk if they exist.

        Returns:
        --------
        dict[str, np.ndarray] | None
            A dictionary mapping sequence indices to their corresponding embeddings, or None if no saved embeddings are found.
        """

        key = self._embedding_key(layer)
        if self._use_memory() and self.sequence_to_features.get(key) is not None:
            print("Embeddings already exist in memory. Returning existing embeddings.")
            return self.sequence_to_features[key]

        if self._use_memory() and self.sequence_to_embeddings:
            layer_embeddings = self._select_layer(self.sequence_to_embeddings, layer)
            self.sequence_to_features[key] = layer_embeddings
            if self._use_disk():
                self._save_pickle(layer_embeddings, self._embedding_path(layer))
            return layer_embeddings

        if self._use_disk():
            save_path = self._embedding_path(layer)
            if save_path.is_file():
                print(f"Loading embeddings from {save_path}")
                embeddings = self._load_pickle(save_path)
                if self._use_memory():
                    self.sequence_to_features[key] = embeddings
                return embeddings
            raise FileNotFoundError(f"No saved embeddings found at {save_path}. Run embed_all_sequences({layer!r}) first.")

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
                            method: Literal['Embeddings', 'PCA', 'SAE', 'SPCA'] = 'Embeddings',
                            method_params: dict | None = None) -> dict[str, np.ndarray]:
        """
        Create an abstraction of the embeddings using the specified method.

        Parameters:
        -----------
        layer : str
            The layer from which to extract features.
        method : Literal['Embeddings', 'PCA', 'SAE', 'SPCA']
            The method to use for creating the abstraction.
        method_params : dict | None
            Parameters for the abstraction method.

        Returns:
        --------
        dict[str, np.ndarray]
            A dictionary mapping sequence indices to their corresponding features.
        """
        if method == 'Embeddings':
            print("No abstraction method specified. Using raw embeddings as features.")
            return self.load_embeddings(layer)

        embeddings = self.load_embeddings(layer)
        key = self._feature_key(method, layer)

        if self._use_memory() and self.sequence_to_features.get(key) is not None:
            print("Abstracted features already exist in memory. Returning existing features.")
            return self.sequence_to_features[key]

        if self._use_disk():
            save_path = self._feature_path(method, layer)
            if save_path.is_file():
                print(f"Abstracted already saved in {save_path}")
                abstracted_features = self._load_pickle(save_path)
                if self._use_memory():
                    self.sequence_to_features[key] = abstracted_features
                return abstracted_features

        print("No abstracted features found. Creating new features.")
        abstracted_features = self._create_feature_space(embeddings, method, method_params)
        if self._use_memory():
            self.sequence_to_features[key] = abstracted_features
        if self._use_disk():
            self._save_pickle(abstracted_features, self._feature_path(method, layer))
        return abstracted_features


    def _create_feature_space(self, embeddings: dict[str, np.ndarray], 
                        method: Literal['Embeddings', 'PCA', 'SAE', 'SPCA'] = 'Embeddings',
                        method_params: dict | None = None) -> dict[str, np.ndarray]:
        # Placeholder for abstraction implementation
        if method == 'Embeddings':
            return embeddings
        elif method == 'PCA':
            # Implement PCA abstraction here
            return self._pca_abstraction(embeddings, method_params)
        elif method == 'SAE':
            # Implement SAE abstraction here
            return self._sae_abstraction(embeddings, method_params)
        elif method == 'SPCA':
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
        raise NotImplementedError("SAE abstraction is not implemented yet.")


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
    

    def _load_abstracted_features(self, layer: str, 
                                    method: Literal['Embeddings', 'PCA', 'SAE', 'SPCA'], 
                                    method_params: dict | None) -> dict[str, np.ndarray]:
        """
        Load abstracted features from disk if they exist.

        Parameters:
        -----------
        layer : str
            The layer from which to extract features.
        method : Literal['Embeddings', 'PCA', 'SAE', 'SPCA']
            The method used for abstraction.
        method_params : dict | None
            Parameters for the abstraction method.

        Returns:
        --------
        dict[str, np.ndarray] | None
            A dictionary mapping sequence indices to their corresponding abstracted features, or None if no saved features are found.
        """
        if method == 'Embeddings':
            print("No abstraction method specified. Using raw embeddings as features.")
            return self.load_embeddings(layer)

        key = self._feature_key(method, layer)
        if self._use_memory() and self.sequence_to_features.get(key) is not None:
            print("Abstracted features already exist in memory. Returning existing features.")
            return self.sequence_to_features[key]

        if self._use_disk():
            save_path = self._feature_path(method, layer)
            if save_path.is_file():
                print(f"Loading abstracted features from {save_path}")
                abstracted_features = self._load_pickle(save_path)
                if self._use_memory():
                    self.sequence_to_features[key] = abstracted_features
                return abstracted_features

        return self.create_feature_space(layer, method, method_params)

    def _normalize_features(self, features, norm_scheme):
        if norm_scheme == "cross_feature":
            return (features - np.mean(features)) / (np.std(features) + 1e-8)
        if norm_scheme == "per_feature":
            return (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        if norm_scheme == "none":
            return features
        raise ValueError(f"Unsupported normalization scheme: {norm_scheme}")

    #TODO: add parameters for model type (linear, non-linear) and regularization scheme (L2, L1, ElasticNet)
    def run_feature_inference(self, layer: str,
                              abstraction_method: Literal['Embeddings', 'PCA', 
                                                          'SAE', 'SPCA'],
                              abstraction_params: dict | None = None,
                              save_results: bool = False) -> dict: 
        """
        Run the abstracted features through the popDMS framework to calculate selection coefficients and fitness.

        Parameters:
        -----------
        layer : str | None
            The layer from which to extract features.
        abstraction_method : Literal['Embeddings', 'PCA', 'SAE', 'SPCA']
            The method used for feature abstraction.
        abstraction_params : dict | None
            Parameters for the abstraction method.

        Returns:
        --------
        dict
            A dictionary containing inferred selection coefficients and fitness values.
        """

        if self.sequence_dataframe is None:
            raise ValueError("Sequence dataframe is not available. Please run process_raw_data() first.")

        abstraction_params = abstraction_params or {}
        norm_scheme = abstraction_params.get('norm_scheme', 'none')
        if self._use_disk():
            # check if the features are already saved to disk
            save_path = self._inference_path(abstraction_method, layer, norm_scheme)
            if save_path.is_file():
                print(f"Loading inference results from {save_path}")
                return self._load_pickle(save_path)
            else:
                print(f"No saved inference results found at {save_path}. Running inference and saving results.")
        
        # load features, seq_to_features type = dict[str, np.ndarray]
        seq_to_features = self._load_abstracted_features(layer, abstraction_method, abstraction_params)
        if norm_scheme is not None and norm_scheme != "none":
            seq_ids = list(seq_to_features)
            features = np.asarray([seq_to_features[seq_id] for seq_id in seq_ids])
            features = self._normalize_features(features, norm_scheme)
            seq_to_features = dict(zip(seq_ids, features))

        inf_result = mini_infer_esm(self.sequence_dataframe, seq_to_features)
        if save_results:
            self._save_inference_results(inf_result, layer, 
                                         abstraction_method, norm_scheme)
        return inf_result


    def _save_inference_results(self, results: dict, layer: str, abstraction_method: str, norm_scheme: str) -> None:
        if self.config.save_dir is None:
            raise ValueError("save_dir must be specified in the configuration to save inference results to disk.")
        
        save_path = self._inference_path(abstraction_method, layer, norm_scheme)
        self._save_pickle(results, save_path)
    
