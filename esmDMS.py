from __future__ import annotations

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


EmbeddingModel = Literal[
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
]

EmbeddingMethod = Literal["mean_pooling", "per_residue", "cls_token"]


@dataclass(frozen=True)
class ESMDMSConfig:
    embedding_model: EmbeddingModel = "esm2_t33_650M_UR50D"
    embedding_method: EmbeddingMethod = "mean_pooling"
    per_residue_mutation_pooling: bool = False


@dataclass(frozen=True)
class CellularDMSInput:
    reference_nuc_path: Path
    mavedb_csv_path: Path
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


    def embed_sequences(self, seq_ids: list[str], out_path: str | None = None) -> np.ndarray:
        """
        Embed protein sequences using the specified ESM model and embedding method.

        Parameters:
        -----------
        seq_ids : list[str]
            A list of sequence indices to embed.
        out_path : str | None
            The path to save the embeddings.

        Returns:
        --------
        np.ndarray
            An array of embeddings corresponding to the input sequences.
        """
        esm_model = f"facebook/esm2_{self.config.embedding_model}"
        embedding_method = self.config.embedding_method

        tokenizer = AutoTokenizer.from_pretrained(esm_model, do_lower_case=False)
        model = AutoModel.from_pretrained(esm_model)
        model.eval()
        sequence_embeddings = []

        mutation_sites = self.get_mutation_sites(seq_ids)
        seq_idx_to_embedding = {}

        for idx in seq_ids:
            layer_embeddings = embed_sequence(idx, tokenizer, model,
                                             embedding_method=embedding_method,
                                             mutation_sites=mutation_sites,
                                             per_residue_mutation_pooling=self.config.per_residue_mutation_pooling)
            seq_idx_to_embedding[idx] = layer_embeddings

        # Save the embeddings to the specified path if it exists
        if out_path is not None:
            np.save(out_path, seq_idx_to_embedding)
        return seq_idx_to_embedding
        

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


    def create_feature_space(self, embeddings: dict[str, np.ndarray], 
                            method: Literal['None', 'PCA', 'SAE', 'SPCA'] = 'None',
                            method_params: dict | None = None) -> dict[str, np.ndarray]:
        """
        Create an abstraction of the embeddings using the specified method.

        Parameters:
        -----------
        embeddings : dict[str, np.ndarray]
            A dictionary mapping sequence indices to their corresponding embeddings.
        method : Literal['None', 'PCA', 'SAE', 'SPCA']
            The method to use for creating the abstraction.
        method_params : dict | None
            Parameters for the abstraction method.

        Returns:
        --------
        dict[str, np.ndarray]
            A dictionary mapping sequence indices to their corresponding abstracted features.
        """
        # Placeholder for abstraction implementation
        if method == 'None':
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
        pass


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
        pass


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
        pass
    
    #TODO: add parameters for model type (linear, non-linear) and regularization scheme (L2, L1, ElasticNet)
    def run_feature_inference(self, features: np.ndarray, layer: str,
                              abstraction_method: Literal['None', 'PCA', 'SAE', 'SPCA'] = 'None') -> dict: 
        """
        Run the abstracted features through the popDMS framework to calculate selection coefficients and fitness.

        Parameters:
        -----------
        features : np.ndarray
            An array of abstracted features corresponding to the input sequences.

        Returns:
        --------
        dict
            A dictionary containing inferred selection coefficients and fitness values.
        """
        pass


    

        

    
    