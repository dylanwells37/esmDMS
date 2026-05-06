import ast
import os
import time
import sys
import gc
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
import pickle
from scipy.stats import pearsonr, rankdata, spearmanr
import matplotlib.pyplot as plt

from popDMS import (mini_infer_independent_esm, infer_gamma_range, mini_infer_fullcov_esm)

#import torch
#from transformers import AutoModel, AutoTokenizer
#from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
#from sklearn.metrics import silhouette_score
#import shutil
#from matplotlib import rc

## GLOBAL VARIABLES
pwd = "/net/dali/home/barton/dhw28/popDMS/esmDMS"
#default_emb_path = pwd + '/data/sequence_data/all_reps_BF520_protein_embeddings.pkl'
default_emb_path = "/net/dali/home/barton/dhw28/popDMS/esmDMS/data/inference_results"
# Pick an ESM-2 model size
"""model_name = "facebook/esm2_t30_150M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
model = AutoModel.from_pretrained(model_name)"""

CODON2AA = {'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',            # Map from codons to amino acids
            'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
            'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
            'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
            'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
            'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
            'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
            'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
            'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
            'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
            'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
            'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
            'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
            'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
            'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
            'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W' }

## FUNCTIONS


def embedding_df_transfer_optimized(embed_df: pd.DataFrame) -> pd.DataFrame:
    """
    An optimized function to transform the embeddings DataFrame.
    It avoids repeated concatenation by building a list of records first.
    
    Format of embed_df:
         PreNums   PostNums   ProteinSequence  Embeddings
    0  [0, 0, 0]  [0, 0, 0]   MKT...           [[...], [...], ...]
    

    Format of RepNDataFrame:
    generation, embedding, frequency, replicate
    
    
    Key Differences:
    We will not have sites and amino acids. Instead, we will have 
    the N embedding dimensions 
    """
    
    # --- Step 1: Extract data into efficient structures ---
    # Using .to_list() and np.array is much faster than accessing pandas Series
    # elements one-by-one inside the loops.
    pre_counts = np.array(embed_df["PreNums"].to_list())
    post_counts = np.array(embed_df["PostNums"].to_list())
    embeddings = embed_df["Embeddings"].to_list()
    
    num_variants = len(embeddings)
    if num_variants == 0:
        return pd.DataFrame(columns=["Generation", "Embedding", "Frequency", "Replicate"])
    
    num_reps = pre_counts.shape[1]
    
    # --- Step 2: Build a list of records (dictionaries) ---
    # Appending to a list is vastly more efficient than concatenating DataFrames.
    data_list = []
    
    # --- Step 3: Loop and append records to the list ---
    # This logic is identical to your original function.
    for i in range(num_variants):
        # Skip rows where the embedding is missing
        if embeddings[i] is None:
            continue
            
        for rep in range(num_reps):
            # --- Handle Generation 0 (pre-selection) ---
            pre_freq = pre_counts[i, rep]
            
            # This is the direct translation of your original condition:
            # `if gen == 0 and freq == 0: continue`
            if pre_freq > 0:
                data_list.append({
                    "Generation": 0,
                    "Embedding": embeddings[i],
                    "Frequency": pre_freq,
                    "Replicate": rep + 1
                })

            # --- Handle Generation 1 (post-selection) ---
            # In your original logic, the post-selection row is always added,
            # so we do the same here.
            post_freq = post_counts[i, rep]
            data_list.append({
                "Generation": 1,
                "Embedding": embeddings[i],
                "Frequency": post_freq,
                "Replicate": rep + 1
            })
            
    # --- Step 4: Create the DataFrame in a single, efficient operation ---
    if not data_list:
        return pd.DataFrame(columns=["Generation", "Embedding", "Frequency", "Replicate"])
        
    return pd.DataFrame(data_list)


def run_inference_calcs(layer_df, output_path, verbose=False, pre_processed=False,
                        n_replicates=3):
    """Run the inference calculations for the layer-specific dataframe"""
    if pre_processed:
        inference_df = layer_df
    else:
        inference_df = embedding_df_transfer_optimized(layer_df)  

        
    data = mini_infer_independent_esm(inference_df, n_replicates=n_replicates,
                                            output_dir=output_path, verbose=verbose)
    return data # data = [dx, icov, s, s_joint, sel_data, gamma_opt, x_array]
    
    
    

def analyze_layers(whole_df, output_path=None, embed_path=None, dataname='BF520',
                   verbose=False):
    """
    WHOLE PIPELINE, DF -> LAYER RESULTS
    
    Input:
    whole_df: dataframe with embeddings for all layers
    output_path: path to save the inference results
    embed_path: path to save the layer-specific embedding dataframes
    dataname: name of the dataset for saving files
    
    Output:
    data = [dx, icov, s, s_joint, sel_data, gamma_opt, x_array] for layer
    output as a list of data for each layer"""
    nonzero_df = whole_df[whole_df["Embeddings"].notnull()].reset_index(drop=True)
    layer_count = nonzero_df["Embeddings"][0].shape[0]
    
    layer_dfs = []
    for layer in range(layer_count):
        layer_df = nonzero_df.copy()
        layer_df["Embeddings"] = layer_df["Embeddings"].apply(lambda x: x[layer])
        layer_dfs.append(layer_df)
        
        if embed_path is None:
            continue
        
        if not os.path.exists(embed_path):
            os.makedirs(embed_path)
        
        # only make the pickle file if it doesn't exist
        if not os.path.exists(embed_path + f'{dataname}_protein_embeddings_layer{layer}.pkl'):
            layer_df.to_pickle(embed_path + f'{dataname}_protein_embeddings_layer{layer}.pkl')
        #print(f"Wrote layer {layer} dataframe to pickle")
    
    layer_results = []
    for layer in range(layer_count):
        if verbose:
            print(f"Analyzing layer {layer}")
        layer_df = layer_dfs[layer]
        
        if output_path is None:
            layer_path = None
        else:
            layer_path = output_path + f'layer{layer}/'
        # data = [dx, icov, s, s_joint, sel_data, gamma_opt, x_array] for layer
        data = run_inference_calcs(layer_df, layer_path, verbose=verbose) 
        layer_results.append(data)
        
        if output_path is None:
            continue
        
        if not os.path.exists(layer_path):
            os.makedirs(layer_path)
        with open(layer_path + 'inference_results.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        if verbose:
            print(f"Wrote inference results for layer {layer}")
        
    return layer_results


def get_unique_df(filepath=default_emb_path):
    """Get the protein dataframe from the embedding pickle file,
    combining all entries with the same protein sequence"""
    
    whole_df = pd.read_pickle(filepath)

    # Combine the prenums and postnums of any entries with the same potein sequence
    whole_df['PreNums'] = whole_df['PreNums'].apply(lambda x: np.array(x))
    whole_df['PostNums'] = whole_df['PostNums'].apply(lambda x: np.array(x))
    whole_df = whole_df.groupby('ProteinSequence').agg({
        'PreNums': lambda x: np.sum(x.tolist(), axis=0),
        'PostNums': lambda x: np.sum(x.tolist(), axis=0),
        'Embeddings': 'first'
    }).reset_index()

    whole_df = whole_df[whole_df["Embeddings"].notnull()].reset_index(drop=True)

    return whole_df


def name_to_path(name):
    """Map a dataset name to the corresponding path containing the layer subdirectories with inference_df.pkl files"""
    name_path_map = {
        "BF520": "/net/dali/home/barton/dhw28/popDMS/esmDMS/data/inference_results",
        "BG505": "/net/dali/home/barton/dhw28/popDMS/esmDMS/data/inference_data/BG505",
        # add more mappings as needed
    }
    if name not in name_path_map:
        raise ValueError(f"Unknown dataset name '{name}'. Available names: {list(name_path_map.keys())}")
    
    return name_path_map.get(name, None)



def get_layer_dataframes(names, layers=list(range(31)), normalize=True, in_paths=None):
    """
    Retrieve and combine layer-specific dataframes from the specified input paths.

    Input:
    in_paths:     list of paths containing layer subdirectories with inference_df.pkl files
    layer_count:  number of layers to retrieve (default: 31)
    normalize:    whether to z-normalize embeddings dimension-wise (default: True)

    Output:
    layer_dfs: list of dataframes, one per layer, with embeddings optionally normalized
    """
    if in_paths is None:
        in_paths = []
        for name in names:
            in_paths.append(name_to_path(name))
            
    layer_dfs = []

    for layer in layers:
        # Load dataframes for this layer from each input path
        layer_dfs_layer = []
        for in_path in in_paths:
            layer_df = pickle.load(open(f"{in_path}/layer{layer}/inference_df.pkl", 'rb'))
            layer_dfs_layer.append(layer_df)

        # Combine dataframes across paths, offsetting replicate indices to avoid collisions
        layer_df = layer_dfs_layer[0].copy()
        num_reps = len(layer_df["Replicate"].unique())

        for path_idx in range(1, len(layer_dfs_layer)):
            df_to_add = layer_dfs_layer[path_idx].copy()
            df_to_add["Replicate"] = df_to_add["Replicate"] + path_idx * num_reps
            layer_df = pd.concat([layer_df, df_to_add], ignore_index=True)
            del df_to_add

        del layer_dfs_layer
        gc.collect()
            
        # Z-normalize each embedding dimension independently across all samples
        if normalize:
            embeddings = np.array(layer_df["Embedding"].to_list())
            for dim in range(embeddings.shape[1]):
                embeddings[:, dim] = z_normalize(embeddings[:, dim])
            layer_df["Embedding"] = [embeddings[i] for i in range(embeddings.shape[0])]
            del embeddings
            gc.collect()

        
        layer_dfs.append(layer_df)

    return layer_dfs

def get_layer_df_processed(names, layer, in_paths=None):
    """Return a processed layer dataframe for a specific layer."""
    if in_paths is None:
        in_paths = [name_to_path(name) for name in names]
    combined_df = None
    for in_path in in_paths:
        layer_df = pickle.load(open(f"{in_path}/layer{layer}/final_df.pkl", 'rb'))
        if len(in_paths) == 1:
            return layer_df
        else:
            if in_path == in_paths[0]:
                combined_df = layer_df.copy()
            else:
                df_to_add = layer_df.copy()
                num_reps = len(df_to_add["Replicate"].unique())
                df_to_add["Replicate"] = df_to_add["Replicate"] + num_reps
                combined_df = pd.concat([combined_df, df_to_add], ignore_index=True)
                del df_to_add
                gc.collect()
    return combined_df


def analyze_layers_cross_variant(whole_df=None, in_paths=None, output_path=None, 
                                 embed_path=None, dataname='BF520', verbose=False,
                                 normalize=True):
    """
    WHOLE PIPELINE, DF -> LAYER RESULTS
    
    Input:
    whole_df: dataframe with embeddings for all layers
    output_path: path to save the inference results
    embed_path: path to save the layer-specific embedding dataframes
    dataname: name of the dataset for saving files
    
    Output:
    data = [dx, icov, s, s_joint, sel_data, gamma_opt, x_array] for layer
    output as a list of data for each layer"""
    #nonzero_df = whole_df[whole_df["Embeddings"].notnull()].reset_index(drop=True)
    layer_count = 31 #nonzero_df["Embeddings"][0].shape[0]
    
    layer_dfs = []
    for layer in range(layer_count):
        layer_dfs_layer = []
        if in_paths is not None:
            for in_path in in_paths:
                layer_df = pickle.load(open(f"{in_path}/layer{layer}/inference_df.pkl", 'rb'))
                layer_dfs_layer.append(layer_df)
        
        # Construct the layer_df by concatenating on the embedding dimension.
        layer_df = layer_dfs_layer[0].copy()
        num_reps = len(layer_df["Replicate"].unique())
        
        total_paths = len(layer_dfs_layer)
        for path_idx in range(1, total_paths):
            # add num_reps * total_paths to the replicate number in the new dataframe
            df_to_add = layer_dfs_layer[path_idx].copy()
            df_to_add["Replicate"] = df_to_add["Replicate"] + path_idx * num_reps
            layer_df = pd.concat([layer_df, df_to_add], ignore_index=True)
        
        if normalize:
            embeddings = np.array([x for x in layer_df["Embedding"].to_list()])
            dimensions = embeddings.shape[1]
            for dim in range(dimensions):
                embeddings[:, dim] = z_normalize(embeddings[:, dim])
            layer_df["Embedding"] = [embeddings[i] for i in range(embeddings.shape[0])]
            layer_dfs.append(layer_df)
    
    layer_results = []
    for layer in range(layer_count):
        if verbose:
            print(f"Analyzing layer {layer}")
        layer_df = layer_dfs[layer]
        
        layer_path = None
        
        num_reps = len(layer_df["Replicate"].unique())
        print(f"Layer {layer} has {num_reps} replicates after combining datasets.")
        #print(layer_df.head())
        # data = [dx, icov, s, s_joint, sel_data, gamma_opt, x_array] for layer
        data = run_inference_calcs(layer_df, layer_path, verbose=verbose,
                                   pre_processed=True, n_replicates=num_reps) 
        layer_results.append(data)
        
    return layer_results


def z_normalize(array: np.ndarray) -> np.ndarray:
    """Z-normalize the input numpy array."""
    mean = np.mean(array)
    std = np.std(array)
    if std == 0:
        return array - mean
    return (array - mean) / std


## SIMULATION FUNCTION ##################################

# Now, let's calculate the fitness for each variant based on its embedding and the selection coefficients
def calculate_fitness_exp(embedding, selection_coefficients):
    fitness = np.exp(np.dot(embedding, selection_coefficients))
    if np.isinf(fitness):
        fitness = 1e10  # Cap infinite fitness to a large number
    return fitness

def calc_all_fitness_exp(embeddings, selection_coefficients,
                         embedding_clip=None):
    fitnesses = []
    for embedding in embeddings:
        if embedding_clip is not None:
            embedding = np.clip(embedding, embedding_clip[0], embedding_clip[1])
        fitness = calculate_fitness_exp(embedding, selection_coefficients)
        fitnesses.append(fitness)
    return np.array(fitnesses)

def calculate_fitness_plus1(embedidng, selection_coefficients):
    fitness = 1 + np.dot(embedidng, selection_coefficients)
    return max(fitness, 0)  # Ensure fitness is not negative

def calc_all_fitness_plus1(embeddings, selection_coefficients,
                            embedding_clip=None):
    fitnesses = []
    for embedding in embeddings:
        if embedding_clip is not None:
            embedding = np.clip(embedding, embedding_clip[0], embedding_clip[1])
        fitness = calculate_fitness_plus1(embedding, selection_coefficients)
        fitnesses.append(fitness)
    return np.array(fitnesses)

def simulate_generation_multinomial(current_counts, fitnesses):
    population_size = np.sum(current_counts)
    total_fitness = np.sum(current_counts * fitnesses)
    probabilities = (current_counts * fitnesses) / total_fitness
    # if a probability is below 0, set it to zero
    # if a probability is above 1, set it to 1
    probabilities = np.clip(probabilities, 0, 1)
    next_counts = np.random.multinomial(population_size, probabilities)
    return next_counts # Check if output is a different scale


def get_df_selection(init_pop, random_seed=42, 
                     selected_layer=12, normalize_embeddings=True, 
                     processed_df=None):
    """Get the starting information for the simulation,
    i.e. the layer dataframe and the initial counts for each replicate.
    
    Args:
        processed_df: Optional pre-built dataframe containing Rep1/2/3_PreNums,
                  Rep1/2/3_PostNums, and an 'Embedding' column for the selected
                  layer. If provided, data loading and layer decomposition are
                  skipped entirely.
    """
    pd.set_option('display.max_columns', 100)
    np.random.seed(random_seed)

    if processed_df is not None:
        # Expect processed_df to already have Rep{1,2,3}_Pre/PostNums and 'Embedding'
        required_cols = [
            'Rep1_PreNums', 'Rep2_PreNums', 'Rep3_PreNums',
            'Rep1_PostNums', 'Rep2_PostNums', 'Rep3_PostNums',
            'Embedding'
        ]
        missing = [c for c in required_cols if c not in processed_df.columns]
        if missing:
            raise ValueError(f"processed_df is missing required columns: {missing}")
        df_selection = processed_df.copy()

    else:
        n_reps = init_pop['PreNums'].iloc[0].shape[0]

        replicate_dfs = []
        for rep in range(n_reps):
            df_rep = init_pop.copy()
            df_rep['PreNums']  = df_rep['PreNums'].apply(lambda x: x[rep])
            df_rep['PostNums'] = df_rep['PostNums'].apply(lambda x: x[rep])
            df_rep = df_rep.drop(columns=['ProteinSequence'])
            replicate_dfs.append(df_rep)

        # Only decompose the selected layer — skip the full loop
        decomposed_dfs = []
        for rep_df in replicate_dfs:
            decomposed_df = rep_df[['PreNums', 'PostNums']].copy()
            decomposed_df[f'Layer{selected_layer}'] = rep_df['Embeddings'].apply(
                lambda x: x[selected_layer]
            )
            decomposed_dfs.append(decomposed_df)

        df_selection = pd.DataFrame()
        for i, decomposed_df in enumerate(decomposed_dfs):
            df_selection[f'Rep{i+1}_PreNums']  = decomposed_df['PreNums']
            df_selection[f'Rep{i+1}_PostNums'] = decomposed_df['PostNums']
        df_selection['Embedding'] = decomposed_dfs[-1][f'Layer{selected_layer}']

    if normalize_embeddings:
        embeddings = np.vstack(df_selection['Embedding'].values)
        dims = embeddings.shape[1]
        for dim in range(dims):
            embeddings[:, dim] = z_normalize(embeddings[:, dim])
        df_selection['Embedding'] = list(embeddings)

    start_counts1 = df_selection["Rep1_PreNums"].values
    start_counts2 = df_selection["Rep2_PreNums"].values
    start_counts3 = df_selection["Rep3_PreNums"].values

    return df_selection, (start_counts1, start_counts2, start_counts3)


def run_simulation(df_selection, selection_coefficients, initial_counts,
                   n_gens=30, embedding_clip=None, save_every=1, fitness='exp',
                   plateau_window=3, plateau_rtol=1e-3):
    """Run a Wright-Fisher multinomial simulation for up to n_gens generations.

    Early stopping: after each generation the mean fitness is averaged across all
    replicates.  Once the relative change in that mean is below plateau_rtol for
    plateau_window consecutive saved time-points, the simulation halts early and
    only the generations up to that point are returned.

    Parameters
    ----------
    plateau_window : int
        Number of consecutive saved generations whose mean-fitness change must
        all fall below plateau_rtol before stopping.  Set to 0 to disable early
        stopping entirely.
    plateau_rtol : float
        Relative-change threshold: |Δw̄| / w̄ < plateau_rtol triggers a plateau
        count increment.
    """
    embeddings = np.vstack(df_selection['Embedding'].values)

    if fitness == 'exp':
        fitnesses = calc_all_fitness_exp(embeddings, selection_coefficients,
                                        embedding_clip=embedding_clip)
    elif fitness == 'plus1':
        fitnesses = calc_all_fitness_plus1(embeddings, selection_coefficients,
                                        embedding_clip=embedding_clip)
    else:
        raise ValueError("Invalid fitness function specified.")

    generation_counts = [initial_counts]
    n_reps = len(initial_counts)
    last_counts = initial_counts

    # Track mean fitness of the most recently saved snapshot for plateau detection
    prev_mean_fitness = None
    plateau_streak = 0

    for gen in range(n_gens):
        this_gen = []
        gen_mean_fitness = 0.0
        for rep in range(n_reps):
            rep_counts = last_counts[rep]
            next_counts = simulate_generation_multinomial(rep_counts, fitnesses)
            this_gen.append(next_counts)
            total = np.sum(next_counts)
            if total > 0:
                gen_mean_fitness += np.dot(next_counts, fitnesses) / total
        gen_mean_fitness /= n_reps

        last_counts = this_gen

        if (gen + 1) % save_every == 0:
            generation_counts.append(this_gen)

            # Plateau detection
            if plateau_window > 0 and prev_mean_fitness is not None:
                rel_change = abs(gen_mean_fitness - prev_mean_fitness) / max(abs(prev_mean_fitness), 1e-12)
                if rel_change < plateau_rtol:
                    plateau_streak += 1
                    if plateau_streak >= plateau_window:
                        break
                else:
                    plateau_streak = 0
            prev_mean_fitness = gen_mean_fitness

    return generation_counts, fitnesses


def simulation_df_transfer(df_selection, generation_counts):
    #["Generation", "Embedding", "Frequency", "Replicate"]
    emb_vals = df_selection["Embedding"].values
    #generation counts format [[gen1data], [gen2data],...]
    data_list = []
    for gen, gen_data in enumerate(generation_counts):
        for rep, rep_data in enumerate(gen_data):
            for i, count in enumerate(rep_data):
                data_list.append({
                    "Generation": gen,
                    "Embedding": emb_vals[i],
                    "Frequency": count,
                    "Replicate": rep + 1
                })
    return pd.DataFrame(data_list)

def simulation_df_transfer_vectorized(df_selection, generation_counts):
    # Convert nested lists to a 3D NumPy array
    counts = np.asarray(generation_counts)
    
    # Get dimensions: Generations (G), Replicates (R), Embeddings (E)
    G, R, E = counts.shape
    
    emb_vals = df_selection["Embedding"].values
    
    # Build columns using vectorization
    # gen changes slowest, rep changes next, emb changes fastest
    generations = np.repeat(np.arange(G), R * E)
    replicates = np.tile(np.repeat(np.arange(1, R + 1), E), G)
    embeddings = np.tile(emb_vals, G * R)
    frequencies = counts.ravel() # Flattens the array efficiently
    
    # Construct DataFrame from a dictionary of flat arrays
    return pd.DataFrame({
        "Generation": generations,
        "Embedding": embeddings,
        "Frequency": frequencies,
        "Replicate": replicates
    })


def run_inference_calcs_sims(df_selection, generation_counts, output_path=None,
                             save_output=False, calc_error_bars=False, variance_cutoff=0.0,
                             infer_ignored_dims=True, method='independent'):
    """Run the inference calculations:
    WHOLE PIPELINE FROM READING IN EMBEDDINGS DATAFRAME

    Parameters
    ----------
    method : str
        Which inference method to use:
        'independent' - original allele-frequency covariance approximation
        'diagonal'    - per-dimension population variance (Approach 1, no matrix inversion)
        'fullcov'     - full per-sequence covariance matrix (Approach 2, guaranteed PSD)
    """
    print("Transferring simulation data to inference dataframe format...")
    inference_df = simulation_df_transfer(df_selection, generation_counts)

    if save_output:
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            inference_df.to_pickle(output_path + 'inference_df.pkl')
            print(f"SAVED INFERENCE DF TO {output_path}")

    print(f"Running inference calculations on simulated data (method='{method}')...")
    n_replicates = len(inference_df["Replicate"].unique())

    _infer_fns = {
        'independent': mini_infer_independent_esm,
        #'diagonal':    mini_infer_diagonal_esm,
        'fullcov':     mini_infer_fullcov_esm,
    }
    if method not in _infer_fns:
        raise ValueError(f"Unknown inference method '{method}'. Choose from {list(_infer_fns)}")

    data = _infer_fns[method](inference_df, n_replicates=n_replicates,
                              output_dir=output_path, verbose=False,
                              calc_error_bars=calc_error_bars,
                              variance_cutoff=variance_cutoff,
                              infer_ignored_dims=infer_ignored_dims)
    return data  # [dx, icov/ivar, s, s_joint, sel_data, gamma_opt, x_array, error_bars, s_joint_error_bars]


def run_gamma_analysis_sims(df_selection, generation_counts, output_path=None,
                             save_output=False, variance_cutoff=0.0, infer_ignored_dims=True):
    """Run the inference calculations:
    WHOLE PIPELINE FROM READING IN EMBEDDINGS DATAFRAME
    """
    print("Transferring simulation data to inference dataframe format...")
    inference_df = simulation_df_transfer(df_selection, generation_counts)

    print("Running inference calculations on simulated data...")
    n_replicates = len(inference_df["Replicate"].unique())
    data = infer_gamma_range(inference_df, n_replicates=n_replicates,
                             variance_cutoff=variance_cutoff,
                             infer_ignored_dims=infer_ignored_dims)
    return data # data = [gammas, s, s_joint]


def normalize_embeddings(df):
    #print(df.head())
    embedding_array = np.array([x for x in df["Embedding"].to_list()])
    #print(embedding_array.shape)
    z_embeddings = np.zeros_like(embedding_array)
    dimensions = embedding_array.shape[1]
    for dim in range(dimensions):
        z_embeddings[:, dim] = z_normalize(embedding_array[:, dim])
        
    # insert back into the dataframe
    df["Embedding"] = [z_embeddings[i] for i in range(z_embeddings.shape[0])]
    return df

def generate_one_selection(embeddings):
    selection_coefficients = np.zeros(embeddings.shape[1])
    embedding_ranges = embeddings.max(axis=0) - embeddings.min(axis=0)
    # Find the indices of the top, lowest, and middle range dimensions
    sorted_indices = np.argsort(embedding_ranges)
    high_range_idx = sorted_indices[-1]
    # Give these dimensions higher selection coefficients
    selection_coefficients[high_range_idx] = 0.10
    return selection_coefficients


## SELECTION FUNCTIONS ############
def gaussian_selection(embeddings_len):
    width = 0.02
    center = 0.0
    sel_coeffs = np.random.normal(loc=center, scale=width, size=embeddings_len)
    return sel_coeffs

def zero_selection(embeddings_len):
    return np.zeros(embeddings_len)

def generate_selection(embeddings_len):
    selection_coefficients = np.zeros(embeddings_len)
    random_index = np.random.choice(embeddings_len)
    selection_coefficients[random_index] = 0.01
    return selection_coefficients

#########################################

def _process_single_layer(args):
    """Worker for get_simulation_results — processes one ESM layer end-to-end.

    Must be a module-level function so it is picklable for ProcessPoolExecutor.
    """
    (layer, embedding_df_path, sel_func, n_gens, save_every, fitness,
     inference, gamma_analysis, calc_error_bars, variance_cutoff,
     infer_ignored_dims, method, plateau_window, plateau_rtol) = args

    print(f"Running layer {layer}...")
    df_selection = load_final_df(layer, embedding_df_path)
    n_reps = len(df_selection.columns) // 2
    initial_counts = []
    for rep in range(n_reps):
        pre_col = f'Rep{rep + 1}_PreNums'
        if pre_col not in df_selection.columns:
            raise ValueError(f"Expected column {pre_col} not found in df_selection: {df_selection.columns}")
        initial_counts.append(df_selection[pre_col].values)

    embeddings_len = df_selection['Embedding'].iloc[0].shape[0]
    selection_coefficients = sel_func(embeddings_len)

    print(f"  Layer {layer}: running simulation...")
    generation_counts, layer_fits = run_simulation(
        df_selection, selection_coefficients, initial_counts,
        n_gens=n_gens, save_every=save_every, fitness=fitness,
        plateau_window=plateau_window, plateau_rtol=plateau_rtol,
    )
    print(f"  Layer {layer}: simulation ran for {len(generation_counts) - 1} saved generations.")

    layer_results = None
    if inference:
        print(f"  Layer {layer}: running inference...")
        data = run_inference_calcs_sims(
            df_selection, generation_counts,
            calc_error_bars=calc_error_bars, variance_cutoff=variance_cutoff,
            infer_ignored_dims=infer_ignored_dims, method=method,
        )
        # [dx, icov/ivar, s, s_joint, sel_data, gamma_opt, x_array, error_bars, s_joint_error_bars]
        #icov_sum = np.sum(data[1], axis=0)
        layer_results = [data[2], data[3], data[7], data[8], data[1], data[5]]

    gamma_data = None
    if gamma_analysis:
        gamma_data = run_gamma_analysis_sims(
            df_selection, generation_counts,
            variance_cutoff=variance_cutoff, infer_ignored_dims=infer_ignored_dims,
        )

    return layer, layer_fits, selection_coefficients, layer_results, gamma_data, generation_counts


def get_simulation_results(n_gens, embedding_df_path=default_emb_path,
                           sel_func=generate_selection,
                           inference=True, gamma_analysis=True, fitness='plus1',
                           save_every=1, layers=[0], variance_cutoff=0.0,
                           calc_error_bars=False, infer_ignored_dims=True,
                           method='independent', n_jobs=1,
                           plateau_window=0, plateau_rtol=1e-3):
    """Run the whole pipeline of reading in the embedding dataframe, generating selection coefficients,
    running the simulation, and running inference calculations on the simulated data.

    Parameters
    ----------
    method : str
        Which inference method to use: 'independent', 'diagonal', or 'fullcov'.
    n_jobs : int
        Number of parallel worker processes for layer processing.
        1 = sequential (default).  -1 = all available CPUs.
        sel_func must be a module-level (picklable) function when n_jobs != 1.
    plateau_window : int
        Passed to run_simulation.  Number of consecutive saved generations with
        relative mean-fitness change below plateau_rtol before early stopping.
        Set to 0 to disable.
    plateau_rtol : float
        Relative-change threshold for plateau detection.
    """
    n_layers = 31

    valid_layers = [l for l in layers if l < n_layers]
    for l in layers:
        if l >= n_layers:
            print(f"Layer {l} is out of bounds (only {n_layers} layers available). Skipping.")

    args_list = [
        (layer, embedding_df_path, sel_func, n_gens, save_every, fitness,
         inference, gamma_analysis, calc_error_bars, variance_cutoff,
         infer_ignored_dims, method, plateau_window, plateau_rtol)
        for layer in valid_layers
    ]

    if n_jobs == 1:
        results = [_process_single_layer(a) for a in args_list]
    else:
        max_workers = None if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_process_single_layer, args_list))

    all_layer_fits = {}
    all_selection_coefficients = {}
    detailed_selection_results = {}
    all_gamma_analysis = {}
    all_generation_counts = {}

    for layer, layer_fits, sel_coeffs, layer_results, gamma_data, gen_counts in results:
        all_layer_fits[layer] = layer_fits
        all_selection_coefficients[layer] = sel_coeffs
        all_generation_counts[layer] = gen_counts
        if layer_results is not None:
            detailed_selection_results[layer] = layer_results
        if gamma_data is not None:
            all_gamma_analysis[layer] = gamma_data

    return all_layer_fits, all_selection_coefficients, detailed_selection_results, all_gamma_analysis, all_generation_counts


def get_eigenvector_simulation_results(n_gens, embedding_df_path=default_emb_path,
                                       sel_func=generate_selection,
                                       inference=True, gamma_analysis=True, fitness='plus1',
                                       save_every=1, layers=[0],
                                       variance_explained_cutoff=0.95,
                                       weight_by_initial_freq=False,
                                       calc_error_bars=False, infer_ignored_dims=True,
                                       method='independent',
                                       plateau_window=3, plateau_rtol=1e-3):
    """Run the simulation pipeline with embeddings projected into the eigenvector
    (PCA) basis of their covariance matrix.

    Parameters
    ----------
    n_gens : int
        Number of generations to simulate.
    embedding_df_path : str
        Path to the embedding data directory containing layer subdirectories.
    sel_func : callable
        Function that takes the number of dimensions and returns selection coefficients.
    inference : bool
        Whether to run popDMS inference on the simulated data.
    gamma_analysis : bool
        Whether to run gamma range analysis on the simulated data.
    fitness : str
        Fitness function to use ('plus1' or 'exp').
    save_every : int
        Save generation counts every this many generations.
    layers : list of int
        Which ESM-2 layers to process.
    variance_explained_cutoff : float
        Fraction of total variance that must be explained by the retained
        eigenvectors (e.g., 0.95 keeps enough PCs to explain 95% of variance).
        Set to 1.0 to keep all eigenvectors.
    weight_by_initial_freq : bool
        If True, weight the covariance matrix by the mean pre-selection counts
        across replicates. If False, use an unweighted covariance matrix.
    calc_error_bars : bool
        Whether to calculate error bars on inferred selection coefficients.
    infer_ignored_dims : bool
        Passed through to the inference function.
    method : str
        Which inference method to use: 'independent', 'diagonal', or 'fullcov'.
    plateau_window : int
        Passed to run_simulation.  Number of consecutive saved generations with
        relative mean-fitness change below plateau_rtol before early stopping.
        Set to 0 to disable.
    plateau_rtol : float
        Relative-change threshold for plateau detection.

    Returns
    -------
    all_layer_fits : dict
        layer -> array of per-variant fitness values used in the simulation.
    all_selection_coefficients : dict
        layer -> true selection coefficients used (in eigenvector space).
    detailed_selection_results : dict
        layer -> [s, s_joint, s_errors, s_joint_errors, icov_sum, gamma_opt] from inference.
    all_gamma_analysis : dict
        layer -> gamma analysis results.
    all_generation_counts : dict
        layer -> generation counts from the simulation.
    eigenvector_info : dict
        layer -> dict with keys 'eigenvalues', 'eigenvectors', 'n_components',
        'variance_explained', and 'projection_matrix'.
    """
    all_layer_fits = {}
    all_selection_coefficients = {}
    detailed_selection_results = {}
    all_generation_counts = {}
    all_gamma_analysis = {}
    eigenvector_info = {}

    n_layers = 31

    for layer in layers:
        if layer >= n_layers:
            print(f"Layer {layer} is out of bounds for the embeddings (only {n_layers} layers available). Skipping.")
            continue
        print(f"Running layer {layer}...")

        df_selection = load_final_df(layer, embedding_df_path)
        n_reps = len([c for c in df_selection.columns if 'PreNums' in c])
        initial_counts = [df_selection[f'Rep{rep + 1}_PreNums'].values for rep in range(n_reps)]

        # --- Build eigenvector basis ---
        embeddings = np.vstack(df_selection['Embedding'].tolist())  # (N, D)

        if weight_by_initial_freq:
            all_pre = np.stack(
                [df_selection[f'Rep{rep + 1}_PreNums'].values for rep in range(n_reps)],
                axis=1
            ).astype(float)  # (N, n_reps)
            weights = all_pre.mean(axis=1)
            weights = np.maximum(weights, 0.0)
            if weights.sum() == 0:
                weights = np.ones(len(embeddings))
            cov_matrix = np.cov(embeddings.T, aweights=weights)
        else:
            cov_matrix = np.cov(embeddings.T)

        # eigh returns eigenvalues in ascending order for a symmetric matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort descending so the most-variance-explaining PC comes first
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]  # columns are eigenvectors, shape (D, D)

        # Clip negative eigenvalues (numerical noise) to zero before computing variance fractions
        eigenvalues_pos = np.maximum(eigenvalues, 0.0)
        total_variance = eigenvalues_pos.sum()
        cumulative_variance = np.cumsum(eigenvalues_pos) / total_variance

        # Number of components needed to reach the cutoff
        n_components = int(np.searchsorted(cumulative_variance, variance_explained_cutoff) + 1)
        n_components = min(n_components, len(eigenvalues))
        n_components = max(n_components, 2)  # Ensure at least two components are kept

        variance_explained = cumulative_variance[n_components - 1]
        print(f"  Layer {layer}: keeping {n_components}/{len(eigenvalues)} eigenvectors "
              f"({variance_explained * 100:.1f}% variance explained)")

        projection_matrix = eigenvectors[:, :n_components]  # (D, k)
        projected_embeddings = embeddings @ projection_matrix  # (N, k)

        eigenvector_info[layer] = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'n_components': n_components,
            'variance_explained': variance_explained,
            'projection_matrix': projection_matrix,
        }

        # Replace embeddings with projected coordinates
        df_pca = df_selection.copy()
        df_pca['Embedding'] = [projected_embeddings[i] for i in range(len(projected_embeddings))]

        selection_coefficients = sel_func(n_components)
        all_selection_coefficients[layer] = selection_coefficients

        print("  Running simulation...")
        generation_counts, layer_fits = run_simulation(
            df_pca, selection_coefficients, initial_counts,
            n_gens=n_gens, save_every=save_every, fitness=fitness,
            plateau_window=plateau_window, plateau_rtol=plateau_rtol,
        )
        print(f"  Layer {layer}: simulation ran for {len(generation_counts) - 1} saved generations.")
        all_generation_counts[layer] = generation_counts
        all_layer_fits[layer] = layer_fits

        if inference:
            print(f"  Running inference for layer {layer} (method='{method}')...")
            data = run_inference_calcs_sims(
                df_pca, generation_counts,
                calc_error_bars=calc_error_bars,
                variance_cutoff=0.0,
                infer_ignored_dims=infer_ignored_dims,
                method=method,
            )
            icov_sum = np.sum(data[1], axis=0)  # (k, k) summed over replicates
            detailed_selection_results[layer] = [
                data[2],   # s
                data[3],   # s_joint
                data[7],   # s errors
                data[8],   # s_joint errors
                icov_sum,  # index 4: covariance matrix at optimal gamma
                data[5],   # index 5: gamma_opt
            ]

        if gamma_analysis:
            gamma_data = run_gamma_analysis_sims(
                df_pca, generation_counts,
                variance_cutoff=0.0,
                infer_ignored_dims=infer_ignored_dims
            )
            all_gamma_analysis[layer] = gamma_data

    return (all_layer_fits, all_selection_coefficients, detailed_selection_results,
            all_gamma_analysis, all_generation_counts, eigenvector_info)

## DEFINITIONS (DIRTY NOW, CLEAN UP LATER)

sim_folder = pwd + "/esm_sim_saves/"

def save_sim_data(sim_data, filename):
    if not os.path.exists(sim_folder):
        os.makedirs(sim_folder)
    with open(sim_folder + filename, 'wb') as f:
        pickle.dump(sim_data, f)
        
        
def load_sim_data(filename):
    with open(sim_folder + filename, 'rb') as f:
        sim_data = pickle.load(f)
    return sim_data

def load_final_df(layer, path=None):
    if path is None:
        path = "/net/dali/home/barton/dhw28/popDMS/esmDMS/data/bg_bf_comb_data"
    return pd.read_pickle(f"{path}/layer{layer}_sim_df.pkl")


def load_inference_df(layer, path="/net/dali/home/barton/dhw28/popDMS/esmDMS/data/inference_results"):
    return pd.read_pickle(f"{path}/layer{layer}_inference_df.pkl")


## ─────────────────────────────────────────────────────────────────────────────
## EIGENVECTOR / PCA ANALYSIS
## ─────────────────────────────────────────────────────────────────────────────

def _extract_embeddings_and_weights(layer_df: pd.DataFrame):
    """
    Return unique embeddings plus per-variant pre/post frequency weights.

    Each unique protein variant appears once per (Generation, Replicate) in
    layer_df.  We use Generation==1, first replicate as the canonical index of
    unique variants (every variant always has a post-selection row).

    Returns
    -------
    embeddings   : ndarray (n_variants, emb_dim)
    pre_weights  : ndarray (n_variants,)  – total pre-selection count across reps
    post_weights : ndarray (n_variants,)  – total post-selection count across reps
    """
    rep1 = layer_df['Replicate'].min()
    base = layer_df[(layer_df['Generation'] == 1) & (layer_df['Replicate'] == rep1)].reset_index(drop=True)
    embeddings = np.array(base['Embedding'].tolist())

    # Hash embeddings for fast groupby (collision probability negligible for 640-dim float64)
    layer_df = layer_df.copy()
    layer_df['_h'] = layer_df['Embedding'].apply(lambda x: hash(x.tobytes()))
    base['_h'] = base['Embedding'].apply(lambda x: hash(x.tobytes()))

    pre_sums  = layer_df[layer_df['Generation'] == 0].groupby('_h')['Frequency'].sum()
    post_sums = layer_df[layer_df['Generation'] == 1].groupby('_h')['Frequency'].sum()

    pre_weights  = np.array([pre_sums.get(h, 0)  for h in base['_h']])
    post_weights = np.array([post_sums.get(h, 0) for h in base['_h']])

    return embeddings, pre_weights, post_weights


def pca_from_embeddings(embeddings: np.ndarray, weights: np.ndarray = None):
    """
    Compute PCA on an (n_variants, emb_dim) embedding matrix.

    If weights are given, they are used to form a weighted covariance matrix
    (each point contributes proportionally to its frequency).

    Returns
    -------
    eigenvalues          : ndarray (emb_dim,)  descending
    eigenvectors         : ndarray (emb_dim, emb_dim)  rows = PCs
    explained_var_ratio  : ndarray (emb_dim,)
    mean                 : ndarray (emb_dim,)
    """
    if weights is not None:
        w = weights.astype(float)
        w_sum = w.sum()
        if w_sum == 0:
            weights = None
        else:
            w = w / w_sum

    if weights is not None:
        mean = np.average(embeddings, axis=0, weights=w)
        centered = embeddings - mean
        # Weighted covariance: C = X^T W X  where W = diag(w)
        cov = (centered * w[:, None]).T @ centered
    else:
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        cov = np.cov(centered.T)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order; reverse to descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].T   # rows = principal components

    # Clip tiny negatives from numerical noise
    eigenvalues = np.clip(eigenvalues, 0, None)
    total = eigenvalues.sum()
    explained_var_ratio = eigenvalues / total if total > 0 else eigenvalues

    return eigenvalues, eigenvectors, explained_var_ratio, mean


def _n_components_for_thresholds(cumvar: np.ndarray, thresholds):
    """Return the number of PCs needed to exceed each variance threshold."""
    return {t: int(np.searchsorted(cumvar, t) + 1) for t in thresholds}


def _participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Effective dimensionality  PR = (Σλ)² / Σλ²
    Equals emb_dim if all eigenvalues equal; equals 1 if one dominates.
    """
    s1 = eigenvalues.sum()
    s2 = (eigenvalues ** 2).sum()
    return float(s1 ** 2 / s2) if s2 > 0 else 0.0


def analyze_eigenvectors(
    data_path: str,
    layers=None,
    variance_thresholds=(0.50, 0.80, 0.90, 0.95, 0.99),
    weight_by: str = 'pre',   # 'pre', 'post', 'uniform'
    store_eigenvectors: bool = False,
    verbose: bool = True,
):
    """
    Full eigenvector / PCA analysis of ESM-2 embedding layers.

    Loads inference_df.pkl for each layer from ``data_path/layer{i}/``,
    computes PCA, and returns a dict with per-layer and cross-layer statistics.

    Parameters
    ----------
    data_path          : root directory containing layer0/, layer1/, …
    layers             : list of layer indices to analyse (default: all found)
    variance_thresholds: variance fractions at which to report # of PCs
    weight_by          : 'pre'  – weight by pre-selection frequency
                         'post' – weight by post-selection frequency
                         'uniform' – equal weight per variant
    store_eigenvectors : if True, store full eigenvector matrices (memory-heavy)
    verbose            : print progress

    Returns
    -------
    results : dict with keys
        'layers'                – list of layer indices analysed
        'n_variants'            – list[int]
        'embedding_dim'         – int
        'eigenvalues'           – list[ndarray]  one per layer, descending
        'explained_var_ratio'   – list[ndarray]  per-PC fraction
        'cumulative_var'        – list[ndarray]  cumulative fraction
        'n_components'          – list[dict]  threshold -> n_components per layer
        'participation_ratio'   – list[float]
        'mean_embeddings'       – list[ndarray]  per-layer mean
        'eigenvectors'          – list[ndarray] or None
        'cross_layer' : dict
            'top1_cosine'       – ndarray (n_layers-1,)  |cos sim| of PC-1 between consecutive layers
            'topk_subspace_overlap' – ndarray (n_layers-1,) k-dim subspace overlap
    """
    if layers is None:
        found = sorted(
            int(d[5:]) for d in os.listdir(data_path)
            if d.startswith('layer') and os.path.isdir(os.path.join(data_path, d))
        )
        layers = found

    variance_thresholds = list(variance_thresholds)

    results = {
        'layers': layers,
        'embedding_dim': None,
        'n_variants': [],
        'eigenvalues': [],
        'explained_var_ratio': [],
        'cumulative_var': [],
        'n_components': [],
        'participation_ratio': [],
        'mean_embeddings': [],
        'eigenvectors': [] if store_eigenvectors else None,
    }

    for layer in layers:
        if verbose:
            print(f"  Layer {layer} ...", end=' ', flush=True)
        df_path = os.path.join(data_path, f'layer{layer}', 'inference_df.pkl')
        layer_df = pickle.load(open(df_path, 'rb'))

        embeddings, pre_w, post_w = _extract_embeddings_and_weights(layer_df)

        if results['embedding_dim'] is None:
            results['embedding_dim'] = embeddings.shape[1]

        if weight_by == 'pre':
            weights = pre_w
        elif weight_by == 'post':
            weights = post_w
        else:
            weights = None

        eigenvalues, eigenvectors, evr, mean = pca_from_embeddings(embeddings, weights)
        cumvar = np.cumsum(evr)

        results['n_variants'].append(len(embeddings))
        results['eigenvalues'].append(eigenvalues)
        results['explained_var_ratio'].append(evr)
        results['cumulative_var'].append(cumvar)
        results['n_components'].append(_n_components_for_thresholds(cumvar, variance_thresholds))
        results['participation_ratio'].append(_participation_ratio(eigenvalues))
        results['mean_embeddings'].append(mean)
        if store_eigenvectors:
            results['eigenvectors'].append(eigenvectors)

        if verbose:
            pr = results['participation_ratio'][-1]
            nc90 = results['n_components'][-1].get(0.90, '?')
            print(f"n={len(embeddings):,}  PR={pr:.1f}  #PC@90%={nc90}")

    # ── Cross-layer statistics ────────────────────────────────────────────────
    if store_eigenvectors and len(layers) > 1:
        top1_cosine = []
        topk_overlap = []
        k = 10  # subspace size for overlap
        for i in range(len(layers) - 1):
            v1 = results['eigenvectors'][i][0]
            v2 = results['eigenvectors'][i + 1][0]
            top1_cosine.append(abs(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))))

            # k-dim subspace overlap: normalised sum of squared cosines between subspaces
            U = results['eigenvectors'][i][:k]      # (k, d)
            V = results['eigenvectors'][i + 1][:k]  # (k, d)
            G = U @ V.T  # (k, k) Gram matrix
            overlap = float(np.linalg.norm(G, 'fro') ** 2) / k
            topk_overlap.append(overlap)

        results['cross_layer'] = {
            'top1_cosine':          np.array(top1_cosine),
            'topk_subspace_overlap': np.array(topk_overlap),
            'k_subspace':           k,
        }
    else:
        results['cross_layer'] = None

    return results


def plot_eigenvector_analysis(results: dict,
                              variance_thresholds=(0.50, 0.80, 0.90, 0.95, 0.99),
                              figsize_base=(5, 4),
                              cmap_name='viridis',
                              save_path: str = None):
    """
    Comprehensive visualisation of the PCA/eigenvector analysis.

    Panels
    ------
    1. Eigenvalue spectra (log scale) per layer – one line per layer, coloured by depth
    2. Cumulative explained variance per layer
    3. Number of PCs needed per threshold vs layer
    4. Participation ratio (effective dimensionality) vs layer
    5. Top-1 PC cosine similarity between consecutive layers  (if cross_layer available)
    6. k-dim subspace overlap between consecutive layers       (if cross_layer available)
    """
    layers = results['layers']
    n_layers = len(layers)
    cmap = plt.get_cmap(cmap_name, n_layers)

    has_cross = results['cross_layer'] is not None
    n_panels = 4 + (2 if has_cross else 0)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_base[0] * ncols,
                                                     figsize_base[1] * nrows))
    axes = axes.flatten()

    variance_thresholds = list(variance_thresholds)
    emb_dim = results['embedding_dim']

    # ── Panel 1: eigenvalue spectra ──────────────────────────────────────────
    ax = axes[0]
    top_k = min(100, emb_dim)
    for idx, layer in enumerate(layers):
        ev = results['eigenvalues'][idx][:top_k]
        ax.plot(np.arange(1, top_k + 1), ev, color=cmap(idx), alpha=0.8, lw=1.2)
    ax.set_yscale('log')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title('Eigenvalue spectra (top 100 PCs)')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(layers[0], layers[-1]))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Layer')

    # ── Panel 2: cumulative variance ─────────────────────────────────────────
    ax = axes[1]
    top_k_var = min(200, emb_dim)
    for idx, layer in enumerate(layers):
        cv = results['cumulative_var'][idx][:top_k_var]
        ax.plot(np.arange(1, len(cv) + 1), cv, color=cmap(idx), alpha=0.8, lw=1.2)
    for t in variance_thresholds:
        ax.axhline(t, color='grey', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Cumulative variance explained')
    ax.set_title('Cumulative variance (first 200 PCs)')
    ax.set_ylim(0, 1.02)
    plt.colorbar(sm, ax=ax, label='Layer')

    # ── Panel 3: #PCs for each threshold vs layer ────────────────────────────
    ax = axes[2]
    thresholds_to_plot = variance_thresholds
    marker_styles = ['o', 's', '^', 'D', 'v', 'P', 'X']
    for ti, t in enumerate(thresholds_to_plot):
        n_pcs = [results['n_components'][i][t] for i in range(n_layers)]
        ax.plot(layers, n_pcs, marker=marker_styles[ti % len(marker_styles)],
                label=f'{int(t*100)}%', lw=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('# PCs')
    ax.set_title('PCs needed per variance threshold')
    ax.legend(fontsize=8)

    # ── Panel 4: participation ratio ─────────────────────────────────────────
    ax = axes[3]
    ax.plot(layers, results['participation_ratio'], 'o-', color='steelblue', lw=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Participation ratio')
    ax.set_title(f'Effective dimensionality  (max = {emb_dim})')
    ax.axhline(emb_dim, color='grey', lw=0.8, ls='--', alpha=0.6)

    if has_cross:
        cl = results['cross_layer']

        # ── Panel 5: top-1 cosine similarity ─────────────────────────────────
        ax = axes[4]
        pairs = [(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        pair_labels = [f'{a}→{b}' for a, b in pairs]
        ax.bar(range(len(pair_labels)), cl['top1_cosine'], color='coral')
        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('|cosine similarity|')
        ax.set_title('PC-1 alignment between consecutive layers')
        ax.set_ylim(0, 1.05)

        # ── Panel 6: k-dim subspace overlap ──────────────────────────────────
        ax = axes[5]
        k = cl['k_subspace']
        ax.bar(range(len(pair_labels)), cl['topk_subspace_overlap'], color='mediumseagreen')
        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Normalised subspace overlap')
        ax.set_title(f'Top-{k} PC subspace overlap (1 = identical)')
        ax.set_ylim(0, 1.05)

    # hide unused axes
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def print_eigenvector_summary(results: dict,
                               variance_thresholds=(0.50, 0.80, 0.90, 0.95, 0.99)):
    """
    Print a concise summary table of the PCA analysis.
    """
    variance_thresholds = list(variance_thresholds)
    header = ['Layer', 'N_variants', 'Eff.dim (PR)'] + [f'#PC@{int(t*100)}%' for t in variance_thresholds]
    rows = []
    for i, layer in enumerate(results['layers']):
        row = [
            layer,
            results['n_variants'][i],
            f"{results['participation_ratio'][i]:.1f}",
        ] + [results['n_components'][i][t] for t in variance_thresholds]
        rows.append(row)

    col_widths = [max(len(str(r[c])) for r in rows + [header]) for c in range(len(header))]
    fmt = '  '.join(f'{{:<{w}}}' for w in col_widths)
    print(fmt.format(*header))
    print('  '.join('-' * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print()
    print(f"Embedding dim : {results['embedding_dim']}")
    print(f"Layers        : {results['layers'][0]} – {results['layers'][-1]}")
    pr_vals = results['participation_ratio']
    print(f"Eff.dim range : {min(pr_vals):.1f} – {max(pr_vals):.1f}  "
          f"(peak at layer {results['layers'][int(np.argmax(pr_vals))]})")


