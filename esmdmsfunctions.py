import os
#import shutil
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
#from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
#from sklearn.metrics import silhouette_score
import pickle
import time

import matplotlib.pyplot as plt

import popDMS
from importlib import reload

# reload popDMS
reload(popDMS)

## GLOBAL VARIABLES


pwd = "/net/dali/home/barton/dhw28/popDMS/esmDMS"

# Pick an ESM-2 model size
model_name = "facebook/esm2_t30_150M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
model = AutoModel.from_pretrained(model_name)

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

default_emb_path = pwd + '/data/sequence_data/all_reps_BF520_protein_embeddings.pkl'


## FUNCTIONS

def load_codoncounts(filepath):
    """Load in the dataframe for the codoncounts file"""
    df = pd.read_csv(filepath)
    column_names = df.columns.tolist()
    column_names = column_names[2:]
    wildtypes = df["wildtype"].tolist()
    df = df.drop(columns=["site", "wildtype"])
    df_array = df.to_numpy()
    return df_array, column_names, wildtypes

def count_unique_proteins(filepath=None, codon_array=None, 
                          column_names=None, wildtypes=None):
    """Count the number of unique proteins in the codon array"""
    if filepath is not None:
        codon_array, column_names, wildtypes = load_codoncounts(filepath)
    else:
        assert codon_array is not None
        assert column_names is not None
        assert wildtypes is not None
    
    count_unique = 0
    # iterate through each row
    for i in range(codon_array.shape[0]):
        row = codon_array[i]
        for j in range(row.shape[0]):
            if row[j] > 0 and column_names[j] != wildtypes[i]:
                count_unique += 1
    return count_unique

def get_reference_sequence(filepath):
    """Get the reference sequence from a text file"""
    with open(filepath, 'r') as f:
        reference_sequence = f.read().strip()
    
    reference_protein_sequence = ""
    for i in range(0, len(reference_sequence), 3):
        codon = reference_sequence[i:i+3]
        aa = CODON2AA.get(codon, 'X')  # Use 'X' for unknown codons
        reference_protein_sequence += aa
    return reference_protein_sequence


def get_day_estimate(filepath):
    codon_array, column_names, wildtypes = load_codoncounts(filepath)
    total_unique = count_unique_proteins(codon_array=codon_array, 
                                         column_names=column_names, 
                                         wildtypes=wildtypes)
    row_number = codon_array.shape[0] # number of sites
    day_estimate = total_unique * row_number / 100000
    return day_estimate

def count_total_proteins(filepath):
    """Sum the total number of proteins in a single row"""
    codon_array, column_names, wildtypes = load_codoncounts(filepath)
    # sum along the first row
    total_proteins = np.sum(codon_array[0, :])
    return total_proteins
    

def write_codon_replicates(replicates_pre: list, replicates_post: list, output_path: str, reference_seq: str):
    """Write the codon changes from multiple replicates to a single file"""
    assert len(replicates_pre) == len(replicates_post), "Number of pre and post replicate files must be the same"
    codon_arrays_pre = []
    codon_arrays_post = []
    loaded_wildtypes, loaded_columns = False, False
    wildtypes_master = []
    columns_master = []
    for filepath in replicates_pre:
        codon_array, column_names, wildtypes = load_codoncounts(filepath)
        codon_arrays_pre.append(codon_array)
        if not loaded_wildtypes:
            loaded_wildtypes = True
            wildtypes_master = wildtypes
        if not loaded_columns:
            loaded_columns = True
            columns_master = column_names
        assert (wildtypes == wildtypes_master), "Wildtypes do not match across replicates"
        assert (column_names == columns_master), "Column names do not match across replicates"
    
    for filepath in replicates_post:
        codon_array, column_names, wildtypes = load_codoncounts(filepath)
        codon_arrays_post.append(codon_array)
        assert (wildtypes == wildtypes_master), "Wildtypes do not match across replicates"
        assert (column_names == columns_master), "Column names do not match across replicates"
        
    column_names_aa = [CODON2AA.get(codon, 'X') for codon in columns_master]
    wildtypes_aa = [CODON2AA.get(codon, 'X') for codon in wildtypes_master]
    new_prot_seqs = []
    pre_num_array = []
    post_num_array = []
    with open (output_path, 'w') as f:
        f.write("PreNums,PostNums,ProteinSequence\n")
        # Iteratre through each row
        for i in range(codon_arrays_pre[0].shape[0]):
            rows_pre = [pre_array[i] for pre_array in codon_arrays_pre]
            rows_post = [post_array[i] for post_array in codon_arrays_post]
            for j in range(rows_pre[0].shape[0]):
                if column_names_aa[j] != wildtypes_aa[i]:
                    pre_nums = [int(pre_row[j]) for pre_row in rows_pre]
                    post_nums = [int(post_row[j]) for post_row in rows_post]
                    
                    pre_num_array.append(pre_nums)
                    post_num_array.append(post_nums)
                    
                    amino_acid = column_names_aa[j]
                    
                    new_prot_seq = reference_seq[:i] + amino_acid + reference_seq[i+1:]
                    new_prot_seqs.append(new_prot_seq)
                    f.write(f"{pre_nums},{post_nums},{new_prot_seq}\n")
                    
    # save a pickle dataframe too
    
    new_df = pd.DataFrame({
        'PreNums': pre_num_array,
        'PostNums': post_num_array,
        'ProteinSequence': new_prot_seqs
    })
    
    def ensure_list(v):
        """Convert a value to a list of numbers if it is a string or array."""
        if isinstance(v, str):
            return ast.literal_eval(v)  # safely convert string like "[1, 2, 3]" → list
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, (list, tuple)):
            return list(v)
        else:
            raise TypeError(f"Unexpected type in PreNums/PostNums: {type(v)}")

    new_df["PreNums"] = new_df["PreNums"].map(ensure_list)
    new_df["PostNums"] = new_df["PostNums"].map(ensure_list)

    new_df = new_df.groupby("ProteinSequence", as_index=False).agg({
        "PreNums": lambda x: [sum(vals) for vals in zip(*x)],
        "PostNums": lambda x: [sum(vals) for vals in zip(*x)],
    })
    
    new_df.to_pickle(output_path.replace('.csv', '.pkl'))
        
    
   
    
def embed_sequence(sequence: str, tokenizer, model) -> np.ndarray:
    """Embed the sequence to a fixed size vector using ESM-2"""

    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():

        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states


    output_embeddings = []
    for layer in hidden_states:
        token_representations = layer
        #print(token_representations.shape)  # Shape: (1, sequence_length, embedding_dim)        s
        sequence_embedding = pool_sequence_representation(token_representations, inputs)
        output_embeddings.append(sequence_embedding)
        #print(sequence_embedding.shape)  # Shape: (embedding_dim,)
    return np.vstack(output_embeddings)  # Shape: (num_layers, embedding_dim)

def pool_sequence_representation(token_representations, inputs) -> np.ndarray:
    """Pool the token representations to get a fixed-size sequence representation."""
    # token_representations shape: (1, sequence_length, embedding_dim)
    # inputs['attention_mask'] shape: (1, sequence_length)
    attention_mask = inputs['attention_mask']
    masked_representations = token_representations * attention_mask.unsqueeze(-1)
    summed = masked_representations.sum(dim=1)
    counts = attention_mask.sum(dim=1).unsqueeze(-1)
    pooled_representation = summed / counts
    return pooled_representation.squeeze(0).cpu().numpy()  # Shape: (embedding_dim,)


def embed_replicates(embedding_df: pd.DataFrame, 
                     output_path: str,
                     embed_zeroes: bool=False,
                     esm_model: str="facebook/esm2_t30_150M_UR50D") -> None:
    """Embed the sequeunces given the replicates embedding dataframe from
    write_codon_replicates() """
    start_time = time.time()
    pre_counts = embedding_df["PreNums"].to_list()
    post_counts = embedding_df["PostNums"].to_list()
    
    tokenizer = AutoTokenizer.from_pretrained(esm_model, do_lower_case=False)
    model = AutoModel.from_pretrained(esm_model)
    
    embeddings = []
    for i, prot_sequence in enumerate(embedding_df["ProteinSequence"].to_list()):
        if embed_zeroes:
            embedding = embed_sequence(prot_sequence, tokenizer, model)
            embeddings.append(embedding)
        else:
            if any([x > 0 for x in pre_counts[i]]):
                embedding = embed_sequence(prot_sequence, tokenizer, model)
                embeddings.append(embedding)
            else:
                embeddings.append(None)
        if i % 100 == 0:
            cur_time = time.time()
            estimate_remaining = (cur_time - start_time) / (i + 1) * (len(embedding_df) - (i + 1))
            print(f"Embedded {i} sequences, time elapsed: {cur_time - start_time:.2f} seconds, estimated time remaining: {estimate_remaining/60:.2f} minutes")
    
    
    embedding_df['Embeddings'] = embeddings
    embedding_df.to_pickle(output_path)
    print(f"Wrote embeddings to {output_path}")


# Now, let's analyze these dang embeddings
def calc_cov_mats(embeddings: np.ndarray, pre_weights: np.ndarray, 
                  post_weights: np.ndarray, layer: int = None) -> np.ndarray:
    """
    Calculate the covariance matrices for the before and after counts
    embeddings: 3 dimensional array of shape (num_sequences, num_layers, embedding_dim)
    layer: which layer to use for the embeddings, if None, use all layers
    """
    
    if layer is not None:
        embeddings = embeddings[:, layer, :]
        
    
    before_cov = np.cov(embeddings.T, aweights=pre_weights)
    after_cov = np.cov(embeddings.T, aweights=post_weights)
    return before_cov, after_cov

def calc_cov_mats_reps(embeddings: np.ndarray, pre_weights: np.ndarray, 
                       post_weights: np.ndarray, layer: int = None) -> np.ndarray:
    """
    Calculate the covariance matrices for the before and after counts
    embeddings: 3 dimensional array of shape (num_sequences, num_layers, embedding_dim)
    layer: which layer to use for the embeddings, if None, use all layers
    """
    num_reps = pre_weights.shape[1]
    if layer is not None:
        embeddings = embeddings[:, layer, :]
        
    before_covs = []
    after_covs = []
    for rep in range(num_reps):
        before_cov = np.cov(embeddings.T, aweights=pre_weights[:, rep])
        after_cov = np.cov(embeddings.T, aweights=post_weights[:, rep])
        before_covs.append(before_cov)
        after_covs.append(after_cov)
    return before_covs, after_covs
    


def embedding_df_transfer(embed_df: pd.DataFrame) -> dict:
    """
    Format of embed_df:
         PreNums   PostNums   ProteinSequence  Embeddings
    0  [0, 0, 0]  [0, 0, 0]   MKT...           [[...], [...], ...]
    

    Format of RepNDataFrame:
    generation, embedding, frequency, replicate
    
    
    Key Differences:
    We will not have sites and amino acids. Instead, we will have 
    the N embedding dimensions 
    """
    
    pre_counts = np.array([np.array(x) for x in embed_df["PreNums"].to_list()])
    post_counts = np.array([np.array(x) for x in embed_df["PostNums"].to_list()])
    embeddings = np.array([x for x in embed_df["Embeddings"].to_list()])
    print("done converting to arrays")
    num_reps = pre_counts.shape[1]
    num_gens = 2 # set to 2 for now, pre and post selection
    
    new_df = pd.DataFrame(columns=["Generation", "Embedding", "Frequency", "Replicate"])
    print("initialized new df")
    for rep in range(num_reps):
        # loop through every row
        print(f"on replicate {rep}")
        for i in range(embed_df.shape[0]):
            print(f"on embedding {i}")
            if embeddings[i] is not None:
                # loop through every generation
                for gen in range(num_gens):
                    print(f"on generation {gen}")
                    if gen == 0:
                        freq = pre_counts[i, rep]
                    else:
                        freq = post_counts[i, rep]
                        
                    if gen == 0 and freq == 0:
                        continue
                        
                    new_row = {
                        "Generation": gen,
                        "Embedding": embeddings[i],
                        "Frequency": freq,
                        "Replicate": rep+1
                    }
                    new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                    
    return new_df
    

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
    data = popDMS.mini_infer_independent_esm(inference_df, n_replicates=n_replicates,
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
        print(layer_df.head())
        # data = [dx, icov, s, s_joint, sel_data, gamma_opt, x_array] for layer
        data = run_inference_calcs(layer_df, layer_path, verbose=verbose,
                                   pre_processed=True, n_replicates=num_reps) 
        layer_results.append(data)
        
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


def z_normalize(array: np.ndarray) -> np.ndarray:
    """Z-normalize the input numpy array."""
    mean = np.mean(array)
    std = np.std(array)
    if std == 0:
        return array - mean
    return (array - mean) / std


## PLOTTING

import scipy as sp 
import scipy.stats as st

def make_grfp_plots_inf(inference_data, normalize=True):
    """Make GRFP plots for the dataframe"""
    # inference_data = [dx, icov, s, s_joint, sel_data, gamma_opt, x_array] for layer
    # inference_data = analyze_layers(df, verbose=False)
    selection_coeffs = []
    for layer in range(len(inference_data)):
        s = inference_data[layer][2]
        #print(f"Selection coefficients for layer {layer}: {s}")
        selection_coeffs.append(s)
        
    num_reps = len(selection_coeffs[0])
    rep_combs = []
    for i in range(num_reps):
        for j in range(i+1, num_reps):
            rep_combs.append((i, j))
    print(f"rep_combs: {rep_combs}")
    num_combs = len(rep_combs)
    # Make a figure with subplots for each replicate combination
    fig, axs = plt.subplots(1, num_combs, figsize=(6*num_combs, 6))
    for comb_index, (rep_i, rep_j) in enumerate(rep_combs):
        ax = axs[comb_index]
        for layer in range(len(selection_coeffs)):
            s = selection_coeffs[layer].copy()
            if normalize:
                s[rep_i] = s[rep_i] / np.max(np.abs(s[rep_i]))
                s[rep_j] = s[rep_j] / np.max(np.abs(s[rep_j]))
            
            ax.scatter(s[rep_i], s[rep_j], label=f'Layer {layer}')
            
        ax.set_title(f'Replicate {rep_i+1} vs Replicate {rep_j+1}')
        ax.set_xlabel(f'Selection Coefficients Replicate {rep_i+1}')
        ax.set_ylabel(f'Selection Coefficients Replicate {rep_j+1}')
        ax.axis('square')
        #ax.legend()
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.suptitle('Replicate Consistency Plots Across Layers', fontsize=16)
    plt.show()


def get_correlations(selection_data):
    """Get the pearson correlation data from the selection data"""
    # Data format:
    # s = [[s_rep_1_layer_1, s_rep_2_layer_1, s_rep_3_layer_1], 
    #     [s_rep_1_layer_2, s_rep_2_layer_2, s_rep_3_layer_2], ...]
    
    num_layers = len(selection_data)
    num_reps = len(selection_data[0])
    rep_combs = []
    for i in range(num_reps):
        for j in range(i+1, num_reps):
            rep_combs.append((i, j))
            
    all_corrs = []
    for layer in range(num_layers):
        s = selection_data[layer]
        layer_corrs = []
        for (rep_i, rep_j) in rep_combs:
            corr = st.pearsonr(s[rep_i], s[rep_j])[0]
            layer_corrs.append(corr)
        all_corrs.append(layer_corrs)
    
    return np.array(all_corrs)  # shape: (num_layers, num_combs)


def plot_from_df(df, verbose=False, table=True):
    """Plot GRFP plots from the dataframe"""
    inference_data = analyze_layers(df, verbose=verbose)
    
    make_grfp_plots_inf(inference_data, normalize=True)
    
    # make a table of the average correlations across replicate combinations for each layer
    if table:
        selection_data = []
        for layer in range(len(inference_data)):
            s = inference_data[layer][2]
            selection_data.append(s)
        corrs = get_correlations(selection_data)
    
        avg_corrs = np.mean(corrs, axis=1)
        print("Average Pearson Correlations Across Replicate Combinations for Each Layer:")
        for layer in range(len(avg_corrs)):
            print(f"Layer {layer}: {avg_corrs[layer]:.4f}")
        print("Full Correlation Table:")
        print(pd.DataFrame(corrs, columns=[f'Rep {i+1} vs Rep {j+1}' for i in range(3) for j in range(i+1, 3)],
                           index=[f'Layer {i}' for i in range(len(avg_corrs))]))
        print(f"Overall average correlation: {np.mean(avg_corrs):.4f}")
    return inference_data


## SHUFFLING FUNCTIONS

def shuffle_replicates(df: pd.DataFrame, replicates: list, random_seed: int = None) -> pd.DataFrame:
    """Shuffle the replicate counts of the given replicates in the dataframe
    
    INPUT: 
    df: dataframe with PreNums and PostNums columns
    replicates: list of replicate indices to shuffle
    random_seed: seed for reproducibility
    
    OUTPUT:
    shuffled_df: dataframe with shuffled replicate counts
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    shuffled_df = df.copy()
    for rep in replicates:
        pre_col = 'PreNums'
        post_col = 'PostNums'
        
        pre_counts = np.array([np.array(x) for x in shuffled_df[pre_col].to_list()])
        post_counts = np.array([np.array(x) for x in shuffled_df[post_col].to_list()])
        
        # extract the replicate column
        pre_rep_counts = pre_counts[:, rep]
        post_rep_counts = post_counts[:, rep]
        
        # shuffle the counts
        np.random.shuffle(pre_rep_counts)
        np.random.shuffle(post_rep_counts)
        
        # put back into the dataframe
        for i in range(shuffled_df.shape[0]):
            pre_counts[i, rep] = pre_rep_counts[i]
            post_counts[i, rep] = post_rep_counts[i]
        
        shuffled_df[pre_col] = [pre_counts[i].tolist() for i in range(shuffled_df.shape[0])]
        shuffled_df[post_col] = [post_counts[i].tolist() for i in range(shuffled_df.shape[0])]
    return shuffled_df
        


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



def get_df_selection(random_seed=42, selected_layer=12, normalize_embeddings=True, input_df=None):
    """Get the starting information for the simulation,
    i.e. the layer dataframe and the initial counts for each replicate.
    
    Args:
        input_df: Optional pre-built dataframe containing Rep1/2/3_PreNums,
                  Rep1/2/3_PostNums, and an 'Embedding' column for the selected
                  layer. If provided, data loading and layer decomposition are
                  skipped entirely.
    """
    rc('text', usetex=True)
    pd.set_option('display.max_columns', 100)
    np.random.seed(random_seed)

    if input_df is not None:
        # Expect input_df to already have Rep{1,2,3}_Pre/PostNums and 'Embedding'
        required_cols = [
            'Rep1_PreNums', 'Rep2_PreNums', 'Rep3_PreNums',
            'Rep1_PostNums', 'Rep2_PostNums', 'Rep3_PostNums',
            'Embedding'
        ]
        missing = [c for c in required_cols if c not in input_df.columns]
        if missing:
            raise ValueError(f"input_df is missing required columns: {missing}")
        df_selection = input_df.copy()

    else:
        pwd = os.getcwd()
        init_pop = get_unique_df(pwd + '/data/sequence_data/all_reps_BF520_protein_embeddings.pkl')
        n_reps = 3

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
        embeddings   = np.vstack(df_selection['Embedding'].tolist())
        z_embeddings = z_normalize(embeddings)
        df_selection['Embedding'] = [z_embeddings[i] for i in range(z_embeddings.shape[0])]

    start_counts1 = df_selection["Rep1_PreNums"].values
    start_counts2 = df_selection["Rep2_PreNums"].values
    start_counts3 = df_selection["Rep3_PreNums"].values

    return df_selection, (start_counts1, start_counts2, start_counts3)


def run_simulation(df_selection, selection_coefficients, initial_counts, 
                   n_gens=30, embedding_clip=None, save_every=1, fitness='exp'):
    embeddings = np.vstack(df_selection['Embedding'].values)
    
    if fitness == 'exp':
        fitnesses = calc_all_fitness_exp(embeddings, selection_coefficients,
                                        embedding_clip=embedding_clip)
    elif fitness == 'plus1':
        fitnesses = calc_all_fitness_plus1(embeddings, selection_coefficients,
                                        embedding_clip=embedding_clip)
    else:
        raise ValueError("Invalid fitness function specified.")
    
    #print("intial counts shape and type:", np.array(initial_counts).shape, type(initial_counts))
    generation_counts = [initial_counts]
    n_reps = len(initial_counts)
    last_counts = initial_counts
    
    print(last_counts)
    
    for gen in range(n_gens):
        
        this_gen = []
        for rep in range(n_reps):
            rep_counts = last_counts[rep]
            next_counts = simulate_generation_multinomial(rep_counts, fitnesses)
            this_gen.append(next_counts)
        if (gen + 1) % save_every == 0:
            #print(f"Completed generation {gen + 1}")
            generation_counts.append(this_gen)
        last_counts = this_gen
    #print(f"generation_counts length: {len(generation_counts)}")
    #print(f"generation_counts shape: {np.array(generation_counts).shape}")
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

def run_inference_calcs_sims(df_selection, generation_counts, output_path,
                             save_output=False):
    """Run the inference calculations:
    WHOLE PIPELINE FROM READING IN EMBEDDINGS DATAFRAME
    """
    inference_df = simulation_df_transfer(df_selection, generation_counts)
    
    if save_output:
        # make directory
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # save inference_df
        inference_df.to_pickle(output_path + 'inference_df.pkl')
        print(f"SAVED INFERENCE DF TO {output_path}")
    
       
    data = popDMS.mini_infer_independent_esm(inference_df, n_replicates=3)
    return data # data = [dx, icov, s, s_joint, sel_data, gamma_opt, x_array]

def other_methods(df_selection, generation_counts, generation=-1):
    """Find the enrichment ratio, log ratio, and log enrichment"""
    first_gen = generation_counts[0]
    last_gen = generation_counts[generation]
    
    enrichments = []
    log_ratios = []
    #log_enrichments = []
    
    embeddings = np.vstack(df_selection['Embedding'].values)
    
    for rep in range(len(first_gen)):
        start_counts = first_gen[rep]
        end_counts = last_gen[rep]
        
        embedding_avg_before = np.sum(embeddings.T * start_counts, axis=1) / np.sum(start_counts)
        embedding_avg_after = np.sum(embeddings.T * end_counts, axis=1) / np.sum(end_counts)
        
        
        embedding_sum = np.sum(embedding_avg_before) + np.sum(embedding_avg_after) / 2
        
        enrichment = (embedding_avg_after / embedding_avg_before) / embedding_sum
        
        log_ratio = np.log((embedding_avg_after / embedding_avg_before) / embedding_sum)
        
        #log_enrichment = np.log(enrichment)
        
        log_ratios.append(log_ratio)
        enrichments.append(enrichment)
        
    return np.array(enrichments), np.array(log_ratios) #, np.array(log_enrichments)

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


def generate_selection(embeddings):
    embedding_ranges = embeddings.max(axis=0) - embeddings.min(axis=0)
    selection_coefficients = np.zeros(embeddings.shape[1])
    # Find the indices of the top, lowest, and middle range dimensions
    sorted_indices = np.argsort(embedding_ranges)
    high_range_idx = sorted_indices[-1]
    # Give these dimensions higher selection coefficients
    selection_coefficients[high_range_idx] = 0.10
    return selection_coefficients

def generate_one_selection(embeddings):
    selection_coefficients = np.zeros(embeddings.shape[1])
    embedding_ranges = embeddings.max(axis=0) - embeddings.min(axis=0)
    # Find the indices of the top, lowest, and middle range dimensions
    sorted_indices = np.argsort(embedding_ranges)
    high_range_idx = sorted_indices[-1]
    # Give these dimensions higher selection coefficients
    selection_coefficients[high_range_idx] = 0.10
    return selection_coefficients

def get_simulation_results(n_layers, n_gens, n_reps, 
                           sel_func=generate_selection,
                           inference=True, fitness='plus1',
                           save_every=1):
    all_layer_fits = {}
    all_selection_coefficients = {}
    detailed_selection_results = {}
    all_generation_counts = {}
    whole_embedding_matrix = {}
    for layer in range(n_layers):
        print(f"Running layer {layer}...")
        df_selection, initial_counts = get_df_selection(random_seed=42, selected_layer=layer,
                                                        normalize_embeddings=False)
        print(df_selection.head())
        df_selection = normalize_embeddings(df_selection)

        # Generate selection coefficients using the provided function
        embedding_matrix = np.vstack(df_selection['Embedding'].tolist())
        whole_embedding_matrix[layer] = embedding_matrix
        selection_coefficients = sel_func(embedding_matrix)
        all_selection_coefficients[layer] = selection_coefficients
        
        print("Running simulation...")
        generation_counts, layer_fits = run_simulation(df_selection, selection_coefficients, 
                                           initial_counts, n_gens=n_gens, save_every=save_every,
                                           fitness=fitness)
        all_generation_counts[layer] = generation_counts
        all_layer_fits[layer] = layer_fits

        if inference:
            gen=n_gens
            print(f"  Analyzing generation {gen}...")
            layer_results = []
            test_path = pwd + f"/simulations/layer_{layer}_gen_{gen}/"
            data = run_inference_calcs_sims(df_selection, generation_counts[:gen + 1], test_path)
            found_sel_coeffs = data[2]
            layer_results.append(found_sel_coeffs)

            detailed_selection_results[layer] = layer_results

    return all_layer_fits, all_selection_coefficients, detailed_selection_results, all_generation_counts, whole_embedding_matrix
    
    
    
def calc_inferred_fits(sim_data, fitness='plus1', layer=0):
    """ Calculate the inferred fitness score of every individual in the population across layer and generation using the inferred selection coefficients and the embeddings"""
    sel_coefs = sim_data[2][layer][0]
    embeddings = sim_data[4][layer]
    n_reps = sel_coefs.shape[0]
    n_indivs = embeddings.shape[0]
    inferred_fits = []
    for indiv in range(n_indivs):
        indiv_fits = []
        for rep in range(n_reps):
            if fitness == 'exp':
                fit = calculate_fitness_exp(embeddings[indiv], sel_coefs[rep])
            elif fitness == 'plus1':
                fit = calculate_fitness_plus1(embeddings[indiv], sel_coefs[rep])
            else:
                raise ValueError("Invalid fitness function specified.")
            indiv_fits.append(fit)
        inferred_fits.append(indiv_fits)
    return np.array(inferred_fits)
    
def comp_inf_vs_real_fits(sim_data, layer, fitness='plus1'):
    """Compare the inferred and real fitness scores."""
    # For each layer, plot the fitness growth over time
    all_layer_fits = sim_data[0][layer]
    all_gen_counts = sim_data[3]
    embeddings = sim_data[4][layer]
    n_gens = len(all_gen_counts[0]) - 1
    n_reps = len(all_gen_counts[0][0])

    # Print the shape of all these data
    #print(fitness)
    inferred_fits = calc_inferred_fits(sim_data, fitness=fitness, 
                                       layer=layer)

    # Z-normalize the real fits
    real_fits = np.array(all_layer_fits)
    real_fits = z_normalize(real_fits)

    # Normalize the inferred fits per replicate
    rep_fits = {}
    for rep in range(n_reps):
        rep_fits[rep] = z_normalize(inferred_fits[:, rep])

    # Make a plot showing the comparison between the real and inferred
    # fitness scores for each replicate, with a diagonal line for reference
    fig, axes = plt.subplots(1, n_reps, figsize=(6 * n_reps, 6), squeeze=False)
    axes = axes.flatten()
    plt.style.use('seaborn-v0_8-darkgrid')

    all_real = []
    all_inferred = []

    for rep in range(n_reps):
        ax = axes[rep]
        r = real_fits
        inf = rep_fits[rep]
        ax.scatter(r, inf, alpha=0.5)
        ax.set_xlabel('Real Fitness (Normalized)')
        ax.set_ylabel('Inferred Fitness (Normalized)')
        ax.set_title(f'Layer {layer} — Replicate {rep}')

        # Diagonal reference line
        lims = [min(r.min(), inf.min()) - 0.5,
                max(r.max(), inf.max()) + 0.5]
        ax.plot(lims, lims, color='red', linestyle='--')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Compute and annotate Pearson r
        
        
        corr, pval = pearsonr(r, inf)
        ax.annotate(f'r = {corr:.3f}\np = {pval:.2e}',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     ha='left', va='top',
                     fontsize=11, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        all_real.extend(r)
        all_inferred.extend(inf)

    plt.suptitle(f'Real vs Inferred Fitness — Layer {layer}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # Also return the overall correlation across all replicates
    overall_corr, overall_pval = pearsonr(all_real, all_inferred)
    print(f'Overall Pearson r = {overall_corr:.4f}, p = {overall_pval:.2e}')

    return rep_fits, overall_corr



def find_fixed_gen(generation_counts, cutoff_pct=0.75):
    # Return the generation at which 90% of the population has the same dominant type
    for gen in range(len(generation_counts)):
        for rep in range(len(generation_counts[gen])):
            rep_counts = generation_counts[gen][rep]
            total_count = np.sum(rep_counts)
            max_count = np.max(rep_counts)
            #print(f"Generation: {gen} Replicate: {rep} Max Count: {max_count} Total Count: {total_count}")
            if max_count / total_count >= cutoff_pct:
                return gen
    return len(generation_counts) - 1  # Return the last generation if never reaches cutoff

def averaged_covariance(sim_data, layer=0):
    layer_df = get_df_selection(random_seed=42, selected_layer=layer,
                                    normalize_embeddings=False)[0]
    layer_df = normalize_embeddings(layer_df)
    embeddings = np.vstack(layer_df['Embedding'].tolist())

    # Get the selection coefficients
    true_selection = sim_data[1][layer]
    layer_generation_counts = sim_data[3][layer]
    inferred_selection = sim_data[2][layer][0]
    best_dim_idx = np.argmax(np.abs(true_selection))

    # Find the covariance between all embeddings with the best dimension
    best_dim_values = embeddings[:, best_dim_idx]
    fixed_gen = find_fixed_gen(layer_generation_counts)
    print(f"Cutoff generation for layer {layer}: {fixed_gen}")
    total_covariances = []
    for gen in range(fixed_gen):
        # Find the covariance using a weighted approach and the population from the first selection event (gen=1)
        weights = np.average(layer_generation_counts[gen], axis=0)
        covariances = []
        for dim in range(embeddings.shape[1]):
            dim_values = embeddings[:, dim]
            mean_best = np.average(best_dim_values, weights=weights)
            mean_dim = np.average(dim_values, weights=weights)
            covariance = np.average((best_dim_values - mean_best) * (dim_values - mean_dim), weights=weights)
            covariances.append(covariance)
        covariances = np.array(covariances)
        total_covariances.append(covariances)
    avg_covariances = np.mean(total_covariances, axis=0)
    # Plot the selection coefficients in order of rank, colored by covariance with best dimension
    # Plot all three replicates
    plt.figure(figsize=(15, 5))
    plt.style.use('seaborn-v0_8-darkgrid')
    for rep in range(3):
        rep_inf_sel = inferred_selection[rep]
        
        normalized_selection = z_normalize(rep_inf_sel)
        
        # Use this:
        sorted_indices = np.argsort(normalized_selection)[::-1]  # Sort descending
        x_vals = np.arange(len(normalized_selection))  # Simple 0, 1, 2, ... for x-axis
        y_vals = normalized_selection[sorted_indices]  # Values in descending order
        covariances_sorted = avg_covariances[sorted_indices]  # Sort covariances to match
        
        plt.subplot(1, 3, rep + 1)
        scatter = plt.scatter(x_vals, y_vals, c=covariances_sorted, cmap='coolwarm', alpha=0.7)
        plt.colorbar(scatter, label='Average Covariance with Best Dimension')
        plt.title(f'Layer {layer} Replicate {rep + 1}')
        plt.xlabel('Rank of Inferred Selection Coefficient')
        plt.ylabel('Inferred Selection Coefficient (Normalized)')
        plt.axhline(0, color='black', linestyle='--')
        
        
        # For the best dimension highlight:
        best_dim_position = np.where(sorted_indices == best_dim_idx)[0]
        plt.scatter(best_dim_position, normalized_selection[best_dim_idx],
                    color='yellow', edgecolor='black', s=100, label='Best Dimension')
        
    plt.tight_layout()
    plt.show()

# plot fitness over time
def plot_fitness_over_time(sim_data):
    # For each layer, plto the fitness growth over time
    
    all_layer_fits = sim_data[0]
    all_gen_counts = sim_data[3]
    
    n_gens = len(all_gen_counts[0]) - 1
    n_reps = len(all_gen_counts[0][0])
    
    avg_fitness_over_time = {}
    for layer in range(len(all_layer_fits.keys())):
        layer_fits = all_layer_fits[layer]
        layer_counts = all_gen_counts[layer]
        avg_fitness_by_rep = []
        for gen in range(len(layer_counts)):
            gen_counts = layer_counts[gen]
            gen_fitnesses = []
            for rep in range(n_reps):
                rep_counts = gen_counts[rep]
                fitnesses = np.array(layer_fits)
                fitnesses = z_normalize(fitnesses)
                fitnesses = fitnesses / np.max(fitnesses)
                avg_fitness = np.sum(fitnesses * rep_counts) / np.sum(rep_counts)
                gen_fitnesses.append(avg_fitness)
            avg_fitness_by_rep.append(gen_fitnesses)
        # Add the replicate information to the layer
        avg_fitness_over_time[layer] = avg_fitness_by_rep

    # Plot the growtih in fitness over time for each layer and replicate
    save_every = 1
    x_vals = np.arange(0, n_gens + 1, save_every)
    
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-darkgrid')
    for layer in range(1, len(avg_fitness_over_time.keys())):
        layer_avg_fitness = np.array(avg_fitness_over_time[layer])
        for rep in range(n_reps):
            #print(x_vals.shape, layer_avg_fitness[:, rep].shape)
            plt.plot(x_vals, layer_avg_fitness[:, rep], label=f'Layer {layer} Replicate {rep + 1}')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness over Generations for Each Layer and Replicate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    
def plot_inferred_vs_true_sel(sim_data):
    # The selections of every coefficients, and the inferred selections of every coefficient across layer and generation
    detailed_selection_results = sim_data[2]
    all_sel_coeffs = sim_data[1]
    n_reps = detailed_selection_results[0][0].shape[0]
    for layer in detailed_selection_results.keys():
        layer_selection = layer
        true_selection = all_sel_coeffs[layer_selection]
        normalized_selection = z_normalize(true_selection)

        for gen in range(len(detailed_selection_results[0])):
            inferred_selection = detailed_selection_results[layer_selection][gen]
            
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            plt.style.use('seaborn-v0_8-darkgrid')
            for rep in range(n_reps):
                x = true_selection
                y = z_normalize(inferred_selection[rep])
                
                
                axs[rep].scatter(x, y, alpha=0.5)
                axs[rep].set_title(f'Layer {layer_selection} Generation {gen * 2 + 1} Replicate {rep + 1}')
                axs[rep].set_xlabel('True Selection Coefficients')
                axs[rep].set_ylabel('Inferred Selection Coefficients')
                """axs[rep].plot([min(normalized_selection), max(normalized_selection)],
                            [min(normalized_selection), max(normalized_selection)],
                            color='red', linestyle='--')"""
            
                axs[rep].set_xlim(-0.05, 0.12)
            
            plt.tight_layout()
            plt.show()

def ordered_cov_plots(sim_data, layer=0):
    layer_df = get_df_selection(random_seed=42, selected_layer=layer,
                                    normalize_embeddings=False)[0]
    layer_df = normalize_embeddings(layer_df)
    embeddings = np.vstack(layer_df['Embedding'].tolist())

    # Get the selection coefficients
    true_selection = sim_data[1][layer]
    layer_generation_counts = sim_data[3][layer]
    inferred_selection = sim_data[2][layer][0]
    best_dim_idx = np.argmax(np.abs(true_selection))

    # Find the covariance between all embeddings with the best dimension
    best_dim_values = embeddings[:, best_dim_idx]
    
    for gen in range(50):
        print(f"Generation: {gen}")
        # Find the covariance using a weighted approach and the population from the first selection event (gen=1)
        weights = np.average(layer_generation_counts[gen], axis=0)
        covariances = []
        for dim in range(embeddings.shape[1]):
            dim_values = embeddings[:, dim]
            mean_best = np.average(best_dim_values, weights=weights)
            mean_dim = np.average(dim_values, weights=weights)
            covariance = np.average((best_dim_values - mean_best) * (dim_values - mean_dim), weights=weights)
            covariances.append(covariance)
        covariances = np.array(covariances)

        # Plot the selection coefficients in order of rank, colored by covariance with best dimension
        # Plot all three replicates
        plt.figure(figsize=(15, 5))
        plt.style.use('seaborn-v0_8-darkgrid')
        for rep in range(3):
            rep_inf_sel = inferred_selection[rep]
            
            normalized_selection = z_normalize(rep_inf_sel)
            
            # Use this:
            sorted_indices = np.argsort(normalized_selection)[::-1]  # Sort descending
            x_vals = np.arange(len(normalized_selection))  # Simple 0, 1, 2, ... for x-axis
            y_vals = normalized_selection[sorted_indices]  # Values in descending order
            covariances_sorted = covariances[sorted_indices]  # Sort covariances to match
            
            plt.subplot(1, 3, rep + 1)
            scatter = plt.scatter(x_vals, y_vals, c=covariances_sorted, cmap='coolwarm', alpha=0.7)
            plt.colorbar(scatter, label='Covariance with Best Dimension')
            plt.title(f'Layer {layer} Replicate {rep + 1}')
            plt.xlabel('Rank of Inferred Selection Coefficient')
            plt.ylabel('Inferred Selection Coefficient (Normalized)')
            plt.axhline(0, color='black', linestyle='--')
            
            
            # For the best dimension highlight:
            best_dim_position = np.where(sorted_indices == best_dim_idx)[0]
            plt.scatter(best_dim_position, normalized_selection[best_dim_idx],
                        color='yellow', edgecolor='black', s=100, label='Best Dimension')
            

        plt.tight_layout()
        plt.show()
    

from scipy.stats import rankdata, spearmanr

def comp_inf_vs_real_fits_rank(sim_data, layer, fitness='plus1'):
    """Compare the inferred and real fitness scores using ranks."""
    # For each layer, plot the fitness growth over time
    all_layer_fits = sim_data[0][layer]
    all_gen_counts = sim_data[3]
    embeddings = sim_data[4][layer]
    n_gens = len(all_gen_counts[0]) - 1
    n_reps = len(all_gen_counts[0][0])

    # Compute inferred fitness
    inferred_fits = calc_inferred_fits(sim_data, fitness=fitness,
                                       layer=layer)

    # Real fits (same across replicates)
    real_fits = np.array(all_layer_fits)

    # Rank-transform helper (average ranks for ties, 1-indexed)
    def rank_transform(x):
        return rankdata(x, method='average')

    # Rank the real fits once
    real_ranks = rank_transform(real_fits)
    n_variants = len(real_fits)

    # Rank inferred fits per replicate
    rep_ranks = {}
    for rep in range(n_reps):
        rep_ranks[rep] = rank_transform(inferred_fits[:, rep])

    # Plot — create figure with constrained_layout instead of tight_layout
    plt.close('all')
    fig, axes = plt.subplots(1, n_reps,
                             figsize=(6 * n_reps, 6))#,
                             #squeeze=False,
                             #constrained_layout=True)
    axes = axes.flatten()

    all_real_ranks = []
    all_inf_ranks = []

    for rep in range(n_reps):
        ax = axes[rep]
        r = real_ranks
        inf = rep_ranks[rep]

        ax.scatter(r, inf, alpha=0.5)
        ax.set_xlabel('Real Fitness (Rank)')
        ax.set_ylabel('Inferred Fitness (Rank)')
        ax.set_title(f'Layer {layer} — Replicate {rep}')
        ax.grid(True, alpha=0.3)

        # Diagonal reference line
        ax.plot([1, n_variants], [1, n_variants], color='red', linestyle='--')
        ax.set_xlim(0, n_variants + 1)
        ax.set_ylim(0, n_variants + 1)

        # Spearman rho
        rho, pval = spearmanr(real_fits, inferred_fits[:, rep])
        ax.annotate(f'\rho = {rho:.3f}\np = {pval:.2e}',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     ha='left', va='top',
                     fontsize=11, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        all_real_ranks.extend(r)
        all_inf_ranks.extend(inf)

    fig.suptitle(f'Real vs Inferred Fitness (Rank) — Layer {layer}',
                 fontsize=14)
    #fig.savefig(f'rank_comparison_layer_{layer}.png', dpi=100, bbox_inches='tight')
    plt.show()
    #plt.close(fig)

    # Overall Spearman correlation across all replicates
    overall_rho, overall_pval = spearmanr(all_real_ranks, all_inf_ranks)
    print(f'Overall Spearman ρ = {overall_rho:.4f}, p = {overall_pval:.2e}')
    return rep_ranks, overall_rho

## SELECTION FUNCTIONS ############
def gaussian_selection(embeddings):
    width = 0.02
    center = 0.0
    sel_coeffs = np.random.normal(loc=center, scale=width, size=embeddings.shape[1])
    return sel_coeffs

def zero_selection(embeddings):
    return np.zeros(embeddings.shape[1])

def generate_selection(embeddings):
    embedding_ranges = embeddings.max(axis=0) - embeddings.min(axis=0)
    # Find the indices of the top, lowest, and middle range dimensions
    selection_coefficients = np.zeros(embeddings.shape[1])
    sorted_indices = np.argsort(embedding_ranges)
    high_range_idx = sorted_indices[-1]
    # Give these dimensions higher selection coefficients
    selection_coefficients[high_range_idx] = 0.10
    return selection_coefficients


## DEFINITIONS (DIRTY NOW, CLEAN UP LATER)
def get_layer_df_piecewise(layer, in_paths, normalize=True):
    """Get the combined layer_df from multiple input sources, together"""
    dfs = []
    for in_path in in_paths:
        df = pickle.load(open(f"{in_path}/layer{layer}/inference_df.pkl", 'rb'))
        dfs.append(df) 
    layer_df = dfs[0].copy()
    num_reps = len(layer_df["Replicate"].unique())
    
    """print(layer_df.columns)"""
    
    total_paths = len(in_paths)
    for path_idx in range(1, total_paths):
        df_to_add = dfs[path_idx].copy()
        df_to_add["Replicate"] = df_to_add["Replicate"] + path_idx * num_reps
        layer_df = pd.concat([layer_df, df_to_add], ignore_index=True)

    if normalize:
        embeddings = np.array([x for x in layer_df["Embedding"].to_list()])
        dimensions = embeddings.shape[1]
        for dim in range(dimensions):
            embeddings[:, dim] = z_normalize(embeddings[:, dim])
        layer_df["Embedding"] = [embeddings[i] for i in range(embeddings.shape[0])]

    return layer_df

def convert_long_to_wide(df):
    """Convert layer_df from long format to wide format.
    
    Before: one row per (Embedding, Replicate, Generation) with a Frequency column.
    After:  one row per unique Embedding with columns Rep{i}_PreNums / Rep{i}_PostNums.
    
    Generation 0 -> PreNums, Generation 1 -> PostNums.
    Replicates are 0-indexed in the input and 1-indexed in the output.
    """
    gen_map = {0: 'PreNums', 1: 'PostNums'}
    # Use tuple as a hashable embedding key
    df = df.copy()
    df['_emb_key'] = df['Embedding'].apply(tuple)
    # Pivot Frequency into (Replicate, Generation) columns
    wide = df.pivot_table(
        index='_emb_key',
        columns=['Replicate', 'Generation'],
        values='Frequency',
        aggfunc='sum'
    ).fillna(0)
    # Flatten and rename columns: (rep, gen) -> Rep{rep+1}_{PreNums|PostNums}
    wide.columns = [
        f'Rep{rep}_{gen_map[gen]}'
        for rep, gen in wide.columns
    ]
    wide = wide.reset_index()
    # Restore numpy arrays and drop the temp key
    wide['Embedding'] = wide['_emb_key'].apply(np.array)
    wide = wide.drop(columns='_emb_key')
    # Reorder: all Pre/Post columns first, then Embedding
    rep_cols = [c for c in wide.columns if c != 'Embedding']
    wide = wide[rep_cols + ['Embedding']].reset_index(drop=True)
    return wide

def get_simulation_results_piecewise(n_layers, n_gens, n_reps,
                                    in_paths, 
                                    sel_func=generate_selection,
                                    inference=True, fitness='plus1',
                                    save_every=1):
    all_layer_fits = {}
    all_selection_coefficients = {}
    detailed_selection_results = {}
    all_generation_counts = {}
    
    for layer in range(n_layers):
        print(f"Running layer {layer}...")
        
        df_layer = get_layer_df_piecewise(layer, in_paths, normalize=True)
        df_selection = convert_long_to_wide(df_layer)
        
        n_reps = len(df_selection.columns) // 2  # Assuming each replicate has PreNums and PostNums
        
        initial_counts = []
        for rep in range(n_reps):
            pre_col = f'Rep{rep + 1}_PreNums'
            if pre_col not in df_selection.columns:
                raise ValueError(f"Expected column {pre_col} not found in df_selection: {df_selection.columns}")
            initial_counts.append(df_selection[pre_col].values)
        

        # Generate selection coefficients using the provided function
        embedding_matrix = np.vstack(df_selection['Embedding'].values)
        selection_coefficients = sel_func(embedding_matrix)
        all_selection_coefficients[layer] = selection_coefficients
        
        print("Running simulation...")
        generation_counts, layer_fits = run_simulation(df_selection, selection_coefficients, 
                                           initial_counts, n_gens=n_gens, save_every=save_every,
                                           fitness=fitness)
        all_generation_counts[layer] = generation_counts
        all_layer_fits[layer] = layer_fits

        if inference:
            gen=n_gens
            print(f"  Analyzing generation {gen}...")
            layer_results = []
            test_path = pwd + f"/simulations/layer_{layer}_gen_{gen}/"
            data = run_inference_calcs_sims(df_selection, generation_counts[:gen + 1], 
                                            test_path, save_output=False)
            found_sel_coeffs = data[2]
            layer_results.append(found_sel_coeffs)
            detailed_selection_results[layer] = layer_results

    return all_layer_fits, all_selection_coefficients, detailed_selection_results, all_generation_counts, embedding_matrix


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