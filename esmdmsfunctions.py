import os
import shutil
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pickle
import time

import matplotlib.pyplot as plt

import popDMS
from importlib import reload

# reload popDMS
reload(popDMS)

## GLOBAL VARIABLES


pwd = os.getcwd()

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




def analyze_layers_piecewise(whole_df=None, in_path=None, output_path=None, embed_path=None, dataname='BF520',
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
    #nonzero_df = whole_df[whole_df["Embeddings"].notnull()].reset_index(drop=True)
    layer_count = 31 #nonzero_df["Embeddings"][0].shape[0]
    
    layer_dfs = []
    for layer in range(layer_count):
        
        if in_path is not None:
            layer_df = pickle.load(open(f"{in_path}/layer{layer}/inference_df.pkl", 'rb'))
        
        #print(layer_df.head())
        
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
        
            
        # data = [dx, icov, s, s_joint, sel_data, gamma_opt, x_array] for layer
        data = run_inference_calcs(layer_df, layer_path, verbose=verbose,
                                   pre_processed=True) 
        layer_results.append(data)
        

        
    return layer_results


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
        
        