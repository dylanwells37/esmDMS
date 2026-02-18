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
    
def run_inference_calcs(layer_df, output_path):
    """Run the inference calculations:
    WHOLE PIPELINE FROM READING IN EMBEDDINGS DATAFRAME
    """
    inference_df = embedding_df_transfer_optimized(layer_df)
    
    # make directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # save inference_df
    inference_df.to_pickle(output_path + 'inference_df.pkl')
    
    inference_df = pd.read_pickle(output_path + 'inference_df.pkl')    
    data = popDMS.mini_infer_independent_esm(inference_df, n_replicates=3,
                                            output_dir=output_path)
    return data



def analyze_layers(whole_df):
    nonzero_df = whole_df[whole_df["Embeddings"].notnull()].reset_index(drop=True)
    layer_count = nonzero_df["Embeddings"][0].shape[0]
    
    layer_dfs = []
    
    for layer in range(layer_count):
        layer_df = nonzero_df.copy()
        layer_df["Embeddings"] = layer_df["Embeddings"].apply(lambda x: x[layer])
        layer_dfs.append(layer_df)
        if not os.path.exists(pwd + f'/data/sequence_data/'):
            os.makedirs(pwd + f'/data/sequence_data/')
        
        # only make the pickle file if it doesn't exist
        if not os.path.exists(pwd + f'/data/sequence_data/BF520_protein_embeddings_layer{layer}.pkl'):
            layer_df.to_pickle(pwd + f'/data/sequence_data/BF520_protein_embeddings_layer{layer}.pkl')
        print(f"Wrote layer {layer} dataframe to pickle")
    
    
    layer_results = []
    for layer in range(layer_count):
        print(f"Analyzing layer {layer}")
        layer_df = layer_dfs[layer]
        output_path = pwd + f'/data/inference_results/layer{layer}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        data = run_inference_calcs(layer_df, output_path)
        layer_results.append(data)
        with open(output_path + 'inference_results.pkl', 'wb') as f:
            pickle.dump(data, f)
        print(f"Wrote inference results for layer {layer}")
    return layer_results


def embed_replicates_streaming(
    embedding_df: pd.DataFrame,
    output_path: str,
    embed_zeroes: bool = False,
    esm_model: str = "facebook/esm2_t30_150M_UR50D",
) -> None:
    """Embed the sequences in streaming mode, writing each embedding to file as it is computed."""
    start_time = time.time()
    pre_counts = embedding_df["PreNums"].to_list()
    post_counts = embedding_df["PostNums"].to_list()
    
    tokenizer = AutoTokenizer.from_pretrained(esm_model, do_lower_case=False)
    model = AutoModel.from_pretrained(esm_model)

    # Open the output file once for appending
    with open(output_path, "ab") as f_out:
        for i, (prot_sequence, pre) in enumerate(zip(embedding_df["ProteinSequence"], pre_counts)):
            if embed_zeroes or any(x > 0 for x in pre):
                embedding = embed_sequence(prot_sequence, tokenizer, model)
            else:
                embedding = None

            # Write a small dict to the pickle file
            record = {
                "Index": i,
                "ProteinSequence": prot_sequence,
                "PreNums": pre_counts[i],
                "PostNums": post_counts[i],
                "Embedding": embedding,
            }
            pickle.dump(record, f_out)

            # Progress logging
            if (i + 1) % 100 == 0:
                cur_time = time.time()
                elapsed = cur_time - start_time
                remaining = elapsed / (i + 1) * (len(embedding_df) - (i + 1))
                print(f"Embedded {i + 1}/{len(embedding_df)} sequences | Elapsed: {elapsed:.1f}s | ETA: {remaining/60:.1f} min")

    print(f"All embeddings written incrementally to {output_path}")
    
    
def read_embeddings_stream(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

records = list(read_embeddings_stream("embeddings.pkl"))
