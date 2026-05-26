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
                  post_weights: np.ndarray, layer = None):
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
                       post_weights: np.ndarray, layer = None):
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


## PLOTTING

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
            corr = pearsonr(s[rep_i], s[rep_j])[0]
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
    """all_gen_counts = sim_data[3]
    embeddings = sim_data[4][layer]
    n_gens = len(all_gen_counts[0]) - 1"""
    n_reps = len(sim_data[2][0][0])

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
    
    fig.suptitle(f'Real vs Inferred Fitness — Layer {layer}',
                 fontsize=14)
    #fig.savefig(f'rank_comparison_layer_{layer}.png', dpi=100, bbox_inches='tight')
    plt.show()
    #plt.close(fig)

    # Overall Spearman correlation across all replicates
    overall_rho, overall_pval = spearmanr(all_real, all_inferred)
    print(f'Overall Spearman ρ = {overall_rho:.4f}, p = {overall_pval:.2e}')
    return overall_rho



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
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    
def plot_inferred_vs_true_sel(sim_data):
    detailed_selection_results = sim_data[2]
    all_sel_coeffs = sim_data[1]
    n_reps = detailed_selection_results[0][0].shape[0]

    for layer in detailed_selection_results.keys():
        layer_selection = layer
        true_selection = all_sel_coeffs[layer_selection]
        normalized_selection = z_normalize(true_selection)

        for gen in range(len(detailed_selection_results[0])):
            inferred_selection = detailed_selection_results[layer_selection][gen]

            fig, axs = plt.subplots(1, n_reps, figsize=(6 * n_reps, 6))
            plt.style.use('seaborn-v0_8-darkgrid')

            # Ensure axs is always iterable (edge case: n_reps == 1)
            if n_reps == 1:
                axs = [axs]

            for rep in range(n_reps):
                x = true_selection
                y = z_normalize(inferred_selection[rep])
                axs[rep].scatter(x, y, alpha=0.5)
                axs[rep].set_title(f'Layer {layer_selection} Generation {gen * 2 + 1} Replicate {rep + 1}')
                axs[rep].set_xlabel('True Selection Coefficients')
                axs[rep].set_ylabel('Inferred Selection Coefficients')
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
    



def comp_inf_vs_real_fits_rank(sim_data, layer, fitness='plus1'):
    """Compare the inferred and real fitness scores using ranks."""
    # For each layer, plot the fitness growth over time
    all_layer_fits = sim_data[0][layer]
    #all_gen_counts = sim_data[3]
    #embeddings = sim_data[4][layer]
    #n_gens = len(all_gen_counts[0]) - 1
    n_reps = len(sim_data[3][0][0])

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



def get_simulation_results_piecewise(layer, n_gens,
                                    sel_func=generate_selection,
                                    inference=True, fitness='plus1',
                                    save_every=1):
    """Get the simulation results for the given layer"""
    
    print(f"Running layer {layer}...")
    df_selection = load_final_df(layer)
    n_reps = len(df_selection.columns) // 2  # Assuming each replicate has PreNums and PostNums
    initial_counts = []
    
    for rep in range(n_reps):
        pre_col = f'Rep{rep + 1}_PreNums'
        if pre_col not in df_selection.columns:
            raise ValueError(f"Expected column {pre_col} not found in df_selection: {df_selection.columns}")
        initial_counts.append(df_selection[pre_col].values)
    
    # Generate selection coefficients using the provided function
    embeddings_len = df_selection['Embedding'].iloc[0].shape[0]
    selection_coefficients = sel_func(embeddings_len)

    print("Running simulation...")
    generation_counts, layer_fits = run_simulation(df_selection, selection_coefficients, 
                                        initial_counts, n_gens=n_gens, 
                                        save_every=save_every, fitness=fitness)

    if inference:
        gen=n_gens
        print(f"  Analyzing generation {gen}...")
        layer_results = []
        test_path = None
        data = run_inference_calcs_sims(df_selection, generation_counts[:gen + 1], test_path)
        found_sel_coeffs = data[2]
        layer_results.append(found_sel_coeffs)

    embedding_matrix = np.vstack(df_selection['Embedding'].values)
    return layer_fits, selection_coefficients, layer_results, generation_counts, embedding_matrix, data

