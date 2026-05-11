import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from esmdmsfunctions import (
    CODON2AA,
    load_final_df,
    z_normalize
)

from paperPop import (
    MAVEDB_NT,
    MAVEDB_WT,
    MAVEDB_ACC,
    MAVEDB_SPLICE,
    read_fancy_comments,
    get_variant_sites_nucs
)

NCOLS = 6
_REP_COLORS = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:brown"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def inferred_fitness(embeddings, sel_coefs, fitness_fn="exp"):
    dot = embeddings @ sel_coefs
    if fitness_fn == "exp":
        return np.exp(dot)
    elif fitness_fn == "plus1":
        return np.maximum(1.0 + dot, 0.0)
    raise ValueError(f"Unknown fitness function: {fitness_fn}")


def get_embeddings(embedding_path, layer):
    df = load_final_df(layer, embedding_path)
    return np.vstack(df["Embedding"].tolist())  # (N, D)


def _scatter_panel(ax, true_z, inf_z, title, color="steelblue"):
    r, _ = pearsonr(true_z, inf_z)
    ax.scatter(true_z, inf_z, alpha=0.4, s=8, color=color, rasterized=True)
    lo = min(true_z.min(), inf_z.min()) - 0.3
    hi = max(true_z.max(), inf_z.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("True (z)", fontsize=7)
    ax.set_ylabel("Inferred (z)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.93), xycoords="axes fraction",
                ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
    return r


def _pearson_summary_plot(ax, layers, rs_joint, rs_per_rep, title, ylabel):
    n_reps = len(rs_per_rep)
    rep_colors = plt.cm.Blues(np.linspace(0.4, 0.75, n_reps))
    for rep in range(n_reps):
        ax.plot(layers, rs_per_rep[rep], color=rep_colors[rep],
                linewidth=1.2, alpha=0.7, linestyle="--", label=f"Rep {rep + 1}")
    ax.plot(layers, rs_joint, color="seagreen", linewidth=2.5,
            marker="o", markersize=4, label="s_joint")
    ax.set_xlabel("ESM-2 Layer", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(layers[::5])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_true_vs_inferred_fitness(all_results, paths, cfg, output_dir, n_cols=NCOLS):
    """True vs inferred fitness across all layers: per-rep scatter grids + Pearson summary."""
    fitness_fn = cfg.get("fitness_fn", "exp") if cfg else "exp"
    os.makedirs(output_dir, exist_ok=True)

    for path_name, results in all_results.items():
        all_layer_fits   = results[0]
        detailed_results = results[2]
        embedding_path   = paths[path_name]

        layers   = sorted(detailed_results.keys())
        n_layers = len(layers)
        n_reps   = len(detailed_results[layers[0]][0])
        nrows    = (n_layers + n_cols - 1) // n_cols

        for rep in range(n_reps):
            fig, axes = plt.subplots(nrows, n_cols, figsize=(4.5 * n_cols, 4 * nrows), squeeze=False)
            fig.suptitle(f"Inferred (s, rep {rep + 1}) vs True Fitness — {path_name}", fontsize=14, y=1.01)
            for idx, layer in enumerate(layers):
                row, col = divmod(idx, n_cols)
                true_z = z_normalize(np.array(all_layer_fits[layer]))
                s      = detailed_results[layer][0]
                emb    = get_embeddings(embedding_path, layer)
                inf_z  = z_normalize(inferred_fitness(emb, s[rep], fitness_fn))
                _scatter_panel(axes[row, col], true_z, inf_z, title=f"Layer {layer}")
            for idx in range(n_layers, nrows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row, col].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"fitness_scatter_rep{rep + 1}_{path_name}.png"),
                        dpi=80, bbox_inches="tight")
            plt.close()

        fig, axes = plt.subplots(nrows, n_cols, figsize=(4.5 * n_cols, 4 * nrows), squeeze=False)
        fig.suptitle(f"s_joint Inferred vs True Fitness — {path_name}", fontsize=14, y=1.01)
        for idx, layer in enumerate(layers):
            row, col = divmod(idx, n_cols)
            true_z  = z_normalize(np.array(all_layer_fits[layer]))
            s_joint = detailed_results[layer][1]
            emb     = get_embeddings(embedding_path, layer)
            inf_z   = z_normalize(inferred_fitness(emb, s_joint, fitness_fn))
            _scatter_panel(axes[row, col], true_z, inf_z, title=f"Layer {layer}", color="seagreen")
        for idx in range(n_layers, nrows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"fitness_scatter_sjoint_{path_name}.png"),
                    dpi=80, bbox_inches="tight")
        plt.close()

        rs_joint, rs_per_rep = [], [[] for _ in range(n_reps)]
        for layer in layers:
            true_z  = z_normalize(np.array(all_layer_fits[layer]))
            emb     = get_embeddings(embedding_path, layer)
            s_joint = detailed_results[layer][1]
            rs_joint.append(pearsonr(true_z, z_normalize(inferred_fitness(emb, s_joint, fitness_fn)))[0])
            s = detailed_results[layer][0]
            for rep in range(n_reps):
                rs_per_rep[rep].append(
                    pearsonr(true_z, z_normalize(inferred_fitness(emb, s[rep], fitness_fn)))[0]
                )

        fig, ax = plt.subplots(figsize=(7, 5))
        _pearson_summary_plot(ax, layers, rs_joint, rs_per_rep,
                              title=f"Dataset: {path_name}",
                              ylabel="Pearson r (inferred vs true fitness)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pearson_r_fitness_{path_name}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close()


def plot_true_vs_inferred_sel_coeffs(all_results, paths, cfg, output_dir, n_cols=NCOLS):
    """True vs inferred selection coefficients across all layers: scatter grid + Pearson summary."""
    os.makedirs(output_dir, exist_ok=True)

    for path_name, results in all_results.items():
        true_sel_coefs   = results[1]
        detailed_results = results[2]

        if true_sel_coefs is None:
            print(f"[{path_name}] No true selection coefficients available, skipping.")
            continue

        layers   = sorted(detailed_results.keys())
        n_layers = len(layers)
        n_reps   = len(detailed_results[layers[0]][0])
        nrows    = (n_layers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(nrows, n_cols, figsize=(4.5 * n_cols, 4 * nrows), squeeze=False)
        fig.suptitle(f"Inferred vs True Selection Coefficients — {path_name}", fontsize=14, y=1.01)
        for idx, layer in enumerate(layers):
            row, col = divmod(idx, n_cols)
            ax      = axes[row, col]
            s_true  = true_sel_coefs[layer]
            s       = detailed_results[layer][0]
            s_joint = detailed_results[layer][1]
            for rep in range(n_reps):
                r, _ = pearsonr(s_true, s[rep])
                ax.scatter(s_true, s[rep], color=_REP_COLORS[rep % len(_REP_COLORS)],
                           alpha=0.4, s=6, rasterized=True, label=f"Rep {rep + 1} (r={r:.2f})")
            r_joint, _ = pearsonr(s_true, s_joint)
            ax.scatter(s_true, s_joint, color="black", alpha=0.6, s=6, marker="D",
                       rasterized=True, label=f"s_joint (r={r_joint:.2f})")
            ax.set_title(f"Layer {layer}", fontsize=9)
            ax.set_xlabel("True s", fontsize=7)
            ax.set_ylabel("Inferred s", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, markerscale=1.5, loc="upper left")
        for idx in range(n_layers, nrows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sel_coef_scatter_{path_name}.png"),
                    dpi=80, bbox_inches="tight")
        plt.close()

        rs_joint, rs_per_rep = [], [[] for _ in range(n_reps)]
        for layer in layers:
            s_true  = true_sel_coefs[layer]
            s       = detailed_results[layer][0]
            s_joint = detailed_results[layer][1]
            rs_joint.append(pearsonr(s_true, s_joint)[0])
            for rep in range(n_reps):
                rs_per_rep[rep].append(pearsonr(s_true, s[rep])[0])

        fig, ax = plt.subplots(figsize=(7, 5))
        _pearson_summary_plot(ax, layers, rs_joint, rs_per_rep,
                              title=f"Dataset: {path_name}",
                              ylabel="Pearson r (inferred s vs true s)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pearson_r_sel_coef_{path_name}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close()


def plot_cross_replicate_consistency(all_results, layers, max_cols=6, output_dir=None, corr='pearson', normalize="none"):
    """Cross-replicate consistency of inferred selection coefficients.

    Produces two complementary plots per layer:
      *_heatmap.png  — symmetric n_reps × n_reps correlation matrix (compact overview)
      *_scatter.png  — individual scatter plot for every replicate pair (detailed view)

    corr : 'pearson' or 'spearman'
    """
    if corr == 'spearman':
        corr_fn, corr_sym = spearmanr, 'ρ'
    else:
        corr_fn, corr_sym = pearsonr, 'r'

    def _norm_annotation(fig):
        if normalize != "none":
            fig.text(0.5, -0.01, f"Normalization: {normalize}", ha="center",
                     fontsize=9, style="italic", color="gray")

    def _save(fig, path):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, path), bbox_inches="tight", dpi=100)
        plt.show()
        plt.close(fig)

    for path_name, results in all_results.items():
        detailed_results = results[2]

        for layer in layers:
            if layer not in detailed_results:
                print(f"[{path_name}]  Layer {layer}: no inference results")
                continue

            s_reps = detailed_results[layer][0]
            n_reps = len(s_reps)
            print(f"n_reps for {path_name} layer {layer}: {n_reps}")

            # Compute all pairwise correlations once; reused by both plots
            corr_matrix   = np.eye(n_reps)
            pair_vals     = {}   # (i, j) -> (val, pval)
            off_diag_vals = []
            for i in range(n_reps):
                for j in range(i + 1, n_reps):
                    val, pval = corr_fn(s_reps[i], s_reps[j])
                    corr_matrix[i, j] = corr_matrix[j, i] = val
                    pair_vals[(i, j)] = (val, pval)
                    off_diag_vals.append(val)

            mean_r = np.mean(off_diag_vals)
            std_r  = np.std(off_diag_vals)
            suptitle_base = f"[{path_name}]  Layer {layer} — Cross-replicate consistency"

            # ── Heatmap ──────────────────────────────────────────────────────
            cell_in = np.clip(5.5 / n_reps, 0.45, 0.85)
            fig_w   = n_reps * cell_in + 1.8
            fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.88))

            im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
            plt.colorbar(im, ax=ax, label=corr_sym, fraction=0.046, pad=0.04)

            fs = int(np.clip(54 / n_reps, 6, 11))
            for i in range(n_reps):
                for j in range(n_reps):
                    v = corr_matrix[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=fs, color="white" if abs(v) > 0.6 else "black")

            tick_labels = [f"Rep {i + 1}" for i in range(n_reps)]
            ax.set_xticks(range(n_reps))
            ax.set_yticks(range(n_reps))
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=fs)
            ax.set_yticklabels(tick_labels, fontsize=fs)
            ax.set_title(
                f"{suptitle_base}\nmean {corr_sym} = {mean_r:.3f} ± {std_r:.3f}"
                f"  ({len(off_diag_vals)} pairs)",
                fontsize=10,
            )
            plt.tight_layout()
            _norm_annotation(fig)
            _save(fig, f"{path_name}_layer{layer}_cross_replicate_heatmap.png")

            # ── Scatter grid ─────────────────────────────────────────────────
            rep_pairs = [(i, j) for i in range(n_reps) for j in range(i + 1, n_reps)]
            n_plots   = len(rep_pairs)
            n_cols    = min(n_plots, max_cols)
            n_rows    = int(np.ceil(n_plots / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
            fig.suptitle(suptitle_base, fontsize=12)

            for idx, (ri, rj) in enumerate(rep_pairs):
                row, col = divmod(idx, n_cols)
                ax = axes[row][col]
                si = z_normalize(s_reps[ri])
                sj = z_normalize(s_reps[rj])
                val, pval = pair_vals[(ri, rj)]
                ax.scatter(si, sj, alpha=0.5, s=6, rasterized=True)
                lim = max(np.abs(si).max(), np.abs(sj).max()) + 0.5
                ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8)
                ax.set_xlabel(f"Rep {ri + 1}", fontsize=7)
                ax.set_ylabel(f"Rep {rj + 1}", fontsize=7)
                ax.set_title(f"{corr_sym} = {val:.3f}", fontsize=8)
                ax.tick_params(labelsize=6)
                ax.axis("equal")

            for idx in range(n_plots, n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row][col].set_visible(False)

            plt.tight_layout()
            _norm_annotation(fig)
            _save(fig, f"{path_name}_layer{layer}_cross_replicate_scatter.png")


def plot_fitness_trajectories(all_results, n_cols=6, output_dir=None):
    """Mean fitness over time for each layer and replicate."""
    for path_name, res in all_results.items():
        layers   = sorted(res[4].keys())
        n_layers = len(layers)
        n_rows   = int(np.ceil(n_layers / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 3.5 * n_rows),
                                 constrained_layout=True)
        axes = np.array(axes).flatten()

        for ax_idx, layer in enumerate(layers):
            ax = axes[ax_idx]
            generation_counts = res[4][layer]
            layer_fits = res[0][layer]
            n_reps = len(generation_counts[0])

            for rep in range(n_reps):
                gen_counts_rep = [gen_counts[rep] for gen_counts in generation_counts]
                rep_mean_fits = []
                for gen_counts in gen_counts_rep:
                    total_count = np.sum(gen_counts)
                    if total_count > 0:
                        freqs = gen_counts / total_count
                        rep_mean_fits.append(np.sum(freqs * layer_fits))
                    else:
                        rep_mean_fits.append(0.0)
                ax.plot(rep_mean_fits, lw=1.2, label=f"Rep {rep+1}")

            ax.set_title(f"Layer {layer}", fontsize=9)
            ax.set_xlabel("Generation", fontsize=8)
            ax.set_ylabel("Mean Fitness", fontsize=8)
            ax.tick_params(labelsize=7)
            if ax_idx == 0:
                ax.legend(fontsize=7)

        for ax in axes[n_layers:]:
            ax.set_visible(False)

        fig.suptitle(f"Mean Fitness Trajectories — {path_name}", fontsize=13)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{path_name}_fitness_trajectories.png"))
        plt.show()


def get_esm_individual_fitness_values(embedding_path, layer, s_joint, fitness_fn='plus1'):
    """Compute ESM-DMS inferred fitness for every sequence in the embedding df.

    Returns:
        pd.Series with protein sequence as index and inferred fitness as values.
    """
    directory_name = os.path.basename(embedding_path.rstrip("/"))
    emb_path = os.path.join(embedding_path, f"{directory_name}_embeddings.pkl")
    emb_full = pd.read_pickle(emb_path).drop_duplicates(subset="ProteinSequence")
    protein_seqs = emb_full["ProteinSequence"].values
    emb_matrix   = np.vstack([np.array(e[layer]) for e in emb_full["Embedding"]])
    fits = inferred_fitness(emb_matrix, s_joint, fitness_fn)
    return pd.Series(fits, index=protein_seqs)


def get_enrichment_ratios(embedding_path, agg='mean'):
    """Compute log enrichment ratio (post/pre frequency) per protein sequence.

    Loads the original embedding pickle (which has Replicate, Generation, Frequency,
    ProteinSequence columns), averages frequencies across replicates at the first and
    last observed generation, then returns log(freq_post / freq_pre).

    Returns:
        pd.Series indexed by protein_seq, NaN-dropped.
    """
    directory_name = os.path.basename(embedding_path.rstrip("/"))
    emb_path = os.path.join(embedding_path, f"{directory_name}_embeddings.pkl")
    emb_full = pd.read_pickle(emb_path)

    gen_min = emb_full["Generation"].min()
    gen_max = emb_full["Generation"].max()

    groupby_fn = "mean" if agg == "mean" else "sum"
    pre  = getattr(emb_full[emb_full["Generation"] == gen_min]
                   .groupby("ProteinSequence")["Frequency"], groupby_fn)()
    post = getattr(emb_full[emb_full["Generation"] == gen_max]
                   .groupby("ProteinSequence")["Frequency"], groupby_fn)()

    pre, post = pre.align(post, join="inner")
    enrichment = np.log(post / pre.replace(0.0, np.nan))
    return enrichment.dropna()


def _align_pair(baseline, esm):
    """Inner-join two Series on index, drop any remaining NaN, return matched arrays."""
    b, e = baseline.align(esm, join="inner")
    b = b.dropna()
    e = e[b.index].dropna()
    b = b[e.index]
    return b.values, e.values


def plot_baseline_vs_esm_comparison(baseline_fits, esm_fits_by_layer, layers,
                                    output_dir, path_name, baseline_label="popDMS Fitness"):
    """Plot a baseline fitness measure vs ESM-DMS inferred fitness.

    Produces:
      1. Pearson r and Spearman rho vs ESM layer (line plot).
      2. Scatter plots for the best-Pearson-r layer plus layers 0, 15, 30.

    baseline_label controls axis labels and output filenames.
    """
    file_tag = baseline_label.lower().replace(" ", "_")

    pearson_rs, spearman_rs, valid_layers = [], [], []
    for layer in layers:
        esm = esm_fits_by_layer.get(layer)
        if esm is None:
            continue
        x, y = _align_pair(baseline_fits, esm)
        if len(x) < 3:
            continue
        pr, _ = pearsonr(x, y)
        sr, _ = spearmanr(x, y)
        pearson_rs.append(pr)
        spearman_rs.append(sr)
        valid_layers.append(layer)

    if not valid_layers:
        print(f"[{path_name}] No valid layers for {baseline_label} vs ESM comparison.")
        return

    layers_arr   = np.array(valid_layers)
    pearson_arr  = np.array(pearson_rs)
    spearman_arr = np.array(spearman_rs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers_arr, pearson_arr,  color="steelblue",  marker="o", markersize=4,
            linewidth=2, label="Pearson r")
    ax.plot(layers_arr, spearman_arr, color="darkorange", marker="s", markersize=4,
            linewidth=2, label="Spearman ρ")
    ax.set_xlabel("ESM-2 Layer", fontsize=12)
    ax.set_ylabel("Correlation", fontsize=12)
    ax.set_title(f"{baseline_label} vs ESM-DMS Correlation — {path_name}", fontsize=13)
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(layers_arr[::5])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_tag}_vs_esm_correlation_{path_name}.png"),
                dpi=100, bbox_inches="tight")
    plt.close()

    best_layer    = valid_layers[int(np.argmax(pearson_arr))]
    select_layers = [l for l in [0, 15, 30] if l in esm_fits_by_layer]
    scatter_layers = sorted(set([best_layer] + select_layers))

    n_plots = len(scatter_layers)
    n_cols  = min(n_plots, 3)
    n_rows  = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    for idx, layer in enumerate(scatter_layers):
        row, col = divmod(idx, n_cols)
        ax  = axes[row][col]
        x, y = _align_pair(baseline_fits, esm_fits_by_layer[layer])
        pr, _ = pearsonr(x, y)
        sr, _ = spearmanr(x, y)
        color = "seagreen" if layer == best_layer else "steelblue"
        label = f"Layer {layer}" + (" (best)" if layer == best_layer else "")
        ax.scatter(x, y, alpha=0.4, s=8, color=color, rasterized=True)
        lo = min(x.min(), y.min()) - 0.05
        hi = max(x.max(), y.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(baseline_label, fontsize=8)
        ax.set_ylabel("ESM-DMS Fitness", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.annotate(f"r = {pr:.3f}\nρ = {sr:.3f}", xy=(0.05, 0.93),
                    xycoords="axes fraction", ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)
    fig.suptitle(f"{baseline_label} vs ESM-DMS Fitness — {path_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_tag}_vs_esm_scatter_{path_name}.png"),
                dpi=80, bbox_inches="tight")
    plt.close()


def get_individual_fitness_values(selection_file, haplotype_counts_file, reference_sequence, rep='joint', comment_char=None):
    '''
    Compute fitness for every individual haplotype in the population.
    
    Fitness is defined as:
        w = 1 + sum_i(x_i * s_i)
    where x_i = 1 if the individual carries mutation i, 0 otherwise,
    and s_i is the selection coefficient for mutation i.
    
    Arguments:
        - selection_file:        Path to selection coefficients .csv.gz from infer_correlated
        - haplotype_counts_file: Path to the MaveDB format haplotype counts file
        - reference_sequence:    Reference nucleotide sequence string
        - rep:                   Which replicate to use for selection coefficients (default: 'joint')
        - comment_char:          Comment character in haplotype counts file (default: None)
    
    Returns:
        - fitness: np.array of shape (n_haplotypes,) with one fitness value per individual,
                   weighted by haplotype count
    '''
    
    # Load selection coefficients (excluding WT, as they contribute 0)
    df_sel = pd.read_csv(selection_file, compression='gzip')
    df_sel = df_sel[df_sel['WT_indicator'] == False]
    
    if rep not in df_sel.columns:
        raise ValueError(f"Column '{rep}' not found. Available: {list(df_sel.columns)}")
    
    # Build lookup: (site, aa) -> selection coefficient
    sel_lookup = {(row['site'], row['amino_acid']): row[rep]
                  for _, row in df_sel.iterrows()}
    
    # Load haplotype counts, drop unneeded columns, remove ambiguous nucleotides
    df_data = read_fancy_comments(haplotype_counts_file, comment_char=comment_char)
    df_data = df_data.replace('NA', np.nan).fillna(0).drop([MAVEDB_ACC], axis=1)
    if MAVEDB_SPLICE in df_data.columns:
        df_data = df_data.drop([MAVEDB_SPLICE], axis=1)
    df_data = df_data[~df_data[MAVEDB_NT].astype('str').str.contains('X', regex=False)]
    df_data.reset_index(drop=True, inplace=True)
    
    # Precompute reference codons and amino acids
    codon_length = 3
    ref_codons = np.array([''.join(reference_sequence[i:i+codon_length])
                           for i in range(0, len(reference_sequence), codon_length)])
    ref_aas = np.array([CODON2AA[c] for c in ref_codons])
    reference_sequence = list(reference_sequence)
    
    ref_protein = ''.join(ref_aas)

    # Compute fitness for each haplotype
    fitness_values = []
    protein_seqs   = []
    for _, row in df_data.iterrows():

        # WT individual carries no mutations, fitness = 1
        if row[MAVEDB_NT] == MAVEDB_WT:
            fitness_values.append(1.0)
            protein_seqs.append(ref_protein)
            continue

        # Reconstruct variant sequence and amino acids
        variant_sites, variant_nucs = get_variant_sites_nucs(row[MAVEDB_NT], shift_by_one=True)
        variant_sequence = reference_sequence.copy()
        for v_site, v_nuc in zip(variant_sites, variant_nucs):
            variant_sequence[v_site] = v_nuc
        variant_codons = np.array([''.join(variant_sequence[i:i+codon_length])
                                   for i in range(0, len(variant_sequence), codon_length)])
        variant_aas = np.array([CODON2AA[c] for c in variant_codons])

        # Sum selection coefficients over all mutated sites
        s_sum = 0.0
        for i, (var_aa, ref_aa) in enumerate(zip(variant_aas, ref_aas)):
            if var_aa != ref_aa:
                site = i + 1  # 1-indexed sites, consistent with infer_correlated
                s_sum += sel_lookup.get((site, var_aa), 0.0)

        fitness_values.append(1.0 + s_sum)
        protein_seqs.append(''.join(variant_aas))

    return pd.Series(fitness_values, index=protein_seqs)
