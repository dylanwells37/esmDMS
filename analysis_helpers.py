import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, rankdata

sns.set_theme(style="darkgrid")

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


def plot_cross_replicate_consistency(all_results, layers, max_cols=6, output_dir=None, corr='pearson', normalize="none", every_n=1):
    """Cross-replicate consistency of inferred selection coefficients.

    Produces per-layer plots (heatmap + scatter) for every ``every_n``-th layer
    (index 0, n, 2n, …) and a summary line plot of mean correlation across all layers.

      *_heatmap.png  — symmetric n_reps × n_reps correlation matrix (compact overview)
      *_scatter.png  — individual scatter plot for every replicate pair (detailed view)
      *_summary.png  — mean ± std correlation vs layer across all layers

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

        # summary data: layer -> (mean_r, std_r)
        summary_stats = {}

        for layer_idx, layer in enumerate(layers):
            if layer not in detailed_results:
                print(f"[{path_name}]  Layer {layer}: no inference results")
                continue

            s_reps = detailed_results[layer][0]
            n_reps = len(s_reps)
            print(f"n_reps for {path_name} layer {layer}: {n_reps}")

            # Compute all pairwise correlations once; reused by both plots + summary
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
            summary_stats[layer] = (mean_r, std_r)
            suptitle_base = f"[{path_name}]  Layer {layer} — Cross-replicate consistency"

            # Only produce per-layer plots for every n-th layer
            if layer_idx % every_n != 0:
                continue

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

        # ── Summary: mean correlation across layers ───────────────────────
        if summary_stats:
            summ_layers = sorted(summary_stats.keys())
            mean_rs = [summary_stats[l][0] for l in summ_layers]
            std_rs  = [summary_stats[l][1] for l in summ_layers]

            fig, ax = plt.subplots(figsize=(max(6, len(summ_layers) * 0.35), 4))
            ax.plot(summ_layers, mean_rs, marker="o", linewidth=1.5, markersize=4)
            ax.fill_between(summ_layers,
                            [m - s for m, s in zip(mean_rs, std_rs)],
                            [m + s for m, s in zip(mean_rs, std_rs)],
                            alpha=0.25)
            ax.set_xlabel("Layer")
            ax.set_ylabel(f"Mean pairwise {corr_sym}")
            ax.set_title(f"[{path_name}]  Cross-replicate consistency across layers")
            ax.set_ylim(-1, 1)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            plt.tight_layout()
            _norm_annotation(fig)
            _save(fig, f"{path_name}_cross_replicate_summary.png")


def plot_shuffled_consistency(layer_stats_by_dataset, layers, output_dir=None, normalize="none"):
    """Compare per-layer shuffled cross-replicate correlations across multiple datasets.

    layer_stats_by_dataset : {dataset_name: {layer: (mean_r, std_r)}}

    Produces two plots:
      shuffled_consistency_overlay.png          — overlaid mean r vs layer, one line per dataset
      shuffled_consistency_profile_correlation.png — heatmap of how correlated the r-per-layer
                                                     profiles are across datasets (only if ≥2)
    """
    def _save(fig, path):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, path), bbox_inches="tight", dpi=100)
        plt.show()
        plt.close(fig)

    def _norm_annotation(fig):
        if normalize != "none":
            fig.text(0.5, -0.01, f"Normalization: {normalize}", ha="center",
                     fontsize=9, style="italic", color="gray")

    all_datasets = list(layer_stats_by_dataset.keys())

    # ── Overlay: mean r vs layer, one line per dataset ────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.35), 4))
    for name, stats in layer_stats_by_dataset.items():
        present = sorted(l for l in layers if l in stats)
        mean_rs = [stats[l][0] for l in present]
        std_rs  = [stats[l][1] for l in present]
        line, = ax.plot(present, mean_rs, marker="o", linewidth=1.5, markersize=4, label=name)
        ax.fill_between(present,
                        [m - s for m, s in zip(mean_rs, std_rs)],
                        [m + s for m, s in zip(mean_rs, std_rs)],
                        alpha=0.15, color=line.get_color())
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean pairwise r (shuffled data)")
    ax.set_title("Cross-replicate consistency on shuffled frequencies — by dataset")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    if len(all_datasets) > 1:
        ax.legend(fontsize=8)
    plt.tight_layout()
    _norm_annotation(fig)
    _save(fig, "shuffled_consistency_overlay.png")

    # ── Profile correlation heatmap (only meaningful with ≥2 datasets) ───
    if len(all_datasets) < 2:
        return

    profile_matrix = np.array([
        [layer_stats_by_dataset[name].get(l, (np.nan, np.nan))[0] for l in layers]
        for name in all_datasets
    ])
    n_ds = len(all_datasets)
    corr_mat = np.full((n_ds, n_ds), np.nan)
    np.fill_diagonal(corr_mat, 1.0)
    for i in range(n_ds):
        for j in range(i + 1, n_ds):
            mask = ~(np.isnan(profile_matrix[i]) | np.isnan(profile_matrix[j]))
            if mask.sum() >= 3:
                val, _ = pearsonr(profile_matrix[i][mask], profile_matrix[j][mask])
            else:
                val = np.nan
            corr_mat[i, j] = corr_mat[j, i] = val

    cell_in = np.clip(5.5 / n_ds, 0.45, 0.85)
    fig_w = n_ds * cell_in + 1.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.88))
    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    plt.colorbar(im, ax=ax, label="r", fraction=0.046, pad=0.04)
    fs = int(np.clip(54 / n_ds, 6, 11))
    for i in range(n_ds):
        for j in range(n_ds):
            v = corr_mat[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "n/a"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=fs, color="white" if (not np.isnan(v) and abs(v) > 0.6) else "black")
    ax.set_xticks(range(n_ds))
    ax.set_yticks(range(n_ds))
    ax.set_xticklabels(all_datasets, rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(all_datasets, fontsize=fs)
    ax.set_title("Correlation of shuffled-r profiles across datasets\n"
                 "(high r → same layers are spuriously correlated)")
    plt.tight_layout()
    _norm_annotation(fig)
    _save(fig, "shuffled_consistency_profile_correlation.png")


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


def get_esm_individual_fitness_values(embedding_path, layer, s_joint, fitness_fn='plus1',
                                      normalize='none'):
    """Compute ESM-DMS inferred fitness for every sequence in the embedding df.

    normalize must match the value used during inference so that embeddings are in
    the same space as s_joint.  Loads from compact format (seq_id_map + layer pkl)
    when available, falling back to the raw embeddings pkl.

    Returns:
        pd.Series with protein sequence as index and inferred fitness as values.
    """
    import pickle as _pickle
    seq_id_map_path = os.path.join(embedding_path, "seq_id_map.pkl")
    layer_emb_path  = os.path.join(embedding_path, f"layer{layer}_seq_to_emb.pkl")

    if os.path.exists(seq_id_map_path) and os.path.exists(layer_emb_path):
        with open(seq_id_map_path, "rb") as f:
            protein_seqs = np.array(_pickle.load(f))
        with open(layer_emb_path, "rb") as f:
            emb_matrix = _pickle.load(f).astype(float)  # (n_unique, L)
    else:
        directory_name = os.path.basename(embedding_path.rstrip("/"))
        emb_path = os.path.join(embedding_path, f"{directory_name}_embeddings.pkl")
        emb_full = pd.read_pickle(emb_path).drop_duplicates(subset="ProteinSequence")
        protein_seqs = emb_full["ProteinSequence"].values
        emb_matrix   = np.vstack([np.array(e[layer]) for e in emb_full["Embedding"]])

    if normalize == "by_layer":
        mean = np.mean(emb_matrix)
        std  = np.std(emb_matrix)
        emb_matrix = (emb_matrix - mean) / (std if std > 0 else 1.0)
    elif normalize == "by_layer_dim":
        means = np.mean(emb_matrix, axis=0)
        stds  = np.std(emb_matrix, axis=0)
        stds[stds == 0] = 1.0
        emb_matrix = (emb_matrix - means) / stds

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
    """Inner-join two Series on index, return matched arrays.

    Deduplicates by index (protein sequence) before aligning — synonymous variants
    share a protein sequence, so multiple haplotypes can map to the same index.
    Duplicate entries are collapsed to their mean.
    """
    b = baseline.groupby(level=0).mean()
    e = esm.groupby(level=0).mean()
    b, e = b.align(e, join="inner")
    mask = b.notna() & e.notna()
    return b[mask].values, e[mask].values


def plot_scatter_comparison(x_fits, y_fits, output_dir, path_name,
                            x_label="popDMS Fitness", y_label="Enrichment Ratio"):
    """Scatter plot directly comparing two per-sequence fitness measures."""
    x, y = _align_pair(x_fits, y_fits)
    if len(x) < 3:
        print(f"[{path_name}] Not enough overlapping sequences for {x_label} vs {y_label}.")
        return
    pr, _ = pearsonr(x, y)
    sr, _ = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.4, s=8, color="steelblue", rasterized=True)
    lo, hi = _ax_limits(x, y)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f"{x_label} vs {y_label} — {path_name}", fontsize=12)
    ax.annotate(f"r = {pr:.3f}\nρ = {sr:.3f}", xy=(0.05, 0.93),
                xycoords="axes fraction", ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
    plt.tight_layout()
    x_tag = x_label.lower().replace(" ", "_")
    y_tag = y_label.lower().replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"{x_tag}_vs_{y_tag}_{path_name}.png"),
                dpi=100, bbox_inches="tight")
    plt.close()


def _ax_limits(x, y, pad_frac=0.05):
    """Axis limits with range-proportional padding, safe for negative values."""
    lo_raw = min(x.min(), y.min())
    hi_raw = max(x.max(), y.max())
    pad = max(hi_raw - lo_raw, 1e-6) * pad_frac
    return lo_raw - pad, hi_raw + pad


def _plot_top_bottom_panel(ax, x, y, top_n, x_label, y_label, title, highlight_axis="x"):
    """Scatter all points; highlight top-N and bottom-N by the chosen axis.

    highlight_axis : 'x' ranks by x (baseline), 'y' ranks by y (ESM-DMS).
    """
    ref = x if highlight_axis == "x" else y
    order = np.argsort(ref)
    idx_bot = order[:top_n]
    idx_top = order[-top_n:]
    idx_mid = order[top_n:-top_n] if len(order) > 2 * top_n else np.array([], dtype=int)

    lo, hi = _ax_limits(x, y)

    ax.scatter(x[idx_mid], y[idx_mid], alpha=0.25, s=6,  color="gray",        rasterized=True, label="rest")
    ax.scatter(x[idx_bot], y[idx_bot], alpha=0.85, s=30, color="tab:purple",   rasterized=True, label=f"bottom {top_n}")
    ax.scatter(x[idx_top], y[idx_top], alpha=0.85, s=30, color="tab:orange",   rasterized=True, label=f"top {top_n}")
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, markerscale=1.4, loc="upper left")


def _plot_rank_panel(ax, x, y, x_label, y_label, title):
    """Scatter rank(x) vs rank(y); annotate with Spearman ρ."""
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    sr, _ = spearmanr(x, y)

    ax.scatter(rx, ry, alpha=0.35, s=6, color="steelblue", rasterized=True)
    n = len(rx)
    ax.plot([1, n], [1, n], "r--", linewidth=0.8)
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, n + 1)
    ax.set_xlabel(f"Rank: {x_label}", fontsize=8)
    ax.set_ylabel(f"Rank: {y_label}", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.annotate(f"ρ = {sr:.3f}", xy=(0.05, 0.93), xycoords="axes fraction",
                ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))


def plot_baseline_vs_esm_comparison(baseline_fits, esm_fits_by_layer, layers,
                                    output_dir, path_name, baseline_label="popDMS Fitness",
                                    z_norm_fits=False, top_n=10):
    """Plot a baseline fitness measure vs ESM-DMS inferred fitness.

    Produces:
      1. Pearson r and Spearman rho vs ESM layer (line plot).
      2. Scatter plots for the best-Pearson-r layer plus layers 0, 15, 30.

    baseline_label controls axis labels and output filenames.
    """
    file_tag = baseline_label.lower().replace(" ", "_")
    if z_norm_fits:
        title_suffix = " (z-normalized)"
    else:
        title_suffix = ""
    
    pearson_rs, spearman_rs, valid_layers = [], [], []
    for layer in layers:
        esm = esm_fits_by_layer.get(layer)
        if esm is None:
            continue
        x, y = _align_pair(baseline_fits, esm)
        if len(x) < 3:
            continue
        if z_norm_fits:
            x = (x - np.mean(x)) / (np.std(x) if np.std(x) > 0 else 1.0)
            y = (y - np.mean(y)) / (np.std(y) if np.std(y) > 0 else 1.0)
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
    plt.savefig(os.path.join(output_dir, f"{file_tag}_vs_esm_correlation_{path_name}{title_suffix}.png"),
                dpi=100, bbox_inches="tight")
    plt.close()

    best_layer    = valid_layers[int(np.argmax(pearson_arr))]
    select_layers = [l for l in [0, 15, 30] if l in esm_fits_by_layer]
    scatter_layers = sorted(set([best_layer] + select_layers))

    n_plots = len(scatter_layers)
    n_cols  = 2
    n_rows  = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    for idx, layer in enumerate(scatter_layers):
        row, col = divmod(idx, n_cols)
        ax  = axes[row][col]
        x, y = _align_pair(baseline_fits, esm_fits_by_layer[layer])
        if z_norm_fits:
            x = (x - np.mean(x)) / (np.std(x) if np.std(x) > 0 else 1.0)
            y = (y - np.mean(y)) / (np.std(y) if np.std(y) > 0 else 1.0)
        pr, _ = pearsonr(x, y)
        sr, _ = spearmanr(x, y)
        color = "seagreen" if layer == best_layer else "steelblue"
        label = f"Layer {layer}" + (" (best)" if layer == best_layer else "")
        ax.scatter(x, y, alpha=0.4, s=8, color=color, rasterized=True)
        lo, hi = _ax_limits(x, y)
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
    plt.savefig(os.path.join(output_dir, f"{file_tag}_vs_esm_scatter_{path_name}{title_suffix}.png"),
                dpi=80, bbox_inches="tight")
    plt.close()

    # ── Top / bottom-N highlight plots (best layer only) ─────────────────────
    x_best, y_best = _align_pair(baseline_fits, esm_fits_by_layer[best_layer])
    if z_norm_fits:
        x_best = (x_best - np.mean(x_best)) / (np.std(x_best) if np.std(x_best) > 0 else 1.0)
        y_best = (y_best - np.mean(y_best)) / (np.std(y_best) if np.std(y_best) > 0 else 1.0)
    eff_n = min(top_n, len(x_best) // 4)  # guard against tiny datasets

    if eff_n >= 1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        _plot_top_bottom_panel(axes[0], x_best, y_best, eff_n,
                               x_label=baseline_label, y_label="ESM-DMS Fitness",
                               title=f"Highlighted by {baseline_label} rank",
                               highlight_axis="x")
        _plot_top_bottom_panel(axes[1], x_best, y_best, eff_n,
                               x_label=baseline_label, y_label="ESM-DMS Fitness",
                               title="Highlighted by ESM-DMS rank",
                               highlight_axis="y")
        fig.suptitle(
            f"Top/bottom {eff_n} — {baseline_label} vs ESM-DMS (layer {best_layer}) — {path_name}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                 f"{file_tag}_vs_esm_top_bottom_{path_name}{title_suffix}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close()

    # ── Rank scatter (best layer) ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    _plot_rank_panel(ax, x_best, y_best,
                     x_label=baseline_label, y_label="ESM-DMS Fitness",
                     title=f"Rank comparison — layer {best_layer} — {path_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             f"{file_tag}_vs_esm_ranks_{path_name}{title_suffix}.png"),
                dpi=100, bbox_inches="tight")
    plt.close()


def get_individual_fitness_values(selection_file, haplotype_counts_file, reference_sequence,
                                  rep='joint', comment_char=None, fitness_fn='plus1'):
    '''
    Compute fitness for every individual haplotype in the population.

    Fitness is defined as:
        plus1: w = 1 + sum_i(x_i * s_i)
        exp:   w = exp(sum_i(x_i * s_i))
    where x_i = 1 if the individual carries mutation i, 0 otherwise.

    Arguments:
        - selection_file:        Path to selection coefficients .csv.gz from infer_correlated
        - haplotype_counts_file: Path to the MaveDB format haplotype counts file
        - reference_sequence:    Reference nucleotide sequence string
        - rep:                   Which replicate to use for selection coefficients (default: 'joint')
        - comment_char:          Comment character in haplotype counts file (default: None)
        - fitness_fn:            'plus1' (default) or 'exp' — must match ESM-DMS fitness_fn

    Returns:
        - fitness: pd.Series indexed by protein sequence with one fitness value per haplotype
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

        # WT individual carries no mutations: s_sum = 0, so w = 1 for both models
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

        if fitness_fn == "exp":
            fitness_values.append(np.exp(s_sum))
        else:
            fitness_values.append(1.0 + s_sum)
        protein_seqs.append(''.join(variant_aas))

    return pd.Series(fitness_values, index=protein_seqs)
