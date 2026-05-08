import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from esmdmsfunctions import (
    load_final_df,
    z_normalize
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


def plot_cross_replicate_consistency(all_results, layers, max_cols=3, output_dir=None, corr='pearson', normalize="none"):
    """Cross-replicate consistency of inferred selection coefficients.

    corr : 'pearson' or 'spearman'
    """
    if corr == 'spearman':
        corr_fn  = spearmanr
        corr_sym = 'ρ'
    else:
        corr_fn  = pearsonr
        corr_sym = 'r'

    for path_name, results in all_results.items():
        detailed_results = results[2]

        for layer in layers:
            if layer not in detailed_results:
                print(f"[{path_name}]  Layer {layer}: no inference results")
                continue

            s_reps    = detailed_results[layer][0]
            n_reps    = len(s_reps)
            print(f"n_reps for {path_name} layer {layer}: {n_reps}")
            rep_pairs = [(i, j) for i in range(n_reps) for j in range(i + 1, n_reps)]
            n_plots   = len(rep_pairs)

            n_cols = min(n_plots, max_cols)
            n_rows = int(np.ceil(n_plots / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

            for idx, (ri, rj) in enumerate(rep_pairs):
                row, col = divmod(idx, n_cols)
                ax = axes[row][col]

                si = z_normalize(s_reps[ri])
                sj = z_normalize(s_reps[rj])
                ax.scatter(si, sj, alpha=0.6, edgecolors='k', linewidths=0.3)
                lim = max(np.abs(si).max(), np.abs(sj).max()) + 0.5
                ax.plot([-lim, lim], [-lim, lim], 'r--')
                ax.set_xlabel(f'Rep {ri + 1} s (normalized)')
                ax.set_ylabel(f'Rep {rj + 1} s (normalized)')
                val, pval = corr_fn(si, sj)
                ax.set_title(f'{corr_sym} = {val:.3f}  (p = {pval:.2e})')
                ax.axis('equal')

            for idx in range(n_plots, n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row][col].set_visible(False)

            fig.suptitle(f'[{path_name}]  Layer {layer} — Cross-replicate consistency', fontsize=13)
            plt.tight_layout()
            if normalize != "none":
                fig.text(0.5, -0.01, f"Normalization: {normalize}", ha='center',
                         fontsize=9, style='italic', color='gray')
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"{path_name}_layer{layer}_cross_replicate.png"),
                            bbox_inches="tight")
            plt.show()


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
