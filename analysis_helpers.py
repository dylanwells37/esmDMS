import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from esmdmsfunctions import (
    load_final_df,
    calculate_fitness_exp,
    calculate_fitness_plus1,
    z_normalize
)


def plot_true_vs_inferred_s(all_results, layers, max_cols=3):
    """True vs inferred selection coefficients for each replicate."""
    for path_name, results in all_results.items():
        all_layer_fits, all_sel_coeffs, detailed_results, gamma_results, all_gen_counts, eig_info = results

        for layer in layers:
            if layer not in detailed_results:
                print(f"[{path_name}]  Layer {layer}: no inference results (run with inference=True)")
                continue

            true_s       = all_sel_coeffs[layer]
            s_reps       = detailed_results[layer][0]
            n_reps       = len(s_reps)
            n_components = eig_info[layer]['n_components']

            n_cols = min(n_reps, max_cols)
            n_rows = int(np.ceil(n_reps / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

            for rep in range(n_reps):
                row, col = divmod(rep, n_cols)
                ax = axes[row][col]

                true_norm     = z_normalize(true_s)
                inferred_norm = z_normalize(s_reps[rep])

                ax.scatter(true_norm, inferred_norm, alpha=0.7, edgecolors='k', linewidths=0.3)
                lim = max(np.abs(true_norm).max(), np.abs(inferred_norm).max()) + 0.5
                ax.plot([-lim, lim], [-lim, lim], 'r--')
                ax.set_xlabel('True s (normalized)')
                ax.set_ylabel('Inferred s (normalized)')
                ax.set_title(f'Rep {rep + 1}')
                ax.axis('equal')

                corr, pval = pearsonr(true_norm, inferred_norm)
                ax.annotate(f'r = {corr:.3f}\np = {pval:.2e}',
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            ha='left', va='top', fontsize=10,
                            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

            for idx in range(n_reps, n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row][col].set_visible(False)

            fig.suptitle(f'[{path_name}]  Layer {layer} — True vs Inferred s  ({n_components} eigenvectors)', fontsize=13)
            plt.tight_layout()
            plt.show()


def plot_cross_replicate_consistency(all_results, layers, max_cols=3):
    """Cross-replicate consistency of inferred selection coefficients."""
    for path_name, results in all_results.items():
        all_layer_fits, all_sel_coeffs, detailed_results, gamma_results, all_gen_counts, eig_info = results

        for layer in layers:
            if layer not in detailed_results:
                print(f"[{path_name}]  Layer {layer}: no inference results")
                continue

            s_reps     = detailed_results[layer][0]
            n_reps     = len(s_reps)
            rep_pairs  = [(i, j) for i in range(n_reps) for j in range(i + 1, n_reps)]
            n_plots    = len(rep_pairs)

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
                corr, pval = pearsonr(si, sj)
                ax.set_title(f'r = {corr:.3f}  (p = {pval:.2e})')
                ax.axis('equal')

            for idx in range(n_plots, n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row][col].set_visible(False)

            fig.suptitle(f'[{path_name}]  Layer {layer} — Cross-replicate consistency  ({eig_info[layer]["n_components"]} eigenvectors)', fontsize=13)
            plt.tight_layout()
            plt.show()


def compute_individual_fitness(all_results, paths, layers, fitness_fn='exp'):
    """
    For each (dataset, layer, replicate) compute per-individual true and inferred fitness.

    Returns
    -------
    fitness_data : dict
        {path_name: {layer: {'true': array(N,),
                              'inferred': array(N, n_reps),
                              'n_reps': int}}}
    """
    calc_fit = calculate_fitness_exp if fitness_fn == 'exp' else calculate_fitness_plus1

    fitness_data = {}
    for path_name, results in all_results.items():
        all_layer_fits, _, detailed_results, _, _, eig_info = results
        fitness_data[path_name] = {}

        for layer in layers:
            if layer not in detailed_results:
                print(f"[{path_name}] layer {layer}: no inference results, skipping.")
                continue

            raw_df = load_final_df(layer, paths[path_name])
            embeddings = np.vstack(raw_df['Embedding'].tolist())          # (N, D)
            proj_mat   = eig_info[layer]['projection_matrix']             # (D, k)
            proj_emb   = embeddings @ proj_mat                            # (N, k)

            true_fits = np.array(all_layer_fits[layer], dtype=float)     # (N,)

            s_reps     = detailed_results[layer][0]                      # list of n_reps arrays (k,)
            n_reps     = len(s_reps)
            inferred   = np.zeros((len(proj_emb), n_reps), dtype=float)
            for rep, s in enumerate(s_reps):
                for i, emb in enumerate(proj_emb):
                    inferred[i, rep] = calc_fit(emb, s)

            fitness_data[path_name][layer] = {
                'true':     true_fits,
                'inferred': inferred,
                'n_reps':   n_reps,
            }
            print(f"[{path_name}] layer {layer}: {len(true_fits)} variants, {n_reps} replicates")

    return fitness_data


def plot_true_vs_inferred_fitness(fitness_data, max_cols=3):
    """True vs inferred individual fitness for each replicate."""
    plt.style.use('seaborn-v0_8-darkgrid')

    for path_name, layer_dict in fitness_data.items():
        for layer, fd in layer_dict.items():
            true_norm = z_normalize(fd['true'])
            n_reps    = fd['n_reps']

            n_cols = min(n_reps, max_cols)
            n_rows = int(np.ceil(n_reps / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

            for rep in range(n_reps):
                row, col = divmod(rep, n_cols)
                ax = axes[row][col]

                inf_norm = z_normalize(fd['inferred'][:, rep])

                ax.scatter(true_norm, inf_norm, alpha=0.5, s=12,
                           edgecolors='k', linewidths=0.2)

                lim = max(np.abs(true_norm).max(), np.abs(inf_norm).max()) + 0.3
                ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1.2, label='y = x')
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_xlabel('True fitness (z-norm)')
                ax.set_ylabel('Inferred fitness (z-norm)')
                ax.set_title(f'Rep {rep + 1}')

                r, pval   = pearsonr(true_norm, inf_norm)
                rho, _    = spearmanr(true_norm, inf_norm)
                rmse      = np.sqrt(np.mean((true_norm - inf_norm) ** 2))
                ax.annotate(
                    f'Pearson r = {r:.3f}  (p={pval:.1e})\nSpearman ρ = {rho:.3f}\nRMSE = {rmse:.3f}',
                    xy=(0.04, 0.96), xycoords='axes fraction',
                    ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.85))

            for idx in range(n_reps, n_rows * n_cols):
                axes[divmod(idx, n_cols)[0]][divmod(idx, n_cols)[1]].set_visible(False)

            fig.suptitle(
                f'[{path_name}]  Layer {layer} — True vs Inferred Individual Fitness',
                fontsize=13)
            plt.tight_layout()
            plt.show()


def plot_fitness_trajectories(all_results, n_cols=6):
    """Mean fitness over time for each layer and replicate."""
    for path_name, res in all_results.items():
        layers = sorted(res[4].keys())
        n_layers = len(layers)
        n_rows = int(np.ceil(n_layers / n_cols))

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
        plt.show()
