"""
# Simulation-based analyses (fitness, sel_coeffs)
python mega_analysis.py Ube4b plots/ --sim_config configs/simulation_config.json --fitness
python mega_analysis.py Ube4b plots/ --sim_config configs/simulation_config.json --sel_coeffs
python mega_analysis.py Ube4b plots/ --sim_config configs/simulation_config.json --fitness --sel_coeffs

# Embedding-only analyses (cross_replicate_consistency, shuffled_frequencies)
python mega_analysis.py Ube4b plots/ --embedding_config configs/inference_config.json --cross_replicate_consistency

# Run all analyses (both configs required)
python mega_analysis.py Ube4b plots/ --sim_config configs/simulation_config.json --embedding_config configs/inference_config.json --all
"""

import os
import sys
import json
import pickle
import argparse
import pandas as pd

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pwd = "/net/dali/home/barton/dhw28/popDMS/esmDMS"
if not os.path.exists(pwd):
    pwd = "/Users/dylanwells/popDMS/esmDMS"
if pwd not in sys.path:
    sys.path.append(pwd)

from esmdmsfunctions import (
    get_simulation_results,
    generate_selection, gaussian_selection, zero_selection,
    load_final_df, load_inference_df,
)

from analysis_helpers import (
    plot_true_vs_inferred_fitness,
    plot_true_vs_inferred_sel_coeffs,
    plot_cross_replicate_consistency,
    plot_fitness_trajectories,
    get_individual_fitness_values,
    get_esm_individual_fitness_values,
    get_enrichment_ratios,
    plot_baseline_vs_esm_comparison,
    plot_scatter_comparison,
)

from popDMS import mini_infer_independent_esm

SEL_FUNC_MAP = {
    "gaussian": gaussian_selection,
    "generate": generate_selection,
    "zero":     zero_selection,
}

NCOLS = 6


# ---------------------------------------------------------------------------
# Config + simulation
# ---------------------------------------------------------------------------

def load_config(config_path, dataset=None):
    with open(config_path) as f:
        cfg = json.load(f)
    if dataset is not None:
        cfg = {k: os.path.expanduser(v.replace("{dataset}", dataset)) if isinstance(v, str) else v
               for k, v in cfg.items()}
    return cfg


def _detect_layers(embedding_path, suffix=None):
    """Return sorted layer indices by scanning for layer{i}_{suffix}.pkl files.

    When suffix is None (inference case), tries 'seq_to_emb' first (new compact
    format) then 'inference_df' (legacy format) so both layouts are handled.
    """
    import re
    try:
        files = os.listdir(embedding_path)
    except OSError:
        return list(range(31))
    for s in ([suffix] if suffix else ["seq_to_emb", "inference_df"]):
        layers = sorted(
            int(m.group(1))
            for f in files
            if (m := re.match(rf"layer(\d+)_{s}\.pkl", f))
        )
        if layers:
            return layers
    return list(range(31))


def emb_df_to_sim_dfs(emb_df, out_path):
    # Format is Embeddings , PreNum_1, PreNum_2, ...
    # Out format will save a dataframe for each layer
    reps = sorted(emb_df["Replicate"].unique())
    pre_df = emb_df[emb_df["Generation"] == 0]

    # One row per unique protein sequence (1-to-1 with embedding)
    result = (
        pre_df.drop_duplicates(subset="ProteinSequence")
              .set_index("ProteinSequence")[["Embedding"]]
    )

    for rep in reps:
        rep_freqs = pre_df[pre_df["Replicate"] == rep].set_index("ProteinSequence")["Frequency"]
        result[f"Rep{rep}_PreNums"] = rep_freqs

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    num_layers = len(result["Embedding"].iloc[0])
    for layer in range(num_layers):
        layer_df = result[["Embedding", *[col for col in result.columns if col.startswith("Rep")]]].copy()
        layer_df["Embedding"] = layer_df["Embedding"].apply(lambda x: x[layer])
        # save the pickle file
        out_path_layer = os.path.join(out_path, f"layer{layer}_sim_df.pkl")
        layer_df.drop(columns=["ProteinSequence"], inplace=True, errors="ignore")
        layer_df.to_pickle(out_path_layer)
        print(f"Saved layer {layer} dataframe to {out_path_layer}")
    print("All layers saved successfully.")


def emb_df_to_inference_dfs(emb_df, out_path):
    """Save compact per-layer inference files.

    Instead of duplicating each sequence's embedding across every (replicate,
    generation) row, we store:
      - layer{i}_seq_to_emb.pkl : (n_unique_seqs, emb_dim) numpy array indexed by seq_id
      - layer{i}_inference_df.pkl: metadata df with (seq_id, Replicate, Generation, Frequency)
      - seq_id_map.pkl           : list of ProteinSequence strings in seq_id order

    load_inference_df() reconstructs the full df with an Embedding column on load.
    This cuts per-layer file size from O(n_rows x emb_dim) to O(n_unique x emb_dim).
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    num_layers = len(emb_df["Embedding"].iloc[0])

    # One row per unique sequence for building compact per-layer arrays
    unique_embs = emb_df.drop_duplicates("ProteinSequence").reset_index(drop=True)
    unique_seqs = unique_embs["ProteinSequence"].tolist()
    seq_to_id = {seq: i for i, seq in enumerate(unique_seqs)}

    # Compact metadata df: integer seq_id replaces both ProteinSequence and Embedding
    meta_df = emb_df[["ProteinSequence", "Replicate", "Generation", "Frequency"]].copy()
    meta_df["seq_id"] = meta_df["ProteinSequence"].map(seq_to_id)
    meta_df = meta_df.drop(columns=["ProteinSequence"]).reset_index(drop=True)

    with open(os.path.join(out_path, "seq_id_map.pkl"), "wb") as f:
        pickle.dump(unique_seqs, f)
    print(f"Saved seq_id_map ({len(unique_seqs):,} unique sequences)")

    # Save metadata once — it is identical across all layers (seq_id, Replicate, Generation, Frequency).
    # load_inference_df checks for this file first; per-layer copies are kept for backward compatibility
    # with any existing data but are no longer written for new saves.
    meta_df.to_pickle(os.path.join(out_path, "inference_metadata.pkl"))
    print(f"Saved inference_metadata ({len(meta_df):,} rows, shared across all layers)")

    for layer in range(num_layers):
        layer_embs = np.array([emb[layer] for emb in unique_embs["Embedding"]])  # (n_unique, emb_dim)

        seq_to_emb_path = os.path.join(out_path, f"layer{layer}_seq_to_emb.pkl")
        with open(seq_to_emb_path, "wb") as f:
            pickle.dump(layer_embs, f)

        print(f"Saved layer {layer}: {layer_embs.nbytes / 1e6:.0f} MB (seq_to_emb)")
    print("All layers saved successfully.")


def run_inference(embedding_path, inference_cfg, save_results=True, force_recompute=False):
    normalize   = inference_cfg.get("normalize", "none")
    replicates  = inference_cfg.get("replicates")  # None → use all
    norm_suffix = f"_{normalize}" if normalize != "none" else ""
    rep_suffix  = ("_reps" + "_".join(map(str, sorted(replicates)))) if replicates else ""
    cache_path  = os.path.join(embedding_path, f"inference_results{norm_suffix}{rep_suffix}.pkl")

    if os.path.exists(cache_path) and not force_recompute:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        # Validate cached n_reps against actual data so stale caches are caught.
        # Load only the compact metadata pkl (no embeddings) to avoid a 3+ GB allocation.
        cached_processed = cached[2]
        if cached_processed:
            probe_layer = next(iter(cached_processed))
            shared = os.path.join(embedding_path, "inference_metadata.pkl")
            per_layer = os.path.join(embedding_path, f"layer{probe_layer}_inference_df.pkl")
            probe_meta = pd.read_pickle(shared if os.path.exists(shared) else per_layer)
            if replicates is not None:
                probe_meta = probe_meta[probe_meta["Replicate"].isin(replicates)]
            expected_n_reps = probe_meta["Replicate"].nunique()
            cached_n_reps = cached_processed[probe_layer][0].shape[0]  # s.shape[0]
            if cached_n_reps != expected_n_reps:
                print(f"Cache has n_reps={cached_n_reps} but data has {expected_n_reps} — recomputing.")
            else:
                print(f"Inference results already exist at {cache_path}. Loading existing results.")
                return cached
        else:
            print(f"Inference results already exist at {cache_path}. Loading existing results.")
            return cached

    layers = inference_cfg.get("layers") or _detect_layers(embedding_path)

    # Compact format requires layer{i}_seq_to_emb.pkl; legacy format uses layer{i}_inference_df.pkl.
    # Check whichever is present; if neither exists, attempt to regenerate from the raw embeddings pkl.
    for layer in layers:
        has_compact = os.path.exists(os.path.join(embedding_path, f"layer{layer}_seq_to_emb.pkl"))
        has_legacy  = os.path.exists(os.path.join(embedding_path, f"layer{layer}_inference_df.pkl"))
        if not has_compact and not has_legacy:
            directory_name = os.path.basename(embedding_path.rstrip("/"))
            emb_path = os.path.join(embedding_path, f"{directory_name}_embeddings.pkl")
            if os.path.exists(emb_path):
                print(f"Layer {layer} data missing — regenerating from {emb_path}", file=sys.stderr)
                embedding_df = pd.read_pickle(emb_path)
                emb_df_to_inference_dfs(embedding_df, embedding_path)
                break
            else:
                print(f"ERROR: No layer data for layer {layer} in {embedding_path} and "
                      f"{directory_name}_embeddings.pkl not found. Cannot run inference.", file=sys.stderr)
                sys.exit(1)

    # Convert to processed in the same loop so full mini_infer return values are not
    # accumulated in memory across all layers simultaneously.
    processed = {}
    for layer in layers:
        layer_df = load_inference_df(layer, embedding_path, normalize=normalize, replicates=replicates)
        n_replicates = layer_df["Replicate"].nunique()
        data = mini_infer_independent_esm(layer_df, n_replicates=n_replicates, gamma=None, corr_cutoff_pct=0.5,
                                          max_reads=1e3, output_dir=None, name='esm_inference', plot_gamma=True,
                                          verbose=True, calc_error_bars=False,
                                          variance_cutoff=0.0, infer_ignored_dims=True)
        processed[layer] = [data[2], data[3], data[7], data[8], data[1], data[5]]
        del layer_df, data

    result_tuple = (None, None, processed, None, None)

    if save_results:
        with open(cache_path, "wb") as f:
            pickle.dump(result_tuple, f, protocol=4)
        print(f"Inference results saved to {cache_path}")
    return result_tuple


def run_simulation(embedding_path, simulation_cfg):
    layers = simulation_cfg.get("layers") or _detect_layers(embedding_path, "sim_df")
    # Check if the dfs for all layers exist before starting the simulation
    for layer in layers:
        df_path = os.path.join(embedding_path, f"layer{layer}_sim_df.pkl")
        if not os.path.exists(df_path):
            print(f"ERROR: Missing final_df.pkl for layer {layer} at {df_path}", file=sys.stderr)
            # Check if the embedding_df exists, if so, make the simulation layer dfs
            directory_name = os.path.basename(embedding_path.rstrip("/"))
            emb_path = os.path.join(embedding_path, f"{directory_name}_embeddings.pkl")
            if os.path.exists(emb_path):
                embedding_df = pd.read_pickle(emb_path)
                emb_df_to_sim_dfs(embedding_df, embedding_path)
                break
            else:
                print(f"ERROR: Missing {directory_name}_embeddings.pkl at {emb_path}. Cannot run simulation.", file=sys.stderr)
                sys.exit(1)

    print(f"EMBEDDING PATH  = {embedding_path}")
    return get_simulation_results(
        n_gens=simulation_cfg.get("n_gens", 30),
        embedding_df_path=embedding_path,
        sel_func=SEL_FUNC_MAP[simulation_cfg.get("sel_func", "gaussian")],
        inference=True,
        gamma_analysis=simulation_cfg.get("run_gamma", False),
        fitness=simulation_cfg.get("fitness_fn", "exp"),
        save_every=simulation_cfg.get("save_every", 1),
        layers=layers,
        calc_error_bars=simulation_cfg.get("calc_error_bars", False),
        infer_ignored_dims=simulation_cfg.get("infer_ignored", True),
        method="fullcov",
    )


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def plot_cross_replicate_consistency_analysis(all_results, paths, cfg, output_dir):
    embedding_path = next(iter(paths.values()))
    layers = cfg.get("layers") or _detect_layers(embedding_path)
    plot_cross_replicate_consistency(all_results, layers, output_dir=output_dir,
                                     normalize=cfg.get("normalize", "none"))



def plot_shuffled_frequencies_analysis(all_results, paths, cfg, output_dir):
    """Cross-replicate consistency after shuffling frequencies within each replicate.

    For each layer, independently shuffles the pre- and post-selection frequency
    values within each replicate (breaking the embedding→count association), then
    re-runs inference and plots cross-replicate consistency. A high r on real data
    but low r here confirms the signal is not an artefact of count structure.
    """
    normalize  = cfg.get("normalize", "none")
    replicates = cfg.get("replicates")
    rng = np.random.default_rng()
    shuffled_all_results = {}

    for name, embedding_path in paths.items():
        layers = cfg.get("layers") or _detect_layers(embedding_path)
        shuffled_processed = {}

        for layer in layers:
            layer_df = load_inference_df(layer, embedding_path, normalize=normalize, replicates=replicates)

            # Shuffle frequencies independently within each (Replicate, Generation) group
            for _, group_idx in layer_df.groupby(["Replicate", "Generation"]).groups.items():
                freqs = layer_df.loc[group_idx, "Frequency"].values.copy()
                rng.shuffle(freqs)
                layer_df.loc[group_idx, "Frequency"] = freqs

            n_reps = layer_df["Replicate"].nunique()
            data = mini_infer_independent_esm(
                layer_df, n_replicates=n_reps, gamma=None, corr_cutoff_pct=0.5,
                max_reads=1e3, output_dir=None, name="esm_inference_shuffled",
                plot_gamma=False, verbose=False, calc_error_bars=False,
                variance_cutoff=0.0, infer_ignored_dims=True,
            )
            shuffled_processed[layer] = [data[2], data[3], data[7], data[8], data[1], data[5]]
            del layer_df, data

        shuffled_all_results[name] = (None, None, shuffled_processed, None, None)

    shuffled_output_dir = os.path.join(output_dir, "shuffled")
    plot_cross_replicate_consistency(shuffled_all_results, layers, output_dir=shuffled_output_dir,
                                     normalize=normalize)


def popDMS_esmDMS_comparison_analysis(all_results, paths, cfg, output_dir):
    """Compare the inferred fitness of every individual within
    the embedding-based inference to the fitness inferred by popDMS on the same data.
    """
    pop_inference_path    = cfg.get("pop_inference_path", None)
    reference_sequence_file = cfg.get("reference_sequence_file", None)
    haplotype_counts_file = cfg.get("haplotype_counts_file", None)
    fitness_fn            = cfg.get("fitness_fn", "plus1")
    comment_char          = cfg.get("comment_char", None)

    if not pop_inference_path or not os.path.exists(pop_inference_path):
        print(f"popDMS inference file not found at {pop_inference_path}. "
              "Run popDMS inference first (e.g. via the data_analysis notebook).")
        return

    with open(reference_sequence_file) as f:
        reference_sequence = f.read().strip()

    normalize = cfg.get("normalize", "none")

    # Part 1: load per-haplotype popDMS fitness values (same model as ESM-DMS)
    popdms_fits = get_individual_fitness_values(
        pop_inference_path, haplotype_counts_file, reference_sequence,
        fitness_fn=fitness_fn, comment_char=comment_char,
    )

    os.makedirs(output_dir, exist_ok=True)

    for path_name, results in all_results.items():
        detailed_results = results[2]
        embedding_path   = paths[path_name]
        layers           = sorted(detailed_results.keys())

        # Part 2: for each layer, compute ESM-DMS inferred fitness per sequence.
        # normalize must match what was used during inference so embeddings are
        # in the same space as s_joint.
        esm_fits_by_layer = {}
        for layer in layers:
            s_joint = detailed_results[layer][1]
            esm_fits_by_layer[layer] = get_esm_individual_fitness_values(
                embedding_path, layer, s_joint, fitness_fn=fitness_fn,
                normalize=normalize,
            )

        # Part 3: plot comparisons
        plot_baseline_vs_esm_comparison(
            popdms_fits, esm_fits_by_layer, layers, output_dir, path_name,
            baseline_label="popDMS Fitness",
        )
        enrichment_ratios = get_enrichment_ratios(embedding_path)
        plot_baseline_vs_esm_comparison(
            enrichment_ratios, esm_fits_by_layer, layers, output_dir, path_name,
            baseline_label="Enrichment Ratio",
        )


def popDMS_enrichment_comparison_analysis(all_results, paths, cfg, output_dir):
    """Scatter plot comparing popDMS inferred fitness directly to log enrichment ratio."""
    pop_inference_path    = cfg.get("pop_inference_path", None)
    reference_sequence_file = cfg.get("reference_sequence_file", None)
    haplotype_counts_file = cfg.get("haplotype_counts_file", None)
    fitness_fn            = cfg.get("fitness_fn", "plus1")
    comment_char          = cfg.get("comment_char", None)

    if not pop_inference_path or not os.path.exists(pop_inference_path):
        print(f"popDMS inference file not found at {pop_inference_path}. "
              "Run popDMS inference first (e.g. via the data_analysis notebook).")
        return

    with open(reference_sequence_file) as f:
        reference_sequence = f.read().strip()

    popdms_fits = get_individual_fitness_values(
        pop_inference_path, haplotype_counts_file, reference_sequence,
        fitness_fn=fitness_fn, comment_char=comment_char,
    )

    os.makedirs(output_dir, exist_ok=True)

    for path_name, _ in all_results.items():
        embedding_path = paths[path_name]
        enrichment_ratios = get_enrichment_ratios(embedding_path)
        plot_scatter_comparison(
            popdms_fits, enrichment_ratios, output_dir, path_name,
            x_label="popDMS Fitness", y_label="Enrichment Ratio",
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ANALYSES = {
    # Simulation-based analyses
    "fitness": (
        plot_true_vs_inferred_fitness,
        "true vs inferred fitness",
    ),
    "sel_coeffs": (
        plot_true_vs_inferred_sel_coeffs,
        "true vs inferred selection coefficients",
    ),
    "gamma": (
        None,  # placeholder for future gamma analysis function
        "gamma analysis (not implemented yet)",
    ),
    "sparsify": (
        None,  # placeholder for future sparsification analysis function
        "sparsification analysis (not implemented yet)",
    ),
    "eigenvalue_distribution": (
        None,  # placeholder for future eigenvalue distribution analysis function
        "eigenvalue distribution analysis (not implemented yet)",
    ),
    # Embedding-only analyses
    "cross_replicate_consistency": (
        plot_cross_replicate_consistency_analysis,
        "Plot the cross-replicate consistency for inferred selection coefficients across layers",
    ),
    "shuffled_frequencies": (
        plot_shuffled_frequencies_analysis,
        "Run cross-replicate consistency analysis on data with shuffled post-selection frequencies",
    ),
    "popDMS_comparison": (
        popDMS_esmDMS_comparison_analysis,
        "Compare per-individual ESM-DMS inferred fitness to popDMS fitness",
    ),
    "popDMS_enrichment_comparison": (
        popDMS_enrichment_comparison_analysis,
        "Scatter plot comparing popDMS inferred fitness directly to log enrichment ratio",
    ),
}

# Now categorize the analyses by which need simulation results vs just embeddings

ANALYSES_REQUIRING_SIM = ("fitness", "sel_coeffs")
ANALYSES_REQUIRING_ONLY_EMB = tuple(k for k in ANALYSES if k not in ANALYSES_REQUIRING_SIM)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run simulation and modular analyses on an embedding dataframe."
    )
    parser.add_argument("dataset",
                        help="Dataset name (e.g. Ube4b); substituted for {dataset} in all config paths")
    parser.add_argument("--sim_config", help="Path to JSON config file (simulation parameters, optional)")
    parser.add_argument("--embedding_config", help="Path to JSON config file (embedding parameters, optional)")
    parser.add_argument("output_dir", help="Directory to write output plots")

    analysis_group = parser.add_argument_group("analyses (at least one required)")
    analysis_group.add_argument("--all", dest="run_all", action="store_true",
                                help="Run all analyses")
    for flag, (_, description) in ANALYSES.items():
        analysis_group.add_argument(f"--{flag}", action="store_true",
                                    help=f"Plot {description}")

    parser.add_argument("--force_recompute", action="store_true",
                        help="Ignore cached inference_results.pkl and recompute from scratch")
    parser.add_argument("--normalize", choices=["none", "by_layer", "by_layer_dim"], default="none",
                        help="Normalize embeddings before inference: none (default), by_layer (global z-score per layer), by_layer_dim (per-dimension z-score per layer)")
    parser.add_argument("--replicates", nargs="+", type=int, default=None, metavar="REP",
                        help="Restrict inference to a subset of replicate IDs, e.g. --replicates 1 2 3 6 7 8")

    args = parser.parse_args()

    dataset = args.dataset

    selected = {flag for flag in ANALYSES if args.run_all or getattr(args, flag, False)}
    if not selected:
        parser.error("No analyses selected. Pass --all or one or more analysis flags.")

    needs_sim = selected & set(ANALYSES_REQUIRING_SIM)
    needs_emb = selected & set(ANALYSES_REQUIRING_ONLY_EMB)

    if needs_sim and not args.sim_config:
        parser.error(f"--sim_config is required for: {', '.join(needs_sim)}")
    if needs_emb and not args.embedding_config:
        parser.error(f"--embedding_config is required for: {', '.join(needs_emb)}")

    sim_cfg = load_config(args.sim_config, dataset) if args.sim_config else {}
    emb_cfg = load_config(args.embedding_config, dataset) if args.embedding_config else {}
    emb_cfg["normalize"]   = args.normalize
    emb_cfg["replicates"]  = args.replicates

    # embedding_path comes from whichever config is loaded; both carry {dataset} → resolved path
    embedding_path = emb_cfg.get("embedding_path") or sim_cfg.get("embedding_path")
    if not embedding_path:
        parser.error("embedding_path not found in any config file.")
    paths = {dataset: embedding_path}

    all_results_sim = None
    all_results_emb = None

    if needs_sim:
        print(f"Running simulation on: {embedding_path}")
        all_results_sim = {dataset: run_simulation(embedding_path, sim_cfg)}

    if needs_emb:
        print(f"Running embedding inference on: {embedding_path}")
        all_results_emb = {dataset: run_inference(embedding_path, emb_cfg,
                                                  force_recompute=args.force_recompute)}

    ran_any = False
    for flag, (fn, description) in ANALYSES.items():
        if flag not in selected:
            continue
        if fn is None:
            print(f"Skipping '{flag}': not yet implemented.")
            continue
        print(f"Running: {description}")
        is_sim = flag in ANALYSES_REQUIRING_SIM
        all_results = all_results_sim if is_sim else all_results_emb
        if is_sim:
            out_subdir = dataset
        else:
            rep_label  = ("reps" + "_".join(map(str, sorted(args.replicates)))) if args.replicates else None
            out_subdir = os.path.join(dataset, args.normalize, *([rep_label] if rep_label else []))
        fn(all_results, paths, sim_cfg if is_sim else emb_cfg,
           output_dir=os.path.join(args.output_dir, out_subdir))
        ran_any = True

    if not ran_any:
        print("Warning: no analyses ran (all selected analyses may be unimplemented).")


if __name__ == "__main__":
    main()
