"""
# Simulation-based analyses (fitness, sel_coeffs)
python mega_analysis.py data/embeddings plots/ --sim_config configs/simulation_config.json --fitness
python mega_analysis.py data/embeddings plots/ --sim_config configs/simulation_config.json --sel_coeffs
python mega_analysis.py data/embeddings plots/ --sim_config configs/simulation_config.json --fitness --sel_coeffs

# Embedding-only analyses (cross_replicate_consistency, shuffled_frequencies)
python mega_analysis.py data/embeddings plots/ --embedding_config configs/inference_config.json --cross_replicate_consistency

# Run all analyses (both configs required)
python mega_analysis.py data/embeddings plots/ --sim_config configs/simulation_config.json --embedding_config configs/inference_config.json --all
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

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)
    

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
    # This function is similar to emb_df_to_sim_dfs but will save the dataframes in a format suitable for inference
    # The main difference is that we won't have the "Rep" columns, and instead we'll just save the embeddings for each layer
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    num_layers = len(emb_df["Embedding"].iloc[0])
    for layer in range(num_layers):
        layer_df = emb_df.copy()
        layer_df["Embedding"] = layer_df["Embedding"].apply(lambda x: x[layer])
        layer_df.drop(columns=["ProteinSequence"], inplace=True, errors="ignore")
        out_path_layer = os.path.join(out_path, f"layer{layer}_inference_df.pkl")
        layer_df.to_pickle(out_path_layer)
        print(f"Saved layer {layer} inference dataframe to {out_path_layer}")
    print("All layers saved successfully.")


def run_inference(embedding_path, inference_cfg, save_results=True, force_recompute=False):
    cache_path = os.path.join(embedding_path, "inference_results.pkl")
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Inference results already exist at {cache_path}. Loading existing results.")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    for layer in inference_cfg.get("layers", list(range(31))):
        df_path = os.path.join(embedding_path, f"layer{layer}_inference_df.pkl")
        if not os.path.exists(df_path):
            print(f"ERROR: Missing inference_df.pkl for layer {layer} at {df_path}", file=sys.stderr)
            # name  = os.path.basename(args.embedding_path.rstrip("/"))
            directory_name = os.path.basename(embedding_path.rstrip("/"))
            emb_path = os.path.join(embedding_path, f"{directory_name}_embeddings.pkl")
            if os.path.exists(emb_path):
                embedding_df = pd.read_pickle(emb_path)
                emb_df_to_inference_dfs(embedding_df, embedding_path)
                break
            else:
                print(f"ERROR: Missing {directory_name}_embeddings.pkl at {emb_path}. Cannot run inference.", file=sys.stderr)
                sys.exit(1)

    raw = {}
    for layer in inference_cfg.get("layers", list(range(31))):
        layer_df = load_inference_df(layer, embedding_path)
        layer_inference_data = mini_infer_independent_esm(layer_df, n_replicates=3, gamma=None, corr_cutoff_pct=0.5,
                                                   max_reads=1e3, output_dir=None, name='esm_inference', plot_gamma=True,
                                                   verbose=True, calc_error_bars=False,
                                                   variance_cutoff=0.0, infer_ignored_dims=True)
        raw[layer] = layer_inference_data

    processed = {}
    for layer, data in raw.items():
        processed[layer] = [data[2], data[3], data[7], data[8], data[1], data[5]]

    result_tuple = (None, None, processed, None, None)

    if save_results:
        with open(cache_path, "wb") as f:
            pickle.dump(result_tuple, f, protocol=4)
        print(f"Inference results saved to {cache_path}")

    return result_tuple


def run_simulation(embedding_path, simulation_cfg):
    # Check if the dfs for all layers exist before starting the simulation
    for layer in simulation_cfg.get("layers", list(range(31))):
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
        layers=simulation_cfg.get("layers", list(range(31))),
        calc_error_bars=simulation_cfg.get("calc_error_bars", False),
        infer_ignored_dims=simulation_cfg.get("infer_ignored", True),
        method="fullcov",
    )


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def plot_cross_replicate_consistency_analysis(all_results, paths, cfg, output_dir):
    layers = cfg.get("layers", list(range(31)))
    plot_cross_replicate_consistency(all_results, layers, output_dir=output_dir)



def plot_shuffled_frequencies_analysis(all_results, paths, cfg, output_dir):
    """Cross-replicate consistency after shuffling frequencies within each replicate.

    For each layer, independently shuffles the pre- and post-selection frequency
    values within each replicate (breaking the embedding→count association), then
    re-runs inference and plots cross-replicate consistency. A high r on real data
    but low r here confirms the signal is not an artefact of count structure.
    """
    layers = cfg.get("layers", list(range(31)))
    rng = np.random.default_rng()
    shuffled_all_results = {}

    for name, embedding_path in paths.items():
        shuffled_processed = {}

        for layer in layers:
            layer_df = load_inference_df(layer, embedding_path).copy()

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

        shuffled_all_results[name] = (None, None, shuffled_processed, None, None)

    shuffled_output_dir = os.path.join(output_dir, "shuffled")
    plot_cross_replicate_consistency(shuffled_all_results, layers, output_dir=shuffled_output_dir)


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
    parser.add_argument("embedding_path",
                        help="Path to embedding data directory (containing layer0/, layer1/, … subdirs)")
    parser.add_argument("--sim_config", help="Path to JSON config file (simulation parameters, optional)")
    parser.add_argument("--embedding_config", help="Path to JSON config file (embedding parameters, optional)")
    parser.add_argument("output_dir", help="Directory to write output plots")

    analysis_group = parser.add_argument_group("analyses (at least one required)")
    analysis_group.add_argument("--all", dest="run_all", action="store_true",
                                help="Run all analyses")
    for flag, (_, description) in ANALYSES.items():
        analysis_group.add_argument(f"--{flag}", action="store_true",
                                    help=f"Plot {description}")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    selected = {flag for flag in ANALYSES if args.run_all or getattr(args, flag, False)}
    if not selected:
        parser.error("No analyses selected. Pass --all or one or more analysis flags.")

    name  = os.path.basename(args.embedding_path.rstrip("/"))
    paths = {name: args.embedding_path}

    needs_sim = selected & set(ANALYSES_REQUIRING_SIM)
    needs_emb = selected & set(ANALYSES_REQUIRING_ONLY_EMB)

    if needs_sim and not args.sim_config:
        parser.error(f"--sim_config is required for: {', '.join(needs_sim)}")
    if needs_emb and not args.embedding_config:
        parser.error(f"--embedding_config is required for: {', '.join(needs_emb)}")

    sim_cfg = load_config(args.sim_config) if args.sim_config else {}
    emb_cfg = load_config(args.embedding_config) if args.embedding_config else {}

    all_results_sim = None
    all_results_emb = None

    if needs_sim:
        print(f"Running simulation on: {args.embedding_path}")
        all_results_sim = {name: run_simulation(args.embedding_path, sim_cfg)}

    if needs_emb:
        print(f"Running embedding inference on: {args.embedding_path}")
        all_results_emb = {name: run_inference(args.embedding_path, emb_cfg)}

    ran_any = False
    for flag, (fn, description) in ANALYSES.items():
        if flag not in selected:
            continue
        if fn is None:
            print(f"Skipping '{flag}': not yet implemented.")
            continue
        print(f"Running: {description}")
        all_results = all_results_sim if flag in ANALYSES_REQUIRING_SIM else all_results_emb
        fn(all_results, paths, sim_cfg if flag in ANALYSES_REQUIRING_SIM else emb_cfg, output_dir=args.output_dir)
        ran_any = True

    if not ran_any:
        print("Warning: no analyses ran (all selected analyses may be unimplemented).")


if __name__ == "__main__":
    main()
