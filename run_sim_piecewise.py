import os
import sys
import json
import shutil
import pickle
import argparse
import tracemalloc
tracemalloc.start()

# Project directory
PROJECT_DIR = "/net/dali/home/barton/dhw28/popDMS/esmDMS"
sys.path.insert(0, PROJECT_DIR)

from esmdmsfunctions import (
    get_simulation_results_piecewise,
    gaussian_selection,
    generate_selection,
    zero_selection,
    # add any other selection functions here
)
import esmdmsfunctions

# --- Map string names to actual functions ---
SELECTION_FUNCTIONS = {
    "gaussian": gaussian_selection,
    "default": generate_selection,
    "zero": zero_selection,
    # add more as needed
}


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Convert sel_func string to actual function
    func_name = config.get("sel_func", "default")
    if func_name not in SELECTION_FUNCTIONS:
        raise ValueError(f"Unknown sel_func '{func_name}'. Options: {list(SELECTION_FUNCTIONS.keys())}")
    config["sel_func"] = SELECTION_FUNCTIONS[func_name]
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("output_file", default=None, help="output file name")
    parser.add_argument("SCRATCH_DIR", help="Path to scratch directory for temporary storage")
    parser.add_argument("--inference", type=int, default=None, help="Override inference parameter")
    parser.add_argument("--layer", type=int, default=None, help="Override layer parameter")
    parser.add_argument("--sel_func", default=None, help="Override selection function (gaussian, default, zero)")
    args = parser.parse_args()
    
    SCRATCH_DIR = args.SCRATCH_DIR
    output_file = args.output_file
    config = load_config(args.config)
    
    # override with command line args if provided
    if args.inference is not None:
        config["inference"] = args.inference
    if args.layer is not None:
        config["layer"] = args.layer
    if args.sel_func is not None:
        if args.sel_func not in SELECTION_FUNCTIONS:
            raise ValueError(f"Unknown sel_func '{args.sel_func}'. Options: {list(SELECTION_FUNCTIONS.keys())}")
        config["sel_func"] = SELECTION_FUNCTIONS[args.sel_func]

    # --- Set up scratch directory ---
    node = os.environ.get("SLURMD_NODENAME", "unknown")
    SCRATCH_SIM_FOLDER = f"{SCRATCH_DIR}/esm_sim_saves/"
    HOME_SIM_FOLDER = f"{PROJECT_DIR}/esm_sim_saves/"

    os.makedirs(SCRATCH_SIM_FOLDER, exist_ok=True)

    # Override paths in utils so any saves go to scratch
    esmdmsfunctions.sim_folder = SCRATCH_SIM_FOLDER
    esmdmsfunctions.pwd = PROJECT_DIR

    # --- Run the simulation ---
    print(f"Running on node: {node}")
    print(f"Writing to scratch: {SCRATCH_SIM_FOLDER}")
    print(f"Config: {json.dumps({k: str(v) for k, v in config.items()}, indent=2)}")
    print("Starting simulation...")
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("\n--- Top 10 memory allocations ---")
    for stat in top_stats[:10]:
        print(stat)
    results = get_simulation_results_piecewise(**config)

    # --- Save results to scratch ---
    scratch_path = os.path.join(SCRATCH_SIM_FOLDER, output_file)
    with open(scratch_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {scratch_path}")

    # --- Copy results back to home directory ---
    os.makedirs(HOME_SIM_FOLDER, exist_ok=True)
    home_path = os.path.join(HOME_SIM_FOLDER, output_file)
    shutil.copy2(scratch_path, home_path)
    print(f"Results copied to {home_path}")

    # --- Clean up scratch ---
    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    print("Scratch cleaned up. Done!")

if __name__ == "__main__":
    main()
