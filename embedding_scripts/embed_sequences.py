import os
import sys
import ast
import json
import time
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# Project directory
PROJECT_DIR = "/net/dali/home/barton/dhw28/popDMS/esmDMS"
sys.path.insert(0, PROJECT_DIR)

from esmdmsfunctions import CODON2AA

DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw_data")
HOME_SEQ_FOLDER = os.path.join(PROJECT_DIR, "data", "sequence_data")


# ---------------------------------------------------------------------------
# Sequence reconstruction
# ---------------------------------------------------------------------------

def load_codoncounts(filepath):
    """Load a codoncounts CSV and return (codon_array, column_names, wildtypes)."""
    df = pd.read_csv(filepath)
    column_names = df.columns.tolist()[2:]
    wildtypes = df["wildtype"].tolist()
    df = df.drop(columns=["site", "wildtype"])
    return df.to_numpy(), column_names, wildtypes


def get_reference_sequence(filepath):
    """Read a nucleotide reference file and return the amino-acid sequence."""
    with open(filepath, "r") as f:
        nuc_seq = f.read().strip()
    aa_seq = ""
    for i in range(0, len(nuc_seq), 3):
        codon = nuc_seq[i:i + 3]
        aa_seq += CODON2AA.get(codon, "X")
    return aa_seq


def build_sequence_dataframe(pre_files, post_files, reference_seq):
    """
    Reconstruct single-mutant protein sequences from codon-count files.

    Returns a DataFrame with columns PreNums, PostNums, ProteinSequence.
    Sequences that appear in multiple replicates are aggregated (counts summed).
    """
    assert len(pre_files) == len(post_files), \
        "Number of pre- and post-selection files must match."

    # Load all arrays, verify consistency across replicates
    pre_arrays, post_arrays = [], []
    wildtypes_master, columns_master = None, None

    for fp_pre, fp_post in zip(pre_files, post_files):
        arr_pre, col_names, wt = load_codoncounts(fp_pre)
        arr_post, col_names_p, wt_p = load_codoncounts(fp_post)

        assert col_names == col_names_p, \
            f"Column names differ between {fp_pre} and {fp_post}"
        assert wt == wt_p, \
            f"Wildtypes differ between {fp_pre} and {fp_post}"

        if wildtypes_master is None:
            wildtypes_master = wt
            columns_master = col_names
        else:
            assert wt == wildtypes_master, "Wildtypes differ across replicates."
            assert col_names == columns_master, "Columns differ across replicates."

        pre_arrays.append(arr_pre)
        post_arrays.append(arr_post)

    col_aa = [CODON2AA.get(c, "X") for c in columns_master]
    wt_aa = [CODON2AA.get(c, "X") for c in wildtypes_master]

    records = []
    n_sites = pre_arrays[0].shape[0]
    n_variants = pre_arrays[0].shape[1]

    for i in range(n_sites):
        rows_pre = [a[i] for a in pre_arrays]
        rows_post = [a[i] for a in post_arrays]
        for j in range(n_variants):
            if col_aa[j] == wt_aa[i]:
                continue
            pre_nums = [int(r[j]) for r in rows_pre]
            post_nums = [int(r[j]) for r in rows_post]
            mut_seq = reference_seq[:i] + col_aa[j] + reference_seq[i + 1:]
            records.append({
                "PreNums": pre_nums,
                "PostNums": post_nums,
                "ProteinSequence": mut_seq,
            })

    df = pd.DataFrame(records)

    # Aggregate duplicate sequences (same mutation observed at multiple codons is rare
    # but possible; also ensures the output is deduplicated)
    df = df.groupby("ProteinSequence", as_index=False).agg({
        "PreNums": lambda x: [sum(v) for v in zip(*x)],
        "PostNums": lambda x: [sum(v) for v in zip(*x)],
    })

    return df


# ---------------------------------------------------------------------------
# ESM embedding
# ---------------------------------------------------------------------------

def pool_sequence_representation(token_representations, inputs):
    """Mean-pool token representations over non-padding positions."""
    attention_mask = inputs["attention_mask"]
    masked = token_representations * attention_mask.unsqueeze(-1)
    summed = masked.sum(dim=1)
    counts = attention_mask.sum(dim=1).unsqueeze(-1)
    return (summed / counts).squeeze(0).cpu().numpy()  # (embedding_dim,)


def embed_sequence(sequence, tokenizer, model):
    """
    Return a (num_layers, embedding_dim) array of mean-pooled hidden states
    for all transformer layers including the embedding layer.
    """
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    layer_embeddings = [
        pool_sequence_representation(layer, inputs)
        for layer in outputs.hidden_states
    ]
    return np.vstack(layer_embeddings)  # (num_layers, embedding_dim)


def embed_dataframe(seq_df, esm_model, embed_zeroes=False):
    """
    Add an Embeddings column to seq_df in-place.
    Sequences whose pre-selection counts are all zero get None unless
    embed_zeroes=True.
    """
    tokenizer = AutoTokenizer.from_pretrained(esm_model, do_lower_case=False)
    model = AutoModel.from_pretrained(esm_model)
    model.eval()

    pre_counts = seq_df["PreNums"].tolist()
    sequences = seq_df["ProteinSequence"].tolist()
    embeddings = []
    start = time.time()

    for i, seq in enumerate(sequences):
        has_observations = any(x > 0 for x in pre_counts[i])
        if embed_zeroes or has_observations:
            emb = embed_sequence(seq, tokenizer, model)
        else:
            emb = None
        embeddings.append(emb)

        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start
            remaining = elapsed / i * (len(sequences) - i)
            print(
                f"  [{i}/{len(sequences)}] elapsed {elapsed/60:.1f} min, "
                f"est. remaining {remaining/60:.1f} min"
            )

    seq_df["Embeddings"] = embeddings
    return seq_df


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    required = ["protein", "pre_selection_files", "post_selection_files",
                "reference_sequence_file"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config missing required key: '{key}'")

    cfg.setdefault("esm_model", "facebook/esm2_t30_150M_UR50D")
    cfg.setdefault("embed_zeroes", False)

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def chunk_output_name(output_file, chunk_idx):
    """Insert _chunk{N} before the file extension: foo.pkl -> foo_chunk3.pkl"""
    stem, ext = os.path.splitext(output_file)
    return f"{stem}_chunk{chunk_idx}{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct protein sequences and compute ESM embeddings."
    )
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("output_file", help="Output pickle filename (e.g. BF520_embeddings.pkl)")
    parser.add_argument("SCRATCH_DIR", help="Path to scratch directory for temporary files")
    parser.add_argument("--embed_zeroes", action="store_true",
                        help="Embed sequences with zero pre-selection counts")
    parser.add_argument("--esm_model", default=None,
                        help="Override ESM model (e.g. facebook/esm2_t6_8M_UR50D)")
    parser.add_argument("--chunk_idx", type=int, default=None,
                        help="Index of this chunk (0-based). Must be used with --n_chunks.")
    parser.add_argument("--n_chunks", type=int, default=None,
                        help="Total number of chunks (array size). Must be used with --chunk_idx.")
    args = parser.parse_args()

    chunked = args.chunk_idx is not None
    if chunked:
        assert args.n_chunks is not None, "--n_chunks is required when --chunk_idx is set"
        assert 0 <= args.chunk_idx < args.n_chunks, \
            f"--chunk_idx {args.chunk_idx} out of range for --n_chunks {args.n_chunks}"

    cfg = load_config(args.config)

    if args.embed_zeroes:
        cfg["embed_zeroes"] = True
    if args.esm_model is not None:
        cfg["esm_model"] = args.esm_model

    # Resolve file paths relative to DATA_DIR if not absolute
    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(DATA_DIR, p)

    pre_files = [resolve(p) for p in cfg["pre_selection_files"]]
    post_files = [resolve(p) for p in cfg["post_selection_files"]]
    ref_file = resolve(cfg["reference_sequence_file"])

    # --- Scratch setup ---
    node = os.environ.get("SLURMD_NODENAME", "unknown")
    SCRATCH_EMBED_FOLDER = os.path.join(args.SCRATCH_DIR, "esm_embed_saves")
    os.makedirs(SCRATCH_EMBED_FOLDER, exist_ok=True)
    os.makedirs(HOME_SEQ_FOLDER, exist_ok=True)

    chunk_label = f"chunk {args.chunk_idx}/{args.n_chunks}" if chunked else "full"
    print(f"Running on node : {node}")
    print(f"Scratch folder  : {SCRATCH_EMBED_FOLDER}")
    print(f"Protein         : {cfg['protein']}")
    print(f"ESM model       : {cfg['esm_model']}")
    print(f"Mode            : {chunk_label}")
    print(f"embed_zeroes    : {cfg['embed_zeroes']}")

    # --- Step 1: reconstruct sequences ---
    print("\n=== Step 1: Reconstructing protein sequences ===")
    reference_seq = get_reference_sequence(ref_file)
    print(f"Reference length: {len(reference_seq)} aa")

    seq_df = build_sequence_dataframe(pre_files, post_files, reference_seq)
    print(f"Unique mutant sequences (total): {len(seq_df)}")

    # Save the full sequence table once (only from chunk 0 or non-chunked runs)
    if not chunked or args.chunk_idx == 0:
        seq_scratch_path = os.path.join(SCRATCH_EMBED_FOLDER, f"{cfg['protein']}_sequences.pkl")
        seq_df.to_pickle(seq_scratch_path)
        seq_home_path = os.path.join(HOME_SEQ_FOLDER, f"{cfg['protein']}_sequences.pkl")
        shutil.copy2(seq_scratch_path, seq_home_path)
        print(f"Sequences saved to {seq_home_path}")

    # Slice the DataFrame for this chunk
    if chunked:
        indices = list(range(args.chunk_idx, len(seq_df), args.n_chunks))
        seq_df = seq_df.iloc[indices].reset_index(drop=True)
        print(f"This chunk embeds {len(seq_df)} sequences "
              f"(every {args.n_chunks} starting at {args.chunk_idx})")

    # --- Step 2: compute embeddings ---
    print("\n=== Step 2: Computing ESM embeddings ===")
    seq_df = embed_dataframe(seq_df, cfg["esm_model"], embed_zeroes=cfg["embed_zeroes"])

    out_name = chunk_output_name(args.output_file, args.chunk_idx) if chunked else args.output_file
    embed_scratch_path = os.path.join(SCRATCH_EMBED_FOLDER, out_name)
    seq_df.to_pickle(embed_scratch_path)
    print(f"Embeddings saved to {embed_scratch_path}")

    embed_home_path = os.path.join(HOME_SEQ_FOLDER, out_name)
    shutil.copy2(embed_scratch_path, embed_home_path)
    print(f"Embeddings copied to {embed_home_path}")

    # --- Cleanup ---
    shutil.rmtree(args.SCRATCH_DIR, ignore_errors=True)
    print("Scratch cleaned up. Done!")


if __name__ == "__main__":
    main()
