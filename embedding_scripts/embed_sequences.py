import io
import os
import re
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


import psutil

# Project directory
#PROJECT_DIR = "/net/dali/home/barton/dhw28/popDMS/esmDMS"
PROJECT_DIR = "/Users/dylanwells/popDMS/esmDMS"
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


def get_reference_sequence_from_wildtypes(codoncounts_filepath):
    """Derive the reference amino-acid sequence from the wildtype codon at each site."""
    _, _, wildtypes = load_codoncounts(codoncounts_filepath)
    return "".join(CODON2AA.get(codon, "X") for codon in wildtypes)


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
# MaveDB sequence reconstruction
# ---------------------------------------------------------------------------

_SNV_RE = re.compile(r'^(\d+)([ACGTacgt])>([ACGTacgt])$')


def get_reference_nuc_sequence(filepath):
    """Read a nucleotide CDS reference file (plain text or FASTA) and return
    the raw uppercase nucleotide string."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    return "".join(
        line.strip() for line in lines if not line.startswith(">")
    ).upper()


def translate_nuc_sequence(nuc_seq):
    """Translate a nucleotide string to an amino-acid string using CODON2AA."""
    return "".join(
        CODON2AA.get(nuc_seq[i:i + 3], "X") for i in range(0, len(nuc_seq) - 2, 3)
    )


def parse_hgvs_nt(hgvs_str):
    """
    Parse an HGVS nucleotide string into a list of (pos_0indexed, ref_nuc, alt_nuc).

    Handles single and multi-substitution entries:
        'c.90G>C'       -> [(89, 'G', 'C')]
        'c.[8C>T;9C>G]' -> [(7, 'C', 'T'), (8, 'C', 'G')]

    Returns an empty list for indels, frameshifts, '_wt', NaN, or any entry
    containing a non-SNV token.
    """
    if not isinstance(hgvs_str, str):
        return []
    hgvs_str = hgvs_str.strip()

    if hgvs_str.startswith("c.[") and hgvs_str.endswith("]"):
        tokens = hgvs_str[3:-1].split(";")
    elif hgvs_str.startswith("c."):
        tokens = [hgvs_str[2:]]
    else:
        return []

    result = []
    for tok in tokens:
        m = _SNV_RE.match(tok.strip())
        if m is None:
            return []  # non-SNV token invalidates the whole entry
        pos_1indexed = int(m.group(1))
        ref_nuc, alt_nuc = m.group(2).upper(), m.group(3).upper()
        if ref_nuc == alt_nuc:
            return []
        result.append((pos_1indexed - 1, ref_nuc, alt_nuc))

    return result


def apply_substitutions(ref_nuc_seq, substitutions):
    """
    Apply a list of (pos_0indexed, ref_nuc, alt_nuc) substitutions to the
    reference CDS. Returns the mutant nucleotide string, or None if any
    position is out of range or the reference nucleotide does not match.
    """
    nuc_list = list(ref_nuc_seq)
    for pos, ref_nuc, alt_nuc in substitutions:
        if pos >= len(nuc_list):
            return None
        if nuc_list[pos] != ref_nuc:
            return None  # reference mismatch
        nuc_list[pos] = alt_nuc
    return "".join(nuc_list)


def build_sequence_dataframe_mavedb(csv_filepath, ref_nuc_seq, skip_stop_codons=True):
    """
    Load a MaveDB-format CSV and return a DataFrame with columns:
        PreNums         - list of pre-selection counts (one per replicate)
        PostNums        - list of post-selection counts (one per replicate)
        ProteinSequence - full single-mutant protein sequence

    Nucleotide substitutions from hgvs_nt are applied to ref_nuc_seq; the
    mutant CDS is translated and kept only when exactly one amino acid differs
    from the reference (single-mutant filter). Double mutants and synonymous
    variants are discarded. Rows mapping to the same mutant protein have their
    counts summed.

    Cannot use pd.read_csv(comment='#') because '#' appears inside MaveDB
    accession values (e.g. 'urn:mavedb:00000043-a-1#1'), which would corrupt
    every data row. Instead, file-level comment lines are stripped manually.
    """
    with open(csv_filepath) as f:
        content = "".join(line for line in f if not line.startswith("#"))
    df = pd.read_csv(io.StringIO(content))

    # Detect replicate pre/post column pairs (Replicate_A_c_0 / Replicate_A_c_1)
    seen = {}
    for col in df.columns:
        m = re.match(r"^(Replicate_\w+)_c_(\d+)$", col)
        if m:
            seen.setdefault(m.group(1), {})[int(m.group(2))] = col
    pre_cols  = [tp[0] for tp in seen.values() if 0 in tp and 1 in tp]
    post_cols = [tp[1] for tp in seen.values() if 0 in tp and 1 in tp]
    if not pre_cols:
        raise ValueError(
            "No replicate count columns found. Expected names like 'Replicate_A_c_0'."
        )
    ref_aa_seq = translate_nuc_sequence(ref_nuc_seq)

    records = []
    for _, row in df.iterrows():
        hgvs_nt = row.get("hgvs_nt")

        if hgvs_nt == "_wt":
            aa_seq = ref_aa_seq
        else:
            substitutions = parse_hgvs_nt(hgvs_nt)
            if not substitutions:
                continue
            mut_nuc_seq = apply_substitutions(ref_nuc_seq, substitutions)
            if mut_nuc_seq is None:
                continue
            aa_seq = translate_nuc_sequence(mut_nuc_seq)

        records.append({
            "PreNums":  [int(row[c]) if pd.notna(row[c]) else 0 for c in pre_cols],
            "PostNums": [int(row[c]) if pd.notna(row[c]) else 0 for c in post_cols],
            "ProteinSequence": aa_seq,
        })

    result = pd.DataFrame(records, columns=["PreNums", "PostNums", "ProteinSequence"])

    if not result.empty:
        result = result.groupby("ProteinSequence", as_index=False).agg({
            "PreNums":  lambda x: [sum(v) for v in zip(*x)],
            "PostNums": lambda x: [sum(v) for v in zip(*x)],
        })

    return result


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

        if i % 5 == 0 and i > 0:
            elapsed = time.time() - start
            remaining = elapsed / i * (len(sequences) - i)
            print(
                f"  [{i}/{len(sequences)}] elapsed {elapsed/60:.1f} min, "
                f"est. remaining {remaining/60:.1f} min"
            )
            print(f"  Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f} MB")

    seq_df["Embeddings"] = embeddings
    return seq_df


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    if "protein" not in cfg:
        raise ValueError("Config missing required key: 'protein'")

    if "mavedb_file" in cfg:
        # MaveDB mode — requires a nucleotide reference file
        if "reference_sequence_file" not in cfg:
            raise ValueError(
                "MaveDB config requires 'reference_sequence_file' (nucleotide CDS)."
            )
        cfg.setdefault("skip_stop_codons", True)
    else:
        # Codon-count mode
        for key in ("pre_selection_files", "post_selection_files"):
            if key not in cfg:
                raise ValueError(f"Config missing required key: '{key}'")

    cfg.setdefault("esm_model", "facebook/esm2_t30_150M_UR50D")
    cfg.setdefault("embed_zeroes", False)

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.embed_zeroes:
        cfg["embed_zeroes"] = True
    if args.esm_model is not None:
        cfg["esm_model"] = args.esm_model

    # Resolve file paths relative to DATA_DIR if not absolute
    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(DATA_DIR, p)

    mavedb_mode = "mavedb_file" in cfg

    # --- Scratch setup ---
    node = os.environ.get("SLURMD_NODENAME", "unknown")
    SCRATCH_EMBED_FOLDER = os.path.join(args.SCRATCH_DIR, "esm_embed_saves")
    os.makedirs(SCRATCH_EMBED_FOLDER, exist_ok=True)
    os.makedirs(HOME_SEQ_FOLDER, exist_ok=True)

    ref_file = resolve(cfg["reference_sequence_file"]) if cfg.get("reference_sequence_file") else None

    print(f"Running on node : {node}")
    print(f"Scratch folder  : {SCRATCH_EMBED_FOLDER}")
    print(f"Protein         : {cfg['protein']}")
    print(f"ESM model       : {cfg['esm_model']}")
    print(f"embed_zeroes    : {cfg['embed_zeroes']}")

    if mavedb_mode:
        mavedb_file = resolve(cfg["mavedb_file"])
        print(f"Mode            : MaveDB")
        print(f"MaveDB file     : {mavedb_file}")
        print(f"Reference seq   : {ref_file}")
        print(f"skip_stop_codons: {cfg['skip_stop_codons']}")
    else:
        pre_files = [resolve(p) for p in cfg["pre_selection_files"]]
        post_files = [resolve(p) for p in cfg["post_selection_files"]]
        ref_source = ref_file if ref_file else f"wildtypes in {os.path.basename(pre_files[0])}"
        print(f"Mode            : codon-count")
        print(f"Pre files       : {pre_files}")
        print(f"Post files      : {post_files}")
        print(f"Reference seq   : {ref_source}")

    # --- Step 1: reconstruct sequences ---
    print("\n=== Step 1: Reconstructing protein sequences ===")

    if mavedb_mode:
        ref_nuc_seq = get_reference_nuc_sequence(ref_file)
        print(f"Reference length: {len(ref_nuc_seq) // 3} aa ({len(ref_nuc_seq)} nt)")
        seq_df = build_sequence_dataframe_mavedb(
            mavedb_file, ref_nuc_seq,
            skip_stop_codons=cfg["skip_stop_codons"],
        )
    else:
        if ref_file:
            reference_seq = get_reference_sequence(ref_file)
        else:
            reference_seq = get_reference_sequence_from_wildtypes(pre_files[0])
            print("No reference file provided — derived reference from wildtype codons.")
        print(f"Reference length: {len(reference_seq)} aa")
        seq_df = build_sequence_dataframe(pre_files, post_files, reference_seq)

    print(f"Unique mutant sequences: {len(seq_df)}")

    seq_scratch_path = os.path.join(SCRATCH_EMBED_FOLDER, f"{cfg['protein']}_sequences.pkl")
    seq_df.to_pickle(seq_scratch_path)
    print(f"Sequences saved to {seq_scratch_path}")

    seq_home_path = os.path.join(HOME_SEQ_FOLDER, f"{cfg['protein']}_sequences.pkl")
    shutil.copy2(seq_scratch_path, seq_home_path)
    print(f"Sequences copied to {seq_home_path}")

    # --- Step 2: compute embeddings ---
    print("\n=== Step 2: Computing ESM embeddings ===")
    seq_df = embed_dataframe(seq_df, cfg["esm_model"], embed_zeroes=cfg["embed_zeroes"])

    embed_scratch_path = os.path.join(SCRATCH_EMBED_FOLDER, args.output_file)
    seq_df.to_pickle(embed_scratch_path)
    print(f"Embeddings saved to {embed_scratch_path}")

    embed_home_path = os.path.join(HOME_SEQ_FOLDER, args.output_file)
    shutil.copy2(embed_scratch_path, embed_home_path)
    print(f"Embeddings copied to {embed_home_path}")

    # --- Cleanup ---
    shutil.rmtree(args.SCRATCH_DIR, ignore_errors=True)
    print("Scratch cleaned up. Done!")


if __name__ == "__main__":
    main()
