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

# Project directory is the root of the repository, assumed to be the parent of the current file's directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.insert(0, PROJECT_DIR)

CODON2AA = {'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',            # Map from codons to amino acids
            'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
            'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
            'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
            'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
            'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
            'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
            'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
            'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
            'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
            'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
            'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
            'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
            'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
            'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
            'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W' }

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
    if len(pre_files) != len(post_files):
        raise ValueError("Number of pre- and post-selection files must match.")
    if not pre_files:
        raise ValueError("At least one pre/post file pair is required.")

    pre_arrays, post_arrays = [], []
    wildtypes_master, columns_master = None, None

    for fp_pre, fp_post in zip(pre_files, post_files):
        arr_pre, col_names, wt = load_codoncounts(fp_pre)
        arr_post, col_names_p, wt_p = load_codoncounts(fp_post)

        if col_names != col_names_p:
            raise ValueError(f"Column names differ between {fp_pre} and {fp_post}")
        if wt != wt_p:
            raise ValueError(f"Wildtypes differ between {fp_pre} and {fp_post}")

        if wildtypes_master is None:
            wildtypes_master = wt
            columns_master = col_names
        else:
            if wt != wildtypes_master:
                raise ValueError("Wildtypes differ across replicates.")
            if col_names != columns_master:
                raise ValueError("Columns differ across replicates.")

        pre_arrays.append(arr_pre)
        post_arrays.append(arr_post)

    col_aa = [CODON2AA.get(c, "X") for c in columns_master]
    wt_aa = [CODON2AA.get(c, "X") for c in wildtypes_master]

    seq_to_index = {}
    index_to_protein_sequence = {}
    index_to_mutation_sites = {}
    counts = {}

    n_sites = pre_arrays[0].shape[0]
    n_variants = pre_arrays[0].shape[1]
    n_reps = len(pre_arrays)

    for i in range(n_sites):
        for j in range(n_variants):
            if col_aa[j] == wt_aa[i]:
                continue

            mut_seq = reference_seq[:i] + col_aa[j] + reference_seq[i + 1:]

            if "*" in mut_seq:
                continue

            if mut_seq not in seq_to_index:
                seq_index = len(seq_to_index)
                seq_to_index[mut_seq] = seq_index
                index_to_protein_sequence[seq_index] = mut_seq
                index_to_mutation_sites[seq_index] = [i]
            else:
                seq_index = seq_to_index[mut_seq]

            for rep_idx in range(n_reps):
                pre_key = (seq_index, rep_idx + 1, 0)
                post_key = (seq_index, rep_idx + 1, 1)

                counts[pre_key] = counts.get(pre_key, 0) + int(pre_arrays[rep_idx][i, j])
                counts[post_key] = counts.get(post_key, 0) + int(post_arrays[rep_idx][i, j])

    records = [
        {
            "SequenceIndex": seq_index,
            "Replicate": rep,
            "Generation": gen,
            "Frequency": freq,
        }
        for (seq_index, rep, gen), freq in counts.items()
    ]

    df = pd.DataFrame(records, columns=["SequenceIndex", "Replicate", "Generation", "Frequency"])
    return df, index_to_protein_sequence, index_to_mutation_sites


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


def build_sequence_dataframe_mavedb(csv_filepath, ref_nuc_seq, 
                                    use_replicates=None, skip_stop_codons=True):
    """
    Load a MaveDB-format CSV and return:
        result_df
            Long-format DataFrame with columns:
                SequenceIndex, Replicate, Generation, Frequency

        index_to_protein_sequence
            Dict mapping SequenceIndex -> full protein sequence

        index_to_mutation_sites
            Dict mapping SequenceIndex -> list of 0-indexed mutation sites

    This avoids storing the full ProteinSequence and MutationSites on every
    replicate/generation/frequency row.
    """
    with open(csv_filepath) as f:
        content = "".join(line for line in f if not line.startswith("#"))
    df = pd.read_csv(io.StringIO(content))

    _COL_RE = re.compile(r"^(.+)_c_(\d+)$")
    seen = {}
    for col in df.columns:
        m = _COL_RE.match(col)
        if m:
            seen.setdefault(m.group(1), {})[int(m.group(2))] = col

    if not seen:
        raise ValueError(
            "No count columns found. Expected names ending in '_c_N' "
            "(e.g. 'Replicate_A_c_0')."
        )

    sorted_rep_names = sorted(seen.keys())
    rep_name_to_idx = {name: idx + 1 for idx, name in enumerate(sorted_rep_names)}

    ref_aa_seq = translate_nuc_sequence(ref_nuc_seq)

    records = []
    seq_to_index = {}
    index_to_protein_sequence = {}
    index_to_mutation_sites = {}
    counts = {}

    for _, row in df.iterrows():
        hgvs_nt = row.get("hgvs_nt")

        if hgvs_nt == "_wt":
            aa_seq = ref_aa_seq
            mutation_sites = []
        else:
            substitutions = parse_hgvs_nt(hgvs_nt)
            if not substitutions:
                continue

            mut_nuc_seq = apply_substitutions(ref_nuc_seq, substitutions)
            if mut_nuc_seq is None:
                continue

            aa_seq = translate_nuc_sequence(mut_nuc_seq)

            if skip_stop_codons and "*" in aa_seq:
                continue

            mutation_sites = [
                i
                for i, (ref_aa, alt_aa) in enumerate(zip(ref_aa_seq, aa_seq))
                if ref_aa != alt_aa
            ]

            if not mutation_sites and aa_seq != ref_aa_seq:
                continue

        if aa_seq not in seq_to_index:
            seq_index = len(seq_to_index)
            seq_to_index[aa_seq] = seq_index
            index_to_protein_sequence[seq_index] = aa_seq
            index_to_mutation_sites[seq_index] = mutation_sites
        else:
            seq_index = seq_to_index[aa_seq]

        for rep_name, gen_dict in seen.items():
            if use_replicates is not None and rep_name not in use_replicates:
                continue
            rep_idx = rep_name_to_idx[rep_name]

            for gen_idx, col_name in sorted(gen_dict.items()):
                freq = int(row[col_name]) if pd.notna(row.get(col_name)) else 0
                key = (seq_index, rep_idx, gen_idx)
                counts[key] = counts.get(key, 0) + freq

    records = [
        {
            "SequenceIndex": seq_index,
            "Replicate": rep_idx,
            "Generation": gen_idx,
            "Frequency": freq,
        }
        for (seq_index, rep_idx, gen_idx), freq in counts.items()
    ]

    result = pd.DataFrame(
        records,
        columns=["SequenceIndex", "Replicate", "Generation", "Frequency"],
    )

    return result, index_to_protein_sequence, index_to_mutation_sites


# ---------------------------------------------------------------------------
# ESM embedding
# ---------------------------------------------------------------------------

def _residue_token_mask(inputs):
    """Mask real residue tokens, excluding CLS/EOS/padding special tokens."""
    attention_mask = inputs["attention_mask"].bool()
    special_mask = inputs.get("special_tokens_mask")
    if special_mask is None:
        return attention_mask
    return attention_mask & ~special_mask.bool()


def pool_sequence_representation(token_representations, inputs):
    """Mean-pool token representations over real residue positions."""
    residue_mask = _residue_token_mask(inputs)
    masked = token_representations * residue_mask.unsqueeze(-1)
    summed = masked.sum(dim=1)
    counts = residue_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return (summed / counts).squeeze(0).cpu().numpy()  # (embedding_dim,)


def all_residue_representation(token_representations, inputs):
    """Return representations for every real residue token."""
    residue_mask = _residue_token_mask(inputs).squeeze(0)
    return token_representations[:, residue_mask, :].squeeze(0).cpu().numpy()


def embed_sequence(sequence, tokenizer, model):
    """
    Return full all-residue hidden-state embeddings for all transformer layers.

    Shape: (num_residues, num_layers, embedding_dim)
    """
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        return_special_tokens_mask=True,
    )
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    layer_embeddings = [
        all_residue_representation(layer, inputs)
        for layer in outputs.hidden_states
    ]
    return np.stack(layer_embeddings, axis=1)


def _normalise_mutation_sites(value):
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(v) for v in value]


def embed_dataframe(seq_df, esm_model, embed_zeroes=False, n_test=None):
    """
    Embed protein sequences and return an annotated DataFrame.

    Codon-count format (columns PreNums / PostNums / ProteinSequence):
        Adds an Embeddings column; returns the same wide-format DataFrame.

    MaveDB long format (columns ProteinSequence / Replicate / Generation / Frequency):
        Embeds each unique sequence once, joins the embedding back, and returns
        a long-format DataFrame with columns:
            ProteinSequence, Generation, Embedding, Frequency, Replicate
    """
    tokenizer = AutoTokenizer.from_pretrained(esm_model, do_lower_case=False)
    model = AutoModel.from_pretrained(esm_model)
    model.eval()


    if n_test is not None:

        keep = seq_df["ProteinSequence"].unique()[:n_test]
        seq_df = seq_df[seq_df["ProteinSequence"].isin(keep)].reset_index(drop=True)

        print(f"n_test={n_test}: embedding {len(seq_df['ProteinSequence'].unique())} sequences")
        
    unique_seqs = seq_df["ProteinSequence"].unique()
    total_freq = seq_df.groupby("ProteinSequence")["Frequency"].sum()
    seq_to_emb = {}
    start = time.time()

    for i, seq in enumerate(unique_seqs):
        has_observations = embed_zeroes or total_freq.get(seq, 0) > 0
        if has_observations:
            seq_to_emb[seq] = embed_sequence(seq, tokenizer, model)
        else:
            seq_to_emb[seq] = None

        if i % 5 == 0 and i > 0:
            elapsed = time.time() - start
            remaining = elapsed / i * (len(unique_seqs) - i)
            print(
                f"  [{i}/{len(unique_seqs)}] elapsed {elapsed/60:.1f} min, "
                f"est. remaining {remaining/60:.1f} min"
            )
            print(f"  Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f} MB")

    out = seq_df.copy()
    out["Embedding"] = out["ProteinSequence"].map(seq_to_emb)
    out["EmbeddingMethod"] = "all_data"
    extra_cols = [c for c in ["MutationSites", "MutationSite", "MutationSiteIndex"] if c in out.columns]
    return out[["ProteinSequence", *extra_cols, "EmbeddingMethod",
                "Generation", "Embedding", "Frequency", "Replicate"]]

    

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
    cfg.setdefault("embedding_method", "all_data")
    cfg["pool_mutations"] = False
    if cfg["embedding_method"] in {"cls", "cls_token"}:
        raise ValueError("CLS embeddings are no longer supported.")
    if cfg["embedding_method"] not in {"all_data", "per_residue"}:
        raise ValueError("embedding_method must be all_data. Derive embedding types at runtime.")
    cfg["embedding_method"] = "all_data"

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
    parser.add_argument("--n_test", type=int, default=None,
                        help="Only embed the first N sequences (for testing)")
    parser.add_argument("--embedding_method", choices=["all_data", "per_residue"],
                        default=None,
                        help="Embedding extraction method. Only full all_data generation is supported.")
    parser.add_argument("--chunk_idx", type=int, default=0,
                        help="Index of this chunk (0-based, used in array jobs)")
    parser.add_argument("--n_chunks", type=int, default=1,
                        help="Total number of chunks (must match --array size)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.embed_zeroes:
        cfg["embed_zeroes"] = True
    if args.esm_model is not None:
        cfg["esm_model"] = args.esm_model
    if args.embedding_method is not None:
        cfg["embedding_method"] = args.embedding_method

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
    print(f"embedding_method: {cfg['embedding_method']}")
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

    print(f"Unique mutant sequences: {len(seq_df['ProteinSequence'].unique())}")

    # Save full sequence dataframe only from chunk 0 (or when not chunking)
    if args.chunk_idx == 0:
        seq_scratch_path = os.path.join(SCRATCH_EMBED_FOLDER, f"{cfg['protein']}_sequences.pkl")
        seq_df.to_pickle(seq_scratch_path)
        print(f"Sequences saved to {seq_scratch_path}")

        seq_home_path = os.path.join(HOME_SEQ_FOLDER, f"{cfg['protein']}_sequences.pkl")
        shutil.copy2(seq_scratch_path, seq_home_path)
        print(f"Sequences copied to {seq_home_path}")

    # --- Chunking ---
    if args.n_chunks > 1:
        chunk_indices = np.array_split(np.arange(len(seq_df)), args.n_chunks)
        idx = chunk_indices[args.chunk_idx]
        seq_df = seq_df.iloc[idx].reset_index(drop=True)
        print(f"\nChunk {args.chunk_idx}/{args.n_chunks - 1}: "
              f"{len(seq_df)} sequences (rows {idx[0]}–{idx[-1]})")

    # --- Step 2: compute embeddings ---
    print("\n=== Step 2: Computing ESM embeddings ===")
    seq_df = embed_dataframe(seq_df, cfg["esm_model"], embed_zeroes=cfg["embed_zeroes"],
                             n_test=args.n_test)

    # Append chunk suffix to output filename when running in array mode
    if args.n_chunks > 1:
        output_filename = f"{args.output_file}.chunk_{args.chunk_idx}"
    else:
        output_filename = args.output_file

    embed_scratch_path = os.path.join(SCRATCH_EMBED_FOLDER, output_filename)
    seq_df.to_pickle(embed_scratch_path)
    print(f"Embeddings saved to {embed_scratch_path}")

    embed_home_path = os.path.join(HOME_SEQ_FOLDER, output_filename)
    shutil.copy2(embed_scratch_path, embed_home_path)
    print(f"Embeddings copied to {embed_home_path}")

    # --- Cleanup ---
    shutil.rmtree(args.SCRATCH_DIR, ignore_errors=True)
    print("Scratch cleaned up. Done!")


if __name__ == "__main__":
    main()
