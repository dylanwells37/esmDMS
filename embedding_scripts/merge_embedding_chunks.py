"""
Merge chunk pkl files produced by submit_embed_array.sh into a single DataFrame.

Usage:
    python embedding_scripts/merge_embedding_chunks.py <output_file> <n_chunks>

Example:
    python embedding_scripts/merge_embedding_chunks.py BF520_embeddings.pkl 10

Reads  data/sequence_data/BF520_embeddings.pkl.chunk_{0..9}
Writes data/sequence_data/BF520_embeddings.pkl  (merged in original order)
"""

import os
import sys
import argparse

import pandas as pd

PROJECT_DIR = "/net/dali/home/barton/dhw28/popDMS/esmDMS"
HOME_SEQ_FOLDER = os.path.join(PROJECT_DIR, "data", "sequence_data")


def chunk_output_name(output_file, chunk_idx):
    return f"{output_file}.chunk_{chunk_idx}"


def main():
    parser = argparse.ArgumentParser(description="Merge ESM embedding chunk files.")
    parser.add_argument("output_file",
                        help="Base output filename used during embedding (e.g. BF520_embeddings.pkl)")
    parser.add_argument("n_chunks", type=int,
                        help="Number of chunks (array size used during submission)")
    parser.add_argument("--delete_chunks", action="store_true",
                        help="Delete individual chunk files after merging")
    args = parser.parse_args()

    chunk_paths = []
    for i in range(args.n_chunks):
        name = chunk_output_name(args.output_file, i)
        path = os.path.join(HOME_SEQ_FOLDER, name)
        if not os.path.exists(path):
            print(f"ERROR: missing chunk file {path}", file=sys.stderr)
            sys.exit(1)
        chunk_paths.append(path)

    print(f"Loading {args.n_chunks} chunks...")
    chunks = [pd.read_pickle(p) for p in chunk_paths]
    merged = pd.concat(chunks, ignore_index=True)

    out_path = os.path.join(HOME_SEQ_FOLDER, args.output_file)
    merged.to_pickle(out_path)
    print(f"Merged {len(merged)} sequences → {out_path}")

    if args.delete_chunks:
        for p in chunk_paths:
            os.remove(p)
        print("Chunk files deleted.")


if __name__ == "__main__":
    main()
