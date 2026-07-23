#!/usr/bin/env python3
"""Load ESM-C and embed one full-length BRCA1 protein sequence."""

from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from esmdms.features import load_language_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="biohub/ESMC-6B")
    parser.add_argument("--input", default="imported_data/MV_BRCA1_Findlay_2018.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--min-gpu-memory-gib", type=float, default=48.0)
    args = parser.parse_args()
    if args.num_sequences <= 0:
        parser.error("--num-sequences must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA initialization failed: either the job received no GPU or the "
            f"node driver is incompatible with this PyTorch CUDA {torch.version.cuda} "
            "build. Check the preceding PyTorch warning and nvidia-smi output."
        )
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024**3
    if total_gib < args.min_gpu_memory_gib:
        raise RuntimeError(
            f"GPU has {total_gib:.1f} GiB, below the requested minimum "
            f"of {args.min_gpu_memory_gib:.1f} GiB"
        )

    with Path(args.input).open(newline="") as handle:
        rows = list(islice(csv.DictReader(handle), args.num_sequences))
    if len(rows) != args.num_sequences:
        raise ValueError(
            f"Requested {args.num_sequences} sequences, but {args.input} "
            f"contains only {len(rows)} rows"
        )
    sequences = [row["mutated_sequence"].strip() for row in rows]
    invalid_lengths = [
        (index, len(sequence))
        for index, sequence in enumerate(sequences)
        if len(sequence) != 1863
    ]
    if invalid_lengths:
        raise ValueError(
            "Expected 1,863-residue BRCA1 sequences; invalid rows: "
            + ", ".join(f"{index} ({length})" for index, length in invalid_lengths)
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    total_started = time.monotonic()
    started = time.monotonic()
    tokenizer, model = load_language_model(args.model, device="cuda", dtype=args.dtype)
    loaded_seconds = time.monotonic() - started

    inference_started = time.monotonic()
    embeddings = []
    sequence_seconds = []
    for index, sequence in enumerate(sequences, start=1):
        sequence_started = time.monotonic()
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        special_mask = inputs.pop("special_tokens_mask").bool().cuda()
        attention_mask = inputs["attention_mask"].bool().cuda()
        sequence_tokens = inputs["input_ids"].cuda()
        residue_mask = (attention_mask & ~special_mask).squeeze(0)
        with torch.inference_mode():
            outputs = model(
                sequence_tokens=sequence_tokens,
                sequence_id=attention_mask,
            )
            residue_embeddings = outputs.embeddings[0, residue_mask]
            pooled = residue_embeddings.float().mean(dim=0).cpu().numpy()
        torch.cuda.synchronize()
        embeddings.append(pooled)
        sequence_seconds.append(time.monotonic() - sequence_started)
        print(
            f"embedded sequence {index}/{len(sequences)} "
            f"in {sequence_seconds[-1]:.3f}s",
            flush=True,
        )
    inference_seconds = time.monotonic() - inference_started
    total_seconds = time.monotonic() - total_started

    embedding_matrix = np.vstack(embeddings)
    embedding_filename = f"brca1_first{len(sequences)}_mean_embeddings.npy"
    np.save(output_dir / embedding_filename, embedding_matrix)
    metrics = {
        "status": "success",
        "model": args.model,
        "dtype": args.dtype,
        "num_sequences": len(sequences),
        "sequence_length": len(sequences[0]),
        "embedding_shape": list(embedding_matrix.shape),
        "embedding_dimension": int(embedding_matrix.shape[1]),
        "embedding_file": embedding_filename,
        "gpu_name": properties.name,
        "gpu_total_gib": total_gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "model_load_seconds": loaded_seconds,
        "inference_seconds": inference_seconds,
        "mean_seconds_per_sequence": float(np.mean(sequence_seconds)),
        "min_seconds_per_sequence": float(np.min(sequence_seconds)),
        "max_seconds_per_sequence": float(np.max(sequence_seconds)),
        "total_python_seconds": total_seconds,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
