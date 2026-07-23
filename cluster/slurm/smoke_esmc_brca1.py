#!/usr/bin/env python3
"""Load ESM-C and embed one full-length BRCA1 protein sequence."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from esmdms.features import load_language_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="biohub/ESMC-6B")
    parser.add_argument(
        "--input", default="imported_data/MV_BRCA1_Findlay_2018.csv"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--min-gpu-memory-gib", type=float, default=48.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the Slurm job did not receive a GPU")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024**3
    if total_gib < args.min_gpu_memory_gib:
        raise RuntimeError(
            f"GPU has {total_gib:.1f} GiB, below the requested minimum "
            f"of {args.min_gpu_memory_gib:.1f} GiB"
        )

    with Path(args.input).open(newline="") as handle:
        row = next(csv.DictReader(handle))
    sequence = row["mutated_sequence"].strip()
    if len(sequence) != 1863:
        raise ValueError(f"Expected a 1,863-residue BRCA1 sequence, got {len(sequence)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    total_started = time.monotonic()
    started = time.monotonic()
    tokenizer, model = load_language_model(
        args.model, device="cuda", dtype=args.dtype
    )
    loaded_seconds = time.monotonic() - started

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

    inference_started = time.monotonic()
    with torch.inference_mode():
        outputs = model(
            sequence_tokens=sequence_tokens,
            sequence_id=attention_mask,
        )
        residue_embeddings = outputs.embeddings[0, residue_mask]
        pooled = residue_embeddings.float().mean(dim=0).cpu().numpy()
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - inference_started
    total_seconds = time.monotonic() - total_started

    np.save(output_dir / "brca1_mean_embedding.npy", pooled)
    metrics = {
        "status": "success",
        "model": args.model,
        "dtype": args.dtype,
        "sequence_length": len(sequence),
        "embedding_dimension": int(pooled.shape[0]),
        "gpu_name": properties.name,
        "gpu_total_gib": total_gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "model_load_seconds": loaded_seconds,
        "inference_seconds": inference_seconds,
        "total_python_seconds": total_seconds,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
