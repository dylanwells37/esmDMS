#!/usr/bin/env python3
"""Estimate ESM-C inference memory for one unbatched protein sequence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


GIB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
    parameters: int
    layers: int
    hidden_size: int
    heads: int
    official_esmc: bool


MODEL_SPECS = {
    "300m": ModelSpec(333_000_000, 30, 960, 15, True),
    "600m": ModelSpec(575_000_000, 36, 1152, 18, True),
    "3b": ModelSpec(3_000_000_000, 36, 2560, 40, False),
    "6b": ModelSpec(6_350_000_000, 80, 2560, 40, True),
}

HOST_RAM_GIB = {"300m": 16, "600m": 24, "3b": 64, "6b": 128}
GPU_VRAM_GIB = {
    "reduced": {"300m": 8, "600m": 12, "3b": 24, "6b": 48},
    "fp32": {"300m": 12, "600m": 16, "3b": 40, "6b": 80},
}


def _gib(value: int | float) -> float:
    return float(value) / GIB


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_SPECS, default="300m")
    parser.add_argument("--length", type=int, default=1863)
    parser.add_argument("--sequences", type=int, default=101)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    if args.length <= 0 or args.sequences <= 0:
        parser.error("length and sequences must be positive")
    spec = MODEL_SPECS[args.model]
    element_bytes = 4 if args.dtype == "fp32" else 2
    tokens = args.length + 2

    weights = spec.parameters * element_bytes
    hidden_stack = (spec.layers + 1) * tokens * spec.hidden_size * element_bytes
    qkv = 3 * tokens * spec.hidden_size * element_bytes
    attention_scores = tokens * tokens * spec.heads * element_bytes
    pooled_artifacts = (
        (spec.layers + 1) * args.sequences * spec.hidden_size * 4
    )
    host_peak = 2 * spec.parameters * 4 + hidden_stack
    vram_key = "fp32" if args.dtype == "fp32" else "reduced"

    print(f"model={args.model} official_esmc={spec.official_esmc}")
    print(
        f"layers={spec.layers} hidden_size={spec.hidden_size} "
        f"heads={spec.heads} residues={args.length}"
    )
    print(f"model weights at {args.dtype}: {_gib(weights):.2f} GiB")
    print(f"all-layer hidden-state stack: {_gib(hidden_stack):.2f} GiB")
    print(f"one-layer QKV tensors: {_gib(qkv):.2f} GiB")
    print(
        "dense attention-score equivalent: "
        f"{_gib(attention_scores):.2f} GiB (SDPA backend dependent)"
    )
    print(
        f"all-layer pooled output for {args.sequences} rows: "
        f"{_gib(pooled_artifacts):.2f} GiB"
    )
    print(f"calculated host loading peak floor: {_gib(host_peak):.2f} GiB")
    print(f"recommended Slurm --mem: {HOST_RAM_GIB[args.model]}G")
    print(
        "recommended minimum GPU VRAM: "
        f"{GPU_VRAM_GIB[vram_key][args.model]} GiB"
    )
    if not spec.official_esmc:
        print(
            "note: Biohub does not publish an official ESM-C 3B checkpoint; "
            "this row estimates a compatible custom 3B configuration."
        )
    if args.dtype != "fp32":
        print(
            "note: host loading uses float32 checkpoint tensors before device "
            "conversion, so reduced precision mainly lowers GPU memory."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
