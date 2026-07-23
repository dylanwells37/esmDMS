"""Command-line interface for the canonical ESM-DMS workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    SAEConfig,
    embed,
    embed_and_llr,
    llr_sequence_ids,
    masked_marginal_llr,
    train_sae,
)
from .inference import infer, prior_sweep, substitution_basis
from .import_data import process_imported_directory
from .metrics import evaluate_fitness
from .schema import Dataset, FeatureArtifact, SEQUENCE_ID
from .workflow import run_analysis_file


def _dataset_create(args: argparse.Namespace) -> None:
    reference = Path(args.reference).read_text()
    Dataset(
        args.name,
        reference,
        pd.read_csv(args.variants, dtype={SEQUENCE_ID: str}),
        pd.read_csv(args.trajectory, dtype={SEQUENCE_ID: str}),
        args.pathogenic_high_selection,
        _truncate(args.truncate),
    ).save(args.output)


def _process(args: argparse.Namespace) -> None:
    summary = process_imported_directory(
        args.input,
        args.output,
        datasets=args.datasets,
        force=args.force,
    )
    print(summary.to_string(index=False))


def _embed(args: argparse.Namespace) -> None:
    dataset = Dataset.load(args.dataset)
    layers = _layers(args.layers)
    artifacts = embed(
        dataset,
        args.model,
        layers=layers,
        pooling=args.pooling,
        window_size=args.window_size,
        truncate=_truncate(args.truncate),
        device=args.device,
        dtype=args.dtype,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for layer, artifact in artifacts.items():
        artifact.save(output / f"layer_{layer}.npz")


def _embed_llr(args: argparse.Namespace) -> None:
    dataset = Dataset.load(args.dataset)

    def progress(stage: str, completed: int, total: int) -> None:
        interval = max(1, total // 10)
        if completed == 1 or completed == total or completed % interval == 0:
            print(f"[{stage}] {completed}/{total}", flush=True)

    embeddings, llr = embed_and_llr(
        dataset,
        args.model,
        layers=_layers(args.layers),
        pooling=args.pooling,
        window_size=args.window_size,
        truncate=_truncate(args.truncate),
        device=args.device,
        dtype=args.dtype,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        progress=progress,
    )
    output = Path(args.output)
    for layer, artifact in embeddings.items():
        artifact.save(output / "embeddings" / f"layer_{layer}.npz")
    llr.save(output / "llr.npz")


def _layers(values: list[str]) -> list[int] | None:
    return None if values == ["all"] else [int(value) for value in values]


def _truncate(values: list[int] | None) -> tuple[int, int] | None:
    return None if values is None else (values[0], values[1])


def _llr(args: argparse.Namespace) -> None:
    masked_marginal_llr(
        Dataset.load(args.dataset),
        args.model,
        window_size=args.window_size,
        truncate=_truncate(args.truncate),
        device=args.device,
        dtype=args.dtype,
    ).save(args.output)


def _merge(args: argparse.Namespace) -> None:
    dataset = Dataset.load(args.dataset)
    artifacts = [FeatureArtifact.load(path) for path in args.artifacts]
    kind = artifacts[0].kind
    if kind in {"embedding", "sae"}:
        expected = dataset.sequence_ids
    elif kind == "llr_prior":
        expected = llr_sequence_ids(dataset)
    else:
        raise ValueError(
            "Cluster merging supports embedding, SAE, and LLR-prior artifacts."
        )
    FeatureArtifact.merge(
        artifacts, expected_sequence_ids=expected
    ).save(args.output)


def _sae(args: argparse.Namespace) -> None:
    config = SAEConfig(
        n_features=args.features,
        sparsity=args.sparsity,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mode=args.mode,
        k=args.k,
        seed=args.seed,
        center_on_reference=not args.no_reference_center,
    )
    train_sae(
        Dataset.load(args.dataset),
        FeatureArtifact.load(args.embeddings),
        config=config,
        model_path=args.model_output,
        device=args.device,
    ).save(args.output)


def _infer(args: argparse.Namespace) -> None:
    dataset = Dataset.load(args.dataset)
    features = (
        substitution_basis(dataset)
        if args.features is None
        else FeatureArtifact.load(args.features)
    )
    prior = FeatureArtifact.load(args.prior) if args.prior else None
    infer(
        dataset,
        features,
        gamma=args.gamma,
        prior=prior,
        prior_scale=args.alpha,
    ).fitness().save(args.output)


def _sweep(args: argparse.Namespace) -> None:
    dataset = Dataset.load(args.dataset)
    prior = FeatureArtifact.load(args.prior)
    gammas = np.logspace(args.gamma_start, args.gamma_stop, args.gamma_points)
    alphas = np.asarray(args.alphas, dtype=float)
    table = prior_sweep(
        dataset,
        prior,
        alphas=alphas,
        gammas=gammas,
        evaluate=lambda fitness: evaluate_fitness(
            dataset, fitness, min_review_stars=args.min_review_stars
        ),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser(
        "process", help="Convert imported count-bearing CSVs into canonical datasets."
    )
    command.add_argument(
        "input", help="Directory containing the imported dataset CSVs."
    )
    command.add_argument("--output", required=True, help="Canonical dataset root.")
    command.add_argument(
        "--datasets", nargs="*", help="Optional dataset stems to process."
    )
    command.add_argument(
        "--force", action="store_true", help="Replace existing dataset directories."
    )
    command.set_defaults(func=_process)

    command = commands.add_parser(
        "dataset", help="Validate and create a canonical dataset directory."
    )
    command.add_argument("--name", required=True)
    command.add_argument(
        "--reference",
        required=True,
        help="Text file containing the reference protein sequence.",
    )
    command.add_argument("--variants", required=True)
    command.add_argument("--trajectory", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--pathogenic-high-selection", action="store_true")
    command.add_argument(
        "--truncate",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Store one 1-based inclusive protein interval in dataset.json.",
    )
    command.set_defaults(func=_dataset_create)

    command = commands.add_parser("embed", help="Generate pooled protein embeddings.")
    command.add_argument("dataset")
    command.add_argument("--model", required=True)
    command.add_argument("--layers", nargs="+", default=["all"])
    command.add_argument("--pooling", choices=["mean", "max"], default="max")
    command.add_argument("--window-size", type=int, default=2048)
    command.add_argument(
        "--truncate",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Override dataset truncation with one 1-based inclusive interval.",
    )
    command.add_argument("--device")
    command.add_argument("--dtype")
    command.add_argument("--output", required=True)
    command.set_defaults(func=_embed)

    command = commands.add_parser(
        "embed-llr",
        help="Generate one row shard of embeddings and LLRs with one model load.",
    )
    command.add_argument("dataset")
    command.add_argument("--model", required=True)
    command.add_argument("--layers", nargs="+", default=["all"])
    command.add_argument("--pooling", choices=["mean", "max"], default="max")
    command.add_argument("--window-size", type=int, default=2048)
    command.add_argument(
        "--truncate",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Override dataset truncation with one 1-based inclusive interval.",
    )
    command.add_argument("--device")
    command.add_argument("--dtype")
    command.add_argument("--shard-index", type=int, default=0)
    command.add_argument("--num-shards", type=int, default=1)
    command.add_argument("--output", required=True)
    command.set_defaults(func=_embed_llr)

    command = commands.add_parser("llr", help="Generate a masked-marginal LLR prior.")
    command.add_argument("dataset")
    command.add_argument("--model", required=True)
    command.add_argument("--window-size", type=int, default=2048)
    command.add_argument(
        "--truncate",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Override dataset truncation with one 1-based inclusive interval.",
    )
    command.add_argument("--device")
    command.add_argument("--dtype")
    command.add_argument("--output", required=True)
    command.set_defaults(func=_llr)

    command = commands.add_parser(
        "merge", help="Merge canonical row-sharded feature artifacts."
    )
    command.add_argument("dataset")
    command.add_argument("artifacts", nargs="+")
    command.add_argument("--output", required=True)
    command.set_defaults(func=_merge)

    command = commands.add_parser("sae", help="Train an SAE on an embedding artifact.")
    command.add_argument("dataset")
    command.add_argument("embeddings")
    command.add_argument("--features", type=int)
    command.add_argument("--sparsity", type=float, default=1e-3)
    command.add_argument("--learning-rate", type=float, default=1e-3)
    command.add_argument("--epochs", type=int, default=200)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument(
        "--mode", choices=["normal", "topk", "batchtopk"], default="normal"
    )
    command.add_argument("--k", type=int)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--device")
    command.add_argument("--no-reference-center", action="store_true")
    command.add_argument("--model-output")
    command.add_argument("--output", required=True)
    command.set_defaults(func=_sae)

    command = commands.add_parser("infer", help="Run one popDMS inference.")
    command.add_argument("dataset")
    command.add_argument(
        "--features", help="Feature artifact; default is the substitution basis."
    )
    command.add_argument(
        "--prior", help="LLR prior artifact for substitution-basis inference."
    )
    command.add_argument("--gamma", type=float, required=True)
    command.add_argument("--alpha", type=float, default=1.0)
    command.add_argument("--output", required=True)
    command.set_defaults(func=_infer)

    command = commands.add_parser(
        "sweep", help="Run an LLR-prior alpha-by-gamma sweep."
    )
    command.add_argument("dataset")
    command.add_argument("prior")
    command.add_argument(
        "--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0]
    )
    command.add_argument("--gamma-start", type=float, default=-5)
    command.add_argument("--gamma-stop", type=float, default=4)
    command.add_argument("--gamma-points", type=int, default=52)
    command.add_argument("--min-review-stars", type=int, default=0)
    command.add_argument("--output", required=True)
    command.set_defaults(func=_sweep)

    command = commands.add_parser(
        "analyze", help="Run a configured multi-dataset analysis."
    )
    command.add_argument("config")
    command.set_defaults(func=lambda args: run_analysis_file(args.config))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
