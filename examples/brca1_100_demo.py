#!/usr/bin/env python3
"""Run the complete ESM-DMS workflow on the first 100 BRCA1 variants."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from esmdms.features import SAEConfig, embed, masked_marginal_llr, train_sae
from esmdms.import_data import process_imported_csv
from esmdms.schema import Dataset, FeatureArtifact, SEQUENCE_ID
from esmdms.workflow import run_analysis_file


SOURCE_NAME = "MV_BRCA1_Findlay_2018"
MODEL_NAME = "biohub/ESMC-300M"
FINAL_LAYER = 30
VARIANT_COUNT = 100


def _progress(message: str) -> None:
    print(f"[brca1-demo] {message}", flush=True)


def _source_dataset() -> Dataset:
    canonical_path = REPOSITORY_ROOT / "datasets" / SOURCE_NAME
    if (canonical_path / "dataset.json").is_file():
        return Dataset.load(canonical_path)

    imported_path = REPOSITORY_ROOT / "imported_data" / f"{SOURCE_NAME}.csv"
    if not imported_path.is_file():
        raise FileNotFoundError(
            f"Missing BRCA1 input {imported_path}. Run this script from a repository "
            "checkout containing imported_data."
        )
    _progress("Converting imported BRCA1 counts to the canonical dataset schema")
    return process_imported_csv(imported_path, canonical_path.parent)


def build_subset_dataset(
    source: Dataset,
    destination: str | Path,
    *,
    variant_count: int = VARIANT_COUNT,
    force: bool = False,
) -> Dataset:
    """Save the first N assayed variants plus one wild-type reference row."""
    destination = Path(destination)
    manifest = destination / "dataset.json"
    if manifest.is_file() and not force:
        existing = Dataset.load(destination)
        selection = existing.metadata.get("selection", {})
        if (
            selection.get("source_dataset") != source.name
            or selection.get("variant_count") != variant_count
        ):
            raise ValueError(
                f"Existing demo dataset at {destination} has different selection "
                "parameters; rerun with --force."
            )
        return existing

    if variant_count <= 0:
        raise ValueError("variant_count must be positive.")
    reference_rows = source.variants[
        source.variants["is_synonymous"]
        & source.variants["protein_sequence"].eq(source.reference_sequence)
    ]
    if reference_rows.empty:
        raise ValueError("Source dataset has no wild-type reference row.")

    assayed = source.variants.loc[~source.variants["is_synonymous"]].head(
        variant_count
    )
    if len(assayed) != variant_count:
        raise ValueError(
            f"Source dataset has only {len(assayed)} assayed variants; "
            f"{variant_count} requested."
        )
    selected_ids = set(assayed[SEQUENCE_ID].astype(str))
    variants = pd.concat([reference_rows.head(1), assayed], ignore_index=True)
    trajectory = source.trajectory[
        source.trajectory[SEQUENCE_ID].astype(str).isin(selected_ids)
    ].copy()
    subset = Dataset(
        name=f"{source.name}__first{variant_count}",
        reference_sequence=source.reference_sequence,
        variants=variants,
        trajectory=trajectory,
        pathogenic_high_selection=source.pathogenic_high_selection,
        truncation=source.truncation,
        metadata={
            "selection": {
                "source_dataset": source.name,
                "rule": "first nonsynonymous rows in canonical source order",
                "variant_count": variant_count,
                "wildtype_reference_row_added": True,
            },
            "source_file": source.metadata.get("source_file"),
            "source_file_sha256": source.metadata.get("source_file_sha256"),
        },
    )
    subset.save(destination)
    return subset


def _load_or_create_artifact(
    path: Path,
    *,
    dataset: Dataset,
    force: bool,
    create,
) -> FeatureArtifact:
    if path.is_file() and not force:
        artifact = FeatureArtifact.load(path)
        if artifact.dataset != dataset.name:
            raise ValueError(
                f"Cached artifact {path} belongs to {artifact.dataset}, not "
                f"{dataset.name}; rerun with --force."
            )
        _progress(f"Reusing {path.relative_to(path.parents[2])}")
        return artifact
    artifact = create()
    artifact.save(path)
    return artifact


def _release_accelerator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()


def _write_analysis_config(
    output_root: Path,
    dataset: Dataset,
    embedding_path: Path,
    sae_path: Path,
    llr_path: Path,
    *,
    pooling: str,
    sae_mode: str,
) -> Path:
    config_path = output_root / "analysis_config.json"
    config = {
        "output_dir": "results",
        "gammas": [0.01, 0.1, 1.0, 10.0],
        "alphas": [0.0, 0.25, 0.5, 1.0, 2.0],
        "review_star_cutoffs": [0, 1, 2, 3],
        "datasets": [
            {
                "path": "dataset",
                "features": {
                    f"ESM-C 300M layer 30 {pooling} pool": str(
                        embedding_path.relative_to(output_root)
                    ),
                    f"ESM-C 300M layer 30 {sae_mode} SAE": str(
                        sae_path.relative_to(output_root)
                    ),
                },
                "priors": {
                    "ESM-C 300M LLR": str(llr_path.relative_to(output_root))
                },
            }
        ],
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


def _write_run_summary(
    output_root: Path,
    dataset: Dataset,
    embedding: FeatureArtifact,
    sae: FeatureArtifact,
    llr: FeatureArtifact,
    results: dict[str, pd.DataFrame],
) -> Path:
    annotations = (
        dataset.variants.loc[~dataset.variants["is_synonymous"], "annotation"]
        .replace("", "unlabeled")
        .value_counts()
        .to_dict()
    )
    payload = {
        "dataset": dataset.name,
        "assayed_variants": int((~dataset.variants["is_synonymous"]).sum()),
        "reference_length": len(dataset.reference_sequence),
        "replicates": int(dataset.trajectory["Replicate"].nunique()),
        "trajectory_rows": len(dataset.trajectory),
        "annotations": annotations,
        "embedding_shape": list(embedding.values.shape),
        "sae_shape": list(sae.values.shape),
        "llr_shape": list(llr.values.shape),
        "artifacts": {
            "embedding": embedding.provenance,
            "sae": sae.provenance,
            "llr": llr.provenance,
        },
        "best_prior_configurations": json.loads(
            results["summary"].to_json(orient="records")
        ),
    }
    path = output_root / "run_summary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "demo" / "brca1_100",
        help="Demo dataset, artifact, and result directory.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create and validate the 100-variant dataset without loading ESM-C.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute dataset and all artifacts."
    )
    parser.add_argument("--device", help="Torch device, for example cuda, mps, or cpu.")
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], help="ESM-C weight dtype."
    )
    parser.add_argument("--pooling", choices=["max", "mean"], default="max")
    parser.add_argument(
        "--sae-mode", choices=["normal", "topk", "batchtopk"], default="batchtopk"
    )
    parser.add_argument("--sae-features", type=int, default=256)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.resolve()
    artifact_dir = output_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    source = _source_dataset()
    dataset = build_subset_dataset(
        source,
        output_root / "dataset",
        force=args.force,
    )
    _progress(
        f"Prepared {dataset.name}: 100 variants, "
        f"{len(dataset.trajectory)} trajectory rows"
    )
    if args.prepare_only:
        _progress(f"Dataset ready at {output_root / 'dataset'}")
        return 0

    embedding_path = (
        artifact_dir / f"esmc_300m_layer30_{args.pooling}_embedding.npz"
    )
    _progress("Generating final-layer ESM-C 300M embeddings")
    embedding = _load_or_create_artifact(
        embedding_path,
        dataset=dataset,
        force=args.force,
        create=lambda: embed(
            dataset,
            MODEL_NAME,
            layers=[FINAL_LAYER],
            pooling=args.pooling,
            device=args.device,
            dtype=args.dtype,
        )[FINAL_LAYER],
    )
    _release_accelerator_memory()

    sae_path = (
        artifact_dir
        / f"esmc_300m_layer30_{args.pooling}_{args.sae_mode}_sae.npz"
    )
    sae_model_path = sae_path.with_suffix(".pt")
    _progress(f"Training one {args.sae_mode} sparse autoencoder")
    sae_config = SAEConfig(
        n_features=args.sae_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mode=args.sae_mode,
        k=args.k if args.sae_mode != "normal" else None,
        seed=args.seed,
    )
    sae = _load_or_create_artifact(
        sae_path,
        dataset=dataset,
        force=args.force,
        create=lambda: train_sae(
            dataset,
            embedding,
            config=sae_config,
            model_path=sae_model_path,
            device=args.device,
        ),
    )
    _release_accelerator_memory()

    llr_path = artifact_dir / "esmc_300m_llr_prior.npz"
    _progress("Computing ESM-C 300M masked-marginal LLR priors")
    llr = _load_or_create_artifact(
        llr_path,
        dataset=dataset,
        force=args.force,
        create=lambda: masked_marginal_llr(
            dataset,
            MODEL_NAME,
            device=args.device,
            dtype=args.dtype,
        ),
    )
    _release_accelerator_memory()

    config_path = _write_analysis_config(
        output_root,
        dataset,
        embedding_path,
        sae_path,
        llr_path,
        pooling=args.pooling,
        sae_mode=args.sae_mode,
    )
    _progress("Running popDMS inference, prior sweeps, consistency, and AUC analysis")
    results = run_analysis_file(config_path)
    summary_path = _write_run_summary(
        output_root, dataset, embedding, sae, llr, results
    )

    print("\nBaselines")
    print(results["baselines"].to_string(index=False))
    print("\nBest LLR-prior configuration")
    print(results["summary"].to_string(index=False))
    _progress(f"Complete. Machine-readable summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
