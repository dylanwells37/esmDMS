#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "clinprotgym_esmc_sae"
DEFAULT_RUN_LABEL = "DeltaEmbSAE_max_pool_batchtopk_k64_nf12800_seed42"


@dataclass(frozen=True)
class Candidate:
    path: str
    kind: str
    size_bytes: int
    dataset: str
    model: str
    layer: str
    reason: str


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def layer_token_present(name: str, layer: str) -> bool:
    return re.search(rf"(?:^|_)Layer_{re.escape(str(layer))}(?:_|\.|$)", name) is not None


def artifact_kind(path: Path) -> str | None:
    name = path.name
    if name.endswith("_inference_results.pkl"):
        return "inference_results"
    if name.endswith("_seq_to_features.pkl"):
        return "seq_to_features"
    if name.endswith("_model.pt"):
        return "model_pt"
    if name.endswith("_viz_data.pkl"):
        return "viz_data"
    if name.endswith("_viz.png"):
        return "viz_png"
    return None


def run_context(run_dir: Path) -> tuple[str, str, str]:
    parts = run_dir.parts
    try:
        dataset = parts[parts.index("datasets") + 1]
        fixed_idx = parts.index("fixed_sae_model_layer_array")
        model = parts[fixed_idx + 1]
        layer = parts[fixed_idx + 2].removeprefix("Layer_")
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not parse SAE run directory context: {run_dir}") from exc
    return dataset, model, layer


def expected_artifact(path: Path, *, dataset: str, model: str, layer: str, kind: str) -> bool:
    name = path.name
    has_dataset = dataset in name
    has_layer = layer_token_present(name, layer)
    if kind in {"inference_results", "seq_to_features"}:
        return has_dataset and model in name and has_layer
    if kind in {"model_pt", "viz_data", "viz_png"}:
        # SAE model/viz filenames do not include the embedding model id, only
        # dataset and layer, so model cannot be used to validate these names.
        return has_dataset and has_layer
    return True


def iter_run_dirs(output_root: Path, run_label: str | None) -> list[Path]:
    pattern = "datasets/*/jobs/fixed_sae_model_layer_array/*/Layer_*/runs/*"
    run_dirs = [path for path in output_root.glob(pattern) if path.is_dir()]
    if run_label:
        run_dirs = [path for path in run_dirs if path.name == run_label]
    return sorted(run_dirs)


def find_stray_artifacts(
    output_root: Path,
    *,
    run_label: str | None,
    kinds: set[str],
    min_age_minutes: float,
) -> tuple[list[Candidate], dict[str, int]]:
    now = time.time()
    stats = {
        "run_dirs": 0,
        "recognized_files": 0,
        "expected_files": 0,
        "expected_bytes": 0,
        "stray_files": 0,
        "stray_bytes": 0,
        "too_new_files": 0,
        "too_new_bytes": 0,
    }
    candidates: list[Candidate] = []
    for run_dir in iter_run_dirs(output_root, run_label):
        stats["run_dirs"] += 1
        dataset, model, layer = run_context(run_dir)
        for path in run_dir.rglob("*"):
            if not path.is_file():
                continue
            kind = artifact_kind(path)
            if kind is None or kind not in kinds:
                continue
            size = path.stat().st_size
            stats["recognized_files"] += 1
            if expected_artifact(path, dataset=dataset, model=model, layer=layer, kind=kind):
                stats["expected_files"] += 1
                stats["expected_bytes"] += size
                continue
            age_minutes = (now - path.stat().st_mtime) / 60.0
            if age_minutes < min_age_minutes:
                stats["too_new_files"] += 1
                stats["too_new_bytes"] += size
                continue
            stats["stray_files"] += 1
            stats["stray_bytes"] += size
            candidates.append(
                Candidate(
                    path=str(path),
                    kind=kind,
                    size_bytes=size,
                    dataset=dataset,
                    model=model,
                    layer=layer,
                    reason="artifact filename does not match containing dataset/model/layer",
                )
            )
    return candidates, stats


def remove_empty_dirs(paths: list[Path], stop_at: Path) -> int:
    removed = 0
    for path in sorted({p.parent for p in paths}, key=lambda p: len(p.parts), reverse=True):
        current = path
        while current != stop_at and stop_at in current.parents:
            try:
                current.rmdir()
            except OSError:
                break
            removed += 1
            current = current.parent
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find and optionally delete stale DeltaEmbSAE artifacts copied into "
            "the wrong ClinProtGym SAE run directories. Dry-run is the default."
        )
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--run-label",
        default=DEFAULT_RUN_LABEL,
        help="Limit cleanup to this run label. Use --run-label '' to scan all run labels.",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["inference_results", "seq_to_features", "model_pt", "viz_data", "viz_png"],
        choices=["inference_results", "seq_to_features", "model_pt", "viz_data", "viz_png"],
        help="Artifact kinds eligible for cleanup.",
    )
    parser.add_argument(
        "--min-age-minutes",
        type=float,
        default=None,
        help=(
            "Skip stray files newer than this. Defaults to 0 for dry-run and "
            "30 for --delete to avoid racing active writers."
        ),
    )
    parser.add_argument("--delete", action="store_true", help="Actually delete the reported stray artifacts.")
    parser.add_argument("--prune-empty-dirs", action="store_true", help="Remove empty directories left after deletion.")
    parser.add_argument("--json-out", help="Optional path to write a JSON report.")
    parser.add_argument("--limit", type=int, default=25, help="Number of candidate examples to print.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if not output_root.is_dir():
        print(f"Output root does not exist: {output_root}", file=sys.stderr)
        return 2
    run_label = args.run_label or None
    min_age_minutes = args.min_age_minutes
    if min_age_minutes is None:
        min_age_minutes = 30.0 if args.delete else 0.0

    candidates, stats = find_stray_artifacts(
        output_root,
        run_label=run_label,
        kinds=set(args.kinds),
        min_age_minutes=min_age_minutes,
    )

    print(f"Output root: {output_root}")
    print(f"Run label: {run_label or '<all>'}")
    print(f"Mode: {'DELETE' if args.delete else 'dry-run'}")
    print(f"Run dirs scanned: {stats['run_dirs']}")
    print(f"Expected artifacts: {stats['expected_files']} ({human_size(stats['expected_bytes'])})")
    print(f"Stray artifacts eligible: {stats['stray_files']} ({human_size(stats['stray_bytes'])})")
    if stats["too_new_files"]:
        print(
            "Stray artifacts skipped as too new: "
            f"{stats['too_new_files']} ({human_size(stats['too_new_bytes'])})"
        )

    if candidates:
        print("\nExamples:")
        for candidate in sorted(candidates, key=lambda row: row.size_bytes, reverse=True)[: args.limit]:
            print(
                f"  {human_size(candidate.size_bytes):>10}  {candidate.kind:<18}  "
                f"{candidate.path}"
            )

    deleted_bytes = 0
    deleted_files = 0
    if args.delete:
        for candidate in candidates:
            path = Path(candidate.path)
            try:
                size = path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue
            deleted_files += 1
            deleted_bytes += size
        print(f"\nDeleted: {deleted_files} files ({human_size(deleted_bytes)})")
        if args.prune_empty_dirs:
            removed_dirs = remove_empty_dirs([Path(row.path) for row in candidates], output_root)
            print(f"Pruned empty directories: {removed_dirs}")
    else:
        print("\nDry-run only. Re-run with --delete to remove the eligible stray artifacts.")

    if args.json_out:
        report = {
            "output_root": str(output_root),
            "run_label": run_label,
            "delete": bool(args.delete),
            "min_age_minutes": min_age_minutes,
            "stats": stats,
            "candidates": [asdict(row) for row in candidates],
        }
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote JSON report: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
