#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "clinprotgym_esmc_sae"
DEFAULT_PAYLOAD = (
    DEFAULT_OUTPUT_ROOT
    / "jobs"
    / "fixed_deltaembsae_layer_array"
    / "clinprotgym_fixed_deltaembsae_layer_payload.pkl"
)


@dataclass(frozen=True)
class PruneCandidate:
    task_idx: int
    dataset: str
    model: str
    layer: int
    reason: str
    path: str
    size_bytes: int


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def model_cache_label(model_name: str) -> str:
    safe = model_name.replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "_", safe)
    return safe.strip("_")


def directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def result_status(result_path: Path) -> tuple[bool, str]:
    if not result_path.exists():
        return False, "no_result_stopped_or_partial"
    try:
        status = json.loads(result_path.read_text()).get("status")
    except Exception:
        return False, "unreadable_result"
    if status == "ok":
        return True, "ok"
    return False, f"result_status_{status or 'missing'}"


def load_tasks(payload_path: Path) -> list[dict]:
    with payload_path.open("rb") as handle:
        payload = pickle.load(handle)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Payload does not contain a task list: {payload_path}")
    return tasks


def find_candidates(output_root: Path, payload_path: Path) -> tuple[list[PruneCandidate], dict[str, int]]:
    tasks = load_tasks(payload_path)
    stats = {
        "total_tasks": len(tasks),
        "ok_tasks": 0,
        "non_success_tasks": 0,
        "existing_non_success_dirs": 0,
        "missing_non_success_dirs": 0,
        "candidate_bytes": 0,
    }
    candidates: list[PruneCandidate] = []

    for task_idx, task in enumerate(tasks):
        dataset = str(task["dataset"])
        model = str(task["model"])
        layer = int(task["layer"])
        layer_dir = (
            output_root
            / "datasets"
            / dataset
            / "jobs"
            / "fixed_sae_model_layer_array"
            / model_cache_label(model)
            / f"Layer_{layer}"
        )
        ok, reason = result_status(layer_dir / "task_result.json")
        if ok:
            stats["ok_tasks"] += 1
            continue

        stats["non_success_tasks"] += 1
        if not layer_dir.exists():
            stats["missing_non_success_dirs"] += 1
            continue

        size = directory_size(layer_dir)
        stats["existing_non_success_dirs"] += 1
        stats["candidate_bytes"] += size
        candidates.append(
            PruneCandidate(
                task_idx=task_idx,
                dataset=dataset,
                model=model,
                layer=layer,
                reason=reason,
                path=str(layer_dir),
                size_bytes=size,
            )
        )

    return candidates, stats


def remove_empty_parents(start: Path, stop_at: Path) -> int:
    removed = 0
    current = start
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
            "Prune ClinProtGym SAE layer directories whose task_result.json is "
            "missing or not status=ok. Dry-run is the default."
        )
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--payload", default=str(DEFAULT_PAYLOAD))
    parser.add_argument("--delete", action="store_true", help="Actually delete non-successful layer directories.")
    parser.add_argument(
        "--prune-empty-dirs",
        action="store_true",
        help="After deletion, remove empty model/job parent directories.",
    )
    parser.add_argument("--json-out", help="Optional path to write a JSON report.")
    parser.add_argument("--limit", type=int, default=40, help="Number of candidate examples to print.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    payload_path = Path(args.payload).resolve()
    if not output_root.is_dir():
        print(f"Output root does not exist: {output_root}", file=sys.stderr)
        return 2
    if not payload_path.is_file():
        print(f"Payload does not exist: {payload_path}", file=sys.stderr)
        return 2

    candidates, stats = find_candidates(output_root, payload_path)
    print(f"Output root: {output_root}")
    print(f"Payload: {payload_path}")
    print(f"Mode: {'DELETE' if args.delete else 'dry-run'}")
    print(f"Total SAE tasks: {stats['total_tasks']}")
    print(f"Successful tasks kept: {stats['ok_tasks']}")
    print(f"Non-success tasks: {stats['non_success_tasks']}")
    print(
        "Non-success layer dirs eligible for removal: "
        f"{stats['existing_non_success_dirs']} ({human_size(stats['candidate_bytes'])})"
    )
    if stats["missing_non_success_dirs"]:
        print(f"Non-success tasks with no layer dir present: {stats['missing_non_success_dirs']}")

    if candidates:
        print("\nExamples:")
        for candidate in candidates[: args.limit]:
            print(
                f"{candidate.task_idx:04d}\t{candidate.reason}\t"
                f"{human_size(candidate.size_bytes)}\t{candidate.path}"
            )
        if len(candidates) > args.limit:
            print(f"... {len(candidates) - args.limit} more")

    deleted = 0
    removed_empty_dirs = 0
    if args.delete:
        for candidate in candidates:
            path = Path(candidate.path)
            shutil.rmtree(path)
            deleted += 1
            if args.prune_empty_dirs:
                removed_empty_dirs += remove_empty_parents(path.parent, output_root)
        print(f"\nDeleted layer dirs: {deleted}")
        if args.prune_empty_dirs:
            print(f"Removed empty parent dirs: {removed_empty_dirs}")
    else:
        print("\nDry run only. Re-run with --delete to remove these directories.")

    if args.json_out:
        report = {
            "output_root": str(output_root),
            "payload": str(payload_path),
            "delete": bool(args.delete),
            "stats": stats,
            "deleted": deleted,
            "removed_empty_dirs": removed_empty_dirs,
            "candidates": [asdict(candidate) for candidate in candidates],
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"Wrote JSON report: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
