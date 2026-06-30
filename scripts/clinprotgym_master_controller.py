#!/usr/bin/env python3
"""Stateful controller for the ClinProtGym ESM-C SAE pipeline.

This script is intentionally a thin orchestrator around the existing pipeline
commands and submission helpers. It keeps a JSON state file, submits one stage
at a time, and exits. When requested, it submits the Slurm wrapper to check
again later instead of sleeping inside an allocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "clinprotgym_esmc_sae"
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "clinprotgym_esmc_sae_pipeline.py"
EMBEDDING_ARRAY_SUBMITTER = REPO_ROOT / "job_scripts" / "submit_clinprotgym_embedding_arrays.sh"
MERGE_SUBMITTER = REPO_ROOT / "job_scripts" / "submit_clinprotgym_embedding_merges.sh"
DEFAULT_CONTROLLER_SCRIPT = REPO_ROOT / "job_scripts" / "run_clinprotgym_master_controller.sh"
DEFAULT_MODELS = ("biohub/ESMC-300M", "biohub/ESMC-600M")
MODEL_LAYER_COUNTS = {"biohub/ESMC-300M": 30, "biohub/ESMC-600M": 36}
CHUNK_RE = re.compile(r"_(mean_pool|max_pool|per_residue)_embeddings_chunk_(\d+)\.pkl$")


@dataclass(frozen=True)
class EmbeddingPolicy:
    n_chunks: int
    mem: str
    max_active: int


def model_short_name(model_name: str) -> str:
    return str(model_name).split("/")[-1].replace("_", "-")


def model_cache_label(model_name: str) -> str:
    return str(model_name).replace("/", "__")


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return dict(default)
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def append_event(state: dict[str, Any], message: str, **extra: Any) -> None:
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "message": message}
    event.update(extra)
    state.setdefault("history", []).append(event)
    state["updated_at"] = event["time"]


def run_command(
    command: list[str],
    *,
    dry_run: bool = False,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(command)
    print(f"+ {printable}")
    if dry_run:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=check,
        capture_output=capture,
        text=True,
    )


def parse_job_id(output: str) -> str:
    text = output.strip()
    if not text:
        return ""
    if ";" in text and text.split(";", 1)[0].isdigit():
        return text.split(";", 1)[0]
    matches = re.findall(r"\b\d+\b", text)
    return matches[-1] if matches else ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_policies(values: list[str] | None) -> list[EmbeddingPolicy]:
    if not values:
        values = ["120:128G:2", "180:192G:1", "240:256G:1"]
    policies: list[EmbeddingPolicy] = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError("--embedding-policy must look like N_CHUNKS:MEM:MAX_ACTIVE")
        policies.append(EmbeddingPolicy(int(parts[0]), parts[1], int(parts[2])))
    return policies


def available_datasets(output_root: Path) -> list[str]:
    dataset_root = output_root / "datasets"
    if not dataset_root.is_dir():
        return []
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


def refresh_cache_status(args: argparse.Namespace, *, dry_run: bool) -> None:
    command = [
        args.python_executable,
        str(PIPELINE_SCRIPT),
        "cache-status",
        "--output-root",
        str(args.output_root),
        "--models",
        *args.models,
    ]
    run_command(command, dry_run=dry_run)


def cache_status_rows(output_root: Path) -> list[dict[str, str]]:
    return read_csv_rows(output_root / "tables" / "clinprotgym_embedding_cache_status.csv")


def embedding_pair_status(
    output_root: Path,
    models: list[str],
) -> tuple[dict[tuple[str, str], tuple[int, int]], list[dict[str, Any]]]:
    wanted_models = set(models)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in cache_status_rows(output_root):
        if row.get("model") not in wanted_models:
            continue
        grouped.setdefault((row["dataset"], row["model"]), []).append(row)

    status: dict[tuple[str, str], tuple[int, int]] = {}
    incomplete: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        exists = sum(str(row.get("exists", "")).lower() == "true" for row in rows)
        total = len(rows)
        status[key] = (exists, total)
        if exists != total:
            dataset, model = key
            incomplete.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "model_short": model_short_name(model),
                    "exists": exists,
                    "total": total,
                }
            )
    return status, sorted(incomplete, key=lambda row: (row["dataset"], row["model"]))


def complete_datasets_for_models(output_root: Path, models: list[str]) -> list[str]:
    status, _ = embedding_pair_status(output_root, models)
    complete: list[str] = []
    for dataset in available_datasets(output_root):
        ok = True
        for model in models:
            exists, total = status.get((dataset, model), (0, 0))
            if total == 0 or exists != total:
                ok = False
                break
        if ok:
            complete.append(dataset)
    return complete


def count_ready_datasets(output_root: Path, datasets: list[str]) -> list[str]:
    manifest = output_root / "tables" / "clinprotgym_processing_manifest.csv"
    rows = read_csv_rows(manifest)
    selected = set(datasets)
    ready: list[str] = []
    for row in rows:
        dataset = row.get("dataset", "")
        if dataset not in selected:
            continue
        try:
            n_traj = int(float(row.get("trajectory_columns", "0") or 0))
        except ValueError:
            n_traj = 0
        if n_traj > 0:
            ready.append(dataset)
    return ready


def embedding_jobs_table(output_root: Path) -> list[dict[str, str]]:
    return read_csv_rows(output_root / "tables" / "clinprotgym_embedding_jobs.csv")


def payload_n_chunks(payload_path: Path) -> int | None:
    if not payload_path.is_file():
        return None
    with payload_path.open("rb") as handle:
        payload = pickle.load(handle)
    return int(payload.get("n_chunks"))


def archive_batch_chunks(batch_dir: Path, archive_name: str, *, dry_run: bool) -> int:
    chunk_files = sorted(batch_dir.glob("*_embeddings_chunk_*.pkl"))
    if not chunk_files:
        return 0
    archive_dir = batch_dir / archive_name
    print(f"Archiving {len(chunk_files)} chunk files from {batch_dir} to {archive_dir}")
    if dry_run:
        return len(chunk_files)
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in chunk_files:
        shutil.move(str(path), archive_dir / path.name)
    return len(chunk_files)


def archive_incomplete_embedding_chunks(
    args: argparse.Namespace,
    incomplete_pairs: list[dict[str, Any]],
    *,
    policy: EmbeddingPolicy,
    reason: str,
    only_if_payload_differs: bool,
    dry_run: bool,
) -> int:
    archived = 0
    archive_name = f"old_outputs_controller_{reason}_{now_stamp()}_n{policy.n_chunks}"
    pair_set = {(row["dataset"], row["model"]) for row in incomplete_pairs}
    for dataset, model in sorted(pair_set):
        batch_dir = (
            args.output_root
            / "datasets"
            / dataset
            / "jobs"
            / "embedding_batches"
            / model_cache_label(model)
        )
        if only_if_payload_differs:
            payloads = sorted(batch_dir.glob("*_embedding_batch_payload.pkl"))
            old_n = payload_n_chunks(payloads[0]) if payloads else None
            if old_n == policy.n_chunks:
                continue
        archived += archive_batch_chunks(batch_dir, archive_name, dry_run=dry_run)
    return archived


def create_embedding_jobs(
    args: argparse.Namespace,
    policy: EmbeddingPolicy,
    *,
    dry_run: bool,
) -> None:
    command = [
        args.python_executable,
        str(PIPELINE_SCRIPT),
        "create-embedding-jobs",
        "--output-root",
        str(args.output_root),
        "--models",
        *args.models,
        "--n-chunks",
        str(policy.n_chunks),
        "--max-active-embedding-tasks",
        str(policy.max_active),
        "--embedding-partition",
        args.embedding_partition,
        "--embedding-cpus",
        str(args.embedding_cpus),
        "--embedding-mem",
        policy.mem,
        "--embedding-time",
        args.embedding_time,
        "--merge-partition",
        args.merge_partition,
        "--merge-mem",
        args.merge_mem,
        "--merge-time",
        args.merge_time,
        "--skip-complete",
    ]
    if args.hf_home:
        command.extend(["--hf-home", str(args.hf_home)])
    run_command(command, dry_run=dry_run)


def embedding_chunk_status(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in embedding_jobs_table(output_root):
        if row.get("action") == "skip_complete" and row.get("cache_complete") == "True":
            continue
        payload_path = Path(row.get("payload_path", ""))
        if not payload_path.is_file():
            rows.append({**row, "n_chunks": 0, "complete_chunks": 0, "missing_chunks": []})
            continue
        with payload_path.open("rb") as handle:
            payload = pickle.load(handle)
        n_chunks = int(payload["n_chunks"])
        batch_dir = Path(payload["batch_dir"])
        present = {"mean_pool": set(), "max_pool": set(), "per_residue": set()}
        for path in batch_dir.glob("*_embeddings_chunk_*.pkl"):
            match = CHUNK_RE.search(path.name)
            if match:
                present[match.group(1)].add(int(match.group(2)))
        complete = set.intersection(*present.values())
        missing = [idx for idx in range(n_chunks) if idx not in complete]
        rows.append(
            {
                **row,
                "n_chunks": n_chunks,
                "complete_chunks": len(complete),
                "missing_chunks": missing,
            }
        )
    return rows


def submit_embedding_arrays(
    args: argparse.Namespace,
    policy: EmbeddingPolicy,
    *,
    dry_run: bool,
) -> list[str]:
    command = [
        "bash",
        str(EMBEDDING_ARRAY_SUBMITTER),
        "--output-root",
        str(args.output_root),
        "--partition",
        args.embedding_partition,
        "--mem",
        policy.mem,
        "--cpus",
        str(args.embedding_cpus),
        "--time",
        args.embedding_time,
        "--max-active",
        str(policy.max_active),
    ]
    if dry_run:
        command.append("--dry-run")
    result = run_command(command, dry_run=False)
    print(result.stdout, end="")
    manifest = args.output_root / "tables" / "clinprotgym_embedding_array_submissions.csv"
    job_ids = [
        row.get("job_id", "")
        for row in read_csv_rows(manifest)
        if row.get("submitted") == "True" and row.get("job_id")
    ]
    return job_ids


def submit_merges(args: argparse.Namespace, *, dry_run: bool) -> list[str]:
    command = [
        "bash",
        str(MERGE_SUBMITTER),
        "--output-root",
        str(args.output_root),
    ]
    if dry_run:
        command.append("--dry-run")
    result = run_command(command, dry_run=False)
    print(result.stdout, end="")
    manifest = args.output_root / "tables" / "clinprotgym_embedding_merge_submissions.csv"
    return [
        row.get("job_id", "")
        for row in read_csv_rows(manifest)
        if row.get("submitted") == "True" and row.get("job_id")
    ]


def job_ids_from_state(state: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for values in state.get("job_ids", {}).values():
        ids.extend(str(value) for value in values if value)
    return sorted(set(ids))


def running_job_ids(job_ids: list[str], *, dry_run: bool) -> list[str]:
    if dry_run or not job_ids:
        return []
    result = run_command(["squeue", "-h", "-j", ",".join(job_ids), "-o", "%A"], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"squeue failed while checking job ids {job_ids}: {result.stderr.strip()}")
    running = sorted(set(line.strip().split("_", 1)[0] for line in result.stdout.splitlines() if line.strip()))
    return running


def schedule_self(args: argparse.Namespace, argv: list[str], state: dict[str, Any], reason: str) -> None:
    append_event(state, "scheduled next controller check", reason=reason)
    write_json(args.state_path, state)
    if not args.resubmit_self:
        print(f"Next check not submitted (--no-resubmit-self). Reason: {reason}")
        return
    command = [
        "sbatch",
        f"--begin=now+{args.poll_minutes}minutes",
        str(args.controller_script),
        *argv,
    ]
    result = run_command(command, dry_run=args.dry_run)
    if result.stdout:
        print(result.stdout.strip())


def submit_pipeline_array(command: list[str], *, dry_run: bool) -> str:
    result = run_command(command, dry_run=dry_run)
    if result.stdout:
        print(result.stdout, end="")
    return parse_job_id(result.stdout)


def create_sae_jobs(args: argparse.Namespace, datasets: list[str], *, dry_run: bool) -> str:
    command = [
        args.python_executable,
        str(PIPELINE_SCRIPT),
        "create-sae-jobs",
        "--output-root",
        str(args.output_root),
        "--datasets",
        *datasets,
        "--models",
        *args.models,
        "--sae-partition",
        args.sae_partition,
        "--sae-cpus",
        str(args.sae_cpus),
        "--sae-mem",
        args.sae_mem,
        "--sae-time",
        args.sae_time,
        "--max-active-sae-tasks",
        str(args.max_active_sae_tasks),
        "--submit",
    ]
    return submit_pipeline_array(command, dry_run=dry_run)


def sae_tasks_ok(output_root: Path) -> tuple[bool, int, int]:
    tasks = read_csv_rows(output_root / "tables" / "clinprotgym_fixed_deltaembsae_layer_tasks.csv")
    if not tasks:
        return False, 0, 0
    ok = 0
    for task in tasks:
        path = (
            output_root
            / "datasets"
            / task["dataset"]
            / "jobs"
            / "fixed_sae_model_layer_array"
            / model_cache_label(task["model"])
            / f"Layer_{int(float(task['layer']))}"
            / "sae_sweep_results.csv"
        )
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty and str(df.get("status", pd.Series([""])).iloc[0]) == "ok":
            ok += 1
    return ok == len(tasks), ok, len(tasks)


def collect_sae(args: argparse.Namespace, datasets: list[str], *, dry_run: bool) -> None:
    command = [
        args.python_executable,
        str(PIPELINE_SCRIPT),
        "collect-sae",
        "--output-root",
        str(args.output_root),
        "--datasets",
        *datasets,
    ]
    run_command(command, dry_run=dry_run)


def create_benchmark_jobs(args: argparse.Namespace, datasets: list[str], *, dry_run: bool) -> str:
    command = [
        args.python_executable,
        str(PIPELINE_SCRIPT),
        "create-benchmark-jobs",
        "--output-root",
        str(args.output_root),
        "--datasets",
        *datasets,
        "--models",
        *args.models,
        "--count-datasets-only",
        "--benchmark-partition",
        args.benchmark_partition,
        "--benchmark-cpus",
        str(args.benchmark_cpus),
        "--benchmark-mem",
        args.benchmark_mem,
        "--benchmark-time",
        args.benchmark_time,
        "--max-active-benchmark-tasks",
        str(args.max_active_benchmark_tasks),
        "--submit",
    ]
    return submit_pipeline_array(command, dry_run=dry_run)


def benchmark_tasks_ok(output_root: Path) -> tuple[bool, int, int]:
    tasks = read_csv_rows(output_root / "tables" / "clinprotgym_benchmark_tasks.csv")
    if not tasks:
        return False, 0, 0
    have = sum(1 for task in tasks if Path(task.get("result_path", "")).is_file())
    return have == len(tasks), have, len(tasks)


def collect_benchmarks(args: argparse.Namespace, datasets: list[str], *, dry_run: bool) -> None:
    command = [
        args.python_executable,
        str(PIPELINE_SCRIPT),
        "collect-benchmarks",
        "--output-root",
        str(args.output_root),
        "--datasets",
        *datasets,
        "--count-datasets-only",
    ]
    run_command(command, dry_run=dry_run)


def ensemble_candidates_exist(output_root: Path, datasets: list[str]) -> bool:
    metrics_path = output_root / "tables" / "clinprotgym_method_metrics.csv"
    if not metrics_path.is_file():
        return False
    df = pd.read_csv(metrics_path)
    if df.empty or "dataset" not in df.columns or "benchmark" not in df.columns:
        return False
    sub = df[
        df["dataset"].astype(str).isin(set(datasets))
        & df["benchmark"].astype(str).eq("Fixed DeltaEmbSAE")
    ].copy()
    if sub.empty or "spearman_rho" not in sub.columns:
        return False
    return pd.to_numeric(sub["spearman_rho"], errors="coerce").notna().any()


def create_ensemble_jobs(args: argparse.Namespace, datasets: list[str], *, dry_run: bool) -> str:
    command = [
        args.python_executable,
        str(PIPELINE_SCRIPT),
        "create-ensemble-jobs",
        "--output-root",
        str(args.output_root),
        "--datasets",
        *datasets,
        "--count-datasets-only",
        "--ensemble-top-n",
        str(args.ensemble_top_n),
        "--ensemble-gamma",
        str(args.ensemble_gamma),
        "--ensemble-partition",
        args.ensemble_partition,
        "--ensemble-cpus",
        str(args.ensemble_cpus),
        "--ensemble-mem",
        args.ensemble_mem,
        "--ensemble-time",
        args.ensemble_time,
        "--max-active-ensemble-tasks",
        str(args.max_active_ensemble_tasks),
        "--submit",
    ]
    return submit_pipeline_array(command, dry_run=dry_run)


def ensemble_tasks_ok(output_root: Path) -> tuple[bool, int, int]:
    tasks = read_csv_rows(output_root / "tables" / "clinprotgym_sae_ensemble_tasks.csv")
    if not tasks:
        return False, 0, 0
    have = 0
    for task in tasks:
        dataset = task["dataset"]
        out_dir = output_root / "datasets" / dataset / "tables" / "sae_ensemble_gamma1"
        if list(out_dir.glob("*_metrics.csv")):
            have += 1
    return have == len(tasks), have, len(tasks)


def collect_ensembles(args: argparse.Namespace, datasets: list[str], *, dry_run: bool) -> None:
    command = [
        args.python_executable,
        str(PIPELINE_SCRIPT),
        "collect-ensembles",
        "--output-root",
        str(args.output_root),
        "--datasets",
        *datasets,
        "--count-datasets-only",
    ]
    run_command(command, dry_run=dry_run)


def summarize(args: argparse.Namespace, datasets: list[str], *, dry_run: bool) -> None:
    for subcommand in ("summarize", "cross-replicate-consistency"):
        command = [
            args.python_executable,
            str(PIPELINE_SCRIPT),
            subcommand,
            "--output-root",
            str(args.output_root),
            "--datasets",
            *datasets,
            "--count-datasets-only",
            "--sahu-brca2-final",
            args.sahu_brca2_final,
        ]
        if subcommand == "cross-replicate-consistency":
            command.append("--omit-empty-datasets")
        run_command(command, dry_run=dry_run)


def handle_embedding(
    args: argparse.Namespace,
    state: dict[str, Any],
    policies: list[EmbeddingPolicy],
    argv: list[str],
) -> str:
    refresh_cache_status(args, dry_run=args.dry_run)
    _, incomplete_pairs = embedding_pair_status(args.output_root, args.models)
    if not incomplete_pairs:
        state["stage"] = "sae"
        state["job_ids"] = {}
        append_event(state, "embedding caches complete")
        write_json(args.state_path, state)
        return "continue"

    policy_index = int(state.get("embedding_policy_index", 0))
    if policy_index >= len(policies):
        if args.allow_partial_downstream:
            state["stage"] = "sae"
            append_event(state, "embedding incomplete; proceeding with complete datasets")
            write_json(args.state_path, state)
            return "continue"
        state["stage"] = "blocked"
        append_event(state, "embedding policies exhausted", incomplete_pairs=incomplete_pairs)
        write_json(args.state_path, state)
        print("Embedding policies exhausted; controller is blocked.")
        return "stop"

    policy = policies[policy_index]
    if state.get("embedding_jobs_n_chunks") != policy.n_chunks:
        archive_incomplete_embedding_chunks(
            args,
            incomplete_pairs,
            policy=policy,
            reason="stale",
            only_if_payload_differs=True,
            dry_run=args.dry_run,
        )
        create_embedding_jobs(args, policy, dry_run=args.dry_run)
        state["embedding_jobs_n_chunks"] = policy.n_chunks
        state["embedding_submitted"] = False
        state["merge_submitted"] = False
        state["job_ids"] = {}
        append_event(state, "created embedding jobs", n_chunks=policy.n_chunks, mem=policy.mem)
        write_json(args.state_path, state)

    chunk_rows = embedding_chunk_status(args.output_root)
    missing_rows = [row for row in chunk_rows if row["missing_chunks"]]
    if missing_rows:
        if state.get("embedding_submitted"):
            if policy_index + 1 < len(policies):
                archive_incomplete_embedding_chunks(
                    args,
                    incomplete_pairs,
                    policy=policy,
                    reason="failed",
                    only_if_payload_differs=False,
                    dry_run=args.dry_run,
                )
                state["embedding_policy_index"] = policy_index + 1
                state["embedding_jobs_n_chunks"] = None
                state["embedding_submitted"] = False
                state["merge_submitted"] = False
                state["job_ids"] = {}
                append_event(
                    state,
                    "escalating embedding policy",
                    from_n_chunks=policy.n_chunks,
                    to_n_chunks=policies[policy_index + 1].n_chunks,
                )
                write_json(args.state_path, state)
                return "continue"
            if int(state.get("final_policy_resubmits", 0)) >= args.max_final_embedding_resubmits:
                if args.allow_partial_downstream:
                    state["stage"] = "sae"
                    append_event(state, "final embedding policy incomplete; proceeding with complete datasets")
                    write_json(args.state_path, state)
                    return "continue"
                state["stage"] = "blocked"
                append_event(state, "final embedding policy failed", missing_rows=len(missing_rows))
                write_json(args.state_path, state)
                print("Final embedding policy still has missing chunks; controller is blocked.")
                return "stop"
            state["final_policy_resubmits"] = int(state.get("final_policy_resubmits", 0)) + 1

        job_ids = submit_embedding_arrays(args, policy, dry_run=args.dry_run)
        state.setdefault("job_ids", {})["embedding"] = job_ids
        state["embedding_submitted"] = True
        append_event(state, "submitted embedding arrays", job_ids=job_ids, policy=policy.__dict__)
        schedule_self(args, argv, state, "embedding arrays submitted")
        return "stop"

    job_ids = submit_merges(args, dry_run=args.dry_run)
    state.setdefault("job_ids", {})["merge"] = job_ids
    state["merge_submitted"] = True
    append_event(state, "submitted embedding merges", job_ids=job_ids)
    schedule_self(args, argv, state, "embedding merges submitted")
    return "stop"


def handle_sae(args: argparse.Namespace, state: dict[str, Any], argv: list[str]) -> str:
    complete_datasets = complete_datasets_for_models(args.output_root, args.models)
    if not complete_datasets:
        state["stage"] = "blocked"
        append_event(state, "no complete embedding datasets for SAE")
        write_json(args.state_path, state)
        return "stop"

    if not state.get("sae_submitted"):
        job_id = create_sae_jobs(args, complete_datasets, dry_run=args.dry_run)
        state.setdefault("job_ids", {})["sae"] = [job_id] if job_id else []
        state["sae_submitted"] = True
        state["sae_datasets"] = complete_datasets
        state["sae_attempts"] = int(state.get("sae_attempts", 0)) + 1
        append_event(state, "submitted SAE jobs", datasets=complete_datasets, job_id=job_id)
        schedule_self(args, argv, state, "SAE jobs submitted")
        return "stop"

    collect_sae(args, list(state.get("sae_datasets", complete_datasets)), dry_run=args.dry_run)
    ok, have, total = sae_tasks_ok(args.output_root)
    if not ok:
        if int(state.get("sae_attempts", 0)) < args.max_sae_attempts:
            state["sae_submitted"] = False
            state["job_ids"] = {}
            append_event(state, "SAE incomplete; retrying", have=have, total=total)
            write_json(args.state_path, state)
            return "continue"
        state["stage"] = "blocked"
        append_event(state, "SAE attempts exhausted", have=have, total=total)
        write_json(args.state_path, state)
        return "stop"

    state["stage"] = "benchmark"
    state["job_ids"] = {}
    append_event(state, "SAE complete", have=have, total=total)
    write_json(args.state_path, state)
    return "continue"


def handle_benchmark(args: argparse.Namespace, state: dict[str, Any], argv: list[str]) -> str:
    complete_datasets = list(state.get("sae_datasets") or complete_datasets_for_models(args.output_root, args.models))
    count_datasets = count_ready_datasets(args.output_root, complete_datasets)
    if not count_datasets:
        state["stage"] = "complete"
        append_event(state, "no count-ready complete datasets; downstream analysis skipped")
        write_json(args.state_path, state)
        return "stop"

    if not state.get("benchmark_submitted"):
        job_id = create_benchmark_jobs(args, count_datasets, dry_run=args.dry_run)
        state.setdefault("job_ids", {})["benchmark"] = [job_id] if job_id else []
        state["benchmark_submitted"] = True
        state["benchmark_datasets"] = count_datasets
        state["benchmark_attempts"] = int(state.get("benchmark_attempts", 0)) + 1
        append_event(state, "submitted benchmark jobs", datasets=count_datasets, job_id=job_id)
        schedule_self(args, argv, state, "benchmark jobs submitted")
        return "stop"

    collect_benchmarks(args, list(state.get("benchmark_datasets", count_datasets)), dry_run=args.dry_run)
    ok, have, total = benchmark_tasks_ok(args.output_root)
    if not ok:
        if int(state.get("benchmark_attempts", 0)) < args.max_benchmark_attempts:
            state["benchmark_submitted"] = False
            state["job_ids"] = {}
            append_event(state, "benchmark incomplete; retrying", have=have, total=total)
            write_json(args.state_path, state)
            return "continue"
        state["stage"] = "blocked"
        append_event(state, "benchmark attempts exhausted", have=have, total=total)
        write_json(args.state_path, state)
        return "stop"

    state["stage"] = "ensemble" if args.run_ensemble else "summarize"
    state["job_ids"] = {}
    append_event(state, "benchmark complete", have=have, total=total)
    write_json(args.state_path, state)
    return "continue"


def handle_ensemble(args: argparse.Namespace, state: dict[str, Any], argv: list[str]) -> str:
    datasets = list(state.get("benchmark_datasets", []))
    if not ensemble_candidates_exist(args.output_root, datasets):
        state["stage"] = "summarize"
        append_event(state, "no ensemble candidates; skipping ensemble")
        write_json(args.state_path, state)
        return "continue"

    if not state.get("ensemble_submitted"):
        job_id = create_ensemble_jobs(args, datasets, dry_run=args.dry_run)
        state.setdefault("job_ids", {})["ensemble"] = [job_id] if job_id else []
        state["ensemble_submitted"] = True
        state["ensemble_attempts"] = int(state.get("ensemble_attempts", 0)) + 1
        append_event(state, "submitted ensemble jobs", datasets=datasets, job_id=job_id)
        schedule_self(args, argv, state, "ensemble jobs submitted")
        return "stop"

    collect_ensembles(args, datasets, dry_run=args.dry_run)
    ok, have, total = ensemble_tasks_ok(args.output_root)
    if not ok:
        if int(state.get("ensemble_attempts", 0)) < args.max_ensemble_attempts:
            state["ensemble_submitted"] = False
            state["job_ids"] = {}
            append_event(state, "ensemble incomplete; retrying", have=have, total=total)
            write_json(args.state_path, state)
            return "continue"
        state["stage"] = "blocked"
        append_event(state, "ensemble attempts exhausted", have=have, total=total)
        write_json(args.state_path, state)
        return "stop"

    state["stage"] = "summarize"
    state["job_ids"] = {}
    append_event(state, "ensemble complete", have=have, total=total)
    write_json(args.state_path, state)
    return "continue"


def handle_summarize(args: argparse.Namespace, state: dict[str, Any]) -> str:
    datasets = list(state.get("benchmark_datasets", []))
    if datasets:
        summarize(args, datasets, dry_run=args.dry_run)
    state["stage"] = "complete"
    append_event(state, "controller complete", datasets=datasets)
    write_json(args.state_path, state)
    print(f"Controller complete. State: {args.state_path}")
    return "stop"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--controller-script", type=Path, default=DEFAULT_CONTROLLER_SCRIPT)
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_OUTPUT_ROOT / "hf_cache")
    parser.add_argument(
        "--embedding-policy",
        action="append",
        help="Retry policy as N_CHUNKS:MEM:MAX_ACTIVE. May be repeated.",
    )
    parser.add_argument("--embedding-partition", default="dept_cpu")
    parser.add_argument("--embedding-cpus", type=int, default=4)
    parser.add_argument("--embedding-time", default="24:00:00")
    parser.add_argument("--merge-partition", default="any_cpu")
    parser.add_argument("--merge-mem", default="16G")
    parser.add_argument("--merge-time", default="02:00:00")
    parser.add_argument("--max-final-embedding-resubmits", type=int, default=1)
    parser.add_argument("--allow-partial-downstream", action="store_true")
    parser.add_argument("--sae-partition", default="any_cpu")
    parser.add_argument("--sae-cpus", type=int, default=4)
    parser.add_argument("--sae-mem", default="48G")
    parser.add_argument("--sae-time", default="08:00:00")
    parser.add_argument("--max-active-sae-tasks", type=int, default=10)
    parser.add_argument("--max-sae-attempts", type=int, default=3)
    parser.add_argument("--benchmark-partition", default="any_cpu")
    parser.add_argument("--benchmark-cpus", type=int, default=4)
    parser.add_argument("--benchmark-mem", default="48G")
    parser.add_argument("--benchmark-time", default="08:00:00")
    parser.add_argument("--max-active-benchmark-tasks", type=int, default=12)
    parser.add_argument("--max-benchmark-attempts", type=int, default=3)
    parser.add_argument("--run-ensemble", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ensemble-top-n", type=int, default=12)
    parser.add_argument("--ensemble-gamma", type=float, default=1.0)
    parser.add_argument("--ensemble-partition", default="any_cpu")
    parser.add_argument("--ensemble-cpus", type=int, default=4)
    parser.add_argument("--ensemble-mem", default="96G")
    parser.add_argument("--ensemble-time", default="18:00:00")
    parser.add_argument("--max-active-ensemble-tasks", type=int, default=4)
    parser.add_argument("--max-ensemble-attempts", type=int, default=3)
    parser.add_argument(
        "--sahu-brca2-final",
        choices=["both", "last2048", "sliding-window", "none"],
        default="both",
    )
    parser.add_argument("--poll-minutes", type=int, default=30)
    parser.add_argument("--resubmit-self", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    args.output_root = args.output_root.resolve()
    args.controller_script = args.controller_script.resolve()
    if args.state_path is None:
        args.state_path = args.output_root / "tables" / "clinprotgym_master_controller_state.json"
    else:
        args.state_path = args.state_path.resolve()
    args.models = list(args.models)
    policies = parse_policies(args.embedding_policy)

    state = read_json(
        args.state_path,
        {
            "version": 1,
            "stage": "embedding",
            "embedding_policy_index": 0,
            "job_ids": {},
            "history": [],
        },
    )
    append_event(state, "controller check started", stage=state.get("stage", "embedding"))

    running = running_job_ids(job_ids_from_state(state), dry_run=args.dry_run)
    if running:
        print(f"Tracked jobs are still running: {', '.join(running)}")
        schedule_self(args, raw_argv, state, "tracked jobs still running")
        return 0

    for _ in range(20):
        stage = state.get("stage", "embedding")
        if stage == "embedding":
            action = handle_embedding(args, state, policies, raw_argv)
        elif stage == "sae":
            action = handle_sae(args, state, raw_argv)
        elif stage == "benchmark":
            action = handle_benchmark(args, state, raw_argv)
        elif stage == "ensemble":
            action = handle_ensemble(args, state, raw_argv)
        elif stage == "summarize":
            action = handle_summarize(args, state)
        elif stage in {"complete", "blocked"}:
            append_event(state, f"controller {stage}")
            write_json(args.state_path, state)
            print(f"Controller stage is {stage}. State: {args.state_path}")
            return 0 if stage == "complete" else 2
        else:
            raise ValueError(f"Unknown controller stage: {stage}")
        if action != "continue":
            return 0 if state.get("stage") != "blocked" else 2

    raise RuntimeError("Controller exceeded internal stage transition limit.")


if __name__ == "__main__":
    raise SystemExit(main())
