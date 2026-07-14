#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "esmDMS_matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from scipy.sparse.linalg import LinearOperator, cg
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import lower_context_helpers as h  # noqa: E402
from esmDMS import CellularDMSInput, ESMDMSConfig, esmDMS  # noqa: E402
from popDMS import get_best_regularization, mini_infer_esm  # noqa: E402


DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "clin_dms_data" / "data" / "final"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "clinprotgym_esmc_sae"
DEFAULT_PYTHON_EXECUTABLE = str(Path(sys.executable).resolve())
DEFAULT_MODELS = ("biohub/ESMC-300M", "biohub/ESMC-600M")
EMBEDDING_TYPE = "max_pool"
NORM_SCHEME = "none"
ABSTRACTION_METHOD = "DeltaEmbSAE"
WILDTYPE_KEY = "__wildtype__"
PATHOGENICITY_LABELS = {"benign", "pathogenic"}
CLINVAR_REVIEW_STAR_CUTOFFS = (0, 1, 2, 3, 4)
ESMC_CONTEXT_LENGTH = 2048
ESMC_WINDOW_STRIDE = ESMC_CONTEXT_LENGTH // 2
SAHU_BRCA2_BASE_DATASET = "MV_BRCA2_Sahu_2025"
SAHU_BRCA2_LAST2048_DATASET = f"{SAHU_BRCA2_BASE_DATASET}__last2048"
SAHU_BRCA2_WINDOW_DATASET = f"{SAHU_BRCA2_BASE_DATASET}__sliding2048_overlap1024"
SAHU_BRCA2_DERIVED_DATASETS = {
    "last2048": SAHU_BRCA2_LAST2048_DATASET,
    "sliding-window": SAHU_BRCA2_WINDOW_DATASET,
}

MODEL_LAYER_COUNTS = {
    "biohub/ESMC-300M": 30,
    "biohub/ESMC-600M": 36,
    # Register a future ESM-C 3B checkpoint here once the exact HF id and
    # layer count are known, or pass --model-layer-count MODEL=N.
}
MODEL_SHORT_NAMES = {
    "biohub/ESMC-300M": "ESMC-300M",
    "biohub/ESMC-600M": "ESMC-600M",
}
MODEL_SIZE_MILLIONS = {
    "biohub/ESMC-300M": 300,
    "biohub/ESMC-600M": 600,
}

BEST_SAE_PARAMS = {
    "n_features": 12800,
    "sparsity_coeff": 1e-3,
    "sparsity_mode": "batchtopk",
    "k": 64,
    "epochs": 200,
    "batch_size": 64,
    "train_frac": 0.8,
    "lr": 1e-3,
    "seed": 42,
    "norm_scheme": NORM_SCHEME,
    "deduplicate_training_sequences": True,
    "sae_training_version": 2,
    "run_label": "DeltaEmbSAE_max_pool_batchtopk_k64_nf12800_seed42_seqdedup_v2",
}

MUTANT_RE = re.compile(r"^([A-Z*])(\d+)([A-Z*])$")
AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    dataset_dir: Path
    table_dir: Path
    sequence_dir: Path
    job_dir: Path
    figure_dir: Path
    reference_path: Path
    counts_path: Path
    scores_path: Path
    sequence_dataframe_path: Path
    metadata_path: Path
    annotations_path: Path
    processing_summary_path: Path
    state_path: Path
    embedding_job_table_path: Path
    embedding_merge_job_table_path: Path
    embedding_cache_status_path: Path
    sae_metrics_path: Path
    benchmark_metrics_path: Path
    ensemble_metrics_path: Path


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-") or "dataset"


def progress(message: str) -> None:
    print(f"[clinprotgym] {message}", file=sys.stderr, flush=True)


def sahu_brca2_derived_dataset_names() -> set[str]:
    return set(SAHU_BRCA2_DERIVED_DATASETS.values())


def is_sahu_brca2_base_dataset(dataset: object) -> bool:
    return safe_name(dataset) == SAHU_BRCA2_BASE_DATASET


def is_sahu_brca2_window_dataset(dataset: object) -> bool:
    return safe_name(dataset) == SAHU_BRCA2_WINDOW_DATASET


def model_short_name(model_name: str) -> str:
    if model_name in MODEL_SHORT_NAMES:
        return MODEL_SHORT_NAMES[model_name]
    tail = str(model_name).split("/")[-1]
    return tail.replace("_", "-")


def model_cache_label(model_name: str) -> str:
    return str(model_name).replace("/", "__")


def model_layers(model_name: str, layer_counts: dict[str, int]) -> list[int]:
    if model_name not in layer_counts:
        raise ValueError(
            f"No layer count is registered for {model_name!r}. "
            "Use --model-layer-count MODEL=N after confirming the checkpoint's layer count."
        )
    return list(range(int(layer_counts[model_name]) + 1))


def model_job_defaults(model_name: str) -> dict:
    size = MODEL_SIZE_MILLIONS.get(model_name, 600)
    is_esmc = "ESMC" in model_name.upper()
    return {
        "partition": "any_cpu",
        "gres": None,
        "constraint": None,
        "mem": "64G" if size >= 300 or is_esmc else "48G",
        "time": "12:00:00",
        "torch_dtype": "float32" if is_esmc else None,
        "allow_cpu_esmc": is_esmc,
        "n_chunks": 40,
    }


def dataset_paths(output_root: Path, dataset: str) -> DatasetPaths:
    dataset_dir = output_root / "datasets" / dataset
    table_dir = dataset_dir / "tables"
    sequence_dir = dataset_dir / "sequence_data"
    job_dir = dataset_dir / "jobs"
    figure_dir = dataset_dir / "figures"
    return DatasetPaths(
        dataset=dataset,
        dataset_dir=dataset_dir,
        table_dir=table_dir,
        sequence_dir=sequence_dir,
        job_dir=job_dir,
        figure_dir=figure_dir,
        reference_path=sequence_dir / f"{dataset}_reference_sequence.dat",
        counts_path=sequence_dir / f"{dataset}_adapter_counts.csv",
        scores_path=sequence_dir / f"{dataset}_scores.csv",
        sequence_dataframe_path=sequence_dir / f"{dataset}_sequence_dataframe.csv",
        metadata_path=table_dir / f"{dataset}_variant_metadata.csv",
        annotations_path=table_dir / f"{dataset}_clinvar_annotations.csv",
        processing_summary_path=table_dir / f"{dataset}_processing_summary.csv",
        state_path=sequence_dir / f"{dataset}_processed_state.pkl",
        embedding_job_table_path=table_dir / f"{dataset}_embedding_jobs.csv",
        embedding_merge_job_table_path=table_dir / f"{dataset}_embedding_merge_jobs.csv",
        embedding_cache_status_path=table_dir / f"{dataset}_embedding_cache_status.csv",
        sae_metrics_path=table_dir / f"{dataset}_fixed_deltaembsae_layer_metrics.csv",
        benchmark_metrics_path=table_dir / f"{dataset}_method_metrics.csv",
        ensemble_metrics_path=table_dir / f"{dataset}_sae_ensemble_metrics.csv",
    )


def ensure_dataset_dirs(paths: DatasetPaths) -> None:
    for directory in (paths.table_dir, paths.sequence_dir, paths.job_dir, paths.figure_dir):
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def completed_sae_task_result(result_path: Path, expected_run_label: str | None = None) -> dict | None:
    if not result_path.is_file():
        return None
    try:
        result = read_json(result_path)
    except Exception:
        return None
    if result.get("status") != "ok":
        return None
    if expected_run_label is not None and result.get("run_label") != expected_run_label:
        return None

    required_keys = ["feature_path", "model_path", "viz_path"]
    if result.get("inference_path"):
        required_keys.append("inference_path")
    for key in required_keys:
        value = result.get(key)
        if not value or not Path(value).is_file():
            return None
    return result


def write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def parse_dataset_selection(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    selected = set()
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                safe_part = safe_name(Path(part).stem)
                selected.add(part)
                selected.add(safe_part)
                if safe_part == SAHU_BRCA2_BASE_DATASET:
                    selected.update(sahu_brca2_derived_dataset_names())
    return selected or None


def iter_input_csvs(input_dir: Path, selected: set[str] | None = None) -> list[Path]:
    files = sorted(Path(input_dir).glob("*.csv"))
    if selected is None:
        return files
    out = []
    for path in files:
        dataset = safe_name(path.stem)
        matches_sahu_derived = dataset == SAHU_BRCA2_BASE_DATASET and bool(
            selected & sahu_brca2_derived_dataset_names()
        )
        if path.name in selected or path.stem in selected or dataset in selected or matches_sahu_derived:
            out.append(path)
    return out


def parse_mutant(mutant: object) -> tuple[str, int, str] | None:
    match = MUTANT_RE.fullmatch(str(mutant).strip())
    if not match:
        return None
    wt_aa, pos, alt_aa = match.groups()
    return wt_aa, int(pos), alt_aa


def infer_reference_sequence(df: pd.DataFrame) -> tuple[str, dict]:
    if "mutant" not in df.columns or "mutated_sequence" not in df.columns:
        raise ValueError("ClinProtGym CSVs must contain mutant and mutated_sequence columns.")
    seq_lengths = df["mutated_sequence"].dropna().astype(str).str.len()
    if seq_lengths.empty or seq_lengths.nunique() != 1:
        raise ValueError(f"Expected one mutated_sequence length, found {sorted(seq_lengths.unique())}.")
    seq_len = int(seq_lengths.iloc[0])
    first_sequence = str(df["mutated_sequence"].dropna().iloc[0]).upper()
    reference = list(first_sequence)
    parse_failures = 0
    position_failures = 0
    alt_mismatches = 0

    for _, row in df.iterrows():
        parsed = parse_mutant(row["mutant"])
        if parsed is None:
            parse_failures += 1
            continue
        wt_aa, pos1, alt_aa = parsed
        idx = pos1 - 1
        if idx < 0 or idx >= seq_len:
            position_failures += 1
            continue
        seq = str(row["mutated_sequence"]).upper()
        if len(seq) != seq_len:
            position_failures += 1
            continue
        if seq[idx] != alt_aa:
            alt_mismatches += 1
        reference[idx] = wt_aa

    reference_sequence = "".join(reference)
    reconstruction_mismatches = 0
    for _, row in df.iterrows():
        parsed = parse_mutant(row["mutant"])
        if parsed is None:
            continue
        _, pos1, alt_aa = parsed
        idx = pos1 - 1
        if idx < 0 or idx >= seq_len:
            continue
        expected = list(reference_sequence)
        expected[idx] = alt_aa
        if "".join(expected) != str(row["mutated_sequence"]).upper():
            reconstruction_mismatches += 1

    summary = {
        "sequence_length": seq_len,
        "parse_failures": parse_failures,
        "position_failures": position_failures,
        "alt_mismatches": alt_mismatches,
        "reconstruction_mismatches": reconstruction_mismatches,
    }
    if parse_failures or position_failures or reconstruction_mismatches:
        raise ValueError(f"Could not consistently infer reference sequence: {summary}")
    return reference_sequence, summary


def mutant_with_position(parsed: tuple[str, int, str], position: int) -> str:
    wt_aa, _, alt_aa = parsed
    return f"{wt_aa}{int(position)}{alt_aa}"


def truncate_dataframe_to_reference_suffix(
    df: pd.DataFrame,
    reference_sequence: str,
    context_length: int = ESMC_CONTEXT_LENGTH,
) -> tuple[pd.DataFrame, str, dict]:
    if len(reference_sequence) <= context_length:
        out, summary = add_full_length_transform_columns(df, reference_sequence, "last2048")
        summary.update({"context_length": int(context_length), "all_mutations_retained": True})
        return out, reference_sequence, summary

    offset = len(reference_sequence) - context_length
    parsed = df["mutant"].map(parse_mutant)
    if parsed.isna().any():
        bad = df.loc[parsed.isna(), "mutant"].head().astype(str).tolist()
        raise ValueError(f"Cannot truncate sequences with unparseable mutants: {bad}")
    positions = parsed.map(lambda value: value[1])
    if int(positions.min()) <= offset:
        raise ValueError(
            f"Last-{context_length} truncation starts at original position {offset + 1}, "
            f"but the earliest mutation is at position {int(positions.min())}."
        )

    out = df.copy()
    out["original_mutant"] = out["mutant"].astype(str)
    out["original_position"] = positions.astype(int)
    out["original_sequence_length"] = int(len(reference_sequence))
    out["analysis_sequence_start"] = int(offset + 1)
    out["analysis_sequence_end"] = int(len(reference_sequence))
    out["analysis_sequence_transform"] = "last2048"
    out["mutated_sequence"] = out["mutated_sequence"].astype(str).str.upper().str.slice(offset)
    out["mutant"] = [
        mutant_with_position(value, int(value[1]) - offset)
        for value in parsed
    ]
    return out, reference_sequence[offset:], {
        "sequence_transform": "last2048",
        "analysis_sequence_start": int(offset + 1),
        "analysis_sequence_end": int(len(reference_sequence)),
        "context_length": int(context_length),
        "all_mutations_retained": True,
        "min_original_position": int(positions.min()),
        "max_original_position": int(positions.max()),
    }


def sliding_window_ranges(length: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    if length < 1:
        raise ValueError("Cannot build windows for an empty sequence.")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive.")
    if length <= window_size:
        return [(0, length)]

    final_start = length - window_size
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(int(start), int(start + window_size)) for start in starts]


def add_full_length_transform_columns(
    df: pd.DataFrame,
    reference_sequence: str,
    sequence_transform: str,
) -> tuple[pd.DataFrame, dict]:
    parsed = df["mutant"].map(parse_mutant)
    if parsed.isna().any():
        bad = df.loc[parsed.isna(), "mutant"].head().astype(str).tolist()
        raise ValueError(f"Cannot prepare sequences with unparseable mutants: {bad}")
    positions = parsed.map(lambda value: value[1])
    out = df.copy()
    out["original_mutant"] = out["mutant"].astype(str)
    out["original_position"] = positions.astype(int)
    out["original_sequence_length"] = int(len(reference_sequence))
    out["analysis_sequence_start"] = 1
    out["analysis_sequence_end"] = int(len(reference_sequence))
    out["analysis_sequence_transform"] = sequence_transform
    return out, {
        "sequence_transform": sequence_transform,
        "analysis_sequence_start": 1,
        "analysis_sequence_end": int(len(reference_sequence)),
        "context_length": int(ESMC_CONTEXT_LENGTH) if sequence_transform != "full" else np.nan,
        "min_original_position": int(positions.min()),
        "max_original_position": int(positions.max()),
    }


def build_sliding_window_embedding_state(
    sequence_to_protein_sequence: dict[str, str],
    window_size: int = ESMC_CONTEXT_LENGTH,
    stride: int = ESMC_WINDOW_STRIDE,
) -> tuple[dict, dict]:
    embedding_sequences: dict[str, str] = {}
    sequence_to_windows: dict[str, list[str]] = {}
    window_ranges: dict[str, dict] = {}
    window_lookup: dict[tuple[int, int, str], str] = {}

    for seq_id, sequence in sequence_to_protein_sequence.items():
        seq_id = str(seq_id)
        sequence = str(sequence).upper()
        window_ids = []
        for window_idx, (start, end) in enumerate(sliding_window_ranges(len(sequence), window_size, stride)):
            window_sequence = sequence[start:end]
            key = (start, end, window_sequence)
            if key not in window_lookup:
                window_id = f"window_{len(window_lookup):07d}_{start + 1}_{end}"
                window_lookup[key] = window_id
                embedding_sequences[window_id] = window_sequence
                window_ranges[window_id] = {
                    "window_index": int(window_idx),
                    "start": int(start),
                    "end": int(end),
                    "start_1based": int(start + 1),
                    "end_1based": int(end),
                    "length": int(end - start),
                }
            window_ids.append(window_lookup[key])
        sequence_to_windows[seq_id] = window_ids

    windows_per_sequence = [len(value) for value in sequence_to_windows.values()]
    window_lengths = [len(value) for value in embedding_sequences.values()]
    state = {
        "embedding_window_method": "sliding_max_pool",
        "embedding_window_size": int(window_size),
        "embedding_window_stride": int(stride),
        "embedding_sequence_to_protein_sequence": embedding_sequences,
        "sequence_to_embedding_windows": sequence_to_windows,
        "embedding_window_ranges": window_ranges,
    }
    summary = {
        "embedding_window_method": "sliding_max_pool",
        "embedding_window_size": int(window_size),
        "embedding_window_stride": int(stride),
        "embedding_window_sequences": int(len(embedding_sequences)),
        "embedding_windows_per_sequence_min": int(min(windows_per_sequence)) if windows_per_sequence else 0,
        "embedding_windows_per_sequence_max": int(max(windows_per_sequence)) if windows_per_sequence else 0,
        "embedding_window_length_min": int(min(window_lengths)) if window_lengths else 0,
        "embedding_window_length_max": int(max(window_lengths)) if window_lengths else 0,
    }
    return state, summary


def make_sequence_ids(df: pd.DataFrame) -> pd.Series:
    base = df["mutant"].astype(str).map(safe_name)
    if not base.duplicated().any():
        return base
    counts: dict[str, int] = {}
    ids = []
    for value in base:
        counts[value] = counts.get(value, 0) + 1
        suffix = "" if counts[value] == 1 else f"__row{counts[value]}"
        ids.append(f"{value}{suffix}")
    return pd.Series(ids, index=df.index)


def collapse_duplicate_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Pool duplicate rows that map to the same full protein sequence.

    The embedding and SAE layers are protein-sequence keyed. If several rows
    share one ``mutated_sequence``, count/frequency columns must be pooled before
    downstream training and fitness analysis; otherwise duplicate protein
    sequences can leak across SAE train/test splits and over-weight the same
    embedding vector.
    """
    if "mutated_sequence" not in df.columns or not df["mutated_sequence"].duplicated().any():
        return df

    has_score = "functional_score" in df.columns
    measurement_cols = [
        col for col in df.columns if str(col).startswith(("count__", "frequency__"))
    ]
    union_cols = {"clinvar_variation_ids", "clinvar_allele_ids"}
    passthrough_cols = [
        c
        for c in df.columns
        if c not in {"functional_score", *measurement_cols, *union_cols}
    ]
    rows = []
    for _, group in df.groupby("mutated_sequence", sort=False):
        best_idx = max(group.index, key=lambda idx: clinvar_representative_rank(group.loc[idx]))
        first = group.loc[best_idx]
        row = {}
        for col in passthrough_cols:
            row[col] = first[col]
        if "has_clinvar" in df.columns:
            row["has_clinvar"] = bool(group["has_clinvar"].map(truthy).any())
        for col in measurement_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[col] = float(values.sum()) if values.notna().any() else np.nan
        if has_score:
            scores = pd.to_numeric(group["functional_score"], errors="coerce").dropna()
            row["functional_score"] = float(scores.mean()) if len(scores) else np.nan
            row["functional_score_n"] = int(len(scores))
            row["functional_score_all"] = ";".join(f"{value:.6g}" for value in scores.tolist())
        for col in union_cols:
            if col in df.columns:
                values: set[str] = set()
                for value in group[col].dropna().astype(str):
                    values.update(part.strip() for part in value.split("|") if part.strip())
                row[col] = "|".join(sorted(values))
        if "mutant" in df.columns:
            mutants = [str(value) for value in group["mutant"].dropna().unique()]
            row["mutant"] = mutants[0] if mutants else first.get("mutant", "")
            if len(mutants) > 1:
                row["pooled_mutants"] = "|".join(mutants)
        row["pooled_protein_sequence_n"] = int(len(group))
        rows.append(row)

    ordered_cols = list(df.columns)
    if "pooled_mutants" in {key for row in rows for key in row} and "pooled_mutants" not in ordered_cols:
        ordered_cols.append("pooled_mutants")
    if "pooled_protein_sequence_n" not in ordered_cols:
        ordered_cols.append("pooled_protein_sequence_n")
    if has_score:
        ordered_cols = ordered_cols + ["functional_score_n", "functional_score_all"]
    return pd.DataFrame(rows, columns=ordered_cols).reset_index(drop=True)


def truthy(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def clinvar_representative_rank(row: pd.Series) -> tuple[int, int, int, int]:
    label = str(row.get("clinvar_significance_normalized", "")).strip().lower()
    kept = label in {"pathogenic", "benign", "uncertain", "uncertain significance"}
    severity = {"pathogenic": 3, "benign": 2, "uncertain": 1, "uncertain significance": 1}.get(label, 0)
    stars = clinvar_review_status_to_stars(row.get("clinvar_review_status", ""))
    return (1 if kept else 0, int(stars), int(severity), 1 if truthy(row.get("has_clinvar", False)) else 0)


def normalize_clinvar_annotation(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    labels = {part.strip().lower() for part in str(value).split("|") if part.strip()}
    if labels == {"benign"}:
        return "benign"
    if labels == {"pathogenic"}:
        return "pathogenic"
    return None


def clinvar_review_status_to_stars(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    text = str(value).strip().lower()
    if not text:
        return 0
    if "practice guideline" in text:
        return 4
    if "reviewed by expert panel" in text:
        return 3
    if "multiple submitters" in text and "no conflicts" in text:
        return 2
    if "no assertion" in text:
        return 0
    if "criteria provided" in text:
        return 1
    return 0


@dataclass(frozen=True)
class TrajectorySpec:
    column: str
    replicate: int
    replicate_name: str
    generation: float
    source_kind: str


def _strip_measure_prefix(column: str) -> str:
    return str(column).split("__", 1)[1] if "__" in str(column) else str(column)


def infer_trajectory_specs(df: pd.DataFrame) -> list[TrajectorySpec]:
    count_cols = [col for col in df.columns if str(col).startswith("count__")]
    freq_cols = [col for col in df.columns if str(col).startswith("frequency__")]
    measure_cols = count_cols if count_cols else freq_cols
    source_kind = "count" if count_cols else "frequency"
    if not measure_cols:
        return []

    day_matches = []
    for col in measure_cols:
        stripped = _strip_measure_prefix(col)
        match = re.fullmatch(r"count_day(?P<day>\d+)_rep(?P<rep>\d+)", stripped)
        if match:
            day_matches.append((col, int(match.group("rep")), float(match.group("day"))))
    if day_matches:
        specs: list[TrajectorySpec] = []
        reps = sorted({rep for _, rep, _ in day_matches})
        library_col = next((col for col in measure_cols if _strip_measure_prefix(col) == "count_library"), None)
        if library_col is not None:
            for rep in reps:
                specs.append(TrajectorySpec(library_col, rep, f"rep{rep}", 0.0, source_kind))
        for col, rep, day in day_matches:
            specs.append(TrajectorySpec(col, rep, f"rep{rep}", day, source_kind))
        return sorted(specs, key=lambda spec: (spec.replicate, spec.generation, spec.column))

    specs = []
    rep_name_to_idx: dict[str, int] = {}

    def rep_idx(rep_name: str) -> int:
        if rep_name not in rep_name_to_idx:
            rep_name_to_idx[rep_name] = len(rep_name_to_idx) + 1
        return rep_name_to_idx[rep_name]

    for col in measure_cols:
        stripped = _strip_measure_prefix(col)
        lower = stripped.lower()

        match = re.fullmatch(r"R(?P<rep>\d+)_(?:(?:D(?P<day>\d+))|(?P<lib>lib))", stripped, flags=re.IGNORECASE)
        if match:
            rep = int(match.group("rep"))
            generation = 0.0 if match.group("lib") else float(match.group("day"))
            specs.append(TrajectorySpec(col, rep, f"R{rep}", generation, source_kind))
            continue

        match = re.fullmatch(r"t(?P<time>\d+)_c_(?P<rep>\d+)", stripped, flags=re.IGNORECASE)
        if match:
            rep_name = f"c{match.group('rep')}"
            specs.append(TrajectorySpec(col, rep_idx(rep_name), rep_name, float(match.group("time")), source_kind))
            continue

        if lower in {"reference_counts", "adjusted_ref_counts", "reference_fraction", "frequency_ref"}:
            specs.append(TrajectorySpec(col, 1, "rep1", 0.0, source_kind))
            continue
        if lower in {"selected_counts", "adjusted_sel_counts", "selected_fraction", "frequency_sel"}:
            specs.append(TrajectorySpec(col, 1, "rep1", 1.0, source_kind))
            continue

        match = re.fullmatch(
            r"(?P<prefix>.+):(?P<stage>nonselect|nonselected|pre|reference|ref|library|lib|select|selected|post)_rep(?P<rep>\d+)(?:_freq)?",
            stripped,
            flags=re.IGNORECASE,
        )
        if match:
            stage = match.group("stage").lower()
            rep = int(match.group("rep"))
            generation = 0.0 if stage in {"nonselect", "nonselected", "pre", "reference", "ref", "library", "lib"} else 1.0
            specs.append(TrajectorySpec(col, rep, f"rep{rep}", generation, source_kind))
            continue

        match = re.fullmatch(r"(?P<prefix>.+):(?P<day>\d+)d", stripped, flags=re.IGNORECASE)
        if match:
            rep_name = safe_name(match.group("prefix"))
            specs.append(TrajectorySpec(col, rep_idx(rep_name), rep_name, float(match.group("day")), source_kind))
            continue

        match = re.fullmatch(r".*:R(?P<rep>\d+)[A-Za-z0-9]*_P(?P<generation>-?\d+)", stripped, flags=re.IGNORECASE)
        if match:
            rep = int(match.group("rep"))
            specs.append(TrajectorySpec(col, rep, f"R{rep}", float(match.group("generation")), source_kind))
            continue

        match = re.fullmatch(r"(?P<name>.+)_c_(?P<generation>\d+)", stripped)
        if match:
            rep_name = safe_name(match.group("name"))
            specs.append(TrajectorySpec(col, rep_idx(rep_name), rep_name, float(match.group("generation")), source_kind))

    if not specs:
        return []

    spec_df = pd.DataFrame([spec.__dict__ for spec in specs])
    valid_reps = []
    for rep, rep_df in spec_df.groupby("replicate"):
        if rep_df["generation"].nunique() >= 2:
            valid_reps.append(rep)
    specs = [spec for spec in specs if spec.replicate in set(valid_reps)]
    return sorted(specs, key=lambda spec: (spec.replicate, spec.generation, spec.column))


def build_sequence_dataframe(df: pd.DataFrame, specs: list[TrajectorySpec]) -> pd.DataFrame:
    columns = ["SequenceIndex", "mutant", "Replicate", "ReplicateName", "Generation", "Frequency", "CountColumn"]
    if not specs:
        return pd.DataFrame(columns=columns)

    records = []
    for spec in specs:
        values = pd.to_numeric(df[spec.column], errors="coerce").fillna(0.0).clip(lower=0.0)
        for seq_id, mutant, value in zip(df["SequenceIndex"], df["mutant"], values):
            records.append(
                {
                    "SequenceIndex": str(seq_id),
                    "mutant": str(mutant),
                    "Replicate": int(spec.replicate),
                    "ReplicateName": spec.replicate_name,
                    "Generation": float(spec.generation),
                    "Frequency": float(value),
                    "CountColumn": spec.column,
                }
            )
    out = pd.DataFrame(records, columns=columns)
    if out.empty:
        return out
    nonzero = out.groupby(["Replicate", "Generation"])["Frequency"].transform("sum").gt(0)
    return out[nonzero].reset_index(drop=True)


def build_metadata(df: pd.DataFrame, reference_sequence: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        wt_aa, pos1, alt_aa = parse_mutant(row["mutant"])
        mutation_sites = [] if wt_aa == alt_aa else [pos1 - 1]
        rows.append(
            {
                "SequenceIndex": str(row["SequenceIndex"]),
                "mutant": str(row["mutant"]),
                "original_mutant": str(row.get("original_mutant", row["mutant"])),
                "wt_aa": wt_aa,
                "position": pos1,
                "original_position": row.get("original_position", pos1),
                "mutant_aa": alt_aa,
                "n_mutation_sites": len(mutation_sites),
                "mutation_sites": json.dumps(mutation_sites),
                "is_synonymous": bool(wt_aa == alt_aa),
                "is_stop": bool(alt_aa == "*" or "*" in str(row["mutated_sequence"])),
                "sequence_length": len(reference_sequence),
                "original_sequence_length": row.get("original_sequence_length", len(reference_sequence)),
                "analysis_sequence_start": row.get("analysis_sequence_start", 1),
                "analysis_sequence_end": row.get("analysis_sequence_end", len(reference_sequence)),
                "analysis_sequence_transform": row.get("analysis_sequence_transform", "full"),
                "functional_score": row.get("functional_score", np.nan),
                "has_clinvar": row.get("has_clinvar", np.nan),
                "clinvar_significance_normalized": row.get("clinvar_significance_normalized", np.nan),
                "clinvar_review_status": row.get("clinvar_review_status", np.nan),
                "clinvar_review_stars": clinvar_review_status_to_stars(row.get("clinvar_review_status", np.nan)),
                "annotation": normalize_clinvar_annotation(row.get("clinvar_significance_normalized")),
            }
        )
    return pd.DataFrame(rows)


def build_annotations(metadata_df: pd.DataFrame) -> pd.DataFrame:
    out = metadata_df[
        [
            "SequenceIndex",
            "mutant",
            "annotation",
            "has_clinvar",
            "clinvar_significance_normalized",
            "clinvar_review_status",
            "clinvar_review_stars",
            "functional_score",
        ]
    ].copy()
    out["stars"] = pd.to_numeric(out["clinvar_review_stars"], errors="coerce").fillna(0).astype(int)
    return out


def prepare_dataset(
    raw_csv: Path,
    output_root: Path,
    drop_stop: bool = True,
    force: bool = False,
    dataset_name: str | None = None,
    sequence_transform: str = "full",
) -> DatasetPaths:
    raw_csv = Path(raw_csv).resolve()
    raw_stat = raw_csv.stat()
    dataset = safe_name(dataset_name or raw_csv.stem)
    paths = dataset_paths(output_root, dataset)
    ensure_dataset_dirs(paths)
    previous_state = None
    if paths.state_path.is_file():
        try:
            previous_state = read_pickle(paths.state_path)
        except Exception:
            previous_state = None
    transform_cache_keys = {
        "sequence_transform": sequence_transform,
        "context_length": int(ESMC_CONTEXT_LENGTH) if sequence_transform != "full" else None,
        "window_stride": int(ESMC_WINDOW_STRIDE) if sequence_transform == "sliding2048_overlap1024" else None,
        # Bump when the ingestion transform changes so stale cached states are
        # rebuilt. v3 pools duplicate full protein sequences and remaps embedding
        # caches by mutated_sequence.
        "ingestion_version": 3,
    }
    if paths.state_path.is_file() and not force:
        try:
            state = previous_state if previous_state is not None else read_pickle(paths.state_path)
            if (
                Path(state.get("raw_csv", "")).resolve() == raw_csv
                and int(state.get("raw_csv_size", -1)) == int(raw_stat.st_size)
                and int(state.get("raw_csv_mtime_ns", -1)) == int(raw_stat.st_mtime_ns)
                and all(state.get(key) == value for key, value in transform_cache_keys.items())
            ):
                return paths
        except Exception:
            pass

    raw_df = pd.read_csv(raw_csv)
    raw_df = collapse_duplicate_variants(raw_df)
    full_reference_sequence, ref_summary = infer_reference_sequence(raw_df)
    df = raw_df.copy()
    df["SequenceIndex"] = make_sequence_ids(df)
    if sequence_transform == "last2048":
        df, reference_sequence, transform_summary = truncate_dataframe_to_reference_suffix(
            df,
            full_reference_sequence,
            context_length=ESMC_CONTEXT_LENGTH,
        )
    elif sequence_transform == "sliding2048_overlap1024":
        reference_sequence = full_reference_sequence
        df, transform_summary = add_full_length_transform_columns(df, reference_sequence, sequence_transform)
        transform_summary.update(
            {
                "context_length": int(ESMC_CONTEXT_LENGTH),
                "window_stride": int(ESMC_WINDOW_STRIDE),
            }
        )
    elif sequence_transform == "full":
        reference_sequence = full_reference_sequence
        df, transform_summary = add_full_length_transform_columns(df, reference_sequence, sequence_transform)
        transform_summary.update({"context_length": None, "window_stride": None})
    else:
        raise ValueError(f"Unsupported sequence_transform: {sequence_transform!r}")

    parsed = df["mutant"].map(parse_mutant)
    df["wt_aa"] = parsed.map(lambda value: value[0])
    df["position"] = parsed.map(lambda value: value[1])
    df["mutant_aa"] = parsed.map(lambda value: value[2])
    df["is_stop"] = df["mutant_aa"].eq("*") | df["mutated_sequence"].astype(str).str.contains("*", regex=False)
    if drop_stop:
        df = df[~df["is_stop"]].copy()

    specs = infer_trajectory_specs(df)
    sequence_dataframe = build_sequence_dataframe(df, specs)
    metadata_df = build_metadata(df, reference_sequence)
    annotations_df = build_annotations(metadata_df)
    scores_df = df[["SequenceIndex", "mutant", "functional_score"]].copy()
    scores_df["score"] = pd.to_numeric(scores_df["functional_score"], errors="coerce")

    sequence_to_protein_sequence = {
        WILDTYPE_KEY: reference_sequence,
        **dict(zip(df["SequenceIndex"].astype(str), df["mutated_sequence"].astype(str).str.upper())),
    }
    sequence_to_mutation_sites = {WILDTYPE_KEY: []}
    for _, row in metadata_df.iterrows():
        sequence_to_mutation_sites[str(row["SequenceIndex"])] = json.loads(row["mutation_sites"])
    window_state = {}
    window_summary = {}
    if sequence_transform == "sliding2048_overlap1024":
        window_state, window_summary = build_sliding_window_embedding_state(
            sequence_to_protein_sequence,
            window_size=ESMC_CONTEXT_LENGTH,
            stride=ESMC_WINDOW_STRIDE,
        )

    paths.reference_path.write_text(f">{dataset}\n{reference_sequence}\n")
    df.to_csv(paths.counts_path, index=False)
    scores_df.to_csv(paths.scores_path, index=False)
    sequence_dataframe.to_csv(paths.sequence_dataframe_path, index=False)
    metadata_df.to_csv(paths.metadata_path, index=False)
    annotations_df.to_csv(paths.annotations_path, index=False)

    summary = {
        "dataset": dataset,
        "raw_csv": str(raw_csv),
        "raw_csv_size": int(raw_stat.st_size),
        "raw_csv_mtime_ns": int(raw_stat.st_mtime_ns),
        "raw_rows": int(len(raw_df)),
        "kept_rows": int(len(df)),
        "dropped_stop_rows": int(len(raw_df) - len(df)) if drop_stop else 0,
        "reference_length": int(len(reference_sequence)),
        "full_reference_length": int(len(full_reference_sequence)),
        "trajectory_columns": int(len(specs)),
        "trajectory_source_kind": specs[0].source_kind if specs else "",
        "trajectory_replicates": int(sequence_dataframe["Replicate"].nunique()) if not sequence_dataframe.empty else 0,
        "trajectory_generations": (
            sorted(sequence_dataframe["Generation"].dropna().unique().tolist()) if not sequence_dataframe.empty else []
        ),
        "functional_score_rows": int(scores_df["score"].notna().sum()),
        "binary_clinvar_rows": int(annotations_df["annotation"].isin(PATHOGENICITY_LABELS).sum()),
        **transform_summary,
        **window_summary,
        **ref_summary,
    }
    pd.DataFrame([summary]).to_csv(paths.processing_summary_path, index=False)

    state = {
        "dataset": dataset,
        "raw_csv": str(raw_csv),
        "raw_csv_size": int(raw_stat.st_size),
        "raw_csv_mtime_ns": int(raw_stat.st_mtime_ns),
        **transform_cache_keys,
        "paths": {key: str(value) for key, value in paths.__dict__.items() if isinstance(value, Path)},
        "reference_sequence": reference_sequence,
        "full_reference_sequence": full_reference_sequence,
        "reference_kind": "protein",
        "sequence_dataframe": sequence_dataframe,
        "sequence_to_protein_sequence": sequence_to_protein_sequence,
        "sequence_to_mutation_sites": sequence_to_mutation_sites,
        "sequence_metadata": metadata_df,
        "scores_dataframe": scores_df,
        "annotations_dataframe": annotations_df,
        "has_real_trajectory": bool(not sequence_dataframe.empty),
        "summary": summary,
        **window_state,
    }
    write_pickle(paths.state_path, state)
    remap_raw_embedding_caches(paths, previous_state, state)
    return paths


def load_dataset_state(output_root: Path, dataset: str) -> dict:
    paths = dataset_paths(output_root, dataset)
    if not paths.state_path.is_file():
        raise FileNotFoundError(f"Missing processed state for {dataset}: {paths.state_path}. Run prepare first.")
    return read_pickle(paths.state_path)


def input_data_for_paths(paths: DatasetPaths) -> CellularDMSInput:
    return CellularDMSInput(
        reference_nuc_path=paths.reference_path,
        mavedb_csv_path=paths.counts_path,
        scores_csv_path=paths.scores_path,
        reference_kind="protein",
        primary_key="SequenceIndex",
        wildtype_key=WILDTYPE_KEY,
    )


def runner_for_dataset(
    output_root: Path,
    dataset: str,
    model_name: str,
    embedding_type: str = EMBEDDING_TYPE,
) -> esmDMS:
    paths = dataset_paths(output_root, dataset)
    state = load_dataset_state(output_root, dataset)
    config = ESMDMSConfig(
        embedding_model=model_name,
        embedding_type=embedding_type,
        local_or_disk="both",
        save_dir=str(paths.sequence_dir),
        dataset_name=dataset,
    )
    runner = esmDMS(input_data_for_paths(paths), config)
    runner.reference_sequence = state["reference_sequence"]
    runner.reference_kind = "protein"
    runner.sequence_dataframe = state["sequence_dataframe"]
    runner.sequence_to_protein_sequence = state["sequence_to_protein_sequence"]
    runner.sequence_to_mutation_sites = state["sequence_to_mutation_sites"]
    runner.sequence_metadata = state["sequence_metadata"]
    runner.scores_dataframe = state["scores_dataframe"]
    return runner


def expected_embedding_sequence_keys(state: dict) -> set[str]:
    return {str(key) for key in state.get("sequence_to_protein_sequence", {})}


def remap_embedding_cache_by_protein_sequence(cache_path: Path, old_state: dict, new_state: dict) -> dict:
    row = {
        "path": str(cache_path),
        "exists": cache_path.is_file(),
        "status": "missing",
        "old_cached_sequences": 0,
        "new_expected_sequences": len(expected_embedding_sequence_keys(new_state)),
        "remapped_sequences": 0,
        "missing_sequences": 0,
        "extra_sequences_before": 0,
    }
    if not cache_path.is_file():
        return row
    try:
        features = read_pickle(cache_path)
    except Exception as exc:
        row.update({"status": "unreadable", "error": repr(exc)})
        return row
    if not isinstance(features, dict):
        row.update({"status": "not_dict", "old_cached_sequences": 0})
        return row

    old_sequence_to_ids: dict[str, list[str]] = {}
    for seq_id, protein_sequence in old_state.get("sequence_to_protein_sequence", {}).items():
        old_sequence_to_ids.setdefault(str(protein_sequence), []).append(str(seq_id))
    new_sequence_map = {
        str(seq_id): str(protein_sequence)
        for seq_id, protein_sequence in new_state.get("sequence_to_protein_sequence", {}).items()
    }
    cached_keys = {str(key) for key in features}
    expected_keys = set(new_sequence_map)
    row["old_cached_sequences"] = int(len(cached_keys))
    row["extra_sequences_before"] = int(len(cached_keys - expected_keys))

    if expected_keys and expected_keys.issubset(cached_keys) and not (cached_keys - expected_keys):
        row.update({"status": "already_current", "remapped_sequences": len(expected_keys), "missing_sequences": 0})
        return row

    remapped = {}
    missing = []
    for new_seq_id, protein_sequence in new_sequence_map.items():
        candidate_ids = old_sequence_to_ids.get(protein_sequence, [])
        old_seq_id = next((candidate for candidate in candidate_ids if candidate in features), None)
        if old_seq_id is None and new_seq_id in features:
            old_seq_id = new_seq_id
        if old_seq_id is None:
            missing.append(new_seq_id)
            continue
        remapped[new_seq_id] = features[old_seq_id]

    if not remapped:
        row.update({"status": "no_sequence_matches", "missing_sequences": len(missing)})
        return row

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    write_pickle(tmp_path, remapped)
    tmp_path.replace(cache_path)
    row.update(
        {
            "status": "remapped" if missing else "remapped_complete",
            "remapped_sequences": int(len(remapped)),
            "missing_sequences": int(len(missing)),
        }
    )
    return row


def remap_raw_embedding_caches(paths: DatasetPaths, old_state: dict | None, new_state: dict) -> list[dict]:
    if not old_state:
        return []
    pattern = f"{paths.dataset}_*_{EMBEDDING_TYPE}_none_Layer_*_seq_to_features.pkl"
    rows = []
    for cache_path in sorted(paths.sequence_dir.glob(pattern)):
        rows.append(remap_embedding_cache_by_protein_sequence(cache_path, old_state, new_state))
    if rows:
        out = paths.table_dir / f"{paths.dataset}_embedding_sequence_remap.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
    return rows


def embedding_cache_key_coverage(path: Path, state: dict) -> dict:
    expected = expected_embedding_sequence_keys(state)
    if not path.is_file():
        return {
            "n_expected_sequences": int(len(expected)),
            "n_cached_sequences": 0,
            "missing_sequence_keys": int(len(expected)),
            "extra_sequence_keys": 0,
            "covers_expected": False,
        }
    try:
        features = read_pickle(path)
    except Exception:
        return {
            "n_expected_sequences": int(len(expected)),
            "n_cached_sequences": 0,
            "missing_sequence_keys": int(len(expected)),
            "extra_sequence_keys": 0,
            "covers_expected": False,
        }
    cached = {str(key) for key in features} if isinstance(features, dict) else set()
    return {
        "n_expected_sequences": int(len(expected)),
        "n_cached_sequences": int(len(cached)),
        "missing_sequence_keys": int(len(expected - cached)),
        "extra_sequence_keys": int(len(cached - expected)),
        "covers_expected": bool(expected.issubset(cached)),
    }


def available_datasets(output_root: Path) -> list[str]:
    manifest_path = Path(output_root) / "tables" / "clinprotgym_processing_manifest.csv"
    if manifest_path.is_file():
        manifest_df = pd.read_csv(manifest_path)
        if "dataset" in manifest_df.columns:
            datasets = []
            for dataset in manifest_df["dataset"].dropna().astype(str):
                state_path = dataset_paths(Path(output_root), dataset).state_path
                if state_path.is_file():
                    datasets.append(dataset)
            if datasets:
                return datasets
    root = output_root / "datasets"
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if (path / "sequence_data" / f"{path.name}_processed_state.pkl").is_file())


def expand_sahu_brca2_base_selection(selected: set[str]) -> set[str]:
    selected = set(selected)
    if SAHU_BRCA2_BASE_DATASET in selected:
        selected.discard(SAHU_BRCA2_BASE_DATASET)
        selected.update(sahu_brca2_derived_dataset_names())
    return selected


def filter_sahu_brca2_final_datasets(datasets: Iterable[str], keep: str = "both") -> list[str]:
    keep = str(keep or "both")
    derived = sahu_brca2_derived_dataset_names()
    if keep == "both":
        keep_derived = derived
    elif keep == "none":
        keep_derived = set()
    elif keep in SAHU_BRCA2_DERIVED_DATASETS:
        keep_derived = {SAHU_BRCA2_DERIVED_DATASETS[keep]}
    else:
        raise ValueError(
            f"Unsupported Sahu BRCA2 final dataset option {keep!r}; "
            "expected one of both, last2048, sliding-window, none."
        )
    out = []
    for dataset in datasets:
        dataset = str(dataset)
        if dataset in derived:
            if dataset in keep_derived:
                out.append(dataset)
        else:
            out.append(dataset)
    return out


def dataset_uses_sliding_window_embeddings(state: dict) -> bool:
    return state.get("embedding_window_method") == "sliding_max_pool"


def filter_count_available_datasets(output_root: Path, datasets: Iterable[str]) -> list[str]:
    output_root = Path(output_root)
    datasets = [str(dataset) for dataset in datasets]
    progress(f"Filtering {len(datasets)} datasets to count-ready datasets")
    out = []
    for idx, dataset in enumerate(datasets, start=1):
        progress(f"  [{idx}/{len(datasets)}] loading processed state for {dataset}")
        state = load_dataset_state(output_root, dataset)
        if bool(state.get("has_real_trajectory", False)):
            out.append(dataset)
    progress(f"Count-ready datasets: {len(out)}/{len(datasets)}")
    return out


def command_datasets(args: argparse.Namespace, output_root: Path, count_datasets_only: bool = False) -> list[str]:
    datasets = list(args.datasets) if getattr(args, "datasets", None) else available_datasets(output_root)
    if count_datasets_only or getattr(args, "count_datasets_only", False):
        datasets = filter_count_available_datasets(output_root, datasets)
    return datasets


def annotation_map_for_dataset(output_root: Path, dataset: str) -> dict[str, str]:
    annotations = load_dataset_state(output_root, dataset)["annotations_dataframe"].copy()
    annotations = annotations[annotations["annotation"].isin(PATHOGENICITY_LABELS)]
    return dict(zip(annotations["SequenceIndex"].astype(str), annotations["annotation"]))


def scores_dataframe_for_dataset(output_root: Path, dataset: str) -> pd.DataFrame:
    return load_dataset_state(output_root, dataset)["scores_dataframe"].copy()


def classification_metrics_for_fitness(fitness_df: pd.DataFrame) -> dict:
    binary_df = fitness_df[fitness_df["annotation"].isin(PATHOGENICITY_LABELS)].dropna(subset=["fitness"]).copy()
    if binary_df.empty:
        return {"auc": np.nan, "n_benign": 0, "n_pathogenic": 0, "n_variants": 0}
    labels = binary_df["annotation"].eq("pathogenic").to_numpy()
    scores = -pd.to_numeric(binary_df["fitness"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        auc = np.nan
    else:
        ranks = pd.Series(scores).rank(method="average").to_numpy()
        auc = float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return {"auc": auc, "n_benign": int(n_neg), "n_pathogenic": int(n_pos), "n_variants": int(len(labels))}


def spearman_for_fitness(fitness_df: pd.DataFrame, scores_df: pd.DataFrame) -> tuple[float, int]:
    score_df = scores_df[["SequenceIndex", "score"]].dropna().copy()
    score_df["SequenceIndex"] = score_df["SequenceIndex"].astype(str)
    merged = fitness_df.merge(score_df, on="SequenceIndex", how="inner")
    fitness = pd.to_numeric(merged["fitness"], errors="coerce").to_numpy(dtype=float)
    score = pd.to_numeric(merged["score"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(fitness) & np.isfinite(score)
    if finite.sum() < 3:
        return np.nan, int(finite.sum())
    return float(spearmanr(fitness[finite], score[finite]).statistic), int(finite.sum())


def metrics_row(
    dataset: str,
    fitness_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    method_family: str,
    benchmark: str,
    method: str,
    model: str = "",
    model_short: str = "",
    layer: int | float = np.nan,
    model_label: str = "",
    embedding_type: str = "",
    feature_path: str | Path | None = "",
    inference_path: str | Path | None = "",
    fitness_path: str | Path | None = "",
    n_features: int | float = np.nan,
    k: int | float = np.nan,
    spearman_target: str = "functional_score",
    extra: dict | None = None,
) -> dict:
    rho, n_score = spearman_for_fitness(fitness_df, scores_df)
    row = {
        "dataset": dataset,
        "method_family": method_family,
        "benchmark": benchmark,
        "method": method,
        "model": model,
        "model_short": model_short,
        "layer": layer,
        "model_label": model_label or method_family,
        "embedding_type": embedding_type,
        "n_features": n_features,
        "k": k,
        "spearman_rho": rho,
        "spearman_target": spearman_target,
        "n_score_sequences": n_score,
        "feature_path": str(feature_path or ""),
        "inference_path": str(inference_path or ""),
        "fitness_path": str(fitness_path or ""),
        **classification_metrics_for_fitness(fitness_df),
    }
    if extra:
        row.update(extra)
    return row


def add_annotations(fitness_df: pd.DataFrame, annotation_map: dict[str, str]) -> pd.DataFrame:
    out = fitness_df.copy()
    out["SequenceIndex"] = out["SequenceIndex"].astype(str)
    out["fitness"] = pd.to_numeric(out["fitness"], errors="coerce")
    out["annotation"] = out["SequenceIndex"].map(annotation_map)
    return out


def enrichment_ratio_fitness(sequence_dataframe: pd.DataFrame, pseudocount: float = 0.5) -> pd.DataFrame:
    if sequence_dataframe.empty:
        raise ValueError("No real trajectory is available for enrichment-ratio baseline.")
    series = []
    for rep, rep_df in sequence_dataframe.groupby("Replicate"):
        generations = sorted(rep_df["Generation"].dropna().unique())
        if len(generations) < 2:
            continue
        initial = rep_df[rep_df["Generation"].eq(generations[0])].groupby("SequenceIndex")["Frequency"].sum()
        final = rep_df[rep_df["Generation"].eq(generations[-1])].groupby("SequenceIndex")["Frequency"].sum()
        ids = sorted(set(initial.index.astype(str)) | set(final.index.astype(str)))
        initial = initial.reindex(ids, fill_value=0.0).astype(float)
        final = final.reindex(ids, fill_value=0.0).astype(float)
        n = len(ids)
        initial_freq = (initial + pseudocount) / (initial.sum() + pseudocount * n)
        final_freq = (final + pseudocount) / (final.sum() + pseudocount * n)
        series.append(np.log2(final_freq / initial_freq).rename(f"rep{rep}"))
    if not series:
        raise ValueError("No replicate had at least two trajectory generations.")
    enrichment = pd.concat(series, axis=1).mean(axis=1, skipna=True)
    return pd.DataFrame({"SequenceIndex": enrichment.index.astype(str), "fitness": enrichment.to_numpy(dtype=float)})


def spearman_target_for_state(state: dict) -> tuple[pd.DataFrame, str]:
    scores_df = state["scores_dataframe"][["SequenceIndex", "score"]].copy()
    scores_df["SequenceIndex"] = scores_df["SequenceIndex"].astype(str)
    scores_df["score"] = pd.to_numeric(scores_df["score"], errors="coerce")
    if scores_df["score"].notna().sum() >= 3:
        return scores_df, "functional_score"

    sequence_df = state["sequence_dataframe"]
    if not sequence_df.empty:
        target_df = enrichment_ratio_fitness(sequence_df).rename(columns={"fitness": "score"})
        return target_df[["SequenceIndex", "score"]], "enrichment_ratio"
    return scores_df, "functional_score_unavailable"


def substitution_feature_index(metadata_df: pd.DataFrame) -> tuple[dict[str, int | None], list[str]]:
    feature_names = []
    feature_to_idx: dict[str, int] = {}
    seq_to_feature: dict[str, int | None] = {}
    for _, row in metadata_df.iterrows():
        seq_id = str(row["SequenceIndex"])
        if bool(row.get("is_synonymous", False)) or str(row.get("mutant_aa")) == "*":
            seq_to_feature[seq_id] = None
            continue
        feature = f"{int(row['position'])}:{row['mutant_aa']}"
        if feature not in feature_to_idx:
            feature_to_idx[feature] = len(feature_names)
            feature_names.append(feature)
        seq_to_feature[seq_id] = feature_to_idx[feature]
    return seq_to_feature, feature_names


def _cg_solve(operator: LinearOperator, rhs: np.ndarray) -> np.ndarray:
    try:
        solution, info = cg(operator, rhs, rtol=1e-6, atol=0.0, maxiter=1000)
    except TypeError:
        solution, info = cg(operator, rhs, tol=1e-6, maxiter=1000)
    if info != 0:
        raise RuntimeError(f"Conjugate-gradient solve did not converge; info={info}")
    return solution


def _replicate_operator_payload(rep_df: pd.DataFrame, seq_to_feature: dict[str, int | None], n_features: int) -> dict:
    generations = np.sort(rep_df["Generation"].dropna().unique())
    if len(generations) < 2:
        raise ValueError("Need at least two generations for popDMS baseline.")
    trap_weights = np.zeros(len(generations), dtype=float)
    trap_weights[0] = (generations[1] - generations[0]) / 2.0
    trap_weights[-1] = (generations[-1] - generations[-2]) / 2.0
    for idx in range(1, len(generations) - 1):
        trap_weights[idx] = (generations[idx + 1] - generations[idx - 1]) / 2.0

    mus = []
    diag = np.zeros(n_features, dtype=float)
    for generation, weight in zip(generations, trap_weights):
        time_df = rep_df[rep_df["Generation"].eq(generation)]
        total = float(time_df["Frequency"].sum())
        mu = np.zeros(n_features, dtype=float)
        if total > 0:
            for seq_id, value in zip(time_df["SequenceIndex"].astype(str), time_df["Frequency"].astype(float)):
                feature_idx = seq_to_feature.get(seq_id)
                if feature_idx is not None:
                    mu[feature_idx] += float(value) / total
        diag += weight * mu
        if weight > 0:
            mus.append((math.sqrt(float(weight)), mu))
    return {"generations": generations, "diag": diag, "mus": mus, "dx": mus[-1][1] - mus[0][1]}


def _solve_substitution_selection(payload: dict, gamma: float) -> np.ndarray:
    diag = payload["diag"]
    mus = payload["mus"]
    n_features = len(diag)

    def matvec(vector: np.ndarray) -> np.ndarray:
        out = (diag + gamma) * vector
        for sqrt_weight, mu in mus:
            out -= sqrt_weight * mu * (sqrt_weight * float(mu @ vector))
        return out

    operator = LinearOperator((n_features, n_features), matvec=matvec, dtype=float)
    return _cg_solve(operator, payload["dx"])


def popdms_substitution_fitness(
    sequence_dataframe: pd.DataFrame,
    metadata_df: pd.DataFrame,
    gamma_values: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sequence_dataframe.empty:
        raise ValueError("No real trajectory is available for popDMS baseline.")
    seq_to_feature, feature_names = substitution_feature_index(metadata_df)
    n_features = len(feature_names)
    if n_features == 0:
        raise ValueError("No non-synonymous substitution features are available.")
    rep_payloads = [
        _replicate_operator_payload(rep_df, seq_to_feature, n_features)
        for _, rep_df in sequence_dataframe.groupby("Replicate")
        if rep_df["Generation"].nunique() >= 2
    ]
    if not rep_payloads:
        raise ValueError("No replicate had at least two trajectory generations.")
    if gamma_values is None:
        gamma_values = np.logspace(np.log10(1 / 100), 4, num=20)

    gamma_values = np.asarray(gamma_values, dtype=float)
    gamma_rows = []
    selected_gamma_consistency = np.nan
    selected_gamma_n_pairs = 0
    if len(rep_payloads) == 1:
        gamma_opt = 1.0
    else:
        corrs = []
        for gamma in gamma_values:
            selections = [_solve_substitution_selection(payload, float(gamma)) for payload in rep_payloads]
            pair_corrs = []
            for i in range(len(selections)):
                for j in range(i + 1, len(selections)):
                    pair_corrs.append(pearsonr(selections[i], selections[j]).statistic)
            corr = float(np.nanmean(pair_corrs)) if pair_corrs else np.nan
            corrs.append(corr)
            gamma_rows.append({"gamma": float(gamma), "mean_pairwise_pearson_r": corr})
        gamma_opt = float(get_best_regularization(corrs, gamma_values, corr_cutoff_pct=0.5))
        selected_gamma_consistency, selected_gamma_n_pairs = mean_pairwise_pearson(
            [_solve_substitution_selection(payload, gamma_opt) for payload in rep_payloads]
        )

    joint_diag = np.sum([payload["diag"] for payload in rep_payloads], axis=0)
    joint_mus = [item for payload in rep_payloads for item in payload["mus"]]
    joint_dx = np.sum([payload["dx"] for payload in rep_payloads], axis=0)
    joint_payload = {"diag": joint_diag, "mus": joint_mus, "dx": joint_dx}
    s_joint = _solve_substitution_selection(joint_payload, gamma_opt * len(rep_payloads))

    rows = []
    for _, row in metadata_df.iterrows():
        seq_id = str(row["SequenceIndex"])
        feature_idx = seq_to_feature.get(seq_id)
        fitness = 1.0 if feature_idx is None else 1.0 + float(s_joint[feature_idx])
        rows.append({"SequenceIndex": seq_id, "fitness": fitness})
    selection_df = pd.DataFrame(
        {
            "feature": feature_names,
            "selection_coefficient": s_joint,
            "gamma_opt": gamma_opt,
            "n_replicates": len(rep_payloads),
            "mean_pairwise_pearson_r": selected_gamma_consistency,
            "n_replicate_pairs": selected_gamma_n_pairs,
        }
    )
    if gamma_rows:
        selection_df.attrs["gamma_diagnostics"] = pd.DataFrame(gamma_rows)
    return pd.DataFrame(rows), selection_df


def mean_pairwise_pearson(matrix: object) -> tuple[float, int]:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.nan, 0
    pair_corrs = []
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            x = arr[i]
            y = arr[j]
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() < 3:
                continue
            x = x[finite]
            y = y[finite]
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                continue
            pair_corrs.append(float(pearsonr(x, y).statistic))
    if not pair_corrs:
        return np.nan, 0
    return float(np.nanmean(pair_corrs)), int(len(pair_corrs))


def replicate_enrichment_consistency(sequence_dataframe: pd.DataFrame, pseudocount: float = 0.5) -> tuple[float, int]:
    replicate_series = []
    for rep, rep_df in sequence_dataframe.groupby("Replicate"):
        generations = sorted(rep_df["Generation"].dropna().unique())
        if len(generations) < 2:
            continue
        initial = rep_df[rep_df["Generation"].eq(generations[0])].groupby("SequenceIndex")["Frequency"].sum()
        final = rep_df[rep_df["Generation"].eq(generations[-1])].groupby("SequenceIndex")["Frequency"].sum()
        ids = sorted(set(initial.index.astype(str)) | set(final.index.astype(str)))
        if not ids:
            continue
        initial = initial.reindex(ids, fill_value=0.0).astype(float)
        final = final.reindex(ids, fill_value=0.0).astype(float)
        n = len(ids)
        initial_freq = (initial + pseudocount) / (initial.sum() + pseudocount * n)
        final_freq = (final + pseudocount) / (final.sum() + pseudocount * n)
        replicate_series.append(np.log2(final_freq / initial_freq).rename(str(rep)))
    if len(replicate_series) < 2:
        return np.nan, 0
    return mean_pairwise_pearson(pd.concat(replicate_series, axis=1).T.to_numpy(dtype=float))


def inference_result_consistency(inference_path: object) -> tuple[float, int]:
    if inference_path is None or pd.isna(inference_path):
        return np.nan, 0
    path = Path(str(inference_path))
    if not path.is_file():
        return np.nan, 0
    result = read_pickle(path)
    s_replicates = getattr(result, "s", None)
    if s_replicates is None:
        try:
            s_replicates = result[2]
        except Exception:
            return np.nan, 0
    return mean_pairwise_pearson(s_replicates)


def cross_replicate_consistency_dataframe(
    output_root: Path,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    output_root = Path(output_root)
    if datasets is None:
        datasets = available_datasets(output_root)
    rows = []
    for dataset in datasets:
        state = load_dataset_state(output_root, dataset)
        sequence_df = state["sequence_dataframe"]
        n_replicates = int(sequence_df["Replicate"].nunique()) if not sequence_df.empty else 0
        if n_replicates < 2:
            continue

        rho, n_pairs = replicate_enrichment_consistency(sequence_df)
        rows.append(
            {
                "dataset": dataset,
                "method_family": "Enrichment ratio baseline",
                "method_label": "Enrichment ratio",
                "cross_replicate_consistency": rho,
                "n_replicates": n_replicates,
                "n_replicate_pairs": n_pairs,
                "selection_gamma": np.nan,
                "model_short": "",
                "layer": np.nan,
                "source": "replicate enrichment ratios",
            }
        )

    metric_frames = []
    for metrics_path in [
        output_root / "tables" / "clinprotgym_method_metrics.csv",
        output_root / "tables" / "clinprotgym_sae_ensemble_metrics.csv",
    ]:
        if metrics_path.is_file():
            metric_frames.append(pd.read_csv(metrics_path))
    metrics_df = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()

    if not metrics_df.empty and "mean_pairwise_pearson_r" in metrics_df.columns:
        popdms_rows = metrics_df[metrics_df["method_family"].eq("popDMS baseline")].copy()
        if not popdms_rows.empty:
            popdms_rows["mean_pairwise_pearson_r"] = pd.to_numeric(
                popdms_rows["mean_pairwise_pearson_r"], errors="coerce"
            )
            if "spearman_rho" in popdms_rows.columns:
                popdms_rows["spearman_rho"] = pd.to_numeric(popdms_rows["spearman_rho"], errors="coerce")
            else:
                popdms_rows["spearman_rho"] = np.nan
            popdms_rows = (
                popdms_rows[np.isfinite(popdms_rows["mean_pairwise_pearson_r"])]
                .sort_values(["dataset", "mean_pairwise_pearson_r", "spearman_rho"], ascending=[True, False, False])
                .groupby("dataset", as_index=False, sort=False)
                .head(1)
            )
            for _, metric_row in popdms_rows.iterrows():
                dataset = str(metric_row.get("dataset", ""))
                if dataset not in set(datasets):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "method_family": "popDMS baseline",
                        "method_label": "popDMS",
                        "cross_replicate_consistency": metric_row["mean_pairwise_pearson_r"],
                        "n_replicates": metric_row.get("n_replicates", np.nan),
                        "n_replicate_pairs": metric_row.get("n_replicate_pairs", np.nan),
                        "selection_gamma": metric_row.get("gamma_opt", np.nan),
                        "model_short": "",
                        "layer": np.nan,
                        "source": str(metric_row.get("fitness_path", "")),
                    }
                )

    artifact_rows = []
    if not metrics_df.empty and "inference_path" in metrics_df.columns:
        method_mask = metrics_df["method_family"].isin(["Raw embeddings", "Raw SAE"])
        for _, metric_row in metrics_df[method_mask].iterrows():
            dataset = str(metric_row.get("dataset", ""))
            if dataset not in set(datasets):
                continue
            rho, n_pairs = inference_result_consistency(metric_row.get("inference_path"))
            if not np.isfinite(rho):
                continue
            artifact_rows.append(
                {
                    "dataset": dataset,
                    "method_family": metric_row.get("method_family", ""),
                    "method_label": (
                        "Raw embeddings" if metric_row.get("method_family") == "Raw embeddings" else "Raw SAE"
                    ),
                    "cross_replicate_consistency": rho,
                    "n_replicates": np.nan,
                    "n_replicate_pairs": n_pairs,
                    "selection_gamma": np.nan,
                    "model_short": metric_row.get("model_short", ""),
                    "layer": metric_row.get("layer", np.nan),
                    "spearman_rho": metric_row.get("spearman_rho", np.nan),
                    "auc": metric_row.get("auc", np.nan),
                    "source": str(metric_row.get("inference_path", "")),
                }
            )

    if artifact_rows:
        artifact_df = pd.DataFrame(artifact_rows)
        for col in ["cross_replicate_consistency", "spearman_rho", "auc"]:
            artifact_df[col] = pd.to_numeric(artifact_df[col], errors="coerce")
        artifact_df = (
            artifact_df.sort_values(
                ["dataset", "method_family", "cross_replicate_consistency", "spearman_rho", "auc"],
                ascending=[True, True, False, False, False],
            )
            .groupby(["dataset", "method_family"], as_index=False, sort=False)
            .head(1)
        )
        rows.extend(artifact_df.to_dict(orient="records"))

    for dataset in datasets:
        source_path = dataset_paths(output_root, dataset).table_dir / "sae_ensemble_gamma1" / f"{dataset}_ensemble_sources.csv"
        component_path = (
            dataset_paths(output_root, dataset).table_dir
            / "sae_ensemble_gamma1"
            / f"{dataset}_ensemble_component_fitness_paths.csv"
        )
        component_df = pd.DataFrame()
        if component_path.is_file():
            component_df = pd.read_csv(component_path)
        if component_df.empty and source_path.is_file():
            source_df = pd.read_csv(source_path)
            if "mean_pairwise_pearson_r" in source_df.columns:
                component_df = source_df.copy()
        if component_df.empty or "mean_pairwise_pearson_r" not in component_df.columns:
            continue
        consistency = pd.to_numeric(component_df["mean_pairwise_pearson_r"], errors="coerce").dropna()
        if consistency.empty:
            continue
        rows.append(
            {
                "dataset": dataset,
                "method_family": "Ensemble SAE model",
                "method_label": "Ensemble SAE",
                "cross_replicate_consistency": float(consistency.mean()),
                "n_replicates": np.nan,
                "n_replicate_pairs": int(
                    pd.to_numeric(component_df.get("n_replicate_pairs", pd.Series(dtype=float)), errors="coerce")
                    .dropna()
                    .max()
                )
                if "n_replicate_pairs" in component_df.columns
                and pd.to_numeric(component_df["n_replicate_pairs"], errors="coerce").notna().any()
                else 0,
                "selection_gamma": 1.0,
                "model_short": "ensemble",
                "layer": np.nan,
                "n_component_model_layers": int(len(consistency)),
                "source": str(component_path if component_path.is_file() else source_path),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["cross_replicate_consistency"] = pd.to_numeric(out["cross_replicate_consistency"], errors="coerce")
    return out


def write_cross_replicate_consistency_outputs(
    output_root: Path,
    datasets: list[str] | None = None,
    include_all_datasets: bool = True,
    table_path: Path | None = None,
    figure_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_root = Path(output_root)
    if datasets is None:
        datasets = available_datasets(output_root)
    table_path = table_path or output_root / "tables" / "clinprotgym_cross_replicate_consistency_by_dataset_method.csv"
    figure_path = figure_path or output_root / "figures" / "clinprotgym_cross_replicate_consistency_by_dataset_method.png"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    consistency_df = cross_replicate_consistency_dataframe(output_root, datasets=datasets)
    consistency_df.to_csv(table_path, index=False)
    plot_df = consistency_df[np.isfinite(consistency_df["cross_replicate_consistency"])].copy() if not consistency_df.empty else pd.DataFrame()
    if plot_df.empty:
        return consistency_df, plot_df, table_path, figure_path

    dataset_order = (
        plot_df.groupby("dataset")["cross_replicate_consistency"].max().sort_values(ascending=False).index.tolist()
    )
    if include_all_datasets:
        dataset_order.extend([dataset for dataset in datasets if dataset not in set(dataset_order)])
    method_order = (
        plot_df.groupby("method_label")["cross_replicate_consistency"].mean().sort_values(ascending=False).index.tolist()
    )
    palette = dict(zip(method_order, sns.color_palette("tab10", n_colors=len(method_order))))

    fig_width = max(10.0, 0.65 * len(dataset_order) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    sns.stripplot(
        data=plot_df,
        x="dataset",
        y="cross_replicate_consistency",
        hue="method_label",
        order=dataset_order,
        hue_order=method_order,
        palette=palette,
        dodge=False,
        jitter=False,
        size=8,
        linewidth=0.6,
        edgecolor="white",
        ax=ax,
    )
    ax.axhline(0, color="0.75", lw=1, zorder=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Dataset (sorted by best available cross-replicate consistency)")
    ax.set_ylabel("Cross-replicate consistency\n(mean pairwise Pearson r)")
    ax.set_title("ClinProtGym cross-replicate consistency by dataset and method")
    ax.tick_params(axis="x", rotation=60)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return consistency_df, plot_df, table_path, figure_path


def fitness_for_feature_mapping(
    seq_to_features: dict,
    inference_result,
    annotation_map: dict[str, str] | None = None,
    norm_scheme: str = NORM_SCHEME,
    baseline: float = 1.0,
) -> pd.DataFrame:
    return h.fitness_for_feature_mapping(
        seq_to_features,
        inference_result,
        annotation_map=annotation_map,
        norm_scheme=norm_scheme,
        baseline=baseline,
    )


def write_slurm_array_script(
    script_path: Path,
    payload_path: Path,
    subcommand: str,
    n_tasks: int,
    job_name: str,
    log_dir: Path,
    partition: str,
    cpus_per_task: int,
    mem: str,
    time: str,
    max_active_tasks: int | None,
    python_executable: str,
) -> Path:
    if n_tasks < 1:
        raise ValueError("n_tasks must be at least 1.")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    array_spec = f"0-{n_tasks - 1}"
    if max_active_tasks is not None:
        array_spec = f"{array_spec}%{max_active_tasks}"
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --array={array_spec}
#SBATCH --output={log_dir}/slurm-%A_%a.out
#SBATCH --error={log_dir}/slurm-%A_%a.err

set -euo pipefail
cd {REPO_ROOT}
export PYTHONUNBUFFERED=1
{python_executable} {Path(__file__).resolve()} {subcommand} {payload_path} "${{SLURM_ARRAY_TASK_ID}}"
"""
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def write_slurm_single_task_script(
    script_path: Path,
    payload_path: Path,
    subcommand: str,
    job_name: str,
    log_dir: Path,
    partition: str,
    cpus_per_task: int,
    mem: str,
    time: str,
    python_executable: str,
) -> Path:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output={log_dir}/merge-%j.out
#SBATCH --error={log_dir}/merge-%j.err

set -euo pipefail
cd {REPO_ROOT}
export PYTHONUNBUFFERED=1
{python_executable} {Path(__file__).resolve()} {subcommand} {payload_path}
"""
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def prepare_runner_for_embedding_job(runner: esmDMS, state: dict) -> tuple[esmDMS, int, str]:
    if not dataset_uses_sliding_window_embeddings(state):
        return runner, len(runner.sequence_to_protein_sequence), ""
    embedding_sequences = {
        str(seq_id): str(sequence)
        for seq_id, sequence in state["embedding_sequence_to_protein_sequence"].items()
    }
    runner.sequence_to_protein_sequence = embedding_sequences
    runner.sequence_to_mutation_sites = {seq_id: [] for seq_id in embedding_sequences}
    return runner, len(embedding_sequences), state["embedding_window_method"]


def layer_label_sort_key(layer_label: object) -> tuple[int, str]:
    label = str(layer_label)
    try:
        return int(label.replace("Layer_", "")), label
    except ValueError:
        return 10**9, label


def aggregate_sliding_window_max_pool_embeddings(
    output_root: Path,
    dataset: str,
    model_name: str,
    job_dir: Path | None = None,
    n_chunks: int | None = None,
) -> dict:
    output_root = Path(output_root)
    state = load_dataset_state(output_root, dataset)
    if not dataset_uses_sliding_window_embeddings(state):
        raise ValueError(f"{dataset} is not configured for sliding-window embedding aggregation.")
    runner = runner_for_dataset(output_root, dataset, model_name)
    if job_dir is None:
        job_dir = dataset_paths(output_root, dataset).job_dir / "embedding_batches" / model_cache_label(model_name)
    batch_dir = runner._batch_dir(job_dir)
    payload_path = runner._batch_payload_path(batch_dir)
    if payload_path.is_file() and n_chunks is None:
        n_chunks = int(read_pickle(payload_path)["n_chunks"])
    if n_chunks is None:
        raise ValueError("n_chunks is required when the embedding batch payload is missing.")

    sequence_to_windows = {
        str(seq_id): [str(window_id) for window_id in window_ids]
        for seq_id, window_ids in state["sequence_to_embedding_windows"].items()
    }
    required_windows = {window_id for window_ids in sequence_to_windows.values() for window_id in window_ids}
    layer_to_window_features: dict[str, dict[str, np.ndarray]] = {}
    missing_chunk_paths = []
    for chunk_idx in range(int(n_chunks)):
        chunk_path = runner._batch_feature_chunk_path(batch_dir, chunk_idx, EMBEDDING_TYPE)
        if not chunk_path.is_file():
            missing_chunk_paths.append(chunk_path)
            continue
        chunk = read_pickle(chunk_path)
        if not (
            isinstance(chunk, dict)
            and chunk.get("format") == "esmDMS_embedding_feature_chunk_v2"
            and chunk.get("embedding_type") == EMBEDDING_TYPE
        ):
            raise ValueError(f"Unexpected max-pool feature chunk format in {chunk_path}.")
        for layer_label, window_features in chunk["features_by_layer"].items():
            layer_features = layer_to_window_features.setdefault(str(layer_label), {})
            for window_id, feature in window_features.items():
                layer_features[str(window_id)] = np.asarray(feature)
    if missing_chunk_paths:
        raise FileNotFoundError(f"Missing max-pool embedding chunk files: {missing_chunk_paths}")

    rows = []
    for layer_label in sorted(layer_to_window_features, key=layer_label_sort_key):
        window_features = layer_to_window_features[layer_label]
        missing_windows = sorted(required_windows - set(window_features))
        if missing_windows:
            raise FileNotFoundError(
                f"Missing {len(missing_windows)} window embeddings for {dataset} {model_name} {layer_label}; "
                f"first missing windows: {missing_windows[:5]}"
            )
        sequence_features = {}
        for seq_id, window_ids in sequence_to_windows.items():
            vectors = [np.asarray(window_features[window_id]) for window_id in window_ids]
            if len(vectors) == 1:
                sequence_features[seq_id] = vectors[0].copy()
            else:
                sequence_features[seq_id] = np.maximum.reduce(vectors)
        embedding_path = runner._embedding_path(layer_label, EMBEDDING_TYPE)
        write_pickle(embedding_path, sequence_features)
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "model_short": model_short_name(model_name),
                "layer": layer_label,
                "embedding_type": EMBEDDING_TYPE,
                "n_sequences": int(len(sequence_features)),
                "n_windows": int(len(required_windows)),
                "path": str(embedding_path),
            }
        )

    paths = dataset_paths(output_root, dataset)
    status_path = paths.table_dir / f"{dataset}_{model_cache_label(model_name)}_window_embedding_merge_status.csv"
    pd.DataFrame(rows).to_csv(status_path, index=False)
    return {
        "status": "ok",
        "dataset": dataset,
        "model": model_name,
        "embedding_type": EMBEDDING_TYPE,
        "n_layers": int(len(rows)),
        "n_sequences": int(len(sequence_to_windows)),
        "n_windows": int(len(required_windows)),
        "status_path": str(status_path),
    }


def create_sliding_window_embedding_merge_job(
    output_root: Path,
    dataset: str,
    model_name: str,
    job_dir: Path,
    n_chunks: int,
    partition: str,
    cpus_per_task: int,
    mem: str,
    time: str,
    python_executable: str,
) -> dict[str, Path | str]:
    batch_dir = Path(job_dir)
    logs_dir = batch_dir / "logs"
    payload = {
        "output_root": str(output_root),
        "dataset": dataset,
        "model": model_name,
        "job_dir": str(job_dir),
        "n_chunks": int(n_chunks),
    }
    payload_path = batch_dir / f"{dataset}_window_embedding_merge_payload.pkl"
    write_pickle(payload_path, payload)
    script_path = write_slurm_single_task_script(
        script_path=batch_dir / "submit_window_embedding_merge.sh",
        payload_path=payload_path,
        subcommand="run-window-embedding-merge",
        job_name=safe_name(f"{dataset}_{model_short_name(model_name)}_winmerge")[:48],
        log_dir=logs_dir,
        partition=partition,
        cpus_per_task=cpus_per_task,
        mem=mem,
        time=time,
        python_executable=python_executable,
    )
    return {"batch_dir": batch_dir, "payload_path": payload_path, "script_path": script_path, "job_id": ""}


def run_window_embedding_merge(args: argparse.Namespace) -> None:
    payload = read_pickle(Path(args.payload_path))
    result = aggregate_sliding_window_max_pool_embeddings(
        output_root=Path(payload["output_root"]),
        dataset=payload["dataset"],
        model_name=payload["model"],
        job_dir=Path(payload["job_dir"]),
        n_chunks=int(payload["n_chunks"]),
    )
    print(json.dumps(result, indent=2, default=str))


def create_embedding_jobs(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    datasets = args.datasets or available_datasets(output_root)
    rows = []
    merge_rows = []
    for dataset in datasets:
        paths = dataset_paths(output_root, dataset)
        state = load_dataset_state(output_root, dataset)
        for model_name in args.models:
            runner = runner_for_dataset(output_root, dataset, model_name)
            defaults = model_job_defaults(model_name)
            job_dir = paths.job_dir / "embedding_batches" / model_cache_label(model_name)
            cache_complete = all(
                runner._embedding_path(layer, EMBEDDING_TYPE).is_file()
                for layer in model_layers(model_name, args.layer_counts)
            )
            action = "skip_complete" if args.skip_complete and cache_complete else "write_script"
            row = {
                "dataset": dataset,
                "model": model_name,
                "model_short": model_short_name(model_name),
                "cache_complete": cache_complete,
                "action": action,
                "script_path": "",
                "payload_path": "",
                "job_id": "",
                "submitted": False,
                "embedding_window_method": state.get("embedding_window_method", ""),
                "n_embedding_sequences": "",
            }
            if action != "skip_complete":
                embedding_runner, n_embedding_sequences, window_method = prepare_runner_for_embedding_job(runner, state)
                row["n_embedding_sequences"] = int(n_embedding_sequences)
                job = embedding_runner.create_embedding_batch_job(
                    job_dir=job_dir,
                    n_chunks=args.n_chunks or defaults["n_chunks"],
                    max_active_jobs=args.max_active_embedding_tasks,
                    job_name=safe_name(f"{dataset}_{model_short_name(model_name)}_embed")[:48],
                    partition=args.embedding_partition or defaults["partition"],
                    gres=defaults["gres"],
                    constraint=defaults["constraint"],
                    cpus_per_task=args.embedding_cpus,
                    mem=args.embedding_mem or defaults["mem"],
                    time=args.embedding_time or defaults["time"],
                    python_executable=args.python_executable,
                    scratch_root=args.scratch_root,
                    hf_home=args.hf_home,
                    torch_dtype=defaults["torch_dtype"],
                    allow_cpu_esmc=defaults["allow_cpu_esmc"],
                    submit=args.submit,
                )
                row.update(
                    {
                        "script_path": str(job["script_path"]),
                        "payload_path": str(job["payload_path"]),
                        "job_id": job["job_id"],
                        "submitted": bool(args.submit),
                    }
                )
                if window_method:
                    merge_job = create_sliding_window_embedding_merge_job(
                        output_root=output_root,
                        dataset=dataset,
                        model_name=model_name,
                        job_dir=job_dir,
                        n_chunks=args.n_chunks or defaults["n_chunks"],
                        partition=args.merge_partition,
                        cpus_per_task=1,
                        mem=args.merge_mem,
                        time=args.merge_time,
                        python_executable=args.python_executable,
                    )
                else:
                    merge_job = runner.create_embedding_batch_merge_job(
                        job_dir=job_dir,
                        layer="all",
                        n_chunks=args.n_chunks or defaults["n_chunks"],
                        save_layers=True,
                        job_name=safe_name(f"{dataset}_{model_short_name(model_name)}_merge")[:48],
                        partition=args.merge_partition,
                        cpus_per_task=1,
                        mem=args.merge_mem,
                        time=args.merge_time,
                        python_executable=args.python_executable,
                        submit=False,
                    )
                merge_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "model_short": model_short_name(model_name),
                        "script_path": str(merge_job["script_path"]),
                        "payload_path": str(merge_job["payload_path"]),
                        "submitted": False,
                        "embedding_window_method": window_method,
                    }
                )
            rows.append(row)
        pd.DataFrame([row for row in rows if row["dataset"] == dataset]).to_csv(paths.embedding_job_table_path, index=False)
        pd.DataFrame([row for row in merge_rows if row["dataset"] == dataset]).to_csv(paths.embedding_merge_job_table_path, index=False)

    all_table = output_root / "tables" / "clinprotgym_embedding_jobs.csv"
    all_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(all_table, index=False)
    pd.DataFrame(merge_rows).to_csv(output_root / "tables" / "clinprotgym_embedding_merge_jobs.csv", index=False)
    print(f"Wrote embedding job table: {all_table}")


def merge_embeddings(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    for dataset in args.datasets or available_datasets(output_root):
        state = load_dataset_state(output_root, dataset)
        for model_name in args.models:
            job_dir = dataset_paths(output_root, dataset).job_dir / "embedding_batches" / model_cache_label(model_name)
            if dataset_uses_sliding_window_embeddings(state):
                result = aggregate_sliding_window_max_pool_embeddings(
                    output_root=output_root,
                    dataset=dataset,
                    model_name=model_name,
                    job_dir=job_dir,
                    n_chunks=args.n_chunks,
                )
                print(f"Merged sliding-window max-pool embeddings for {dataset} {model_name}: {result['status_path']}")
            else:
                runner = runner_for_dataset(output_root, dataset, model_name)
                runner.merge_embedding_batch_outputs(job_dir=job_dir, layer="all", n_chunks=args.n_chunks, save_layers=True)
                print(f"Merged embeddings for {dataset} {model_name}")


def write_embedding_cache_status(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    all_rows = []
    for dataset in args.datasets or available_datasets(output_root):
        state = load_dataset_state(output_root, dataset)
        rows = []
        for model_name in args.models:
            runner = runner_for_dataset(output_root, dataset, model_name)
            for layer in model_layers(model_name, args.layer_counts):
                path = runner._embedding_path(layer, EMBEDDING_TYPE)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "model_short": model_short_name(model_name),
                        "layer": layer,
                        "embedding_type": EMBEDDING_TYPE,
                        "embedding_window_method": state.get("embedding_window_method", ""),
                        "path": str(path),
                        "exists": path.is_file(),
                        **embedding_cache_key_coverage(path, state),
                    }
                )
        paths = dataset_paths(output_root, dataset)
        pd.DataFrame(rows).to_csv(paths.embedding_cache_status_path, index=False)
        all_rows.extend(rows)
    out = output_root / "tables" / "clinprotgym_embedding_cache_status.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"Wrote embedding cache status: {out}")


def create_llr_jobs(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    tasks = []
    for dataset in args.datasets or available_datasets(output_root):
        state = load_dataset_state(output_root, dataset)
        if dataset_uses_sliding_window_embeddings(state):
            continue
        for model_name in args.models:
            paths = dataset_paths(output_root, dataset)
            tasks.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "model_short": model_short_name(model_name),
                    "llr_sites_path": str(paths.table_dir / f"{dataset}_{model_cache_label(model_name)}_llr_sites.csv"),
                    "llr_fitness_path": str(paths.table_dir / f"{dataset}_{model_cache_label(model_name)}_llr_fitness.csv"),
                    "torch_dtype": model_job_defaults(model_name)["torch_dtype"],
                }
            )
    task_table_path = output_root / "tables" / "clinprotgym_llr_tasks.csv"
    task_table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tasks).to_csv(task_table_path, index=False)
    if not tasks:
        print(f"No LLR tasks were created. Wrote empty task table: {task_table_path}")
        return
    payload = {"output_root": str(output_root), "tasks": tasks}
    job_root = output_root / "jobs" / "llr"
    payload_path = job_root / "clinprotgym_llr_payload.pkl"
    write_pickle(payload_path, payload)
    script_path = write_slurm_array_script(
        script_path=job_root / "submit_clinprotgym_llr_array.sh",
        payload_path=payload_path,
        subcommand="run-llr-task",
        n_tasks=len(tasks),
        job_name="clinpgym_llr",
        log_dir=job_root / "logs",
        partition=args.llr_partition,
        cpus_per_task=args.llr_cpus,
        mem=args.llr_mem,
        time=args.llr_time,
        max_active_tasks=args.max_active_llr_tasks,
        python_executable=args.python_executable,
    )
    print(f"Wrote LLR payload: {payload_path}")
    print(f"Wrote LLR array script: {script_path}")
    if args.submit:
        completed = subprocess.run(["sbatch", str(script_path)], check=True, capture_output=True, text=True)
        print(completed.stdout.strip())


def aa_token_id(tokenizer, amino_acid: str) -> int | None:
    encoded = tokenizer(amino_acid, add_special_tokens=False)
    token_ids = encoded.get("input_ids", [])
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def compute_llr_for_task(payload_path: Path, task_idx: int) -> dict:
    payload = read_pickle(payload_path)
    task = payload["tasks"][task_idx]
    output_root = Path(payload["output_root"])
    dataset = task["dataset"]
    model_name = task["model"]
    state = load_dataset_state(output_root, dataset)
    metadata_df = state["sequence_metadata"].copy()
    reference = state["reference_sequence"]

    tokenizer, model = esmDMS._load_embedding_model(model_name, torch_dtype=task.get("torch_dtype"))
    try:
        device = model.device
    except AttributeError:
        device = next(model.parameters()).device
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError(f"Tokenizer for {model_name} does not define a mask token.")
    aa_ids = {aa: aa_token_id(tokenizer, aa) for aa in AA_ALPHABET}

    base_inputs = tokenizer(
        reference,
        return_tensors="pt",
        add_special_tokens=True,
        return_special_tokens_mask=True,
    )
    special_mask = base_inputs.pop("special_tokens_mask").bool().squeeze(0)
    attention = base_inputs.get("attention_mask", torch.ones_like(base_inputs["input_ids"])).bool().squeeze(0)
    residue_positions = torch.where(attention & ~special_mask)[0]
    if len(residue_positions) != len(reference):
        raise ValueError(
            f"Residue-token count {len(residue_positions)} does not match reference length {len(reference)}."
        )
    base_inputs = {key: value.to(device) for key, value in base_inputs.items()}

    site_rows = []
    for position, site_df in metadata_df.groupby("position", sort=True):
        position = int(position)
        wt_aa = str(site_df["wt_aa"].iloc[0])
        mutant_aas = sorted({str(value) for value in site_df["mutant_aa"].dropna().unique()})
        wt_id = aa_ids.get(wt_aa)
        if wt_id is None:
            for mutant_aa in mutant_aas:
                site_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "position": position,
                        "wt_aa": wt_aa,
                        "mutant_aa": mutant_aa,
                        "logp_wt": np.nan,
                        "logp_mutant": np.nan,
                        "llr": np.nan,
                        "status": "missing_wt_token",
                    }
                )
            continue
        inputs = {key: value.clone() for key, value in base_inputs.items()}
        token_position = int(residue_positions[position - 1])
        inputs["input_ids"][0, token_position] = mask_token_id
        with torch.no_grad():
            logits = model(**inputs).logits[0, token_position].float()
            log_probs = torch.log_softmax(logits, dim=-1)
        logp_wt = float(log_probs[wt_id].detach().cpu())
        for mutant_aa in mutant_aas:
            mut_id = aa_ids.get(mutant_aa)
            if mutant_aa == wt_aa:
                logp_mut = logp_wt
                status = "ok"
            elif mut_id is None:
                logp_mut = np.nan
                status = "missing_mutant_token"
            else:
                logp_mut = float(log_probs[mut_id].detach().cpu())
                status = "ok"
            site_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "model_short": task["model_short"],
                    "position": position,
                    "wt_aa": wt_aa,
                    "mutant_aa": mutant_aa,
                    "logp_wt": logp_wt,
                    "logp_mutant": logp_mut,
                    "llr": logp_mut - logp_wt if np.isfinite(logp_mut) else np.nan,
                    "status": status,
                }
            )
    sites_df = pd.DataFrame(site_rows)
    Path(task["llr_sites_path"]).parent.mkdir(parents=True, exist_ok=True)
    sites_df.to_csv(task["llr_sites_path"], index=False)

    llr_lookup = {
        (int(row["position"]), str(row["mutant_aa"])): row["llr"]
        for _, row in sites_df.iterrows()
        if row.get("status") == "ok"
    }
    fitness_rows = []
    for _, row in metadata_df.iterrows():
        llr = 0.0 if bool(row["is_synonymous"]) else llr_lookup.get((int(row["position"]), str(row["mutant_aa"])), np.nan)
        fitness_rows.append({"SequenceIndex": str(row["SequenceIndex"]), "fitness": llr, "mutant": row["mutant"]})
    fitness_df = pd.DataFrame(fitness_rows)
    fitness_df.to_csv(task["llr_fitness_path"], index=False)
    result = {
        "task_idx": task_idx,
        "status": "ok",
        "dataset": dataset,
        "model": model_name,
        "llr_sites_path": task["llr_sites_path"],
        "llr_fitness_path": task["llr_fitness_path"],
        "n_sites": int(sites_df["position"].nunique()),
        "n_variants": int(len(fitness_df)),
    }
    status_dir = output_root / "jobs" / "llr" / "status"
    write_json(status_dir / f"{task_idx:04d}_{dataset}_{model_cache_label(model_name)}.json", result)
    return result


def run_payload_task(payload_path: Path, task_idx: int, runner_fn) -> None:
    try:
        result = runner_fn(payload_path, task_idx)
        print(json.dumps(result, indent=2, default=str))
    except Exception as exc:
        payload = read_pickle(payload_path)
        output_root = Path(payload.get("output_root", DEFAULT_OUTPUT_ROOT))
        failure = {
            "task_idx": task_idx,
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        status_dir = output_root / "jobs" / "failed_tasks"
        write_json(status_dir / f"{Path(payload_path).stem}_{task_idx:04d}_failed.json", failure)
        raise


def create_sae_jobs(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    tasks = []
    for dataset in command_datasets(args, output_root):
        state = load_dataset_state(output_root, dataset)
        for model_name in args.models:
            for layer in model_layers(model_name, args.layer_counts):
                tasks.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "model_short": model_short_name(model_name),
                        "layer": layer,
                        "method": ABSTRACTION_METHOD,
                        "embedding_type": EMBEDDING_TYPE,
                        "params": dict(BEST_SAE_PARAMS),
                        "run_label": BEST_SAE_PARAMS["run_label"],
                        "run_inference": bool(state.get("has_real_trajectory", False)),
                    }
                )
    job_root = output_root / "jobs" / "fixed_deltaembsae_layer_array"
    payload = {
        "output_root": str(output_root),
        "tasks": tasks,
        "force_recompute": bool(args.force_recompute),
    }
    payload_path = job_root / "clinprotgym_fixed_deltaembsae_layer_payload.pkl"
    write_pickle(payload_path, payload)
    script_path = write_slurm_array_script(
        script_path=job_root / "submit_clinprotgym_fixed_deltaembsae_layer_array.sh",
        payload_path=payload_path,
        subcommand="run-sae-task",
        n_tasks=len(tasks),
        job_name="clinpgym_sae",
        log_dir=job_root / "logs",
        partition=args.sae_partition,
        cpus_per_task=args.sae_cpus,
        mem=args.sae_mem,
        time=args.sae_time,
        max_active_tasks=args.max_active_sae_tasks,
        python_executable=args.python_executable,
    )
    task_df = pd.DataFrame(tasks)
    task_df.insert(0, "task_idx", range(len(task_df)))
    out = output_root / "tables" / "clinprotgym_fixed_deltaembsae_layer_tasks.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    task_df.to_csv(out, index=False)
    print(f"Wrote SAE payload: {payload_path}")
    print(f"Wrote SAE array script: {script_path}")
    if args.submit:
        completed = subprocess.run(["sbatch", str(script_path)], check=True, capture_output=True, text=True)
        print(completed.stdout.strip())


def run_sae_task(payload_path: Path, task_idx: int) -> dict:
    payload = read_pickle(payload_path)
    output_root = Path(payload["output_root"])
    task = payload["tasks"][task_idx]
    dataset = task["dataset"]
    model_name = task["model"]
    layer = task["layer"]
    paths = dataset_paths(output_root, dataset)
    sweep_dir = paths.job_dir / "fixed_sae_model_layer_array" / model_cache_label(model_name) / f"Layer_{layer}"
    output_run_root = sweep_dir / "runs"
    existing_result = completed_sae_task_result(
        sweep_dir / "task_result.json",
        expected_run_label=task.get("run_label"),
    )
    if existing_result is not None:
        existing_result = dict(existing_result)
        existing_result.update(
            {
                "task_idx": task_idx,
                "dataset": dataset,
                "model": model_name,
                "model_short": task["model_short"],
                "layer_index": layer,
                "model_label": f"{task['model_short']} Layer_{layer} {task['run_label']}",
                "skipped_existing_ok": True,
            }
        )
        return existing_result

    runner = runner_for_dataset(output_root, dataset, model_name)
    run_inference = bool(task.get("run_inference", bool(load_dataset_state(output_root, dataset).get("has_real_trajectory", False))))
    embedding_path = runner._embedding_path(layer, EMBEDDING_TYPE)
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Missing embedding cache for {dataset} {model_name} layer {layer}: {embedding_path}")

    sweep_dir.mkdir(parents=True, exist_ok=True)
    output_run_root.mkdir(parents=True, exist_ok=True)
    params = dict(task["params"])
    params["run_label"] = task["run_label"]
    single_payload = {
        "input_data": input_data_for_paths(paths),
        "config": {
            "embedding_model": model_name,
            "embedding_type": EMBEDDING_TYPE,
            "embedding_method": None,
            "local_or_disk": "both",
            "save_dir": str(paths.sequence_dir),
            "dataset_name": dataset,
        },
        "sequence_dataframe": runner.sequence_dataframe,
        "sequence_to_mutation_sites": runner.sequence_to_mutation_sites,
        "sequence_to_protein_sequence": runner.sequence_to_protein_sequence,
        "sequence_metadata": runner.sequence_metadata,
        "scores_dataframe": runner.scores_dataframe,
        "layer": layer,
        "method": ABSTRACTION_METHOD,
        "embedding_type": EMBEDDING_TYPE,
        "embedding_path": str(embedding_path),
        "output_root": str(output_run_root),
        "sweep_dir": str(sweep_dir),
        "configs": [{"run_label": task["run_label"], "params": params}],
        "gpus": 1,
        "max_parallel_runs": 1,
        "run_inference": run_inference,
        "force_recompute": bool(payload.get("force_recompute", False)),
        "require_cuda": False,
    }
    scratch_dir = None
    if os.environ.get("TMPDIR"):
        scratch_dir = (
            Path(os.environ["TMPDIR"])
            / "clinprotgym_sae_tasks"
            / safe_name(f"task_{task_idx:04d}_{dataset}_{model_cache_label(model_name)}_Layer_{layer}")
        )
    result = esmDMS._run_sae_sweep_config_safe(
        single_payload,
        config_idx=0,
        gpu_idx=None,
        scratch_dir=scratch_dir,
    )
    result.update(
        {
            "task_idx": task_idx,
            "dataset": dataset,
            "model": model_name,
            "model_short": task["model_short"],
            "layer_index": layer,
            "model_label": f"{task['model_short']} Layer_{layer} {task['run_label']}",
        }
    )
    result.update(h.reconstruction_metrics(result.get("viz_path")))
    pd.DataFrame([result]).to_csv(sweep_dir / "sae_sweep_results.csv", index=False)
    write_json(sweep_dir / "task_result.json", result)
    if result.get("status") != "ok":
        raise RuntimeError(f"SAE task failed for {dataset} {model_name} layer {layer}: {result.get('error')}")
    return result


def collect_sae(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    all_rows = []
    for dataset in command_datasets(args, output_root):
        paths = dataset_paths(output_root, dataset)
        result_paths = sorted((paths.job_dir / "fixed_sae_model_layer_array").glob("*/Layer_*/sae_sweep_results.csv"))
        rows = [pd.read_csv(path) for path in result_paths]
        dataset_df = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
        dataset_df.to_csv(paths.sae_metrics_path, index=False)
        all_rows.append(dataset_df)
    all_df = pd.concat([df for df in all_rows if not df.empty], ignore_index=True, sort=False) if all_rows else pd.DataFrame()
    out = output_root / "tables" / "clinprotgym_fixed_deltaembsae_layer_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out, index=False)
    print(f"Wrote SAE metrics: {out}")


def create_benchmark_jobs(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    tasks = []
    sae_metrics_path = output_root / "tables" / "clinprotgym_fixed_deltaembsae_layer_metrics.csv"
    progress(f"create-benchmark-jobs: reading SAE metrics from {sae_metrics_path}")
    sae_metrics_df = pd.read_csv(sae_metrics_path) if sae_metrics_path.is_file() else pd.DataFrame()
    progress(f"create-benchmark-jobs: loaded {len(sae_metrics_df)} SAE metric rows")
    datasets = command_datasets(args, output_root)
    progress(f"create-benchmark-jobs: building tasks for {len(datasets)} datasets")
    for dataset_idx, dataset in enumerate(datasets, start=1):
        dataset_task_start = len(tasks)
        progress(f"[{dataset_idx}/{len(datasets)}] {dataset}: loading processed state")
        state = load_dataset_state(output_root, dataset)
        has_trajectory = bool(state["has_real_trajectory"])
        progress(f"[{dataset_idx}/{len(datasets)}] {dataset}: has_real_trajectory={has_trajectory}")
        paths = dataset_paths(output_root, dataset)
        tasks.append({"dataset": dataset, "task_type": "functional_score", "model_label": "DMS functional score"})
        if has_trajectory:
            tasks.append({"dataset": dataset, "task_type": "enrichment_ratio", "model_label": "Enrichment ratio"})
            tasks.append({"dataset": dataset, "task_type": "popdms_substitution", "model_label": "Regular popDMS substitution"})
        for model_idx, model_name in enumerate(args.models, start=1):
            model_task_start = len(tasks)
            progress(f"[{dataset_idx}/{len(datasets)}] {dataset}: checking model {model_idx}/{len(args.models)} {model_name}")
            llr_path = paths.table_dir / f"{dataset}_{model_cache_label(model_name)}_llr_fitness.csv"
            if llr_path.is_file():
                tasks.append(
                    {
                        "dataset": dataset,
                        "task_type": "llr",
                        "model": model_name,
                        "model_short": model_short_name(model_name),
                        "fitness_source_path": str(llr_path),
                        "model_label": f"{model_short_name(model_name)} LLR",
                    }
                )
            if has_trajectory:
                runner = runner_for_dataset(output_root, dataset, model_name)
                layers = model_layers(model_name, args.layer_counts)
                progress(
                    f"[{dataset_idx}/{len(datasets)}] {dataset}: scanning {len(layers)} embedding layers for {model_name}"
                )
                for layer in layers:
                    embedding_path = runner._embedding_path(layer, EMBEDDING_TYPE)
                    if embedding_path.is_file():
                        tasks.append(
                            {
                                "dataset": dataset,
                                "task_type": "raw_embedding",
                                "model": model_name,
                                "model_short": model_short_name(model_name),
                                "layer": layer,
                                "feature_path": str(embedding_path),
                                "model_label": f"{model_short_name(model_name)} Layer_{layer} raw max_pool",
                            }
                        )
            progress(
                f"[{dataset_idx}/{len(datasets)}] {dataset}: added {len(tasks) - model_task_start} tasks for {model_name}"
            )
        if has_trajectory and not sae_metrics_df.empty:
            sub = sae_metrics_df[(sae_metrics_df["dataset"].eq(dataset)) & (sae_metrics_df["status"].eq("ok"))].copy()
            progress(f"[{dataset_idx}/{len(datasets)}] {dataset}: adding {len(sub)} successful SAE benchmark tasks")
            for _, row in sub.iterrows():
                tasks.append(
                    {
                        "dataset": dataset,
                        "task_type": "fixed_deltaembsae",
                        "model": row.get("model", ""),
                        "model_short": row.get("model_short", ""),
                        "layer": int(row.get("layer_index")),
                        "n_features": row.get("n_features", BEST_SAE_PARAMS["n_features"]),
                        "k": row.get("k", BEST_SAE_PARAMS["k"]),
                        "feature_path": row.get("feature_path", ""),
                        "inference_path": row.get("inference_path", ""),
                        "model_label": row.get("model_label", "Fixed DeltaEmbSAE"),
                    }
                )
        progress(f"[{dataset_idx}/{len(datasets)}] {dataset}: total added {len(tasks) - dataset_task_start} tasks")
    progress(f"create-benchmark-jobs: assigning output paths for {len(tasks)} tasks")
    for idx, task in enumerate(tasks):
        task["task_idx"] = idx
        safe_label = safe_name(f"{task['dataset']}_{task['model_label']}")[:120]
        paths = dataset_paths(output_root, task["dataset"])
        task["result_path"] = str(paths.job_dir / "benchmark_row_analysis" / "results" / f"{idx:05d}_{safe_label}.csv")
        task["fitness_path"] = str(paths.job_dir / "benchmark_row_analysis" / "fitness" / f"{idx:05d}_{safe_label}_fitness.csv")

    job_root = output_root / "jobs" / "benchmark_row_analysis"
    payload = {"output_root": str(output_root), "tasks": tasks, "force_recompute": bool(args.force_recompute)}
    payload_path = job_root / "clinprotgym_benchmark_row_analysis_payload.pkl"
    progress(f"create-benchmark-jobs: writing payload to {payload_path}")
    write_pickle(payload_path, payload)
    progress("create-benchmark-jobs: writing Slurm array script")
    script_path = write_slurm_array_script(
        script_path=job_root / "submit_clinprotgym_benchmark_row_analysis_array.sh",
        payload_path=payload_path,
        subcommand="run-benchmark-task",
        n_tasks=len(tasks),
        job_name="clinpgym_bench",
        log_dir=job_root / "logs",
        partition=args.benchmark_partition,
        cpus_per_task=args.benchmark_cpus,
        mem=args.benchmark_mem,
        time=args.benchmark_time,
        max_active_tasks=args.max_active_benchmark_tasks,
        python_executable=args.python_executable,
    )
    task_df = pd.DataFrame(tasks)
    out = output_root / "tables" / "clinprotgym_benchmark_tasks.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    progress(f"create-benchmark-jobs: writing task table to {out}")
    task_df.to_csv(out, index=False)
    print(f"Wrote benchmark payload: {payload_path}")
    print(f"Wrote benchmark array script: {script_path}")
    if args.submit:
        completed = subprocess.run(["sbatch", str(script_path)], check=True, capture_output=True, text=True)
        print(completed.stdout.strip())


def run_benchmark_task(payload_path: Path, task_idx: int) -> dict:
    payload = read_pickle(payload_path)
    output_root = Path(payload["output_root"])
    task = payload["tasks"][task_idx]
    dataset = task["dataset"]
    state = load_dataset_state(output_root, dataset)
    annotation_map = annotation_map_for_dataset(output_root, dataset)
    scores_df = state["scores_dataframe"]
    target_scores_df, spearman_target = spearman_target_for_state(state)
    sequence_df = state["sequence_dataframe"]
    metadata_df = state["sequence_metadata"]
    task_type = task["task_type"]
    row_extra = {}

    if task_type == "functional_score":
        fitness_df = scores_df[["SequenceIndex", "score"]].rename(columns={"score": "fitness"})
        method_family = "DMS functional score"
        benchmark = "DMS functional score"
        method = "functional score"
    elif task_type == "enrichment_ratio":
        fitness_df = enrichment_ratio_fitness(sequence_df)
        method_family = "Enrichment ratio baseline"
        benchmark = "Enrichment ratio"
        method = "enrichment ratio"
    elif task_type == "popdms_substitution":
        fitness_df, selection_df = popdms_substitution_fitness(sequence_df, metadata_df)
        selection_path = Path(task["fitness_path"]).with_name(Path(task["fitness_path"]).stem.replace("_fitness", "_selection") + ".csv")
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_df.to_csv(selection_path, index=False)
        gamma_diag = selection_df.attrs.get("gamma_diagnostics")
        if gamma_diag is not None:
            gamma_diag.to_csv(selection_path.with_name(selection_path.stem + "_gamma_diagnostics.csv"), index=False)
        selection_meta = selection_df.iloc[0] if not selection_df.empty else {}
        row_extra = {
            "gamma_opt": selection_meta.get("gamma_opt", np.nan),
            "mean_pairwise_pearson_r": selection_meta.get("mean_pairwise_pearson_r", np.nan),
            "n_replicates": selection_meta.get("n_replicates", np.nan),
            "n_replicate_pairs": selection_meta.get("n_replicate_pairs", np.nan),
            "selection_path": str(selection_path),
        }
        method_family = "popDMS baseline"
        benchmark = "Regular popDMS"
        method = "substitution popDMS"
    elif task_type == "llr":
        fitness_df = pd.read_csv(task["fitness_source_path"])[["SequenceIndex", "fitness"]]
        method_family = "LLR baseline"
        benchmark = "ESM-C LLR"
        method = "masked marginal LLR"
    elif task_type == "raw_embedding":
        runner = runner_for_dataset(output_root, dataset, task["model"])
        layer = int(task["layer"])
        inference_path = runner._inference_path("none", layer, NORM_SCHEME, EMBEDDING_TYPE)
        if payload.get("force_recompute") and inference_path.is_file():
            inference_path.unlink()
        result = runner.run_feature_inference(
            layer=layer,
            abstraction_method="none",
            abstraction_params={"norm_scheme": NORM_SCHEME},
            embedding_type=EMBEDDING_TYPE,
        )
        feature_path = runner._embedding_path(layer, EMBEDDING_TYPE)
        fitness_df = fitness_for_feature_mapping(runner.load_embeddings(layer, EMBEDDING_TYPE), result)
        method_family = "Raw embeddings"
        benchmark = "Raw embeddings"
        method = "raw ESM"
    elif task_type == "fixed_deltaembsae":
        feature_path = Path(task["feature_path"])
        inference_path = Path(task["inference_path"])
        if not inference_path.is_file() or payload.get("force_recompute"):
            seq_to_features_for_inference = read_pickle(feature_path)
            inference_sequence_df, inference_features = esmDMS._drop_missing_features(
                sequence_df,
                seq_to_features_for_inference,
                f"{dataset} fixed DeltaEmbSAE benchmark",
            )
            inference_result = mini_infer_esm(inference_sequence_df, inference_features)
            inference_path.parent.mkdir(parents=True, exist_ok=True)
            write_pickle(inference_path, inference_result)
        seq_to_features = read_pickle(feature_path)
        inference_result = read_pickle(inference_path)
        fitness_df = fitness_for_feature_mapping(seq_to_features, inference_result)
        method_family = "Raw SAE"
        benchmark = "Fixed DeltaEmbSAE"
        method = "DeltaEmbSAE"
    else:
        raise ValueError(f"Unsupported benchmark task_type: {task_type}")

    fitness_df = add_annotations(fitness_df, annotation_map)
    fitness_path = Path(task["fitness_path"])
    fitness_path.parent.mkdir(parents=True, exist_ok=True)
    fitness_df.to_csv(fitness_path, index=False)
    row = metrics_row(
        dataset=dataset,
        fitness_df=fitness_df,
        scores_df=target_scores_df,
        method_family=method_family,
        benchmark=benchmark,
        method=method,
        model=task.get("model", ""),
        model_short=task.get("model_short", ""),
        layer=task.get("layer", np.nan),
        model_label=task.get("model_label", method_family),
        embedding_type=EMBEDDING_TYPE if task_type in {"raw_embedding", "fixed_deltaembsae"} else "",
        feature_path=task.get("feature_path", ""),
        inference_path=task.get("inference_path", ""),
        fitness_path=fitness_path,
        n_features=task.get("n_features", np.nan),
        k=task.get("k", np.nan),
        spearman_target=spearman_target,
        extra=row_extra,
    )
    result_path = Path(task["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(result_path, index=False)
    return {"task_idx": task_idx, "status": "ok", "dataset": dataset, "task_type": task_type, "result_path": str(result_path)}


def collect_benchmarks(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    active_datasets = command_datasets(args, output_root)
    task_table_path = output_root / "tables" / "clinprotgym_benchmark_tasks.csv"
    task_table = pd.read_csv(task_table_path) if task_table_path.is_file() else pd.DataFrame()
    all_frames = []
    for dataset in active_datasets:
        paths = dataset_paths(output_root, dataset)
        if not task_table.empty and "result_path" in task_table.columns:
            result_paths = [
                Path(path)
                for path in task_table[task_table["dataset"].eq(dataset)]["result_path"].dropna().astype(str)
            ]
        else:
            result_paths = sorted((paths.job_dir / "benchmark_row_analysis" / "results").glob("*.csv"))
        result_paths = [path for path in result_paths if path.is_file()]
        frames = [pd.read_csv(path) for path in result_paths]
        dataset_df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        dataset_df.to_csv(paths.benchmark_metrics_path, index=False)
        if not dataset_df.empty:
            all_frames.append(dataset_df)
    all_df = pd.concat(all_frames, ignore_index=True, sort=False) if all_frames else pd.DataFrame()
    out = output_root / "tables" / "clinprotgym_method_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out, index=False)
    print(f"Wrote benchmark metrics: {out}")


def create_ensemble_jobs(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    tasks = []
    metrics_path = output_root / "tables" / "clinprotgym_method_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing benchmark metrics: {metrics_path}. Run collect-benchmarks first.")
    metrics_df = pd.read_csv(metrics_path)
    for dataset in command_datasets(args, output_root):
        state = load_dataset_state(output_root, dataset)
        if not state["has_real_trajectory"]:
            continue
        candidates = metrics_df[
            metrics_df["dataset"].eq(dataset)
            & metrics_df["benchmark"].eq("Fixed DeltaEmbSAE")
            & np.isfinite(pd.to_numeric(metrics_df["spearman_rho"], errors="coerce"))
        ].copy()
        if candidates.empty:
            continue
        candidates["spearman_rho"] = pd.to_numeric(candidates["spearman_rho"], errors="coerce")
        candidates["auc"] = pd.to_numeric(candidates["auc"], errors="coerce")
        candidates = candidates.sort_values(["spearman_rho", "auc"], ascending=[False, False]).head(args.ensemble_top_n)
        tasks.append(
            {
                "dataset": dataset,
                "aggregation": "rank_worst",
                "gamma": float(args.ensemble_gamma),
                "source_rows": candidates.to_dict(orient="records"),
                "method_label": f"Top{args.ensemble_top_n} SAE rank_worst gamma={args.ensemble_gamma:g}",
            }
        )
    job_root = output_root / "jobs" / "sae_ensemble_gamma1"
    payload = {"output_root": str(output_root), "tasks": tasks}
    payload_path = job_root / "clinprotgym_sae_ensemble_gamma1_payload.pkl"
    write_pickle(payload_path, payload)
    script_path = write_slurm_array_script(
        script_path=job_root / "submit_clinprotgym_sae_ensemble_gamma1_array.sh",
        payload_path=payload_path,
        subcommand="run-ensemble-task",
        n_tasks=len(tasks),
        job_name="clinpgym_ens",
        log_dir=job_root / "logs",
        partition=args.ensemble_partition,
        cpus_per_task=args.ensemble_cpus,
        mem=args.ensemble_mem,
        time=args.ensemble_time,
        max_active_tasks=args.max_active_ensemble_tasks,
        python_executable=args.python_executable,
    )
    task_df = pd.DataFrame([{k: v for k, v in task.items() if k != "source_rows"} for task in tasks])
    out = output_root / "tables" / "clinprotgym_sae_ensemble_tasks.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    task_df.to_csv(out, index=False)
    print(f"Wrote ensemble payload: {payload_path}")
    print(f"Wrote ensemble array script: {script_path}")
    if args.submit:
        completed = subprocess.run(["sbatch", str(script_path)], check=True, capture_output=True, text=True)
        print(completed.stdout.strip())


def run_ensemble_task(payload_path: Path, task_idx: int) -> dict:
    payload = read_pickle(payload_path)
    output_root = Path(payload["output_root"])
    task = payload["tasks"][task_idx]
    dataset = task["dataset"]
    state = load_dataset_state(output_root, dataset)
    annotation_map = annotation_map_for_dataset(output_root, dataset)
    sequence_df = state["sequence_dataframe"]
    target_scores_df, spearman_target = spearman_target_for_state(state)
    paths = dataset_paths(output_root, dataset)
    out_dir = paths.table_dir / "sae_ensemble_gamma1"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = pd.DataFrame(task["source_rows"])
    source_rows.to_csv(out_dir / f"{dataset}_ensemble_sources.csv", index=False)

    series = []
    component_rows = []
    for source_idx, row in source_rows.reset_index(drop=True).iterrows():
        feature_path = Path(row["feature_path"])
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing ensemble feature path: {feature_path}")
        seq_to_features = read_pickle(feature_path)
        inference_sequence_df, inference_features = esmDMS._drop_missing_features(
            sequence_df,
            seq_to_features,
            f"{dataset} ensemble source {source_idx}",
        )
        result = mini_infer_esm(inference_sequence_df, inference_features, gamma=float(task["gamma"]))
        component_consistency, component_pairs = mean_pairwise_pearson(result.s)
        fitness_df = fitness_for_feature_mapping(inference_features, result)
        comp_path = out_dir / f"{dataset}_source_{source_idx:02d}_gamma_{task['gamma']:g}_fitness.csv"
        fitness_df.to_csv(comp_path, index=False)
        name = f"{row.get('model_short', row.get('model', 'model'))}_L{int(float(row['layer']))}"
        series.append(pd.Series(fitness_df["fitness"].to_numpy(dtype=float), index=fitness_df["SequenceIndex"].astype(str), name=name))
        component_rows.append(
            {
                "source_idx": source_idx,
                "model": row.get("model", ""),
                "model_short": row.get("model_short", ""),
                "layer": row.get("layer", np.nan),
                "gamma": float(task["gamma"]),
                "gamma_opt": float(result.gamma_opt),
                "mean_pairwise_pearson_r": component_consistency,
                "n_replicate_pairs": component_pairs,
                "fitness_path": str(comp_path),
            }
        )
    score_matrix = pd.concat(series, axis=1)
    ensemble_fitness = h.rank_ensemble_fitness_from_score_matrix(score_matrix, aggregation=task["aggregation"])
    ensemble_fitness = add_annotations(ensemble_fitness, annotation_map)
    fitness_path = out_dir / f"{dataset}_top{len(series)}_sae_{task['aggregation']}_gamma_{task['gamma']:g}_fitness.csv"
    ensemble_fitness.to_csv(fitness_path, index=False)
    pd.DataFrame(component_rows).to_csv(out_dir / f"{dataset}_ensemble_component_fitness_paths.csv", index=False)
    row = metrics_row(
        dataset=dataset,
        fitness_df=ensemble_fitness,
        scores_df=target_scores_df,
        method_family="Ensemble SAE model",
        benchmark="ESMC Fixed DeltaEmbSAE rank ensemble",
        method="DeltaEmbSAE rank ensemble",
        model_label=task["method_label"],
        embedding_type=EMBEDDING_TYPE,
        fitness_path=fitness_path,
        n_features=np.nan,
        k=np.nan,
        spearman_target=spearman_target,
        extra={
            "aggregation": task["aggregation"],
            "gamma": float(task["gamma"]),
            "n_component_model_layers": int(len(series)),
            "mean_component_pairwise_pearson_r": (
                float(np.nanmean([row["mean_pairwise_pearson_r"] for row in component_rows]))
                if component_rows
                else np.nan
            ),
        },
    )
    metrics_path = out_dir / f"{dataset}_top{len(series)}_sae_{task['aggregation']}_gamma_{task['gamma']:g}_metrics.csv"
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    return {"task_idx": task_idx, "status": "ok", "dataset": dataset, "metrics_path": str(metrics_path)}


def collect_ensembles(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    frames = []
    for dataset in command_datasets(args, output_root):
        paths = dataset_paths(output_root, dataset)
        metric_paths = sorted((paths.table_dir / "sae_ensemble_gamma1").glob("*_metrics.csv"))
        dataset_df = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True, sort=False) if metric_paths else pd.DataFrame()
        dataset_df.to_csv(paths.ensemble_metrics_path, index=False)
        if not dataset_df.empty:
            frames.append(dataset_df)
    all_df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    out = output_root / "tables" / "clinprotgym_sae_ensemble_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out, index=False)
    print(f"Wrote ensemble metrics: {out}")


def method_plot_label(method_family: object) -> str:
    labels = {
        "Enrichment ratio baseline": "Enrichment ratio",
        "popDMS baseline": "popDMS",
        "LLR baseline": "LLR",
        "Raw embeddings": "Raw ESM",
        "Raw SAE": "Single SAE",
        "Ensemble SAE model": "SAE ensemble",
    }
    return labels.get(str(method_family), str(method_family))


def review_star_cutoff_label(min_stars: int) -> str:
    if int(min_stars) <= 0:
        return "all binary"
    suffix = "star" if int(min_stars) == 1 else "stars"
    return f">={int(min_stars)} {suffix}"


def auc_discrimination_value(auc: object) -> float:
    auc_value = pd.to_numeric(pd.Series([auc]), errors="coerce").iloc[0]
    if not np.isfinite(auc_value):
        return np.nan
    return float(max(auc_value, 1.0 - auc_value))


def auc_direction_flipped(auc: object) -> bool:
    auc_value = pd.to_numeric(pd.Series([auc]), errors="coerce").iloc[0]
    return bool(np.isfinite(auc_value) and auc_value < 0.5)


def best_auc_rows_by_dataset_method(metrics_df: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    rows = []
    if metrics_df.empty or "fitness_path" not in metrics_df.columns:
        return pd.DataFrame()
    candidate_df = metrics_df[metrics_df["method_family"].isin(method_order)].copy()
    candidate_df = candidate_df[candidate_df["fitness_path"].notna() & candidate_df["fitness_path"].astype(str).str.len().gt(0)]
    if candidate_df.empty:
        return pd.DataFrame()
    candidate_df["auc"] = pd.to_numeric(candidate_df.get("auc", np.nan), errors="coerce")
    candidate_df["auc_discrimination"] = candidate_df["auc"].map(auc_discrimination_value)
    candidate_df["auc_flipped"] = candidate_df["auc"].map(auc_direction_flipped)
    candidate_df["spearman_rho"] = pd.to_numeric(candidate_df.get("spearman_rho", np.nan), errors="coerce")
    candidate_df["layer_numeric"] = pd.to_numeric(candidate_df.get("layer", np.nan), errors="coerce")
    layer_method = candidate_df["method_family"].isin(["Raw embeddings", "Raw SAE"])
    candidate_df = candidate_df[~(layer_method & candidate_df["layer_numeric"].eq(0))].copy()
    for (_, _), group in candidate_df.groupby(["dataset", "method_family"], sort=False):
        finite = group[np.isfinite(group["auc_discrimination"])].copy()
        if finite.empty:
            continue
        rows.append(finite.sort_values(["auc_discrimination", "spearman_rho"], ascending=[False, False]).iloc[0])
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=candidate_df.columns)


def auc_by_review_star_cutoff_rows(
    output_root: Path,
    dataset: str,
    selected_rows: pd.DataFrame,
) -> pd.DataFrame:
    if selected_rows.empty:
        return pd.DataFrame()
    annotations = load_dataset_state(output_root, dataset)["annotations_dataframe"].copy()
    annotations["SequenceIndex"] = annotations["SequenceIndex"].astype(str)
    annotations["stars"] = pd.to_numeric(annotations.get("stars", 0), errors="coerce").fillna(0).astype(int)
    annotation_lookup = dict(zip(annotations["SequenceIndex"], annotations["annotation"]))
    star_lookup = dict(zip(annotations["SequenceIndex"], annotations["stars"]))

    rows = []
    for _, row in selected_rows.iterrows():
        fitness_path = row.get("fitness_path", "")
        if pd.isna(fitness_path) or not str(fitness_path):
            continue
        path = Path(str(fitness_path))
        if not path.is_file():
            continue
        try:
            fitness_df = pd.read_csv(path)
        except Exception:
            continue
        if not {"SequenceIndex", "fitness"}.issubset(fitness_df.columns):
            continue
        fitness_df["SequenceIndex"] = fitness_df["SequenceIndex"].astype(str)
        fitness_df["fitness"] = pd.to_numeric(fitness_df["fitness"], errors="coerce")
        annotation_scheme = str(row.get("annotation_scheme", "")).strip().lower()
        has_embedded_annotations = {"annotation", "stars"}.issubset(fitness_df.columns)
        if annotation_scheme == "hgvs" and has_embedded_annotations:
            fitness_df["stars"] = pd.to_numeric(fitness_df["stars"], errors="coerce").fillna(0).astype(int)
        else:
            fitness_df["annotation"] = fitness_df["SequenceIndex"].map(annotation_lookup)
            fitness_df["stars"] = fitness_df["SequenceIndex"].map(star_lookup).fillna(0).astype(int)

        for min_stars in CLINVAR_REVIEW_STAR_CUTOFFS:
            cutoff_df = fitness_df if min_stars <= 0 else fitness_df[fitness_df["stars"].ge(min_stars)].copy()
            auc_metrics = classification_metrics_for_fitness(cutoff_df)
            auc_metrics["auc_discrimination"] = auc_discrimination_value(auc_metrics["auc"])
            auc_metrics["auc_flipped"] = auc_direction_flipped(auc_metrics["auc"])
            rows.append(
                {
                    "dataset": dataset,
                    "method_family": row.get("method_family", ""),
                    "method_plot_label": method_plot_label(row.get("method_family", "")),
                    "model_label": row.get("model_label", ""),
                    "benchmark": row.get("benchmark", ""),
                    "method": row.get("method", ""),
                    "model_short": row.get("model_short", ""),
                    "layer": row.get("layer", np.nan),
                    "review_cutoff": review_star_cutoff_label(min_stars),
                    "min_review_stars": int(min_stars),
                    "fitness_path": str(path),
                    "annotation_scheme": row.get("annotation_scheme", "protein"),
                    **auc_metrics,
                }
            )
    return pd.DataFrame(rows)


def write_auc_by_review_star_cutoff_plots(
    output_root: Path,
    auc_selected_rows: pd.DataFrame,
    active_dataset_list: list[str],
    method_order: list[str],
) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    written: list[Path] = []
    if auc_selected_rows.empty:
        return written

    global_rows = []
    method_label_order = [method_plot_label(value) for value in method_order]
    cutoff_order = [review_star_cutoff_label(value) for value in CLINVAR_REVIEW_STAR_CUTOFFS]

    for dataset in active_dataset_list:
        dataset_rows = auc_selected_rows[auc_selected_rows["dataset"].astype(str).eq(dataset)].copy()
        if dataset_rows.empty:
            continue
        paths = dataset_paths(Path(output_root), dataset)
        table_df = auc_by_review_star_cutoff_rows(Path(output_root), dataset, dataset_rows)
        if table_df.empty:
            continue
        global_rows.append(table_df)

        table_path = paths.table_dir / f"{dataset}_best_method_auc_by_clinvar_review_stars.csv"
        table_df.to_csv(table_path, index=False)

        plot_df = table_df[np.isfinite(pd.to_numeric(table_df["auc_discrimination"], errors="coerce"))].copy()
        if plot_df.empty:
            continue
        plot_df["auc_discrimination"] = pd.to_numeric(plot_df["auc_discrimination"], errors="coerce")
        plot_df["review_cutoff"] = pd.Categorical(plot_df["review_cutoff"], categories=cutoff_order, ordered=True)
        present_labels = set(plot_df["method_plot_label"].dropna().astype(str))
        hue_order = [label for label in method_label_order if label in present_labels]

        progress(f"summarize: plotting {dataset}_best_method_auc_by_clinvar_review_stars.png")
        fig_width = max(8.8, 1.35 * len(cutoff_order) + 3.5)
        fig, ax = plt.subplots(figsize=(fig_width, 5.4))
        sns.barplot(
            data=plot_df.sort_values(["review_cutoff", "method_plot_label"]),
            x="review_cutoff",
            y="auc_discrimination",
            hue="method_plot_label",
            order=cutoff_order,
            hue_order=hue_order or None,
            ax=ax,
        )
        ax.axhline(0.5, color="0.65", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("ClinVar review-star cutoff")
        ax.set_ylabel("Direction-normalized pathogenic-vs-benign AUC")
        ax.set_title(f"{dataset}: best non-layer-0 method AUC by ClinVar review-star cutoff")
        ax.tick_params(axis="x", rotation=25)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        sns.despine(ax=ax)
        fig.tight_layout()
        figure_path = paths.figure_dir / f"{dataset}_best_method_auc_by_clinvar_review_stars.png"
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(figure_path)

    if global_rows:
        global_path = Path(output_root) / "tables" / "clinprotgym_best_method_auc_by_clinvar_review_stars.csv"
        pd.concat(global_rows, ignore_index=True, sort=False).to_csv(global_path, index=False)

    return written


def write_summary_plots(
    output_root: Path,
    metrics_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    method_order: list[str],
    active_dataset_list: list[str],
) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    progress("summarize: writing summary plots")
    figure_dir = Path(output_root) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    sns.set_theme(style="whitegrid")

    selected_plot = selected_df.copy()
    if not selected_plot.empty:
        selected_plot["method_plot_label"] = selected_plot["method_family"].map(method_plot_label)
    auc_selected_plot = best_auc_rows_by_dataset_method(metrics_df, method_order)
    if not auc_selected_plot.empty:
        auc_selected_plot["method_plot_label"] = auc_selected_plot["method_family"].map(method_plot_label)

    method_labels_present = set()
    if not selected_plot.empty:
        method_labels_present.update(selected_plot["method_plot_label"].dropna().astype(str))
    if not auc_selected_plot.empty:
        method_labels_present.update(auc_selected_plot["method_plot_label"].dropna().astype(str))
    method_label_order = [method_plot_label(value) for value in method_order if method_plot_label(value) in method_labels_present]

    plot_dataset_values = set()
    if not selected_plot.empty:
        plot_dataset_values.update(selected_plot["dataset"].dropna().astype(str))
    if not auc_selected_plot.empty:
        plot_dataset_values.update(auc_selected_plot["dataset"].dropna().astype(str))
    dataset_order = [dataset for dataset in active_dataset_list if dataset in plot_dataset_values]
    if not dataset_order:
        dataset_order = sorted(plot_dataset_values)

    if not selected_plot.empty:
        plot_df = selected_plot[np.isfinite(pd.to_numeric(selected_plot["spearman_rho"], errors="coerce"))].copy()
        if not plot_df.empty:
            progress("summarize: plotting clinprotgym_best_method_spearman_by_dataset.png")
            plot_df["spearman_rho"] = pd.to_numeric(plot_df["spearman_rho"], errors="coerce")
            fig_width = max(10.0, 1.2 * len(dataset_order) + 3.2)
            fig, ax = plt.subplots(figsize=(fig_width, 6.0))
            sns.barplot(
                data=plot_df,
                x="dataset",
                y="spearman_rho",
                hue="method_plot_label",
                order=dataset_order,
                hue_order=method_label_order or None,
                ax=ax,
            )
            ax.axhline(0, color="0.75", linewidth=1.0, zorder=0)
            ax.set_xlabel("Dataset")
            ax.set_ylabel("Spearman rho")
            ax.set_title("Best method per family by dataset: Spearman rho")
            ax.tick_params(axis="x", rotation=55)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
            ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
            sns.despine(ax=ax)
            fig.tight_layout()
            path = figure_dir / "clinprotgym_best_method_spearman_by_dataset.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            written.append(path)

    if not auc_selected_plot.empty:
        plot_df = auc_selected_plot[
            np.isfinite(pd.to_numeric(auc_selected_plot["auc_discrimination"], errors="coerce"))
        ].copy()
        if not plot_df.empty:
            progress("summarize: plotting clinprotgym_best_method_auc_by_dataset.png")
            plot_df["auc_discrimination"] = pd.to_numeric(plot_df["auc_discrimination"], errors="coerce")
            fig_width = max(10.0, 1.2 * len(dataset_order) + 3.2)
            fig, ax = plt.subplots(figsize=(fig_width, 6.0))
            sns.barplot(
                data=plot_df,
                x="dataset",
                y="auc_discrimination",
                hue="method_plot_label",
                order=dataset_order,
                hue_order=method_label_order or None,
                ax=ax,
            )
            ax.axhline(0.5, color="0.65", linewidth=1.0, linestyle="--", zorder=0)
            ax.set_ylim(0.0, 1.02)
            ax.set_xlabel("Dataset")
            ax.set_ylabel("Direction-normalized ClinVar AUC")
            ax.set_title("Best non-layer-0 method per family by dataset: ClinVar AUC")
            ax.tick_params(axis="x", rotation=55)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
            ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
            sns.despine(ax=ax)
            fig.tight_layout()
            path = figure_dir / "clinprotgym_best_method_auc_by_dataset.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            written.append(path)

        scatter_df = auc_selected_plot[
            np.isfinite(pd.to_numeric(auc_selected_plot["spearman_rho"], errors="coerce"))
            & np.isfinite(pd.to_numeric(auc_selected_plot["auc_discrimination"], errors="coerce"))
        ].copy()
        if not scatter_df.empty:
            progress("summarize: plotting clinprotgym_best_method_spearman_vs_auc.png")
            scatter_df["spearman_rho"] = pd.to_numeric(scatter_df["spearman_rho"], errors="coerce")
            scatter_df["auc_discrimination"] = pd.to_numeric(scatter_df["auc_discrimination"], errors="coerce")
            fig, ax = plt.subplots(figsize=(8.2, 6.4))
            sns.scatterplot(
                data=scatter_df,
                x="spearman_rho",
                y="auc_discrimination",
                hue="method_plot_label",
                style="dataset",
                hue_order=method_label_order or None,
                s=95,
                edgecolor="white",
                linewidth=0.7,
                ax=ax,
            )
            ax.axvline(0, color="0.80", linewidth=1.0, zorder=0)
            ax.set_xlabel("Spearman rho")
            ax.set_ylabel("Direction-normalized ClinVar AUC")
            ax.set_title("Best non-layer-0 method per family: Spearman rho vs ClinVar AUC")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
            sns.despine(ax=ax)
            fig.tight_layout()
            path = figure_dir / "clinprotgym_best_method_spearman_vs_auc.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            written.append(path)

    if not selected_plot.empty:
        heatmap_df = selected_plot.copy()
        heatmap_df["spearman_rho"] = pd.to_numeric(heatmap_df["spearman_rho"], errors="coerce")
        heatmap_df = heatmap_df[np.isfinite(heatmap_df["spearman_rho"])]
        if not heatmap_df.empty:
            progress("summarize: plotting clinprotgym_best_method_spearman_heatmap.png")
            pivot = heatmap_df.pivot_table(
                index="dataset",
                columns="method_plot_label",
                values="spearman_rho",
                aggfunc="first",
            )
            pivot = pivot.reindex(index=dataset_order)
            pivot = pivot[[col for col in method_label_order if col in pivot.columns]]
            fig_width = max(7.5, 0.75 * len(pivot.columns) + 3.5)
            fig_height = max(4.8, 0.45 * len(pivot.index) + 2.0)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="vlag",
                center=0,
                linewidths=0.4,
                linecolor="white",
                cbar_kws={"label": "Spearman rho"},
                ax=ax,
            )
            ax.set_xlabel("Method")
            ax.set_ylabel("Dataset")
            ax.set_title("Best method per family: Spearman rho heatmap")
            fig.tight_layout()
            path = figure_dir / "clinprotgym_best_method_spearman_heatmap.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            written.append(path)

    summary_plot = summary_df.copy()
    if not summary_plot.empty:
        summary_plot["method_plot_label"] = summary_plot["method_family"].map(method_plot_label)
        summary_plot = summary_plot[
            np.isfinite(pd.to_numeric(summary_plot["mean_spearman_rho"], errors="coerce"))
        ].copy()
        if not summary_plot.empty:
            progress("summarize: plotting clinprotgym_average_spearman_by_method.png")
            summary_plot["mean_spearman_rho"] = pd.to_numeric(summary_plot["mean_spearman_rho"], errors="coerce")
            summary_plot["sem_spearman_rho"] = pd.to_numeric(summary_plot["sem_spearman_rho"], errors="coerce").fillna(0.0)
            summary_order = [method_plot_label(value) for value in method_order if method_plot_label(value) in set(summary_plot["method_plot_label"])]
            plot_df = summary_plot.set_index("method_plot_label").reindex(summary_order).dropna(subset=["mean_spearman_rho"]).reset_index()
            fig, ax = plt.subplots(figsize=(9.0, 5.2))
            ax.bar(
                plot_df["method_plot_label"],
                plot_df["mean_spearman_rho"],
                yerr=plot_df["sem_spearman_rho"],
                color=sns.color_palette("tab10", n_colors=len(plot_df)),
                capsize=3,
            )
            ax.axhline(0, color="0.75", linewidth=1.0, zorder=0)
            ax.set_xlabel("Method")
            ax.set_ylabel("Mean Spearman rho across datasets")
            ax.set_title("Average best-method Spearman rho")
            ax.tick_params(axis="x", rotation=35)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
            sns.despine(ax=ax)
            fig.tight_layout()
            path = figure_dir / "clinprotgym_average_spearman_by_method.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            written.append(path)

    layer_df = metrics_df[metrics_df["method_family"].isin(["Raw embeddings", "Raw SAE"])].copy()
    if not layer_df.empty and {"dataset", "method_family", "layer", "model_short", "spearman_rho"}.issubset(layer_df.columns):
        layer_df["layer"] = pd.to_numeric(layer_df["layer"], errors="coerce")
        layer_df["spearman_rho"] = pd.to_numeric(layer_df["spearman_rho"], errors="coerce")
        layer_df = layer_df[np.isfinite(layer_df["layer"]) & np.isfinite(layer_df["spearman_rho"])].copy()
        if not layer_df.empty:
            progress("summarize: plotting clinprotgym_layer_spearman_by_dataset_model.png")
            layer_df["method_plot_label"] = layer_df["method_family"].map(method_plot_label)
            layer_df["dataset"] = pd.Categorical(layer_df["dataset"].astype(str), categories=active_dataset_list, ordered=True)
            grid = sns.relplot(
                data=layer_df.sort_values(["dataset", "method_plot_label", "model_short", "layer"]),
                x="layer",
                y="spearman_rho",
                hue="model_short",
                style="method_plot_label",
                col="dataset",
                col_wrap=2,
                kind="line",
                marker="o",
                height=3.7,
                aspect=1.55,
                facet_kws={"sharey": False, "sharex": True},
            )
            grid.set_axis_labels("ESM-C layer", "Spearman rho")
            grid.set_titles("{col_name}")
            grid.figure.suptitle("Layer-wise benchmark performance", y=1.02)
            grid.figure.tight_layout()
            path = figure_dir / "clinprotgym_layer_spearman_by_dataset_model.png"
            grid.figure.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(grid.figure)
            written.append(path)

    if not auc_selected_plot.empty:
        written.extend(
            write_auc_by_review_star_cutoff_plots(
                Path(output_root),
                auc_selected_plot,
                active_dataset_list,
                method_order,
            )
        )

    best_fitness_rows = []
    if not auc_selected_plot.empty and "fitness_path" in auc_selected_plot.columns:
        for dataset, group in auc_selected_plot.groupby("dataset", sort=False):
            finite = group[np.isfinite(pd.to_numeric(group["auc_discrimination"], errors="coerce"))].copy()
            if finite.empty:
                finite = group[np.isfinite(pd.to_numeric(group["spearman_rho"], errors="coerce"))].copy()
            if finite.empty:
                continue
            finite["spearman_rho"] = pd.to_numeric(finite["spearman_rho"], errors="coerce")
            finite["auc"] = pd.to_numeric(finite["auc"], errors="coerce")
            finite["auc_discrimination"] = pd.to_numeric(finite["auc_discrimination"], errors="coerce")
            best_fitness_rows.append(
                finite.sort_values(["auc_discrimination", "spearman_rho"], ascending=[False, False]).iloc[0]
            )
    distribution_panels = []
    for row in best_fitness_rows:
        fitness_path = row.get("fitness_path", "")
        if pd.isna(fitness_path) or not str(fitness_path):
            continue
        path = Path(str(fitness_path))
        if not path.is_file():
            continue
        try:
            fitness_df = pd.read_csv(path)
        except Exception:
            continue
        if not {"fitness", "annotation"}.issubset(fitness_df.columns):
            continue
        fitness_df = fitness_df[fitness_df["annotation"].isin(PATHOGENICITY_LABELS)].copy()
        fitness_df["fitness"] = pd.to_numeric(fitness_df["fitness"], errors="coerce")
        fitness_df = fitness_df[np.isfinite(fitness_df["fitness"])]
        if fitness_df.empty:
            continue
        distribution_panels.append((row, fitness_df))
    if distribution_panels:
        progress("summarize: plotting clinprotgym_best_method_fitness_distributions_by_clinvar.png")
        ncols = 2
        nrows = int(math.ceil(len(distribution_panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.9 * nrows), squeeze=False)
        axes_flat = axes.ravel()
        palette = {"benign": "#2c7fb8", "pathogenic": "#d7301f"}
        for ax, (row, fitness_df) in zip(axes_flat, distribution_panels):
            values = fitness_df["fitness"].to_numpy(dtype=float)
            if np.nanmin(values) == np.nanmax(values):
                center = float(np.nanmin(values))
                bins = np.array([center - 0.5, center + 0.5])
            else:
                bins = np.histogram_bin_edges(values, bins=min(24, max(8, int(np.sqrt(len(values))))))
            for annotation in ["benign", "pathogenic"]:
                subset = fitness_df.loc[fitness_df["annotation"].eq(annotation), "fitness"].to_numpy(dtype=float)
                if len(subset) == 0:
                    continue
                ax.hist(
                    subset,
                    bins=bins,
                    color=palette[annotation],
                    alpha=0.55,
                    label=f"{annotation} (n={len(subset)})",
                    edgecolor="white",
                    linewidth=0.35,
                )
            ax.set_title(
                f"{row.get('dataset')}\n"
                f"{method_plot_label(row.get('method_family'))}: "
                f"AUC*={float(row.get('auc_discrimination', np.nan)):.3f}",
                fontsize=10,
            )
            ax.set_xlabel("Fitness")
            ax.set_ylabel("Variant count")
            ax.legend(fontsize=8)
            ax.grid(color="0.90", linewidth=0.65)
        for ax in axes_flat[len(distribution_panels):]:
            ax.axis("off")
        fig.suptitle("Best method fitness distributions by ClinVar annotation", y=1.02)
        fig.tight_layout()
        path = figure_dir / "clinprotgym_best_method_fitness_distributions_by_clinvar.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written


def summarize(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    progress("summarize: resolving active datasets")
    active_dataset_list = filter_sahu_brca2_final_datasets(
        command_datasets(args, output_root),
        getattr(args, "sahu_brca2_final", "both"),
    )
    active_datasets = set(active_dataset_list)
    frames = []
    progress("summarize: reading method and ensemble metric tables")
    for path in [
        output_root / "tables" / "clinprotgym_method_metrics.csv",
        output_root / "tables" / "clinprotgym_sae_ensemble_metrics.csv",
    ]:
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No method or ensemble metrics were found to summarize.")
    metrics_df = pd.concat(frames, ignore_index=True, sort=False)
    if "dataset" in metrics_df.columns:
        metrics_df = metrics_df[metrics_df["dataset"].astype(str).isin(active_datasets)].copy()
    progress(f"summarize: loaded {len(metrics_df)} metric rows for {len(active_datasets)} active datasets")
    wanted = [
        "Enrichment ratio baseline",
        "popDMS baseline",
        "LLR baseline",
        "Raw embeddings",
        "Raw SAE",
        "Ensemble SAE model",
    ]
    metrics_df = metrics_df[metrics_df["method_family"].isin(wanted)].copy()
    for col in ["spearman_rho", "auc"]:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")
    if "spearman_target" not in metrics_df.columns:
        metrics_df["spearman_target"] = ""

    selected_rows = []
    progress("summarize: selecting best row per dataset and method family")
    for (dataset, family), group in metrics_df.groupby(["dataset", "method_family"], sort=False):
        finite = group[np.isfinite(group["spearman_rho"])].copy()
        if finite.empty:
            finite = group[np.isfinite(group["auc"])].copy()
        if finite.empty:
            selected_rows.append(group.iloc[0])
            continue
        selected_rows.append(finite.sort_values(["spearman_rho", "auc"], ascending=[False, False]).iloc[0])
    selected_df = pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame(columns=metrics_df.columns)
    summary_rows = []
    for family in wanted:
        group = selected_df[selected_df["method_family"].eq(family)]
        summary_rows.append(
            {
                "method_family": family,
                "n_datasets": int(group["dataset"].nunique()),
                "mean_spearman_rho": float(group["spearman_rho"].mean(skipna=True)) if not group.empty else np.nan,
                "sem_spearman_rho": float(group["spearman_rho"].sem(skipna=True)) if len(group) > 1 else np.nan,
                "mean_auc": float(group["auc"].mean(skipna=True)) if not group.empty else np.nan,
                "sem_auc": float(group["auc"].sem(skipna=True)) if len(group) > 1 else np.nan,
                "finite_spearman_datasets": int(np.isfinite(group["spearman_rho"]).sum()),
                "finite_auc_datasets": int(np.isfinite(group["auc"]).sum()),
                "spearman_targets": ", ".join(
                    sorted(target for target in group["spearman_target"].dropna().astype(str).unique() if target)
                ),
            }
        )
    out_dir = output_root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = out_dir / "clinprotgym_best_method_rows_by_dataset.csv"
    summary_path = out_dir / "clinprotgym_average_spearman_auc_summary.csv"
    selected_df.to_csv(selected_path, index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    figure_paths = write_summary_plots(output_root, metrics_df, selected_df, summary_df, wanted, active_dataset_list)
    print(f"Wrote selected rows: {selected_path}")
    print(f"Wrote average Spearman/AUC summary: {summary_path}")
    for path in figure_paths:
        print(f"Wrote summary figure: {path}")


def cross_replicate_consistency(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    datasets = filter_sahu_brca2_final_datasets(
        command_datasets(args, output_root),
        getattr(args, "sahu_brca2_final", "both"),
    )
    consistency_df, plot_df, table_path, figure_path = write_cross_replicate_consistency_outputs(
        output_root,
        datasets=datasets,
        include_all_datasets=not args.omit_empty_datasets,
    )
    print(f"Wrote cross-replicate consistency table: {table_path}")
    if plot_df.empty:
        print("No finite cross-replicate consistency rows were available to plot.")
    else:
        print(f"Wrote cross-replicate consistency figure: {figure_path}")
        print(
            consistency_df[np.isfinite(consistency_df["cross_replicate_consistency"])]
            .sort_values(["dataset", "cross_replicate_consistency"], ascending=[True, False])[
                ["dataset", "method_label", "cross_replicate_consistency", "n_replicate_pairs", "selection_gamma"]
            ]
            .to_string(index=False)
        )


def preparation_specs_for_csv(raw_csv: Path) -> list[dict[str, str]]:
    dataset = safe_name(Path(raw_csv).stem)
    if dataset == SAHU_BRCA2_BASE_DATASET:
        return [
            {"dataset_name": SAHU_BRCA2_LAST2048_DATASET, "sequence_transform": "last2048"},
            {"dataset_name": SAHU_BRCA2_WINDOW_DATASET, "sequence_transform": "sliding2048_overlap1024"},
        ]
    return [{"dataset_name": dataset, "sequence_transform": "full"}]


def prepare(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    selected = parse_dataset_selection(args.datasets)
    csvs = iter_input_csvs(input_dir, selected)
    if not csvs:
        raise FileNotFoundError(f"No CSV files matched under {input_dir}.")
    rows = []
    for raw_csv in csvs:
        for spec in preparation_specs_for_csv(raw_csv):
            paths = prepare_dataset(
                raw_csv,
                output_root,
                drop_stop=not args.keep_stop,
                force=args.force,
                dataset_name=spec["dataset_name"],
                sequence_transform=spec["sequence_transform"],
            )
            summary = pd.read_csv(paths.processing_summary_path).iloc[0].to_dict()
            rows.append(summary)
            print(f"Prepared {paths.dataset}: {paths.state_path}")
    manifest_path = output_root / "tables" / "clinprotgym_processing_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Wrote processing manifest: {manifest_path}")


def parse_model_layer_counts(values: list[str] | None) -> dict[str, int]:
    counts = dict(MODEL_LAYER_COUNTS)
    if not values:
        return counts
    for value in values:
        if "=" not in value:
            raise ValueError("--model-layer-count values must look like MODEL=N")
        model, count = value.split("=", 1)
        counts[model] = int(count)
    return counts


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Pipeline output root.")
    parser.add_argument("--datasets", nargs="*", help="Dataset stems/files to include.")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Embedding model ids.")
    parser.add_argument("--model-layer-count", action="append", help="Register an extra model layer count as MODEL=N.")


def add_sahu_brca2_final_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sahu-brca2-final",
        choices=["both", "last2048", "sliding-window", "none"],
        default="both",
        help=(
            "Which derived Sahu BRCA2 ESM-C context analysis to include in final summary/plot commands. "
            "Both derived datasets can still be calculated before this filter is applied."
        ),
    )


def add_count_dataset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--count-datasets-only",
        action="store_true",
        help="Restrict this analysis command to datasets with real trajectory/count data in the processed state.",
    )


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if hasattr(args, "model_layer_count"):
        args.layer_counts = parse_model_layer_counts(args.model_layer_count)
    if hasattr(args, "datasets") and args.datasets:
        selected = parse_dataset_selection(args.datasets)
        if selected is not None and not hasattr(args, "input_dir"):
            selected = expand_sahu_brca2_base_selection(selected)
            available = available_datasets(Path(args.output_root))
            args.datasets = [dataset for dataset in available if dataset in selected]
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Adapt ClinProtGym final CSVs into esmDMS-ready protein datasets.")
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    add_common_args(p)
    p.add_argument("--keep-stop", action="store_true", help="Keep stop mutants. Default drops them for ESM compatibility.")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=prepare)

    p = sub.add_parser("create-embedding-jobs", help="Write ESM-C embedding and merge jobs.")
    add_common_args(p)
    add_model_args(p)
    p.add_argument("--python-executable", default=DEFAULT_PYTHON_EXECUTABLE)
    p.add_argument("--scratch-root", default="/scr")
    p.add_argument("--hf-home")
    p.add_argument("--n-chunks", type=int)
    p.add_argument("--max-active-embedding-tasks", type=int, default=10)
    p.add_argument("--embedding-partition")
    p.add_argument("--embedding-cpus", type=int, default=4)
    p.add_argument("--embedding-mem")
    p.add_argument("--embedding-time")
    p.add_argument("--merge-partition", default="any_cpu")
    p.add_argument("--merge-mem", default="16G")
    p.add_argument("--merge-time", default="02:00:00")
    p.add_argument("--skip-complete", action="store_true")
    p.add_argument("--submit", action="store_true")
    p.set_defaults(func=create_embedding_jobs)

    p = sub.add_parser("merge-embeddings", help="Merge completed embedding chunks into per-layer caches.")
    add_common_args(p)
    add_model_args(p)
    p.add_argument("--n-chunks", type=int)
    p.set_defaults(func=merge_embeddings)

    p = sub.add_parser("run-window-embedding-merge")
    p.add_argument("payload_path")
    p.set_defaults(func=run_window_embedding_merge)

    p = sub.add_parser("cache-status", help="Write embedding cache status tables.")
    add_common_args(p)
    add_model_args(p)
    p.set_defaults(func=write_embedding_cache_status)

    p = sub.add_parser("create-llr-jobs", help="Write masked-marginal ESM-C LLR jobs.")
    add_common_args(p)
    add_model_args(p)
    p.add_argument("--python-executable", default=DEFAULT_PYTHON_EXECUTABLE)
    p.add_argument("--llr-partition", default="any_cpu")
    p.add_argument("--llr-cpus", type=int, default=4)
    p.add_argument("--llr-mem", default="64G")
    p.add_argument("--llr-time", default="12:00:00")
    p.add_argument("--max-active-llr-tasks", type=int, default=4)
    p.add_argument("--submit", action="store_true")
    p.set_defaults(func=create_llr_jobs)

    p = sub.add_parser("run-llr-task")
    p.add_argument("payload_path")
    p.add_argument("task_idx", type=int)
    p.set_defaults(func=lambda args: run_payload_task(Path(args.payload_path), args.task_idx, compute_llr_for_task))

    p = sub.add_parser("create-sae-jobs", help="Write fixed DeltaEmbSAE layer-array jobs.")
    add_common_args(p)
    add_model_args(p)
    p.add_argument("--python-executable", default=DEFAULT_PYTHON_EXECUTABLE)
    p.add_argument("--sae-partition", default="any_cpu")
    p.add_argument("--sae-cpus", type=int, default=4)
    p.add_argument("--sae-mem", default="48G")
    p.add_argument("--sae-time", default="08:00:00")
    p.add_argument("--max-active-sae-tasks", type=int, default=10)
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--submit", action="store_true")
    p.set_defaults(func=create_sae_jobs)

    p = sub.add_parser("run-sae-task")
    p.add_argument("payload_path")
    p.add_argument("task_idx", type=int)
    p.set_defaults(func=lambda args: run_payload_task(Path(args.payload_path), args.task_idx, run_sae_task))

    p = sub.add_parser("collect-sae", help="Collect fixed SAE layer metrics.")
    add_common_args(p)
    p.set_defaults(func=collect_sae)

    p = sub.add_parser("create-benchmark-jobs", help="Write method benchmark jobs.")
    add_common_args(p)
    add_model_args(p)
    add_count_dataset_arg(p)
    p.add_argument("--python-executable", default=DEFAULT_PYTHON_EXECUTABLE)
    p.add_argument("--benchmark-partition", default="any_cpu")
    p.add_argument("--benchmark-cpus", type=int, default=4)
    p.add_argument("--benchmark-mem", default="48G")
    p.add_argument("--benchmark-time", default="08:00:00")
    p.add_argument("--max-active-benchmark-tasks", type=int, default=12)
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--submit", action="store_true")
    p.set_defaults(func=create_benchmark_jobs)

    p = sub.add_parser("run-benchmark-task")
    p.add_argument("payload_path")
    p.add_argument("task_idx", type=int)
    p.set_defaults(func=lambda args: run_payload_task(Path(args.payload_path), args.task_idx, run_benchmark_task))

    p = sub.add_parser("collect-benchmarks", help="Collect benchmark metrics.")
    add_common_args(p)
    add_count_dataset_arg(p)
    p.set_defaults(func=collect_benchmarks)

    p = sub.add_parser("create-ensemble-jobs", help="Write gamma=1 SAE rank-worst ensemble jobs.")
    add_common_args(p)
    add_count_dataset_arg(p)
    p.add_argument("--ensemble-top-n", type=int, default=12)
    p.add_argument("--ensemble-gamma", type=float, default=1.0)
    p.add_argument("--python-executable", default=DEFAULT_PYTHON_EXECUTABLE)
    p.add_argument("--ensemble-partition", default="any_cpu")
    p.add_argument("--ensemble-cpus", type=int, default=4)
    p.add_argument("--ensemble-mem", default="96G")
    p.add_argument("--ensemble-time", default="18:00:00")
    p.add_argument("--max-active-ensemble-tasks", type=int, default=4)
    p.add_argument("--submit", action="store_true")
    p.set_defaults(func=create_ensemble_jobs)

    p = sub.add_parser("run-ensemble-task")
    p.add_argument("payload_path")
    p.add_argument("task_idx", type=int)
    p.set_defaults(func=lambda args: run_payload_task(Path(args.payload_path), args.task_idx, run_ensemble_task))

    p = sub.add_parser("collect-ensembles", help="Collect ensemble metrics.")
    add_common_args(p)
    add_count_dataset_arg(p)
    p.set_defaults(func=collect_ensembles)

    p = sub.add_parser("summarize", help="Write average Spearman/AUC summary across datasets.")
    add_common_args(p)
    add_count_dataset_arg(p)
    add_sahu_brca2_final_arg(p)
    p.set_defaults(func=summarize)

    p = sub.add_parser(
        "cross-replicate-consistency",
        help="Write the ClinProtGym cross-replicate consistency table and dot plot.",
    )
    add_common_args(p)
    add_count_dataset_arg(p)
    add_sahu_brca2_final_arg(p)
    p.add_argument(
        "--omit-empty-datasets",
        action="store_true",
        help="Only show datasets with at least one finite consistency point on the x-axis.",
    )
    p.set_defaults(func=cross_replicate_consistency)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args(argv))
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
