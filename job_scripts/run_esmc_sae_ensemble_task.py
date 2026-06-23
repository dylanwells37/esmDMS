#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lower_context_helpers as h  # noqa: E402


def _load_payload(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _task_label(task: dict) -> str:
    parts = [task["task_type"], task["collection_name"]]
    if task.get("aggregation"):
        parts.append(task["aggregation"])
    gamma_objective = task.get("gamma_selection_objective")
    if gamma_objective and gamma_objective != "cross_replicate_consistency":
        parts.append(str(gamma_objective))
    return h.safe_file_component("__".join(parts))


def _load_inputs(payload: dict):
    repo_root = Path(payload.get("repo_root", Path.cwd())).resolve()
    paths = h.default_brca1_paths(repo_root)
    metrics_df = h.load_metrics_table(payload.get("metrics_path", paths.spearman_auc_table_path))
    sae_metrics_path = Path(payload.get("sae_metrics_path", paths.fixed_sae_metrics_path))
    sae_metrics_df = pd.read_csv(sae_metrics_path) if sae_metrics_path.is_file() else pd.DataFrame()
    collections = h.build_esmc_sae_collections(
        metrics_df,
        sae_metrics_df,
        embedding_type=payload.get("embedding_type", "max_pool"),
        top_n=int(payload.get("collection_top_n", 12)),
        balanced_per_model=int(payload.get("balanced_per_model", 4)),
    )
    runner = h.processed_brca1_runner(
        paths,
        model_name=payload.get("state_model_name", "biohub/ESMC-300M"),
        dataset_name=payload.get("dataset_name", "BRCA1"),
        embedding_type=payload.get("embedding_type", "max_pool"),
    )
    return paths, metrics_df, sae_metrics_df, collections, runner


def _metrics_for_spec(payload: dict, paths: h.BRCA1Paths, runner, spec: dict) -> pd.DataFrame:
    return h.ensemble_metrics_by_review_stars(
        [spec],
        clinvar_cache_path=payload.get("clinvar_annotation_cache_path", paths.clinvar_annotation_cache_path),
        scores_dataframe=runner.scores_dataframe,
        dataset_name=payload.get("dataset_name", "BRCA1"),
        primary_key=payload.get("primary_key", "hgvs_nt"),
        thresholds=payload.get("clinvar_review_star_thresholds", h.CLINVAR_REVIEW_STAR_THRESHOLDS),
    )


def _run_rank_task(payload: dict, paths: h.BRCA1Paths, runner, source_rows: pd.DataFrame, task: dict) -> dict:
    output_root = Path(payload["output_root"])
    label = _task_label(task)
    fitness_dir = output_root / "fitness"
    metrics_dir = output_root / "metrics"
    source_dir = output_root / "sources"
    for directory in (fitness_dir, metrics_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / f"{label}_sources.csv"
    source_rows.to_csv(source_path, index=False)
    fitness_df, n_components = h.aggregate_fitness_rows(source_rows, task["aggregation"])
    fitness_path = fitness_dir / f"{label}_fitness.csv"
    fitness_df.to_csv(fitness_path, index=False)

    spec = {
        "method_label": task.get("method_label", f"{task['collection_name']} {task['aggregation']}"),
        "short_label": task.get("short_label", f"{task['collection_name']} {task['aggregation']}"),
        "benchmark": "ESMC Fixed DeltaEmbSAE rank ensemble",
        "method": "DeltaEmbSAE rank ensemble",
        "embedding_type": payload.get("embedding_type", "max_pool"),
        "aggregation": task["aggregation"],
        "model_filter": task["collection_name"],
        "n_component_model_layers": n_components,
        "fitness_df": fitness_df,
        "fitness_path": fitness_path,
    }
    metrics_df = _metrics_for_spec(payload, paths, runner, spec)
    metrics_path = metrics_dir / f"{label}_metrics_by_review_stars.csv"
    metrics_df.to_csv(metrics_path, index=False)

    return {
        "task_type": "rank_ensemble",
        "collection_name": task["collection_name"],
        "aggregation": task["aggregation"],
        "n_sources": int(len(source_rows)),
        "n_component_model_layers": int(n_components),
        "fitness_path": str(fitness_path),
        "metrics_path": str(metrics_path),
        "source_path": str(source_path),
    }


def _run_pca_task(payload: dict, paths: h.BRCA1Paths, runner, source_rows: pd.DataFrame, task: dict) -> dict:
    output_root = Path(payload["output_root"])
    label = _task_label(task)
    gamma_selection_objective = task.get(
        "gamma_selection_objective",
        payload.get("gamma_selection_objective", "cross_replicate_consistency"),
    )
    pca_dir = output_root / "pca_concat" / gamma_selection_objective / label
    fitness_dir = output_root / "fitness"
    metrics_dir = output_root / "metrics"
    source_dir = output_root / "sources"
    for directory in (pca_dir, fitness_dir, metrics_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / f"{label}_sources.csv"
    source_rows.to_csv(source_path, index=False)
    gamma_values = task.get("gamma_values", payload.get("gamma_values"))
    if gamma_values is not None:
        gamma_values = np.asarray(gamma_values, dtype=float)

    run_info = h.run_pca_concatenated_sae_inference(
        source_rows,
        runner.sequence_dataframe,
        output_dir=pca_dir,
        run_label=label,
        max_components=int(task.get("max_components", payload.get("max_components", 1000))),
        per_source_components=int(task.get("per_source_components", payload.get("per_source_components", 64))),
        standardize=bool(task.get("standardize", payload.get("standardize_pca_inputs", True))),
        importance_metric=task.get("importance_metric", payload.get("importance_metric", "explained_variance_ratio")),
        seed=int(task.get("seed", payload.get("seed", 42))),
        gamma_values=gamma_values,
        gamma_selection_objective=gamma_selection_objective,
        scores_dataframe=runner.scores_dataframe,
        score_col=payload.get("score_col", "score"),
    )

    annotation_map = h.clinvar_binary_annotation_map(
        payload.get("clinvar_annotation_cache_path", paths.clinvar_annotation_cache_path),
        primary_key=payload.get("primary_key", "hgvs_nt"),
        min_review_stars=0,
    )
    fitness_df = h.fitness_for_feature_mapping(
        run_info["seq_to_features"],
        run_info["inference_result"],
        annotation_map=annotation_map,
        norm_scheme="none",
    )
    fitness_path = fitness_dir / f"{label}_fitness.csv"
    fitness_df.to_csv(fitness_path, index=False)

    spec = {
        "method_label": task.get("method_label", f"{task['collection_name']} PCA concat"),
        "short_label": task.get("short_label", f"{task['collection_name']} PCA"),
        "benchmark": "ESMC Fixed DeltaEmbSAE PCA concat",
        "method": "DeltaEmbSAE PCA concat",
        "embedding_type": payload.get("embedding_type", "max_pool"),
        "aggregation": "pca_concat",
        "model_filter": task["collection_name"],
        "n_component_model_layers": run_info["n_sources"],
        "fitness_df": fitness_df,
        "fitness_path": fitness_path,
        "feature_path": run_info["feature_path"],
        "inference_path": run_info["inference_path"],
        "gamma_plot_path": run_info["gamma_plot_path"],
        "gamma_table_path": run_info["gamma_table_path"],
        "gamma_pair_table_path": run_info["gamma_pair_table_path"],
        "gamma_selection_objective": gamma_selection_objective,
        "selected_gamma_mean_pairwise_pearson_r": run_info["selected_gamma_mean_pairwise_pearson_r"],
        "selected_gamma_mavedb_spearman": run_info["selected_gamma_mavedb_spearman"],
        "selected_gamma_grid": run_info["selected_gamma_grid"],
        "component_table_path": run_info["component_table_path"],
    }
    metrics_df = _metrics_for_spec(payload, paths, runner, spec)
    metrics_df["gamma_opt"] = run_info["gamma_opt"]
    metrics_df["gamma_selection_objective"] = gamma_selection_objective
    metrics_df["selected_gamma_mean_pairwise_pearson_r"] = run_info["selected_gamma_mean_pairwise_pearson_r"]
    metrics_df["selected_gamma_mavedb_spearman"] = run_info["selected_gamma_mavedb_spearman"]
    metrics_df["selected_gamma_grid"] = run_info["selected_gamma_grid"]
    metrics_df["n_selected_components"] = run_info["n_selected_components"]
    metrics_df["per_source_components"] = int(task.get("per_source_components", payload.get("per_source_components", 64)))
    metrics_df["importance_metric"] = task.get("importance_metric", payload.get("importance_metric", "explained_variance_ratio"))
    metrics_df["standardize_pca_inputs"] = bool(task.get("standardize", payload.get("standardize_pca_inputs", True)))
    metrics_path = metrics_dir / f"{label}_metrics_by_review_stars.csv"
    metrics_df.to_csv(metrics_path, index=False)

    return {
        "task_type": "pca_concat",
        "collection_name": task["collection_name"],
        "n_sources": int(run_info["n_sources"]),
        "n_selected_components": int(run_info["n_selected_components"]),
        "gamma_opt": float(run_info["gamma_opt"]),
        "gamma_selection_objective": gamma_selection_objective,
        "selected_gamma_mean_pairwise_pearson_r": float(run_info["selected_gamma_mean_pairwise_pearson_r"]),
        "selected_gamma_mavedb_spearman": float(run_info["selected_gamma_mavedb_spearman"]),
        "feature_path": str(run_info["feature_path"]),
        "inference_path": str(run_info["inference_path"]),
        "fitness_path": str(fitness_path),
        "metrics_path": str(metrics_path),
        "component_table_path": str(run_info["component_table_path"]),
        "source_table_path": str(run_info["source_table_path"]),
        "source_path": str(source_path),
        "gamma_plot_path": str(run_info["gamma_plot_path"]),
        "gamma_table_path": str(run_info["gamma_table_path"]),
        "gamma_pair_table_path": str(run_info["gamma_pair_table_path"]),
    }


def run_task(payload_path: str | Path, task_idx: int) -> dict:
    payload = _load_payload(payload_path)
    tasks = payload["tasks"]
    if task_idx < 0 or task_idx >= len(tasks):
        raise IndexError(f"task_idx {task_idx} is outside 0..{len(tasks) - 1}")
    task = tasks[task_idx]
    output_root = Path(payload["output_root"])
    status_dir = output_root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"{task_idx:04d}_{_task_label(task)}_status.json"
    try:
        paths, _, _, collections, runner = _load_inputs(payload)
        if task["collection_name"] not in collections:
            raise KeyError(f"Unknown collection {task['collection_name']!r}. Available: {sorted(collections)}")
        source_rows = collections[task["collection_name"]].copy()
        if source_rows.empty:
            raise ValueError(f"Collection {task['collection_name']!r} is empty.")
        if task["task_type"] == "rank_ensemble":
            result = _run_rank_task(payload, paths, runner, source_rows, task)
        elif task["task_type"] == "pca_concat":
            result = _run_pca_task(payload, paths, runner, source_rows, task)
        else:
            raise ValueError(f"Unsupported task_type: {task['task_type']!r}")
        result.update({"task_idx": task_idx, "status": "ok"})
        h.write_json_summary(status_path, result)
        return result
    except Exception as exc:
        failure = {
            "task_idx": task_idx,
            "status": "failed",
            "task": task,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        h.write_json_summary(status_path, failure)
        raise


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: run_esmc_sae_ensemble_task.py PAYLOAD.pkl TASK_IDX", file=sys.stderr)
        return 2
    result = run_task(argv[1], int(argv[2]))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
