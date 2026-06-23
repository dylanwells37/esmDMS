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
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: run_rank_ensemble_gamma_sweep.py PAYLOAD.pkl", file=sys.stderr)
        return 2
    payload_path = Path(argv[1])
    payload = _load_payload(payload_path)
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"{payload.get('run_label', 'rank_gamma_sweep')}_status.json"
    try:
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
        collection_name = payload["collection_name"]
        if collection_name not in collections:
            raise KeyError(f"Unknown collection {collection_name!r}. Available: {sorted(collections)}")
        runner = h.processed_brca1_runner(
            paths,
            model_name=payload.get("state_model_name", "biohub/ESMC-300M"),
            dataset_name=payload.get("dataset_name", "BRCA1"),
            embedding_type=payload.get("embedding_type", "max_pool"),
        )
        gamma_values = np.asarray(payload["gamma_values"], dtype=float)
        annotation_map = h.clinvar_binary_annotation_map(
            payload.get("clinvar_annotation_cache_path", paths.clinvar_annotation_cache_path),
            primary_key=payload.get("primary_key", "hgvs_nt"),
            min_review_stars=int(payload.get("min_review_stars", 0)),
        )
        metrics_out, consistency_out = h.rank_ensemble_gamma_sweep(
            collections[collection_name],
            runner.sequence_dataframe,
            runner.scores_dataframe,
            annotation_map=annotation_map,
            gamma_values=gamma_values,
            aggregation=payload.get("aggregation", "rank_worst"),
            score_col=payload.get("score_col", "score"),
        )
        metrics_path = Path(payload["metrics_output_path"])
        consistency_path = Path(payload["component_consistency_output_path"])
        figure_path = Path(payload["figure_output_path"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        consistency_path.parent.mkdir(parents=True, exist_ok=True)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.to_csv(metrics_path, index=False)
        consistency_out.to_csv(consistency_path, index=False)
        fig = h.plot_gamma_spearman_auc_scatter(
            metrics_out,
            figure_path,
            payload.get("plot_title", "Rank ensemble gamma sweep"),
            figure_dpi=int(payload.get("figure_dpi", 300)),
        )
        if fig is not None:
            import matplotlib.pyplot as plt

            plt.close(fig)
        result = {
            "status": "ok",
            "payload_path": str(payload_path),
            "metrics_path": str(metrics_path),
            "component_consistency_path": str(consistency_path),
            "figure_path": str(figure_path),
            "n_gamma": int(len(gamma_values)),
            "collection_name": collection_name,
            "aggregation": payload.get("aggregation", "rank_worst"),
        }
        h.write_json_summary(status_path, result)
        print(result)
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "payload_path": str(payload_path),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        h.write_json_summary(status_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
