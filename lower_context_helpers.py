from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "esmDMS_matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

from esmDMS import CellularDMSInput, ESMDMSConfig, EmbeddingModel, esmDMS
from popDMS import _prepare_eigendecomp, compute_dx_covariance_esm, infer_gamma_range, mini_infer_esm


PATHOGENICITY_LABELS = {"benign", "pathogenic"}
CLINVAR_REVIEW_STAR_THRESHOLDS = [1, 2, 3, 4]

MODEL_SIZE_MILLIONS = {
    "facebook/esm2_t6_8M_UR50D": 8,
    "facebook/esm2_t12_35M_UR50D": 35,
    "facebook/esm2_t30_150M_UR50D": 150,
    "facebook/esm2_t33_650M_UR50D": 650,
    "facebook/esm2_t36_3B_UR50D": 3000,
    "biohub/ESMC-300M": 300,
    "biohub/ESMC-600M": 600,
    "biohub/ESMC-6B": 6000,
}
MODEL_LAYER_COUNTS = {
    "biohub/ESMC-300M": 30,
    "biohub/ESMC-600M": 36,
    "biohub/ESMC-6B": 80,
}
MODEL_SHORT_NAMES = {
    "facebook/esm2_t6_8M_UR50D": "ESM2-8M",
    "facebook/esm2_t12_35M_UR50D": "ESM2-35M",
    "facebook/esm2_t30_150M_UR50D": "ESM2-150M",
    "facebook/esm2_t33_650M_UR50D": "ESM2-650M",
    "facebook/esm2_t36_3B_UR50D": "ESM2-3B",
    "biohub/ESMC-300M": "ESMC-300M",
    "biohub/ESMC-600M": "ESMC-600M",
    "biohub/ESMC-6B": "ESMC-6B",
}


@dataclass(frozen=True)
class BRCA1Paths:
    repo_root: Path
    data_dir: Path
    analysis_dir: Path
    baseline_analysis_dir: Path
    sequence_dir: Path
    figure_dir: Path
    table_dir: Path
    job_dir: Path
    baseline_table_dir: Path
    baseline_sequence_dir: Path
    clinvar_annotation_cache_path: Path
    exact_popdms_fitness_path: Path
    exact_popdms_selection_path: Path
    enrichment_ratio_fitness_path: Path
    spearman_auc_table_path: Path
    spearman_auc_star_combined_table_path: Path
    fixed_sae_metrics_path: Path


def default_brca1_paths(repo_root: str | Path) -> BRCA1Paths:
    repo_root = Path(repo_root)
    data_dir = repo_root / "data" / "mavedb_data"
    analysis_dir = repo_root / "data" / "esm_data_analysis" / "BRCA1_plm_layer_sae"
    baseline_analysis_dir = repo_root / "data" / "esm_data_analysis" / "BRCA1_experimental"
    sequence_dir = analysis_dir / "sequence_data"
    figure_dir = analysis_dir / "figures"
    table_dir = analysis_dir / "tables"
    job_dir = sequence_dir / "jobs"
    baseline_table_dir = baseline_analysis_dir / "tables"
    baseline_sequence_dir = baseline_analysis_dir / "sequence_data"
    return BRCA1Paths(
        repo_root=repo_root,
        data_dir=data_dir,
        analysis_dir=analysis_dir,
        baseline_analysis_dir=baseline_analysis_dir,
        sequence_dir=sequence_dir,
        figure_dir=figure_dir,
        table_dir=table_dir,
        job_dir=job_dir,
        baseline_table_dir=baseline_table_dir,
        baseline_sequence_dir=baseline_sequence_dir,
        clinvar_annotation_cache_path=baseline_table_dir / "BRCA1_clinvar_annotations_by_hgvs.csv",
        exact_popdms_fitness_path=baseline_table_dir / "BRCA1_exact_popDMS_fitness_values.csv",
        exact_popdms_selection_path=baseline_sequence_dir / "regular_popdms" / "BRCA1_exact_popDMS_selection_coefficients.csv.gz",
        enrichment_ratio_fitness_path=table_dir / "BRCA1_enrichment_ratio_fitness_values.csv",
        spearman_auc_table_path=table_dir / "BRCA1_plm_layer_mavedb_spearman_vs_clinvar_auc.csv",
        spearman_auc_star_combined_table_path=table_dir / "BRCA1_plm_layer_mavedb_spearman_vs_clinvar_auc_by_review_stars.csv",
        fixed_sae_metrics_path=table_dir / "BRCA1_fixed_deltaembsae_layer_metrics.csv",
    )


def brca1_input(paths: BRCA1Paths) -> CellularDMSInput:
    return CellularDMSInput(
        reference_nuc_path=paths.data_dir / "BRCA1_reference_sequence.dat",
        mavedb_csv_path=paths.data_dir / "BRCA1_counts.csv",
        scores_csv_path=paths.data_dir / "BRCA1_scores.csv",
        reference_kind="protein",
        primary_key="hgvs_nt",
    )


def model_cache_label(model_name: str) -> str:
    return str(model_name).replace("/", "__")


def safe_file_component(value: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
    return safe or "run"


def n_transformer_layers(model_name: str) -> int:
    match = re.search(r"esm2_t(\d+)_", model_name)
    if match:
        return int(match.group(1))
    if model_name in MODEL_LAYER_COUNTS:
        return MODEL_LAYER_COUNTS[model_name]
    raise ValueError(f"No layer count registered for {model_name!r}.")


def model_family(model_name: str) -> str:
    return "ESMC" if "ESMC" in str(model_name) else "ESM2"


def model_job_defaults(model_name: str, size_millions: int | float) -> dict:
    is_esmc = "ESMC" in str(model_name)
    large_model = size_millions > 1000
    return {
        "embedding_partition": "big_memory" if large_model else "any_cpu",
        "embedding_gres": None,
        "embedding_constraint": None,
        "embedding_mem": "64G" if (size_millions >= 300 or is_esmc) else "48G",
        "embedding_time": "24:00:00" if large_model else "12:00:00" if (size_millions >= 300 or is_esmc) else "08:00:00",
        "torch_dtype": "float32" if is_esmc else None,
        "allow_cpu_esmc": is_esmc,
        "n_chunks": 40,
    }


def build_model_specs(model_names: Iterable[str] | None = None) -> list[dict]:
    if model_names is None:
        model_names = list(EmbeddingModel.__args__)
    specs = []
    for model_name in model_names:
        size = MODEL_SIZE_MILLIONS[model_name]
        n_layers = n_transformer_layers(model_name)
        spec = {
            "model": model_name,
            "model_short": MODEL_SHORT_NAMES.get(model_name, model_name),
            "model_cache_label": model_cache_label(model_name),
            "family": model_family(model_name),
            "size_millions": size,
            "n_transformer_layers": n_layers,
            "n_hidden_state_layers": n_layers + 1,
            "layers": list(range(n_layers + 1)),
            "default_include": size <= 1000,
            "requires_custom_large_model_job": size > 1000,
        }
        spec.update(model_job_defaults(model_name, size))
        specs.append(spec)
    return specs


def make_runner(
    model_name: str,
    input_data: CellularDMSInput,
    save_dir: str | Path,
    dataset_name: str = "BRCA1",
    embedding_type: str = "max_pool",
    template_runner: esmDMS | None = None,
) -> esmDMS:
    config = ESMDMSConfig(
        embedding_model=model_name,
        embedding_type=embedding_type,
        local_or_disk="both",
        save_dir=str(save_dir),
        dataset_name=dataset_name,
    )
    runner = esmDMS(input_data=input_data, config=config)
    if template_runner is not None and getattr(template_runner, "sequence_dataframe", None) is not None:
        runner.reference_sequence = template_runner.reference_sequence
        runner.sequence_dataframe = template_runner.sequence_dataframe
        runner.sequence_to_mutation_sites = template_runner.sequence_to_mutation_sites
        runner.sequence_to_protein_sequence = template_runner.sequence_to_protein_sequence
        runner.sequence_metadata = template_runner.sequence_metadata
        runner.reference_kind = template_runner.reference_kind
        runner.scores_dataframe = template_runner.scores_dataframe
    return runner


def processed_brca1_runner(
    paths: BRCA1Paths,
    model_name: str = "biohub/ESMC-300M",
    dataset_name: str = "BRCA1",
    embedding_type: str = "max_pool",
) -> esmDMS:
    runner = make_runner(
        model_name,
        brca1_input(paths),
        paths.sequence_dir,
        dataset_name=dataset_name,
        embedding_type=embedding_type,
    )
    runner.process_raw_data(drop_stop_codons=True)
    runner.load_functional_scores()
    return runner


def as_existing_path(value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    path = Path(value)
    return path if path.is_file() else None


def load_pickle(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_pickle(value, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)
    return path


def reconstruction_metrics(viz_path: object) -> dict:
    viz_path = as_existing_path(viz_path)
    if viz_path is None:
        return {
            "reconstruction_mse": np.nan,
            "reconstruction_r2": np.nan,
            "activation_sparsity": np.nan,
            "activation_density": np.nan,
            "n_active_features": np.nan,
            "active_feature_fraction": np.nan,
            "final_train_loss": np.nan,
            "final_test_loss": np.nan,
        }
    viz = load_pickle(viz_path)
    x = np.asarray(viz["X_original"], dtype=float)
    x_recon = np.asarray(viz["X_reconstructed"], dtype=float)
    test_idx = np.asarray(viz.get("test_idx") or [], dtype=int)
    eval_idx = test_idx if len(test_idx) else np.arange(x.shape[0])
    x_eval = x[eval_idx]
    x_recon_eval = x_recon[eval_idx]
    residual = x_eval - x_recon_eval
    mse = float(np.mean(residual**2))
    sse = float(np.sum(residual**2))
    centered = x_eval - x_eval.mean(axis=0, keepdims=True)
    sst = float(np.sum(centered**2))
    r2 = 1.0 - (sse / sst) if sst > 0 else np.nan
    z_all = np.asarray(viz["Z_all"])
    activation_density = float((z_all > 0).mean())
    active_mask = np.asarray(viz["active_mask"], dtype=bool)
    train_losses = viz.get("train_losses") or []
    test_losses = viz.get("test_losses") or []
    return {
        "reconstruction_mse": mse,
        "reconstruction_r2": r2,
        "activation_sparsity": 1.0 - activation_density,
        "activation_density": activation_density,
        "n_active_features": int(active_mask.sum()),
        "active_feature_fraction": float(active_mask.mean()),
        "final_train_loss": float(train_losses[-1]) if train_losses else np.nan,
        "final_test_loss": float(test_losses[-1]) if test_losses else np.nan,
    }


def clinvar_annotation_table(cache_path: str | Path, primary_key: str = "hgvs_nt") -> pd.DataFrame:
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing cached ClinVar annotation table: {cache_path}. "
            "Run the annotation section of the baseline BRCA1 analysis first."
        )
    annotations = pd.read_csv(cache_path)
    annotations["SequenceIndex"] = annotations[primary_key].astype(str)
    annotations["stars"] = pd.to_numeric(annotations.get("stars", 0), errors="coerce").fillna(0)
    return annotations


def clinvar_binary_annotation_map(
    cache_path: str | Path,
    primary_key: str = "hgvs_nt",
    min_review_stars: int = 0,
) -> dict[str, str]:
    annotations = clinvar_annotation_table(cache_path, primary_key=primary_key)
    keep = annotations["annotation"].isin(PATHOGENICITY_LABELS)
    if min_review_stars > 0:
        keep = keep & annotations["stars"].ge(min_review_stars)
    annotations = annotations[keep].copy()
    return dict(zip(annotations["SequenceIndex"], annotations["annotation"]))


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


def spearman_for_fitness_dataframe(
    fitness_df: pd.DataFrame,
    scores_dataframe: pd.DataFrame,
    score_col: str = "score",
) -> tuple[float, int]:
    score_df = scores_dataframe[["SequenceIndex", score_col]].dropna().copy()
    score_df["SequenceIndex"] = score_df["SequenceIndex"].astype(str)
    comparison_df = fitness_df.merge(score_df, on="SequenceIndex", how="inner")
    fitness = pd.to_numeric(comparison_df["fitness"], errors="coerce").to_numpy(dtype=float)
    scores = pd.to_numeric(comparison_df[score_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(fitness) & np.isfinite(scores)
    if finite.sum() < 3:
        return np.nan, int(finite.sum())
    rho = spearmanr(fitness[finite], scores[finite]).statistic
    return float(rho), int(finite.sum())


def functional_score_fitness_dataframe(
    scores_dataframe: pd.DataFrame,
    annotation_map: dict[str, str],
    score_col: str = "score",
) -> pd.DataFrame:
    score_df = scores_dataframe[["SequenceIndex", score_col]].dropna().copy()
    score_df["SequenceIndex"] = score_df["SequenceIndex"].astype(str)
    score_df["fitness"] = pd.to_numeric(score_df[score_col], errors="coerce")
    score_df["annotation"] = score_df["SequenceIndex"].map(annotation_map)
    return score_df[["SequenceIndex", "fitness", "annotation"]]


def enrichment_ratio_fitness_dataframe(
    mavedb_csv_path: str | Path,
    primary_key: str,
    annotation_map: dict[str, str] | None = None,
    output_path: str | Path | None = None,
    pseudocount: float = 0.5,
) -> pd.DataFrame:
    counts_df = pd.read_csv(mavedb_csv_path).copy()
    counts_df["SequenceIndex"] = counts_df[primary_key].astype(str)
    day_count_cols = []
    for column in counts_df.columns:
        match = re.fullmatch(r"count_day(?P<day>\d+)_rep(?P<rep>\d+)", str(column))
        if match:
            day_count_cols.append((int(match.group("day")), int(match.group("rep")), column))
    if not day_count_cols:
        raise ValueError("No count_day<day>_rep<rep> columns were found for enrichment ratio baseline.")
    final_day = max(day for day, _, _ in day_count_cols)
    final_cols = [(rep, column) for day, rep, column in day_count_cols if day == final_day]
    day0_cols = {rep: column for day, rep, column in day_count_cols if day == 0}
    n_rows = len(counts_df)
    replicate_scores = []
    for rep, final_col in sorted(final_cols):
        initial_col = day0_cols.get(rep, "count_library")
        initial_counts = pd.to_numeric(counts_df[initial_col], errors="coerce").fillna(0.0)
        final_counts = pd.to_numeric(counts_df[final_col], errors="coerce").fillna(0.0)
        initial_freq = (initial_counts + pseudocount) / (initial_counts.sum() + pseudocount * n_rows)
        final_freq = (final_counts + pseudocount) / (final_counts.sum() + pseudocount * n_rows)
        replicate_scores.append(np.log2(final_freq / initial_freq))
    enrichment = pd.concat(replicate_scores, axis=1).mean(axis=1)
    fitness_df = pd.DataFrame({"SequenceIndex": counts_df["SequenceIndex"], "fitness": enrichment})
    if annotation_map is not None:
        fitness_df["annotation"] = fitness_df["SequenceIndex"].map(annotation_map)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fitness_df.to_csv(output_path, index=False)
    return fitness_df


def regular_popdms_fitness_dataframe(
    exact_popdms_fitness_path: str | Path,
    annotation_map: dict[str, str],
) -> pd.DataFrame:
    exact_popdms_fitness_path = Path(exact_popdms_fitness_path)
    if not exact_popdms_fitness_path.is_file():
        raise FileNotFoundError(f"Missing exact popDMS fitness table: {exact_popdms_fitness_path}")
    fitness_df = pd.read_csv(exact_popdms_fitness_path)
    fitness_df["SequenceIndex"] = fitness_df["SequenceIndex"].astype(str)
    fitness_df["annotation"] = fitness_df["SequenceIndex"].map(annotation_map)
    return fitness_df[["SequenceIndex", "fitness", "annotation"]]


def fitness_for_feature_mapping(
    seq_to_features: dict,
    inference_result,
    annotation_map: dict[str, str] | None = None,
    norm_scheme: str = "none",
    baseline: float = 1.0,
) -> pd.DataFrame:
    _, seq_to_features = esmDMS._drop_missing_features(None, seq_to_features, "Fitness scoring")
    esmDMS._require_vector_features(seq_to_features, "Fitness scoring")
    seq_ids = list(seq_to_features)
    features = np.asarray([seq_to_features[seq_id] for seq_id in seq_ids], dtype=float)
    if norm_scheme is not None and norm_scheme != "none":
        features = esmDMS._normalize_features(features, norm_scheme)
    if getattr(inference_result, "s_joint", None) is not None:
        fitness = baseline + features @ inference_result.s_joint
    else:
        fitness = baseline + np.asarray(
            [features @ inference_result.s[rep_idx] for rep_idx in range(inference_result.s.shape[0])]
        ).mean(axis=0)
    annotation_map = annotation_map or {}
    return pd.DataFrame(
        {
            "SequenceIndex": [str(seq_id) for seq_id in seq_ids],
            "fitness": fitness,
            "annotation": [annotation_map.get(str(seq_id)) for seq_id in seq_ids],
        }
    )


def metrics_row_from_fitness(
    fitness_df: pd.DataFrame,
    annotation_map: dict[str, str],
    scores_dataframe: pd.DataFrame,
    dataset_name: str,
    model_label: str,
    model_group: str,
    benchmark: str,
    method: str,
    embedding_type: str,
    model: str = "",
    model_short: str = "",
    layer: int | float = np.nan,
    n_features: int | float = np.nan,
    k: int | float = np.nan,
    feature_path: str | Path | None = "",
    inference_path: str | Path | None = "",
    fitness_path: str | Path | None = "",
    plot_group: str | None = None,
) -> dict:
    if "annotation" not in fitness_df.columns:
        fitness_df = fitness_df.copy()
        fitness_df["annotation"] = fitness_df["SequenceIndex"].astype(str).map(annotation_map)
    spearman_rho, n_score_sequences = spearman_for_fitness_dataframe(fitness_df, scores_dataframe)
    auc_metrics = classification_metrics_for_fitness(fitness_df)
    return {
        "dataset": dataset_name,
        "model": model,
        "model_short": model_short,
        "layer": layer,
        "model_label": model_label,
        "model_group": model_group,
        "benchmark": benchmark,
        "method": method,
        "embedding_type": embedding_type,
        "n_features": n_features,
        "k": k,
        "spearman_rho": spearman_rho,
        "n_score_sequences": n_score_sequences,
        "feature_path": str(feature_path) if feature_path else "",
        "inference_path": str(inference_path) if inference_path else "",
        "fitness_path": str(fitness_path) if fitness_path else "",
        "method_pool": f"{method} / {embedding_type}",
        "plot_group": plot_group or model_group,
        **auc_metrics,
    }


def load_metrics_table(table_path: str | Path, metrics_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if metrics_df is not None and not metrics_df.empty:
        out = metrics_df.copy()
    else:
        table_path = Path(table_path)
        if not table_path.is_file():
            raise FileNotFoundError(f"Missing metrics table: {table_path}")
        out = pd.read_csv(table_path)
    for column in ["layer", "auc", "spearman_rho", "n_features", "k", "n_variants"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def review_star_specs(thresholds: Iterable[int] = CLINVAR_REVIEW_STAR_THRESHOLDS) -> list[tuple[int, str]]:
    return [(0, "all binary annotations")] + [(int(stars), f">= {int(stars)} star(s)") for stars in thresholds]


def review_star_label(min_review_stars: int) -> str:
    min_review_stars = int(min_review_stars)
    return "all binary annotations" if min_review_stars == 0 else f">= {min_review_stars} star(s)"


def is_esmc_row(df: pd.DataFrame) -> pd.Series:
    model = df.get("model", pd.Series("", index=df.index)).astype(str)
    model_short = df.get("model_short", pd.Series("", index=df.index)).astype(str)
    return model.str.contains("ESMC", case=False, na=False) | model_short.str.startswith("ESMC-", na=False)


def ensemble_source_rows(
    metrics_df: pd.DataFrame,
    benchmark: str,
    method: str,
    embedding_type: str = "max_pool",
    model_short_prefix: str | None = None,
    esmc_only: bool = True,
    exclude_layer0: bool = True,
) -> pd.DataFrame:
    rows = metrics_df[
        metrics_df["benchmark"].eq(benchmark)
        & metrics_df["method"].eq(method)
        & metrics_df["embedding_type"].eq(embedding_type)
        & np.isfinite(metrics_df["layer"])
    ].copy()
    if exclude_layer0:
        rows = rows[rows["layer"].gt(0)].copy()
    if model_short_prefix is not None:
        rows = rows[rows["model_short"].astype(str).str.startswith(model_short_prefix)].copy()
    if esmc_only:
        rows = rows[is_esmc_row(rows)].copy()
    return rows.sort_values(["benchmark", "model_short", "layer"]).reset_index(drop=True)


def load_ensemble_fitness_series(row: pd.Series) -> pd.Series:
    fitness_path = as_existing_path(row.get("fitness_path", ""))
    if fitness_path is None:
        raise FileNotFoundError(f"Missing fitness_path for ensemble row {row.get('model_label', '')!r}")
    fitness_df = pd.read_csv(fitness_path)
    fitness_df = fitness_df[["SequenceIndex", "fitness"]].copy()
    fitness_df["SequenceIndex"] = fitness_df["SequenceIndex"].astype(str)
    fitness_df["fitness"] = pd.to_numeric(fitness_df["fitness"], errors="coerce")
    fitness_df = fitness_df.dropna(subset=["fitness"])
    layer = row.get("layer", "x")
    layer = int(layer) if pd.notna(layer) and np.isfinite(float(layer)) else layer
    series_name = f"{row.get('model_short', row.get('model', 'model'))}_L{layer}"
    return fitness_df.groupby("SequenceIndex")["fitness"].mean().rename(series_name)


def aggregate_fitness_rows(source_rows: pd.DataFrame, aggregation: str) -> tuple[pd.DataFrame, int]:
    if source_rows.empty:
        raise ValueError("No source rows were available for the requested ensemble.")
    score_series = [load_ensemble_fitness_series(row) for _, row in source_rows.iterrows()]
    score_matrix = pd.concat(score_series, axis=1)
    n_components = score_matrix.notna().sum(axis=1)

    if aggregation == "fitness_mean":
        fitness = score_matrix.mean(axis=1, skipna=True)
        extra = {"mean_fitness": fitness}
    else:
        rank_matrix = score_matrix.rank(method="average", ascending=True)
        average_score = score_matrix.mean(axis=1, skipna=True)
        if aggregation == "rank_mean":
            selected_rank = rank_matrix.mean(axis=1, skipna=True)
        elif aggregation == "rank_median":
            selected_rank = rank_matrix.median(axis=1, skipna=True)
        elif aggregation == "rank_best":
            selected_rank = rank_matrix.max(axis=1, skipna=True)
        elif aggregation == "rank_worst":
            selected_rank = rank_matrix.min(axis=1, skipna=True)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        tmp = pd.DataFrame(
            {
                "selected_rank": selected_rank,
                "average_score": average_score,
                "n_component_model_layers": n_components,
            }
        )
        tmp = tmp[tmp["n_component_model_layers"].gt(0)].dropna(subset=["selected_rank", "average_score"]).copy()
        tmp["_sequence_index_sort"] = tmp.index.astype(str)
        tmp = tmp.sort_values(
            ["selected_rank", "average_score", "_sequence_index_sort"],
            ascending=[True, True, True],
        )
        reranked = pd.Series(np.arange(1, len(tmp) + 1, dtype=float), index=tmp.index)
        fitness = pd.Series(np.nan, index=score_matrix.index, dtype=float)
        fitness.loc[reranked.index] = reranked
        extra = {
            "selected_rank": selected_rank,
            "average_score": average_score,
            f"{aggregation}_with_average_score_tiebreak": fitness,
        }

    output_df = pd.DataFrame(
        {
            "SequenceIndex": score_matrix.index.astype(str),
            "fitness": fitness.to_numpy(dtype=float),
            "n_component_model_layers": n_components.to_numpy(dtype=int),
        }
    )
    for name, values in extra.items():
        output_df[name] = values.to_numpy(dtype=float)
    output_df = output_df[output_df["n_component_model_layers"].gt(0)].dropna(subset=["fitness"]).reset_index(drop=True)
    return output_df, len(score_series)


def rank_ensemble_fitness_from_score_matrix(
    score_matrix: pd.DataFrame,
    aggregation: str,
) -> pd.DataFrame:
    if score_matrix.empty:
        raise ValueError("score_matrix is empty.")
    n_components = score_matrix.notna().sum(axis=1)
    rank_matrix = score_matrix.rank(method="average", ascending=True)
    average_score = score_matrix.mean(axis=1, skipna=True)
    if aggregation == "rank_mean":
        selected_rank = rank_matrix.mean(axis=1, skipna=True)
    elif aggregation in {"rank_median", "median_rank"}:
        selected_rank = rank_matrix.median(axis=1, skipna=True)
    elif aggregation in {"rank_best", "best_rank"}:
        selected_rank = rank_matrix.max(axis=1, skipna=True)
    elif aggregation in {"rank_worst", "worst_rank"}:
        selected_rank = rank_matrix.min(axis=1, skipna=True)
    else:
        raise ValueError(f"Unsupported rank aggregation: {aggregation}")
    tmp = pd.DataFrame(
        {
            "selected_rank": selected_rank,
            "average_score": average_score,
            "n_component_model_layers": n_components,
        }
    )
    tmp = tmp[tmp["n_component_model_layers"].gt(0)].dropna(subset=["selected_rank", "average_score"]).copy()
    tmp["_sequence_index_sort"] = tmp.index.astype(str)
    tmp = tmp.sort_values(["selected_rank", "average_score", "_sequence_index_sort"], ascending=[True, True, True])
    reranked = pd.Series(np.arange(1, len(tmp) + 1, dtype=float), index=tmp.index)
    out = pd.DataFrame(
        {
            "SequenceIndex": score_matrix.index.astype(str),
            "fitness": pd.Series(np.nan, index=score_matrix.index, dtype=float),
            "selected_rank": selected_rank,
            "average_score": average_score,
            "n_component_model_layers": n_components.astype(int),
        }
    )
    out.loc[reranked.index, "fitness"] = reranked
    return out[out["n_component_model_layers"].gt(0)].dropna(subset=["fitness"]).reset_index(drop=True)


def rank_ensemble_gamma_sweep(
    source_rows: pd.DataFrame,
    sequence_dataframe: pd.DataFrame,
    scores_dataframe: pd.DataFrame,
    annotation_map: dict[str, str],
    gamma_values: np.ndarray,
    aggregation: str = "rank_worst",
    score_col: str = "score",
    baseline: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source_rows.empty:
        raise ValueError("No source rows were provided for gamma sweep.")
    gamma_values = np.asarray(gamma_values, dtype=float)
    component_payloads = []
    common_ids: set[str] | None = None
    for source_idx, (_, row) in enumerate(source_rows.reset_index(drop=True).iterrows()):
        feature_path = as_existing_path(row.get("feature_path", ""))
        if feature_path is None:
            raise FileNotFoundError(f"Missing feature_path for source row {row.get('model_label', source_idx)!r}.")
        seq_to_features = load_pickle(feature_path)
        inference_sequence_df, seq_to_features = esmDMS._drop_missing_features(
            sequence_dataframe,
            seq_to_features,
            f"rank gamma sweep source {source_idx}",
        )
        esmDMS._require_vector_features(seq_to_features, f"rank gamma sweep source {source_idx}")
        gamma_values_out, s_by_gamma, s_joint_by_gamma = infer_gamma_range(
            inference_sequence_df,
            seq_to_features,
            gamma_values=gamma_values,
        )
        if not np.allclose(gamma_values_out, gamma_values):
            raise ValueError("infer_gamma_range returned unexpected gamma values.")
        seq_ids = [str(seq_id) for seq_id in seq_to_features]
        common_ids = set(seq_ids) if common_ids is None else common_ids & set(seq_ids)
        component_payloads.append(
            {
                "source_idx": source_idx,
                "row": row,
                "seq_to_features": {str(k): np.asarray(v, dtype=float) for k, v in seq_to_features.items()},
                "s_by_gamma": s_by_gamma,
                "s_joint_by_gamma": s_joint_by_gamma,
            }
        )
    if not common_ids:
        raise ValueError("No common SequenceIndex values were shared by rank gamma sweep sources.")
    common_seq_ids = sorted(common_ids)

    rows = []
    component_consistency_rows = []
    for gamma_idx, gamma in enumerate(gamma_values):
        score_series = []
        source_consistency = []
        for payload in component_payloads:
            features = np.asarray(
                [payload["seq_to_features"][seq_id] for seq_id in common_seq_ids],
                dtype=float,
            )
            fitness = baseline + features @ payload["s_joint_by_gamma"][gamma_idx]
            row = payload["row"]
            series_name = f"{row.get('model_short', row.get('model', 'model'))}_L{int(row['layer'])}"
            score_series.append(pd.Series(fitness, index=common_seq_ids, name=series_name))
            s_reps = payload["s_by_gamma"][gamma_idx]
            pair_corrs = []
            for rep_i in range(s_reps.shape[0]):
                for rep_j in range(rep_i + 1, s_reps.shape[0]):
                    corr = pearsonr(s_reps[rep_i], s_reps[rep_j]).statistic
                    pair_corrs.append(corr)
            consistency = float(np.nanmean(pair_corrs)) if pair_corrs else np.nan
            source_consistency.append(consistency)
            component_consistency_rows.append(
                {
                    "gamma": gamma,
                    "source_idx": payload["source_idx"],
                    "model": row.get("model", ""),
                    "model_short": row.get("model_short", ""),
                    "layer": row.get("layer", np.nan),
                    "mean_pairwise_pearson_r": consistency,
                }
            )
        score_matrix = pd.concat(score_series, axis=1)
        fitness_df = rank_ensemble_fitness_from_score_matrix(score_matrix, aggregation=aggregation)
        fitness_df["annotation"] = fitness_df["SequenceIndex"].astype(str).map(annotation_map)
        spearman_rho, n_score_sequences = spearman_for_fitness_dataframe(fitness_df, scores_dataframe, score_col=score_col)
        auc_metrics = classification_metrics_for_fitness(fitness_df)
        rows.append(
            {
                "gamma": gamma,
                "log10_gamma": np.log10(gamma),
                "aggregation": aggregation,
                "n_component_model_layers": int(len(component_payloads)),
                "n_sequences": int(len(fitness_df)),
                "n_score_sequences": n_score_sequences,
                "spearman_rho": spearman_rho,
                "mean_component_rep_consistency": float(np.nanmean(source_consistency)) if source_consistency else np.nan,
                **auc_metrics,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(component_consistency_rows)


def plot_gamma_spearman_auc_scatter(
    gamma_metrics_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    figure_dpi: int = 300,
):
    plot_df = gamma_metrics_df[
        np.isfinite(gamma_metrics_df["spearman_rho"])
        & np.isfinite(gamma_metrics_df["auc"])
        & np.isfinite(gamma_metrics_df["gamma"])
    ].copy()
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sc = ax.scatter(
        plot_df["spearman_rho"],
        plot_df["auc"],
        c=plot_df["gamma"],
        cmap="viridis",
        norm=mcolors.LogNorm(vmin=plot_df["gamma"].min(), vmax=plot_df["gamma"].max()),
        s=58,
        edgecolors="white",
        linewidths=0.7,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("gamma")
    ax.set_xlabel("MaveDB Spearman rho")
    ax.set_ylabel("ClinVar pathogenic-vs-benign AUC")
    ax.set_title(title)
    ax.grid(color="0.90", linewidth=0.8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    return fig


def attach_sae_reconstruction_metrics(source_rows: pd.DataFrame, sae_metrics_df: pd.DataFrame | None) -> pd.DataFrame:
    rows = source_rows.copy()
    if sae_metrics_df is None or sae_metrics_df.empty:
        return rows
    metrics = sae_metrics_df.copy()
    if "layer_index" not in metrics.columns and "layer" in metrics.columns:
        metrics["layer_index"] = metrics["layer"].astype(str).str.replace("Layer_", "", regex=False).astype(float)
    metrics["layer_index"] = pd.to_numeric(metrics["layer_index"], errors="coerce")
    rows["layer"] = pd.to_numeric(rows["layer"], errors="coerce")
    join_cols = ["model", "model_short"]
    keep_cols = [
        col
        for col in [
            "model",
            "model_short",
            "layer_index",
            "status",
            "reconstruction_mse",
            "reconstruction_r2",
            "activation_sparsity",
            "activation_density",
            "n_active_features",
            "active_feature_fraction",
            "final_train_loss",
            "final_test_loss",
        ]
        if col in metrics.columns
    ]
    merged = rows.merge(
        metrics[keep_cols].rename(columns={"layer_index": "layer"}),
        on=join_cols + ["layer"],
        how="left",
        suffixes=("", "_sae"),
    )
    return merged


def _finite_sort_head(df: pd.DataFrame, sort_cols: list[str], ascending: list[bool], n: int) -> pd.DataFrame:
    finite = df.copy()
    for col in sort_cols:
        finite = finite[np.isfinite(pd.to_numeric(finite[col], errors="coerce"))]
    return finite.sort_values(sort_cols, ascending=ascending).head(n).reset_index(drop=True)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(skipna=True)
    if not np.isfinite(std) or std == 0:
        return values * 0.0
    return (values - values.mean(skipna=True)) / std


def build_esmc_sae_collections(
    metrics_df: pd.DataFrame,
    sae_metrics_df: pd.DataFrame | None = None,
    embedding_type: str = "max_pool",
    top_n: int = 12,
    balanced_per_model: int = 4,
) -> dict[str, pd.DataFrame]:
    base = ensemble_source_rows(
        metrics_df,
        benchmark="Fixed DeltaEmbSAE",
        method="DeltaEmbSAE",
        embedding_type=embedding_type,
        esmc_only=True,
        exclude_layer0=True,
    )
    base = attach_sae_reconstruction_metrics(base, sae_metrics_df)
    collections: dict[str, pd.DataFrame] = {
        "all_esmc_sae_layers": base.reset_index(drop=True),
        f"best_auc_top{top_n}": _finite_sort_head(base, ["auc", "spearman_rho"], [False, False], top_n),
        f"best_mavedb_spearman_top{top_n}": _finite_sort_head(base, ["spearman_rho", "auc"], [False, False], top_n),
    }
    if "reconstruction_r2" in base.columns:
        collections[f"best_reconstruction_r2_top{top_n}"] = _finite_sort_head(
            base,
            ["reconstruction_r2", "auc"],
            [False, False],
            top_n,
        )
        scored = base.copy()
        scored["balanced_score"] = _zscore(scored["auc"]) + _zscore(scored["spearman_rho"]) + _zscore(scored["reconstruction_r2"])
        balanced = (
            scored[np.isfinite(scored["balanced_score"])]
            .sort_values(["model_short", "balanced_score"], ascending=[True, False])
            .groupby("model_short", as_index=False, group_keys=False)
            .head(balanced_per_model)
            .sort_values("balanced_score", ascending=False)
            .reset_index(drop=True)
        )
        collections[f"balanced_auc_spearman_reconstruction_top{balanced_per_model}_per_model"] = balanced
    return {name: rows.reset_index(drop=True) for name, rows in collections.items() if not rows.empty}


def ensemble_metrics_by_review_stars(
    ensemble_specs: list[dict],
    clinvar_cache_path: str | Path,
    scores_dataframe: pd.DataFrame,
    dataset_name: str = "BRCA1",
    primary_key: str = "hgvs_nt",
    thresholds: Iterable[int] = CLINVAR_REVIEW_STAR_THRESHOLDS,
) -> pd.DataFrame:
    rows = []
    for min_review_stars, review_filter in review_star_specs(thresholds):
        annotation_map = clinvar_binary_annotation_map(
            clinvar_cache_path,
            primary_key=primary_key,
            min_review_stars=min_review_stars,
        )
        functional_auc = classification_metrics_for_fitness(
            functional_score_fitness_dataframe(scores_dataframe, annotation_map)
        )["auc"]
        for spec in ensemble_specs:
            fitness_df = spec["fitness_df"].copy()
            fitness_df["SequenceIndex"] = fitness_df["SequenceIndex"].astype(str)
            fitness_df["annotation"] = fitness_df["SequenceIndex"].map(annotation_map)
            spearman_rho, n_score_sequences = spearman_for_fitness_dataframe(fitness_df, scores_dataframe)
            auc_metrics = classification_metrics_for_fitness(fitness_df)
            rows.append(
                {
                    "dataset": dataset_name,
                    "review_filter": review_filter,
                    "min_review_stars": min_review_stars,
                    "method_label": spec["method_label"],
                    "short_label": spec.get("short_label", spec["method_label"]),
                    "benchmark": spec["benchmark"],
                    "method": spec["method"],
                    "embedding_type": spec["embedding_type"],
                    "aggregation": spec["aggregation"],
                    "model_filter": spec.get("model_filter", "ESMC SAE collection"),
                    "n_component_model_layers": spec["n_component_model_layers"],
                    "n_score_sequences": n_score_sequences,
                    "spearman_rho": spearman_rho,
                    "functional_score_auc": functional_auc,
                    "fitness_path": str(spec.get("fitness_path", "")),
                    "feature_path": str(spec.get("feature_path", "")),
                    "inference_path": str(spec.get("inference_path", "")),
                    "gamma_plot_path": str(spec.get("gamma_plot_path", "")),
                    "gamma_table_path": str(spec.get("gamma_table_path", "")),
                    "gamma_pair_table_path": str(spec.get("gamma_pair_table_path", "")),
                    "gamma_selection_objective": str(spec.get("gamma_selection_objective", "")),
                    "selected_gamma_mean_pairwise_pearson_r": spec.get("selected_gamma_mean_pairwise_pearson_r", np.nan),
                    "selected_gamma_mavedb_spearman": spec.get("selected_gamma_mavedb_spearman", np.nan),
                    "selected_gamma_grid": spec.get("selected_gamma_grid", np.nan),
                    "component_table_path": str(spec.get("component_table_path", "")),
                    **auc_metrics,
                }
            )
    return pd.DataFrame(rows)


def plot_ensemble_review_star_panels(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    marker_specs: dict | None = None,
    figure_dpi: int = 300,
):
    plot_df = metrics_df[np.isfinite(metrics_df["spearman_rho"]) & np.isfinite(metrics_df["auc"])].copy()
    if plot_df.empty:
        return None
    marker_specs = marker_specs or {}
    review_order = [label for _, label in review_star_specs() if label in set(plot_df["review_filter"])]
    ncols = min(3, len(review_order))
    nrows = int(np.ceil(len(review_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.35 * ncols, 3.75 * nrows), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.ravel()
    x_values = plot_df["spearman_rho"].to_numpy(dtype=float)
    y_values = np.concatenate([plot_df["auc"].to_numpy(dtype=float), plot_df["functional_score_auc"].dropna().to_numpy(dtype=float)])
    x_pad = max(0.025, 0.12 * (np.nanmax(x_values) - np.nanmin(x_values)))
    y_pad = max(0.025, 0.12 * (np.nanmax(y_values) - np.nanmin(y_values)))
    xlim = (max(-1.0, np.nanmin(x_values) - x_pad), min(1.0, np.nanmax(x_values) + x_pad))
    ylim = (max(0.0, min(np.nanmin(y_values) - y_pad, 0.47)), min(1.0, max(np.nanmax(y_values) + y_pad, 0.53)))
    for ax, review_filter in zip(axes_flat, review_order):
        panel_df = plot_df[plot_df["review_filter"].eq(review_filter)]
        for _, row in panel_df.iterrows():
            spec = marker_specs.get(row["method_label"], {})
            ax.scatter(
                row["spearman_rho"],
                row["auc"],
                s=spec.get("size", 115),
                marker=spec.get("marker", "o"),
                color=spec.get("color", "0.35"),
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )
            ax.annotate(
                row.get("short_label", row["method_label"]),
                (row["spearman_rho"], row["auc"]),
                xytext=spec.get("offset", (6, -9)),
                textcoords="offset points",
                fontsize=7,
                color="0.18",
            )
        functional_auc_values = panel_df["functional_score_auc"].dropna().unique()
        if len(functional_auc_values):
            ax.axhline(functional_auc_values[0], color="0.20", linestyle="-.", linewidth=1.05, zorder=1)
        ax.axhline(0.5, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
        ax.axvline(0.0, color="0.72", linestyle=":", linewidth=0.9, zorder=0)
        ax.set_title(review_filter)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(axis="both", color="0.90", linewidth=0.8)
    for ax in axes_flat[len(review_order):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("MaveDB functional-score Spearman rho")
    for ax in axes[:, 0]:
        ax.set_ylabel("ClinVar pathogenic-vs-benign AUC")
    handles = [
        Line2D(
            [0],
            [0],
            marker=spec.get("marker", "o"),
            color="none",
            markerfacecolor=spec.get("color", "0.35"),
            markeredgecolor="white",
            markersize=8,
            label=label,
        )
        for label, spec in marker_specs.items()
    ]
    handles.append(Line2D([0], [0], color="0.20", linestyle="-.", linewidth=1.05, label="MaveDB functional score AUC"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 3), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.08)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    return fig


def combined_axis_limits(previous_df: pd.DataFrame, ensemble_df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    previous_finite = previous_df[np.isfinite(previous_df["spearman_rho"]) & np.isfinite(previous_df["auc"])]
    ensemble_finite = ensemble_df[np.isfinite(ensemble_df["spearman_rho"]) & np.isfinite(ensemble_df["auc"])]
    x_values = pd.concat([previous_finite["spearman_rho"], ensemble_finite["spearman_rho"]], ignore_index=True).to_numpy(dtype=float)
    y_values = pd.concat(
        [previous_finite["auc"], ensemble_finite["auc"], ensemble_df["functional_score_auc"].dropna()],
        ignore_index=True,
    ).to_numpy(dtype=float)
    x_values = x_values[np.isfinite(x_values)]
    y_values = y_values[np.isfinite(y_values)]
    if len(x_values) == 0:
        x_values = np.asarray([0.0])
    if len(y_values) == 0:
        y_values = np.asarray([0.5])
    x_pad = max(0.025, 0.08 * (float(np.nanmax(x_values)) - float(np.nanmin(x_values))))
    y_pad = max(0.025, 0.08 * (float(np.nanmax(y_values)) - float(np.nanmin(y_values))))
    xlim = (max(-1.0, float(np.nanmin(x_values)) - x_pad), min(1.0, float(np.nanmax(x_values)) + x_pad))
    ylim = (max(0.0, min(float(np.nanmin(y_values)) - y_pad, 0.47)), min(1.0, max(float(np.nanmax(y_values)) + y_pad, 0.53)))
    return xlim, ylim


def previous_comparison_marker_specs() -> dict:
    return {
        "Raw embeddings": {"marker": "D", "size": 30, "color": "#f58518", "label": "Previous raw max_pool layers", "alpha": 0.36},
        "Fixed DeltaEmbSAE": {"marker": "o", "size": 31, "color": "0.55", "label": "Previous fixed SAE layers", "alpha": 0.42},
        "Regular popDMS": {"marker": "*", "size": 145, "color": "#111111", "label": "Regular popDMS", "alpha": 0.95},
        "Enrichment ratio": {"marker": "v", "size": 95, "color": "#009e73", "label": "Enrichment ratio", "alpha": 0.95},
    }


def ensemble_comparison_handles(ensemble_marker_specs: dict) -> list:
    previous_marker_specs = previous_comparison_marker_specs()
    handles = [
        Line2D(
            [0],
            [0],
            marker=spec["marker"],
            color="none",
            markerfacecolor=spec["color"],
            markeredgecolor="white",
            markersize=7,
            alpha=spec["alpha"],
            label=spec["label"],
        )
        for spec in previous_marker_specs.values()
    ]
    handles += [
        Line2D(
            [0],
            [0],
            marker=spec.get("marker", "P"),
            color="none",
            markerfacecolor=spec.get("color", "#4c78a8"),
            markeredgecolor="white",
            markersize=9,
            label=label,
        )
        for label, spec in ensemble_marker_specs.items()
    ]
    handles.append(Line2D([0], [0], color="0.20", linestyle="-.", linewidth=1.05, label="MaveDB functional score AUC"))
    handles.append(Line2D([0], [0], color="0.35", linestyle="--", linewidth=0.9, label="Random AUC = 0.5"))
    return handles


def plot_ensemble_comparison_legend(
    ensemble_marker_specs: dict,
    output_path: str | Path,
    figure_dpi: int = 300,
    ncol: int = 3,
):
    handles = ensemble_comparison_handles(ensemble_marker_specs)
    fig_height = max(1.8, 0.28 * np.ceil(len(handles) / max(ncol, 1)) + 1.1)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=ncol, frameon=False)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    return fig


def _normalized_gamma_objective(row: pd.Series) -> str:
    value = row.get("gamma_selection_objective", "")
    if pd.isna(value) or str(value).strip() == "":
        return "cross_replicate_consistency" if row.get("aggregation") == "pca_concat" else "not_applicable"
    return str(value)


def performance_summary_table(
    metrics_df: pd.DataFrame,
    review_labels: Iterable[str] | None = None,
) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    if review_labels is None:
        review_labels = [review_star_label(0), review_star_label(1), review_star_label(2), review_star_label(3)]
    review_labels = list(review_labels)
    df = metrics_df.copy()
    df["gamma_selection_objective"] = df.apply(_normalized_gamma_objective, axis=1)
    for column in [
        "auc",
        "spearman_rho",
        "functional_score_auc",
        "gamma_opt",
        "selected_gamma_mean_pairwise_pearson_r",
        "selected_gamma_mavedb_spearman",
        "selected_gamma_grid",
        "n_component_model_layers",
        "n_selected_components",
        "n_variants",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    group_cols = [
        "method_label",
        "benchmark",
        "method",
        "aggregation",
        "model_filter",
        "gamma_selection_objective",
    ]
    optional_cols = ["n_component_model_layers", "n_selected_components", "gamma_opt", "selected_gamma_mean_pairwise_pearson_r", "selected_gamma_mavedb_spearman", "selected_gamma_grid", "fitness_path", "feature_path", "inference_path", "gamma_plot_path"]
    summary_rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        first = group.iloc[0]
        for column in optional_cols:
            if column in group.columns:
                value = first.get(column, np.nan)
                if column.startswith("n_") and pd.notna(value):
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        pass
                row[column] = value
        all_binary = group[group["review_filter"].eq(review_star_label(0))]
        if not all_binary.empty:
            row["mavedb_spearman"] = float(all_binary["spearman_rho"].iloc[0])
            row["functional_score_auc"] = float(all_binary["functional_score_auc"].iloc[0]) if "functional_score_auc" in all_binary else np.nan
        for review_label in review_labels:
            panel = group[group["review_filter"].eq(review_label)]
            suffix = (
                "all_binary"
                if review_label == review_star_label(0)
                else review_label.replace(">=", "min").replace(" star(s)", "_stars").replace(" ", "_")
            )
            if panel.empty:
                row[f"auc_{suffix}"] = np.nan
                row[f"n_variants_{suffix}"] = np.nan
            else:
                row[f"auc_{suffix}"] = float(panel["auc"].iloc[0])
                row[f"n_variants_{suffix}"] = int(panel["n_variants"].iloc[0]) if "n_variants" in panel else np.nan
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    sort_cols = [col for col in ["aggregation", "model_filter", "gamma_selection_objective", "method_label"] if col in summary_df.columns]
    if sort_cols:
        summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)
    for column in ["mavedb_spearman", "auc_all_binary"]:
        if column not in summary_df.columns:
            summary_df[column] = np.nan
        summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce")
    summary_df["mavedb_spearman_rank"] = summary_df["mavedb_spearman"].rank(
        method="min",
        ascending=False,
        na_option="bottom",
    )
    summary_df["auc_all_binary_rank"] = summary_df["auc_all_binary"].rank(
        method="min",
        ascending=False,
        na_option="bottom",
    )
    summary_df["spearman_auc_rank_score"] = summary_df[["mavedb_spearman_rank", "auc_all_binary_rank"]].mean(axis=1)
    summary_df["spearman_auc_rank"] = summary_df["spearman_auc_rank_score"].rank(method="min", ascending=True).astype(int)
    summary_df = summary_df.sort_values(
        ["spearman_auc_rank_score", "mavedb_spearman_rank", "auc_all_binary_rank", "method_label"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return summary_df


def gamma_objective_comparison_table(performance_df: pd.DataFrame) -> pd.DataFrame:
    if performance_df.empty:
        return pd.DataFrame()
    pca_df = performance_df[
        performance_df["aggregation"].eq("pca_concat")
        & performance_df["gamma_selection_objective"].isin(["cross_replicate_consistency", "mavedb_spearman"])
    ].copy()
    if pca_df.empty:
        return pd.DataFrame()
    value_cols = [
        "gamma_opt",
        "selected_gamma_mean_pairwise_pearson_r",
        "selected_gamma_mavedb_spearman",
        "mavedb_spearman",
        "auc_all_binary",
        "auc_min_1_stars",
        "auc_min_2_stars",
        "auc_min_3_stars",
    ]
    value_cols = [col for col in value_cols if col in pca_df.columns]
    rows = []
    for model_filter, group in pca_df.groupby("model_filter", dropna=False):
        by_objective = {row["gamma_selection_objective"]: row for _, row in group.iterrows()}
        if "cross_replicate_consistency" not in by_objective or "mavedb_spearman" not in by_objective:
            continue
        row = {"model_filter": model_filter}
        for objective, prefix in [
            ("cross_replicate_consistency", "crossrep"),
            ("mavedb_spearman", "mavedb_gamma"),
        ]:
            source = by_objective[objective]
            row[f"{prefix}_method_label"] = source.get("method_label", "")
            for column in value_cols:
                row[f"{prefix}_{column}"] = source.get(column, np.nan)
        for column in value_cols:
            left = row.get(f"crossrep_{column}", np.nan)
            right = row.get(f"mavedb_gamma_{column}", np.nan)
            if pd.notna(left) and pd.notna(right):
                row[f"delta_{column}"] = right - left
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("model_filter").reset_index(drop=True)


def plot_gamma_objective_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
    figure_dpi: int = 300,
):
    if comparison_df.empty:
        return None
    specs = [
        (
            "selected_gamma_mean_pairwise_pearson_r",
            "Selected-gamma replicate consistency",
        ),
        ("mavedb_spearman", "MaveDB Spearman"),
        ("auc_all_binary", "ClinVar AUC, all binary"),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(5.0 * len(specs), 4.4), squeeze=False)
    axes_flat = axes.ravel()
    for ax, (metric, title) in zip(axes_flat, specs):
        x_col = f"crossrep_{metric}"
        y_col = f"mavedb_gamma_{metric}"
        if x_col not in comparison_df.columns or y_col not in comparison_df.columns:
            ax.axis("off")
            continue
        plot_df = comparison_df[np.isfinite(comparison_df[x_col]) & np.isfinite(comparison_df[y_col])].copy()
        if plot_df.empty:
            ax.text(0.5, 0.5, "No finite data", transform=ax.transAxes, ha="center", va="center")
            ax.axis("off")
            continue
        ax.scatter(plot_df[x_col], plot_df[y_col], s=70, color="#4c78a8", edgecolors="white", linewidths=0.8)
        for _, row in plot_df.iterrows():
            ax.annotate(str(row["model_filter"]).replace("_", "\n"), (row[x_col], row[y_col]), xytext=(5, 4), textcoords="offset points", fontsize=7)
        lo = float(np.nanmin(np.r_[plot_df[x_col].to_numpy(dtype=float), plot_df[y_col].to_numpy(dtype=float)]))
        hi = float(np.nanmax(np.r_[plot_df[x_col].to_numpy(dtype=float), plot_df[y_col].to_numpy(dtype=float)]))
        pad = max(0.01, 0.08 * (hi - lo))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.45", linestyle="--", linewidth=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Cross-replicate gamma")
        ax.set_ylabel("MaveDB-Spearman gamma")
        ax.set_title(title)
        ax.grid(color="0.90", linewidth=0.8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    return fig


def plot_ensemble_with_previous(
    previous_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    ensemble_marker_specs: dict,
    figure_dpi: int = 300,
    review_star_thresholds: Iterable[int] | None = None,
    annotate_points: bool = True,
    include_legend: bool = True,
):
    if review_star_thresholds is None:
        review_star_thresholds = CLINVAR_REVIEW_STAR_THRESHOLDS
    review_order = [review_star_label(0)] + [review_star_label(stars) for stars in review_star_thresholds]
    review_order = [label for label in review_order if label in set(previous_df["review_filter"]) or label in set(ensemble_df["review_filter"])]
    ncols = min(3, max(1, len(review_order)))
    nrows = int(np.ceil(len(review_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.55 * ncols, 3.85 * nrows), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.ravel()
    axis_previous_df = previous_df[previous_df["review_filter"].isin(review_order)].copy()
    axis_ensemble_df = ensemble_df[ensemble_df["review_filter"].isin(review_order)].copy()
    xlim, ylim = combined_axis_limits(axis_previous_df, axis_ensemble_df)
    previous_marker_specs = previous_comparison_marker_specs()
    for ax, review_filter in zip(axes_flat, review_order):
        panel_previous = previous_df[previous_df["review_filter"].eq(review_filter)].copy()
        panel_previous = panel_previous[np.isfinite(panel_previous["spearman_rho"]) & np.isfinite(panel_previous["auc"])]
        panel_ensemble = ensemble_df[ensemble_df["review_filter"].eq(review_filter)].copy()
        for benchmark, spec in previous_marker_specs.items():
            group_df = panel_previous[panel_previous["benchmark"].eq(benchmark)]
            if group_df.empty:
                continue
            ax.scatter(
                group_df["spearman_rho"],
                group_df["auc"],
                s=spec["size"],
                marker=spec["marker"],
                color=spec["color"],
                alpha=spec["alpha"],
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
        functional_auc = panel_previous.loc[panel_previous["benchmark"].eq("DMS functional score"), "auc"].dropna()
        if functional_auc.empty and "functional_score_auc" in panel_ensemble.columns:
            functional_auc = panel_ensemble["functional_score_auc"].dropna()
        if not functional_auc.empty:
            ax.axhline(float(functional_auc.iloc[0]), color="0.20", linestyle="-.", linewidth=1.05, zorder=1)
        finite_ensemble = panel_ensemble[np.isfinite(panel_ensemble["spearman_rho"]) & np.isfinite(panel_ensemble["auc"])]
        for _, row in finite_ensemble.iterrows():
            spec = ensemble_marker_specs.get(row["method_label"], {})
            ax.scatter(
                row["spearman_rho"],
                row["auc"],
                s=spec.get("size", 145),
                marker=spec.get("marker", "P"),
                color=spec.get("color", "#4c78a8"),
                edgecolors="white",
                linewidths=1.0,
                zorder=5,
            )
            if annotate_points:
                ax.annotate(
                    row.get("short_label", row["method_label"]),
                    (row["spearman_rho"], row["auc"]),
                    xytext=spec.get("offset", (6, -9)),
                    textcoords="offset points",
                    fontsize=7,
                    color="0.12",
                )
        if panel_previous.empty and finite_ensemble.empty:
            ax.text(0.5, 0.5, "No finite AUC", transform=ax.transAxes, ha="center", va="center", color="0.35", fontsize=9)
        ax.axhline(0.5, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
        ax.axvline(0.0, color="0.72", linestyle=":", linewidth=0.9, zorder=0)
        ax.set_title(review_filter)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(axis="both", color="0.90", linewidth=0.8)
    for ax in axes_flat[len(review_order):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("MaveDB functional-score Spearman rho")
    for ax in axes[:, 0]:
        ax.set_ylabel("ClinVar pathogenic-vs-benign AUC")
    if include_legend:
        handles = ensemble_comparison_handles(ensemble_marker_specs)
        fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, y=1.04 if not include_legend else 1.09)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    return fig


def _feature_matrix_from_mapping(seq_to_features: dict, seq_ids: list[str]) -> np.ndarray:
    return np.asarray([seq_to_features[seq_id] for seq_id in seq_ids], dtype=np.float32)


def _standardize_matrix(x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.squeeze(0), std.squeeze(0)


def _pca_transform(
    x: np.ndarray,
    n_components: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
        transformed = pca.fit_transform(x)
        return transformed.astype(np.float32), pca.explained_variance_.astype(float), pca.explained_variance_ratio_.astype(float)
    except Exception:
        x_centered = x - x.mean(axis=0, keepdims=True)
        _, s, vt = np.linalg.svd(x_centered, full_matrices=False)
        components = vt[:n_components]
        transformed = x_centered @ components.T
        explained_variance = (s[:n_components] ** 2) / max(x.shape[0] - 1, 1)
        total_variance = np.var(x_centered, axis=0, ddof=1).sum()
        explained_ratio = explained_variance / total_variance if total_variance > 0 else np.zeros_like(explained_variance)
        return transformed.astype(np.float32), explained_variance.astype(float), explained_ratio.astype(float)


def pca_concatenated_feature_mapping(
    source_rows: pd.DataFrame,
    max_components: int = 1000,
    per_source_components: int = 64,
    standardize: bool = True,
    importance_metric: str = "explained_variance_ratio",
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    if source_rows.empty:
        raise ValueError("No source rows were provided for PCA concatenation.")
    if importance_metric not in {"explained_variance", "explained_variance_ratio"}:
        raise ValueError("importance_metric must be 'explained_variance' or 'explained_variance_ratio'.")

    loaded = []
    common_ids: set[str] | None = None
    for source_idx, (_, row) in enumerate(source_rows.reset_index(drop=True).iterrows()):
        feature_path = as_existing_path(row.get("feature_path", ""))
        if feature_path is None:
            raise FileNotFoundError(f"Missing feature_path for PCA source {row.get('model_label', source_idx)!r}")
        seq_to_features = load_pickle(feature_path)
        _, seq_to_features = esmDMS._drop_missing_features(None, seq_to_features, f"PCA source {source_idx}")
        esmDMS._require_vector_features(seq_to_features, f"PCA source {source_idx}")
        seq_id_set = {str(seq_id) for seq_id in seq_to_features}
        common_ids = seq_id_set if common_ids is None else common_ids & seq_id_set
        loaded.append((source_idx, row, {str(k): np.asarray(v, dtype=np.float32) for k, v in seq_to_features.items()}))
    if not common_ids:
        raise ValueError("No common sequence IDs were shared by the selected PCA sources.")
    seq_ids = sorted(common_ids)

    transformed_blocks = []
    component_rows = []
    source_summary_rows = []
    for source_idx, row, seq_to_features in loaded:
        x = _feature_matrix_from_mapping(seq_to_features, seq_ids)
        x_in = x
        if standardize:
            x_in, _, _ = _standardize_matrix(x)
        n_components = min(int(per_source_components), x_in.shape[0] - 1, x_in.shape[1])
        if n_components < 1:
            continue
        transformed, explained_variance, explained_ratio = _pca_transform(x_in, n_components=n_components, seed=seed + source_idx)
        block_idx = len(transformed_blocks)
        transformed_blocks.append(transformed)
        source_summary_rows.append(
            {
                "source_idx": source_idx,
                "block_idx": block_idx,
                "model": row.get("model", ""),
                "model_short": row.get("model_short", ""),
                "layer": row.get("layer", np.nan),
                "model_label": row.get("model_label", ""),
                "feature_path": row.get("feature_path", ""),
                "n_input_features": x.shape[1],
                "n_pca_components": n_components,
                "standardized": standardize,
            }
        )
        for pc_idx in range(n_components):
            component_rows.append(
                {
                    "source_idx": source_idx,
                    "block_idx": block_idx,
                    "pc_idx": pc_idx,
                    "model": row.get("model", ""),
                    "model_short": row.get("model_short", ""),
                    "layer": row.get("layer", np.nan),
                    "model_label": row.get("model_label", ""),
                    "explained_variance": float(explained_variance[pc_idx]),
                    "explained_variance_ratio": float(explained_ratio[pc_idx]),
                    "importance": float(explained_ratio[pc_idx] if importance_metric == "explained_variance_ratio" else explained_variance[pc_idx]),
                }
            )

    component_df = pd.DataFrame(component_rows)
    if component_df.empty:
        raise ValueError("No PCA components were produced.")
    selected = component_df.sort_values(["importance", "explained_variance"], ascending=[False, False]).head(max_components).copy()
    selected = selected.reset_index(drop=True)
    selected["concat_component_idx"] = np.arange(len(selected))
    concat_columns = [
        transformed_blocks[int(row.block_idx)][:, int(row.pc_idx)]
        for row in selected.itertuples(index=False)
    ]
    concat = np.column_stack(concat_columns).astype(np.float32)
    feature_mapping = {seq_id: concat[idx] for idx, seq_id in enumerate(seq_ids)}
    component_df = component_df.merge(
        selected[["source_idx", "pc_idx", "concat_component_idx"]],
        on=["source_idx", "pc_idx"],
        how="left",
    )
    component_df["selected"] = component_df["concat_component_idx"].notna()
    source_summary_df = pd.DataFrame(source_summary_rows)
    return feature_mapping, component_df, source_summary_df


def plot_gamma_consistency_for_features(
    sequence_dataframe: pd.DataFrame,
    seq_to_features: dict,
    output_path: str | Path,
    gamma_values: np.ndarray | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    sequence_dataframe, seq_to_features = esmDMS._drop_missing_features(
        sequence_dataframe,
        seq_to_features,
        "PCA concatenated gamma plot",
    )
    gamma_values, s_by_gamma, _ = infer_gamma_range(sequence_dataframe, seq_to_features, gamma_values=gamma_values)
    rows = []
    for gamma_idx, gamma in enumerate(gamma_values):
        pair_corrs = []
        for rep_i in range(s_by_gamma.shape[1]):
            for rep_j in range(rep_i + 1, s_by_gamma.shape[1]):
                corr = pearsonr(s_by_gamma[gamma_idx, rep_i], s_by_gamma[gamma_idx, rep_j]).statistic
                pair_corrs.append(corr)
                rows.append({"gamma": gamma, "rep_i": rep_i + 1, "rep_j": rep_j + 1, "pearson_r": corr})
        rows.append(
            {
                "gamma": gamma,
                "rep_i": "mean",
                "rep_j": "mean",
                "pearson_r": float(np.nanmean(pair_corrs)) if pair_corrs else np.nan,
            }
        )
    reg_df = pd.DataFrame(rows)
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mean_df = reg_df[reg_df["rep_i"].eq("mean")]
    ax.plot(mean_df["gamma"], mean_df["pearson_r"], marker="o", label="Mean pairwise r")
    pair_df = reg_df[~reg_df["rep_i"].eq("mean")]
    for (rep_i, rep_j), pair_rows in pair_df.groupby(["rep_i", "rep_j"]):
        ax.plot(pair_rows["gamma"], pair_rows["pearson_r"], alpha=0.35, linewidth=1, label=f"Rep {rep_i} vs {rep_j}")
    ax.set_xscale("log")
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Regularization strength (gamma)")
    ax.set_ylabel("Selection coefficient Pearson r")
    ax.set_title("PCA-concatenated SAE regularization sweep")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    return fig, reg_df


def gamma_diagnostics_for_features(
    sequence_dataframe: pd.DataFrame,
    seq_to_features: dict,
    scores_dataframe: pd.DataFrame,
    gamma_values: np.ndarray | None = None,
    score_col: str = "score",
    max_reads: float = 1e2,
    baseline: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sequence_dataframe, seq_to_features = esmDMS._drop_missing_features(
        sequence_dataframe,
        seq_to_features,
        "Gamma diagnostics",
    )
    esmDMS._require_vector_features(seq_to_features, "Gamma diagnostics")
    seq_ids = [str(seq_id) for seq_id in seq_to_features]
    features = np.asarray([seq_to_features[seq_id] for seq_id in seq_to_features], dtype=float)
    score_df = scores_dataframe[["SequenceIndex", score_col]].copy()
    score_df["SequenceIndex"] = score_df["SequenceIndex"].astype(str)
    score_map = score_df.set_index("SequenceIndex")[score_col]
    scores = pd.to_numeric(pd.Series(seq_ids).map(score_map), errors="coerce").to_numpy(dtype=float)

    n_replicates = len(sequence_dataframe["Replicate"].unique())
    if gamma_values is None:
        gamma_values = np.logspace(np.log10(1 / max_reads), 4, num=20)
    gamma_values = np.asarray(gamma_values, dtype=float)

    dx, icov, _ = compute_dx_covariance_esm(sequence_dataframe, seq_to_features)
    _, eig_lam, eig_vec, vt_dx, lam_j, v_j, vt_dx_j = _prepare_eigendecomp(dx, icov, n_replicates)

    summary_rows = []
    pair_rows = []
    for gamma in gamma_values:
        s_reps = np.zeros((n_replicates, len(dx[0])))
        for rep_idx in range(n_replicates):
            s_reps[rep_idx] = eig_vec[rep_idx] @ (vt_dx[rep_idx] / (eig_lam[rep_idx] + gamma))
        pair_corrs = []
        for rep_i in range(n_replicates):
            for rep_j in range(rep_i + 1, n_replicates):
                corr = pearsonr(s_reps[rep_i], s_reps[rep_j]).statistic
                pair_corrs.append(corr)
                pair_rows.append(
                    {
                        "gamma": gamma,
                        "rep_i": rep_i + 1,
                        "rep_j": rep_j + 1,
                        "pearson_r": corr,
                    }
                )
        s_joint = v_j @ (vt_dx_j / (lam_j + n_replicates * gamma))
        fitness = baseline + features @ s_joint
        finite = np.isfinite(fitness) & np.isfinite(scores)
        if finite.sum() >= 3:
            mavedb_spearman = float(spearmanr(fitness[finite], scores[finite]).statistic)
        else:
            mavedb_spearman = np.nan
        summary_rows.append(
            {
                "gamma": gamma,
                "mean_pairwise_pearson_r": float(np.nanmean(pair_corrs)) if pair_corrs else np.nan,
                "std_pairwise_pearson_r": float(np.nanstd(pair_corrs)) if pair_corrs else np.nan,
                "mavedb_spearman": mavedb_spearman,
                "n_score_sequences": int(finite.sum()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    pair_df = pd.DataFrame(pair_rows)
    metadata = {
        "n_replicates": n_replicates,
        "n_features": int(features.shape[1]),
        "n_sequences": int(features.shape[0]),
    }
    return summary_df, pair_df, metadata


def select_gamma_from_diagnostics(
    diagnostics_df: pd.DataFrame,
    objective: str,
    cross_replicate_gamma: float | None = None,
) -> tuple[float, dict]:
    objective = str(objective)
    if objective == "cross_replicate_consistency":
        if cross_replicate_gamma is None:
            finite = diagnostics_df[np.isfinite(diagnostics_df["mean_pairwise_pearson_r"])].copy()
            if finite.empty:
                raise ValueError("No finite cross-replicate consistency values are available for gamma selection.")
            row = finite.sort_values("mean_pairwise_pearson_r", ascending=False).iloc[0]
            gamma = float(row["gamma"])
        else:
            gamma = float(cross_replicate_gamma)
            row = nearest_gamma_diagnostics_row(diagnostics_df, gamma)
    elif objective == "mavedb_spearman":
        finite = diagnostics_df[np.isfinite(diagnostics_df["mavedb_spearman"])].copy()
        if finite.empty:
            raise ValueError("No finite MaveDB Spearman values are available for gamma selection.")
        row = finite.sort_values(
            ["mavedb_spearman", "mean_pairwise_pearson_r"],
            ascending=[False, False],
        ).iloc[0]
        gamma = float(row["gamma"])
    else:
        raise ValueError(
            "gamma_selection_objective must be 'cross_replicate_consistency' or 'mavedb_spearman'."
        )
    return gamma, {
        "selected_gamma_grid": float(row["gamma"]),
        "selected_gamma_mean_pairwise_pearson_r": float(row.get("mean_pairwise_pearson_r", np.nan)),
        "selected_gamma_mavedb_spearman": float(row.get("mavedb_spearman", np.nan)),
    }


def nearest_gamma_diagnostics_row(diagnostics_df: pd.DataFrame, gamma: float) -> pd.Series:
    if diagnostics_df.empty:
        raise ValueError("diagnostics_df is empty.")
    gamma_values = pd.to_numeric(diagnostics_df["gamma"], errors="coerce").to_numpy(dtype=float)
    idx = int(np.nanargmin(np.abs(np.log10(gamma_values) - np.log10(float(gamma)))))
    return diagnostics_df.iloc[idx]


def plot_gamma_diagnostics(
    diagnostics_df: pd.DataFrame,
    selected_gamma: float,
    output_path: str | Path,
    objective: str,
    title: str = "PCA-concatenated SAE gamma diagnostics",
) -> tuple[plt.Figure, pd.DataFrame]:
    plot_df = diagnostics_df.copy()
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(
        plot_df["gamma"],
        plot_df["mean_pairwise_pearson_r"],
        marker="o",
        color="#4c78a8",
        label="Mean replicate consistency",
    )
    ax.plot(
        plot_df["gamma"],
        plot_df["mavedb_spearman"],
        marker="s",
        color="#f58518",
        label="MaveDB Spearman",
    )
    ax.axvline(selected_gamma, color="0.15", linestyle="--", linewidth=1.1, label=f"Selected gamma ({objective})")
    ax.set_xscale("log")
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Regularization strength (gamma)")
    ax.set_ylabel("Correlation")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    return fig, plot_df


def run_pca_concatenated_sae_inference(
    source_rows: pd.DataFrame,
    sequence_dataframe: pd.DataFrame,
    output_dir: str | Path,
    run_label: str,
    max_components: int = 1000,
    per_source_components: int = 64,
    standardize: bool = True,
    importance_metric: str = "explained_variance_ratio",
    seed: int = 42,
    gamma_values: np.ndarray | None = None,
    gamma_selection_objective: str = "cross_replicate_consistency",
    scores_dataframe: pd.DataFrame | None = None,
    score_col: str = "score",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = safe_file_component(run_label)
    feature_path = output_dir / f"{safe_label}_seq_to_features.pkl"
    inference_path = output_dir / f"{safe_label}_inference_results.pkl"
    component_table_path = output_dir / f"{safe_label}_pca_components.csv"
    source_table_path = output_dir / f"{safe_label}_sources.csv"
    gamma_objective_label = safe_file_component(gamma_selection_objective)
    gamma_plot_path = output_dir / f"{safe_label}_{gamma_objective_label}_gamma_diagnostics.png"
    gamma_table_path = output_dir / f"{safe_label}_{gamma_objective_label}_gamma_diagnostics.csv"
    gamma_pair_table_path = output_dir / f"{safe_label}_{gamma_objective_label}_gamma_pair_consistency.csv"

    seq_to_features, component_df, source_summary_df = pca_concatenated_feature_mapping(
        source_rows,
        max_components=max_components,
        per_source_components=per_source_components,
        standardize=standardize,
        importance_metric=importance_metric,
        seed=seed,
    )
    save_pickle(seq_to_features, feature_path)
    component_df.to_csv(component_table_path, index=False)
    source_summary_df.to_csv(source_table_path, index=False)

    inference_sequence_df, inference_features = esmDMS._drop_missing_features(
        sequence_dataframe,
        seq_to_features,
        "PCA concatenated inference",
    )
    if scores_dataframe is None:
        result = mini_infer_esm(inference_sequence_df, inference_features)
        selected_gamma = float(result.gamma_opt)
        diagnostics_summary = {
            "selected_gamma_grid": np.nan,
            "selected_gamma_mean_pairwise_pearson_r": np.nan,
            "selected_gamma_mavedb_spearman": np.nan,
        }
        fig, gamma_df = plot_gamma_consistency_for_features(
            inference_sequence_df,
            inference_features,
            gamma_plot_path,
            gamma_values=gamma_values,
        )
        gamma_pair_df = gamma_df[~gamma_df["rep_i"].eq("mean")].copy()
        gamma_df = gamma_df[gamma_df["rep_i"].eq("mean")].rename(columns={"pearson_r": "mean_pairwise_pearson_r"})
    else:
        diagnostics_df, gamma_pair_df, _ = gamma_diagnostics_for_features(
            inference_sequence_df,
            inference_features,
            scores_dataframe=scores_dataframe,
            gamma_values=gamma_values,
            score_col=score_col,
        )
        if gamma_selection_objective == "cross_replicate_consistency":
            cross_rep_result = mini_infer_esm(inference_sequence_df, inference_features)
            selected_gamma, diagnostics_summary = select_gamma_from_diagnostics(
                diagnostics_df,
                gamma_selection_objective,
                cross_replicate_gamma=float(cross_rep_result.gamma_opt),
            )
            result = cross_rep_result
        else:
            selected_gamma, diagnostics_summary = select_gamma_from_diagnostics(
                diagnostics_df,
                gamma_selection_objective,
            )
            result = mini_infer_esm(inference_sequence_df, inference_features, gamma=selected_gamma)
        fig, gamma_df = plot_gamma_diagnostics(
            diagnostics_df,
            selected_gamma=selected_gamma,
            output_path=gamma_plot_path,
            objective=gamma_selection_objective,
        )
    save_pickle(result, inference_path)
    plt.close(fig)
    gamma_df.to_csv(gamma_table_path, index=False)
    gamma_pair_df.to_csv(gamma_pair_table_path, index=False)
    return {
        "feature_path": feature_path,
        "inference_path": inference_path,
        "component_table_path": component_table_path,
        "source_table_path": source_table_path,
        "gamma_plot_path": gamma_plot_path,
        "gamma_table_path": gamma_table_path,
        "gamma_pair_table_path": gamma_pair_table_path,
        "n_sources": len(source_rows),
        "n_sequences": len(seq_to_features),
        "n_selected_components": int(component_df["selected"].sum()),
        "gamma_opt": float(result.gamma_opt),
        "gamma_selection_objective": gamma_selection_objective,
        **diagnostics_summary,
        "inference_result": result,
        "seq_to_features": seq_to_features,
    }


def load_previous_metrics_for_combined_plot(
    all_binary_table_path: str | Path,
    table_dir: str | Path,
    star_combined_table_path: str | Path | None = None,
    star_table_template: str = "BRCA1_plm_layer_mavedb_spearman_vs_clinvar_auc_min_{stars}_stars.csv",
) -> pd.DataFrame:
    all_binary_table_path = Path(all_binary_table_path)
    if not all_binary_table_path.is_file():
        raise FileNotFoundError(f"Missing benchmark metrics table: {all_binary_table_path}")
    frames = [pd.read_csv(all_binary_table_path).assign(min_review_stars=0)]
    if star_combined_table_path is not None and Path(star_combined_table_path).is_file():
        frames.append(pd.read_csv(star_combined_table_path))
    else:
        star_frames = []
        for min_review_stars in CLINVAR_REVIEW_STAR_THRESHOLDS:
            star_path = Path(table_dir) / star_table_template.format(stars=min_review_stars)
            if star_path.is_file():
                star_frames.append(pd.read_csv(star_path).assign(min_review_stars=min_review_stars))
        if star_frames:
            frames.append(pd.concat(star_frames, ignore_index=True, sort=False))
    previous_df = pd.concat(frames, ignore_index=True, sort=False)
    previous_df["min_review_stars"] = pd.to_numeric(previous_df["min_review_stars"], errors="coerce").fillna(0).astype(int)
    previous_df["review_filter"] = previous_df["min_review_stars"].map(review_star_label)
    previous_df["auc"] = pd.to_numeric(previous_df["auc"], errors="coerce")
    previous_df["spearman_rho"] = pd.to_numeric(previous_df.get("spearman_rho", np.nan), errors="coerce")
    if "plot_group" not in previous_df.columns:
        previous_df["plot_group"] = previous_df["benchmark"]
    return previous_df


def active_slurm_jobs_by_name(job_names: Iterable[str]) -> dict[str, list[dict]]:
    wanted = set(job_names)
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not user:
        return {}
    try:
        completed = subprocess.run(
            ["squeue", "-h", "-u", user, "-o", "%j|%T|%i"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    active: dict[str, list[dict]] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        job_name, state, job_id = (line.split("|", 2) + ["", ""])[:3]
        job_name = job_name.strip()
        if job_name in wanted:
            active.setdefault(job_name, []).append({"job_id": job_id.strip(), "state": state.strip(), "job_name": job_name})
    return active


def active_slot_limits(items: list[str], total_slots: int) -> dict[str, int]:
    if not items:
        return {}
    if len(items) > total_slots:
        return {item: 1 for item in items}
    base = total_slots // len(items)
    extra = total_slots % len(items)
    return {item: base + (idx < extra) for idx, item in enumerate(items)}


def write_slurm_array_job(
    script_path: str | Path,
    payload_path: str | Path,
    runner_script: str | Path,
    n_tasks: int,
    job_name: str,
    repo_root: str | Path,
    log_dir: str | Path,
    partition: str = "any_cpu",
    cpus_per_task: int = 4,
    mem: str = "32G",
    time: str = "06:00:00",
    max_active_tasks: int | None = None,
    python_executable: str = "python3",
) -> Path:
    if n_tasks < 1:
        raise ValueError("n_tasks must be at least 1.")
    script_path = Path(script_path)
    payload_path = Path(payload_path)
    runner_script = Path(runner_script)
    repo_root = Path(repo_root)
    log_dir = Path(log_dir)
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
cd {repo_root}
export PYTHONUNBUFFERED=1
{python_executable} {runner_script} {payload_path} "${{SLURM_ARRAY_TASK_ID}}"
"""
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def write_json_summary(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


# Compatibility names for helpers that originally lived inside lower_context.ipynb.
_model_cache_label = model_cache_label
_safe_file_component = safe_file_component
_n_transformer_layers = n_transformer_layers
_model_family = model_family
_model_job_defaults = model_job_defaults
_active_slot_limits = active_slot_limits
_as_existing_path = as_existing_path
_load_pickle = load_pickle
_reconstruction_metrics = reconstruction_metrics
_clinvar_annotation_table = clinvar_annotation_table
_clinvar_binary_annotation_map = clinvar_binary_annotation_map
_classification_metrics_for_fitness = classification_metrics_for_fitness
_spearman_for_fitness_dataframe = spearman_for_fitness_dataframe
_functional_score_fitness_dataframe = functional_score_fitness_dataframe
_enrichment_ratio_fitness_dataframe = enrichment_ratio_fitness_dataframe
_regular_popdms_fitness_dataframe = regular_popdms_fitness_dataframe
_fitness_for_feature_mapping = fitness_for_feature_mapping
_metrics_row_from_fitness = metrics_row_from_fitness
_review_star_specs = review_star_specs
_combined_review_star_label = review_star_label
_ensemble_source_rows = ensemble_source_rows
_load_ensemble_fitness_series = load_ensemble_fitness_series
_aggregate_fitness_rows = aggregate_fitness_rows
_ensemble_metrics_by_review_stars = ensemble_metrics_by_review_stars
_plot_ensemble_review_star_panels = plot_ensemble_review_star_panels
_combined_axis_limits = combined_axis_limits
_plot_ensemble_with_previous = plot_ensemble_with_previous


def _embedding_job_name(spec: dict) -> str:
    return safe_file_component(f"brca1_{spec['model_short']}_embed")[:48]


def _active_slurm_embedding_jobs(models: Iterable[str], skip_active: bool = True) -> dict[str, list[dict]]:
    if not skip_active:
        return {}
    specs_by_model = {spec["model"]: spec for spec in build_model_specs()}
    wanted_names = {
        _embedding_job_name(specs_by_model[model_name]): model_name
        for model_name in models
        if model_name in specs_by_model
    }
    active_by_name = active_slurm_jobs_by_name(wanted_names)
    return {
        wanted_names[job_name]: rows
        for job_name, rows in active_by_name.items()
        if job_name in wanted_names
    }


def _embedding_cache_complete(model_runner: esmDMS, spec: dict, embedding_type: str = "max_pool") -> bool:
    return all(model_runner._embedding_path(layer, embedding_type).is_file() for layer in spec["layers"])


def _load_ensemble_metrics_table(table_path: str | Path, metrics_df: pd.DataFrame | None = None) -> pd.DataFrame:
    return load_metrics_table(table_path, metrics_df)


def _load_previous_metrics_for_combined_ensemble_plot(
    all_binary_table_path: str | Path,
    table_dir: str | Path,
    star_combined_table_path: str | Path | None = None,
) -> pd.DataFrame:
    return load_previous_metrics_for_combined_plot(
        all_binary_table_path=all_binary_table_path,
        table_dir=table_dir,
        star_combined_table_path=star_combined_table_path,
    )


def _best_worst_rank_fitness_dataframe(source_rows: pd.DataFrame, rank_mode: str) -> tuple[pd.DataFrame, int]:
    if rank_mode == "best_rank":
        return aggregate_fitness_rows(source_rows, "rank_best")
    if rank_mode == "worst_rank":
        return aggregate_fitness_rows(source_rows, "rank_worst")
    raise ValueError(f"Unsupported rank_mode: {rank_mode}")
