#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import clinprotgym_esmc_sae_pipeline as pipeline  # noqa: E402


DEFAULT_COUNT_READY_DATASETS = (
    "MV_BRCA1_Findlay_2018",
    "MV_BRCA2_Huang_2025",
    "MV_MSH2_Jia_2020",
    "MV_TP53_Kotler_2018",
)

DEFAULT_PROCESSED_VARIANT_PATHS = (
    REPO_ROOT / "data/clin_dms_data/data/processed/mavedb_clinical/variants.csv",
    REPO_ROOT / "data/clin_dms_data/data/processed/mavedb_native/variants.csv",
)

DEFAULT_CLINVAR_VARIANT_SUMMARY = (
    REPO_ROOT / "data/clin_dms_data/data/raw/clinvar/variant_summary_2026-06.txt.gz"
)

DATASET_GENE = {
    "MV_BRCA1_Findlay_2018": "BRCA1",
    "MV_BRCA2_Huang_2025": "BRCA2",
    "MV_BRCA2_Sahu_2025": "BRCA2",
    "MV_BRCA2_Sahu_2025__last2048": "BRCA2",
    "MV_BRCA2_Sahu_2025__sliding2048_overlap1024": "BRCA2",
    "MV_GCK_Gersing_2022_activity": "GCK",
    "MV_GCK_Gersing_2023_abundance": "GCK",
    "MV_KCNH2_Kroncke_2024": "KCNH2",
    "MV_LDLR_Clausen_2025": "LDLR",
    "MV_MSH2_Jia_2020": "MSH2",
    "MV_PALB2_SGE_2024": "PALB2",
    "MV_PTEN_Matreyek_2018": "PTEN",
    "MV_PTEN_Mighell_2018": "PTEN",
    "MV_TP53_Kotler_2018": "TP53",
    "MV_VHL_Buckley_2024": "VHL",
}

HGVS_MAP_COLUMNS = [
    "dataset",
    "protein_sequence_index",
    "mutant",
    "original_mutant",
    "functional_score",
    "mavedb_accession",
    "hgvs_nt",
    "hgvs_pro",
    "post_mapped_hgvs_p",
    "clingen_allele_id",
    "mapping_source",
]

ANNOTATION_COLUMNS_TO_REPLACE = [
    "annotation",
    "stars",
    "has_clinvar",
    "clinvar_significance",
    "clinvar_significance_normalized",
    "clinvar_review_status",
    "clinvar_variation_ids",
    "clinvar_allele_ids",
    "clinvar_conditions",
    "clinvar_record_count",
    "annotation_source",
]


@dataclass
class RefreshStats:
    table_path: Path
    rows: int = 0
    rows_with_fitness_path: int = 0
    refreshed_rows: int = 0
    hgvs_rows: int = 0
    protein_rows: int = 0
    missing_fitness_files: int = 0
    unusable_fitness_files: int = 0


@dataclass
class HgvsContext:
    dataset: str
    active: bool
    reason: str
    variant_map: pd.DataFrame
    annotations: pd.DataFrame
    variant_map_path: Path
    annotation_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh ClinProtGym AUC/count metrics after ClinVar annotations changed, "
            "without rerunning benchmark or ensemble jobs. The HGVS scheme expands "
            "protein-level fitness files back to exact nucleotide HGVS rows when "
            "hgvs_nt is recoverable from processed MaveDB variants."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_COUNT_READY_DATASETS),
        help="Datasets to refresh. Defaults to the current count-ready dataset set.",
    )
    parser.add_argument(
        "--annotation-scheme",
        choices=["auto", "hgvs", "protein"],
        default="auto",
        help=(
            "auto uses exact nucleotide-HGVS ClinVar annotations when available and "
            "falls back to protein annotations otherwise. hgvs requests HGVS only, "
            "with fallback noted in the audit table if hgvs_nt is not available."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=str(pipeline.DEFAULT_INPUT_DIR),
        help="Directory containing updated ClinProtGym final CSVs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(pipeline.DEFAULT_OUTPUT_ROOT),
        help="ClinProtGym ESM-C SAE output root.",
    )
    parser.add_argument(
        "--processed-variants",
        action="append",
        type=Path,
        default=None,
        help=(
            "Processed variants.csv with HGVS columns. Repeatable. Defaults to the "
            "mavedb_clinical and mavedb_native processed variant tables."
        ),
    )
    parser.add_argument(
        "--clinvar-variant-summary",
        type=Path,
        default=DEFAULT_CLINVAR_VARIANT_SUMMARY,
        help="ClinVar variant_summary.txt.gz used to rebuild exact HGVS annotations.",
    )
    parser.add_argument(
        "--hgvs-cache-dir",
        type=Path,
        default=None,
        help="Directory for per-dataset exact-HGVS ClinVar annotation caches.",
    )
    parser.add_argument(
        "--force-rebuild-hgvs-cache",
        action="store_true",
        help="Rescan the ClinVar bulk file even if a per-dataset HGVS cache exists.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip reparsing final CSVs. Use only if processed states already contain the updated annotations.",
    )
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Refresh tables only; do not redraw summary plots.",
    )
    parser.add_argument(
        "--sahu-brca2-final",
        choices=["both", "last2048", "sliding-window", "none"],
        default="both",
        help="Passed through to summarize if Sahu BRCA2 derived datasets are included.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def is_blank(value: object) -> bool:
    return not clean_text(value)


def first_present(row: pd.Series, columns: list[str]) -> str:
    for column in columns:
        if column in row and not is_blank(row[column]):
            return clean_text(row[column])
    return ""


def relative_display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def base_dataset_name(dataset: str) -> str:
    return str(dataset).split("__", 1)[0]


def infer_gene(dataset: str) -> str:
    if dataset in DATASET_GENE:
        return DATASET_GENE[dataset]
    base = base_dataset_name(dataset)
    if base in DATASET_GENE:
        return DATASET_GENE[base]
    parts = str(dataset).split("_")
    return parts[1] if len(parts) > 1 else ""


def hgvs_nt_key(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    match = re.search(r"c\.[^\s,;()]+", text)
    return match.group(0) if match else ""


def normalize_clinvar_significance(value: object) -> str:
    text = clean_text(value).lower()
    if not text:
        return "other"
    if "conflict" in text:
        return "conflicting"
    if "pathogenic" in text:
        return "pathogenic"
    if "benign" in text:
        return "benign"
    if "uncertain" in text:
        return "uncertain"
    return "other"


def exact_binary_annotation(normalized_values: set[str]) -> str | None:
    values = {value for value in normalized_values if value}
    if values == {"benign"}:
        return "benign"
    if values == {"pathogenic"}:
        return "pathogenic"
    return None


def joined_unique(records: list[dict[str, str]], key: str) -> str:
    return "|".join(sorted({clean_text(row.get(key, "")) for row in records if clean_text(row.get(key, ""))}))


def load_processed_variants(paths: list[Path] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths or list(DEFAULT_PROCESSED_VARIANT_PATHS):
        if not path.is_file():
            continue
        frame = pd.read_csv(path, dtype=str, low_memory=False)
        frame["processed_variants_path"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def run_prepare(datasets: list[str], input_dir: Path, output_root: Path) -> None:
    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_root=str(output_root),
        datasets=datasets,
        keep_stop=False,
        force=True,
    )
    pipeline.prepare(args)


def score_key(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(12).astype("string")


def standardize_variant_map(frame: pd.DataFrame, dataset: str, source: str) -> pd.DataFrame:
    out = frame.copy()
    out["dataset"] = dataset
    out["mapping_source"] = source
    for column in HGVS_MAP_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    out = out[HGVS_MAP_COLUMNS].copy()
    out["protein_sequence_index"] = out["protein_sequence_index"].astype(str)
    out["hgvs_nt"] = out["hgvs_nt"].map(clean_text)
    out = out[out["hgvs_nt"].astype(bool)].copy()
    out = out.drop_duplicates(["protein_sequence_index", "hgvs_nt"], keep="first")
    return out


def build_hgvs_variant_map(dataset: str, output_root: Path, processed_variants: pd.DataFrame) -> pd.DataFrame:
    paths = pipeline.dataset_paths(output_root, dataset)
    if not paths.counts_path.is_file():
        return pd.DataFrame(columns=HGVS_MAP_COLUMNS)

    counts = pd.read_csv(paths.counts_path, dtype=str, low_memory=False)
    if "SequenceIndex" not in counts.columns:
        return pd.DataFrame(columns=HGVS_MAP_COLUMNS)

    if "hgvs_nt" in counts.columns and counts["hgvs_nt"].map(clean_text).astype(bool).any():
        direct = pd.DataFrame(
            {
                "protein_sequence_index": counts["SequenceIndex"].astype(str),
                "mutant": counts.get("mutant", ""),
                "original_mutant": counts.get("original_mutant", counts.get("mutant", "")),
                "functional_score": counts.get("functional_score", ""),
                "mavedb_accession": counts.get("mavedb_accession", ""),
                "hgvs_nt": counts.get("hgvs_nt", ""),
                "hgvs_pro": counts.get("hgvs_pro", ""),
                "post_mapped_hgvs_p": counts.get("post_mapped_hgvs_p", ""),
                "clingen_allele_id": counts.get("clingen_allele_id", ""),
            }
        )
        return standardize_variant_map(direct, dataset, "adapter_counts_hgvs_nt")

    if processed_variants.empty or "dataset_id" not in processed_variants.columns:
        return pd.DataFrame(columns=HGVS_MAP_COLUMNS)

    source_dataset = base_dataset_name(dataset)
    variants = processed_variants[processed_variants["dataset_id"].eq(source_dataset)].copy()
    if variants.empty or "hgvs_nt" not in variants.columns:
        return pd.DataFrame(columns=HGVS_MAP_COLUMNS)
    variants = variants[variants["hgvs_nt"].map(clean_text).astype(bool)].copy()
    if variants.empty:
        return pd.DataFrame(columns=HGVS_MAP_COLUMNS)

    left = counts.copy()
    left["protein_sequence_index"] = left["SequenceIndex"].astype(str)
    left["join_mutant"] = left.apply(
        lambda row: first_present(row, ["original_mutant", "mutant"]), axis=1
    )
    right = variants.copy()
    right["join_mutant"] = right["mutant"].map(clean_text)

    left_has_score = "functional_score" in left.columns and left["functional_score"].map(clean_text).astype(bool).any()
    right_has_score = "mavedb_score" in right.columns and right["mavedb_score"].map(clean_text).astype(bool).any()
    if left_has_score and right_has_score:
        left["join_score"] = score_key(left["functional_score"])
        right["join_score"] = score_key(right["mavedb_score"])
        merged = left.merge(right, on=["join_mutant", "join_score"], how="left", suffixes=("", "_variant"))
        source = "processed_variants_mutant_functional_score"
    else:
        merged = left.merge(right, on="join_mutant", how="left", suffixes=("", "_variant"))
        source = "processed_variants_mutant"

    mapped = pd.DataFrame(
        {
            "protein_sequence_index": merged["protein_sequence_index"],
            "mutant": merged.get("mutant_variant", merged.get("mutant", "")),
            "original_mutant": merged.get("original_mutant", merged.get("mutant", "")),
            "functional_score": merged.get("functional_score", ""),
            "mavedb_accession": merged.get("mavedb_accession", ""),
            "hgvs_nt": merged.get("hgvs_nt", ""),
            "hgvs_pro": merged.get("hgvs_pro", ""),
            "post_mapped_hgvs_p": merged.get("post_mapped_hgvs_p", ""),
            "clingen_allele_id": merged.get("clingen_allele_id", ""),
        }
    )
    return standardize_variant_map(mapped, dataset, source)


def aggregate_clinvar_records(hgvs_nt: str, records: list[dict[str, str]]) -> dict[str, object]:
    if not records:
        return {
            "hgvs_nt": hgvs_nt,
            "has_clinvar": False,
            "clinvar_significance": "",
            "clinvar_significance_normalized": "",
            "annotation": None,
            "clinvar_review_status": "",
            "stars": 0,
            "clinvar_variation_ids": "",
            "clinvar_allele_ids": "",
            "clinvar_conditions": "",
            "clinvar_record_count": 0,
            "annotation_source": "clinvar_hgvs_nt",
        }

    normalized = {normalize_clinvar_significance(row.get("ClinicalSignificance", "")) for row in records}
    review_status = joined_unique(records, "ReviewStatus")
    return {
        "hgvs_nt": hgvs_nt,
        "has_clinvar": True,
        "clinvar_significance": joined_unique(records, "ClinicalSignificance"),
        "clinvar_significance_normalized": "|".join(sorted(normalized)),
        "annotation": exact_binary_annotation(normalized),
        "clinvar_review_status": review_status,
        "stars": pipeline.clinvar_review_status_to_stars(review_status),
        "clinvar_variation_ids": joined_unique(records, "VariationID"),
        "clinvar_allele_ids": joined_unique(records, "#AlleleID"),
        "clinvar_conditions": joined_unique(records, "PhenotypeList"),
        "clinvar_record_count": len(records),
        "annotation_source": "clinvar_hgvs_nt",
    }


def build_hgvs_annotation_cache(
    dataset: str,
    variant_map: pd.DataFrame,
    clinvar_path: Path,
    cache_path: Path,
    force: bool,
) -> pd.DataFrame:
    if cache_path.is_file() and not force:
        return pd.read_csv(cache_path, dtype={"hgvs_nt": str})

    hgvs_values = sorted({clean_text(value) for value in variant_map.get("hgvs_nt", []) if clean_text(value)})
    if not hgvs_values:
        return pd.DataFrame()
    if not clinvar_path.is_file():
        return pd.DataFrame()

    gene = infer_gene(dataset)
    target_by_key: dict[str, list[str]] = {}
    for hgvs_nt in hgvs_values:
        key = hgvs_nt_key(hgvs_nt)
        if key:
            target_by_key.setdefault(key, []).append(hgvs_nt)

    records_by_hgvs: dict[str, list[dict[str, str]]] = {value: [] for value in hgvs_values}
    for row in clinvar_rows_for_gene(clinvar_path, gene):
        if row.get("Assembly") != "GRCh38":
            continue
        key = hgvs_nt_key(row.get("Name", ""))
        if not key:
            continue
        for hgvs_nt in target_by_key.get(key, []):
            records_by_hgvs[hgvs_nt].append(row)

    rows = [aggregate_clinvar_records(hgvs_nt, records_by_hgvs[hgvs_nt]) for hgvs_nt in hgvs_values]
    out = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def clinvar_rows_for_gene(clinvar_path: Path, gene: str) -> list[dict[str, str]]:
    if not gene:
        return []
    if clinvar_path.suffix == ".gz":
        with gzip.open(clinvar_path, "rt", encoding="utf-8", newline="") as handle:
            header = handle.readline().rstrip("\n")
        command = ["zgrep", "-F", f"\t{gene}\t", str(clinvar_path)]
    else:
        with clinvar_path.open(encoding="utf-8", newline="") as handle:
            header = handle.readline().rstrip("\n")
        command = ["grep", "-F", f"\t{gene}\t", str(clinvar_path)]

    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode not in {0, 1}:
        raise RuntimeError(f"Failed to filter ClinVar rows for {gene}: {proc.stderr.strip()}")
    lines = [line for line in proc.stdout.splitlines() if line]
    if not lines:
        return []
    return list(csv.DictReader([header, *lines], delimiter="\t"))


def build_hgvs_context(
    dataset: str,
    output_root: Path,
    processed_variants: pd.DataFrame,
    cache_dir: Path,
    clinvar_path: Path,
    force_cache: bool,
) -> HgvsContext:
    paths = pipeline.dataset_paths(output_root, dataset)
    variant_map = build_hgvs_variant_map(dataset, output_root, processed_variants)
    variant_map_path = paths.table_dir / f"{dataset}_hgvs_variant_map.csv"
    paths.table_dir.mkdir(parents=True, exist_ok=True)
    variant_map.to_csv(variant_map_path, index=False)

    if variant_map.empty:
        return HgvsContext(dataset, False, "no recoverable hgvs_nt mapping", variant_map, pd.DataFrame(), variant_map_path, None)

    annotation_path = cache_dir / f"{dataset}_hgvs_clinvar_annotations.csv"
    annotations = build_hgvs_annotation_cache(dataset, variant_map, clinvar_path, annotation_path, force_cache)
    if annotations.empty:
        reason = f"no HGVS ClinVar annotation cache; missing or unusable {clinvar_path}"
        return HgvsContext(dataset, False, reason, variant_map, annotations, variant_map_path, annotation_path)

    binary_count = int(annotations["annotation"].isin(pipeline.PATHOGENICITY_LABELS).sum())
    if binary_count == 0:
        return HgvsContext(dataset, False, "HGVS annotations found no binary benign/pathogenic labels", variant_map, annotations, variant_map_path, annotation_path)

    return HgvsContext(dataset, True, "using exact nucleotide HGVS ClinVar annotations", variant_map, annotations, variant_map_path, annotation_path)


def source_fitness_path(row: pd.Series) -> Path | None:
    source_path = row.get("source_fitness_path", "")
    if not is_blank(source_path):
        return Path(clean_text(source_path))
    fitness_path = row.get("fitness_path", "")
    if is_blank(fitness_path):
        return None
    return Path(clean_text(fitness_path))


def write_column_value(df: pd.DataFrame, idx: int, column: str, value: object) -> None:
    if column not in df.columns:
        df[column] = pd.NA
    df.loc[idx, column] = value


def refresh_protein_fitness(
    df: pd.DataFrame,
    idx: int,
    source_path: Path,
    fitness_df: pd.DataFrame,
    annotation_map: dict[str, str],
) -> dict:
    fitness_df = fitness_df.copy()
    fitness_df["SequenceIndex"] = fitness_df["SequenceIndex"].astype(str)
    fitness_df["annotation"] = fitness_df["SequenceIndex"].map(annotation_map)
    fitness_df.to_csv(source_path, index=False)
    write_column_value(df, idx, "fitness_path", str(source_path))
    write_column_value(df, idx, "source_fitness_path", "")
    write_column_value(df, idx, "annotation_scheme", "protein")
    write_column_value(df, idx, "annotation_source_path", "")
    return pipeline.classification_metrics_for_fitness(fitness_df)


def expand_fitness_to_hgvs(
    dataset: str,
    source_path: Path,
    fitness_df: pd.DataFrame,
    context: HgvsContext,
    output_dir: Path,
) -> tuple[pd.DataFrame, Path] | None:
    protein_df = fitness_df.copy()
    protein_df["SequenceIndex"] = protein_df["SequenceIndex"].astype(str)
    protein_df = protein_df.rename(columns={"SequenceIndex": "protein_sequence_index"})
    protein_df = protein_df.drop(columns=[c for c in ANNOTATION_COLUMNS_TO_REPLACE if c in protein_df.columns], errors="ignore")

    map_columns = [
        "protein_sequence_index",
        "mutant",
        "original_mutant",
        "functional_score",
        "mavedb_accession",
        "hgvs_nt",
        "hgvs_pro",
        "post_mapped_hgvs_p",
        "clingen_allele_id",
    ]
    map_df = context.variant_map[[c for c in map_columns if c in context.variant_map.columns]].copy()
    map_df["protein_sequence_index"] = map_df["protein_sequence_index"].astype(str)
    map_df = map_df[map_df["hgvs_nt"].map(clean_text).astype(bool)].copy()
    if map_df.empty:
        return None

    expanded = protein_df.merge(map_df, on="protein_sequence_index", how="inner")
    if expanded.empty:
        return None

    annotations = context.annotations.copy()
    annotations["hgvs_nt"] = annotations["hgvs_nt"].map(clean_text)
    annotation_cols = [
        "hgvs_nt",
        "annotation",
        "stars",
        "has_clinvar",
        "clinvar_significance",
        "clinvar_significance_normalized",
        "clinvar_review_status",
        "clinvar_variation_ids",
        "clinvar_allele_ids",
        "clinvar_conditions",
        "clinvar_record_count",
        "annotation_source",
    ]
    annotations = annotations[[c for c in annotation_cols if c in annotations.columns]].copy()
    expanded = expanded.merge(annotations, on="hgvs_nt", how="left")
    expanded.insert(0, "SequenceIndex", expanded["hgvs_nt"].astype(str))
    expanded["dataset"] = dataset
    expanded["annotation_scheme"] = "hgvs"
    expanded["stars"] = pd.to_numeric(expanded.get("stars", 0), errors="coerce").fillna(0).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pipeline.safe_name(source_path.stem)}__hgvs_fitness.csv"
    expanded.to_csv(out_path, index=False)
    return expanded, out_path


def refresh_hgvs_fitness(
    df: pd.DataFrame,
    idx: int,
    dataset: str,
    source_path: Path,
    fitness_df: pd.DataFrame,
    context: HgvsContext,
) -> dict | None:
    paths = pipeline.dataset_paths(Path(df.attrs["output_root"]), dataset)
    result = expand_fitness_to_hgvs(dataset, source_path, fitness_df, context, paths.table_dir / "hgvs_auc_fitness")
    if result is None:
        return None
    expanded_df, hgvs_path = result
    write_column_value(df, idx, "fitness_path", str(hgvs_path))
    write_column_value(df, idx, "source_fitness_path", str(source_path))
    write_column_value(df, idx, "annotation_scheme", "hgvs")
    write_column_value(df, idx, "annotation_source_path", str(context.annotation_path or ""))
    write_column_value(df, idx, "hgvs_variant_map_path", str(context.variant_map_path))
    write_column_value(df, idx, "n_hgvs_mapped_fitness_rows", int(len(expanded_df)))
    return pipeline.classification_metrics_for_fitness(expanded_df)


def refresh_metrics_table(
    metrics_path: Path,
    dataset: str,
    output_root: Path,
    annotation_map: dict[str, str],
    context: HgvsContext,
    annotation_scheme: str,
) -> tuple[pd.DataFrame, RefreshStats]:
    stats = RefreshStats(table_path=metrics_path)
    df = pd.read_csv(metrics_path)
    df.attrs["output_root"] = str(output_root)
    stats.rows = int(len(df))
    use_hgvs = annotation_scheme in {"auto", "hgvs"} and context.active

    for idx, row in df.iterrows():
        source_path = source_fitness_path(row)
        if source_path is None:
            continue
        stats.rows_with_fitness_path += 1
        if not source_path.is_file():
            stats.missing_fitness_files += 1
            continue

        fitness_df = pd.read_csv(source_path)
        if not {"SequenceIndex", "fitness"}.issubset(fitness_df.columns):
            stats.unusable_fitness_files += 1
            continue

        if use_hgvs:
            auc_metrics = refresh_hgvs_fitness(df, idx, dataset, source_path, fitness_df, context)
            if auc_metrics is None:
                if annotation_scheme == "hgvs":
                    stats.unusable_fitness_files += 1
                    continue
                auc_metrics = refresh_protein_fitness(df, idx, source_path, fitness_df, annotation_map)
                stats.protein_rows += 1
            else:
                stats.hgvs_rows += 1
        else:
            auc_metrics = refresh_protein_fitness(df, idx, source_path, fitness_df, annotation_map)
            stats.protein_rows += 1

        for key, value in auc_metrics.items():
            write_column_value(df, idx, key, value)
        stats.refreshed_rows += 1

    df.to_csv(metrics_path, index=False)
    return df, stats


def refresh_dataset(
    dataset: str,
    output_root: Path,
    processed_variants: pd.DataFrame,
    cache_dir: Path,
    clinvar_path: Path,
    force_cache: bool,
    annotation_scheme: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[RefreshStats], dict[str, object]]:
    paths = pipeline.dataset_paths(output_root, dataset)
    annotation_map = pipeline.annotation_map_for_dataset(output_root, dataset)
    context = build_hgvs_context(dataset, output_root, processed_variants, cache_dir, clinvar_path, force_cache)
    stats: list[RefreshStats] = []

    effective_scheme = "hgvs" if annotation_scheme in {"auto", "hgvs"} and context.active else "protein"
    audit = {
        "dataset": dataset,
        "requested_annotation_scheme": annotation_scheme,
        "effective_annotation_scheme": effective_scheme,
        "hgvs_context": context.reason,
        "hgvs_variant_rows": int(len(context.variant_map)),
        "hgvs_annotation_rows": int(len(context.annotations)),
        "hgvs_binary_annotation_rows": int(context.annotations.get("annotation", pd.Series(dtype=str)).isin(pipeline.PATHOGENICITY_LABELS).sum())
        if not context.annotations.empty
        else 0,
        "hgvs_variant_map_path": str(context.variant_map_path),
        "hgvs_annotation_path": str(context.annotation_path or ""),
    }

    method_df = None
    if paths.benchmark_metrics_path.is_file():
        method_df, method_stats = refresh_metrics_table(
            paths.benchmark_metrics_path,
            dataset,
            output_root,
            annotation_map,
            context,
            annotation_scheme,
        )
        stats.append(method_stats)
    else:
        print(f"Missing method metrics for {dataset}: {paths.benchmark_metrics_path}")

    ensemble_df = None
    if paths.ensemble_metrics_path.is_file():
        ensemble_df, ensemble_stats = refresh_metrics_table(
            paths.ensemble_metrics_path,
            dataset,
            output_root,
            annotation_map,
            context,
            annotation_scheme,
        )
        stats.append(ensemble_stats)
    else:
        print(f"Missing ensemble metrics for {dataset}: {paths.ensemble_metrics_path}")

    extra_metrics = sorted(paths.table_dir.glob("sae_ensemble_gamma*/*_metrics.csv"))
    primary_paths = {paths.benchmark_metrics_path.resolve(), paths.ensemble_metrics_path.resolve()}
    for metrics_path in extra_metrics:
        if metrics_path.resolve() in primary_paths:
            continue
        _, extra_stats = refresh_metrics_table(
            metrics_path,
            dataset,
            output_root,
            annotation_map,
            context,
            annotation_scheme,
        )
        stats.append(extra_stats)

    return method_df, ensemble_df, stats, audit


def write_global_table(frames: list[pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(path, index=False)
    else:
        pd.DataFrame().to_csv(path, index=False)


def run_summarize(datasets: list[str], output_root: Path, sahu_brca2_final: str) -> None:
    args = argparse.Namespace(
        output_root=str(output_root),
        datasets=datasets,
        count_datasets_only=True,
        sahu_brca2_final=sahu_brca2_final,
    )
    pipeline.summarize(args)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    datasets = [str(dataset) for dataset in args.datasets]
    cache_dir = args.hgvs_cache_dir or (output_root / "tables" / "hgvs_clinvar_annotations")

    if not args.skip_prepare:
        print("Repreparing updated final CSVs...")
        run_prepare(datasets, input_dir, output_root)

    processed_variants = load_processed_variants(args.processed_variants)
    method_frames: list[pd.DataFrame] = []
    ensemble_frames: list[pd.DataFrame] = []
    all_stats: list[RefreshStats] = []
    audit_rows: list[dict[str, object]] = []

    for dataset in datasets:
        print(f"Refreshing annotation-derived metrics for {dataset}...")
        method_df, ensemble_df, stats, audit = refresh_dataset(
            dataset,
            output_root,
            processed_variants,
            cache_dir,
            Path(args.clinvar_variant_summary),
            args.force_rebuild_hgvs_cache,
            args.annotation_scheme,
        )
        if method_df is not None:
            method_frames.append(method_df)
        if ensemble_df is not None:
            ensemble_frames.append(ensemble_df)
        all_stats.extend(stats)
        audit_rows.append(audit)

    tables_dir = output_root / "tables"
    write_global_table(method_frames, tables_dir / "clinprotgym_method_metrics.csv")
    write_global_table(ensemble_frames, tables_dir / "clinprotgym_sae_ensemble_metrics.csv")
    pd.DataFrame(audit_rows).to_csv(tables_dir / "clinprotgym_annotation_scheme_audit.csv", index=False)

    print("\nAnnotation scheme audit:")
    for row in audit_rows:
        print(
            f"{row['dataset']}: {row['effective_annotation_scheme']} "
            f"({row['hgvs_context']}; hgvs_binary={row['hgvs_binary_annotation_rows']})"
        )

    print("\nRefresh summary:")
    for stat in all_stats:
        print(
            f"{relative_display(stat.table_path)}: rows={stat.rows}, refreshed={stat.refreshed_rows}, "
            f"hgvs={stat.hgvs_rows}, protein={stat.protein_rows}, "
            f"missing_fitness={stat.missing_fitness_files}, unusable_fitness={stat.unusable_fitness_files}"
        )

    if not args.no_summarize:
        print("\nRedrawing summary plots...")
        run_summarize(datasets, output_root, args.sahu_brca2_final)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
