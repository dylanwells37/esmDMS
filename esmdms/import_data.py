"""Convert the count-bearing imported CSVs into the canonical dataset schema."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import Dataset, SEQUENCE_ID


MUTANT_PATTERN = re.compile(r"^(?P<wt>[A-Z])(?P<position>[1-9]\d*)(?P<mutant>[A-Z*])$")
WILDTYPE_ID = "__wildtype__"
REQUIRED_IMPORTED_COLUMNS = {
    "mutant",
    "mutated_sequence",
    "has_clinvar",
    "clinvar_significance",
    "clinvar_significance_normalized",
    "clinvar_review_status",
    "clinvar_conditions",
    "clinvar_variation_ids",
    "functional_score",
}

PATHOGENIC_HIGH_SELECTION = {
    "MV_BRCA1_Findlay_2018": False,
    "MV_MSH2_Jia_2020": True,
    "MV_BRCA2_Huang_2025": False,
    "MV_VHL_Buckley_2024": False,
    "MV_TP53_Kotler_2018": True,
}

DATASET_TRUNCATIONS = {
    "MV_BRCA2_Huang_2025": (1371, 3418),
}


@dataclass(frozen=True)
class Measurement:
    replicate: str
    generation: float
    columns: tuple[str, ...]
    measurement_kind: str


def _sequence_id(sequence: str) -> str:
    digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()[:20]
    return f"seq_{digest}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_mutants(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_IMPORTED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Imported CSV is missing columns: {', '.join(missing)}")
    if frame[["mutant", "mutated_sequence"]].isna().any(axis=None):
        raise ValueError("mutant and mutated_sequence cannot contain missing values.")
    parsed = frame["mutant"].astype(str).str.strip().str.extract(MUTANT_PATTERN)
    if parsed.isna().any(axis=None):
        bad = (
            frame.loc[parsed.isna().any(axis=1), "mutant"].astype(str).head(5).tolist()
        )
        raise ValueError(f"Could not parse mutant labels: {bad}")
    parsed["position"] = parsed["position"].astype(int)
    return parsed


def _reference_sequence(frame: pd.DataFrame, parsed: pd.DataFrame) -> str:
    candidates = set()
    missense = parsed["mutant"].ne("*")
    for index in frame.index[missense]:
        sequence = str(frame.at[index, "mutated_sequence"]).strip().upper()
        position = int(parsed.at[index, "position"])
        wildtype = str(parsed.at[index, "wt"])
        mutant = str(parsed.at[index, "mutant"])
        if position > len(sequence) or sequence[position - 1] != mutant:
            raise ValueError(
                f"Mutation label does not match mutated_sequence at row {index}."
            )
        candidates.add(sequence[: position - 1] + wildtype + sequence[position:])
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one reconstructed reference sequence; found {len(candidates)}."
        )
    return candidates.pop()


def _review_stars(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    text = str(value).strip().lower()
    if "practice guideline" in text:
        return 4
    if "reviewed by expert panel" in text:
        return 3
    if "multiple submitters" in text and "no conflicts" in text:
        return 2
    if "criteria provided" in text and "conflicting" not in text:
        return 1
    return 0


def _boolean(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _measurements(dataset: str, columns: Iterable[str]) -> list[Measurement]:
    columns = set(columns)

    def require(*names: str) -> tuple[str, ...]:
        missing = sorted(set(names).difference(columns))
        if missing:
            raise ValueError(
                f"{dataset} is missing measurement columns: {', '.join(missing)}"
            )
        return tuple(names)

    if dataset == "MV_BRCA1_Findlay_2018":
        specs = []
        library = "count__count_library"
        for replicate in (1, 2):
            specs.extend(
                [
                    Measurement(f"rep{replicate}", 0.0, require(library), "count"),
                    Measurement(
                        f"rep{replicate}",
                        5.0,
                        require(f"count__count_day5_rep{replicate}"),
                        "count",
                    ),
                    Measurement(
                        f"rep{replicate}",
                        11.0,
                        require(f"count__count_day11_rep{replicate}"),
                        "count",
                    ),
                ]
            )
        return specs

    if dataset == "MV_BRCA2_Huang_2025":
        specs = []
        for replicate in range(1, 7):
            for generation, stage in ((0.0, "lib"), (5.0, "D5"), (14.0, "D14")):
                column = f"count__R{replicate}_{stage}"
                specs.append(
                    Measurement(f"R{replicate}", generation, require(column), "count")
                )
        return specs

    if dataset == "MV_MSH2_Jia_2020":
        prefix = "count__GSE162130_MSH2_HAP1_raw_count.tsv.gz:"
        specs = []
        for replicate in range(1, 4):
            initial = f"{prefix}R{replicate}D_P0"
            for condition in ("DB", "DB6"):
                label = f"R{replicate}_{condition}"
                endpoint = f"{prefix}R{replicate}{condition}_P2"
                specs.extend(
                    [
                        Measurement(label, 0.0, require(initial), "count"),
                        Measurement(label, 2.0, require(endpoint), "count"),
                    ]
                )
        return specs

    if dataset == "MV_VHL_Buckley_2024":
        specs = []
        for arm in ("tHDR", "rLD2_tHDR"):
            initial = f"count__{arm}_pre"
            for endpoint in ("post", "post2"):
                label = f"{arm}_{endpoint}"
                specs.extend(
                    [
                        Measurement(label, 0.0, require(initial), "count"),
                        Measurement(
                            label, 1.0, require(f"count__{arm}_{endpoint}"), "count"
                        ),
                    ]
                )
        return specs

    if dataset == "MV_TP53_Kotler_2018":
        pattern = re.compile(r"^frequency__(?P<replicate>.+):(?P<day>\d+)d$")
        found = []
        for column in sorted(columns):
            match = pattern.fullmatch(column)
            if match:
                found.append(
                    Measurement(
                        match.group("replicate"),
                        float(match.group("day")),
                        (column,),
                        "frequency",
                    )
                )
        replicate_counts = pd.Series([spec.replicate for spec in found]).value_counts()
        if len(replicate_counts) != 6 or not replicate_counts.eq(4).all():
            raise ValueError(
                "TP53 must contain six four-timepoint frequency replicates."
            )
        return sorted(found, key=lambda spec: (spec.replicate, spec.generation))

    supported = ", ".join(sorted(PATHOGENIC_HIGH_SELECTION))
    raise ValueError(
        f"Unsupported imported dataset {dataset!r}. Supported datasets: {supported}"
    )


def _variant_table(
    frame: pd.DataFrame, parsed: pd.DataFrame, reference: str
) -> tuple[pd.DataFrame, int]:
    stop_mask = parsed["mutant"].eq("*")
    kept = frame.loc[~stop_mask].copy()
    kept_parsed = parsed.loc[~stop_mask].copy()
    kept["mutated_sequence"] = (
        kept["mutated_sequence"].astype(str).str.strip().str.upper()
    )

    for index in kept.index:
        position = int(kept_parsed.at[index, "position"])
        expected = list(reference)
        expected[position - 1] = str(kept_parsed.at[index, "mutant"])
        if kept.at[index, "mutated_sequence"] != "".join(expected):
            raise ValueError(
                f"Row {index} is not exactly the declared single substitution."
            )

    sequence_ids = kept["mutated_sequence"].map(_sequence_id)
    if sequence_ids.duplicated().any():
        raise ValueError("Imported missense rows must have unique protein sequences.")
    variants = pd.DataFrame(
        {
            SEQUENCE_ID: sequence_ids,
            "protein_sequence": kept["mutated_sequence"],
            "position": kept_parsed["position"].astype(int),
            "wt_aa": kept_parsed["wt"],
            "mutant_aa": kept_parsed["mutant"],
            "is_synonymous": False,
            "mutant": kept["mutant"].astype(str),
            "functional_score": pd.to_numeric(
                kept.get("functional_score"), errors="coerce"
            ),
            "annotation": kept.get("clinvar_significance_normalized", "")
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower(),
            "review_stars": kept.get("clinvar_review_status", "").map(_review_stars),
            "has_clinvar": kept.get("has_clinvar", False).map(_boolean),
            "clinvar_significance": kept.get("clinvar_significance", "").fillna(""),
            "clinvar_review_status": kept.get("clinvar_review_status", "").fillna(""),
            "clinvar_conditions": kept.get("clinvar_conditions", "").fillna(""),
            "clinvar_variation_ids": kept.get("clinvar_variation_ids", "").fillna(""),
        }
    )
    wildtype = {column: "" for column in variants.columns}
    wildtype.update(
        {
            SEQUENCE_ID: WILDTYPE_ID,
            "protein_sequence": reference,
            "position": 1,
            "wt_aa": reference[0],
            "mutant_aa": reference[0],
            "is_synonymous": True,
            "mutant": f"{reference[0]}1{reference[0]}",
            "functional_score": np.nan,
            "annotation": "",
            "review_stars": 0,
            "has_clinvar": False,
        }
    )
    variants = pd.concat([pd.DataFrame([wildtype]), variants], ignore_index=True)
    return variants, int(stop_mask.sum())


def _trajectory_table(
    frame: pd.DataFrame, variants: pd.DataFrame, measurements: list[Measurement]
) -> pd.DataFrame:
    id_by_sequence = dict(
        zip(variants["protein_sequence"].astype(str), variants[SEQUENCE_ID].astype(str))
    )
    source = frame[frame["mutated_sequence"].astype(str).isin(id_by_sequence)].copy()
    source_ids = source["mutated_sequence"].astype(str).map(id_by_sequence)
    records = []
    for measurement in measurements:
        numeric = source.loc[:, measurement.columns].apply(
            pd.to_numeric, errors="coerce"
        )
        if (numeric < 0).any(axis=None):
            raise ValueError(f"Negative measurements found in {measurement.columns}.")
        values = numeric.fillna(0.0).sum(axis=1)
        records.append(
            pd.DataFrame(
                {
                    SEQUENCE_ID: source_ids,
                    "Replicate": measurement.replicate,
                    "Generation": measurement.generation,
                    "Frequency": values,
                    "MeasurementKind": measurement.measurement_kind,
                    "SourceColumns": "|".join(measurement.columns),
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def process_imported_csv(
    path: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
    collection_metadata: dict | None = None,
) -> Dataset:
    path = Path(path)
    dataset_name = path.stem
    if dataset_name not in PATHOGENIC_HIGH_SELECTION:
        raise ValueError(f"No processing specification exists for {dataset_name!r}.")
    destination = Path(output_root) / dataset_name
    if destination.exists() and not force:
        raise FileExistsError(
            f"Output dataset already exists: {destination}. Use force=True to replace it."
        )

    frame = pd.read_csv(path)
    parsed = _parse_mutants(frame)
    reference = _reference_sequence(frame, parsed)
    measurements = _measurements(dataset_name, frame.columns)
    variants, dropped_stops = _variant_table(frame, parsed, reference)
    trajectory = _trajectory_table(frame, variants, measurements)

    metadata = {
        "source_file": path.name,
        "source_file_sha256": _file_sha256(path),
        "source_rows": int(len(frame)),
        "kept_missense_rows": int(len(variants) - 1),
        "dropped_stop_rows": dropped_stops,
        "wildtype_row_added": True,
        "sequence_id": "sha256(protein_sequence)[:20]",
        "reference_reconstruction": "replace mutant residue with mutant label wildtype residue",
        "trajectory": [
            {
                "replicate": spec.replicate,
                "generation": spec.generation,
                "columns": list(spec.columns),
                "measurement_kind": spec.measurement_kind,
            }
            for spec in measurements
        ],
        "collection": dict(collection_metadata or {}),
    }
    dataset = Dataset(
        dataset_name,
        reference,
        variants,
        trajectory,
        pathogenic_high_selection=PATHOGENIC_HIGH_SELECTION[dataset_name],
        truncation=DATASET_TRUNCATIONS.get(dataset_name),
        metadata=metadata,
    )
    dataset.save(destination)
    return dataset


def process_imported_directory(
    input_dir: str | Path,
    output_root: str | Path,
    *,
    datasets: Iterable[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    selected = set(datasets or PATHOGENIC_HIGH_SELECTION)
    unknown = sorted(selected.difference(PATHOGENIC_HIGH_SELECTION))
    if unknown:
        raise ValueError(f"Unknown dataset selections: {', '.join(unknown)}")
    snapshot_path = input_dir / "CLINVAR_SNAPSHOT.json"
    collection_metadata = (
        json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else {}
    )
    rows = []
    for dataset_name in sorted(selected):
        path = input_dir / f"{dataset_name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing imported dataset: {path}")
        dataset = process_imported_csv(
            path,
            output_root,
            force=force,
            collection_metadata=collection_metadata,
        )
        rows.append(
            {
                "dataset": dataset.name,
                "variants": len(dataset.variants) - 1,
                "trajectory_rows": len(dataset.trajectory),
                "replicates": dataset.trajectory["Replicate"].nunique(),
                "reference_length": len(dataset.reference_sequence),
                "dropped_stops": dataset.metadata["dropped_stop_rows"],
            }
        )
    return pd.DataFrame(rows)
