"""Canonical in-memory and on-disk data structures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


ARTIFACT_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 2
SEQUENCE_ID = "SequenceIndex"
ArtifactKind = Literal["embedding", "sae", "llr_prior", "basis", "fitness"]
ARTIFACT_KINDS = {"embedding", "sae", "llr_prior", "basis", "fitness"}

VARIANT_COLUMNS = (
    SEQUENCE_ID,
    "protein_sequence",
    "position",
    "wt_aa",
    "mutant_aa",
    "is_synonymous",
)
TRAJECTORY_COLUMNS = (SEQUENCE_ID, "Replicate", "Generation", "Frequency")


def _string_ids(values: object, *, label: str) -> tuple[str, ...]:
    ids = tuple(str(value) for value in values)
    if not ids or any(not value for value in ids):
        raise ValueError(f"{label} must contain non-empty identifiers.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} must be unique.")
    return ids


def _boolean_values(series: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    invalid = sorted(set(normalized).difference(mapping))
    if invalid:
        raise ValueError(
            f"{label} contains invalid boolean values: {', '.join(invalid)}"
        )
    return normalized.map(mapping).astype(bool)


@dataclass(frozen=True)
class FeatureArtifact:
    """A finite feature matrix with explicit row and column identities.

    Raw embeddings, SAE activations, LLR priors, substitution bases, and inferred
    fitness values all use this representation. Rows are always keyed by
    ``SequenceIndex``; scalar scores use a one-column matrix.
    """

    sequence_ids: tuple[str, ...]
    values: np.ndarray
    feature_names: tuple[str, ...]
    kind: ArtifactKind
    dataset: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sequence_ids = _string_ids(self.sequence_ids, label="sequence_ids")
        feature_names = _string_ids(self.feature_names, label="feature_names")
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError(
                f"Artifact values must be 2D; received shape {values.shape}."
            )
        if values.shape != (len(sequence_ids), len(feature_names)):
            raise ValueError(
                "Artifact shape does not match its identifiers: "
                f"{values.shape} != ({len(sequence_ids)}, {len(feature_names)})."
            )
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("Artifact values must be numeric.")
        if not np.isfinite(values).all():
            raise ValueError("Artifact values must all be finite.")
        if not self.dataset:
            raise ValueError("dataset must be non-empty.")
        if self.kind not in ARTIFACT_KINDS:
            raise ValueError(f"Unsupported artifact kind {self.kind!r}.")
        try:
            json.dumps(self.provenance)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Artifact provenance must be JSON serializable."
            ) from error
        object.__setattr__(self, "sequence_ids", sequence_ids)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        kind: ArtifactKind,
        dataset: str,
        provenance: dict[str, Any] | None = None,
    ) -> "FeatureArtifact":
        if SEQUENCE_ID not in frame:
            raise ValueError(f"Feature table must contain {SEQUENCE_ID!r}.")
        feature_names = tuple(column for column in frame if column != SEQUENCE_ID)
        if not feature_names:
            raise ValueError("Feature table must contain at least one feature column.")
        return cls(
            tuple(frame[SEQUENCE_ID].astype(str)),
            frame.loc[:, feature_names].to_numpy(),
            feature_names,
            kind,
            dataset,
            dict(provenance or {}),
        )

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.values, columns=self.feature_names)
        frame.insert(0, SEQUENCE_ID, self.sequence_ids)
        return frame

    def align(self, sequence_ids: object) -> np.ndarray:
        requested = tuple(str(value) for value in sequence_ids)
        row_by_id = {
            sequence_id: index for index, sequence_id in enumerate(self.sequence_ids)
        }
        missing = sorted(set(requested).difference(row_by_id))
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Artifact is missing {len(missing)} sequence ids: {preview}"
            )
        return self.values[[row_by_id[sequence_id] for sequence_id in requested]]

    def scalar_series(self, name: str | None = None) -> pd.Series:
        if self.values.shape[1] != 1:
            raise ValueError("Expected a one-column artifact.")
        return pd.Series(
            self.values[:, 0],
            index=pd.Index(self.sequence_ids, name=SEQUENCE_ID),
            name=name or self.feature_names[0],
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Feature artifacts must use the .npz extension.")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "kind": self.kind,
            "dataset": self.dataset,
            "provenance": self.provenance,
        }
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                sequence_ids=np.asarray(self.sequence_ids, dtype=str),
                values=self.values,
                feature_names=np.asarray(self.feature_names, dtype=str),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureArtifact":
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported artifact schema {metadata.get('schema_version')!r}; "
                    f"expected {ARTIFACT_SCHEMA_VERSION}."
                )
            return cls(
                tuple(payload["sequence_ids"].astype(str)),
                payload["values"],
                tuple(payload["feature_names"].astype(str)),
                metadata["kind"],
                metadata["dataset"],
                metadata.get("provenance", {}),
            )

    @classmethod
    def merge(
        cls,
        artifacts: list["FeatureArtifact"],
        *,
        expected_sequence_ids: object,
    ) -> "FeatureArtifact":
        """Merge row-sharded artifacts and restore canonical dataset order."""
        if not artifacts:
            raise ValueError("At least one artifact is required for merging.")
        expected = _string_ids(
            expected_sequence_ids, label="expected_sequence_ids"
        )
        first = artifacts[0]

        def base_provenance(artifact: "FeatureArtifact") -> dict[str, Any]:
            return {
                key: value
                for key, value in artifact.provenance.items()
                if key not in {"shard_index", "num_shards", "merged_shards"}
            }

        base = base_provenance(first)
        for artifact in artifacts[1:]:
            if artifact.kind != first.kind:
                raise ValueError("Cannot merge artifacts with different kinds.")
            if artifact.dataset != first.dataset:
                raise ValueError("Cannot merge artifacts from different datasets.")
            if artifact.feature_names != first.feature_names:
                raise ValueError("Cannot merge artifacts with different features.")
            if base_provenance(artifact) != base:
                raise ValueError("Cannot merge artifacts with different provenance.")

        shard_fields = [
            (
                artifact.provenance.get("shard_index"),
                artifact.provenance.get("num_shards"),
            )
            for artifact in artifacts
        ]
        if any(index is not None or count is not None for index, count in shard_fields):
            if any(index is None or count is None for index, count in shard_fields):
                raise ValueError("Shard provenance is incomplete.")
            counts = {int(count) for _, count in shard_fields}
            if len(counts) != 1:
                raise ValueError("Shard artifacts disagree on num_shards.")
            count = counts.pop()
            indices = {int(index) for index, _ in shard_fields}
            if len(artifacts) != count or indices != set(range(count)):
                raise ValueError(
                    f"Expected shard indices 0 through {count - 1}; received "
                    f"{sorted(indices)}."
                )

        row_by_id: dict[str, np.ndarray] = {}
        for artifact in artifacts:
            for sequence_id, values in zip(artifact.sequence_ids, artifact.values):
                if sequence_id in row_by_id:
                    raise ValueError(
                        f"Duplicate SequenceIndex across shards: {sequence_id}"
                    )
                row_by_id[sequence_id] = values
        missing = sorted(set(expected).difference(row_by_id))
        extra = sorted(set(row_by_id).difference(expected))
        if missing or extra:
            raise ValueError(
                "Merged artifact rows do not match the expected dataset rows: "
                f"{len(missing)} missing, {len(extra)} extra."
            )

        provenance = dict(base)
        provenance["merged_shards"] = len(artifacts)
        return cls(
            expected,
            np.vstack([row_by_id[sequence_id] for sequence_id in expected]),
            first.feature_names,
            first.kind,
            first.dataset,
            provenance,
        )


@dataclass(frozen=True)
class Dataset:
    """A DMS dataset using the sole supported repository data layout."""

    name: str
    reference_sequence: str
    variants: pd.DataFrame
    trajectory: pd.DataFrame
    pathogenic_high_selection: bool = False
    truncation: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Dataset name must be non-empty.")
        if not isinstance(self.pathogenic_high_selection, bool):
            raise TypeError("pathogenic_high_selection must be a boolean.")
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as error:
            raise ValueError("Dataset metadata must be JSON serializable.") from error
        reference = "".join(str(self.reference_sequence).split()).upper()
        if not reference or not reference.isalpha():
            raise ValueError(
                "reference_sequence must be a non-empty amino-acid sequence."
            )
        truncation = self.truncation
        if truncation is not None:
            if (
                not isinstance(truncation, (tuple, list))
                or len(truncation) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in truncation
                )
            ):
                raise TypeError(
                    "truncation must be a pair of integer 1-based endpoints or None."
                )
            start, end = truncation
            if start < 1 or end < start or end > len(reference):
                raise ValueError(
                    "truncation must be a valid 1-based inclusive reference interval."
                )
            truncation = (start, end)

        variants = self.variants.copy()
        missing = sorted(set(VARIANT_COLUMNS).difference(variants.columns))
        if missing:
            raise ValueError(f"Variant table is missing columns: {', '.join(missing)}")
        if variants.loc[:, VARIANT_COLUMNS].isna().any().any():
            raise ValueError("Required variant columns cannot contain missing values.")
        variants[SEQUENCE_ID] = variants[SEQUENCE_ID].astype(str)
        if variants[SEQUENCE_ID].duplicated().any():
            raise ValueError("Variant table must have one row per SequenceIndex.")
        variants["position"] = pd.to_numeric(
            variants["position"], errors="raise"
        ).astype(int)
        if not variants["position"].between(1, len(reference)).all():
            raise ValueError(
                "Variant positions must use one-based reference coordinates."
            )
        variants["is_synonymous"] = _boolean_values(
            variants["is_synonymous"], label="is_synonymous"
        )
        variants["protein_sequence"] = (
            variants["protein_sequence"]
            .astype(str)
            .str.replace(r"\s+", "", regex=True)
            .str.upper()
        )
        invalid_sequences = ~variants["protein_sequence"].str.fullmatch(
            rf"[A-Z]{{{len(reference)}}}"
        )
        if invalid_sequences.any():
            raise ValueError(
                "Every protein_sequence must be amino-acid text matching the reference length."
            )
        variants["wt_aa"] = variants["wt_aa"].astype(str).str.upper()
        variants["mutant_aa"] = variants["mutant_aa"].astype(str).str.upper()
        expected_wildtype = variants["position"].map(
            lambda position: reference[position - 1]
        )
        if not variants["wt_aa"].eq(expected_wildtype).all():
            raise ValueError("wt_aa must match reference_sequence at position.")
        observed_mutant = np.asarray(
            [
                sequence[position - 1]
                for sequence, position in zip(
                    variants["protein_sequence"], variants["position"]
                )
            ]
        )
        if not variants["mutant_aa"].eq(observed_mutant).all():
            raise ValueError("mutant_aa must match protein_sequence at position.")
        synonymous_from_change = variants["wt_aa"].eq(variants["mutant_aa"])
        if not variants["is_synonymous"].eq(synonymous_from_change).all():
            raise ValueError("is_synonymous must agree with wt_aa and mutant_aa.")
        substitutions = variants.loc[~variants["is_synonymous"], "position"]
        if truncation is not None and not substitutions.empty:
            start, end = truncation
            if substitutions.min() < start or substitutions.max() > end:
                raise ValueError(
                    "Dataset truncation must cover every assayed substitution."
                )
        for row in variants.itertuples(index=False):
            sequence = str(row.protein_sequence)
            differences = [
                index
                for index, (wildtype, observed) in enumerate(
                    zip(reference, sequence), start=1
                )
                if wildtype != observed
            ]
            expected = [] if bool(row.is_synonymous) else [int(row.position)]
            if differences != expected:
                raise ValueError(
                    "Each protein_sequence must be the reference or exactly the declared substitution."
                )

        trajectory = self.trajectory.copy()
        missing = sorted(set(TRAJECTORY_COLUMNS).difference(trajectory.columns))
        if missing:
            raise ValueError(
                f"Trajectory table is missing columns: {', '.join(missing)}"
            )
        if trajectory.loc[:, TRAJECTORY_COLUMNS].isna().any().any():
            raise ValueError(
                "Required trajectory columns cannot contain missing values."
            )
        trajectory[SEQUENCE_ID] = trajectory[SEQUENCE_ID].astype(str)
        unknown = sorted(set(trajectory[SEQUENCE_ID]).difference(variants[SEQUENCE_ID]))
        if unknown:
            raise ValueError(
                f"Trajectory contains {len(unknown)} unknown SequenceIndex values."
            )
        for column in ("Generation", "Frequency"):
            trajectory[column] = pd.to_numeric(trajectory[column], errors="raise")
        if not np.isfinite(trajectory[["Generation", "Frequency"]].to_numpy()).all():
            raise ValueError("Trajectory generations and frequencies must be finite.")
        if trajectory["Frequency"].lt(0).any():
            raise ValueError("Trajectory frequencies must be non-negative.")
        if trajectory.duplicated([SEQUENCE_ID, "Replicate", "Generation"]).any():
            raise ValueError(
                "Trajectory rows must be unique by SequenceIndex, Replicate, and Generation."
            )
        generations = trajectory.groupby("Replicate")["Generation"].nunique()
        if generations.empty or generations.lt(2).any():
            raise ValueError("Every replicate must contain at least two generations.")
        totals = trajectory.groupby(["Replicate", "Generation"])["Frequency"].sum()
        if totals.le(0).any():
            raise ValueError(
                "Every replicate generation must have positive total frequency."
            )

        object.__setattr__(self, "reference_sequence", reference)
        object.__setattr__(self, "truncation", truncation)
        object.__setattr__(self, "variants", variants.reset_index(drop=True))
        object.__setattr__(self, "trajectory", trajectory.reset_index(drop=True))

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(self.variants[SEQUENCE_ID])

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.variants.to_csv(directory / "variants.csv", index=False)
        self.trajectory.to_csv(directory / "trajectory.csv", index=False)
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "name": self.name,
            "reference_sequence": self.reference_sequence,
            "pathogenic_high_selection": self.pathogenic_high_selection,
            "truncation": (
                None
                if self.truncation is None
                else {"start": self.truncation[0], "end": self.truncation[1]}
            ),
            "metadata": self.metadata,
        }
        (directory / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "Dataset":
        directory = Path(directory)
        manifest = json.loads((directory / "dataset.json").read_text())
        if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported dataset schema {manifest.get('schema_version')!r}; "
                f"expected {DATASET_SCHEMA_VERSION}."
            )
        truncation = manifest["truncation"]
        if truncation is not None:
            truncation = (truncation["start"], truncation["end"])
        return cls(
            name=manifest["name"],
            reference_sequence=manifest["reference_sequence"],
            variants=pd.read_csv(directory / "variants.csv", dtype={SEQUENCE_ID: str}),
            trajectory=pd.read_csv(
                directory / "trajectory.csv", dtype={SEQUENCE_ID: str}
            ),
            pathogenic_high_selection=bool(
                manifest.get("pathogenic_high_selection", False)
            ),
            truncation=truncation,
            metadata=manifest.get("metadata", {}),
        )
