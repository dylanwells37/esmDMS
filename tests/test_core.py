"""Schema, inference, metric, and workflow tests. These must not require torch."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from esmdms.inference import (
    SubstitutionBasis,
    assay_oriented_scores,
    build_problem,
    infer,
    prior_sweep,
    prior_vector,
    substitution_basis,
)
from esmdms.import_data import process_imported_csv
from esmdms.metrics import (
    enrichment_fitness,
    evaluate_fitness,
    functional_score_fitness,
)
from esmdms.schema import Dataset, FeatureArtifact
from esmdms.workflow import _summarize, run_analysis

from .conftest import synthetic_dataset, synthetic_prior


def test_dataset_and_artifact_round_trip(tmp_path):
    source = synthetic_dataset()
    dataset = Dataset(
        source.name,
        source.reference_sequence,
        source.variants,
        source.trajectory,
        truncation=(1, 3),
    )
    dataset.save(tmp_path / "dataset")
    loaded_dataset = Dataset.load(tmp_path / "dataset")
    assert loaded_dataset.sequence_ids == dataset.sequence_ids
    assert loaded_dataset.reference_sequence == "AAA"
    assert loaded_dataset.truncation == (1, 3)
    manifest = json.loads((tmp_path / "dataset" / "dataset.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["truncation"] == {"start": 1, "end": 3}

    artifact = synthetic_prior(dataset)
    artifact.save(tmp_path / "prior.npz")
    loaded_artifact = FeatureArtifact.load(tmp_path / "prior.npz")
    assert loaded_artifact.kind == "llr_prior"
    np.testing.assert_array_equal(loaded_artifact.values, artifact.values)


def test_imported_csv_processor_builds_canonical_dataset(tmp_path):
    frame = pd.DataFrame(
        {
            "mutant": ["A1C", "A2D", "A3*"],
            "mutated_sequence": ["CAA", "ADA", "AA"],
            "has_clinvar": ["true", "false", "true"],
            "clinvar_significance": ["Pathogenic", "", "Pathogenic"],
            "clinvar_significance_normalized": ["pathogenic", "", "pathogenic"],
            "clinvar_review_status": [
                "reviewed by expert panel",
                "",
                "criteria provided, single submitter",
            ],
            "clinvar_conditions": ["condition", "", "condition"],
            "clinvar_variation_ids": ["1", "", "2"],
            "count__count_day11_rep1": [30, 20, 1],
            "count__count_day11_rep2": [31, 21, 1],
            "count__count_day5_rep1": [20, 15, 1],
            "count__count_day5_rep2": [21, 16, 1],
            "count__count_library": [10, 10, 1],
            "count__count_negative_control": [0, 0, 0],
            "count__count_rna_rep1": [1, 1, 0],
            "count__count_rna_rep2": [1, 1, 0],
            "functional_score": [-1.0, 0.5, -2.0],
        }
    )
    source = tmp_path / "MV_BRCA1_Findlay_2018.csv"
    frame.to_csv(source, index=False)
    dataset = process_imported_csv(source, tmp_path / "datasets")

    assert dataset.reference_sequence == "AAA"
    assert dataset.metadata["dropped_stop_rows"] == 1
    assert dataset.metadata["source_file"] == source.name
    assert len(dataset.metadata["source_file_sha256"]) == 64
    assert dataset.variants.iloc[0]["SequenceIndex"] == "__wildtype__"
    assert len(dataset.variants) == 3
    assert len(dataset.trajectory) == 12
    assert set(dataset.trajectory["Replicate"]) == {"rep1", "rep2"}
    assert set(dataset.trajectory["Generation"]) == {0.0, 5.0, 11.0}
    assert not dataset.trajectory["SequenceIndex"].eq("__wildtype__").any()
    assert sorted(
        path.name for path in (tmp_path / "datasets" / source.stem).iterdir()
    ) == [
        "dataset.json",
        "trajectory.csv",
        "variants.csv",
    ]


def test_partial_functional_scores_exclude_missing_rows():
    dataset = synthetic_dataset()
    variants = dataset.variants.copy()
    variants.loc[[0, 2], "functional_score"] = np.nan
    partial = Dataset(
        dataset.name,
        dataset.reference_sequence,
        variants,
        dataset.trajectory,
    )
    fitness = functional_score_fitness(partial)
    assert fitness.sequence_ids == ("v1", "v2", "v3")
    assert np.isfinite(fitness.values).all()


def test_substitution_prior_inference_and_metrics():
    dataset = synthetic_dataset()
    basis = substitution_basis(dataset)
    assert basis.feature_names == ("1:C", "2:D", "3:E")
    result = infer(
        dataset, basis, gamma=0.1, prior=synthetic_prior(dataset), prior_scale=0.5
    )
    assert result.replicate_coefficients.shape == (2, 3)
    assert result.joint_coefficients.shape == (3,)
    assert np.isfinite(result.joint_coefficients).all()
    fitness = result.fitness()
    assert fitness.values.shape == (5, 1)
    metrics = evaluate_fitness(dataset, fitness)
    assert 0 <= metrics["auc"] <= 1
    assert metrics["n_pathogenic"] == 3


def test_substitution_basis_is_sparse():
    dataset = synthetic_dataset()
    basis = substitution_basis(dataset)

    # One column index per variant row; the wild-type row is outside the basis.
    assert isinstance(basis, SubstitutionBasis)
    assert basis.columns.tolist() == [-1, 0, 0, 1, 2]
    coefficients = np.asarray([0.5, -0.25, 2.0])
    np.testing.assert_allclose(
        basis.project(coefficients), [0.0, 0.5, 0.5, -0.25, 2.0]
    )


def test_inference_problem_is_reusable_across_gamma_and_alpha():
    dataset = synthetic_dataset()
    basis = substitution_basis(dataset)
    problem = build_problem(dataset, basis)
    prior = prior_vector(dataset, basis, synthetic_prior(dataset))

    for gamma in (0.01, 1.0):
        for alpha in (0.0, 1.0):
            reused = problem.solve(gamma=gamma, prior_values=alpha * prior)
            fresh = infer(
                dataset,
                basis,
                gamma=gamma,
                prior=synthetic_prior(dataset),
                prior_scale=alpha,
            )
            np.testing.assert_allclose(
                reused.joint_coefficients, fresh.joint_coefficients, atol=1e-12
            )

    with pytest.raises(ValueError, match="expected"):
        problem.solve(gamma=1.0, prior_values=np.zeros(2))
    with pytest.raises(ValueError, match="gamma must be positive"):
        problem.solve(gamma=0.0)


def test_raw_llr_orientation_follows_dataset_selection_direction():
    source = synthetic_dataset()
    dataset = Dataset(
        "synthetic-high",
        source.reference_sequence,
        source.variants,
        source.trajectory,
        pathogenic_high_selection=True,
    )
    raw_prior = synthetic_prior(dataset, orientation="raw_llr")
    result = infer(dataset, substitution_basis(dataset), gamma=1.0, prior=raw_prior)
    np.testing.assert_allclose(result.prior, [2.0, 1.0, 0.25])

    # The same helper drives inference and the workflow's raw-prior baseline.
    np.testing.assert_allclose(
        assay_oriented_scores(dataset, raw_prior).to_numpy(),
        [2.0, 2.0, 1.0, 0.25],
    )
    # A prior already expressed in selection units is left alone.
    selection_prior = synthetic_prior(dataset, orientation="selection")
    np.testing.assert_allclose(
        assay_oriented_scores(dataset, selection_prior).to_numpy(),
        [-2.0, -2.0, -1.0, -0.25],
    )


def test_prior_sweep_and_enrichment():
    dataset = synthetic_dataset()
    sweep = prior_sweep(
        dataset,
        synthetic_prior(dataset),
        alphas=[0.0, 1.0],
        gammas=[0.01, 1.0],
        evaluate=lambda fitness: evaluate_fitness(dataset, fitness),
    )
    assert len(sweep) == 4
    assert set(("alpha", "gamma", "auc", "cross_replicate_consistency")).issubset(sweep)
    enrichment = enrichment_fitness(dataset)
    assert enrichment.values.shape == (5, 1)
    assert np.isfinite(enrichment.values).all()


def test_summary_keeps_one_intact_row_per_dataset_and_prior():
    # The winning row carries a NaN consistency. GroupBy.first() would backfill
    # it from the runner-up; the summary must report the winner's own values.
    sweeps = pd.DataFrame(
        {
            "dataset": ["d"] * 3,
            "prior": ["p"] * 3,
            "alpha": [1.0, 0.0, 0.5],
            "gamma": [10.0, 1.0, 2.0],
            "cross_replicate_consistency": [np.nan, 0.11, 0.22],
            "auc_stars_0": [0.9, 0.8, 0.7],
        }
    )
    summary = _summarize(sweeps, 0)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["alpha"] == 1.0
    assert row["gamma"] == 10.0
    assert np.isnan(row["cross_replicate_consistency"])
    assert row["auc_stars_0"] == 0.9
    # The zero-prior control and the gain over it are reported alongside.
    assert row["auc_alpha0"] == 0.8
    assert row["auc_gain_over_alpha0"] == pytest.approx(0.1)
    assert bool(row["prior_used"]) is True


def test_summary_flags_a_winner_that_used_no_prior():
    sweeps = pd.DataFrame(
        {
            "dataset": ["d"] * 2,
            "prior": ["p"] * 2,
            "alpha": [0.0, 1.0],
            "gamma": [1.0, 1.0],
            "cross_replicate_consistency": [0.5, 0.4],
            "auc_stars_0": [0.9, 0.6],
        }
    )
    row = _summarize(sweeps, 0).iloc[0]
    assert row["alpha"] == 0.0
    assert bool(row["prior_used"]) is False
    assert row["auc_gain_over_alpha0"] == pytest.approx(0.0)


def test_configured_workflow(tmp_path):
    dataset = synthetic_dataset()
    dataset.save(tmp_path / "dataset")
    synthetic_prior(dataset, orientation="raw_llr").save(tmp_path / "prior.npz")
    config = {
        "_config_dir": str(tmp_path),
        "output_dir": "results",
        "gammas": [0.01, 1.0],
        "alpha_mode": "fixed",
        "alphas": [0.0, 1.0],
        "review_star_cutoffs": [0, 2],
        "datasets": [{"path": "dataset", "priors": {"Test LLR": "prior.npz"}}],
    }
    results = run_analysis(config)
    assert len(results["sweeps"]) == 4
    assert len(results["summary"]) == 1
    assert (tmp_path / "results" / "summary.csv").is_file()

    baselines = results["baselines"]
    # Every gamma-selected baseline records the gamma and consistency it used.
    # The gamma is chosen by the popDMS elbow over its own grid, not the config
    # sweep gammas, so only require a valid positive strength here.
    regular = baselines.set_index("method").loc["Regular popDMS"]
    assert regular["gamma"] > 0
    assert np.isfinite(regular["cross_replicate_consistency"])
    # Methods with no gamma leave it explicitly missing rather than absent.
    assert np.isnan(baselines.set_index("method").loc["Enrichment ratio", "gamma"])

    # Spearman is cutoff-independent, so it appears once and is never suffixed.
    assert "spearman_rho" in baselines
    assert not [column for column in baselines if column.startswith("spearman_rho_")]
    assert {"auc_stars_0", "auc_stars_2"}.issubset(baselines.columns)


def test_workflow_raw_prior_baseline_uses_assay_oriented_scores(tmp_path):
    source = synthetic_dataset()
    dataset = Dataset(
        "synthetic-high",
        source.reference_sequence,
        source.variants,
        source.trajectory,
        pathogenic_high_selection=True,
    )
    dataset.save(tmp_path / "dataset")
    synthetic_prior(dataset, orientation="raw_llr").save(tmp_path / "prior.npz")
    results = run_analysis(
        {
            "_config_dir": str(tmp_path),
            "output_dir": "results",
            "gammas": [1.0],
            "alphas": [0.0],
            "review_star_cutoffs": [0],
            "datasets": [{"path": "dataset", "priors": {"LLR": "prior.npz"}}],
        }
    )
    raw = results["baselines"].set_index("method").loc["Raw LLR"]
    # Higher oriented LLR must mean higher selection, which on this dataset
    # means more pathogenic, matching every other row's convention.
    oriented = assay_oriented_scores(dataset, synthetic_prior(dataset, orientation="raw_llr"))
    expected = evaluate_fitness(
        dataset,
        FeatureArtifact(
            tuple(oriented.index.astype(str)),
            oriented.to_numpy()[:, None],
            ("fitness",),
            "fitness",
            dataset.name,
        ),
    )
    assert raw["auc_stars_0"] == pytest.approx(expected["auc"])
    assert raw["spearman_rho"] == pytest.approx(expected["spearman_rho"])
