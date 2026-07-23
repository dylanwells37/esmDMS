from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from esmdms.inference import (
    build_problem,
    matched_alpha_grid,
    prior_vector,
    substitution_basis,
)
from esmdms.regularization import (
    find_last_below_threshold,
    get_best_regularization,
    popdms_gamma_grid,
    select_elbow_gamma,
)

from .conftest import synthetic_dataset, synthetic_prior

# Optional byte-for-byte cross-check against the canonical popDMS source, when it
# is present on this machine. The elbow logic must agree exactly.
POPDMS_SOURCE = Path("/ihome/jbarton/dhw28/popDMS/popDMS.py")


def _reference_selector():
    """Exec only the two elbow functions from the popDMS source.

    Importing the whole module pulls in plotting dependencies (``mplot``); the
    selector itself needs only numpy, so we lift the two function definitions out
    by name and exec them in an isolated namespace. This still checks against the
    exact source text of the canonical implementation.
    """
    tree = ast.parse(POPDMS_SOURCE.read_text())
    wanted = {"find_last_below_threshold", "get_best_regularization"}
    namespace: dict = {"np": np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), str(POPDMS_SOURCE), "exec"), namespace)
    return namespace["get_best_regularization"]


def _elbow_curve():
    """A correlation curve that rises to a plateau: elbow before the flat top."""
    gammas = np.logspace(-5, 4, 20)
    corrs = np.array(
        [0.10, 0.15, 0.22, 0.34, 0.50, 0.66, 0.78, 0.86, 0.90, 0.915]
        + [0.92] * 10
    )
    return corrs, gammas


def test_get_best_regularization_picks_elbow_not_argmax():
    corrs, gammas = _elbow_curve()
    gamma = get_best_regularization(corrs, gammas, 0.10)
    # The elbow sits below the flat maximum, so the chosen gamma is well under the
    # argmax gamma (which would be the last, most over-regularized grid point).
    assert gamma < gammas[int(np.argmax(corrs))]
    assert gamma in set(gammas)


def test_find_last_below_threshold():
    assert find_last_below_threshold([0.0, 0.05, 0.2, 0.3], th=0.1) == 1
    assert find_last_below_threshold([0.2, 0.3], th=0.1) == 0
    assert find_last_below_threshold([0.0, 0.0, 0.0], th=0.1) == 2


@pytest.mark.skipif(not POPDMS_SOURCE.is_file(), reason="popDMS source not present")
def test_matches_popdms_reference_exactly():
    reference = _reference_selector()
    corrs, gammas = _elbow_curve()
    for cutoff in (0.05, 0.10, 0.25, 0.5):
        assert get_best_regularization(corrs, gammas, cutoff) == reference(
            corrs, gammas, cutoff
        )


def test_popdms_gamma_grid_ascending_and_bounded():
    grid = popdms_gamma_grid(synthetic_dataset())
    assert grid.shape == (20,)
    assert np.all(np.diff(grid) > 0)
    assert grid[-1] == pytest.approx(1e4)
    # Floor of 1e5 reads keeps the lower bound at or below 1e-5.
    assert grid[0] <= 1e-5 + 1e-12


def test_select_elbow_gamma_returns_curve():
    dataset = synthetic_dataset()
    problem = build_problem(dataset, substitution_basis(dataset))
    grid = popdms_gamma_grid(dataset)
    gamma, corrs = select_elbow_gamma(problem, grid)
    assert gamma in set(grid)
    assert corrs.shape == grid.shape


def test_matched_alpha_grid_scales_prior_to_coefficient_spread():
    dataset = synthetic_dataset()
    basis = substitution_basis(dataset)
    problem = build_problem(dataset, basis)
    prior = synthetic_prior(dataset, orientation="raw_llr")

    alphas, records, meta = matched_alpha_grid(
        dataset,
        prior,
        reference_gamma=1.0,
        basis=basis,
        problem=problem,
    )

    s_star = meta["matched_scale"]
    p = prior_vector(dataset, basis, prior)
    c = problem.solve(gamma=1.0).joint_coefficients
    # By construction the matched scale equalizes the standard deviations.
    assert np.std(s_star * p) == pytest.approx(np.std(c))
    assert meta["sigma_coeff"] == pytest.approx(np.std(c))
    assert meta["sigma_prior"] == pytest.approx(np.std(p))

    # Control, seven multiples of s*, and the unscaled raw-LLR point.
    assert alphas[0] == 0.0
    assert 1.0 in alphas
    assert any(record["unscaled_raw_llr"] for record in records)
    scaled = [r for r in records if not r["unscaled_raw_llr"] and r["alpha"] > 0]
    assert {r["scale_multiple"] for r in scaled} == {0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0}
    # The 1x multiple lands exactly at the matched scale.
    one_x = next(r for r in scaled if r["scale_multiple"] == 1.0)
    assert one_x["alpha"] == pytest.approx(s_star)


def test_matched_alpha_grid_is_default_in_workflow(tmp_path):
    from esmdms.workflow import run_analysis

    dataset = synthetic_dataset()
    dataset.save(tmp_path / "dataset")
    synthetic_prior(dataset, orientation="raw_llr").save(tmp_path / "prior.npz")
    config = {
        "_config_dir": str(tmp_path),
        "output_dir": "results",
        "gammas": [0.01, 1.0],
        "review_star_cutoffs": [0],
        "datasets": [{"path": "dataset", "priors": {"Test LLR": "prior.npz"}}],
    }
    results = run_analysis(config)
    sweeps = results["sweeps"]
    # Default matched grid: 9 alphas (0, seven multiples, unscaled) x 2 gammas.
    assert set(sweeps["alpha"]).__len__() == 9
    assert len(sweeps) == 18
    for column in ("scale_multiple", "matched_scale", "sigma_coeff", "unscaled_raw_llr"):
        assert column in sweeps.columns
    assert sweeps["unscaled_raw_llr"].any()
