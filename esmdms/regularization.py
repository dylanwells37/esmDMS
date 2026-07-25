"""popDMS regularization-strength (gamma) selection by the correlation elbow.

The two selector functions ``find_last_below_threshold`` and
``get_best_regularization`` are copied, unchanged in logic, from the canonical
Barton-lab popDMS implementation (``popDMS.py`` in the ``popDMS`` repository,
functions at lines 1109-1162). Keeping them byte-for-byte faithful means the
gamma chosen for regular popDMS here matches the reference tool. The thin
``popdms_gamma_grid`` and ``select_elbow_gamma`` helpers only adapt the esmDMS
``InferenceProblem`` / ``Dataset`` objects to the shape those functions expect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .inference import InferenceProblem
    from .schema import Dataset

# The ESM-DMS notebooks and popDMS notebooks both use CORR_CUTOFF_PCT = 0.10.
DEFAULT_CORR_CUTOFF_PCT = 0.10
# popDMS floors the maximum read depth at 1e5 when building the gamma grid.
DEFAULT_MAX_READS = 1e5
GAMMA_GRID_POINTS = 20


def find_last_below_threshold(nums_in, th=0.1):
    idx_out = 0
    for i, value in enumerate(nums_in):
        if value >= th:
            break
        idx_out = i
    return idx_out


def get_best_regularization(corrs, gamma_values, corr_cutoff_pct=DEFAULT_CORR_CUTOFF_PCT):
    '''
    Compute best regularization strength from correlation data.
    '''

    max_corrs = max(corrs)
    corr_thresh = (max_corrs**2 - corrs[0]**2) * corr_cutoff_pct

    gamma_opt = 0.1
    if abs(max_corrs**2 - corrs[0]**2) < 0.01:
        gamma_opt = gamma_values[0]
    else:
        gamma_set = False

        delta_cor_set, i_set = [], []
        for i in range(np.argmax(corrs), 1, -1):
            delta_cor_num = corrs[i]**2 - corrs[i-1]**2
            delta_cor_den = (np.log10(gamma_values[i]) - np.log10(gamma_values[i-1]))
            delta_cor_ratio = abs(delta_cor_num / delta_cor_den)
            delta_cor_set.append(abs(max_corrs) - abs(corrs[i]))
            i_set.append(i)
            if delta_cor_ratio >= corr_thresh:
                gamma_opt = gamma_values[i]
                gamma_set = True
                break
        # if the loop completes without finding a gamma, set gamma_opt to the value that gives 1% of the drop in R
        if not gamma_set:
            i_before_below10percent = find_last_below_threshold(delta_cor_set)
            gamma_opt = gamma_values[i_set[i_before_below10percent]]

    return gamma_opt


def max_reads(dataset: "Dataset", *, floor: float = DEFAULT_MAX_READS) -> float:
    """Return the largest per-timepoint total read mass, floored like popDMS.

    popDMS derives its gamma grid lower bound from ``1 / max_reads`` with
    ``max_reads`` floored at 1e5. esmDMS trajectories store counts (or, for the
    TP53 frequency dataset, normalized frequencies) in the ``Frequency`` column,
    so the analogous depth is the maximum total mass summed over variants at any
    one replicate/generation. The floor dominates for the frequency dataset and
    keeps the lower bound at 1e-5, matching the esmDMS default grid.
    """
    totals = dataset.trajectory.groupby(["Replicate", "Generation"])["Frequency"].sum()
    observed = float(totals.max()) if len(totals) else 0.0
    return max(observed, float(floor))


def popdms_gamma_grid(
    dataset: "Dataset", *, num: int = GAMMA_GRID_POINTS
) -> np.ndarray:
    """popDMS gamma grid: ``logspace(log10(1/max_reads), 4, num)`` (ascending)."""
    lower = np.log10(1.0 / max_reads(dataset))
    return np.logspace(lower, 4, num=num)


def select_elbow_gamma(
    problem: "InferenceProblem",
    gammas: np.ndarray,
    *,
    corr_cutoff_pct: float = DEFAULT_CORR_CUTOFF_PCT,
) -> tuple[float, np.ndarray]:
    """Choose gamma at the popDMS correlation elbow over an ascending grid.

    Returns the selected gamma and the cross-replicate consistency curve (one
    value per grid point) so callers can log or plot it. With fewer than two
    replicates the correlation is undefined, so popDMS fixes gamma = 1; we do the
    same.
    """
    gammas = np.asarray(gammas, dtype=float)
    if len(problem.replicate_labels) < 2:
        return 1.0, np.full(len(gammas), np.nan)
    corrs = np.array(
        [problem.solve(gamma=float(g)).cross_replicate_consistency for g in gammas],
        dtype=float,
    )
    gamma_opt = float(get_best_regularization(corrs, gammas, corr_cutoff_pct))
    return gamma_opt, corrs
