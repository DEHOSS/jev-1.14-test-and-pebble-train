"""Query scoring: disagreement percentiles and skill-based indistinguishability."""

from __future__ import annotations

import numpy as np


def disagreement_threshold(history: np.ndarray, percentile: float) -> float:
    """Adaptive floor: p-th percentile of recent candidate-pair ensemble disagreements."""
    if percentile <= 0 or len(history) == 0:
        return 0.0
    if percentile > 100:
        raise ValueError("disagreement percentile must be in [0, 100]")
    return float(np.percentile(history, percentile))


def skill_total_variation(post_a: np.ndarray, post_b: np.ndarray) -> float:
    """Total variation between two skill posteriors in [0, 1]."""
    a = np.asarray(post_a, dtype=np.float64)
    b = np.asarray(post_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a / max(a.sum(), 1e-8)
    b = b / max(b.sum(), 1e-8)
    return 0.5 * float(np.abs(a - b).sum())


def pair_indistinguishable(post_a: np.ndarray, post_b: np.ndarray, disagreement: float,
                           min_skill_tv: float, min_disagreement: float) -> bool:
    """True when skill detection and/or disagreement say the pair is not worth querying.

    A pair is indistinguishable if skill posteriors are too close *or* (once a
    disagreement floor is active) the ensemble is not uncertain enough.
    """
    if skill_total_variation(post_a, post_b) < min_skill_tv:
        return True
    if min_disagreement > 0 and disagreement < min_disagreement:
        return True
    return False


def minmax_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-8:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def sepoa_scores(skill_gaps: np.ndarray, disagreements: np.ndarray) -> np.ndarray:
    """S-EPOA I = (1 + |ΔR_skill|) (1 + Var) after min-max normalizing each term."""
    skill_term = 1.0 + minmax_unit(skill_gaps)
    dis_term = 1.0 + minmax_unit(disagreements)
    return skill_term * dis_term
