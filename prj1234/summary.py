"""
summary.py
==========
Statistical analysis of experiment results.

All public functions accept either:
  * a raw ``List[Dict]`` of game_results rows, or
  * an experiment ID + db_path (for lazy loading).
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from storage.database import DEFAULT_DB_PATH, fetch_runs, get_experiment
from storage.models import ExperimentConfig, ExperimentStats

logger = logging.getLogger(__name__)

OUTCOMES = ["CC", "CD", "DC", "DD"]


# ── core statistics ───────────────────────────────────────────────────────────

def compute_outcome_frequencies(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count occurrences of each outcome.

    Parameters
    ----------
    rows : List[Dict]
        Raw game_results rows (must contain ``'outcome'`` key).

    Returns
    -------
    Dict[str, int]
        ``{"CC": n, "CD": n, "DC": n, "DD": n}``
    """
    counts: Dict[str, int] = {o: 0 for o in OUTCOMES}
    for row in rows:
        outcome = row["outcome"]
        if outcome in counts:
            counts[outcome] += 1
        else:
            logger.warning("Unknown outcome '%s' ignored.", outcome)
    return counts


def compute_score_stats(
    rows: List[Dict[str, Any]],
    player: str = "a",
) -> Dict[str, float]:
    """Compute descriptive statistics for one player's scores.

    Parameters
    ----------
    rows : List[Dict]
    player : str
        ``"a"`` or ``"b"``.

    Returns
    -------
    Dict[str, float]
        Keys: ``mean``, ``median``, ``std``, ``min``, ``max``, ``variance``.
    """
    key = f"player_{player}_score"
    scores = [float(r[key]) for r in rows if key in r]
    if not scores:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "variance": 0.0}
    arr = np.array(scores, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "variance": float(np.var(arr)),
    }


def compute_experiment_stats(
    experiment_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> ExperimentStats:
    """Aggregate all statistics for an experiment.

    Parameters
    ----------
    experiment_id : int
    db_path : str

    Returns
    -------
    ExperimentStats
    """
    # Validate experiment exists
    get_experiment(experiment_id, db_path)  # raises KeyError if missing
    rows = fetch_runs(experiment_id, db_path)
    if not rows:
        raise ValueError(f"No game_results found for experiment_id={experiment_id}.")

    outcome_counts = compute_outcome_frequencies(rows)
    stats_a = compute_score_stats(rows, "a")
    stats_b = compute_score_stats(rows, "b")

    probs_a = [r["player_a_prob"] for r in rows]
    probs_b = [r["player_b_prob"] for r in rows]

    return ExperimentStats(
        experiment_id=experiment_id,
        total_runs=len(rows),
        outcome_counts=outcome_counts,
        avg_score_a=stats_a["mean"],
        avg_score_b=stats_b["mean"],
        median_score_a=stats_a["median"],
        median_score_b=stats_b["median"],
        std_score_a=stats_a["std"],
        std_score_b=stats_b["std"],
        avg_prob_a=float(np.mean(probs_a)),
        avg_prob_b=float(np.mean(probs_b)),
    )


def compute_prob_vs_score(
    rows: List[Dict[str, Any]],
    player: str = "a",
    n_bins: int = 20,
) -> List[Tuple[float, float]]:
    """Bin probability values and compute average score per bin.

    Parameters
    ----------
    rows : List[Dict]
    player : str
        ``"a"`` or ``"b"``.
    n_bins : int
        Number of equal-width probability bins.

    Returns
    -------
    List[Tuple[float, float]]
        ``[(prob_centre, avg_score), ...]``  sorted by probability.
    """
    prob_key = f"player_{player}_prob"
    score_key = f"player_{player}_score"
    if not rows:
        return []

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_scores: Dict[int, List[float]] = {i: [] for i in range(n_bins)}

    for row in rows:
        p = float(row[prob_key])
        s = float(row[score_key])
        idx = int(np.searchsorted(bins, p, side="right")) - 1
        idx = max(0, min(idx, n_bins - 1))
        bin_scores[idx].append(s)

    result = []
    for i in range(n_bins):
        centre = float((bins[i] + bins[i + 1]) / 2)
        if bin_scores[i]:
            result.append((centre, float(np.mean(bin_scores[i]))))
    return result


def compute_rolling_scores(
    rows: List[Dict[str, Any]],
    window: int = 50,
) -> Tuple[List[float], List[float]]:
    """Compute rolling average scores for Player A and B over all runs.

    Parameters
    ----------
    rows : List[Dict]
        Ordered game_results rows.
    window : int
        Rolling window size.

    Returns
    -------
    Tuple[List[float], List[float]]
        ``(rolling_a, rolling_b)``
    """
    scores_a = np.array([float(r["player_a_score"]) for r in rows])
    scores_b = np.array([float(r["player_b_score"]) for r in rows])
    kernel = np.ones(window) / window
    ra = np.convolve(scores_a, kernel, mode="valid").tolist()
    rb = np.convolve(scores_b, kernel, mode="valid").tolist()
    return ra, rb


def compute_grid_heatmap(
    rows: List[Dict[str, Any]],
    player: str = "a",
) -> Tuple[List[float], List[float], List[List[float]]]:
    """Compute a 2-D average-score grid over (pA, pB).

    Parameters
    ----------
    rows : List[Dict]
    player : str
        ``"a"`` or ``"b"``.

    Returns
    -------
    Tuple[List[float], List[float], List[List[float]]]
        ``(pa_values, pb_values, z_grid)``
        where ``z_grid[i][j]`` is the average score for Player *player*
        at ``pa_values[i]``, ``pb_values[j]``.
    """
    score_key = f"player_{player}_score"
    pa_set = sorted({round(float(r["player_a_prob"]), 10) for r in rows})
    pb_set = sorted({round(float(r["player_b_prob"]), 10) for r in rows})
    pa_idx = {v: i for i, v in enumerate(pa_set)}
    pb_idx = {v: j for j, v in enumerate(pb_set)}

    accum: Dict[Tuple[int, int], List[float]] = {}
    for row in rows:
        pa = round(float(row["player_a_prob"]), 10)
        pb = round(float(row["player_b_prob"]), 10)
        i = pa_idx.get(pa)
        j = pb_idx.get(pb)
        if i is not None and j is not None:
            accum.setdefault((i, j), []).append(float(row[score_key]))

    z: List[List[float]] = [
        [
            float(np.mean(accum.get((i, j), [0.0])))
            for j in range(len(pb_set))
        ]
        for i in range(len(pa_set))
    ]
    return pa_set, pb_set, z


def print_summary(stats: ExperimentStats, experiment: ExperimentConfig) -> None:
    """Print a formatted summary table to stdout."""
    sep = "-" * 50
    print(sep)
    print(f"Experiment #{stats.experiment_id}: {experiment.name}")
    print(f"  Strategy  : {experiment.strategy_type}")
    print(f"  Payoff    : {experiment.payoff_mode}")
    print(f"  Total runs: {stats.total_runs:,}")
    print()
    print("  Outcome frequencies:")
    total = stats.total_runs
    for o, c in stats.outcome_counts.items():
        pct = 100.0 * c / total if total else 0.0
        print(f"    {o}: {c:>7,}  ({pct:5.1f}%)")
    print()
    print(f"  Avg score  A: {stats.avg_score_a:.4f}  (±{stats.std_score_a:.4f})")
    print(f"  Avg score  B: {stats.avg_score_b:.4f}  (±{stats.std_score_b:.4f})")
    print(f"  Cooperation rate A: {stats.cooperation_rate_a:.3f}")
    print(f"  Cooperation rate B: {stats.cooperation_rate_b:.3f}")
    print(sep)
