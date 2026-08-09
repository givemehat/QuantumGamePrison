"""
plots.py
========
Matplotlib visualisations for Quantum Prisoner's Dilemma experiments.

All functions save plots to disk and optionally display them.

Plot types
----------
* outcome_bar        – outcome frequency bar chart
* score_line         – Player A & B score over runs (with rolling average)
* prob_vs_score      – probability vs average score scatter/line
* score_histogram    – distribution histogram for both players
* heatmap_payoff     – 2-D average payoff heatmap (grid experiments)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import numpy as np

from analysis.summary import (
    OUTCOMES,
    compute_grid_heatmap,
    compute_outcome_frequencies,
    compute_prob_vs_score,
    compute_rolling_scores,
    compute_score_stats,
)
from storage.database import DEFAULT_DB_PATH, fetch_runs

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_and_show(fig: plt.Figure, filepath: str, show: bool) -> str:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    logger.info("Saved plot: '%s'.", filepath)
    if show:
        plt.show()
    plt.close(fig)
    return filepath


# ── outcome bar chart ─────────────────────────────────────────────────────────


def plot_outcome_bar(
    experiment_id: int,
    save_dir: str = "plots",
    show: bool = False,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Bar chart of outcome frequencies for an experiment.

    Parameters
    ----------
    experiment_id : int
    save_dir : str
        Directory to save the PNG.
    show : bool
        If True, call ``plt.show()`` after saving.
    db_path : str

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    rows = fetch_runs(experiment_id, db_path)
    counts = compute_outcome_frequencies(rows)

    labels = list(counts.keys())
    values = list(counts.values())
    total = sum(values) or 1

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=plt.cm.tab10.colors[:4])
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Count")
    ax.set_title(f"Outcome Distribution — Experiment #{experiment_id}")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{100*val/total:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    filepath = str(Path(save_dir) / f"experiment_{experiment_id}_outcome_bar.png")
    return _save_and_show(fig, filepath, show)


# ── score over runs line plot ─────────────────────────────────────────────────


def plot_score_line(
    experiment_id: int,
    rolling_window: int = 50,
    save_dir: str = "plots",
    show: bool = False,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Rolling-average score over the sequence of game rounds.

    Parameters
    ----------
    rolling_window : int
        Window size for the rolling average.
    """
    rows = fetch_runs(experiment_id, db_path)
    if not rows:
        raise ValueError(f"No data for experiment_id={experiment_id}.")

    ra, rb = compute_rolling_scores(rows, window=rolling_window)
    x = np.arange(rolling_window - 1, len(rows))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, ra, label="Player A", linewidth=1.2, alpha=0.85)
    ax.plot(x, rb, label="Player B", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Score (rolling avg, w={rolling_window})")
    ax.set_title(f"Score Over Rounds — Experiment #{experiment_id}")
    ax.legend()
    fig.tight_layout()
    filepath = str(Path(save_dir) / f"experiment_{experiment_id}_score_line.png")
    return _save_and_show(fig, filepath, show)


# ── probability vs score scatter ──────────────────────────────────────────────


def plot_prob_vs_score(
    experiment_id: int,
    save_dir: str = "plots",
    show: bool = False,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Average score binned by cooperation probability for both players."""
    rows = fetch_runs(experiment_id, db_path)
    pa_pts = compute_prob_vs_score(rows, player="a")
    pb_pts = compute_prob_vs_score(rows, player="b")

    fig, ax = plt.subplots(figsize=(8, 4))
    if pa_pts:
        xs_a, ys_a = zip(*pa_pts)
        ax.plot(xs_a, ys_a, marker="o", markersize=4, label="Player A", linewidth=1.5)
    if pb_pts:
        xs_b, ys_b = zip(*pb_pts)
        ax.plot(xs_b, ys_b, marker="s", markersize=4, label="Player B", linewidth=1.5)
    ax.set_xlabel("Cooperation Probability")
    ax.set_ylabel("Average Score")
    ax.set_title(f"Probability vs Average Score — Experiment #{experiment_id}")
    ax.legend()
    fig.tight_layout()
    filepath = str(Path(save_dir) / f"experiment_{experiment_id}_prob_vs_score.png")
    return _save_and_show(fig, filepath, show)


# ── score histogram ────────────────────────────────────────────────────────────


def plot_score_histogram(
    experiment_id: int,
    bins: int = 20,
    save_dir: str = "plots",
    show: bool = False,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Overlaid histograms of score distributions for both players."""
    rows = fetch_runs(experiment_id, db_path)
    scores_a = [float(r["player_a_score"]) for r in rows]
    scores_b = [float(r["player_b_score"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores_a, bins=bins, alpha=0.6, label="Player A", density=True)
    ax.hist(scores_b, bins=bins, alpha=0.6, label="Player B", density=True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — Experiment #{experiment_id}")
    ax.legend()
    fig.tight_layout()
    filepath = str(Path(save_dir) / f"experiment_{experiment_id}_score_hist.png")
    return _save_and_show(fig, filepath, show)


# ── payoff heatmap (grid experiments) ─────────────────────────────────────────


def plot_heatmap(
    experiment_id: int,
    player: str = "a",
    save_dir: str = "plots",
    show: bool = False,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """2-D heatmap of average payoff over the (pA, pB) grid.

    Parameters
    ----------
    player : str
        ``"a"`` or ``"b"``.
    """
    rows = fetch_runs(experiment_id, db_path)
    pa_vals, pb_vals, z = compute_grid_heatmap(rows, player=player)

    if not pa_vals or not pb_vals:
        raise ValueError(f"Experiment #{experiment_id} has no grid data for a heatmap.")

    z_arr = np.array(z)  # shape (n_pa, n_pb)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(pb_vals, pa_vals, z_arr, shading="auto")
    fig.colorbar(im, ax=ax, label=f"Avg Score Player {player.upper()}")
    ax.set_xlabel("Player B Cooperation Probability")
    ax.set_ylabel("Player A Cooperation Probability")
    ax.set_title(
        f"Payoff Heatmap (Player {player.upper()}) — Experiment #{experiment_id}"
    )
    fig.tight_layout()
    filepath = str(Path(save_dir) / f"experiment_{experiment_id}_heatmap_{player}.png")
    return _save_and_show(fig, filepath, show)


# ── quantum vs classical comparison ──────────────────────────────────────────


def plot_quantum_vs_classical(
    quantum_experiment_id: int,
    classical_experiment_id: int,
    save_dir: str = "plots",
    show: bool = False,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Side-by-side outcome-frequency bar chart comparing two experiments.

    Useful for comparing quantum vs classical strategies with the same pA/pB.
    """
    rows_q = fetch_runs(quantum_experiment_id, db_path)
    rows_c = fetch_runs(classical_experiment_id, db_path)
    counts_q = compute_outcome_frequencies(rows_q)
    counts_c = compute_outcome_frequencies(rows_c)

    total_q = sum(counts_q.values()) or 1
    total_c = sum(counts_c.values()) or 1

    labels = OUTCOMES
    freqs_q = [counts_q[o] / total_q for o in labels]
    freqs_c = [counts_c[o] / total_c for o in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, freqs_q, width, label=f"Quantum (#{quantum_experiment_id})")
    ax.bar(
        x + width / 2, freqs_c, width, label=f"Classical (#{classical_experiment_id})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative Frequency")
    ax.set_title("Quantum vs Classical Outcome Comparison")
    ax.legend()
    fig.tight_layout()
    filepath = str(
        Path(save_dir)
        / f"quantum_{quantum_experiment_id}_vs_classical_{classical_experiment_id}.png"
    )
    return _save_and_show(fig, filepath, show)


# ── generate all standard plots ───────────────────────────────────────────────


def generate_all_plots(
    experiment_id: int,
    save_dir: str = "plots",
    show: bool = False,
    db_path: str = DEFAULT_DB_PATH,
    is_grid: bool = False,
) -> Dict[str, str]:
    """Generate all standard plots for an experiment.

    Parameters
    ----------
    is_grid : bool
        If True, also generate heatmaps (for grid experiments).

    Returns
    -------
    Dict[str, str]
        Mapping of plot type → file path.
    """
    generated: Dict[str, str] = {}

    def _try(key: str, fn, **kwargs):
        try:
            generated[key] = fn(
                experiment_id, save_dir=save_dir, show=show, db_path=db_path, **kwargs
            )
        except Exception as exc:
            logger.warning("Could not generate '%s' plot: %s", key, exc)

    _try("outcome_bar", plot_outcome_bar)
    _try("score_line", plot_score_line)
    _try("prob_vs_score", plot_prob_vs_score)
    _try("score_hist", plot_score_histogram)

    if is_grid:
        _try("heatmap_a", plot_heatmap, player="a")
        _try("heatmap_b", plot_heatmap, player="b")

    return generated
