"""
reports.py
==========
Markdown report generation for Quantum Prisoner's Dilemma experiments.

Reports are designed to be human-readable and suitable as a starting
point for academic writing or technical documentation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from analysis.summary import (
    OUTCOMES,
    compute_experiment_stats,
    compute_outcome_frequencies,
)
from storage.database import DEFAULT_DB_PATH, fetch_runs, get_experiment
from storage.models import ExperimentConfig, ExperimentStats

logger = logging.getLogger(__name__)


def _outcome_table(counts: Dict[str, int], total: int) -> str:
    """Render outcome counts as a Markdown table."""
    lines = [
        "| Outcome | Count | Frequency |",
        "|---------|------:|----------:|",
    ]
    for o in OUTCOMES:
        c = counts.get(o, 0)
        pct = 100.0 * c / total if total else 0.0
        lines.append(f"| {o}      | {c:,} | {pct:.2f}% |")
    return "\n".join(lines)


def _score_table(stats: ExperimentStats) -> str:
    """Render player score statistics as a Markdown table."""
    return (
        "| Metric      | Player A | Player B |\n"
        "|-------------|----------:|----------:|\n"
        f"| Mean        | {stats.avg_score_a:.4f} | {stats.avg_score_b:.4f} |\n"
        f"| Median      | {stats.median_score_a:.4f} | {stats.median_score_b:.4f} |\n"
        f"| Std Dev     | {stats.std_score_a:.4f} | {stats.std_score_b:.4f} |\n"
        f"| Coop. Rate  | {stats.cooperation_rate_a:.3f} | {stats.cooperation_rate_b:.3f} |"
    )


def _interpret(stats: ExperimentStats, experiment: ExperimentConfig) -> str:
    """Generate a brief plain-language interpretation paragraph."""
    dominant = max(stats.outcome_counts, key=lambda k: stats.outcome_counts[k])
    dom_pct = 100.0 * stats.outcome_counts[dominant] / stats.total_runs

    lines = [
        f"The experiment used the **{experiment.strategy_type}** strategy with the "
        f"**{experiment.payoff_mode}** payoff model over **{stats.total_runs:,}** rounds.",
        "",
        f"The most frequent outcome was **{dominant}** ({dom_pct:.1f}% of rounds).",
    ]
    if stats.cooperation_rate_a > 0.6 and stats.cooperation_rate_b > 0.6:
        lines.append(
            "Both players displayed predominantly cooperative behaviour, "
            "consistent with mutual benefit strategies."
        )
    elif stats.cooperation_rate_a < 0.4 and stats.cooperation_rate_b < 0.4:
        lines.append(
            "Both players displayed predominantly defecting behaviour, "
            "indicating convergence toward the Nash equilibrium of mutual defection."
        )
    else:
        lines.append(
            "Players exhibited mixed strategies, suggesting asymmetric cooperation dynamics."
        )

    lines += [
        "",
        f"Player A achieved an average score of **{stats.avg_score_a:.3f}** "
        f"(std {stats.std_score_a:.3f}), while Player B averaged "
        f"**{stats.avg_score_b:.3f}** (std {stats.std_score_b:.3f}).",
    ]
    return "\n".join(lines)


def generate_report(
    experiment_id: int,
    output_path: str,
    plot_paths: Optional[Dict[str, str]] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Generate a Markdown report for a completed experiment.

    Parameters
    ----------
    experiment_id : int
    output_path : str
        Destination file path (e.g. ``"reports/experiment_1.md"``).
    plot_paths : dict, optional
        Mapping of plot type → file path (from ``plots.generate_all_plots``).
    db_path : str

    Returns
    -------
    str
        Path to the written report file.
    """
    experiment = get_experiment(experiment_id, db_path)
    stats = compute_experiment_stats(experiment_id, db_path)
    plot_paths = plot_paths or {}

    # ── config details from stored JSON ──
    cfg = experiment.config_json
    strategy_type = experiment.strategy_type
    encoding = cfg.get("encoding", "ry")
    entanglement = cfg.get("entanglement", "none")
    iterations = (
        cfg.get("iterations") or cfg.get("iterations_per_cell") or stats.total_runs
    )
    pa_prob = cfg.get("player_a_prob", stats.avg_prob_a)
    pb_prob = cfg.get("player_b_prob", stats.avg_prob_b)
    seed = cfg.get("seed", "not set")
    is_grid = "pa_values" in cfg

    # ── assemble Markdown ──
    sections: List[str] = []

    sections.append(f"# Experiment Report: {experiment.name}")
    sections.append(
        f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*\n"
    )

    sections.append("## 1. Experiment Metadata\n")
    sections.append(
        f"| Field            | Value |\n"
        f"|------------------|-------|\n"
        f"| Experiment ID    | {experiment_id} |\n"
        f"| Name             | {experiment.name} |\n"
        f"| Description      | {experiment.description or '—'} |\n"
        f"| Strategy Type    | `{strategy_type}` |\n"
        f"| Payoff Model     | `{experiment.payoff_mode}` |\n"
        f"| Encoding         | `{encoding}` |\n"
        f"| Entanglement     | `{entanglement}` |\n"
        f"| Experiment Kind  | {'Grid sweep' if is_grid else 'Single fixed probability'} |\n"
        f"| Total Rounds     | {stats.total_runs:,} |\n"
        f"| Random Seed      | {seed} |\n"
        f"| Created (UTC)    | {experiment.created_at.strftime('%Y-%m-%d %H:%M')} |\n"
    )

    if not is_grid:
        sections.append(
            f"| Player A prob    | {pa_prob:.3f} |\n"
            f"| Player B prob    | {pb_prob:.3f} |\n"
        )

    sections.append("## 2. Outcome Frequencies\n")
    sections.append(_outcome_table(stats.outcome_counts, stats.total_runs))
    sections.append("")

    sections.append("## 3. Score Statistics\n")
    sections.append(_score_table(stats))
    sections.append("")

    sections.append("## 4. Interpretation\n")
    sections.append(_interpret(stats, experiment))
    sections.append("")

    if plot_paths:
        sections.append("## 5. Generated Plots\n")
        for plot_type, path in plot_paths.items():
            rel = Path(path).name
            sections.append(f"### {plot_type.replace('_', ' ').title()}\n")
            sections.append(f"![{plot_type}]({rel})\n")

    sections.append("---")
    sections.append(
        "*This report was generated automatically by the "
        "Quantum Prisoner's Dilemma research framework.*"
    )

    content = "\n".join(sections)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    logger.info("Report written to '%s'.", output_path)
    return output_path
