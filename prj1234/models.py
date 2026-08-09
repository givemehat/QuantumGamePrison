"""
models.py
=========
Typed data-access models for experiments and game results.

These dataclasses sit above the raw SQLite layer and provide
a clean, type-safe interface to the data tier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class GameRun:
    """Single game round stored in the ``game_results`` table.

    Attributes
    ----------
    id : int | None
        Database primary key (None before insertion).
    experiment_id : int
        FK → experiments.id.
    player_a_prob, player_b_prob : float
        Cooperation probabilities used in this round.
    outcome : str
        Two-character label: ``"CC"``, ``"CD"``, ``"DC"``, or ``"DD"``.
    player_a_score, player_b_score : float
        Payoff scores for this round.
    timestamp : datetime
        Insertion time (UTC).
    """

    experiment_id: int
    player_a_prob: float
    player_b_prob: float
    outcome: str
    player_a_score: float
    player_b_score: float
    id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.outcome not in {"CC", "CD", "DC", "DD"}:
            raise ValueError(f"Invalid outcome '{self.outcome}'.")
        for name, val in (
            ("player_a_prob", self.player_a_prob),
            ("player_b_prob", self.player_b_prob),
        ):
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"{name}={val!r} must be in [0, 1].")


@dataclass
class ExperimentConfig:
    """Metadata stored in the ``experiments`` table.

    Attributes
    ----------
    name : str
        Short human-readable experiment name.
    description : str
        Longer description (stored verbatim).
    strategy_type : str
        E.g. ``"quantum_ry"``, ``"classical"``.
    payoff_mode : str
        E.g. ``"standard_pd"``.
    config_json : dict
        Arbitrary extra configuration serialised as JSON.
    id : int | None
        Database primary key (None before insertion).
    created_at : datetime
    """

    name: str
    description: str
    strategy_type: str
    payoff_mode: str
    config_json: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def config_json_str(self) -> str:
        """Serialise config_json to a JSON string."""
        return json.dumps(self.config_json, sort_keys=True)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ExperimentConfig":
        """Construct from a database row dict."""
        cfg_raw = row.get("config_json") or "{}"
        cfg = json.loads(cfg_raw) if isinstance(cfg_raw, str) else cfg_raw
        return cls(
            id=row["id"],
            name=row["name"],
            description=row.get("description", ""),
            strategy_type=row["strategy_type"],
            payoff_mode=row["payoff_mode"],
            config_json=cfg,
            created_at=(
                datetime.fromisoformat(row["created_at"])
                if isinstance(row.get("created_at"), str)
                else row.get("created_at", datetime.utcnow())
            ),
        )

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for printing or report tables."""
        return {
            "id": self.id,
            "name": self.name,
            "strategy_type": self.strategy_type,
            "payoff_mode": self.payoff_mode,
            "created_at": self.created_at.isoformat(timespec="seconds"),
        }


@dataclass
class ExperimentStats:
    """Aggregated statistics for one experiment, returned by the analysis layer."""

    experiment_id: int
    total_runs: int
    outcome_counts: Dict[str, int]
    avg_score_a: float
    avg_score_b: float
    median_score_a: float
    median_score_b: float
    std_score_a: float
    std_score_b: float
    avg_prob_a: float
    avg_prob_b: float

    @property
    def cooperation_rate_a(self) -> float:
        """Empirical cooperation rate for Player A."""
        cc = self.outcome_counts.get("CC", 0)
        cd = self.outcome_counts.get("CD", 0)
        total = self.total_runs
        return (cc + cd) / total if total else 0.0

    @property
    def cooperation_rate_b(self) -> float:
        """Empirical cooperation rate for Player B."""
        cc = self.outcome_counts.get("CC", 0)
        dc = self.outcome_counts.get("DC", 0)
        total = self.total_runs
        return (cc + dc) / total if total else 0.0
