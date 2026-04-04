"""
engine.py
=========
Main game engine for the Quantum Prisoner's Dilemma.

The engine ties together:
  - Quantum outcome sampling  (quantum_core)
  - Payoff computation        (payoff)
  - Per-round result records  (storage.models)

Usage example
-------------
>>> from game.engine import GameEngine
>>> from game.payoff import STANDARD_PD
>>> from quantum.quantum_core import EncodingScheme, EntanglementMode
>>> engine = GameEngine(payoff_model=STANDARD_PD)
>>> results = engine.play_rounds(prob_a=0.7, prob_b=0.5, n_rounds=500)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from game.payoff import PayoffModel, STANDARD_PD
from quantum.quantum_core import (
    EncodingScheme,
    EntanglementMode,
    sample_outcomes,
)

logger = logging.getLogger(__name__)


# ── round result dataclass ────────────────────────────────────────────────────

class RoundResult:
    """Container for the result of a single game round.

    Attributes
    ----------
    move_a, move_b : str
        ``"C"`` or ``"D"``.
    score_a, score_b : float
        Payoff scores for this round.
    outcome : str
        Two-character label, e.g. ``"CC"``, ``"CD"``.
    """

    __slots__ = ("move_a", "move_b", "score_a", "score_b", "outcome",
                 "player_a_prob", "player_b_prob")

    def __init__(
        self,
        move_a: str,
        move_b: str,
        score_a: float,
        score_b: float,
        player_a_prob: float,
        player_b_prob: float,
    ) -> None:
        self.move_a = move_a
        self.move_b = move_b
        self.score_a = score_a
        self.score_b = score_b
        self.outcome = move_a + move_b
        self.player_a_prob = player_a_prob
        self.player_b_prob = player_b_prob

    def __repr__(self) -> str:
        return (
            f"RoundResult(outcome={self.outcome!r}, "
            f"scores=({self.score_a:.1f}, {self.score_b:.1f}))"
        )


# ── game engine ───────────────────────────────────────────────────────────────

class GameEngine:
    """Orchestrates quantum sampling and payoff computation for many rounds.

    Parameters
    ----------
    payoff_model : PayoffModel
        The payoff matrix to use (default: standard Prisoner's Dilemma).
    encoding : EncodingScheme
        Quantum angle encoding strategy.
    entanglement : EntanglementMode
        Post-encoding entanglement mode.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        payoff_model: PayoffModel = STANDARD_PD,
        encoding: EncodingScheme = EncodingScheme.RY,
        entanglement: EntanglementMode = EntanglementMode.NONE,
        seed: Optional[int] = None,
    ) -> None:
        self.payoff_model = payoff_model
        self.encoding = encoding
        self.entanglement = entanglement
        self._rng = np.random.default_rng(seed)

    # ── public API ───────────────────────────────────────────────────────────

    def play_rounds(
        self,
        prob_a: float,
        prob_b: float,
        n_rounds: int,
    ) -> List[RoundResult]:
        """Simulate *n_rounds* of the game and return all round results.

        Parameters
        ----------
        prob_a : float
            Player A cooperation probability in [0, 1].
        prob_b : float
            Player B cooperation probability in [0, 1].
        n_rounds : int
            Number of rounds to play.

        Returns
        -------
        List[RoundResult]
            One entry per round.

        Raises
        ------
        ValueError
            For invalid probabilities or non-positive n_rounds.
        """
        self._validate_inputs(prob_a, prob_b, n_rounds)

        logger.debug(
            "Playing %d rounds: pA=%.3f, pB=%.3f, encoding=%s, entanglement=%s",
            n_rounds, prob_a, prob_b, self.encoding.value, self.entanglement.value,
        )

        move_pairs = sample_outcomes(
            prob_a=prob_a,
            prob_b=prob_b,
            shots=n_rounds,
            encoding=self.encoding,
            entanglement=self.entanglement,
            rng=self._rng,
        )

        results: List[RoundResult] = []
        for move_a, move_b in move_pairs:
            score_a, score_b = self.payoff_model.get_payoff(move_a, move_b)
            results.append(
                RoundResult(
                    move_a=move_a,
                    move_b=move_b,
                    score_a=score_a,
                    score_b=score_b,
                    player_a_prob=prob_a,
                    player_b_prob=prob_b,
                )
            )

        return results

    def play_one_round(self, prob_a: float, prob_b: float) -> RoundResult:
        """Play exactly one round.  Convenience wrapper around play_rounds."""
        return self.play_rounds(prob_a, prob_b, 1)[0]

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_inputs(prob_a: float, prob_b: float, n_rounds: int) -> None:
        errors = []
        for name, val in (("prob_a", prob_a), ("prob_b", prob_b)):
            if not (0.0 <= float(val) <= 1.0):
                errors.append(f"{name}={val!r} must be in [0, 1].")
        if n_rounds <= 0:
            errors.append(f"n_rounds={n_rounds!r} must be a positive integer.")
        if errors:
            raise ValueError(" ".join(errors))
