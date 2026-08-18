"""
strategies.py
=============
Strategy abstractions for the Quantum Prisoner's Dilemma.

A *strategy* encapsulates how a player determines their cooperation
probability for a given round.  All strategies expose a single method:
``get_cooperation_prob() -> float``.

Strategy types
--------------
ClassicalFixed        : constant probability p ∈ [0, 1]
ClassicalAlwaysC      : always cooperate (p=1.0)
ClassicalAlwaysD      : always defect   (p=0.0)
ClassicalTitForTat    : copy opponent's last move (needs history)
QuantumRy             : Ry-encoded quantum strategy
QuantumRx             : Rx-encoded quantum strategy
QuantumEntangled      : strategy that requires entanglement (flag only)
ParameterizedStrategy : any strategy described by (encoding, entanglement)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from quantum.quantum_core import EncodingScheme, EntanglementMode

logger = logging.getLogger(__name__)


# ── base class ────────────────────────────────────────────────────────────────


class BaseStrategy(ABC):
    """Abstract base for all player strategies."""

    @abstractmethod
    def get_cooperation_prob(self) -> float:
        """Return the cooperation probability for the current round."""

    @property
    def name(self) -> str:
        """Return the human-readable name of the strategy."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}()"


# ── classical strategies ──────────────────────────────────────────────────────


class ClassicalFixed(BaseStrategy):
    """Always return the same cooperation probability.

    Parameters
    ----------
    prob : float
        Cooperation probability in [0, 1].
    """

    def __init__(self, prob: float) -> None:
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"Probability must be in [0, 1], got {prob}.")
        self._prob = prob

    def get_cooperation_prob(self) -> float:
        """Return the strategy's fixed cooperation probability."""
        return self._prob

    @property
    def name(self) -> str:
        """Return the identifier for this strategy type."""
        return "ClassicalFixed"

    def __repr__(self) -> str:
        return f"ClassicalFixed(prob={self._prob:.3f})"


class ClassicalAlwaysC(ClassicalFixed):
    """Always cooperate."""

    def __init__(self) -> None:
        super().__init__(1.0)

    @property
    def name(self) -> str:
        """Return the identifier for the always-cooperate strategy."""
        return "AlwaysCooperate"


class ClassicalAlwaysD(ClassicalFixed):
    """Always defect."""

    def __init__(self) -> None:
        super().__init__(0.0)

    @property
    def name(self) -> str:
        """Return the identifier for the always-defect strategy."""
        return "AlwaysDefect"


class ClassicalTitForTat(BaseStrategy):
    """Tit-For-Tat: cooperate first, then mirror opponent's previous move.

    Parameters
    ----------
    opponent_history : list of str
        Running list of opponent moves (``"C"`` or ``"D"``).
    """

    def __init__(self, opponent_history: Optional[List[str]] = None) -> None:
        self._history: List[str] = opponent_history or []

    def record_opponent_move(self, move: str) -> None:
        """Append the opponent's latest move to history."""
        if move not in ("C", "D"):
            raise ValueError(f"Move must be 'C' or 'D', got '{move}'.")
        self._history.append(move)

    def get_cooperation_prob(self) -> float:
        if not self._history:
            return 1.0  # first round: cooperate
        return 1.0 if self._history[-1] == "C" else 0.0

    @property
    def name(self) -> str:
        return "TitForTat"

    def __repr__(self) -> str:
        return f"TitForTat(history_len={len(self._history)})"


# ── quantum strategies ────────────────────────────────────────────────────────


class QuantumRy(BaseStrategy):
    """Quantum strategy using Ry-angle encoding.

    Parameters
    ----------
    prob : float
        Target cooperation probability.
    """

    def __init__(self, prob: float) -> None:
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"Probability must be in [0, 1], got {prob}.")
        self._prob = prob

    def get_cooperation_prob(self) -> float:
        """Return the configured quantum cooperation probability."""
        return self._prob
    @property
    def encoding(self) -> EncodingScheme:
        """Return the Ry encoding scheme used by this strategy."""
        return EncodingScheme.RY

    @property
    def name(self) -> str:
        """Return the identifier for the Ry quantum strategy."""
        return "QuantumRy"

    def __repr__(self) -> str:
        return f"QuantumRy(prob={self._prob:.3f})"


class QuantumRx(BaseStrategy):
    """Quantum strategy using Rx-angle encoding."""

    def __init__(self, prob: float) -> None:
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"Probability must be in [0, 1], got {prob}.")
        self._prob = prob

    def get_cooperation_prob(self) -> float:
        """Return the configured quantum cooperation probability."""
        return self._prob

    @property
    def encoding(self) -> EncodingScheme:
        """Return the Rx encoding scheme used by this strategy."""
        return EncodingScheme.RX
    @property
    def name(self) -> str:
        """Return the identifier for the Rx quantum strategy."""
        return "QuantumRx"


class QuantumHRy(BaseStrategy):
    """Quantum strategy using H+Ry encoding (Hadamard superposition then rotation)."""

    def __init__(self, prob: float) -> None:
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"Probability must be in [0, 1], got {prob}.")
        self._prob = prob

    def get_cooperation_prob(self) -> float:
        """Return the configured quantum cooperation probability."""
        return self._prob

    @property
    def encoding(self) -> EncodingScheme:
        """Return the H+Ry encoding scheme used by this strategy."""
        return EncodingScheme.H_RY

    @property
    def name(self) -> str:
        """Return the identifier for the H+Ry quantum strategy."""
        return "QuantumHRy"


# ── parameterized strategy descriptor ─────────────────────────────────────────


@dataclass
class StrategySpec:
    """Complete specification for a player's strategy in an experiment.

    Attributes
    ----------
    prob : float
        Cooperation probability.
    encoding : EncodingScheme
        Angle encoding (only relevant for quantum strategies).
    entanglement : EntanglementMode
        Entanglement mode (shared between both players in the game engine).
    strategy_type : str
        Human-readable type label stored in the database.
    """

    prob: float
    encoding: EncodingScheme = EncodingScheme.RY
    entanglement: EntanglementMode = EntanglementMode.NONE
    strategy_type: str = "quantum_ry"

    def __post_init__(self) -> None:
        if not (0.0 <= self.prob <= 1.0):
            raise ValueError(f"prob must be in [0, 1], got {self.prob}.")

    def to_strategy(self) -> BaseStrategy:
        """Instantiate the matching BaseStrategy object."""
        mapping = {
            "classical": ClassicalFixed,
            "quantum_ry": QuantumRy,
            "quantum_rx": QuantumRx,
            "quantum_h_ry": QuantumHRy,
            "always_c": lambda p: ClassicalAlwaysC(),
            "always_d": lambda p: ClassicalAlwaysD(),
        }
        cls = mapping.get(self.strategy_type)
        if cls is None:
            raise ValueError(f"Unknown strategy_type: '{self.strategy_type}'.")
        return cls(self.prob)


STRATEGY_TYPES = [
    "classical",
    "quantum_ry",
    "quantum_rx",
    "quantum_h_ry",
    "always_c",
    "always_d",
]
"""All valid strategy type identifiers."""
