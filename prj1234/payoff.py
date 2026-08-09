"""
payoff.py
=========
Payoff matrix abstractions for the Prisoner's Dilemma and variants.

A payoff matrix is a mapping:

    (move_A, move_B) → (score_A, score_B)

where each move is ``"C"`` (cooperate) or ``"D"`` (defect).

Named variants
--------------
standard_pd : classic PD  (C,C)→(3,3)  (C,D)→(0,5)  (D,C)→(5,0)  (D,D)→(1,1)
harsh_pd    : harsher mutual cooperation penalty
lenient_pd  : extra reward for mutual cooperation
stag_hunt   : coordination game
chicken     : anti-coordination game
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Type aliases
Move = str  # "C" or "D"
Outcome = Tuple[Move, Move]  # (move_A, move_B)
Payoff = Tuple[float, float]  # (score_A, score_B)
PayoffMatrix = Dict[Outcome, Payoff]


@dataclass(frozen=True)
class PayoffModel:
    """Immutable payoff model for a symmetric 2-player game.

    Parameters
    ----------
    matrix : PayoffMatrix
        Must define all four outcomes: CC, CD, DC, DD.
    name : str
        Human-readable identifier (stored in experiments table).
    description : str
        Brief description of the game variant.
    """

    matrix: PayoffMatrix
    name: str
    description: str = ""

    def __post_init__(self) -> None:
        required = {("C", "C"), ("C", "D"), ("D", "C"), ("D", "D")}
        if not required.issubset(self.matrix.keys()):
            missing = required - set(self.matrix.keys())
            raise ValueError(f"PayoffMatrix is missing outcomes: {missing}.")

    def get_payoff(self, move_a: Move, move_b: Move) -> Payoff:
        """Return (score_A, score_B) for a given pair of moves.

        Parameters
        ----------
        move_a : str
            Player A's move (``"C"`` or ``"D"``).
        move_b : str
            Player B's move (``"C"`` or ``"D"``).

        Returns
        -------
        Tuple[float, float]

        Raises
        ------
        ValueError
            If either move is not ``"C"`` or ``"D"``.
        """
        if move_a not in ("C", "D") or move_b not in ("C", "D"):
            raise ValueError(
                f"Moves must be 'C' or 'D', got move_a='{move_a}', move_b='{move_b}'."
            )
        return self.matrix[(move_a, move_b)]

    def outcome_label(self, move_a: Move, move_b: Move) -> str:
        """Return the outcome label string, e.g. ``'CC'``, ``'CD'``."""
        return move_a + move_b


# ── built-in game definitions ─────────────────────────────────────────────────

STANDARD_PD = PayoffModel(
    name="standard_pd",
    description="Classic Prisoner's Dilemma: mutual cooperation rewarded, "
    "defection tempting.",
    matrix={
        ("C", "C"): (3.0, 3.0),
        ("C", "D"): (0.0, 5.0),
        ("D", "C"): (5.0, 0.0),
        ("D", "D"): (1.0, 1.0),
    },
)

HARSH_PD = PayoffModel(
    name="harsh_pd",
    description="Harsh PD: mutual defection is more punishing.",
    matrix={
        ("C", "C"): (3.0, 3.0),
        ("C", "D"): (0.0, 5.0),
        ("D", "C"): (5.0, 0.0),
        ("D", "D"): (0.0, 0.0),
    },
)

LENIENT_PD = PayoffModel(
    name="lenient_pd",
    description="Lenient PD: mutual cooperation yields higher reward.",
    matrix={
        ("C", "C"): (5.0, 5.0),
        ("C", "D"): (0.0, 7.0),
        ("D", "C"): (7.0, 0.0),
        ("D", "D"): (1.0, 1.0),
    },
)

STAG_HUNT = PayoffModel(
    name="stag_hunt",
    description="Stag Hunt coordination game: mutual cooperation maximises joint payoff.",
    matrix={
        ("C", "C"): (4.0, 4.0),
        ("C", "D"): (0.0, 3.0),
        ("D", "C"): (3.0, 0.0),
        ("D", "D"): (3.0, 3.0),
    },
)

CHICKEN = PayoffModel(
    name="chicken",
    description="Chicken (Hawk-Dove): mutual defection is catastrophic.",
    matrix={
        ("C", "C"): (3.0, 3.0),
        ("C", "D"): (2.0, 4.0),
        ("D", "C"): (4.0, 2.0),
        ("D", "D"): (0.0, 0.0),
    },
)

NAMED_GAMES: Dict[str, PayoffModel] = {
    "standard_pd": STANDARD_PD,
    "harsh_pd": HARSH_PD,
    "lenient_pd": LENIENT_PD,
    "stag_hunt": STAG_HUNT,
    "chicken": CHICKEN,
}
"""Registry of all named game variants."""


def get_payoff_model(name: str) -> PayoffModel:
    """Retrieve a named PayoffModel.

    Parameters
    ----------
    name : str
        Game identifier (e.g. ``"standard_pd"``).

    Raises
    ------
    ValueError
        If *name* is not a registered game.
    """
    model = NAMED_GAMES.get(name)
    if model is None:
        raise ValueError(
            f"Unknown payoff mode '{name}'. Valid options: {list(NAMED_GAMES)}."
        )
    return model


def payoff_model_from_dict(
    name: str,
    matrix_dict: Dict[str, Tuple[float, float]],
    description: str = "",
) -> PayoffModel:
    """Build a custom PayoffModel from a plain dict.

    Parameters
    ----------
    name : str
        Identifier for the new model.
    matrix_dict : dict
        Keys like ``"CC"``, ``"CD"``, ``"DC"``, ``"DD"``; values ``(score_a, score_b)``.
    description : str, optional

    Example
    -------
    >>> m = payoff_model_from_dict("my_game", {"CC":(4,4),"CD":(0,6),"DC":(6,0),"DD":(2,2)})
    """
    key_map = {"CC": ("C", "C"), "CD": ("C", "D"), "DC": ("D", "C"), "DD": ("D", "D")}
    matrix: PayoffMatrix = {}
    for k, v in matrix_dict.items():
        outcome = key_map.get(k.upper())
        if outcome is None:
            raise ValueError(f"Invalid outcome key '{k}'. Use 'CC', 'CD', 'DC', 'DD'.")
        matrix[outcome] = (float(v[0]), float(v[1]))
    return PayoffModel(name=name, matrix=matrix, description=description)
