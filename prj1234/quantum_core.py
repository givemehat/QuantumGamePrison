"""
quantum_core.py
===============
Low-level quantum circuit construction and probability sampling for the
Quantum Prisoner's Dilemma.

All circuits operate on 2 qubits (qubit 0 → Player A, qubit 1 → Player B).
Measurement convention:
    |0⟩ → Cooperate (C)
    |1⟩ → Defect    (D)

Encoding schemes
----------------
Ry  : Ry(θ)|0⟩  → cos(θ/2)|0⟩ + sin(θ/2)|1⟩
      P(C) = cos²(θ/2)  →  θ = 2·arccos(√p)
Rx  : Rx(θ)|0⟩  → cos(θ/2)|0⟩ − i·sin(θ/2)|1⟩
      P(C) = cos²(θ/2)  →  θ = 2·arccos(√p)  (identical magnitude)
H+Ry: Hadamard followed by Ry (uniform superposition then rotation)

Entanglement modes
------------------
none       : independent qubits
cnot       : CNOT(0→1) after encoding — correlates strategies
cz         : CZ(0,1) after encoding
bell_init  : initialise in Bell state |Φ+⟩ then apply individual rotations
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── availability guards ──────────────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not found. Falling back to numpy-only sampling.")


# ── enumerations ─────────────────────────────────────────────────────────────


class EncodingScheme(str, Enum):
    """Probability-to-angle encoding strategy."""

    RY = "ry"
    RX = "rx"
    H_RY = "h_ry"


class EntanglementMode(str, Enum):
    """Post-encoding entangling operation."""

    NONE = "none"
    CNOT = "cnot"
    CZ = "cz"
    BELL_INIT = "bell_init"


# ── move labels ──────────────────────────────────────────────────────────────

BITSTRING_TO_MOVES: Dict[str, Tuple[str, str]] = {
    "00": ("C", "C"),
    "01": ("C", "D"),
    "10": ("D", "C"),
    "11": ("D", "D"),
}
"""
Maps 2-qubit measurement bitstrings to (Player A move, Player B move).
Qiskit's bitstring ordering: rightmost bit = qubit 0 (Player A).
  "00" → A=C, B=C
  "01" → A=C, B=D   (qubit 1 measured 1 → B defects)
  "10" → A=D, B=C   (qubit 0 measured 1 → A defects)
  "11" → A=D, B=D
"""


def bitstring_to_moves(bitstring: str) -> Tuple[str, str]:
    """Return (move_A, move_B) from a 2-bit measurement string.

    Parameters
    ----------
    bitstring : str
        Two-character string of '0'/'1', e.g. ``"01"``.

    Returns
    -------
    Tuple[str, str]
        ``("C" | "D", "C" | "D")`` for Player A and Player B.

    Raises
    ------
    ValueError
        If *bitstring* is not a valid 2-bit string.
    """
    if bitstring not in BITSTRING_TO_MOVES:
        raise ValueError(
            f"Invalid bitstring '{bitstring}'. Must be one of {list(BITSTRING_TO_MOVES)}."
        )
    return BITSTRING_TO_MOVES[bitstring]


# ── angle helpers ─────────────────────────────────────────────────────────────


def prob_to_ry_angle(prob: float) -> float:
    """Convert a cooperation probability to an Ry rotation angle.

    Uses the relation P(C) = cos²(θ/2), so θ = 2·arccos(√p).

    Parameters
    ----------
    prob : float
        Cooperation probability in [0, 1].

    Returns
    -------
    float
        Rotation angle θ in radians.
    """
    prob = float(np.clip(prob, 0.0, 1.0))
    return 2.0 * math.acos(math.sqrt(prob))


def prob_to_rx_angle(prob: float) -> float:
    """Convert a cooperation probability to an Rx rotation angle.

    Identical magnitude relationship to Ry: P(C) = cos²(θ/2).
    """
    return prob_to_ry_angle(prob)


# ── circuit builders ──────────────────────────────────────────────────────────


def build_circuit(
    prob_a: float,
    prob_b: float,
    encoding: EncodingScheme = EncodingScheme.RY,
    entanglement: EntanglementMode = EntanglementMode.NONE,
) -> "QuantumCircuit":
    """Build a 2-qubit Qiskit circuit for the Prisoner's Dilemma.

    Parameters
    ----------
    prob_a : float
        Player A cooperation probability in [0, 1].
    prob_b : float
        Player B cooperation probability in [0, 1].
    encoding : EncodingScheme
        Angle encoding scheme.
    entanglement : EntanglementMode
        Post-encoding entanglement operation.

    Returns
    -------
    QuantumCircuit
        Unmeasured 2-qubit circuit (measurement added separately).

    Raises
    ------
    ImportError
        If Qiskit is not installed.
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required to build quantum circuits.")

    qc = QuantumCircuit(2)

    if entanglement == EntanglementMode.BELL_INIT:
        # Start in Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        qc.h(0)
        qc.cx(0, 1)

    # --- per-player encoding ---
    theta_a = _compute_angle(prob_a, encoding)
    theta_b = _compute_angle(prob_b, encoding)

    if encoding == EncodingScheme.RY:
        qc.ry(theta_a, 0)
        qc.ry(theta_b, 1)
    elif encoding == EncodingScheme.RX:
        qc.rx(theta_a, 0)
        qc.rx(theta_b, 1)
    elif encoding == EncodingScheme.H_RY:
        qc.h(0)
        qc.ry(theta_a, 0)
        qc.h(1)
        qc.ry(theta_b, 1)

    # --- entanglement layer ---
    if entanglement == EntanglementMode.CNOT:
        qc.cx(0, 1)
    elif entanglement == EntanglementMode.CZ:
        qc.cz(0, 1)
    # BELL_INIT already handled above; NONE needs nothing

    return qc


def _compute_angle(prob: float, encoding: EncodingScheme) -> float:
    """Internal helper: dispatch angle computation by encoding scheme."""
    if encoding in (EncodingScheme.RY, EncodingScheme.H_RY):
        return prob_to_ry_angle(prob)
    elif encoding == EncodingScheme.RX:
        return prob_to_rx_angle(prob)
    raise ValueError(f"Unknown encoding scheme: {encoding}")


# ── probability distributions ─────────────────────────────────────────────────


def get_exact_probabilities(
    prob_a: float,
    prob_b: float,
    encoding: EncodingScheme = EncodingScheme.RY,
    entanglement: EntanglementMode = EntanglementMode.NONE,
) -> Dict[str, float]:
    """Compute the exact outcome probability distribution via Statevector.

    Parameters
    ----------
    prob_a, prob_b : float
        Cooperation probabilities.
    encoding : EncodingScheme
    entanglement : EntanglementMode

    Returns
    -------
    Dict[str, float]
        Mapping ``{"00": p, "01": p, "10": p, "11": p}`` where values sum to 1.
    """
    if not QISKIT_AVAILABLE:
        return _numpy_exact_probs(prob_a, prob_b)

    qc = build_circuit(prob_a, prob_b, encoding, entanglement)
    sv = Statevector(qc)
    probs = sv.probabilities_dict()
    # Ensure all 4 keys are present, pad with 0
    full: Dict[str, float] = {bs: probs.get(bs, 0.0) for bs in BITSTRING_TO_MOVES}
    return full


def _numpy_exact_probs(prob_a: float, prob_b: float) -> Dict[str, float]:
    """Numpy-only fallback for independent (non-entangled) Ry encoding."""
    pa_c = float(np.clip(prob_a, 0.0, 1.0))
    pb_c = float(np.clip(prob_b, 0.0, 1.0))
    pa_d = 1.0 - pa_c
    pb_d = 1.0 - pb_c
    return {
        "00": pa_c * pb_c,
        "01": pa_c * pb_d,
        "10": pa_d * pb_c,
        "11": pa_d * pb_d,
    }


def sample_outcomes(
    prob_a: float,
    prob_b: float,
    shots: int,
    encoding: EncodingScheme = EncodingScheme.RY,
    entanglement: EntanglementMode = EntanglementMode.NONE,
    rng: np.random.Generator | None = None,
) -> List[Tuple[str, str]]:
    """Sample *shots* (move_A, move_B) outcomes from the quantum circuit.

    Uses exact Statevector probabilities and samples with numpy for speed.

    Parameters
    ----------
    prob_a, prob_b : float
        Cooperation probabilities.
    shots : int
        Number of samples to draw.
    encoding : EncodingScheme
    entanglement : EntanglementMode
    rng : numpy Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    List[Tuple[str, str]]
        Each element is (``"C"``|``"D"``, ``"C"``|``"D"``).
    """
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}.")

    rng = rng or np.random.default_rng()
    probs = get_exact_probabilities(prob_a, prob_b, encoding, entanglement)
    bitstrings = list(probs.keys())
    weights = np.array([probs[b] for b in bitstrings], dtype=float)
    weights /= weights.sum()  # normalise for floating-point safety

    chosen = rng.choice(len(bitstrings), size=shots, p=weights)
    return [bitstring_to_moves(bitstrings[i]) for i in chosen]


def sample_single_outcome(
    prob_a: float,
    prob_b: float,
    encoding: EncodingScheme = EncodingScheme.RY,
    entanglement: EntanglementMode = EntanglementMode.NONE,
    rng: np.random.Generator | None = None,
) -> Tuple[str, str]:
    """Convenience wrapper: draw exactly one outcome."""
    return sample_outcomes(prob_a, prob_b, 1, encoding, entanglement, rng)[0]
