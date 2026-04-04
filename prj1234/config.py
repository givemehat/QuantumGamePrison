"""
config.py
=========
Experiment configuration objects for the simulation layer.

Configurations can be constructed programmatically or loaded from
JSON / YAML files, making it easy to version-control experiments.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from quantum.quantum_core import EncodingScheme, EntanglementMode

logger = logging.getLogger(__name__)


@dataclass
class SingleRunConfig:
    """Configuration for a single fixed-probability experiment.

    Attributes
    ----------
    player_a_prob : float
        Player A cooperation probability in [0, 1].
    player_b_prob : float
        Player B cooperation probability in [0, 1].
    iterations : int
        Total number of game rounds.
    strategy_type : str
        E.g. ``"quantum_ry"``, ``"classical"``.
    payoff_mode : str
        Named game variant, e.g. ``"standard_pd"``.
    encoding : EncodingScheme
    entanglement : EntanglementMode
    name : str
        Short experiment name.
    description : str
    seed : int | None
        Random seed for reproducibility.
    """

    player_a_prob: float
    player_b_prob: float
    iterations: int = 1_000
    strategy_type: str = "quantum_ry"
    payoff_mode: str = "standard_pd"
    encoding: EncodingScheme = EncodingScheme.RY
    entanglement: EntanglementMode = EntanglementMode.NONE
    name: str = "single_run"
    description: str = ""
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        errors: List[str] = []
        for attr in ("player_a_prob", "player_b_prob"):
            val = getattr(self, attr)
            if not (0.0 <= float(val) <= 1.0):
                errors.append(f"{attr}={val!r} must be in [0, 1].")
        if self.iterations <= 0:
            errors.append(f"iterations={self.iterations!r} must be positive.")
        if errors:
            raise ValueError("Invalid SingleRunConfig: " + " ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["encoding"] = self.encoding.value
        d["entanglement"] = self.entanglement.value
        return d


@dataclass
class GridRunConfig:
    """Configuration for a 2-D probability sweep experiment.

    Parameters
    ----------
    pa_values : list of float
        Grid of Player A cooperation probabilities.
    pb_values : list of float
        Grid of Player B cooperation probabilities.
    iterations_per_cell : int
        Rounds played at each (pA, pB) grid point.
    strategy_type, payoff_mode, encoding, entanglement, name, description, seed
        Same semantics as SingleRunConfig.
    """

    pa_values: List[float]
    pb_values: List[float]
    iterations_per_cell: int = 500
    strategy_type: str = "quantum_ry"
    payoff_mode: str = "standard_pd"
    encoding: EncodingScheme = EncodingScheme.RY
    entanglement: EntanglementMode = EntanglementMode.NONE
    name: str = "grid_run"
    description: str = ""
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        errors: List[str] = []
        for lst_name, lst in (("pa_values", self.pa_values), ("pb_values", self.pb_values)):
            for v in lst:
                if not (0.0 <= float(v) <= 1.0):
                    errors.append(f"{lst_name} contains {v!r} which is not in [0, 1].")
        if self.iterations_per_cell <= 0:
            errors.append(f"iterations_per_cell={self.iterations_per_cell!r} must be positive.")
        if not self.pa_values or not self.pb_values:
            errors.append("pa_values and pb_values must be non-empty.")
        if errors:
            raise ValueError("Invalid GridRunConfig: " + " ".join(errors))

    @property
    def total_runs(self) -> int:
        """Total number of game rounds across all grid cells."""
        return len(self.pa_values) * len(self.pb_values) * self.iterations_per_cell

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pa_values": self.pa_values,
            "pb_values": self.pb_values,
            "iterations_per_cell": self.iterations_per_cell,
            "strategy_type": self.strategy_type,
            "payoff_mode": self.payoff_mode,
            "encoding": self.encoding.value,
            "entanglement": self.entanglement.value,
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
        }


# ── loaders ───────────────────────────────────────────────────────────────────

def _parse_config_dict(data: Dict[str, Any]) -> SingleRunConfig | GridRunConfig:
    """Dispatch a raw config dict to the appropriate config dataclass."""
    kind = data.get("kind", "single")
    enc = EncodingScheme(data.get("encoding", "ry"))
    ent = EntanglementMode(data.get("entanglement", "none"))
    common = {
        "strategy_type": data.get("strategy_type", "quantum_ry"),
        "payoff_mode": data.get("payoff_mode", "standard_pd"),
        "encoding": enc,
        "entanglement": ent,
        "name": data.get("name", "experiment"),
        "description": data.get("description", ""),
        "seed": data.get("seed"),
    }
    if kind == "grid":
        return GridRunConfig(
            pa_values=data["pa_values"],
            pb_values=data["pb_values"],
            iterations_per_cell=data.get("iterations_per_cell", 500),
            **common,
        )
    return SingleRunConfig(
        player_a_prob=data["player_a_prob"],
        player_b_prob=data["player_b_prob"],
        iterations=data.get("iterations", 1000),
        **common,
    )


def load_config_from_json(path: str) -> SingleRunConfig | GridRunConfig:
    """Load an experiment config from a JSON file.

    Parameters
    ----------
    path : str
        Path to a ``.json`` config file.

    Example JSON (single run)
    -------------------------
    .. code-block:: json

        {
            "kind": "single",
            "name": "my_experiment",
            "player_a_prob": 0.7,
            "player_b_prob": 0.5,
            "iterations": 2000,
            "strategy_type": "quantum_ry",
            "payoff_mode": "standard_pd",
            "encoding": "ry",
            "entanglement": "cnot"
        }
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: '{path}'.")
    with p.open() as fh:
        data = json.load(fh)
    return _parse_config_dict(data)


def load_config_from_yaml(path: str) -> SingleRunConfig | GridRunConfig:
    """Load an experiment config from a YAML file (requires PyYAML).

    Parameters
    ----------
    path : str
        Path to a ``.yaml`` / ``.yml`` config file.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for YAML config loading.") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: '{path}'.")
    with p.open() as fh:
        data = yaml.safe_load(fh)
    return _parse_config_dict(data)


def make_grid_from_range(
    pa_start: float,
    pa_end: float,
    pa_step: float,
    pb_start: float,
    pb_end: float,
    pb_step: float,
    **kwargs: Any,
) -> GridRunConfig:
    """Helper: create a GridRunConfig by specifying ranges instead of lists.

    Parameters
    ----------
    pa_start, pa_end, pa_step : float
        Range spec for Player A.
    pb_start, pb_end, pb_step : float
        Range spec for Player B.
    **kwargs
        Forwarded to GridRunConfig (iterations_per_cell, name, etc.).

    Example
    -------
    >>> cfg = make_grid_from_range(0.0, 1.0, 0.1, 0.0, 1.0, 0.1, iterations_per_cell=200)
    """
    import numpy as np

    def _range(start: float, end: float, step: float) -> List[float]:
        vals = np.arange(start, end + step / 2, step)
        return [round(float(v), 10) for v in vals if 0.0 <= v <= 1.0]

    pa_vals = _range(pa_start, pa_end, pa_step)
    pb_vals = _range(pb_start, pb_end, pb_step)
    if not pa_vals or not pb_vals:
        raise ValueError("Grid range produced no valid probability values.")
    return GridRunConfig(pa_values=pa_vals, pb_values=pb_vals, **kwargs)
