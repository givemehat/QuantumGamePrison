"""
runner.py
=========
Experiment runner — ties together config, game engine, and database.

Supports:
  * Single fixed-probability experiments
  * 2-D probability grid sweeps

For large experiments (e.g. 50 k–100 k runs) all inserts are batched
in a single transaction to minimise SQLite overhead.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

from game.engine import GameEngine, RoundResult
from game.payoff import get_payoff_model
from quantum.quantum_core import EncodingScheme, EntanglementMode
from simulation.config import GridRunConfig, SingleRunConfig
from storage.database import (
    DEFAULT_DB_PATH,
    export_csv,
    insert_experiment,
    insert_game_runs,
)
from storage.models import ExperimentConfig, GameRun

logger = logging.getLogger(__name__)

# ── batch size for DB inserts ─────────────────────────────────────────────────
_INSERT_BATCH = 10_000


class ExperimentRunner:
    """Run simulation experiments and persist results to SQLite.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database (default ``"quantum_game.db"``).
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path

    # ── public API ────────────────────────────────────────────────────────────

    def run_single(self, config: SingleRunConfig) -> int:
        """Run a single fixed-probability experiment.

        Parameters
        ----------
        config : SingleRunConfig

        Returns
        -------
        int
            The experiment ID assigned by the database.
        """
        payoff_model = get_payoff_model(config.payoff_mode)
        engine = GameEngine(
            payoff_model=payoff_model,
            encoding=config.encoding,
            entanglement=config.entanglement,
            seed=config.seed,
        )
        exp_config = ExperimentConfig(
            name=config.name,
            description=config.description,
            strategy_type=config.strategy_type,
            payoff_mode=config.payoff_mode,
            config_json=config.to_dict(),
        )
        experiment_id = insert_experiment(exp_config, self.db_path)

        logger.info(
            "Starting single experiment id=%d  pA=%.3f  pB=%.3f  iterations=%d",
            experiment_id,
            config.player_a_prob,
            config.player_b_prob,
            config.iterations,
        )
        t0 = time.perf_counter()

        results = engine.play_rounds(
            config.player_a_prob, config.player_b_prob, config.iterations
        )
        runs = self._results_to_game_runs(results, experiment_id)
        self._batch_insert(runs)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Experiment id=%d complete: %d rounds in %.2fs.",
            experiment_id,
            config.iterations,
            elapsed,
        )
        return experiment_id

    def run_grid(self, config: GridRunConfig) -> int:
        """Run a 2-D grid sweep experiment.

        Each (pA, pB) cell is played for ``config.iterations_per_cell`` rounds.
        All results are stored under a single experiment ID.

        Parameters
        ----------
        config : GridRunConfig

        Returns
        -------
        int
            The experiment ID.
        """
        payoff_model = get_payoff_model(config.payoff_mode)
        engine = GameEngine(
            payoff_model=payoff_model,
            encoding=config.encoding,
            entanglement=config.entanglement,
            seed=config.seed,
        )
        exp_config = ExperimentConfig(
            name=config.name,
            description=config.description,
            strategy_type=config.strategy_type,
            payoff_mode=config.payoff_mode,
            config_json=config.to_dict(),
        )
        experiment_id = insert_experiment(exp_config, self.db_path)

        total_cells = len(config.pa_values) * len(config.pb_values)
        logger.info(
            "Starting grid experiment id=%d  cells=%d  total_runs=%d",
            experiment_id,
            total_cells,
            config.total_runs,
        )
        t0 = time.perf_counter()

        buffer: List[GameRun] = []
        cells_done = 0

        for pa in config.pa_values:
            for pb in config.pb_values:
                results = engine.play_rounds(pa, pb, config.iterations_per_cell)
                buffer.extend(self._results_to_game_runs(results, experiment_id))
                cells_done += 1

                if len(buffer) >= _INSERT_BATCH:
                    self._batch_insert(buffer)
                    buffer.clear()
                    logger.debug("Grid progress: %d/%d cells", cells_done, total_cells)

        if buffer:
            self._batch_insert(buffer)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Grid experiment id=%d complete: %d cells, %d total runs in %.2fs.",
            experiment_id,
            cells_done,
            config.total_runs,
            elapsed,
        )
        return experiment_id

    def export_results_csv(self, experiment_id: int, output_path: str) -> int:
        """Export all results for an experiment to CSV.

        Returns the number of exported rows.
        """
        return export_csv(experiment_id, output_path, self.db_path)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _results_to_game_runs(
        results: List[RoundResult],
        experiment_id: int,
    ) -> List[GameRun]:
        return [
            GameRun(
                experiment_id=experiment_id,
                player_a_prob=r.player_a_prob,
                player_b_prob=r.player_b_prob,
                outcome=r.outcome,
                player_a_score=r.score_a,
                player_b_score=r.score_b,
            )
            for r in results
        ]

    def _batch_insert(self, runs: List[GameRun]) -> None:
        insert_game_runs(runs, self.db_path)
