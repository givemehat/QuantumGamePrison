"""
tests/test_suite.py
===================
Unit and integration tests for the Quantum Prisoner's Dilemma framework.

Run with:
    pytest tests/test_suite.py -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# quantum_core tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBitstringMapping(unittest.TestCase):
    """Validate bitstring → move mapping logic."""

    def test_all_four_outcomes(self):
        from quantum.quantum_core import bitstring_to_moves

        self.assertEqual(bitstring_to_moves("00"), ("C", "C"))
        self.assertEqual(bitstring_to_moves("01"), ("C", "D"))
        self.assertEqual(bitstring_to_moves("10"), ("D", "C"))
        self.assertEqual(bitstring_to_moves("11"), ("D", "D"))

    def test_invalid_bitstring_raises(self):
        from quantum.quantum_core import bitstring_to_moves

        with self.assertRaises(ValueError):
            bitstring_to_moves("22")
        with self.assertRaises(ValueError):
            bitstring_to_moves("0")


class TestAngleConversion(unittest.TestCase):
    """Validate probability-to-angle encoding."""

    def test_always_cooperate(self):
        from quantum.quantum_core import prob_to_ry_angle

        angle = prob_to_ry_angle(1.0)
        self.assertAlmostEqual(angle, 0.0, places=6)

    def test_always_defect(self):
        from quantum.quantum_core import prob_to_ry_angle

        angle = prob_to_ry_angle(0.0)
        self.assertAlmostEqual(angle, math.pi, places=6)

    def test_half_probability(self):
        from quantum.quantum_core import prob_to_ry_angle

        angle = prob_to_ry_angle(0.5)
        # cos²(angle/2)=0.5 → angle=π/2
        self.assertAlmostEqual(angle, math.pi / 2, places=6)

    def test_round_trip(self):
        from quantum.quantum_core import prob_to_ry_angle

        for p in [0.1, 0.3, 0.7, 0.9]:
            angle = prob_to_ry_angle(p)
            recovered = math.cos(angle / 2) ** 2
            self.assertAlmostEqual(recovered, p, places=6)


class TestProbabilities(unittest.TestCase):
    """Exact probabilities must sum to 1 and match independent encoding."""

    def test_probs_sum_to_one(self):
        from quantum.quantum_core import get_exact_probabilities

        probs = get_exact_probabilities(0.6, 0.4)
        total = sum(probs.values())
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_always_cooperate_gives_cc(self):
        from quantum.quantum_core import get_exact_probabilities

        probs = get_exact_probabilities(1.0, 1.0)
        self.assertAlmostEqual(probs["00"], 1.0, places=6)
        self.assertAlmostEqual(probs["11"], 0.0, places=6)

    def test_always_defect_gives_dd(self):
        from quantum.quantum_core import get_exact_probabilities

        probs = get_exact_probabilities(0.0, 0.0)
        self.assertAlmostEqual(probs["11"], 1.0, places=6)

    def test_independent_product(self):
        from quantum.quantum_core import get_exact_probabilities, EntanglementMode

        pa, pb = 0.7, 0.4
        probs = get_exact_probabilities(pa, pb, entanglement=EntanglementMode.NONE)
        # C=0, D=1 → P(00) = P(A=C)*P(B=C) = pa*pb
        self.assertAlmostEqual(probs["00"], pa * pb, places=4)


class TestSampling(unittest.TestCase):
    """Sampling must return valid moves and approximate target probabilities."""

    def test_sample_returns_valid_moves(self):
        from quantum.quantum_core import sample_outcomes

        outcomes = sample_outcomes(0.6, 0.5, shots=100)
        for a, b in outcomes:
            self.assertIn(a, ("C", "D"))
            self.assertIn(b, ("C", "D"))

    def test_sample_length(self):
        from quantum.quantum_core import sample_outcomes

        outcomes = sample_outcomes(0.5, 0.5, shots=50)
        self.assertEqual(len(outcomes), 50)

    def test_sample_invalid_shots(self):
        from quantum.quantum_core import sample_outcomes

        with self.assertRaises(ValueError):
            sample_outcomes(0.5, 0.5, shots=0)

    def test_always_cooperate_sampling(self):
        from quantum.quantum_core import sample_outcomes

        rng = np.random.default_rng(42)
        outcomes = sample_outcomes(1.0, 1.0, shots=200, rng=rng)
        self.assertTrue(all(a == "C" and b == "C" for a, b in outcomes))


# ═══════════════════════════════════════════════════════════════════════════════
# payoff tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPayoffMatrix(unittest.TestCase):

    def setUp(self):
        from game.payoff import STANDARD_PD

        self.model = STANDARD_PD

    def test_cc_payoff(self):
        self.assertEqual(self.model.get_payoff("C", "C"), (3.0, 3.0))

    def test_cd_payoff(self):
        self.assertEqual(self.model.get_payoff("C", "D"), (0.0, 5.0))

    def test_dc_payoff(self):
        self.assertEqual(self.model.get_payoff("D", "C"), (5.0, 0.0))

    def test_dd_payoff(self):
        self.assertEqual(self.model.get_payoff("D", "D"), (1.0, 1.0))

    def test_invalid_move_raises(self):
        with self.assertRaises(ValueError):
            self.model.get_payoff("X", "C")

    def test_all_named_games_valid(self):
        from game.payoff import NAMED_GAMES

        for name, model in NAMED_GAMES.items():
            for a in ("C", "D"):
                for b in ("C", "D"):
                    sa, sb = model.get_payoff(a, b)
                    self.assertIsInstance(sa, float)
                    self.assertIsInstance(sb, float)

    def test_custom_payoff_from_dict(self):
        from game.payoff import payoff_model_from_dict

        m = payoff_model_from_dict(
            "test", {"CC": (4, 4), "CD": (0, 6), "DC": (6, 0), "DD": (2, 2)}
        )
        self.assertEqual(m.get_payoff("C", "C"), (4.0, 4.0))


# ═══════════════════════════════════════════════════════════════════════════════
# game engine tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGameEngine(unittest.TestCase):

    def setUp(self):
        from game.engine import GameEngine
        from game.payoff import STANDARD_PD

        self.engine = GameEngine(payoff_model=STANDARD_PD, seed=42)

    def test_play_rounds_returns_correct_count(self):
        results = self.engine.play_rounds(0.5, 0.5, 100)
        self.assertEqual(len(results), 100)

    def test_play_round_valid_outcomes(self):
        results = self.engine.play_rounds(0.6, 0.4, 50)
        for r in results:
            self.assertIn(r.move_a, ("C", "D"))
            self.assertIn(r.move_b, ("C", "D"))
            self.assertIn(r.outcome, ("CC", "CD", "DC", "DD"))

    def test_play_round_scores_nonnegative(self):
        results = self.engine.play_rounds(0.5, 0.5, 50)
        for r in results:
            self.assertGreaterEqual(r.score_a, 0.0)
            self.assertGreaterEqual(r.score_b, 0.0)

    def test_invalid_prob_raises(self):
        from game.engine import GameEngine

        engine = GameEngine()
        with self.assertRaises(ValueError):
            engine.play_rounds(1.5, 0.5, 10)

    def test_invalid_n_rounds_raises(self):
        from game.engine import GameEngine

        engine = GameEngine()
        with self.assertRaises(ValueError):
            engine.play_rounds(0.5, 0.5, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# database tests (in-memory SQLite)
# ═══════════════════════════════════════════════════════════════════════════════

import os
import tempfile

# Each test class that needs a real persistent SQLite file gets one via
# get_test_db() below.  We use a module-level temp file so tests within
# the same class share state (needed for FK constraints between tables).
_tmp_db_fd, _tmp_db_path = tempfile.mkstemp(suffix=".db", prefix="qpd_test_")
os.close(_tmp_db_fd)
DB = _tmp_db_path


class TestDatabase(unittest.TestCase):

    def setUp(self):
        from storage.database import initialise_db

        initialise_db(DB)

    def _make_experiment(self, name: str = "test_exp"):
        from storage.models import ExperimentConfig

        return ExperimentConfig(
            name=name,
            description="unit test",
            strategy_type="quantum_ry",
            payoff_mode="standard_pd",
        )

    def test_insert_and_retrieve_experiment(self):
        from storage.database import get_experiment, insert_experiment

        cfg = self._make_experiment()
        exp_id = insert_experiment(cfg, DB)
        retrieved = get_experiment(exp_id, DB)
        self.assertEqual(retrieved.name, "test_exp")
        self.assertEqual(retrieved.strategy_type, "quantum_ry")

    def test_missing_experiment_raises_key_error(self):
        from storage.database import get_experiment

        with self.assertRaises(KeyError):
            get_experiment(9999, DB)

    def test_insert_and_fetch_runs(self):
        from storage.database import (
            count_runs,
            fetch_runs,
            insert_experiment,
            insert_game_runs,
        )
        from storage.models import ExperimentConfig, GameRun

        cfg = self._make_experiment("batch_test")
        exp_id = insert_experiment(cfg, DB)

        runs = [
            GameRun(
                experiment_id=exp_id,
                player_a_prob=0.7,
                player_b_prob=0.5,
                outcome="CC",
                player_a_score=3.0,
                player_b_score=3.0,
            )
            for _ in range(20)
        ]
        insert_game_runs(runs, DB)
        self.assertEqual(count_runs(exp_id, DB), 20)
        fetched = fetch_runs(exp_id, DB)
        self.assertEqual(len(fetched), 20)
        self.assertEqual(fetched[0]["outcome"], "CC")

    def test_invalid_outcome_raises(self):
        from storage.models import GameRun

        with self.assertRaises(ValueError):
            GameRun(
                experiment_id=1,
                player_a_prob=0.5,
                player_b_prob=0.5,
                outcome="XY",
                player_a_score=3.0,
                player_b_score=3.0,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# simulation runner tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSimulationRunner(unittest.TestCase):

    def _runner(self):
        from simulation.runner import ExperimentRunner

        return ExperimentRunner(db_path=DB)

    def test_single_run_produces_results(self):
        from simulation.config import SingleRunConfig

        config = SingleRunConfig(
            player_a_prob=0.6,
            player_b_prob=0.4,
            iterations=50,
            seed=42,
        )
        runner = self._runner()
        exp_id = runner.run_single(config)

        from storage.database import count_runs

        self.assertEqual(count_runs(exp_id, DB), 50)

    def test_grid_run_correct_total_count(self):
        from simulation.config import GridRunConfig

        config = GridRunConfig(
            pa_values=[0.3, 0.7],
            pb_values=[0.3, 0.7],
            iterations_per_cell=10,
            seed=42,
        )
        runner = self._runner()
        exp_id = runner.run_grid(config)

        from storage.database import count_runs

        # 2 × 2 × 10 = 40
        self.assertEqual(count_runs(exp_id, DB), 40)


# ═══════════════════════════════════════════════════════════════════════════════
# analysis tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create a small experiment to analyse."""
        from simulation.config import SingleRunConfig
        from simulation.runner import ExperimentRunner

        runner = ExperimentRunner(db_path=DB)
        config = SingleRunConfig(
            player_a_prob=0.8,
            player_b_prob=0.6,
            iterations=200,
            seed=0,
        )
        cls.exp_id = runner.run_single(config)

    def test_outcome_frequencies_sum(self):
        from storage.database import fetch_runs
        from analysis.summary import compute_outcome_frequencies

        rows = fetch_runs(self.exp_id, DB)
        counts = compute_outcome_frequencies(rows)
        self.assertEqual(sum(counts.values()), 200)

    def test_score_stats_keys(self):
        from storage.database import fetch_runs
        from analysis.summary import compute_score_stats

        rows = fetch_runs(self.exp_id, DB)
        stats = compute_score_stats(rows, player="a")
        for key in ("mean", "median", "std", "min", "max", "variance"):
            self.assertIn(key, stats)

    def test_experiment_stats(self):
        from analysis.summary import compute_experiment_stats

        stats = compute_experiment_stats(self.exp_id, DB)
        self.assertEqual(stats.total_runs, 200)
        self.assertGreaterEqual(stats.cooperation_rate_a, 0.0)
        self.assertLessEqual(stats.cooperation_rate_a, 1.0)

    def test_rolling_scores_length(self):
        from storage.database import fetch_runs
        from analysis.summary import compute_rolling_scores

        rows = fetch_runs(self.exp_id, DB)
        ra, rb = compute_rolling_scores(rows, window=10)
        self.assertEqual(len(ra), 200 - 10 + 1)


# ═══════════════════════════════════════════════════════════════════════════════
# strategies tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategies(unittest.TestCase):

    def test_classical_fixed(self):
        from quantum.strategies import ClassicalFixed

        s = ClassicalFixed(0.75)
        self.assertAlmostEqual(s.get_cooperation_prob(), 0.75)

    def test_always_c(self):
        from quantum.strategies import ClassicalAlwaysC

        self.assertAlmostEqual(ClassicalAlwaysC().get_cooperation_prob(), 1.0)

    def test_always_d(self):
        from quantum.strategies import ClassicalAlwaysD

        self.assertAlmostEqual(ClassicalAlwaysD().get_cooperation_prob(), 0.0)

    def test_tit_for_tat_first_round(self):
        from quantum.strategies import ClassicalTitForTat

        tft = ClassicalTitForTat()
        self.assertAlmostEqual(tft.get_cooperation_prob(), 1.0)

    def test_tit_for_tat_after_defect(self):
        from quantum.strategies import ClassicalTitForTat

        tft = ClassicalTitForTat()
        tft.record_opponent_move("D")
        self.assertAlmostEqual(tft.get_cooperation_prob(), 0.0)

    def test_invalid_prob_raises(self):
        from quantum.strategies import ClassicalFixed

        with self.assertRaises(ValueError):
            ClassicalFixed(1.5)

    def test_strategy_spec_to_strategy(self):
        from quantum.strategies import StrategySpec
        from quantum.quantum_core import EncodingScheme, EntanglementMode

        spec = StrategySpec(prob=0.5, strategy_type="classical")
        s = spec.to_strategy()
        self.assertAlmostEqual(s.get_cooperation_prob(), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
