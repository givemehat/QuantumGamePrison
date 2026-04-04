"""
database.py
===========
SQLite data-access layer for the Quantum Prisoner's Dilemma.

Schema
------
experiments(id, name, description, strategy_type, payoff_mode, config_json, created_at)
game_results(id, experiment_id, player_a_prob, player_b_prob, outcome,
             player_a_score, player_b_score, timestamp)

All public functions accept an explicit ``db_path`` argument so the
database location can be overridden in tests (use ``:memory:``).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from storage.models import ExperimentConfig, GameRun

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "quantum_game.db"

# ── DDL ──────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS experiments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    description   TEXT    NOT NULL DEFAULT '',
    strategy_type TEXT    NOT NULL,
    payoff_mode   TEXT    NOT NULL,
    config_json   TEXT    NOT NULL DEFAULT '{}',
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS game_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    player_a_prob   REAL    NOT NULL,
    player_b_prob   REAL    NOT NULL,
    outcome         TEXT    NOT NULL,
    player_a_score  REAL    NOT NULL,
    player_b_score  REAL    NOT NULL,
    timestamp       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gr_experiment
    ON game_results(experiment_id);
"""


# ── connection helper ─────────────────────────────────────────────────────────

@contextmanager
def _connect(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding an initialised connection with row_factory set.

    The schema DDL runs on every open (``CREATE TABLE IF NOT EXISTS`` makes
    it idempotent).  For real databases this costs a trivial amount; it
    ensures that even a freshly-created file is always ready to use.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_DDL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── initialisation ────────────────────────────────────────────────────────────

def initialise_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Ensure the database schema exists (idempotent).

    Parameters
    ----------
    db_path : str
        Path to the SQLite file (or ``:memory:`` for tests).
    """
    with _connect(db_path) as conn:
        pass  # DDL is run inside _connect; this call just confirms connectivity
    logger.debug("Database initialised at '%s'.", db_path)


# ── experiments ───────────────────────────────────────────────────────────────

def insert_experiment(
    config: ExperimentConfig,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Insert a new experiment row and return its assigned ID.

    Parameters
    ----------
    config : ExperimentConfig
    db_path : str

    Returns
    -------
    int
        The auto-assigned primary key.
    """
    initialise_db(db_path)
    sql = """
        INSERT INTO experiments (name, description, strategy_type,
                                 payoff_mode, config_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with _connect(db_path) as conn:
        cur = conn.execute(
            sql,
            (
                config.name,
                config.description,
                config.strategy_type,
                config.payoff_mode,
                config.config_json_str(),
                datetime.utcnow().isoformat(),
            ),
        )
        experiment_id = cur.lastrowid
    logger.info("Inserted experiment id=%d name='%s'.", experiment_id, config.name)
    return experiment_id


def get_experiment(experiment_id: int, db_path: str = DEFAULT_DB_PATH) -> ExperimentConfig:
    """Retrieve an experiment by its primary key.

    Raises
    ------
    KeyError
        If no experiment with that ID exists.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"No experiment with id={experiment_id}.")
    return ExperimentConfig.from_row(dict(row))


def list_experiments(db_path: str = DEFAULT_DB_PATH) -> List[ExperimentConfig]:
    """Return all experiments ordered by creation time (newest first)."""
    initialise_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC"
        ).fetchall()
    return [ExperimentConfig.from_row(dict(r)) for r in rows]


def delete_experiment(experiment_id: int, db_path: str = DEFAULT_DB_PATH) -> int:
    """Delete an experiment and all its game results (cascade).

    Returns the number of deleted game_results rows.
    """
    with _connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM game_results WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()[0]
        conn.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
    logger.info("Deleted experiment id=%d (%d results).", experiment_id, count)
    return count


# ── game results ──────────────────────────────────────────────────────────────

def insert_game_runs(
    runs: List[GameRun],
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Batch-insert game run records (single transaction for speed).

    Parameters
    ----------
    runs : List[GameRun]
    db_path : str
    """
    if not runs:
        return
    sql = """
        INSERT INTO game_results
            (experiment_id, player_a_prob, player_b_prob,
             outcome, player_a_score, player_b_score, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    ts = datetime.utcnow().isoformat()
    data = [
        (
            r.experiment_id,
            r.player_a_prob,
            r.player_b_prob,
            r.outcome,
            r.player_a_score,
            r.player_b_score,
            ts,
        )
        for r in runs
    ]
    with _connect(db_path) as conn:
        conn.executemany(sql, data)
    logger.debug("Inserted %d game_results for experiment_id=%d.", len(runs), runs[0].experiment_id)


def fetch_runs(
    experiment_id: int,
    db_path: str = DEFAULT_DB_PATH,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch game_results for a given experiment as a list of dicts.

    Parameters
    ----------
    experiment_id : int
    db_path : str
    limit : int, optional
        If given, return at most *limit* rows.

    Returns
    -------
    List[Dict[str, Any]]
    """
    sql = "SELECT * FROM game_results WHERE experiment_id = ? ORDER BY id"
    params: tuple = (experiment_id,)
    if limit is not None:
        sql += " LIMIT ?"
        params = (experiment_id, limit)
    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def fetch_runs_by_prob(
    experiment_id: int,
    player_a_prob: float,
    player_b_prob: float,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Fetch rows matching exact probability values."""
    sql = """
        SELECT * FROM game_results
        WHERE experiment_id=? AND player_a_prob=? AND player_b_prob=?
        ORDER BY id
    """
    with _connect(db_path) as conn:
        rows = conn.execute(sql, (experiment_id, player_a_prob, player_b_prob)).fetchall()
    return [dict(r) for r in rows]


def count_runs(experiment_id: int, db_path: str = DEFAULT_DB_PATH) -> int:
    """Return the number of game_results for an experiment."""
    with _connect(db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM game_results WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()[0]


# ── export ────────────────────────────────────────────────────────────────────

def export_csv(
    experiment_id: int,
    output_path: str,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Export all game_results for an experiment to a CSV file.

    Parameters
    ----------
    experiment_id : int
    output_path : str
        Destination file path (created or overwritten).
    db_path : str

    Returns
    -------
    int
        Number of rows exported.
    """
    import csv

    rows = fetch_runs(experiment_id, db_path)
    if not rows:
        logger.warning("No data found for experiment_id=%d.", experiment_id)
        return 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d rows to '%s'.", len(rows), output_path)
    return len(rows)
