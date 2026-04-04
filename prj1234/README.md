# Quantum Prisoner's Dilemma — Research Framework

A modular, research-grade simulation and analysis tool for studying the
Prisoner's Dilemma using quantum-inspired strategies.

---

## Directory Structure

```
quantum_pd/
├── quantum/
│   ├── quantum_core.py     # Circuits, statevector, sampling, angle encoding
│   └── strategies.py       # Classical & quantum strategy abstractions
├── game/
│   ├── payoff.py           # Payoff matrix models (standard PD, variants)
│   └── engine.py           # Game engine: sampling → payoffs → RoundResult
├── storage/
│   ├── database.py         # SQLite CRUD, migrations, batch inserts, CSV export
│   └── models.py           # GameRun, ExperimentConfig, ExperimentStats
├── simulation/
│   ├── config.py           # SingleRunConfig, GridRunConfig, JSON/YAML loaders
│   └── runner.py           # ExperimentRunner (single & grid)
├── analysis/
│   ├── summary.py          # Descriptive stats, frequency counts, heatmap data
│   ├── plots.py            # Matplotlib charts (bar, line, scatter, heatmap)
│   └── reports.py          # Markdown report generation
├── cli/
│   └── main.py             # Full CLI entry point (argparse)
├── tests/
│   └── test_suite.py       # Unit + integration tests
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## CLI Quick Reference

All commands are run as:
```bash
python -m cli.main [--db PATH] <command> ...
```

### Run a single experiment
```bash
python -m cli.main simulate single \
  --player-a 0.7 --player-b 0.5 \
  --iterations 2000 \
  --strategy-type quantum_ry \
  --payoff-mode standard_pd \
  --encoding ry \
  --entanglement none \
  --name "baseline_experiment" \
  --description "Quantum Ry encoding, no entanglement"
```

### Run a 2-D grid sweep
```bash
python -m cli.main simulate grid \
  --pa-start 0.0 --pa-end 1.0 --pa-step 0.1 \
  --pb-start 0.0 --pb-end 1.0 --pb-step 0.1 \
  --iterations-per-cell 300 \
  --strategy-type quantum_ry \
  --name "full_grid_sweep"
```

### Run with CNOT entanglement
```bash
python -m cli.main simulate single \
  --player-a 0.6 --player-b 0.6 \
  --iterations 5000 \
  --entanglement cnot \
  --name "cnot_entangled"
```

### List all stored experiments
```bash
python -m cli.main list
```

### Print summary statistics
```bash
python -m cli.main analyze summary --experiment-id 1
```

### Generate plots
```bash
python -m cli.main analyze plots \
  --experiment-id 1 \
  --save-dir plots/ \
  --grid            # add heatmaps for grid experiments
```

### Generate a Markdown report
```bash
python -m cli.main report generate \
  --experiment-id 1 \
  --output reports/experiment_1.md \
  --include-plots \
  --plots-dir plots/
```

### Export results to CSV
```bash
python -m cli.main export csv \
  --experiment-id 1 \
  --output data/experiment_1.csv
```

### Run from a config file
```bash
python -m cli.main simulate from-file config/my_experiment.json
```

---

## Config File Format (JSON)

```json
{
  "kind": "single",
  "name": "cnot_sweep",
  "description": "Testing CNOT entanglement effect",
  "player_a_prob": 0.7,
  "player_b_prob": 0.5,
  "iterations": 3000,
  "strategy_type": "quantum_ry",
  "payoff_mode": "standard_pd",
  "encoding": "ry",
  "entanglement": "cnot",
  "seed": 42
}
```

For a grid experiment, use `"kind": "grid"` and replace `player_a_prob`/`player_b_prob`/`iterations`
with `pa_values`, `pb_values`, and `iterations_per_cell`.

---

## Programmatic API

### Extending a strategy

```python
from quantum.strategies import BaseStrategy

class GrimTriggerStrategy(BaseStrategy):
    def __init__(self) -> None:
        self._betrayed = False

    def report_betrayal(self) -> None:
        self._betrayed = True

    def get_cooperation_prob(self) -> float:
        return 0.0 if self._betrayed else 1.0
```

### Custom payoff model

```python
from game.payoff import payoff_model_from_dict

my_game = payoff_model_from_dict(
    name="asymmetric_pd",
    matrix_dict={"CC": (4, 4), "CD": (0, 7), "DC": (7, 0), "DD": (1, 1)},
    description="Asymmetric temptation game",
)
```

### Programmatic experiment

```python
from simulation.runner import ExperimentRunner
from simulation.config import GridRunConfig, make_grid_from_range
from quantum.quantum_core import EncodingScheme, EntanglementMode

config = make_grid_from_range(
    0.0, 1.0, 0.2, 0.0, 1.0, 0.2,
    iterations_per_cell=500,
    entanglement=EntanglementMode.CNOT,
    name="cnot_grid",
)
runner = ExperimentRunner(db_path="results.db")
exp_id = runner.run_grid(config)

# Export for external analysis
runner.export_results_csv(exp_id, "data/cnot_grid.csv")
```

### Analysis in Python

```python
from analysis.summary import compute_experiment_stats
from analysis.plots import generate_all_plots
from analysis.reports import generate_report

stats = compute_experiment_stats(exp_id)
print(f"Avg score A: {stats.avg_score_a:.3f}")
print(f"Cooperation rate A: {stats.cooperation_rate_a:.3f}")

paths = generate_all_plots(exp_id, save_dir="plots/", is_grid=True)
generate_report(exp_id, "reports/exp.md", plot_paths=paths)
```

---

## Running Tests

```bash
pytest tests/test_suite.py -v
```

---

## Strategy Types

| ID            | Description                              |
|---------------|------------------------------------------|
| `quantum_ry`  | Ry-rotation quantum encoding             |
| `quantum_rx`  | Rx-rotation quantum encoding             |
| `quantum_h_ry`| Hadamard + Ry (uniform superposition)    |
| `classical`   | Pure classical fixed probability         |
| `always_c`    | Always cooperate                         |
| `always_d`    | Always defect                            |

## Entanglement Modes

| ID          | Description                                           |
|-------------|-------------------------------------------------------|
| `none`      | Independent qubits (default)                          |
| `cnot`      | CNOT gate applied after encoding                      |
| `cz`        | CZ gate applied after encoding                        |
| `bell_init` | Start in Bell state |Φ+⟩, then apply rotations       |

## Named Games

| ID            | (C,C) | (C,D) | (D,C) | (D,D) |
|---------------|-------|-------|-------|-------|
| `standard_pd` | 3,3   | 0,5   | 5,0   | 1,1   |
| `harsh_pd`    | 3,3   | 0,5   | 5,0   | 0,0   |
| `lenient_pd`  | 5,5   | 0,7   | 7,0   | 1,1   |
| `stag_hunt`   | 4,4   | 0,3   | 3,0   | 3,3   |
| `chicken`     | 3,3   | 2,4   | 4,2   | 0,0   |
