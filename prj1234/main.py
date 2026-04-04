"""
cli/main.py
===========
Professional command-line interface for the Quantum Prisoner's Dilemma framework.

Usage examples
--------------
Single experiment:
    python -m cli.main simulate single \\
        --player-a 0.7 --player-b 0.5 --iterations 2000 \\
        --strategy-type quantum_ry --payoff-mode standard_pd \\
        --name "my_exp" --description "Test run"

Grid sweep:
    python -m cli.main simulate grid \\
        --pa-start 0.0 --pa-end 1.0 --pa-step 0.1 \\
        --pb-start 0.0 --pb-end 1.0 --pb-step 0.1 \\
        --iterations-per-cell 300 --name "full_grid"

Analysis:
    python -m cli.main analyze summary --experiment-id 1
    python -m cli.main analyze plots   --experiment-id 1 --save-dir plots/
    python -m cli.main report generate --experiment-id 1 --output reports/exp1.md

Export CSV:
    python -m cli.main export csv --experiment-id 1 --output data/exp1.csv

List experiments:
    python -m cli.main list
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qpd.cli")


# ── lazy imports (keep startup fast) ─────────────────────────────────────────

def _runner():
    from simulation.runner import ExperimentRunner
    return ExperimentRunner


def _single_cfg():
    from simulation.config import SingleRunConfig
    return SingleRunConfig


def _grid_cfg():
    from simulation.config import GridRunConfig, make_grid_from_range
    return GridRunConfig, make_grid_from_range


def _enc_ent():
    from quantum.quantum_core import EncodingScheme, EntanglementMode
    return EncodingScheme, EntanglementMode


# ── argument helpers ──────────────────────────────────────────────────────────

def _prob(value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected a number, got '{value}'.")
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"Probability must be in [0, 1], got {v}.")
    return v


def _positive_int(value: str) -> int:
    try:
        v = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected an integer, got '{value}'.")
    if v <= 0:
        raise argparse.ArgumentTypeError(f"Value must be positive, got {v}.")
    return v


# ── CLI builder ───────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qpd",
        description="Quantum Prisoner's Dilemma — Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db", default="quantum_game.db", metavar="PATH",
        help="SQLite database file (default: quantum_game.db).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── list ──────────────────────────────────────────────────────────────────
    sub.add_parser("list", help="List all stored experiments.")

    # ── simulate ──────────────────────────────────────────────────────────────
    sim = sub.add_parser("simulate", help="Run simulation experiments.")
    sim_sub = sim.add_subparsers(dest="sim_command", required=True)

    # simulate single
    s = sim_sub.add_parser("single", help="Run a single fixed-probability experiment.")
    s.add_argument("--player-a", type=_prob, required=True, metavar="FLOAT",
                   help="Player A cooperation probability [0,1].")
    s.add_argument("--player-b", type=_prob, required=True, metavar="FLOAT",
                   help="Player B cooperation probability [0,1].")
    s.add_argument("--iterations", type=_positive_int, default=1000,
                   help="Number of game rounds (default: 1000).")
    _add_common_sim_args(s)

    # simulate grid
    g = sim_sub.add_parser("grid", help="Run a 2-D probability sweep experiment.")
    g.add_argument("--pa-start", type=_prob, default=0.0)
    g.add_argument("--pa-end",   type=_prob, default=1.0)
    g.add_argument("--pa-step",  type=float, default=0.1)
    g.add_argument("--pb-start", type=_prob, default=0.0)
    g.add_argument("--pb-end",   type=_prob, default=1.0)
    g.add_argument("--pb-step",  type=float, default=0.1)
    g.add_argument("--iterations-per-cell", type=_positive_int, default=300,
                   help="Rounds per grid cell (default: 300).")
    _add_common_sim_args(g)

    # simulate from-file
    f = sim_sub.add_parser("from-file", help="Run experiment from JSON/YAML config file.")
    f.add_argument("config_file", help="Path to JSON or YAML config file.")

    # ── analyze ───────────────────────────────────────────────────────────────
    ana = sub.add_parser("analyze", help="Analyse experiment results.")
    ana_sub = ana.add_subparsers(dest="ana_command", required=True)

    smry = ana_sub.add_parser("summary", help="Print summary statistics.")
    smry.add_argument("--experiment-id", type=int, required=True)

    plts = ana_sub.add_parser("plots", help="Generate and save plots.")
    plts.add_argument("--experiment-id", type=int, required=True)
    plts.add_argument("--save-dir", default="plots")
    plts.add_argument("--show", action="store_true", help="Display plots interactively.")
    plts.add_argument("--grid", action="store_true", help="Include heatmap plots (grid experiments).")

    # ── report ────────────────────────────────────────────────────────────────
    rep = sub.add_parser("report", help="Generate reports.")
    rep_sub = rep.add_subparsers(dest="rep_command", required=True)

    gen = rep_sub.add_parser("generate", help="Generate a Markdown report.")
    gen.add_argument("--experiment-id", type=int, required=True)
    gen.add_argument("--output", default=None, metavar="PATH",
                     help="Output Markdown file (default: reports/experiment_<id>.md).")
    gen.add_argument("--include-plots", action="store_true",
                     help="Also generate plots and embed references.")
    gen.add_argument("--plots-dir", default="plots")
    gen.add_argument("--grid", action="store_true",
                     help="Include heatmaps (for grid experiments).")

    # ── export ────────────────────────────────────────────────────────────────
    exp = sub.add_parser("export", help="Export data.")
    exp_sub = exp.add_subparsers(dest="exp_command", required=True)

    csv_p = exp_sub.add_parser("csv", help="Export results to CSV.")
    csv_p.add_argument("--experiment-id", type=int, required=True)
    csv_p.add_argument("--output", required=True, metavar="PATH")

    return parser


def _add_common_sim_args(p: argparse.ArgumentParser) -> None:
    """Attach strategy/payoff/meta arguments shared by all simulate subcommands."""
    p.add_argument(
        "--strategy-type", default="quantum_ry",
        choices=["quantum_ry", "quantum_rx", "quantum_h_ry", "classical",
                 "always_c", "always_d"],
        help="Strategy encoding type (default: quantum_ry).",
    )
    p.add_argument(
        "--payoff-mode", default="standard_pd",
        choices=["standard_pd", "harsh_pd", "lenient_pd", "stag_hunt", "chicken"],
        help="Payoff matrix variant (default: standard_pd).",
    )
    p.add_argument(
        "--encoding", default="ry", choices=["ry", "rx", "h_ry"],
        help="Quantum angle encoding (default: ry).",
    )
    p.add_argument(
        "--entanglement", default="none",
        choices=["none", "cnot", "cz", "bell_init"],
        help="Entanglement mode (default: none).",
    )
    p.add_argument("--name", default="experiment", help="Short experiment name.")
    p.add_argument("--description", default="", help="Longer description.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")


# ── command handlers ──────────────────────────────────────────────────────────

def cmd_list(args: argparse.Namespace) -> int:
    from storage.database import list_experiments
    experiments = list_experiments(args.db)
    if not experiments:
        print("No experiments found.")
        return 0
    print(f"\n{'ID':>4}  {'Name':<30}  {'Strategy':<15}  {'Payoff':<15}  Created")
    print("-" * 90)
    for exp in experiments:
        d = exp.to_summary_dict()
        print(
            f"{d['id']:>4}  {d['name']:<30}  {d['strategy_type']:<15}  "
            f"{d['payoff_mode']:<15}  {d['created_at']}"
        )
    print()
    return 0


def cmd_simulate_single(args: argparse.Namespace) -> int:
    SingleRunConfig = _single_cfg()
    EncodingScheme, EntanglementMode = _enc_ent()

    config = SingleRunConfig(
        player_a_prob=args.player_a,
        player_b_prob=args.player_b,
        iterations=args.iterations,
        strategy_type=args.strategy_type,
        payoff_mode=args.payoff_mode,
        encoding=EncodingScheme(args.encoding),
        entanglement=EntanglementMode(args.entanglement),
        name=args.name,
        description=args.description,
        seed=args.seed,
    )
    runner = _runner()(db_path=args.db)
    exp_id = runner.run_single(config)
    print(f"\n✓ Experiment #{exp_id} complete.")
    print(f"  Rounds    : {config.iterations:,}")
    print(f"  Player A  : p={config.player_a_prob:.3f}  strategy={config.strategy_type}")
    print(f"  Player B  : p={config.player_b_prob:.3f}  strategy={config.strategy_type}")
    print(f"  Payoff    : {config.payoff_mode}")
    print(f"  Database  : {args.db}\n")
    return 0


def cmd_simulate_grid(args: argparse.Namespace) -> int:
    GridRunConfig, make_grid_from_range = _grid_cfg()
    EncodingScheme, EntanglementMode = _enc_ent()

    config = make_grid_from_range(
        pa_start=args.pa_start, pa_end=args.pa_end, pa_step=args.pa_step,
        pb_start=args.pb_start, pb_end=args.pb_end, pb_step=args.pb_step,
        iterations_per_cell=args.iterations_per_cell,
        strategy_type=args.strategy_type,
        payoff_mode=args.payoff_mode,
        encoding=EncodingScheme(args.encoding),
        entanglement=EntanglementMode(args.entanglement),
        name=args.name,
        description=args.description,
        seed=args.seed,
    )
    runner = _runner()(db_path=args.db)
    exp_id = runner.run_grid(config)
    print(f"\n✓ Grid experiment #{exp_id} complete.")
    print(f"  Grid cells: {len(config.pa_values)} × {len(config.pb_values)}")
    print(f"  Total runs: {config.total_runs:,}")
    print(f"  Database  : {args.db}\n")
    return 0


def cmd_simulate_from_file(args: argparse.Namespace) -> int:
    path = args.config_file
    if path.endswith(".yaml") or path.endswith(".yml"):
        from simulation.config import load_config_from_yaml
        config = load_config_from_yaml(path)
    else:
        from simulation.config import load_config_from_json
        config = load_config_from_json(path)

    from simulation.config import SingleRunConfig, GridRunConfig
    runner = _runner()(db_path=args.db)

    if isinstance(config, GridRunConfig):
        exp_id = runner.run_grid(config)
    else:
        exp_id = runner.run_single(config)

    print(f"\n✓ Experiment #{exp_id} complete (from file: {path}).\n")
    return 0


def cmd_analyze_summary(args: argparse.Namespace) -> int:
    from analysis.summary import compute_experiment_stats, print_summary
    from storage.database import get_experiment
    exp = get_experiment(args.experiment_id, args.db)
    stats = compute_experiment_stats(args.experiment_id, args.db)
    print_summary(stats, exp)
    return 0


def cmd_analyze_plots(args: argparse.Namespace) -> int:
    from analysis.plots import generate_all_plots
    paths = generate_all_plots(
        args.experiment_id,
        save_dir=args.save_dir,
        show=args.show,
        db_path=args.db,
        is_grid=args.grid,
    )
    print(f"\n✓ Plots saved for experiment #{args.experiment_id}:")
    for name, path in paths.items():
        print(f"  {name:<20} → {path}")
    print()
    return 0


def cmd_report_generate(args: argparse.Namespace) -> int:
    from analysis.reports import generate_report
    plot_paths = {}
    if args.include_plots:
        from analysis.plots import generate_all_plots
        plot_paths = generate_all_plots(
            args.experiment_id,
            save_dir=args.plots_dir,
            show=False,
            db_path=args.db,
            is_grid=args.grid,
        )
    output = args.output or f"reports/experiment_{args.experiment_id}.md"
    path = generate_report(
        args.experiment_id,
        output_path=output,
        plot_paths=plot_paths,
        db_path=args.db,
    )
    print(f"\n✓ Report written to: {path}\n")
    return 0


def cmd_export_csv(args: argparse.Namespace) -> int:
    from storage.database import export_csv
    n = export_csv(args.experiment_id, args.output, args.db)
    print(f"\n✓ Exported {n:,} rows to: {args.output}\n")
    return 0


# ── dispatch ──────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        if args.command == "list":
            return cmd_list(args)

        elif args.command == "simulate":
            if args.sim_command == "single":
                return cmd_simulate_single(args)
            elif args.sim_command == "grid":
                return cmd_simulate_grid(args)
            elif args.sim_command == "from-file":
                return cmd_simulate_from_file(args)

        elif args.command == "analyze":
            if args.ana_command == "summary":
                return cmd_analyze_summary(args)
            elif args.ana_command == "plots":
                return cmd_analyze_plots(args)

        elif args.command == "report":
            if args.rep_command == "generate":
                return cmd_report_generate(args)

        elif args.command == "export":
            if args.exp_command == "csv":
                return cmd_export_csv(args)

        parser.print_help()
        return 1

    except KeyError as exc:
        print(f"\n✗ Not found: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"\n✗ Validation error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"\n✗ File error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        logger.exception("Unexpected error")
        print(f"\n✗ Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
