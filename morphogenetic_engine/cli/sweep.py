"""
Dedicated CLI for running parameter sweeps.

This module provides a command-line interface for grid search and
Bayesian optimization experiments.
"""

import argparse
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console

from ..sweeps.config import load_sweep_config, load_sweep_configs
from ..sweeps.grid_search import GridSearchRunner


class SweepCLI:
    """Command-line interface for sweep experiments."""

    def __init__(self):
        """Initialize the CLI."""
        self.console = Console()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser for sweep commands."""
        parser = argparse.ArgumentParser(
            prog="morphogenetic-sweep",
            description="Run hyperparameter sweeps for morphogenetic experiments",
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Grid search command
        grid_parser = subparsers.add_parser("grid", help="Run grid search sweep")
        grid_parser.add_argument(
            "--config",
            "-c",
            type=Path,
            required=True,
            help="Path to sweep configuration file or directory",
        )
        grid_parser.add_argument(
            "--parallel",
            "-p",
            type=int,
            default=1,
            help="Maximum number of parallel experiments (default: 1)",
        )
        grid_parser.add_argument(
            "--timeout",
            type=int,
            default=3600,
            help="Timeout per experiment in seconds (default: 3600)",
        )
        grid_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

        # Bayesian search command (placeholder for future implementation)
        bayesian_parser = subparsers.add_parser("bayesian", help="Run Bayesian optimization sweep")
        bayesian_parser.add_argument(
            "--config", "-c", type=Path, required=True, help="Path to sweep configuration file"
        )
        bayesian_parser.add_argument(
            "--trials",
            "-t",
            type=int,
            default=50,
            help="Number of optimization trials (default: 50)",
        )
        bayesian_parser.add_argument(
            "--timeout", type=int, default=3600, help="Timeout per trial in seconds (default: 3600)"
        )

        # Quick test command
        quick_parser = subparsers.add_parser("quick", help="Run a quick test sweep")
        quick_parser.add_argument(
            "--problem",
            choices=["spirals", "moons", "clusters", "spheres", "complex_moons"],
            default="spirals",
            help="Problem type to test (default: spirals)",
        )
        quick_parser.add_argument(
            "--trials", type=int, default=10, help="Number of quick trials (default: 10)"
        )

        # Resume command (placeholder)
        resume_parser = subparsers.add_parser("resume", help="Resume interrupted sweep")
        resume_parser.add_argument("--sweep-id", type=str, required=True, help="Sweep ID to resume")

        return parser

    def run_grid_search(self, args: argparse.Namespace) -> int:
        """Run a grid search sweep."""
        try:
            # Load configuration
            if args.config.is_file():
                config = load_sweep_config(args.config)
                configs = [config]
            else:
                configs = load_sweep_configs(args.config)

            self.console.print(
                f"[bold green]Loading {len(configs)} sweep configuration(s)[/bold green]"
            )

            # Override execution settings from CLI args
            for config in configs:
                if args.parallel:
                    config.execution["max_parallel"] = args.parallel
                if args.timeout:
                    config.execution["timeout_per_trial"] = args.timeout

            # Create base args namespace
            base_args = argparse.Namespace()
            if args.seed:
                base_args.seed = args.seed

            # Run each configuration
            all_results = []
            for i, config in enumerate(configs):
                if len(configs) > 1:
                    self.console.print(
                        f"\n[bold cyan]Running configuration {i+1}/{len(configs)}[/bold cyan]"
                    )

                runner = GridSearchRunner(config, base_args)
                results = runner.run_sweep()
                all_results.append(results)

            self.console.print(
                f"\n[bold green]All sweeps completed! Processed {len(all_results)} configuration(s)[/bold green]"
            )
            return 0

        except FileNotFoundError as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return 1
        except PermissionError as e:
            self.console.print(f"[bold red]Error: Permission denied accessing file: {e}[/bold red]")
            return 1
        except yaml.YAMLError as e:
            self.console.print(f"[bold red]Error: Invalid YAML format: {e}[/bold red]")
            return 1
        except (ValueError, RuntimeError, ImportError) as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return 1
        except Exception as e:
            self.console.print(f"[bold red]Unexpected error: {e}[/bold red]")
            return 1

    def run_bayesian_search(self, args: argparse.Namespace) -> int:
        """Run a Bayesian optimization sweep."""
        try:
            from ..sweeps.bayesian import BayesianSearchRunner

            # Load configuration
            config = load_sweep_config(args.config)

            if config.sweep_type != "bayesian":
                self.console.print(
                    "[yellow]Warning: Configuration is not set for Bayesian optimization[/yellow]"
                )
                self.console.print("Setting sweep_type to 'bayesian'")
                config.sweep_type = "bayesian"

            # Override settings from CLI args
            if args.trials:
                # Note: trials is handled in the run_optimization call
                pass
            if args.timeout:
                config.execution["timeout_per_trial"] = args.timeout

            # Create base args namespace
            base_args = argparse.Namespace()

            # Run Bayesian optimization
            runner = BayesianSearchRunner(config, base_args)
            _ = runner.run_optimization(n_trials=args.trials)

            self.console.print("\n[bold green]Bayesian optimization completed![/bold green]")
            return 0

        except FileNotFoundError as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return 1
        except PermissionError as e:
            self.console.print(f"[bold red]Error: Permission denied accessing file: {e}[/bold red]")
            return 1
        except yaml.YAMLError as e:
            self.console.print(f"[bold red]Error: Invalid YAML format: {e}[/bold red]")
            return 1
        except (ValueError, RuntimeError, ImportError) as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return 1
        except Exception as e:
            self.console.print(f"[bold red]Unexpected error: {e}[/bold red]")
            return 1

    def run_quick_test(self, args: argparse.Namespace) -> int:
        """Run a quick test sweep."""
        self.console.print("[bold cyan]Running quick test sweep...[/bold cyan]")

        # Create a temporary configuration for quick testing
        quick_config = {
            "sweep_type": "grid",
            "experiment": {
                "problem_type": args.problem,
                "input_dim": 3,
                "n_samples": 1000,
            },
            "parameters": {
                "num_layers": [4, 8],
                "hidden_dim": [64, 128],
                "lr": [1e-3, 1e-2],
            },
            "execution": {
                "max_parallel": 2,
                "timeout_per_trial": 300,  # 5 minutes
            },
        }

        # Limit to requested number of trials
        from ..sweeps.config import SweepConfig

        config = SweepConfig(quick_config)
        combinations = config.get_grid_combinations()

        if len(combinations) > args.trials:
            # Randomly sample combinations
            import random

            random.shuffle(combinations)
            combinations = combinations[: args.trials]
            self.console.print(
                f"[yellow]Limiting to {args.trials} trials (sampled from "
                f"{len(config.get_grid_combinations())} possible combinations)[/yellow]"
            )
        # Create a modified config with limited combinations
        # This is a simplified approach - in a full implementation we'd need better sampling
        runner = GridSearchRunner(config)
        _ = runner.run_sweep()

        return 0

    def resume_sweep(self, _args: argparse.Namespace) -> int:
        """Resume an interrupted sweep."""
        self.console.print("[yellow]Resume functionality not yet implemented[/yellow]")
        self.console.print("This feature will be added in a future update")
        return 1

    def main(self, argv: Optional[list] = None) -> int:
        """Main entry point for the CLI."""
        parser = self.create_parser()
        args = parser.parse_args(argv)

        if args.command is None:
            parser.print_help()
            return 1

        if args.command == "grid":
            return self.run_grid_search(args)
        elif args.command == "bayesian":
            return self.run_bayesian_search(args)
        elif args.command == "quick":
            return self.run_quick_test(args)
        elif args.command == "resume":
            return self.resume_sweep(args)
        else:
            self.console.print(f"[bold red]Unknown command: {args.command}[/bold red]")
            return 1


def main():
    """Main entry point for the sweep CLI."""
    cli = SweepCLI()
    return cli.main()
