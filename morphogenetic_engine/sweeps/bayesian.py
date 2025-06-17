"""
Bayesian optimization using Optuna for morphogenetic experiments.

This module provides intelligent parameter search using Optuna's
Tree-structured Parzen Estimator (TPE) and other optimization algorithms.
"""

import argparse
import hashlib
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
from rich.console import Console
from rich.progress import Progress

from .config import SweepConfig
from .results import SweepResults


class BayesianSearchRunner:
    """Bayesian optimization runner using Optuna."""

    def __init__(self, config: SweepConfig, base_args: Optional[argparse.Namespace] = None):
        """Initialize the Bayesian search runner."""
        self.config = config
        self.base_args = base_args or argparse.Namespace()
        self.console = Console()
        self.project_root = Path(__file__).parent.parent.parent

        # Create results directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sweep_dir = self.project_root / "results" / "sweeps" / f"bayesian_{self.timestamp}"
        self.sweep_dir.mkdir(parents=True, exist_ok=True)

        self.results = SweepResults(self.sweep_dir)

        # Initialize Optuna study
        self.study_name = f"morphogenetic_{self.timestamp}"
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.config.direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner(),
        )

        self.trial_counter = 0

    def _create_sampler(self):
        """Create the Optuna sampler based on configuration."""
        sampler_name = self.config.optimization.get("sampler", "TPE")

        if sampler_name == "TPE":
            return optuna.samplers.TPESampler()
        if sampler_name == "CmaEs":
            return optuna.samplers.CmaEsSampler()
        if sampler_name == "Random":
            return optuna.samplers.RandomSampler()
        self.console.print(f"[yellow]Unknown sampler {sampler_name}, using TPE[/yellow]")
        return optuna.samplers.TPESampler()

    def _create_pruner(self):
        """Create the Optuna pruner based on configuration."""
        pruner_name = self.config.optimization.get("pruner", None)

        if pruner_name == "MedianPruner":
            return optuna.pruners.MedianPruner()
        elif pruner_name == "PercentilePruner":
            return optuna.pruners.PercentilePruner(25.0)
        elif pruner_name is None:
            return optuna.pruners.NopPruner()  # No pruning
        else:
            self.console.print(f"[yellow]Unknown pruner {pruner_name}, using no pruning[/yellow]")
            return optuna.pruners.NopPruner()

    def _suggest_parameters(self, trial) -> Dict[str, Any]:
        """Suggest parameters for a trial based on the search space."""
        params = {}
        search_space = self.config.get_bayesian_search_space()

        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                # Structured parameter definition
                param_type = param_config.get("type", "categorical")

                if param_type == "categorical":
                    choices = param_config.get("choices", [])
                    params[param_name] = trial.suggest_categorical(param_name, choices)

                elif param_type == "int":
                    low = param_config.get("low", 1)
                    high = param_config.get("high", 100)
                    log = param_config.get("log", False)
                    if log:
                        params[param_name] = trial.suggest_int(param_name, low, high, log=True)
                    else:
                        params[param_name] = trial.suggest_int(param_name, low, high)

                elif param_type == "float":
                    low = param_config.get("low", 0.0)
                    high = param_config.get("high", 1.0)
                    log = param_config.get("log", False)
                    if log:
                        params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, low, high)

                else:
                    # Fallback to categorical
                    self.console.print(
                        f"[yellow]Unknown parameter type {param_type} for {param_name}, treating as categorical[/yellow]"
                    )
                    choices = param_config.get("choices", [param_config])
                    params[param_name] = trial.suggest_categorical(param_name, choices)

            elif isinstance(param_config, list):
                # Simple categorical list
                params[param_name] = trial.suggest_categorical(param_name, param_config)

            else:
                # Single value - treat as fixed
                params[param_name] = param_config

        # Add experiment fixed parameters
        for key, value in self.config.experiment.items():
            if key not in params:
                params[key] = value

        return params

    def _objective(self, trial) -> float:
        """Objective function for a single trial."""
        # Get parameters for this trial
        params = self._suggest_parameters(trial)

        # Generate run slug
        run_slug = self._generate_run_slug(params, self.trial_counter)
        self.trial_counter += 1

        try:
            # Build and run command
            cmd = self._build_command(params, run_slug)

            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_per_trial,
                cwd=self.project_root,
                check=False,
            )
            end_time = time.time()

            # Process results
            experiment_result = {
                "trial_number": trial.number,
                "run_slug": run_slug,
                "parameters": params,
                "runtime": end_time - start_time,
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            if result.returncode == 0:
                # Extract metrics
                metrics = self._extract_metrics(result.stdout)
                experiment_result.update(metrics)

                # Get target metric
                target_metric = self.config.target_metric
                if target_metric in metrics:
                    metric_value = metrics[target_metric]

                    # Store result
                    self.results.add_result(experiment_result)

                    return float(metric_value)
                else:
                    self.console.print(
                        f"[yellow]Target metric '{target_metric}' not found in results[/yellow]"
                    )
                    experiment_result["error"] = f"Target metric '{target_metric}' not found"
                    self.results.add_result(experiment_result)
                    raise optuna.TrialPruned()

            else:
                experiment_result["error"] = f"Process failed with code {result.returncode}"
                self.results.add_result(experiment_result)
                raise optuna.TrialPruned()

        except subprocess.TimeoutExpired as e:
            experiment_result = {
                "trial_number": trial.number,
                "run_slug": run_slug,
                "parameters": params,
                "success": False,
                "error": f"Timeout after {self.config.timeout_per_trial} seconds",
            }
            self.results.add_result(experiment_result)
            raise optuna.TrialPruned() from e

        except Exception as e:
            experiment_result = {
                "trial_number": trial.number,
                "run_slug": run_slug,
                "parameters": params,
                "success": False,
                "error": str(e),
            }
            self.results.add_result(experiment_result)
            raise optuna.TrialPruned() from e

    def run_optimization(self, n_trials: int = 50, timeout: Optional[int] = None) -> SweepResults:
        """Run the Bayesian optimization."""
        self.console.print(
            f"[bold green]Starting Bayesian optimization with {n_trials} trials[/bold green]"
        )
        self.console.print(f"Results will be saved to: {self.sweep_dir}")

        with Progress() as progress:
            task = progress.add_task("[green]Running optimization...", total=n_trials)

            def callback(_study, trial):
                progress.update(task, advance=1)

                # Print trial results
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    self.console.print(
                        f"Trial {trial.number}: {self.config.target_metric} = {trial.value:.4f}"
                    )
                elif trial.state == optuna.trial.TrialState.PRUNED:
                    self.console.print(f"Trial {trial.number}: Pruned")
                elif trial.state == optuna.trial.TrialState.FAIL:
                    self.console.print(f"Trial {trial.number}: Failed")

            # Run optimization
            self.study.optimize(
                self._objective, n_trials=n_trials, timeout=timeout, callbacks=[callback]
            )

        # Finalize results
        self.results.finalize()
        self._save_study_results()
        self._print_summary()

        return self.results

    def _build_command(self, params: Dict[str, Any], _run_slug: str) -> List[str]:
        """Build the command to run a single experiment."""
        cmd = [sys.executable, "scripts/run_morphogenetic_experiment.py"]

        # Add parameters
        for key, value in params.items():
            cmd.extend([f"--{key}", str(value)])

        # Add base arguments
        if hasattr(self.base_args, "seed") and self.base_args.seed is not None:
            cmd.extend(["--seed", str(self.base_args.seed)])

        return cmd

    def _generate_run_slug(self, params: Dict[str, Any], trial_number: int) -> str:
        """Generate a unique slug for a trial."""
        # Create a short hash of the parameters
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        return f"trial_{trial_number:03d}_{param_hash}"

    def _extract_metrics(self, stdout: str) -> Dict[str, Any]:
        """Extract metrics from experiment output."""
        metrics = {}

        lines = stdout.split("\n")
        for line in lines:
            line = line.strip()

            # Look for common metric patterns
            if "Final accuracy:" in line:
                try:
                    acc_str = line.split("Final accuracy:")[1].strip()
                    metrics["final_acc"] = float(acc_str)
                except (IndexError, ValueError):
                    pass

            if "Best accuracy:" in line:
                try:
                    acc_str = line.split("Best accuracy:")[1].strip()
                    metrics["best_acc"] = float(acc_str)
                except (IndexError, ValueError):
                    pass

            if "val_acc:" in line:
                try:
                    acc_str = line.split("val_acc:")[1].strip().split()[0]
                    metrics["val_acc"] = float(acc_str)
                except (IndexError, ValueError):
                    pass

        return metrics

    def _save_study_results(self):
        """Save Optuna study results."""
        study_file = self.sweep_dir / "optuna_study.json"

        # Create study summary
        study_summary = {
            "study_name": self.study.study_name,
            "direction": self.study.direction.name,
            "n_trials": len(self.study.trials),
            "best_trial": (
                {
                    "number": self.study.best_trial.number,
                    "value": self.study.best_value,
                    "params": self.study.best_params,
                }
                if self.study.best_trial
                else None
            ),
            "trials": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                }
                for trial in self.study.trials
            ],
        }

        import json

        with open(study_file, "w", encoding="utf-8") as f:
            json.dump(study_summary, f, indent=2, default=str)

    def _print_summary(self):
        """Print optimization summary."""
        if not self.study.best_trial:
            self.console.print("[bold red]No successful trials![/bold red]")
            return

        from rich.table import Table

        table = Table(title="Bayesian Optimization Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total trials", str(len(self.study.trials)))
        table.add_row(
            "Successful trials",
            str(len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])),
        )
        table.add_row("Best value", f"{self.study.best_value:.4f}")
        table.add_row("Best trial", str(self.study.best_trial.number))

        self.console.print(table)

        # Print best parameters
        self.console.print("\n[bold green]Best parameters:[/bold green]")
        for key, value in self.study.best_params.items():
            self.console.print(f"  {key}: {value}")
