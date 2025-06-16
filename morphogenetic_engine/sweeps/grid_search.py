"""
Enhanced grid search runner for morphogenetic experiments.

This module extracts and improves the existing grid search functionality
with parallel execution, progress tracking, and better error handling.
"""

import argparse
import hashlib
import multiprocessing as mp
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from .config import SweepConfig
from .results import SweepResults


class GridSearchRunner:
    """Enhanced grid search runner with parallel execution and progress tracking."""

    def __init__(self, config: SweepConfig, base_args: Optional[argparse.Namespace] = None):
        """Initialize the grid search runner."""
        self.config = config
        self.base_args = base_args or argparse.Namespace()
        self.console = Console()
        self.project_root = Path(__file__).parent.parent.parent

        # Create results directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sweep_dir = self.project_root / "results" / "sweeps" / f"grid_{self.timestamp}"
        self.sweep_dir.mkdir(parents=True, exist_ok=True)

        self.results = SweepResults(self.sweep_dir)

    def run_sweep(self) -> SweepResults:
        """Execute the grid search sweep."""
        combinations = self.config.get_grid_combinations()

        self.console.print(
            f"[bold green]Starting grid search with {len(combinations)} combinations[/bold green]"
        )
        self.console.print(f"Results will be saved to: {self.sweep_dir}")

        # Create progress tracking
        with Progress() as progress:
            task = progress.add_task("[green]Running experiments...", total=len(combinations))

            if self.config.max_parallel > 1:
                self._run_parallel(combinations, progress, task)
            else:
                self._run_sequential(combinations, progress, task)

        # Finalize results
        self.results.finalize()
        self._print_summary()

        return self.results

    def _run_sequential(self, combinations: List[Dict[str, Any]], progress: Progress, task: TaskID):
        """Run experiments sequentially."""
        for i, combo in enumerate(combinations):
            result = self._run_single_experiment(combo, i)
            self.results.add_result(result)
            progress.update(task, advance=1)

    def _run_parallel(self, combinations: List[Dict[str, Any]], progress: Progress, task: TaskID):
        """Run experiments in parallel."""
        max_workers = min(self.config.max_parallel, mp.cpu_count())

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_combo = {
                executor.submit(self._run_single_experiment, combo, i): (combo, i)
                for i, combo in enumerate(combinations)
            }

            # Collect results as they complete
            for future in as_completed(future_to_combo):
                result = future.result()
                self.results.add_result(result)
                progress.update(task, advance=1)

    def _run_single_experiment(self, combo: Dict[str, Any], run_index: int) -> Dict[str, Any]:
        """Run a single experiment with the given parameter combination."""
        run_slug = self._generate_run_slug(combo, run_index)

        try:
            # Prepare command
            cmd = self._build_command(combo, run_slug)

            # Run experiment
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

            # Parse results
            experiment_result = {
                "run_id": run_index,
                "run_slug": run_slug,
                "parameters": combo,
                "runtime": end_time - start_time,
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            if result.returncode == 0:
                # Try to extract metrics from output
                metrics = self._extract_metrics(result.stdout)
                experiment_result.update(metrics)
            else:
                experiment_result["error"] = f"Process failed with code {result.returncode}"

            return experiment_result

        except subprocess.TimeoutExpired:
            return {
                "run_id": run_index,
                "run_slug": run_slug,
                "parameters": combo,
                "success": False,
                "error": f"Timeout after {self.config.timeout_per_trial} seconds",
            }
        except (ValueError, RuntimeError, OSError) as e:
            return {
                "run_id": run_index,
                "run_slug": run_slug,
                "parameters": combo,
                "success": False,
                "error": str(e),
            }

    def _build_command(self, combo: Dict[str, Any], _run_slug: str) -> List[str]:
        """Build the command to run a single experiment."""
        cmd = [sys.executable, "scripts/run_morphogenetic_experiment.py"]

        # Add parameters from combination
        for key, value in combo.items():
            cmd.extend([f"--{key}", str(value)])

        # Add any base arguments
        if hasattr(self.base_args, "seed") and self.base_args.seed is not None:
            cmd.extend(["--seed", str(self.base_args.seed)])

        return cmd

    def _generate_run_slug(self, combo: Dict[str, Any], run_index: int) -> str:
        """Generate a unique slug for a parameter combination."""
        # Create a short hash of the parameter combination
        combo_str = "_".join(f"{k}={v}" for k, v in sorted(combo.items()))
        combo_hash = hashlib.md5(combo_str.encode()).hexdigest()[:8]

        return f"run_{run_index:03d}_{combo_hash}"

    def _extract_metrics(self, stdout: str) -> Dict[str, Any]:
        """Extract metrics from experiment output."""
        metrics = {}

        # Look for common metric patterns in output
        lines = stdout.split("\n")
        for line in lines:
            line = line.strip()

            # Look for final accuracy metrics
            if "Final accuracy:" in line:
                try:
                    acc_str = line.split("Final accuracy:")[1].strip()
                    metrics["final_acc"] = float(acc_str)
                except (IndexError, ValueError):
                    pass

            # Look for best accuracy metrics
            if "Best accuracy:" in line:
                try:
                    acc_str = line.split("Best accuracy:")[1].strip()
                    metrics["best_acc"] = float(acc_str)
                except (IndexError, ValueError):
                    pass

            # Look for validation accuracy
            if "val_acc:" in line:
                try:
                    acc_str = line.split("val_acc:")[1].strip().split()[0]
                    metrics["val_acc"] = float(acc_str)
                except (IndexError, ValueError):
                    pass

        return metrics

    def _print_summary(self):
        """Print a summary of the sweep results."""
        successful_results = [r for r in self.results.results if r.get("success", False)]

        if not successful_results:
            self.console.print("[bold red]No successful experiments![/bold red]")
            return

        # Find best result
        metric = self.config.target_metric
        if self.config.direction == "maximize":
            best_result = max(successful_results, key=lambda x: x.get(metric, 0))
        else:
            best_result = min(successful_results, key=lambda x: x.get(metric, float("inf")))

        # Create summary table
        table = Table(title="Grid Search Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total experiments", str(len(self.results.results)))
        table.add_row("Successful", str(len(successful_results)))
        table.add_row("Failed", str(len(self.results.results) - len(successful_results)))
        table.add_row(f"Best {metric}", f"{best_result.get(metric, 'N/A'):.4f}")
        table.add_row("Best run", best_result.get("run_slug", "N/A"))

        self.console.print(table)

        # Print best parameters
        if "parameters" in best_result:
            self.console.print("\n[bold green]Best parameters:[/bold green]")
            for key, value in best_result["parameters"].items():
                self.console.print(f"  {key}: {value}")
