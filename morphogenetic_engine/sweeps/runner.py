"""
Comprehensive sweep execution functionality for morphogenetic experiments.

This module provides parameter sweep capabilities including grid search
and result management, extracted from the main experiment script.
"""

import argparse
import csv
import hashlib
import itertools
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from morphogenetic_engine.cli.arguments import get_valid_argument_names


def parse_value_list(value: Union[str, List, int, float]) -> List[Any]:
    """Parse a value that could be a comma-separated string, list, or single value."""
    if isinstance(value, str):
        # Handle comma-separated strings
        if "," in value:
            return [item.strip() for item in value.split(",")]
        else:
            return [value]
    elif isinstance(value, list):
        return value
    else:
        # Single numeric or other value
        return [value]


def validate_sweep_config(sweep_config: Dict[str, Any], valid_args: set) -> None:
    """Validate that all keys in the sweep config are valid argument names."""
    for key in sweep_config:
        # Remove leading dashes if present to match argument names
        clean_key = key.lstrip("-")
        if clean_key not in valid_args:
            raise ValueError(
                f"Unknown parameter in sweep config: '{key}'. Valid parameters: {sorted(valid_args)}"
            )


def load_sweep_configs(config_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load YAML sweep configuration(s) from a file or directory."""
    config_path = Path(config_path)
    configs = []

    if config_path.is_file():
        if config_path.suffix.lower() in [".yml", ".yaml"]:
            with open(config_path, "r", encoding="utf-8") as f:
                configs.append(yaml.safe_load(f))
        else:
            raise ValueError(f"Sweep config file must have .yml or .yaml extension: {config_path}")
    elif config_path.is_dir():
        yaml_files = list(config_path.glob("*.yml")) + list(config_path.glob("*.yaml"))
        if not yaml_files:
            raise ValueError(f"No YAML files found in directory: {config_path}")
        for yaml_file in sorted(yaml_files):
            with open(yaml_file, "r", encoding="utf-8") as f:
                configs.append(yaml.safe_load(f))
    else:
        raise ValueError(f"Sweep config path does not exist: {config_path}")

    return configs


def expand_grid(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a sweep configuration into a grid of parameter combinations."""
    if not sweep_config:
        return [{}]

    # Parse all values into lists
    param_lists = {}
    for key, value in sweep_config.items():
        param_lists[key] = parse_value_list(value)

    # Create cartesian product
    keys = list(param_lists.keys())
    value_combinations = itertools.product(*param_lists.values())

    # Convert to list of dictionaries
    grid = []
    for combination in value_combinations:
        combo_dict = dict(zip(keys, combination))
        grid.append(combo_dict)

    return grid


def merge_args_with_combo(
    base_args: argparse.Namespace, combo: Dict[str, Any]
) -> argparse.Namespace:
    """Merge base CLI arguments with a parameter combination from the sweep grid."""
    # Create a new namespace with base args
    merged_args = argparse.Namespace(**vars(base_args))

    # Override with combo values, converting types appropriately
    for key, value in combo.items():
        # Convert string values to appropriate types based on the original arg type
        if hasattr(merged_args, key):
            original_value = getattr(merged_args, key)
            if isinstance(original_value, bool):
                # Handle boolean conversion
                if isinstance(value, str):
                    converted_value = value.lower() in ("true", "1", "yes", "on")
                else:
                    converted_value = bool(value)
            elif isinstance(original_value, int):
                converted_value = int(value)
            elif isinstance(original_value, float):
                converted_value = float(value)
            else:
                converted_value = value

            setattr(merged_args, key, converted_value)
        else:
            # New parameter not in base args - add as-is
            setattr(merged_args, key, value)

    return merged_args


def generate_run_slug(combo: Dict[str, Any], run_index: int) -> str:
    """Generate a unique slug for a parameter combination."""
    # Create a short hash of the parameter combination
    combo_str = "_".join(f"{k}={v}" for k, v in sorted(combo.items()))
    combo_hash = hashlib.md5(combo_str.encode()).hexdigest()[:8]

    return f"run_{run_index:03d}_{combo_hash}"


def create_sweep_results_summary(sweep_runs: List[Dict[str, Any]], sweep_dir: Path) -> None:
    """Create a CSV summary of all sweep runs and their results."""
    if not sweep_runs:
        return

    summary_path = sweep_dir / "sweep_summary.csv"

    # Collect all parameter names and result keys
    all_params = set()
    all_results = set()

    for run in sweep_runs:
        all_params.update(run.get("parameters", {}).keys())
        all_results.update(run.get("results", {}).keys())

    # Write CSV
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = ["run_id", "run_slug"] + sorted(all_params) + sorted(all_results)
        writer.writerow(header)

        # Data rows
        for run in sweep_runs:
            row = [run.get("run_id", ""), run.get("run_slug", "")]

            # Add parameter values
            params = run.get("parameters", {})
            for param in sorted(all_params):
                row.append(params.get(param, ""))

            # Add result values
            results = run.get("results", {})
            for result in sorted(all_results):
                row.append(results.get(result, ""))

            writer.writerow(row)

    print(f"Sweep summary saved to: {summary_path}")


def create_run_directory_and_setup(sweep_dir: Path, run_slug: str, original_setup_func):
    """Create run directory and setup function for a single sweep run."""
    run_dir = sweep_dir / run_slug
    run_dir.mkdir(exist_ok=True)

    def setup_experiment_for_sweep(
        args_inner, run_dir_param=run_dir, original_setup_param=original_setup_func
    ):
        """Modified setup_experiment that puts logs in the run directory."""
        from torch.utils.tensorboard import SummaryWriter

        (
            logger_inner,
            tb_writer_inner,
            log_f_inner,
            device_inner,
            config_inner,
            slug_inner,
            project_root_inner,
        ) = original_setup_param(args_inner)

        # Move the log file to the run directory
        original_log_path = Path(log_f_inner.name)
        new_log_path = run_dir_param / original_log_path.name
        log_f_inner.close()

        if original_log_path.exists():
            shutil.move(str(original_log_path), str(new_log_path))

        # Reopen with new path
        log_f_new = new_log_path.open("w", encoding="utf-8")

        # Close the original TensorBoard writer and create a new one in the run directory
        tb_writer_inner.close()
        tb_dir_new = run_dir_param / "tensorboard"
        tb_writer_new = SummaryWriter(log_dir=str(tb_dir_new))

        return (
            logger_inner,
            tb_writer_new,
            log_f_new,
            device_inner,
            config_inner,
            slug_inner,
            project_root_inner,
        )

    return run_dir, setup_experiment_for_sweep


def process_single_sweep_config(
    config_idx: int,
    sweep_config: Dict[str, Any],
    args: argparse.Namespace,
    sweep_dir: Path,
    run_counter: int,
    valid_args: set,
) -> tuple[List[Dict[str, Any]], int]:
    """Process a single sweep configuration and return results and updated counter."""
    from morphogenetic_engine.runners import run_single_experiment, setup_experiment

    print(f"\nProcessing sweep config {config_idx + 1}")

    # Validate configuration
    try:
        validate_sweep_config(sweep_config, valid_args)
    except ValueError as e:
        print(f"Error in sweep config {config_idx + 1}: {e}")
        return [], run_counter

    # Expand parameter grid
    param_grid = expand_grid(sweep_config)
    print(f"Generated {len(param_grid)} parameter combinations")

    config_runs = []
    original_setup = setup_experiment

    # Run experiments for this config
    for combo in param_grid:
        run_counter += 1
        run_slug = generate_run_slug(combo, run_counter)

        print(f"\nRun {run_counter}: {run_slug}")
        print(f"Parameters: {combo}")

        # Merge CLI args with parameter combination
        run_args = merge_args_with_combo(args, combo)

        # Create run-specific setup
        _, setup_func = create_run_directory_and_setup(sweep_dir, run_slug, original_setup)

        # Temporarily replace setup_experiment
        import morphogenetic_engine.runners as runners_module

        original_setup_ref = runners_module.setup_experiment
        runners_module.setup_experiment = setup_func

        try:
            # Run the experiment
            experiment_results = run_single_experiment(run_args, run_slug)

            # Separate parameters from results
            run_record = {
                "run_id": run_slug,
                "run_slug": run_slug,
                "parameters": combo,
                "results": {
                    k: v for k, v in experiment_results.items() if k not in ["run_id", "parameters"]
                },
            }
            config_runs.append(run_record)

            print(f"  Final accuracy: {experiment_results.get('best_acc', 0.0):.4f}")
            if experiment_results.get("error"):
                print(f"  Error: {experiment_results['error']}")

        finally:
            # Restore original setup_experiment
            runners_module.setup_experiment = original_setup_ref

    return config_runs, run_counter


def run_parameter_sweep(args: argparse.Namespace) -> None:
    """Run a parameter sweep based on the sweep configuration."""
    print(f"Loading sweep configuration from: {args.sweep_config}")

    # Load and validate sweep configs
    try:
        sweep_configs = load_sweep_configs(args.sweep_config)
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        print(f"Error loading sweep config: {e}")
        return

    valid_args = get_valid_argument_names()

    # Create sweep results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).parent.parent.parent
    sweep_dir = project_root / "results" / "sweeps" / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep results will be saved to: {sweep_dir}")

    # Process all sweep configs
    all_runs = []
    run_counter = 0

    for config_idx, sweep_config in enumerate(sweep_configs):
        config_runs, run_counter = process_single_sweep_config(
            config_idx, sweep_config, args, sweep_dir, run_counter, valid_args
        )
        all_runs.extend(config_runs)

    # Create summary and print results
    create_sweep_results_summary(all_runs, sweep_dir)

    print(f"\nSweep completed! {len(all_runs)} experiments run.")
    print(f"Results saved to: {sweep_dir}")

    # Print quick summary
    if all_runs:
        successful_runs = [r for r in all_runs if not r.get("results", {}).get("error")]
        if successful_runs:
            best_run = max(successful_runs, key=lambda x: x.get("results", {}).get("best_acc", 0.0))
            best_acc = best_run.get("results", {}).get("best_acc", 0.0)
            print(f"Best accuracy: {best_acc:.4f} (run: {best_run.get('run_slug', 'unknown')})")
