"""
Run a morphogenetic-architecture experiment on various datasets.

• Phase 1 – train the full network for warm_up_epochs
• Phase 2 – freeze the trunk, let Kasmina germinate seeds on a plateau

This is the main entry point for running single experiments or parameter sweeps.
The actual experiment logic has been refactored into the morphogenetic_engine package.
"""

import random
import sys

import numpy as np
import torch

from morphogenetic_engine.cli.arguments import parse_experiment_arguments
from morphogenetic_engine.runners import run_single_experiment, setup_experiment_for_tests
from morphogenetic_engine.sweeps.runner import run_parameter_sweep


def main():
    """Main function to orchestrate single experiments or parameter sweeps."""
    args = parse_experiment_arguments()

    if args.sweep_config:
        # Run parameter sweep - use the real implementation, not compatibility version
        from morphogenetic_engine.sweeps.runner import run_parameter_sweep as run_sweep_impl
        run_sweep_impl(args)
    else:
        # Run single experiment
        run_single_experiment(args)


if __name__ == "__main__":
    # Set global random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()


# Backward compatibility functions for tests
def setup_experiment(args):
    """Backward-compatible setup_experiment for tests that expect 5 return values."""
    # Import here to ensure we use the patched versions if tests are running
    import logging
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter  # Import inside function for proper patching
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Set device based on args and availability
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    # Create configuration using the imported function
    from morphogenetic_engine.utils import create_experiment_config, generate_experiment_slug
    config = create_experiment_config(args, device)
    slug = generate_experiment_slug(args)

    # Initialize Prometheus monitoring
    from morphogenetic_engine.monitoring import initialize_monitoring
    initialize_monitoring(experiment_id=slug, port=8000)

    # Determine log location and initialise logger
    project_root = Path(__file__).parent.parent

    # Configure MLflow if available
    from morphogenetic_engine.utils import is_testing_mode
    MLFLOW_AVAILABLE = not is_testing_mode()
    if MLFLOW_AVAILABLE:
        try:
            import mlflow
            mlruns_dir = project_root / "mlruns"
            mlruns_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(config.get("mlflow_uri", "file://" + str(mlruns_dir)))
            mlflow.set_experiment(config.get("experiment_name", args.problem_type))
        except ImportError:
            pass
    
    log_dir = project_root / "results"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"results_{slug}.log"

    logger = ExperimentLogger(str(log_path), config)
    log_f = log_path.open("w", encoding="utf-8")

    # Create TensorBoard writer - this will use the patched SummaryWriter if testing
    tb_dir = project_root / "runs" / slug
    # Access SummaryWriter through globals() to ensure patches work
    tb_writer = globals()['SummaryWriter'](log_dir=str(tb_dir))

    return logger, tb_writer, log_f, device, config


def parse_arguments():
    """Backward-compatible argument parser."""
    return parse_experiment_arguments()


def run_parameter_sweep_compat(args):
    """Backward-compatible run_parameter_sweep that uses script-level imports for testing."""
    import yaml
    
    print(f"Loading sweep configuration from: {args.sweep_config}")

    # Load and validate sweep configs - uses script-level imported function
    try:
        sweep_configs = load_sweep_configs(args.sweep_config)
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        print(f"Error loading sweep config: {e}")
        return

    valid_args = get_valid_argument_names()

    # Create sweep results directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).parent.parent
    sweep_dir = project_root / "results" / "sweeps" / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep results will be saved to: {sweep_dir}")

    # Process all sweep configs - simplified for test compatibility
    all_runs = []
    run_counter = 0

    for config_idx, sweep_config in enumerate(sweep_configs):
        print(f"\nProcessing sweep config {config_idx + 1}")

        # Validate configuration
        try:
            validate_sweep_config(sweep_config, valid_args)
        except ValueError as e:
            print(f"Error in sweep config {config_idx + 1}: {e}")
            continue

        # Expand parameter grid
        param_grid = expand_grid(sweep_config)
        print(f"Generated {len(param_grid)} parameter combinations")

        # Run experiments for this config
        for combo in param_grid:
            run_counter += 1
            run_slug = generate_run_slug(combo, run_counter)

            print(f"\nRun {run_counter}: {run_slug}")
            print(f"Parameters: {combo}")

            # Merge CLI args with parameter combination
            run_args = merge_args_with_combo(args, combo)

            # Run the experiment using script-level imported function
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
            all_runs.append(run_record)

            print(f"  Final accuracy: {experiment_results.get('best_acc', 0.0):.4f}")
            if experiment_results.get("error"):
                print(f"  Error: {experiment_results['error']}")

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


# Alias for backward compatibility with tests
run_parameter_sweep = run_parameter_sweep_compat


def _is_testing_mode() -> bool:
    """Check if we're in testing mode."""
    try:
        return "pytest" in sys.modules or "unittest" in sys.modules
    except (ImportError, AttributeError):
        return False


# Legacy imports for backward compatibility
try:
    from morphogenetic_engine import datasets  # For tests that patch datasets
    from morphogenetic_engine.logger import ExperimentLogger  # For tests that patch ExperimentLogger
    from morphogenetic_engine.runners import (
        get_dataloaders,
        log_final_summary,
        export_metrics_for_dvc,
        run_single_experiment  # For tests that patch run_single_experiment
    )
    from morphogenetic_engine.experiment import build_model_and_agents
    from morphogenetic_engine.utils import (
        write_experiment_log_header as write_log_header,
        generate_experiment_slug,
        create_experiment_config as setup_experiment_config
    )
    # Import sweep functionality for backward compatibility
    from morphogenetic_engine.sweeps.runner import (
        expand_grid,
        generate_run_slug,
        load_sweep_configs,
        merge_args_with_combo,
        parse_value_list,
        create_sweep_results_summary,
        validate_sweep_config,
        process_single_sweep_config
        # Note: run_parameter_sweep is NOT imported here - we use the compatibility version
    )
    from morphogenetic_engine.cli.arguments import get_valid_argument_names
    from pathlib import Path  # For tests that patch Path
    from torch.utils.tensorboard import SummaryWriter  # For tests that patch SummaryWriter
except ImportError:
    # Fallback if modules aren't available
    pass


def setup_experiment_for_tests(args):
    """Script-level setup_experiment_for_tests that uses script imports for proper test patching."""
    return setup_experiment(args)
