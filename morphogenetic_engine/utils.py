"""
Utility functions for morphogenetic experiments.

This module provides various utility functions for experiment setup,
data processing, and file operations.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from morphogenetic_engine.training import clear_seed_report_cache


def is_testing_mode() -> bool:
    """Check if we're in testing mode."""
    try:
        return "pytest" in sys.modules or "unittest" in sys.modules
    except (ImportError, AttributeError):
        return False


def generate_experiment_slug(args) -> str:
    """Generate a unique slug for an experiment configuration."""
    return (
        f"{args.problem_type}_dim{args.input_dim}_{args.device}"
        f"_h{args.hidden_dim}_bs{args.blend_steps}"
        f"_lr{args.shadow_lr}_pt{args.progress_thresh}"
        f"_dw{args.drift_warn}"
    )


def create_experiment_config(args, device) -> Dict[str, Any]:
    """Create a configuration dictionary from parsed arguments."""
    return {
        "problem_type": args.problem_type,
        "n_samples": args.n_samples,
        "input_dim": args.input_dim,
        "train_frac": args.train_frac,
        "batch_size": args.batch_size,
        "device": str(device),
        "seed": args.seed,
        "warm_up_epochs": args.warm_up_epochs,
        "adaptation_epochs": args.adaptation_epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "seeds_per_layer": args.seeds_per_layer,
        "blend_steps": args.blend_steps,
        "shadow_lr": args.shadow_lr,
        "progress_thresh": args.progress_thresh,
        "drift_warn": args.drift_warn,
        "acc_threshold": args.acc_threshold,
    }


def write_experiment_log_header(log_f, config: Dict[str, Any], args) -> None:
    """Write the detailed configuration header to the log file."""
    clear_seed_report_cache()

    # Write comprehensive configuration header
    log_f.write("# Morphogenetic Architecture Experiment Log\n")
    log_f.write(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write("# Configuration:\n")

    # Basic parameters
    for key in [
        "problem_type",
        "n_samples",
        "input_dim",
        "train_frac",
        "batch_size",
        "device",
        "seed",
    ]:
        if key in config:
            log_f.write(f"# {key}: {config[key]}\n")

    # Dataset-specific parameters
    if args.problem_type == "spirals":
        log_f.write(f"# noise: {args.noise}\n")
        log_f.write(f"# rotations: {args.rotations}\n")
    elif args.problem_type in ["moons", "complex_moons"]:
        log_f.write(f"# moon_noise: {args.moon_noise}\n")
        if args.problem_type == "moons":
            log_f.write(f"# moon_sep: {args.moon_sep}\n")
    elif args.problem_type == "clusters":
        log_f.write(f"# cluster_count: {args.cluster_count}\n")
        log_f.write(f"# cluster_size: {args.cluster_size}\n")
        log_f.write(f"# cluster_std: {args.cluster_std}\n")
        log_f.write(f"# cluster_sep: {args.cluster_sep}\n")
    elif args.problem_type == "spheres":
        log_f.write(f"# sphere_count: {args.sphere_count}\n")
        log_f.write(f"# sphere_size: {args.sphere_size}\n")
        log_f.write(f"# sphere_radii: {args.sphere_radii}\n")
        log_f.write(f"# sphere_noise: {args.sphere_noise}\n")

    # Morphogenetic architecture parameters
    morpho_params = [
        "warm_up_epochs",
        "adaptation_epochs",
        "lr",
        "hidden_dim",
        "num_layers",
        "seeds_per_layer",
        "blend_steps",
        "shadow_lr",
        "progress_thresh",
        "drift_warn",
        "acc_threshold",
    ]
    for param in morpho_params:
        if param in config:
            log_f.write(f"# {param}: {config[param]}\n")

    log_f.write("#\n")
    log_f.write("# Data format: epoch,seed,state,alpha\n")
    log_f.write("epoch,seed,state,alpha\n")


def write_experiment_log_footer(log_f, final_stats: Dict[str, Any], seed_manager) -> None:
    """Write the summary footer to the log file."""
    log_f.write("#\n")
    log_f.write("# Experiment completed successfully\n")
    log_f.write(f"# End timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write(f"# Final best accuracy: {final_stats['best_acc']:.4f}\n")

    if final_stats.get("seeds_activated", False):
        active_seeds = sum(
            1 for info in seed_manager.seeds.values() if info["module"].state == "active"
        )
        log_f.write(f"# Seeds activated: {active_seeds}/{len(seed_manager.seeds)}\n")
    else:
        log_f.write(f"# Seeds activated: 0/{len(seed_manager.seeds)}\n")

    log_f.write("# ===== LOG COMPLETE =====\n")


def export_metrics_for_dvc(final_stats: Dict[str, Any], slug: str, project_root: Path) -> None:
    """Export metrics in JSON format for DVC tracking."""
    metrics = {
        "best_acc": final_stats["best_acc"],
        "accuracy_dip": final_stats.get("accuracy_dip"),
        "recovery_time": final_stats.get("recovery_time"),
        "seeds_activated": final_stats.get("seeds_activated", False),
    }

    # Remove None values
    metrics = {k: v for k, v in metrics.items() if v is not None}

    # Save metrics JSON
    metrics_path = project_root / "results" / f"metrics_{slug}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)  # Create results directory if it doesn't exist
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
