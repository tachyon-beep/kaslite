"""
Command-line argument parsing for morphogenetic experiments.

This module provides a comprehensive argument parser for configuring
morphogenetic architecture experiments with various datasets and parameters.
"""

import argparse
from typing import List, Optional


def parse_float_list(value: str) -> List[float]:
    """Parse a comma-separated string of floats."""
    return [float(x.strip()) for x in value.split(",")]


def create_experiment_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for morphogenetic experiments."""
    parser = argparse.ArgumentParser(
        description="Run morphogenetic-architecture experiments on various datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Sweep configuration
    parser.add_argument(
        "--sweep_config",
        "-s",
        type=str,
        default=None,
        help="Path to YAML sweep configuration file or directory",
    )

    # Existing morphogenetic parameters
    parser.add_argument(
        "--blend_steps",
        type=int,
        default=30,
        help="Number of blend steps for morphogenetic adaptation",
    )
    parser.add_argument(
        "--shadow_lr", type=float, default=1e-3, help="Learning rate for shadow networks"
    )
    parser.add_argument(
        "--progress_thresh", type=float, default=0.6, help="Progress threshold for seed activation"
    )
    parser.add_argument(
        "--drift_warn",
        type=float,
        default=0.12,
        help="Drift warning threshold (0=disable)",
    )

    # Problem type and general parameters
    parser.add_argument(
        "--problem_type",
        choices=["spirals", "moons", "clusters", "spheres", "complex_moons", "cifar10"],
        default="spirals",
        help="Type of problem to solve",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of output classes (auto-detected for known datasets)",
    )
    parser.add_argument("--n_samples", type=int, default=2000, help="Total samples (split evenly)")
    parser.add_argument(
        "--input_dim",
        type=int,
        default=3,
        help="Embedding/dimension for clusters and spheres",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.8, help="Train/validation split fraction"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help='"cpu" or "cuda"')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Spirals-specific parameters
    parser.add_argument("--noise", type=float, default=0.25, help="Spirals noise")
    parser.add_argument("--rotations", type=int, default=4, help="Spirals turns")

    # Moons-specific parameters
    parser.add_argument("--moon_noise", type=float, default=0.1, help="Gaussian noise for moons")
    parser.add_argument("--moon_sep", type=float, default=0.5, help="Separation between half-moons")

    # Clusters-specific parameters
    parser.add_argument("--cluster_count", type=int, default=2, help="Number of Gaussian blobs")
    parser.add_argument("--cluster_size", type=int, default=500, help="Points per cluster")
    parser.add_argument("--cluster_std", type=float, default=0.5, help="Cluster standard deviation")
    parser.add_argument(
        "--cluster_sep",
        type=float,
        default=3.0,
        help="Distance between cluster centers",
    )

    # Spheres-specific parameters
    parser.add_argument(
        "--sphere_count",
        type=int,
        default=2,
        help="Number of concentric spherical shells",
    )
    parser.add_argument("--sphere_size", type=int, default=500, help="Points per sphere shell")
    parser.add_argument(
        "--sphere_radii", type=parse_float_list, default="1,2", help="Comma-separated list of radii"
    )
    parser.add_argument(
        "--sphere_noise",
        type=float,
        default=0.05,
        help="Jitter magnitude on sphere surface",
    )

    # Additional hyper-parameters
    parser.add_argument(
        "--warm_up_epochs",
        type=int,
        default=50,
        help="Number of warm-up epochs before adaptation phase",
    )
    parser.add_argument(
        "--adaptation_epochs",
        type=int,
        default=200,
        help="Number of epochs for the adaptation phase",
    )
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension size for the network",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=8,
        help="Number of hidden layers in the network (each with a corresponding seed)",
    )
    parser.add_argument(
        "--seeds_per_layer",
        type=int,
        default=1,
        help="Number of sentinel seeds per layer (allows multiple adaptive paths per layer)",
    )
    parser.add_argument(
        "--acc_threshold",
        type=float,
        default=0.95,
        help="Accuracy threshold for Kasmina germination",
    )

    return parser


def parse_experiment_arguments(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments for morphogenetic experiments."""
    parser = create_experiment_parser()
    return parser.parse_args(args)


def get_valid_argument_names() -> set:
    """Get the set of valid argument names for sweep validation."""
    return {
        "sweep_config",
        "blend_steps",
        "shadow_lr",
        "progress_thresh",
        "drift_warn",
        "problem_type",
        "n_samples",
        "input_dim",
        "train_frac",
        "batch_size",
        "device",
        "seed",
        "noise",
        "rotations",
        "moon_noise",
        "moon_sep",
        "cluster_count",
        "cluster_size",
        "cluster_std",
        "cluster_sep",
        "sphere_count",
        "sphere_size",
        "sphere_radii",
        "sphere_noise",
        "warm_up_epochs",
        "adaptation_epochs",
        "lr",
        "hidden_dim",
        "num_layers",
        "seeds_per_layer",
        "acc_threshold",
        "num_classes",
    }
