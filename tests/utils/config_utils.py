"""Configuration utilities for testing.

This module provides utilities for creating test configurations,
experiment parameters, and sweep configurations.
"""

from typing import Any


def create_test_experiment_config(**overrides: Any) -> dict[str, Any]:
    """Create a test experiment configuration.

    Args:
        **overrides: Configuration values to override

    Returns:
        Dictionary containing experiment configuration
    """
    default_config = {
        "problem_type": "spirals",
        "n_samples": 1000,
        "input_dim": 2,
        "train_frac": 0.8,
        "batch_size": 32,
        "device": "cpu",
        "seed": 42,
        "warm_up_epochs": 10,
        "adaptation_epochs": 50,
        "lr": 0.001,
        "hidden_dim": 64,
        "num_layers": 3,
        "seeds_per_layer": 4,
        "blend_steps": 30,
        "shadow_lr": 0.001,
        "progress_thresh": 0.6,
        "drift_warn": 0.12,
        "acc_threshold": 0.95,
    }
    default_config.update(overrides)
    return default_config


def create_test_final_stats(**overrides: Any) -> dict[str, Any]:
    """Create test final statistics for experiments.

    Args:
        **overrides: Statistics values to override

    Returns:
        Dictionary containing final experiment statistics
    """
    default_stats = {
        "best_acc": 0.95,
        "accuracy_dip": 0.05,
        "recovery_time": 10,
        "seeds_activated": True,
        "final_acc": 0.94,
        "training_time": 120.5,
        "convergence_epoch": 45,
    }
    default_stats.update(overrides)
    return default_stats


def create_test_sweep_config(**overrides: Any) -> dict[str, Any]:
    """Create test sweep configuration for hyperparameter optimization.

    Args:
        **overrides: Sweep configuration values to override

    Returns:
        Dictionary containing sweep configuration
    """
    default_config = {
        "method": "grid",
        "parameters": {
            "lr": {"values": [0.001, 0.01, 0.1]},
            "batch_size": {"values": [16, 32, 64]},
            "hidden_dim": {"values": [32, 64, 128]},
        },
        "metric": {"name": "best_acc", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 10},
    }
    default_config.update(overrides)
    return default_config
