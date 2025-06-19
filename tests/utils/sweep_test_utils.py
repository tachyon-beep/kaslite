"""
Test utilities for sweep CLI testing.

This module provides helper functions and builders for creating
test configurations and mock objects.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


class SweepConfigBuilder:
    """Builder pattern for creating test sweep configurations."""

    def __init__(self):
        """Initialize with default configuration."""
        self.config = {
            "sweep_type": "grid",
            "experiment": {"problem_type": "spirals", "input_dim": 3, "n_samples": 1000},
            "parameters": {"lr": [0.01], "hidden_dim": [64]},
            "execution": {"max_parallel": 1, "timeout_per_trial": 300},
        }

    def with_sweep_type(self, sweep_type: str) -> "SweepConfigBuilder":
        """Set the sweep type."""
        self.config["sweep_type"] = sweep_type
        return self

    def with_parameters(self, parameters: Dict[str, Any]) -> "SweepConfigBuilder":
        """Set the parameter sweep configuration."""
        self.config["parameters"] = parameters
        return self

    def with_execution(self, execution: Dict[str, Any]) -> "SweepConfigBuilder":
        """Set the execution configuration."""
        self.config["execution"] = execution
        return self

    def with_experiment(self, experiment: Dict[str, Any]) -> "SweepConfigBuilder":
        """Set the experiment configuration."""
        self.config["experiment"] = experiment
        return self

    def build(self) -> Dict[str, Any]:
        """Build and return the configuration dictionary."""
        return self.config.copy()

    def save_to_file(self, file_path: Path) -> Path:
        """Save the configuration to a YAML file."""
        file_path.write_text(yaml.dump(self.build()))
        return file_path


def create_invalid_yaml_content() -> str:
    """Create invalid YAML content for testing error handling."""
    return "invalid: yaml: content: ["


def create_empty_config_content() -> str:
    """Create empty configuration content."""
    return ""


def create_minimal_valid_config() -> Dict[str, Any]:
    """Create the minimal valid configuration for testing."""
    return SweepConfigBuilder().build()


def create_complex_grid_config() -> Dict[str, Any]:
    """Create a complex grid search configuration for integration testing."""
    return (
        SweepConfigBuilder()
        .with_parameters(
            {
                "lr": [0.001, 0.01, 0.1],
                "hidden_dim": [32, 64, 128],
                "num_layers": [2, 4, 6],
                "dropout": [0.0, 0.1, 0.2],
            }
        )
        .with_execution({"max_parallel": 4, "timeout_per_trial": 1800})
        .build()
    )


def create_bayesian_config() -> Dict[str, Any]:
    """Create a Bayesian optimization configuration."""
    return (
        SweepConfigBuilder()
        .with_sweep_type("bayesian")
        .with_parameters(
            {
                "lr": {"type": "float", "low": 0.0001, "high": 0.1},
                "hidden_dim": {"type": "int", "low": 32, "high": 256},
                "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            }
        )
        .build()
    )
