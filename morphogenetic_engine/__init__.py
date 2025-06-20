"""
Morphogenetic Engine - A neural architecture framework.

This package provides core components for building and training
morphogenetic neural networks with adaptive architecture capabilities.
"""

from . import cli, ui_dashboard, components, core, datasets, experiment, logger, monitoring, sweeps, training

__all__ = [
    "ui_dashboard",
    "components",
    "core",
    "datasets",
    "experiment",
    "logger",
    "monitoring",
    "training",
    "sweeps",
    "cli",
]
