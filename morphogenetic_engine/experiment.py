"""
Experiment utilities for morphogenetic architecture experiments.

This module provides functions for building models, managing experiments,
and utility functions for experiment management.
"""

from __future__ import annotations

from torch import nn

from morphogenetic_engine.components import BaseNet
from morphogenetic_engine.core import BlendingConfig, KasminaMicro, SeedManager
from morphogenetic_engine.logger import ExperimentLogger


def build_model_and_agents(
    # Model architecture
    hidden_dim: int,
    input_dim: int,
    num_layers: int,
    seeds_per_layer: int,
    # Model behavior
    shadow_lr: float,
    drift_warn: float,
    # Agents
    acc_threshold: float,
    # Infrastructure
    device: str,
    logger: ExperimentLogger,
    # Dataset-specific
    problem_type: str,
    # Optional blending configuration
    blend_steps: int | None = None,  # Still accept for config creation
    **kwargs,  # Capture any other unused args like sweep_config, tb_writer
):
    """Initialize the SeedManager, BaseNet model, loss function, and KasminaMicro."""
    # kwargs intentionally captures unused parameters for compatibility with experiment runners
    _ = kwargs  # Suppress unused parameter warning
    seed_manager = SeedManager(logger=logger)

    # Create blending configuration with centralized parameters
    blend_config = BlendingConfig(
        fixed_steps=blend_steps if blend_steps is not None else 30,  # Default if not provided
        # Other parameters use defaults for now
    )

    # Determine number of classes based on problem type
    if problem_type == "cifar10":
        num_classes = 10
    else:
        # All synthetic datasets are binary classification
        num_classes = 2

    model = BaseNet(
        hidden_dim=hidden_dim,
        seed_manager=seed_manager,
        input_dim=input_dim,
        output_dim=num_classes,
        num_layers=num_layers,
        seeds_per_layer=seeds_per_layer,
        shadow_lr=shadow_lr,
        drift_warn=drift_warn,
        blend_cfg=blend_config,  # Pass the centralized config
    ).to(device)

    # The `tamiyo` agent from the high-level plan
    tamiyo = KasminaMicro(
        seed_manager,
        patience=15,  # This could be parameterized
        delta=5e-4,  # This could be parameterized
        acc_threshold=acc_threshold,
        logger=logger,
        blending_config=blend_config,  # Pass the same config instance
    )

    # The `karn` agent is implicitly the training loop logic itself
    karn = None  # Placeholder, as Karn's logic is not a separate class yet

    return model, seed_manager, karn, tamiyo
