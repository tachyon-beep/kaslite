"""
Experiment utilities for morphogenetic architecture experiments.

This module provides functions for building models, managing experiments,
and utility functions for experiment management.
"""

from __future__ import annotations

from torch import nn

from morphogenetic_engine.components import BaseNet
from morphogenetic_engine.core import KasminaMicro, SeedManager
from morphogenetic_engine.logger import ExperimentLogger


def build_model_and_agents(
    # Model architecture
    hidden_dim: int,
    input_dim: int,
    num_layers: int,
    seeds_per_layer: int,
    # Model behavior
    blend_steps: int,
    shadow_lr: float,
    progress_thresh: float,
    drift_warn: float,
    # Agents
    acc_threshold: float,
    # Infrastructure
    device: str,
    logger: ExperimentLogger,
    tb_writer,  # Add type hint if available
    # Dataset-specific
    problem_type: str,
    **kwargs,  # Capture any other unused args
):
    """Initialize the SeedManager, BaseNet model, loss function, and KasminaMicro."""
    seed_manager = SeedManager()

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
        blend_steps=blend_steps,
        shadow_lr=shadow_lr,
        progress_thresh=progress_thresh,
        drift_warn=drift_warn,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)

    # The `tamiyo` agent from the high-level plan
    tamiyo = KasminaMicro(
        seed_manager,
        patience=15,  # This could be parameterized
        delta=5e-4,  # This could be parameterized
        acc_threshold=acc_threshold,
        logger=logger,
    )

    # The `karn` agent is implicitly the training loop logic itself
    karn = None  # Placeholder, as Karn's logic is not a separate class yet

    return model, seed_manager, karn, tamiyo
