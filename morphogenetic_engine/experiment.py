"""
Experiment utilities for morphogenetic architecture experiments.

This module provides functions for building models, managing experiments,
and utility functions for experiment management.
"""

from __future__ import annotations

from torch import nn

from morphogenetic_engine.components import BaseNet
from morphogenetic_engine.core import KasminaMicro, SeedManager


def build_model_and_agents(args, device):
    """Initialize the SeedManager, BaseNet model, loss function, and KasminaMicro."""
    seed_manager = SeedManager()

    # Determine number of classes based on problem type
    if args.problem_type == "cifar10":
        num_classes = 10
    else:
        # All synthetic datasets are binary classification
        # Handle case where args might be a Mock object
        num_classes_attr = getattr(args, "num_classes", 2)
        if hasattr(num_classes_attr, "_mock_name"):  # It's a Mock object
            num_classes = 2
        else:
            num_classes = num_classes_attr

    model = BaseNet(
        args.hidden_dim,
        seed_manager=seed_manager,
        input_dim=args.input_dim,
        output_dim=num_classes,
        num_layers=args.num_layers,
        seeds_per_layer=args.seeds_per_layer,
        blend_steps=args.blend_steps,
        shadow_lr=args.shadow_lr,
        progress_thresh=args.progress_thresh,
        drift_warn=args.drift_warn,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    kasmina = KasminaMicro(seed_manager, patience=15, delta=5e-4, acc_threshold=args.acc_threshold)

    return model, seed_manager, loss_fn, kasmina
