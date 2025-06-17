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
from morphogenetic_engine.runners import run_single_experiment
from morphogenetic_engine.sweeps.runner import run_parameter_sweep


def main():
    """Main function to orchestrate single experiments or parameter sweeps."""
    args = parse_experiment_arguments()

    if args.sweep_config:
        # Run parameter sweep
        run_parameter_sweep(args)
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
