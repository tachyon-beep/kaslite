"""
Comprehensive test suite for the run_morphogenetic_experiment script.

This module provides backward compatibility by importing all test classes
from the refactored test modules. The tests have been split into smaller,
more manageable files:

- test_datasets.py: Dataset generation tests
- test_training.py: Training and evaluation tests
- test_cli.py: CLI argument parsing and main function tests
- test_integration.py: Integration and architecture scaling tests

For new development, import directly from the specific test modules.
This file maintains backward compatibility for existing test runners.
"""

# Import all test classes from refactored modules for backward compatibility
from .test_datasets import (
    TestCreateSpirals,
    TestCreateComplexMoons, 
    TestNewDatasets,
)

from .test_training import (
    TestTrainEpoch,
    TestEvaluate,
)

from .test_cli import (
    TestMainFunction,
    TestCLIDispatch,
    TestNewCLIFlags,
    TestNewCLIArguments,
)

from .test_integration import (
    TestIntegration,
    TestHighDimensionalIntegration,
    TestArchitectureScaling,
)

# Re-export all test classes to maintain existing imports
__all__ = [
    "TestCreateSpirals",
    "TestCreateComplexMoons", 
    "TestNewDatasets",
    "TestTrainEpoch",
    "TestEvaluate",
    "TestMainFunction",
    "TestCLIDispatch", 
    "TestNewCLIFlags",
    "TestNewCLIArguments",
    "TestIntegration",
    "TestHighDimensionalIntegration",
    "TestArchitectureScaling",
]


# All tests are now imported from the refactored modules above.
# This ensures backward compatibility while keeping code organized.
