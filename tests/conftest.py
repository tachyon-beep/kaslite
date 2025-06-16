"""
Shared test configuration and fixtures.

This module provides common test fixtures and utilities used across
all test modules in the morphogenetic engine test suite.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Provide a device for testing (CPU by default)."""
    return torch.device("cpu")


@pytest.fixture
def random_seed():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def small_dataset():
    """Provide a small dataset for testing."""
    X = torch.randn(32, 2)
    y = torch.randint(0, 2, (32,))
    return X, y


@pytest.fixture
def mock_args():
    """Provide mock arguments for testing."""
    class MockArgs:
        """Mock arguments class for testing with default values."""
        hidden_dim = 32
        input_dim = 2
        num_layers = 3
        seeds_per_layer = 2
        blend_steps = 30
        shadow_lr = 1e-3
        progress_thresh = 0.6
        drift_warn = 0.12
        acc_threshold = 0.95
        lr = 1e-3
        batch_size = 16
        problem_type = "moons"
    
    return MockArgs()
