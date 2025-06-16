"""Common test utilities for components testing."""

import torch
from morphogenetic_engine.core import SeedManager


def create_test_seed_manager():
    """Create a seed manager for testing."""
    return SeedManager()


def create_test_input(batch_size: int = 4, input_dim: int = 2) -> torch.Tensor:
    """Create test input tensor."""
    return torch.randn(batch_size, input_dim)


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected {expected_shape}, got {tensor.shape}"


def count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
