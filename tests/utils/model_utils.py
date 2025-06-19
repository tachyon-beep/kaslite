"""Model testing utilities.

This module provides utilities for testing PyTorch models, including
parameter counting, output shape validation, and mode assertions.
"""

import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in a model.

    Args:
        model: PyTorch model to analyze

    Returns:
        Total number of parameters in the model

    Raises:
        TypeError: If model is not a PyTorch module
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model)}")

    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch model to analyze

    Returns:
        Number of trainable parameters in the model

    Raises:
        TypeError: If model is not a PyTorch module
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model)}")

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_model_output_shape(
    model: torch.nn.Module, input_tensor: torch.Tensor, expected_output_shape: tuple[int, ...]
) -> None:
    """Assert model produces output of expected shape.

    Args:
        model: Model to test
        input_tensor: Input to pass through the model
        expected_output_shape: Expected shape of model output

    Raises:
        AssertionError: If output shape doesn't match expected shape
        TypeError: If model is not a PyTorch module
        RuntimeError: If model forward pass fails
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model)}")

    model.eval()
    try:
        with torch.no_grad():
            output = model(input_tensor)
    except Exception as e:
        raise RuntimeError(f"Model forward pass failed: {e}") from e

    from .tensor_utils import assert_tensor_shape

    assert_tensor_shape(output, expected_output_shape)


def create_mock_model(input_dim: int, output_dim: int, hidden_dim: int = 64) -> torch.nn.Module:
    """Create a simple mock model for testing.

    Args:
        input_dim: Input dimension size
        output_dim: Output dimension size
        hidden_dim: Hidden layer dimension size

    Returns:
        Simple sequential model for testing

    Raises:
        ValueError: If any dimension is non-positive
    """
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    )


def assert_model_in_eval_mode(model: torch.nn.Module) -> None:
    """Assert that model is in evaluation mode.

    Args:
        model: Model to check

    Raises:
        AssertionError: If model is not in eval mode
        TypeError: If model is not a PyTorch module
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model)}")

    assert not model.training, "Model should be in eval mode"


def assert_model_in_train_mode(model: torch.nn.Module) -> None:
    """Assert that model is in training mode.

    Args:
        model: Model to check

    Raises:
        AssertionError: If model is not in training mode
        TypeError: If model is not a PyTorch module
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model)}")

    assert model.training, "Model should be in training mode"
