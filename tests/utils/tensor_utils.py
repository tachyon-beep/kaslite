"""Tensor testing utilities.

This module provides utilities for creating and validating PyTorch tensors
in tests, with comprehensive assertion functions and helper utilities.
"""

import torch


def create_test_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    requires_grad: bool = False,
    seed: int | None = None,
) -> torch.Tensor:
    """Create a test tensor with specified properties.

    Args:
        shape: Shape of the tensor to create
        dtype: Data type of the tensor
        device: Device to place the tensor on
        requires_grad: Whether tensor should require gradients
        seed: Optional seed for reproducible random values

    Returns:
        Test tensor with specified properties

    Raises:
        ValueError: If shape contains non-positive dimensions
        RuntimeError: If device is not available
    """
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"All dimensions must be positive, got {shape}")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")

    if seed is not None:
        torch.manual_seed(seed)

    return torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)


def create_test_input(
    batch_size: int = 4, input_dim: int = 2, seed: int | None = None
) -> torch.Tensor:
    """Create test input tensor.

    Args:
        batch_size: Number of samples in the batch
        input_dim: Dimensionality of each input sample
        seed: Optional seed for reproducible values

    Returns:
        Input tensor of shape (batch_size, input_dim)

    Raises:
        ValueError: If batch_size or input_dim are non-positive
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")

    return create_test_tensor((batch_size, input_dim), seed=seed)


def create_test_target(
    batch_size: int = 4, num_classes: int = 2, seed: int | None = None
) -> torch.Tensor:
    """Create test target tensor for classification.

    Args:
        batch_size: Number of samples in the batch
        num_classes: Number of classes in the classification problem
        seed: Optional seed for reproducible values

    Returns:
        Target tensor of shape (batch_size,) with integer class labels

    Raises:
        ValueError: If batch_size or num_classes are non-positive
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    if seed is not None:
        torch.manual_seed(seed)

    return torch.randint(0, num_classes, (batch_size,))


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    """Assert tensor has expected shape.

    Args:
        tensor: Tensor to check
        expected_shape: Expected shape tuple

    Raises:
        AssertionError: If tensor shape doesn't match expected shape
    """
    actual_shape = tuple(tensor.shape)
    assert actual_shape == expected_shape, f"Expected shape {expected_shape}, got {actual_shape}"


def assert_tensor_properties(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: torch.dtype | None = None,
    expected_device: str | None = None,
    requires_grad: bool | None = None,
) -> None:
    """Assert multiple tensor properties at once.

    Args:
        tensor: Tensor to check
        expected_shape: Expected shape (optional)
        expected_dtype: Expected data type (optional)
        expected_device: Expected device (optional)
        requires_grad: Expected gradient requirement (optional)

    Raises:
        AssertionError: If any property doesn't match expectations
    """
    if expected_shape is not None:
        assert_tensor_shape(tensor, expected_shape)

    if expected_dtype is not None:
        assert (
            tensor.dtype == expected_dtype
        ), f"Expected dtype {expected_dtype}, got {tensor.dtype}"

    if expected_device is not None:
        device_str = str(tensor.device)
        assert device_str == expected_device, f"Expected device {expected_device}, got {device_str}"

    if requires_grad is not None:
        assert (
            tensor.requires_grad == requires_grad
        ), f"Expected requires_grad {requires_grad}, got {tensor.requires_grad}"


def assert_tensors_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> None:
    """Assert two tensors are close within tolerance.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        rtol: Relative tolerance parameter
        atol: Absolute tolerance parameter
        equal_nan: Whether to treat NaN values as equal

    Raises:
        AssertionError: If tensors are not close within tolerance
        ValueError: If tensors have different shapes
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Tensors must have same shape: {tensor1.shape} vs {tensor2.shape}")

    close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if not close:
        max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
        assert False, f"Tensors not close within tolerance: max diff = {max_diff:.2e}"
