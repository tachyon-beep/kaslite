"""Environment validation utilities for testing.

This module provides utilities for validating the test environment
and setting up logging configurations for tests.
"""

import torch
from unittest.mock import MagicMock


def validate_test_environment() -> None:
    """Validate that the test environment is properly configured.

    Raises:
        RuntimeError: If test environment is not properly configured
        ImportError: If required dependencies are missing
    """
    # Check that basic tensor operations work
    try:
        test_tensor = torch.randn(2, 2)
        _ = test_tensor + test_tensor
    except Exception as e:
        raise RuntimeError(f"Basic tensor operations failed: {e}") from e

    # Check random seed functionality
    try:
        torch.manual_seed(42)
        tensor1 = torch.randn(5)
        torch.manual_seed(42)
        tensor2 = torch.randn(5)
        if not torch.allclose(tensor1, tensor2):
            raise RuntimeError("Random seed functionality not working")
    except Exception as e:
        raise RuntimeError(f"Random seed validation failed: {e}") from e


def setup_test_logging() -> MagicMock:
    """Set up logging for tests.

    Returns:
        Mock logger configured for testing
    """
    from .mock_utils import create_mock_logger

    logger = create_mock_logger()
    # Configure any default behaviors
    logger.level = "DEBUG"
    return logger
