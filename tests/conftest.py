"""
Shared test configuration and fixtures.

This module provides common test fixtures and utilities used across
all test modules in the morphogenetic engine test suite.
"""

import argparse
from typing import Dict, Optional
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from mlflow.entities.model_registry import ModelVersion


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


@pytest.fixture
def mock_mlflow_client():
    """Fixture providing a mocked MLflow client."""
    with patch("mlflow.tracking.MlflowClient") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def sample_model_version():
    """Fixture providing a sample ModelVersion object."""
    version = Mock(spec=ModelVersion)
    version.version = "1"
    version.current_stage = "Staging"
    version.run_id = "test_run_123"
    version.creation_timestamp = 1609459200000  # 2021-01-01 00:00:00
    return version


@pytest.fixture
def register_args():
    """Fixture providing arguments for model registration."""
    return argparse.Namespace(
        model_name="TestModel",
        run_id="test_run_123",
        description="Test description",
        val_acc=0.85,
        train_loss=0.23,
        seeds_activated=True,
        tags=["env=test", "version=1.0"],
    )


@pytest.fixture
def register_args_minimal():
    """Fixture providing minimal arguments for model registration."""
    return argparse.Namespace(
        model_name="TestModel",
        run_id="test_run_456",
        description=None,
        val_acc=None,
        train_loss=None,
        seeds_activated=None,
        tags=None,
    )


@pytest.fixture
def promote_args():
    """Fixture providing arguments for model promotion."""
    return argparse.Namespace(
        model_name="TestModel", stage="Production", version="3", archive_existing=True
    )


@pytest.fixture
def list_args():
    """Fixture providing arguments for model listing."""
    return argparse.Namespace(model_name="TestModel", stage=None)


@pytest.fixture
def list_args_with_stage():
    """Fixture providing arguments for model listing with stage filter."""
    return argparse.Namespace(model_name="TestModel", stage="Production")


@pytest.fixture
def best_args():
    """Fixture providing arguments for best model retrieval."""
    return argparse.Namespace(
        model_name="TestModel", metric="val_acc", stage=None, higher_is_better=True
    )


@pytest.fixture
def production_args():
    """Fixture providing arguments for production model retrieval."""
    return argparse.Namespace(model_name="TestModel")


class MockModelVersion:
    """Helper class to create realistic ModelVersion mocks."""

    @staticmethod
    def create(
        version: str = "1",
        stage: str = "Staging",
        run_id: str = "test_run_123",
        timestamp: Optional[int] = None,
        aliases: Optional[list[str]] = None,
    ) -> Mock:
        """Create a mock ModelVersion with specified attributes."""
        mock_version = Mock(spec=ModelVersion)
        mock_version.version = version
        mock_version.current_stage = stage
        mock_version.run_id = run_id
        mock_version.creation_timestamp = timestamp or 1609459200000

        # Support both old stage-based and new alias-based API
        if aliases is None:
            # Convert stage to alias for modern API compatibility
            if stage and stage != "None":
                aliases = [stage]
            else:
                aliases = []
        mock_version.aliases = aliases

        return mock_version


def assert_output_contains(output: str, expected_items: list[str]) -> None:
    """Helper function to assert that output contains all expected items."""
    for item in expected_items:
        assert item in output, f"Expected '{item}' not found in output: {output}"


def create_mock_run(metrics: Dict[str, float]) -> Mock:
    """Create a mock MLflow run with specified metrics."""
    mock_run = Mock()
    mock_run.data.metrics = metrics
    return mock_run
