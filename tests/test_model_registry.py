"""
Unit tests for the MLflow Model Registry integration.

Tests the ModelRegistry class and its integration with MLflow for model
lifecycle management, versioning, and metadata handling.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from morphogenetic_engine.model_registry import ModelRegistry


class TestModelRegistry:
    """Test suite for ModelRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_name = "TestKasminaModel"
        self.registry = ModelRegistry(self.model_name)

        # Sample metrics for testing
        self.sample_metrics = {"val_acc": 0.85, "train_loss": 0.23, "seeds_activated": True}

        # Sample tags for testing
        self.sample_tags = {"problem_type": "spirals", "device": "cpu", "training_mode": "test"}

    def _mock_registry_client(self, mock_client_class):
        """Helper method to properly mock the registry's MLflow client."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        self.registry.client = mock_client
        return mock_client

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    @patch("morphogenetic_engine.model_registry.mlflow.register_model")
    def test_register_best_model_success(self, mock_register, mock_client_class):
        """Test successful model registration."""
        # Setup mocks
        mock_client = self._mock_registry_client(mock_client_class)

        mock_model_version = Mock()
        mock_model_version.version = "1"
        mock_register.return_value = mock_model_version

        # Test registration
        result = self.registry.register_best_model(
            run_id="test_run_123",
            metrics=self.sample_metrics,
            description="Test model",
            tags=self.sample_tags,
        )

        # Assertions
        assert result == mock_model_version
        mock_register.assert_called_once_with(
            model_uri="runs:/test_run_123/model", name=self.model_name, tags=self.sample_tags
        )
        mock_client.update_model_version.assert_called_once_with(
            name=self.model_name, version="1", description="Test model"
        )

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    @patch("morphogenetic_engine.model_registry.mlflow.register_model")
    def test_register_best_model_auto_description(self, mock_register, mock_client_class):
        """Test model registration with auto-generated description."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        self.registry.client = mock_client

        mock_model_version = Mock()
        mock_model_version.version = "2"
        mock_register.return_value = mock_model_version

        # Test registration without description
        result = self.registry.register_best_model(
            run_id="test_run_456", metrics=self.sample_metrics
        )

        # Check that auto-description was generated
        call_args = mock_client.update_model_version.call_args[1]
        description = call_args["description"]
        assert "Val Acc: 0.8500" in description
        assert "Train Loss: 0.2300" in description
        assert "Seeds Activated: True" in description

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    @patch("morphogenetic_engine.model_registry.mlflow.register_model")
    def test_register_best_model_failure(self, mock_register, mock_client_class):
        """Test model registration failure handling."""
        # Setup mocks to raise exception
        mock_register.side_effect = Exception("Registration failed")

        # Test registration failure
        result = self.registry.register_best_model(
            run_id="test_run_fail", metrics=self.sample_metrics
        )

        # Should return None on failure
        assert result is None

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_get_best_model_version_by_accuracy(self, mock_client_class):
        """Test finding best model version by validation accuracy."""
        # Setup mocks
        mock_client = self._mock_registry_client(mock_client_class)

        # Create mock model versions
        version1 = Mock()
        version1.version = "1"
        version1.run_id = "run_1"
        version1.current_stage = "None"

        version2 = Mock()
        version2.version = "2"
        version2.run_id = "run_2"
        version2.current_stage = "Staging"

        mock_client.search_model_versions.return_value = [version1, version2]

        # Mock run data with different accuracies
        run1 = Mock()
        run1.data.metrics = {"val_acc": 0.75}

        run2 = Mock()
        run2.data.metrics = {"val_acc": 0.90}

        mock_client.get_run.side_effect = [run1, run2]

        # Test finding best model
        result = self.registry.get_best_model_version(metric_name="val_acc")

        # Should return version 2 (higher accuracy)
        assert result == version2

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_get_best_model_version_with_stage_filter(self, mock_client_class):
        """Test finding best model with stage filtering."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        self.registry.client = mock_client

        # Create mock model versions
        version1 = Mock()
        version1.version = "1"
        version1.run_id = "run_1"
        version1.current_stage = "Production"

        version2 = Mock()
        version2.version = "2"
        version2.run_id = "run_2"
        version2.current_stage = "Staging"

        mock_client.search_model_versions.return_value = [version1, version2]

        # Mock run data
        run1 = Mock()
        run1.data.metrics = {"val_acc": 0.85}

        mock_client.get_run.return_value = run1

        # Test with stage filter
        result = self.registry.get_best_model_version(stage="Production")

        # Should only consider Production models
        assert result == version1

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_get_best_model_version_no_versions(self, mock_client_class):
        """Test behavior when no model versions exist."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.search_model_versions.return_value = []

        # Test with no versions
        result = self.registry.get_best_model_version()

        # Should return None
        assert result is None

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_promote_model_success(self, mock_client_class):
        """Test successful model promotion."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        self.registry.client = mock_client

        # Mock search_model_versions to return empty list (no existing versions to archive)
        mock_client.search_model_versions.return_value = []

        # Test promotion
        result = self.registry.promote_model(version="3", stage="Production")

        # Assertions
        assert result is True
        mock_client.transition_model_version_stage.assert_called_with(
            name=self.model_name, version="3", stage="Production"
        )

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_promote_model_with_archiving(self, mock_client_class):
        """Test model promotion with archiving of existing models."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        self.registry.client = mock_client

        # Create existing production model
        existing_version = Mock()
        existing_version.version = "2"
        existing_version.current_stage = "Production"

        mock_client.search_model_versions.return_value = [existing_version]

        # Test promotion with archiving
        result = self.registry.promote_model(version="3", stage="Production", archive_existing=True)

        # Should archive existing version and promote new one
        assert result is True
        assert mock_client.transition_model_version_stage.call_count == 2

        # Check archiving call
        calls = mock_client.transition_model_version_stage.call_args_list
        archive_call = calls[0]
        assert archive_call[1]["stage"] == "Archived"
        assert archive_call[1]["version"] == "2"

        # Check promotion call
        promote_call = calls[1]
        assert promote_call[1]["stage"] == "Production"
        assert promote_call[1]["version"] == "3"

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_promote_best_model_auto_selection(self, mock_client_class):
        """Test promoting best model when version not specified."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        self.registry.client = mock_client

        # Mock search_model_versions to return empty list (no existing versions to archive)
        mock_client.search_model_versions.return_value = []

        # Mock the get_best_model_version call
        with patch.object(self.registry, "get_best_model_version") as mock_get_best:
            mock_best_version = Mock()
            mock_best_version.version = "4"
            mock_get_best.return_value = mock_best_version

            # Test promotion without specifying version
            result = self.registry.promote_model(stage="Staging")

            # Should find best version and promote it
            assert result is True
            mock_get_best.assert_called_once_with(self.model_name)
            mock_client.transition_model_version_stage.assert_called_with(
                name=self.model_name, version="4", stage="Staging"
            )

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_list_model_versions(self, mock_client_class):
        """Test listing model versions."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        self.registry.client = mock_client

        # Create mock versions
        version1 = Mock()
        version1.version = "1"
        version2 = Mock()
        version2.version = "2"
        version3 = Mock()
        version3.version = "3"

        mock_client.search_model_versions.return_value = [version1, version3, version2]

        # Test listing
        result = self.registry.list_model_versions()

        # Should return sorted versions (descending)
        assert len(result) == 3
        assert result[0].version == "3"
        assert result[1].version == "2"
        assert result[2].version == "1"

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_list_model_versions_with_stage_filter(self, mock_client_class):
        """Test listing model versions with stage filtering."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        self.registry.client = mock_client

        # Create mock versions with different stages
        version1 = Mock()
        version1.version = "1"
        version1.current_stage = "Production"

        version2 = Mock()
        version2.version = "2"
        version2.current_stage = "Staging"

        mock_client.search_model_versions.return_value = [version1, version2]

        # Test with stage filter
        result = self.registry.list_model_versions(stage="Production")

        # Should only return Production models
        assert len(result) == 1
        assert result[0].version == "1"

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_get_production_model_uri(self, mock_client_class):
        """Test getting production model URI."""
        # Mock list_model_versions to return production model
        with patch.object(self.registry, "list_model_versions") as mock_list:
            mock_version = Mock()
            mock_version.version = "5"
            mock_list.return_value = [mock_version]

            # Test getting production URI
            result = self.registry.get_production_model_uri()

            # Should return correct URI format
            expected_uri = f"models:/{self.model_name}/5"
            assert result == expected_uri
            mock_list.assert_called_once_with(self.model_name, stage="Production")

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_get_production_model_uri_no_production_model(self, mock_client_class):
        """Test getting production model URI when none exists."""
        # Mock list_model_versions to return empty list
        with patch.object(self.registry, "list_model_versions") as mock_list:
            mock_list.return_value = []

            # Test getting production URI
            result = self.registry.get_production_model_uri()

            # Should return None
            assert result is None

    def test_model_registry_initialization(self):
        """Test ModelRegistry initialization."""
        # Test default model name
        default_registry = ModelRegistry()
        assert default_registry.model_name == "KasminaModel"

        # Test custom model name
        custom_registry = ModelRegistry("CustomModel")
        assert custom_registry.model_name == "CustomModel"

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_promote_model_failure(self, mock_client_class):
        """Test model promotion failure handling."""
        # Setup mocks to raise exception
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.transition_model_version_stage.side_effect = Exception("Promotion failed")

        # Test promotion failure
        result = self.registry.promote_model(version="1", stage="Production")

        # Should return False on failure
        assert result is False

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_promote_model_no_best_version_found(self, mock_client_class):
        """Test promotion when no best version can be found."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock get_best_model_version to return None
        with patch.object(self.registry, "get_best_model_version") as mock_get_best:
            mock_get_best.return_value = None

            # Test promotion without version when no best version exists
            result = self.registry.promote_model(stage="Production")

            # Should return False
            assert result is False
            # Should not attempt promotion
            mock_client.transition_model_version_stage.assert_not_called()

    @patch("morphogenetic_engine.model_registry.MlflowClient")
    def test_get_best_model_version_lower_is_better(self, mock_client_class):
        """Test finding best model version with lower_is_better metric."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        self.registry.client = mock_client

        # Create mock model versions
        version1 = Mock()
        version1.version = "1"
        version1.run_id = "run_1"

        version2 = Mock()
        version2.version = "2"
        version2.run_id = "run_2"

        mock_client.search_model_versions.return_value = [version1, version2]

        # Mock run data with different loss values
        run1 = Mock()
        run1.data.metrics = {"train_loss": 0.25}

        run2 = Mock()
        run2.data.metrics = {"train_loss": 0.15}

        mock_client.get_run.side_effect = [run1, run2]

        # Test finding best model with lower_is_better=True
        result = self.registry.get_best_model_version(
            metric_name="train_loss", higher_is_better=False
        )

        # Should return version 2 (lower loss)
        assert result == version2
