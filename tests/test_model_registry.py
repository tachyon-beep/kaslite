"""
Unit and integration tests for the MLflow Model Registry integration.

Tests the ModelRegistry class and its integration with MLflow for model
lifecycle management, versioning, and metadata handling.
"""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import tempfile
from unittest.mock import Mock, patch

import mlflow
import pytest
from hypothesis import given
from hypothesis import strategies as st

from morphogenetic_engine.model_registry import ModelRegistry

# =============================================================================
# Test Constants and Data Factories
# =============================================================================


class TestConstants:
    """Constants for test data to avoid magic numbers."""

    HIGH_ACCURACY = 0.90
    LOW_ACCURACY = 0.75
    MEDIUM_ACCURACY = 0.82

    HIGH_LOSS = 0.25
    LOW_LOSS = 0.15
    MEDIUM_LOSS = 0.20

    DEFAULT_MODEL_NAME = "TestKasminaModel"
    SAMPLE_RUN_IDS = ["run_123", "run_456", "run_789"]

    SAMPLE_METRICS = {"val_acc": HIGH_ACCURACY, "train_loss": LOW_LOSS, "seeds_activated": True}

    SAMPLE_TAGS = {"problem_type": "spirals", "device": "cpu", "training_mode": "test"}


@pytest.fixture
def model_registry() -> ModelRegistry:
    """Create a ModelRegistry instance for testing."""
    return ModelRegistry(TestConstants.DEFAULT_MODEL_NAME)


@pytest.fixture
def mock_mlflow_client(mocker):
    """Centralized MLflow client mock that patches at the service boundary."""
    mock_client = mocker.Mock()
    mocker.patch("morphogenetic_engine.model_registry.MlflowClient", return_value=mock_client)
    return mock_client


@pytest.fixture
def sample_model_versions():
    """Create reusable test data for model versions."""
    versions = []
    for i, (stage, run_id) in enumerate(
        [("None", "run_1"), ("Staging", "run_2"), ("Production", "run_3")], 1
    ):
        version = Mock()
        version.version = str(i)
        version.run_id = run_id
        version.current_stage = stage
        versions.append(version)
    return versions


@pytest.fixture
def sample_run_metrics():
    """Create sample run metrics for testing."""
    return {
        "run_1": {"val_acc": TestConstants.LOW_ACCURACY},
        "run_2": {"val_acc": TestConstants.HIGH_ACCURACY},
        "run_3": {"val_acc": TestConstants.MEDIUM_ACCURACY},
    }


# =============================================================================
# Unit Tests
# =============================================================================


@pytest.mark.unit
class TestModelRegistryUnit:
    """Unit tests for ModelRegistry class."""

    def test_model_registry_initialization_default(self) -> None:
        """Test ModelRegistry initialization with default model name."""
        registry = ModelRegistry()
        assert registry.model_name == "KasminaModel", "Default model name should be 'KasminaModel'"

    def test_model_registry_initialization_custom(self) -> None:
        """Test ModelRegistry initialization with custom model name."""
        custom_name = "CustomTestModel"
        registry = ModelRegistry(custom_name)
        assert registry.model_name == custom_name, f"Model name should be '{custom_name}'"

    @pytest.mark.parametrize(
        "run_id,expected_uri",
        [
            ("test_run_123", "runs:/test_run_123/model"),
            ("run_456", "runs:/run_456/model"),
            ("special-run-789", "runs:/special-run-789/model"),
        ],
    )
    def test_model_uri_formatting(
        self, model_registry: ModelRegistry, mock_mlflow_client, run_id: str, expected_uri: str
    ) -> None:
        """Test that model URIs are formatted correctly."""
        # Setup mock
        mock_version = Mock()
        mock_version.version = "1"

        with patch(
            "morphogenetic_engine.model_registry.mlflow.register_model", return_value=mock_version
        ) as mock_register:

            model_registry.register_best_model(run_id=run_id, metrics=TestConstants.SAMPLE_METRICS)

            # Verify URI format
            mock_register.assert_called_once()
            call_args = mock_register.call_args[1]
            assert call_args["model_uri"] == expected_uri, f"Model URI should be '{expected_uri}'"

    def test_register_best_model_success(
        self, model_registry: ModelRegistry, mock_mlflow_client
    ) -> None:
        """Test successful model registration with proper mock setup."""
        # Setup mocks
        mock_version = Mock()
        mock_version.version = "1"

        with patch(
            "morphogenetic_engine.model_registry.mlflow.register_model", return_value=mock_version
        ) as mock_register:

            # Test registration
            result = model_registry.register_best_model(
                run_id=TestConstants.SAMPLE_RUN_IDS[0],
                metrics=TestConstants.SAMPLE_METRICS,
                description="Test model description",
                tags=TestConstants.SAMPLE_TAGS,
            )

            # Assertions with descriptive messages
            assert result == mock_version, "Should return the registered model version"
            mock_register.assert_called_once_with(
                model_uri=f"runs:/{TestConstants.SAMPLE_RUN_IDS[0]}/model",
                name=TestConstants.DEFAULT_MODEL_NAME,
                tags=TestConstants.SAMPLE_TAGS,
            )
            mock_mlflow_client.update_model_version.assert_called_once_with(
                name=TestConstants.DEFAULT_MODEL_NAME,
                version="1",
                description="Test model description",
            )

    def test_register_best_model_auto_description(
        self, model_registry: ModelRegistry, mock_mlflow_client
    ) -> None:
        """Test model registration with auto-generated description."""
        mock_version = Mock()
        mock_version.version = "2"

        with patch(
            "morphogenetic_engine.model_registry.mlflow.register_model", return_value=mock_version
        ):

            # Test registration without explicit description
            result = model_registry.register_best_model(
                run_id=TestConstants.SAMPLE_RUN_IDS[1], metrics=TestConstants.SAMPLE_METRICS
            )

            # Verify auto-description was generated correctly
            call_args = mock_mlflow_client.update_model_version.call_args[1]
            description = call_args["description"]

            assert "Val Acc: 0.9000" in description, "Description should include formatted accuracy"
            assert "Train Loss: 0.1500" in description, "Description should include formatted loss"
            assert "Seeds Activated: True" in description, "Description should include seeds status"
            assert result == mock_version, "Should return the registered model version"

    def test_register_best_model_failure_handling(
        self, model_registry: ModelRegistry, mock_mlflow_client
    ) -> None:
        """Test model registration failure handling."""
        with patch(
            "morphogenetic_engine.model_registry.mlflow.register_model",
            side_effect=Exception("Registration failed"),
        ):

            result = model_registry.register_best_model(
                run_id="test_run_fail", metrics=TestConstants.SAMPLE_METRICS
            )

            assert result is None, "Should return None on registration failure"

    @pytest.mark.parametrize(
        "metric_name,metric_values,higher_is_better,expected_version_idx",
        [
            (
                "val_acc",
                [
                    TestConstants.LOW_ACCURACY,
                    TestConstants.HIGH_ACCURACY,
                    TestConstants.MEDIUM_ACCURACY,
                ],
                True,
                1,
            ),
            (
                "train_loss",
                [TestConstants.HIGH_LOSS, TestConstants.LOW_LOSS, TestConstants.MEDIUM_LOSS],
                False,
                1,
            ),
            ("f1_score", [0.60, 0.85, 0.75], True, 1),
            ("mse", [0.10, 0.05, 0.08], False, 1),
        ],
    )
    def test_get_best_model_version_optimization(
        self,
        model_registry: ModelRegistry,
        mock_mlflow_client,
        sample_model_versions,
        metric_name: str,
        metric_values: list[float],
        higher_is_better: bool,
        expected_version_idx: int,
    ) -> None:
        """Test best model selection with different optimization criteria."""
        # Setup mock model versions
        mock_mlflow_client.search_model_versions.return_value = sample_model_versions

        # Setup mock runs with different metric values
        mock_runs = []
        for value in metric_values:
            run = Mock()
            run.data.metrics = {metric_name: value}
            mock_runs.append(run)

        mock_mlflow_client.get_run.side_effect = mock_runs

        # Test best model selection
        result = model_registry.get_best_model_version(
            metric_name=metric_name, higher_is_better=higher_is_better
        )

        expected_version = sample_model_versions[expected_version_idx]
        assert result == expected_version, (
            f"Expected version {expected_version.version} with {metric_name}="
            f"{metric_values[expected_version_idx]} (higher_is_better={higher_is_better})"
        )

    def test_get_best_model_version_with_stage_filter(
        self, model_registry: ModelRegistry, mock_mlflow_client, sample_model_versions
    ) -> None:
        """Test finding best model with stage filtering."""
        # Setup mocks
        mock_mlflow_client.search_model_versions.return_value = sample_model_versions

        # Mock run data for production model only
        production_run = Mock()
        production_run.data.metrics = {"val_acc": TestConstants.MEDIUM_ACCURACY}
        mock_mlflow_client.get_run.return_value = production_run

        # Test with stage filter
        result = model_registry.get_best_model_version(stage="Production")

        production_version = sample_model_versions[2]  # Third version is Production
        assert result == production_version, "Should return the Production model version"

    def test_get_best_model_version_no_versions(
        self, model_registry: ModelRegistry, mock_mlflow_client
    ) -> None:
        """Test behavior when no model versions exist."""
        mock_mlflow_client.search_model_versions.return_value = []

        result = model_registry.get_best_model_version()

        assert result is None, "Should return None when no model versions exist"

    @pytest.mark.parametrize(
        "archive_existing,expected_call_count",
        [
            (True, 2),  # Archive existing + promote new
            (False, 1),  # Only promote new
        ],
    )
    def test_promote_model_archiving_behavior(
        self,
        model_registry: ModelRegistry,
        mock_mlflow_client,
        archive_existing: bool,
        expected_call_count: int,
    ) -> None:
        """Test model promotion with and without archiving existing models."""
        # Setup existing production model
        existing_version = Mock()
        existing_version.version = "2"
        existing_version.current_stage = "Production"

        if archive_existing:
            mock_mlflow_client.search_model_versions.return_value = [existing_version]
        else:
            mock_mlflow_client.search_model_versions.return_value = []

        # Test promotion
        result = model_registry.promote_model(
            version="3", stage="Production", archive_existing=archive_existing
        )

        assert result is True, "Promotion should succeed"
        assert (
            mock_mlflow_client.transition_model_version_stage.call_count == expected_call_count
        ), f"Expected {expected_call_count} stage transitions"

        if archive_existing:
            # Verify archiving and promotion calls
            calls = mock_mlflow_client.transition_model_version_stage.call_args_list

            # First call should archive existing version
            archive_call = calls[0]
            assert (
                archive_call[1]["stage"] == "Archived"
            ), "First call should archive existing model"
            assert archive_call[1]["version"] == "2", "Should archive version 2"

            # Second call should promote new version
            promote_call = calls[1]
            assert (
                promote_call[1]["stage"] == "Production"
            ), "Second call should promote to Production"
            assert promote_call[1]["version"] == "3", "Should promote version 3"

    def test_promote_model_auto_selection(
        self, model_registry: ModelRegistry, mock_mlflow_client
    ) -> None:
        """Test promoting best model when version not specified."""
        mock_mlflow_client.search_model_versions.return_value = []

        # Mock the get_best_model_version call
        mock_best_version = Mock()
        mock_best_version.version = "4"

        with patch.object(
            model_registry, "get_best_model_version", return_value=mock_best_version
        ) as mock_get_best:

            result = model_registry.promote_model(stage="Staging")

            assert result is True, "Auto-promotion should succeed"
            mock_get_best.assert_called_once_with(TestConstants.DEFAULT_MODEL_NAME)
            mock_mlflow_client.transition_model_version_stage.assert_called_with(
                name=TestConstants.DEFAULT_MODEL_NAME, version="4", stage="Staging"
            )

    def test_promote_model_failure_handling(
        self, model_registry: ModelRegistry, mock_mlflow_client
    ) -> None:
        """Test model promotion failure handling."""
        mock_mlflow_client.transition_model_version_stage.side_effect = Exception(
            "Promotion failed"
        )

        result = model_registry.promote_model(version="1", stage="Production")

        assert result is False, "Should return False on promotion failure"

    def test_promote_model_no_best_version_found(
        self, model_registry: ModelRegistry, mock_mlflow_client
    ) -> None:
        """Test promotion when no best version can be found."""
        with patch.object(model_registry, "get_best_model_version", return_value=None):

            result = model_registry.promote_model(stage="Production")

            assert result is False, "Should return False when no best version found"
            mock_mlflow_client.transition_model_version_stage.assert_not_called()

    def test_list_model_versions_sorting(
        self, model_registry: ModelRegistry, mock_mlflow_client
    ) -> None:
        """Test that model versions are returned in descending order."""
        # Create mock versions in random order
        version1 = Mock()
        version1.version = "1"
        version3 = Mock()
        version3.version = "3"
        version2 = Mock()
        version2.version = "2"

        # Return them in non-sorted order
        mock_mlflow_client.search_model_versions.return_value = [version1, version3, version2]

        result = model_registry.list_model_versions()

        assert len(result) == 3, "Should return all 3 versions"
        assert result[0].version == "3", "First version should be '3' (highest)"
        assert result[1].version == "2", "Second version should be '2'"
        assert result[2].version == "1", "Third version should be '1' (lowest)"

    def test_list_model_versions_with_stage_filter(
        self, model_registry: ModelRegistry, mock_mlflow_client, sample_model_versions
    ) -> None:
        """Test listing model versions with stage filtering."""
        mock_mlflow_client.search_model_versions.return_value = sample_model_versions

        result = model_registry.list_model_versions(stage="Production")

        assert len(result) == 1, "Should return only Production models"
        assert (
            result[0].current_stage == "Production"
        ), "Returned model should be in Production stage"

    def test_get_production_model_uri_success(self, model_registry: ModelRegistry) -> None:
        """Test getting production model URI when model exists."""
        mock_version = Mock()
        mock_version.version = "5"

        with patch.object(
            model_registry, "list_model_versions", return_value=[mock_version]
        ) as mock_list:

            result = model_registry.get_production_model_uri()

            expected_uri = f"models:/{TestConstants.DEFAULT_MODEL_NAME}/5"
            assert result == expected_uri, f"Should return URI: {expected_uri}"
            mock_list.assert_called_once_with(TestConstants.DEFAULT_MODEL_NAME, stage="Production")

    def test_get_production_model_uri_no_model(self, model_registry: ModelRegistry) -> None:
        """Test getting production model URI when no production model exists."""
        with patch.object(model_registry, "list_model_versions", return_value=[]):

            result = model_registry.get_production_model_uri()

            assert result is None, "Should return None when no production model exists"


# =============================================================================
# Property-Based Tests
# =============================================================================


@pytest.mark.property
class TestModelRegistryPropertyBased:
    """Property-based tests using Hypothesis for edge case discovery."""

    @given(
        metrics=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10,
        )
    )
    def test_register_model_with_arbitrary_metrics(
        self, model_registry: ModelRegistry, mock_mlflow_client, metrics: dict[str, float]
    ) -> None:
        """Test model registration with generated metric combinations."""
        # Convert values to ensure they are Python floats
        converted_metrics = {k: float(v) for k, v in metrics.items()}

        mock_version = Mock()
        mock_version.version = "1"

        with patch(
            "morphogenetic_engine.model_registry.mlflow.register_model", return_value=mock_version
        ):

            result = model_registry.register_best_model(
                run_id="test_run", metrics=converted_metrics
            )

            # Should handle any valid metrics dictionary
            assert (
                result == mock_version
            ), "Should successfully register model with any valid metrics"

    @given(model_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))
    def test_model_registry_initialization_arbitrary_names(self, model_name: str) -> None:
        """Test ModelRegistry initialization with generated model names."""
        registry = ModelRegistry(model_name)
        assert registry.model_name == model_name, f"Model name should be '{model_name}'"


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestModelRegistryIntegration:
    """Integration tests with real MLflow using temporary tracking URI."""

    @pytest.fixture(autouse=True)
    def setup_temp_mlflow(self, tmp_path):
        """Setup temporary MLflow tracking for integration tests."""
        # Set temporary tracking URI
        original_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

        yield

        # Restore original URI
        mlflow.set_tracking_uri(original_uri)

    def test_register_and_promote_workflow_integration(self, tmp_path) -> None:
        """Test full model registration and promotion workflow with real MLflow."""
        registry = ModelRegistry("IntegrationTestModel")

        # Start an MLflow run to create a model artifact
        with mlflow.start_run() as run:
            # Log a simple model (dummy data)
            import torch  # pylint: disable=import-outside-toplevel
            import torch.nn as nn  # pylint: disable=import-outside-toplevel

            model = nn.Linear(10, 1)
            mlflow.pytorch.log_model(model, "model")

            # Log some metrics
            mlflow.log_metrics(
                {"val_acc": TestConstants.HIGH_ACCURACY, "train_loss": TestConstants.LOW_LOSS}
            )

            run_id = run.info.run_id

        # Test registration
        metrics = {"val_acc": TestConstants.HIGH_ACCURACY, "train_loss": TestConstants.LOW_LOSS}
        version = registry.register_best_model(run_id=run_id, metrics=metrics)

        assert version is not None, "Model registration should succeed"
        assert version.version == "1", "First registered model should be version 1"

        # Test promotion
        promote_result = registry.promote_model(version="1", stage="Staging")
        assert promote_result is True, "Model promotion should succeed"

        # Test listing and URI generation
        versions = registry.list_model_versions(stage="Staging")
        assert len(versions) == 1, "Should have one staging model"
        assert versions[0].current_stage == "Staging", "Model should be in Staging stage"

        # Promote to production and test URI
        promote_result = registry.promote_model(version="1", stage="Production")
        assert promote_result is True, "Production promotion should succeed"

        production_uri = registry.get_production_model_uri()
        expected_uri = "models:/IntegrationTestModel/1"
        assert production_uri == expected_uri, f"Production URI should be {expected_uri}"


# =============================================================================
# Error Boundary Tests
# =============================================================================


@pytest.mark.unit
class TestModelRegistryErrorBoundaries:
    """Tests for error handling and edge cases."""

    def test_register_model_with_malformed_metrics(
        self, model_registry: ModelRegistry, mock_mlflow_client
    ) -> None:
        """Test handling of malformed metrics."""
        malformed_metrics = {
            "val_acc": float("inf"),
            "train_loss": float("nan"),
            "invalid": "not_a_number",
        }

        mock_version = Mock()
        mock_version.version = "1"

        with patch(
            "morphogenetic_engine.model_registry.mlflow.register_model", return_value=mock_version
        ):

            # Should handle malformed metrics gracefully
            result = model_registry.register_best_model(
                run_id="test_run", metrics=malformed_metrics
            )

            # Registration should still work, but description generation should be robust
            assert result == mock_version, "Should handle malformed metrics gracefully"

    def test_get_best_model_version_with_missing_metrics(
        self, model_registry: ModelRegistry, mock_mlflow_client, sample_model_versions
    ) -> None:
        """Test best model selection when some runs are missing the target metric."""
        mock_mlflow_client.search_model_versions.return_value = sample_model_versions

        # Mock runs where some are missing the target metric
        run_with_metric = Mock()
        run_with_metric.data.metrics = {"val_acc": TestConstants.HIGH_ACCURACY}

        run_without_metric = Mock()
        run_without_metric.data.metrics = {"other_metric": 0.5}

        mock_mlflow_client.get_run.side_effect = [
            run_without_metric,
            run_with_metric,
            run_without_metric,
        ]

        result = model_registry.get_best_model_version(metric_name="val_acc")

        # Should return the version with the metric
        assert result == sample_model_versions[1], "Should return version with available metric"

    def test_mlflow_client_initialization_failure(self, mocker) -> None:
        """Test handling of MLflow client initialization failure."""
        # Mock MlflowClient to raise exception during initialization
        mocker.patch(
            "morphogenetic_engine.model_registry.MlflowClient",
            side_effect=Exception("MLflow connection failed"),
        )

        # Should handle initialization failure gracefully
        with pytest.raises(Exception, match="MLflow connection failed"):
            ModelRegistry("TestModel")

    def test_concurrent_model_registration_simulation(
        self, model_registry: ModelRegistry, mock_mlflow_client
    ) -> None:
        """Test simulation of concurrent model registration scenarios."""
        # Simulate concurrent registration by having register_model succeed then fail
        mock_version = Mock()
        mock_version.version = "1"

        with patch("morphogenetic_engine.model_registry.mlflow.register_model") as mock_register:
            # First call succeeds, second fails (simulating race condition)
            mock_register.side_effect = [mock_version, Exception("Model name already exists")]

            # First registration should succeed
            result1 = model_registry.register_best_model(
                run_id="run_1", metrics=TestConstants.SAMPLE_METRICS
            )
            assert result1 == mock_version, "First registration should succeed"

            # Second registration should fail gracefully
            result2 = model_registry.register_best_model(
                run_id="run_2", metrics=TestConstants.SAMPLE_METRICS
            )
            assert result2 is None, "Second registration should fail gracefully"


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.benchmark
class TestModelRegistryPerformance:
    """Performance tests for large datasets and operations."""

    def test_list_large_number_of_model_versions(
        self, model_registry: ModelRegistry, mock_mlflow_client, benchmark
    ) -> None:
        """Test performance with large number of model versions."""
        # Create 1000 mock model versions
        large_version_list = []
        for i in range(1000):
            version = Mock()
            version.version = str(i + 1)
            version.current_stage = "None"
            large_version_list.append(version)

        mock_mlflow_client.search_model_versions.return_value = large_version_list

        # Benchmark the sorting operation
        def list_versions():
            return model_registry.list_model_versions()

        result = benchmark(list_versions)

        assert len(result) == 1000, "Should handle large number of versions"
        assert result[0].version == "1000", "Should maintain correct sorting"
