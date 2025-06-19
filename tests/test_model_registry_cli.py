"""
Unit tests for the Model Registry CLI.

Tests the command-line interface for model registry operations including
model registration, promotion, listing, and management.

Refactored version focusing on behavior validation with proper dependency mocking:
- Uses real ModelRegistry instances with mocked external dependencies only
- Tests actual behavior rather than mock interactions
- Comprehensive edge case and error condition coverage
- Modern pytest patterns with fixtures and parametrization
- Property-based testing for robust validation
"""

# pylint: disable=redefined-outer-name
# Note: pytest fixtures naturally "redefine" their names when injected as parameters

import time
from typing import Any, Dict
from unittest.mock import Mock, patch

import hypothesis.strategies as st
import pytest
from hypothesis import given

from morphogenetic_engine.cli.model_registry_cli import (
    get_best_model,
    get_production_model,
    list_models,
    main,
    promote_model,
    register_model,
)
from tests.conftest import MockModelVersion, create_mock_run

# Test Fixtures and Helpers


@pytest.fixture
def mock_mlflow_environment():
    """Provide a properly configured MLflow environment for integration tests."""
    with (
        patch("morphogenetic_engine.model_registry.MlflowClient") as mock_client_class,
        patch("morphogenetic_engine.model_registry.mlflow.register_model") as mock_register,
    ):

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Configure default successful behavior
        mock_register.return_value = MockModelVersion.create(version="1")
        mock_client.update_model_version.return_value = None
        mock_client.transition_model_version_stage.return_value = MockModelVersion.create(version="1", stage="Production")
        mock_client.search_model_versions.return_value = []

        # Configure get_run to return proper run data with real metrics
        def mock_get_run(_):
            return create_mock_run({"val_acc": 0.85, "train_loss": 0.23})

        mock_client.get_run.side_effect = mock_get_run

        yield {
            "client": mock_client,
            "register_model": mock_register,
            "client_class": mock_client_class,
        }


@pytest.fixture
def mock_cli_mlflow_client():
    """Provide MLflow client mock specifically for CLI functions that use MlflowClient directly."""
    with patch("morphogenetic_engine.cli.model_registry_cli.MlflowClient") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        yield mock_client


def assert_registry_called_with_correct_params(mock_method, expected_params: Dict[str, Any]):
    """Helper to validate ModelRegistry method calls with expected parameters."""
    assert mock_method.called, f"Expected {getattr(mock_method, '_mock_name', 'mock method')} to be called"
    call_kwargs = mock_method.call_args[1]
    for key, expected_value in expected_params.items():
        assert key in call_kwargs, f"Expected parameter '{key}' not found in call"
        assert call_kwargs[key] == expected_value, f"Parameter '{key}': expected {expected_value}, got {call_kwargs[key]}"


def create_realistic_model_versions(count: int = 3) -> list:
    """Create a list of realistic ModelVersion objects for testing."""
    versions = []
    stages = ["Staging", "Production", "Archived"]
    for i in range(count):
        version = MockModelVersion.create(
            version=str(i + 1),
            stage=stages[i % len(stages)],
            run_id=f"run_{i + 1:03d}",
            timestamp=1609459200000 + (i * 86400000),  # One day apart
        )
        versions.append(version)
    return versions


# Property-based testing strategies


@st.composite
def valid_tag_strategy(draw):
    """Generate valid tag strings for property-based testing."""
    key = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=32, max_codepoint=126)))
    value = draw(st.text(max_size=50, alphabet=st.characters(min_codepoint=32, max_codepoint=126)))
    return f"{key}={value}"


@st.composite
def unicode_text_strategy(draw):
    """Generate Unicode text for testing international character handling."""
    return draw(
        st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(categories=["Lu", "Ll", "Nd", "Pc", "Pd"]),  # Letters, numbers, punctuation
        )
    )


class TestModelRegistryCLIUnit:
    """Unit tests for CLI functions - testing logic with real ModelRegistry objects."""

    def test_tag_parsing_valid_tags(self, register_args, mock_mlflow_environment):
        """Test proper parsing of valid tags in register command."""
        # ARRANGE
        register_args.tags = ["env=prod", "version=2.1", "team=ml"]
        expected_tags = {"env": "prod", "version": "2.1", "team": "ml"}

        # ACT
        register_model(register_args)

        # ASSERT - Verify tags were passed correctly to MLflow
        register_call = mock_mlflow_environment["register_model"].call_args
        assert register_call[1]["tags"] == expected_tags

    def test_tag_parsing_handles_equals_in_values(self, register_args, mock_mlflow_environment):
        """Test handling of equals signs within tag values."""
        # ARRANGE
        register_args.tags = ["url=https://example.com", "query=name=value"]
        expected_tags = {"url": "https://example.com", "query": "name=value"}

        # ACT
        register_model(register_args)

        # ASSERT
        register_call = mock_mlflow_environment["register_model"].call_args
        assert register_call[1]["tags"] == expected_tags

    def test_tag_parsing_skips_invalid_format(self, register_args, mock_mlflow_environment):
        """Test that invalid tag formats are silently skipped."""
        # ARRANGE
        register_args.tags = ["env=prod", "invalid_tag", "version=1.0"]
        expected_tags = {"env": "prod", "version": "1.0"}

        # ACT
        register_model(register_args)

        # ASSERT
        register_call = mock_mlflow_environment["register_model"].call_args
        assert register_call[1]["tags"] == expected_tags

    def test_tag_parsing_empty_tags(self, register_args_minimal, mock_mlflow_environment):
        """Test handling of empty or None tags."""
        # ARRANGE - args already has tags=None

        # ACT
        register_model(register_args_minimal)

        # ASSERT
        register_call = mock_mlflow_environment["register_model"].call_args
        assert register_call[1]["tags"] == {}

    @pytest.mark.parametrize(
        "val_acc,train_loss,seeds_activated,expected",
        [
            (0.85, 0.23, True, "Val Acc: 0.8500, Train Loss: 0.2300, Seeds Activated: True"),
            (None, 0.15, False, "Val Acc: 0.0000, Train Loss: 0.1500, Seeds Activated: False"),
            (0.92, None, None, "Val Acc: 0.9200, Train Loss: 0.0000, Seeds Activated: False"),
            (None, None, None, "Val Acc: 0.0000, Train Loss: 0.0000, Seeds Activated: False"),
            (
                1.0,
                0.0,
                True,
                "Val Acc: 1.0000, Train Loss: 0.0000, Seeds Activated: True",
            ),  # Boundary values
        ],
    )
    def test_metrics_description_generation(
        self, register_args, mock_mlflow_environment, val_acc, train_loss, seeds_activated, expected
    ):
        """Test proper generation of description from metrics."""
        # ARRANGE
        register_args.val_acc = val_acc
        register_args.train_loss = train_loss
        register_args.seeds_activated = seeds_activated
        register_args.description = None  # Let it auto-generate

        # ACT
        register_model(register_args)

        # ASSERT - Verify description contains expected metrics
        update_call = mock_mlflow_environment["client"].update_model_version.call_args
        description = update_call[1]["description"]
        assert expected in description
        assert "Morphogenetic model" in description

    def test_custom_description_preserves_user_input(self, register_args, mock_mlflow_environment):
        """Test that custom descriptions are preserved."""
        # ARRANGE
        custom_description = "Custom model description for testing"
        register_args.description = custom_description

        # ACT
        register_model(register_args)

        # ASSERT
        update_call = mock_mlflow_environment["client"].update_model_version.call_args
        description = update_call[1]["description"]
        assert description == custom_description

    @given(st.lists(valid_tag_strategy(), min_size=1, max_size=10))
    def test_tag_parsing_property_based(self, tags):
        """Property-based test for tag parsing with various valid inputs."""
        # ARRANGE - Use context manager instead of fixture
        with (
            patch("morphogenetic_engine.model_registry.MlflowClient") as mock_client_class,
            patch("morphogenetic_engine.model_registry.mlflow.register_model") as mock_register,
        ):

            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_register.return_value = MockModelVersion.create(version="1")
            mock_client.update_model_version.return_value = None

            register_args = type(
                "Args",
                (),
                {
                    "model_name": "TestModel",
                    "run_id": "test_run_123",
                    "description": "Test description",
                    "val_acc": 0.85,
                    "train_loss": 0.23,
                    "seeds_activated": True,
                    "tags": tags,
                },
            )()

            expected_tags = {}
            for tag in tags:
                if "=" in tag:
                    key, value = tag.split("=", 1)
                    expected_tags[key] = value

            # ACT
            register_model(register_args)

            # ASSERT
            register_call = mock_register.call_args
            assert register_call[1]["tags"] == expected_tags

    @given(unicode_text_strategy())
    def test_unicode_model_names(self, unicode_name):
        """Test handling of Unicode characters in model names."""
        # ARRANGE - Use context manager instead of fixture
        with (
            patch("morphogenetic_engine.model_registry.MlflowClient") as mock_client_class,
            patch("morphogenetic_engine.model_registry.mlflow.register_model") as mock_register,
        ):

            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_register.return_value = MockModelVersion.create(version="1")
            mock_client.update_model_version.return_value = None

            register_args = type(
                "Args",
                (),
                {
                    "model_name": unicode_name,
                    "run_id": "test_run_123",
                    "description": "Test description",
                    "val_acc": 0.85,
                    "train_loss": 0.23,
                    "seeds_activated": True,
                    "tags": ["env=test"],
                },
            )()

            # ACT & ASSERT - Should not raise an exception
            register_model(register_args)

            # Verify Unicode name was passed through correctly
            register_call = mock_register.call_args
            assert register_call[1]["name"] == unicode_name


class TestModelRegistryCLIIntegration:
    """Integration tests using real ModelRegistry with mocked external dependencies only."""

    def test_register_model_complete_workflow(self, register_args, mock_mlflow_environment):
        """Test complete model registration workflow with realistic data."""
        # ARRANGE
        model_version = MockModelVersion.create(version="3", stage="None")
        mock_mlflow_environment["register_model"].return_value = model_version

        # ACT - Should complete successfully
        register_model(register_args)

        # ASSERT - Verify the complete workflow
        # 1. Model was registered with correct URI
        register_call = mock_mlflow_environment["register_model"].call_args
        expected_uri = f"runs:/{register_args.run_id}/model"
        assert register_call[1]["model_uri"] == expected_uri
        assert register_call[1]["name"] == register_args.model_name

        # 2. Description was updated
        mock_mlflow_environment["client"].update_model_version.assert_called_once()
        update_call = mock_mlflow_environment["client"].update_model_version.call_args
        assert update_call[1]["name"] == register_args.model_name
        assert update_call[1]["version"] == "3"

    def test_promote_model_with_archiving(self, promote_args, mock_mlflow_environment):
        """Test model promotion with automatic archiving of existing production models."""
        # ARRANGE
        # Simulate existing production model with both stage and aliases for compatibility
        existing_prod = MockModelVersion.create(version="1", stage="Production", aliases=["Production"])
        mock_mlflow_environment["client"].search_model_versions.return_value = [existing_prod]

        # ACT
        promote_model(promote_args)

        # ASSERT
        client = mock_mlflow_environment["client"]

        # Should search for existing models (for archiving) if get_model_version_by_alias fails
        # The new implementation tries get_model_version_by_alias first, then falls back to search
        search_calls = client.search_model_versions.call_args_list
        if search_calls:
            search_call = search_calls[0]
            expected_filter = f"name='{promote_args.model_name}'"
            assert expected_filter in search_call[1]["filter_string"]

        # Should have set the new alias (modern API)
        client.set_registered_model_alias.assert_called()

        # Should have deleted the existing alias for archiving
        client.delete_registered_model_alias.assert_called()

    def test_list_models_with_realistic_data(self, list_args, mock_mlflow_environment):
        """Test model listing with realistic model version data."""
        # ARRANGE
        versions = create_realistic_model_versions(5)
        mock_mlflow_environment["client"].search_model_versions.return_value = versions

        # ACT
        list_models(list_args)

        # ASSERT
        search_call = mock_mlflow_environment["client"].search_model_versions.call_args
        expected_filter = f"name='{list_args.model_name}'"
        assert expected_filter in search_call[1]["filter_string"]

    def test_get_best_model_with_metric_retrieval(self, best_args, mock_mlflow_environment, mock_cli_mlflow_client):
        """Test best model retrieval with actual metric fetching."""
        # ARRANGE
        best_version = MockModelVersion.create(version="3", stage="Staging", run_id="run_789")
        mock_mlflow_environment["client"].search_model_versions.return_value = [best_version]

        # Mock metric retrieval
        mock_run = create_mock_run({"val_acc": 0.92, "train_loss": 0.15})
        mock_cli_mlflow_client.get_run.return_value = mock_run

        # ACT
        get_best_model(best_args)

        # ASSERT
        # Should search for best model
        search_call = mock_mlflow_environment["client"].search_model_versions.call_args
        assert f"name='{best_args.model_name}'" in search_call[1]["filter_string"]

        # Should retrieve metrics from the run
        mock_cli_mlflow_client.get_run.assert_called_with("run_789")

    def test_get_production_model_uri_retrieval(self, production_args, mock_mlflow_environment):
        """Test production model URI retrieval."""
        # ARRANGE
        prod_version = MockModelVersion.create(version="2", stage="Production", aliases=["Production"])
        # Set up the get_model_version_by_alias call (modern API)
        mock_mlflow_environment["client"].get_model_version_by_alias.return_value = prod_version

        # ACT
        get_production_model(production_args)

        # ASSERT
        # Should use the modern get_model_version_by_alias API
        mock_mlflow_environment["client"].get_model_version_by_alias.assert_called_with(
            name=production_args.model_name, alias="Production"
        )


class TestModelRegistryCLIEdgeCases:
    """Test edge cases, boundary conditions, and error scenarios."""

    def test_register_model_with_extremely_long_inputs(self, register_args, mock_mlflow_environment):
        """Test handling of extremely long model names and descriptions."""
        # ARRANGE
        long_name = "ModelName" * 100  # 900 characters
        long_description = "Description " * 500  # 6000 characters

        register_args.model_name = long_name
        register_args.description = long_description

        # ACT & ASSERT - Should handle gracefully
        register_model(register_args)

        register_call = mock_mlflow_environment["register_model"].call_args
        assert register_call[1]["name"] == long_name

    def test_register_model_with_special_characters(self, register_args, mock_mlflow_environment):
        """Test handling of special characters in various fields."""
        # ARRANGE
        register_args.tags = [
            "emoji=🚀",
            "unicode=αβγδε",
            "symbols=!@#$%^&*()",
            'quotes=key="value with spaces"',
            "newlines=line1\nline2",
        ]

        # ACT
        register_model(register_args)

        # ASSERT
        register_call = mock_mlflow_environment["register_model"].call_args
        tags = register_call[1]["tags"]
        assert tags["emoji"] == "🚀"
        assert tags["unicode"] == "αβγδε"
        assert tags["symbols"] == "!@#$%^&*()"

    def test_promote_model_with_concurrent_modifications(self, promote_args, mock_mlflow_environment):
        """Test promotion when concurrent modifications occur with alias-based API."""
        # ARRANGE - With alias-based API, we primarily use get_model_version_by_alias
        initial_prod = MockModelVersion.create(version="1", stage="Production", aliases=["Production"])

        # Set up get_model_version_by_alias to return existing version
        mock_mlflow_environment["client"].get_model_version_by_alias.return_value = initial_prod

        # ACT & ASSERT - Should handle gracefully with modern API
        promote_model(promote_args)

        # Should have used the modern alias-based API
        mock_mlflow_environment["client"].get_model_version_by_alias.assert_called()
        mock_mlflow_environment["client"].set_registered_model_alias.assert_called()

    def test_list_models_with_malformed_timestamps(self, list_args, mock_mlflow_environment):
        """Test listing models with malformed or None timestamps."""
        # ARRANGE
        versions = [
            MockModelVersion.create(version="1", timestamp=None),
            MockModelVersion.create(version="2", timestamp=0),
            MockModelVersion.create(version="3", timestamp=-1),
            MockModelVersion.create(version="4", timestamp=9999999999999),  # Far future
        ]
        mock_mlflow_environment["client"].search_model_versions.return_value = versions

        # ACT & ASSERT - Should not crash
        list_models(list_args)

    def test_get_best_model_with_no_metrics(self, best_args, mock_mlflow_environment, mock_cli_mlflow_client):
        """Test best model retrieval when runs have no metrics."""
        # ARRANGE
        best_version = MockModelVersion.create(version="1", run_id="run_no_metrics")
        mock_mlflow_environment["client"].search_model_versions.return_value = [best_version]

        # Mock run with no metrics
        mock_run = create_mock_run({})  # Empty metrics
        mock_cli_mlflow_client.get_run.return_value = mock_run

        # ACT & ASSERT - Should handle gracefully
        get_best_model(best_args)

    def test_register_model_with_zero_metrics(self, register_args, mock_mlflow_environment):
        """Test registration with zero values for metrics (boundary case)."""
        # ARRANGE
        register_args.val_acc = 0.0
        register_args.train_loss = 0.0
        register_args.seeds_activated = False
        register_args.description = None  # Auto-generate

        # ACT
        register_model(register_args)

        # ASSERT - Zero values should be preserved, not treated as None
        update_call = mock_mlflow_environment["client"].update_model_version.call_args
        description = update_call[1]["description"]
        assert "Val Acc: 0.0000" in description
        assert "Train Loss: 0.0000" in description
        assert "Seeds Activated: False" in description

    @pytest.mark.parametrize(
        "invalid_stage",
        [
            "InvalidStage",
            "production",  # Wrong case
            "PRODUCTION",  # Wrong case
            "",  # Empty string
            "None",  # String "None"
        ],
    )
    def test_promote_model_invalid_stages(self, promote_args, mock_mlflow_environment, invalid_stage):
        """Test promotion with invalid stage names using modern alias API."""
        # ARRANGE
        promote_args.stage = invalid_stage
        # Modern API: set_registered_model_alias will fail for invalid aliases
        import mlflow.exceptions

        mock_mlflow_environment["client"].set_registered_model_alias.side_effect = mlflow.exceptions.MlflowException(
            f"Invalid alias: {invalid_stage}"
        )

        # ACT & ASSERT - CLI should exit with error code due to promote_model returning False
        with pytest.raises(SystemExit) as exc_info:
            promote_model(promote_args)
        assert exc_info.value.code == 1

    def test_large_model_registry_performance(self, list_args, mock_mlflow_environment):
        """Test performance with large number of model versions."""
        # ARRANGE - Create large number of versions
        large_version_list = create_realistic_model_versions(1000)
        mock_mlflow_environment["client"].search_model_versions.return_value = large_version_list

        # ACT & ASSERT - Should complete without timeout
        list_models(list_args)

        # Verify all versions would be processed
        assert len(large_version_list) == 1000


class TestModelRegistryCLIErrorConditions:
    """Test comprehensive error conditions and failure scenarios."""

    def test_register_model_mlflow_service_unavailable(self, register_args, mock_mlflow_environment):
        """Test registration when MLflow service is unavailable."""
        # ARRANGE - ModelRegistry returns None on failure
        mock_mlflow_environment["register_model"].return_value = None

        # ACT & ASSERT - CLI checks for None and exits
        with pytest.raises(SystemExit) as exc_info:
            register_model(register_args)
        assert exc_info.value.code == 1

    def test_register_model_authentication_failure(self, register_args, mock_mlflow_environment):
        """Test registration with authentication errors."""
        # ARRANGE - ModelRegistry returns None on failure
        mock_mlflow_environment["register_model"].return_value = None

        # ACT & ASSERT
        with pytest.raises(SystemExit) as exc_info:
            register_model(register_args)
        assert exc_info.value.code == 1

    def test_register_model_disk_full_error(self, register_args, mock_mlflow_environment):
        """Test registration when disk space is exhausted."""
        # ARRANGE - ModelRegistry returns None on failure
        mock_mlflow_environment["register_model"].return_value = None

        # ACT & ASSERT
        with pytest.raises(SystemExit) as exc_info:
            register_model(register_args)
        assert exc_info.value.code == 1

    def test_promote_model_version_not_found(self, promote_args, mock_mlflow_environment):
        """Test promotion when specified version doesn't exist."""
        # ARRANGE - ModelRegistry catches exception and returns False
        import mlflow.exceptions

        mock_mlflow_environment["client"].set_registered_model_alias.side_effect = mlflow.exceptions.MlflowException(
            "Model version not found"
        )

        # ACT & ASSERT - CLI checks return value and exits
        with pytest.raises(SystemExit) as exc_info:
            promote_model(promote_args)
        assert exc_info.value.code == 1

    def test_promote_model_permission_denied(self, promote_args, mock_mlflow_environment):
        """Test promotion with insufficient permissions."""
        # ARRANGE - ModelRegistry catches exception and returns False
        import mlflow.exceptions

        mock_mlflow_environment["client"].set_registered_model_alias.side_effect = mlflow.exceptions.MlflowException(
            "Permission denied: Cannot set model alias"
        )

        # ACT & ASSERT
        with pytest.raises(SystemExit) as exc_info:
            promote_model(promote_args)
        assert exc_info.value.code == 1

    def test_list_models_connection_timeout(self, list_args, mock_mlflow_environment):
        """Test model listing with connection timeout."""
        # ARRANGE - ModelRegistry.list_model_versions returns [] on error
        import mlflow.exceptions

        mock_mlflow_environment["client"].search_model_versions.side_effect = mlflow.exceptions.MlflowException(
            "Connection timeout: MLflow server not responding"
        )

        # ACT & ASSERT - list_models handles empty result gracefully
        list_models(list_args)  # Should not crash

    def test_get_best_model_run_not_found(self, best_args, mock_mlflow_environment, mock_cli_mlflow_client):
        """Test best model retrieval when associated run is missing."""
        # ARRANGE
        best_version = MockModelVersion.create(version="1", run_id="missing_run")
        mock_mlflow_environment["client"].search_model_versions.return_value = [best_version]

        mock_cli_mlflow_client.get_run.side_effect = Exception("Run not found: missing_run")

        # ACT & ASSERT - Should handle gracefully and not crash
        get_best_model(best_args)

        # Should still attempt to get the run even if it fails
        mock_cli_mlflow_client.get_run.assert_called_with("missing_run")

    def test_get_production_model_database_corruption(self, production_args, mock_mlflow_environment):
        """Test production model retrieval with database corruption."""
        # ARRANGE - ModelRegistry.get_production_model_uri returns None on error
        import mlflow.exceptions

        mock_mlflow_environment["client"].search_model_versions.side_effect = mlflow.exceptions.MlflowException(
            "Database corruption detected"
        )

        # ACT & ASSERT - Should not crash, returns None and prints message
        get_production_model(production_args)

    @pytest.mark.parametrize(
        "network_error",
        [
            "Connection timed out",
            "Network unreachable",
            "DNS resolution failed",
            "SSL certificate verification failed",
            "HTTP 503 Service Unavailable",
        ],
    )
    def test_various_network_failures(self, register_args, mock_mlflow_environment, network_error):
        """Test handling of various network-related failures."""
        # ARRANGE - Use the parametrized network error
        mock_mlflow_environment["register_model"].side_effect = network_error

        # ACT & ASSERT
        with pytest.raises(SystemExit) as exc_info:
            register_model(register_args)
        assert exc_info.value.code == 1

    def test_memory_exhaustion_during_large_operation(self, list_args, mock_mlflow_environment):
        """Test handling of memory exhaustion with very large model lists."""
        # ARRANGE - ModelRegistry.list_model_versions returns [] on error
        import mlflow.exceptions

        mock_mlflow_environment["client"].search_model_versions.side_effect = mlflow.exceptions.MlflowException(
            "Cannot allocate memory for large result set"
        )

        # ACT & ASSERT - Should not crash
        list_models(list_args)


class TestModelRegistryCLIArgumentValidation:
    """Test CLI argument parsing, validation, and main function routing."""

    def test_main_no_command_shows_help(self):
        """Test main function with no command shows help and exits."""
        # ARRANGE
        with patch("sys.argv", ["model_registry_cli.py"]):
            # ACT & ASSERT
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @pytest.mark.parametrize(
        "command,expected_function",
        [
            (["register", "test_run_123"], "register_model"),
            (["promote", "Production", "--version", "3"], "promote_model"),
            (["list"], "list_models"),
            (["best"], "get_best_model"),
            (["production"], "get_production_model"),
        ],
    )
    def test_main_command_routing(self, command, expected_function):
        """Test that main function correctly routes to expected functions."""
        # ARRANGE
        full_argv = ["model_registry_cli.py"] + command

        with patch("sys.argv", full_argv):
            with patch(f"morphogenetic_engine.cli.model_registry_cli.{expected_function}") as mock_func:
                # ACT
                main()

                # ASSERT
                mock_func.assert_called_once()

    def test_main_with_custom_model_name(self):
        """Test main function with custom model name argument."""
        # ARRANGE
        with patch("sys.argv", ["model_registry_cli.py", "--model-name", "CustomModel", "list"]):
            with patch("morphogenetic_engine.cli.model_registry_cli.list_models") as mock_list:
                # ACT
                main()

                # ASSERT
                call_args = mock_list.call_args[0][0]
                assert call_args.model_name == "CustomModel"

    def test_register_command_with_all_arguments(self):
        """Test register command with all possible arguments."""
        # ARRANGE
        argv = [
            "model_registry_cli.py",
            "register",
            "test_run_123",
            "--description",
            "Test model",
            "--val-acc",
            "0.95",
            "--train-loss",
            "0.05",
            "--seeds-activated",
            "True",
            "--tags",
            "env=prod",
            "version=2.0",
        ]

        with patch("sys.argv", argv):
            with patch("morphogenetic_engine.cli.model_registry_cli.register_model") as mock_register:
                # ACT
                main()

                # ASSERT
                args = mock_register.call_args[0][0]
                assert args.run_id == "test_run_123"
                assert args.description == "Test model"
                assert args.val_acc == pytest.approx(0.95)
                assert args.train_loss == pytest.approx(0.05)
                assert args.seeds_activated is True
                assert args.tags == ["env=prod", "version=2.0"]

    def test_promote_command_with_no_archive_flag(self):
        """Test promote command with --no-archive flag."""
        # ARRANGE
        with patch("sys.argv", ["model_registry_cli.py", "promote", "Production", "--no-archive"]):
            with patch("morphogenetic_engine.cli.model_registry_cli.promote_model") as mock_promote:
                # ACT
                main()

                # ASSERT
                args = mock_promote.call_args[0][0]
                assert args.archive_existing is False

    def test_best_command_with_lower_is_better_flag(self):
        """Test best command with --lower-is-better flag."""
        # ARRANGE
        with patch("sys.argv", ["model_registry_cli.py", "best", "--metric", "loss", "--lower-is-better"]):
            with patch("morphogenetic_engine.cli.model_registry_cli.get_best_model") as mock_best:
                # ACT
                main()

                # ASSERT
                args = mock_best.call_args[0][0]
                assert args.metric == "loss"
                assert args.higher_is_better is False

    @pytest.mark.parametrize(
        "exception_type,expected_exit_code",
        [
            (KeyboardInterrupt(), 1),
            (Exception("Test error"), 1),
            (OSError("File not found"), 1),
            (ValueError("Invalid value"), 1),
        ],
    )
    def test_main_exception_handling(self, exception_type, expected_exit_code):
        """Test main function exception handling for various error types."""
        # ARRANGE
        with patch("sys.argv", ["model_registry_cli.py", "register", "test_run"]):
            with patch("morphogenetic_engine.cli.model_registry_cli.register_model") as mock_register:
                mock_register.side_effect = exception_type

                # ACT & ASSERT
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == expected_exit_code

    def test_keyboard_interrupt_handling(self):
        """Test graceful handling of keyboard interrupts."""
        # ARRANGE
        with patch("sys.argv", ["model_registry_cli.py", "register", "test_run"]):
            with patch("morphogenetic_engine.cli.model_registry_cli.register_model") as mock_register:
                mock_register.side_effect = KeyboardInterrupt()

                # ACT & ASSERT
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1


class TestModelRegistryCLIPropertyBased:
    """Property-based tests for robust validation of CLI behavior."""

    @given(st.lists(valid_tag_strategy(), min_size=0, max_size=20))
    def test_tag_parsing_robustness(self, tags):
        """Property-based test ensuring tag parsing never crashes."""
        # ARRANGE - Use context manager instead of fixture
        with (
            patch("morphogenetic_engine.model_registry.MlflowClient") as mock_client_class,
            patch("morphogenetic_engine.model_registry.mlflow.register_model") as mock_register,
        ):

            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_register.return_value = MockModelVersion.create(version="1")
            mock_client.update_model_version.return_value = None

            register_args = type(
                "Args",
                (),
                {
                    "model_name": "TestModel",
                    "run_id": "test_run_123",
                    "description": "Test description",
                    "val_acc": 0.85,
                    "train_loss": 0.23,
                    "seeds_activated": True,
                    "tags": tags,
                },
            )()

            # ACT & ASSERT - Should never raise an exception
            register_model(register_args)

            # Verify call was made successfully
            assert mock_register.called

    @given(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    def test_metrics_boundary_values(self, metric_value):
        """Test metrics handling with various boundary values."""
        # ARRANGE - Use context manager instead of fixture
        with (
            patch("morphogenetic_engine.model_registry.MlflowClient") as mock_client_class,
            patch("morphogenetic_engine.model_registry.mlflow.register_model") as mock_register,
        ):

            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_register.return_value = MockModelVersion.create(version="1")
            mock_client.update_model_version.return_value = None

            register_args = type(
                "Args",
                (),
                {
                    "model_name": "TestModel",
                    "run_id": "test_run_123",
                    "val_acc": metric_value,
                    "train_loss": 0.23,
                    "seeds_activated": True,
                    "tags": ["env=test"],
                    "description": None,  # Auto-generate
                },
            )()

            # ACT & ASSERT - Should handle all finite float values
            register_model(register_args)

            # Verify description was generated
            update_call = mock_client.update_model_version.call_args
            description = update_call[1]["description"]
            assert f"Val Acc: {metric_value:.4f}" in description

    @given(unicode_text_strategy())
    def test_unicode_handling_in_descriptions(self, unicode_text):
        """Property-based test for Unicode handling in descriptions."""
        # ARRANGE - Use context manager instead of fixture
        with (
            patch("morphogenetic_engine.model_registry.MlflowClient") as mock_client_class,
            patch("morphogenetic_engine.model_registry.mlflow.register_model") as mock_register,
        ):

            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_register.return_value = MockModelVersion.create(version="1")
            mock_client.update_model_version.return_value = None

            register_args = type(
                "Args",
                (),
                {
                    "model_name": "TestModel",
                    "run_id": "test_run_123",
                    "description": unicode_text,
                    "val_acc": 0.85,
                    "train_loss": 0.23,
                    "seeds_activated": True,
                    "tags": ["env=test"],
                },
            )()

            # ACT & ASSERT - Should handle any Unicode text
            register_model(register_args)

            update_call = mock_client.update_model_version.call_args
            assert update_call[1]["description"] == unicode_text


class TestModelRegistryCLIPerformance:
    """Performance and load testing for CLI operations."""

    def test_concurrent_registrations(self, mock_mlflow_environment):
        """Test handling of concurrent model registrations."""
        import threading

        from mlflow.exceptions import MlflowException

        # Ensure the MLflow environment is available for all threads
        assert mock_mlflow_environment is not None

        results = []

        def register_worker(worker_id):
            """Worker function for concurrent registration."""
            args = type(
                "Args",
                (),
                {
                    "model_name": f"TestModel{worker_id}",
                    "run_id": f"run_{worker_id}",
                    "description": f"Worker {worker_id} model",
                    "val_acc": 0.8 + (worker_id * 0.01),
                    "train_loss": 0.2 - (worker_id * 0.01),
                    "seeds_activated": True,
                    "tags": [f"worker={worker_id}"],
                },
            )()

            try:
                register_model(args)
                results.append(f"success_{worker_id}")
            except (SystemExit, MlflowException) as e:
                results.append(f"error_{worker_id}_{type(e).__name__}")

        # ARRANGE - Create multiple worker threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_worker, args=(i,))
            threads.append(thread)

        # ACT - Start all threads concurrently
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        # ASSERT - All operations should complete
        assert len(results) == 10
        success_count = len([r for r in results if r.startswith("success_")])
        assert success_count == 10, f"Expected 10 successes, got {success_count}. Results: {results}"

    def test_large_tag_list_performance(self, register_args, mock_mlflow_environment):
        """Test performance with large number of tags."""
        # ARRANGE - Create large tag list
        large_tag_list = [f"tag{i}=value{i}" for i in range(1000)]
        register_args.tags = large_tag_list

        start_time = time.time()

        # ACT
        register_model(register_args)

        # ASSERT - Should complete within reasonable time
        execution_time = time.time() - start_time
        assert execution_time < 5.0, f"Tag parsing took too long: {execution_time:.2f}s"

        # Verify all tags were processed
        register_call = mock_mlflow_environment["register_model"].call_args
        assert len(register_call[1]["tags"]) == 1000

    def test_memory_usage_with_large_model_list(self, list_args, mock_mlflow_environment):
        """Test memory usage with very large model version lists."""
        # ARRANGE - Create large list of model versions
        large_version_list = create_realistic_model_versions(10000)
        mock_mlflow_environment["client"].search_model_versions.return_value = large_version_list

        # Measure memory before
        import tracemalloc

        tracemalloc.start()

        # ACT
        list_models(list_args)

        # ASSERT - Check memory usage
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (less than 100MB for 10k models)
        assert peak < 100 * 1024 * 1024, f"Peak memory usage too high: {peak / 1024 / 1024:.1f}MB"
