"""
Unit tests for the Model Registry CLI.

Tests the command-line interface for model registry operations including
model registration, promotion, listing, and management.
"""

import argparse
import sys
from io import StringIO
from unittest.mock import Mock, call, patch

import pytest

from morphogenetic_engine.cli.model_registry_cli import (
    get_best_model,
    get_production_model,
    list_models,
    main,
    promote_model,
    register_model,
)


class TestModelRegistryCLI:
    """Test suite for Model Registry CLI functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Sample arguments for testing
        self.register_args = argparse.Namespace(
            model_name="TestModel",
            run_id="test_run_123",
            description="Test description",
            val_acc=0.85,
            train_loss=0.23,
            seeds_activated=True,
            tags=["env=test", "version=1.0"],
        )

        self.promote_args = argparse.Namespace(
            model_name="TestModel", stage="Production", version="3", archive_existing=True
        )

        self.list_args = argparse.Namespace(model_name="TestModel", stage=None)

        self.best_args = argparse.Namespace(
            model_name="TestModel", metric="val_acc", stage=None, higher_is_better=True
        )

    @patch("morphogenetic_engine.cli.model_registry_cli.ModelRegistry")
    def test_register_model_success(self, mock_registry_class):
        """Test successful model registration via CLI."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_model_version = Mock()
        mock_model_version.version = "2"
        mock_registry.register_best_model.return_value = mock_model_version

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            register_model(self.register_args)

            # Check output
            output = mock_stdout.getvalue()
            assert "✅ Successfully registered model version 2" in output
            assert "Run ID: test_run_123" in output
            assert "Model Name: TestModel" in output

        # Check registry was called correctly
        mock_registry.register_best_model.assert_called_once_with(
            run_id="test_run_123",
            metrics={"val_acc": 0.85, "train_loss": 0.23, "seeds_activated": True},
            description="Test description",
            tags={"env": "test", "version": "1.0"},
        )

    @patch("morphogenetic_engine.cli.model_registry_cli.ModelRegistry")
    def test_register_model_failure(self, mock_registry_class):
        """Test model registration failure via CLI."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.register_best_model.return_value = None

        # Test should exit with code 1
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout, pytest.raises(
            SystemExit
        ) as exc_info:
            register_model(self.register_args)

            assert exc_info.value.code == 1
            assert "❌ Failed to register model" in mock_stdout.getvalue()

    @patch("morphogenetic_engine.cli.model_registry_cli.ModelRegistry")
    def test_register_model_no_tags(self, mock_registry_class):
        """Test model registration without tags."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_model_version = Mock()
        mock_model_version.version = "1"
        mock_registry.register_best_model.return_value = mock_model_version

        # Args without tags
        args_no_tags = argparse.Namespace(
            model_name="TestModel",
            run_id="test_run_456",
            description=None,
            val_acc=None,
            train_loss=None,
            seeds_activated=None,
            tags=None,
        )

        register_model(args_no_tags)

        # Should call with empty metrics and tags
        mock_registry.register_best_model.assert_called_once_with(
            run_id="test_run_456", metrics={}, description=None, tags={}
        )

    @patch("morphogenetic_engine.cli.model_registry_cli.ModelRegistry")
    def test_promote_model_success(self, mock_registry_class):
        """Test successful model promotion via CLI."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.promote_model.return_value = True

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            promote_model(self.promote_args)

            # Check output
            output = mock_stdout.getvalue()
            assert "✅ Successfully promoted model TestModel v3 to Production" in output

        # Check registry was called correctly
        mock_registry.promote_model.assert_called_once_with(
            version="3", stage="Production", archive_existing=True
        )

    @patch("morphogenetic_engine.cli.model_registry_cli.ModelRegistry")
    def test_promote_model_failure(self, mock_registry_class):
        """Test model promotion failure via CLI."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.promote_model.return_value = False

        # Test should exit with code 1
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout, pytest.raises(
            SystemExit
        ) as exc_info:
            promote_model(self.promote_args)

            assert exc_info.value.code == 1
            assert "Failed to promote model" in mock_stdout.getvalue()

    @patch("morphogenetic_engine.cli.model_registry_cli.ModelRegistry")
    def test_list_models_success(self, mock_registry_class):
        """Test successful model listing via CLI."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        # Create mock model versions
        version1 = Mock()
        version1.version = "1"
        version1.current_stage = "Archived"
        version1.run_id = "run_123"
        version1.creation_timestamp = 1609459200000  # 2021-01-01 00:00:00

        version2 = Mock()
        version2.version = "2"
        version2.current_stage = "Production"
        version2.run_id = "run_456"
        version2.creation_timestamp = 1609545600000  # 2021-01-02 00:00:00

        mock_registry.list_model_versions.return_value = [version1, version2]

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            list_models(self.list_args)

            # Check output format
            output = mock_stdout.getvalue()
            assert "Model versions for TestModel:" in output
            assert "Version" in output
            assert "Stage" in output
            assert "Run ID" in output
            assert "Created" in output
            assert "1" in output
            assert "2" in output
            assert "Archived" in output
            assert "Production" in output

    @patch("morphogenetic_engine.cli.model_registry_cli.ModelRegistry")
    def test_list_models_empty(self, mock_registry_class):
        """Test model listing when no models exist."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.list_model_versions.return_value = []

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            list_models(self.list_args)

            # Check output
            output = mock_stdout.getvalue()
            assert "No model versions found for TestModel" in output

    @patch("morphogenetic_engine.cli.model_registry_cli.ModelRegistry")
    def test_list_models_with_stage_filter(self, mock_registry_class):
        """Test model listing with stage filter."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.list_model_versions.return_value = []

        # Args with stage filter
        args_with_stage = argparse.Namespace(model_name="TestModel", stage="Production")

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            list_models(args_with_stage)

            # Check output
            output = mock_stdout.getvalue()
            assert "(filtered by stage: Production)" in output

        # Check registry was called with stage filter
        mock_registry.list_model_versions.assert_called_once_with(stage="Production")

    @patch("morphogenetic_engine.cli.model_registry_cli.ModelRegistry")
    @patch("mlflow.tracking.MlflowClient")
    def test_get_best_model_success(self, mock_client_class, mock_registry_class):
        """Test successful best model retrieval via CLI."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_best_version = Mock()
        mock_best_version.version = "3"
        mock_best_version.current_stage = "Staging"
        mock_best_version.run_id = "run_789"
        mock_registry.get_best_model_version.return_value = mock_best_version

        # Mock MLflow client for metric retrieval
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_run = Mock()
        mock_run.data.metrics = {"val_acc": 0.92}
        mock_client.get_run.return_value = mock_run

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            get_best_model(self.best_args)

            # Check output
            output = mock_stdout.getvalue()
            assert "✅ Best model version: 3" in output
            assert "Stage: Staging" in output
            assert "Run ID: run_789" in output
            assert "val_acc: 0.92" in output

        # Check registry was called correctly
        mock_registry.get_best_model_version.assert_called_once_with(
            stage=None, metric_name="val_acc", higher_is_better=True
        )

    @patch("morphogenetic_engine.cli.model_registry_cli.ModelRegistry")
    def test_get_best_model_not_found(self, mock_registry_class):
        """Test best model retrieval when no model found."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.get_best_model_version.return_value = None

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            get_best_model(self.best_args)

            # Check output
            output = mock_stdout.getvalue()
            assert "No model versions found matching criteria" in output

    @patch("morphogenetic_engine.cli.model_registry_cli.ModelRegistry")
    def test_get_production_model_success(self, mock_registry_class):
        """Test successful production model retrieval via CLI."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.get_production_model_uri.return_value = "models:/TestModel/5"

        args = argparse.Namespace(model_name="TestModel")

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            get_production_model(args)

            # Check output
            output = mock_stdout.getvalue()
            assert "✅ Production model URI: models:/TestModel/5" in output

    @patch("morphogenetic_engine.cli.model_registry_cli.ModelRegistry")
    def test_get_production_model_not_found(self, mock_registry_class):
        """Test production model retrieval when no model found."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.get_production_model_uri.return_value = None

        args = argparse.Namespace(model_name="TestModel")

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            get_production_model(args)

            # Check output
            output = mock_stdout.getvalue()
            assert "No production model found for TestModel" in output

    @patch("sys.argv", ["model_registry_cli.py"])
    def test_main_no_command(self):
        """Test main function with no command."""
        with patch("sys.stdout", new_callable=StringIO), pytest.raises(SystemExit) as exc_info:
            main()

            assert exc_info.value.code == 1

    @patch("sys.argv", ["model_registry_cli.py", "register", "test_run_123"])
    @patch("morphogenetic_engine.cli.model_registry_cli.register_model")
    def test_main_register_command(self, mock_register):
        """Test main function with register command."""
        main()
        mock_register.assert_called_once()

    @patch("sys.argv", ["model_registry_cli.py", "promote", "Production", "--version", "3"])
    @patch("morphogenetic_engine.cli.model_registry_cli.promote_model")
    def test_main_promote_command(self, mock_promote):
        """Test main function with promote command."""
        main()
        mock_promote.assert_called_once()

    @patch("sys.argv", ["model_registry_cli.py", "list"])
    @patch("morphogenetic_engine.cli.model_registry_cli.list_models")
    def test_main_list_command(self, mock_list):
        """Test main function with list command."""
        main()
        mock_list.assert_called_once()

    @patch("sys.argv", ["model_registry_cli.py", "best"])
    @patch("morphogenetic_engine.cli.model_registry_cli.get_best_model")
    def test_main_best_command(self, mock_best):
        """Test main function with best command."""
        main()
        mock_best.assert_called_once()

    @patch("sys.argv", ["model_registry_cli.py", "production"])
    @patch("morphogenetic_engine.cli.model_registry_cli.get_production_model")
    def test_main_production_command(self, mock_production):
        """Test main function with production command."""
        main()
        mock_production.assert_called_once()

    @patch("sys.argv", ["model_registry_cli.py", "register", "test_run"])
    @patch("morphogenetic_engine.cli.model_registry_cli.register_model")
    def test_main_keyboard_interrupt(self, mock_register):
        """Test main function with keyboard interrupt."""
        mock_register.side_effect = KeyboardInterrupt()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout, pytest.raises(
            SystemExit
        ) as exc_info:
            main()

            assert exc_info.value.code == 1
            assert "Operation cancelled by user" in mock_stdout.getvalue()

    @patch("sys.argv", ["model_registry_cli.py", "register", "test_run"])
    @patch("morphogenetic_engine.cli.model_registry_cli.register_model")
    @patch("morphogenetic_engine.cli.model_registry_cli.logger")
    def test_main_general_exception(self, mock_logger, mock_register):
        """Test main function with general exception."""
        mock_register.side_effect = Exception("Test error")

        with pytest.raises(SystemExit) as exc_info:
            main()

            assert exc_info.value.code == 1
            mock_logger.error.assert_called_with("Command failed: Test error")

    def test_register_model_tag_parsing(self):
        """Test proper parsing of tags in register command."""
        # Args with multiple tags
        args = argparse.Namespace(
            model_name="TestModel",
            run_id="test_run",
            description=None,
            val_acc=None,
            train_loss=None,
            seeds_activated=None,
            tags=["env=prod", "version=2.1", "team=ml"],
        )

        with patch(
            "morphogenetic_engine.cli.model_registry_cli.ModelRegistry"
        ) as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry

            mock_model_version = Mock()
            mock_model_version.version = "1"
            mock_registry.register_best_model.return_value = mock_model_version

            register_model(args)

            # Check that tags were parsed correctly
            call_args = mock_registry.register_best_model.call_args[1]
            expected_tags = {"env": "prod", "version": "2.1", "team": "ml"}
            assert call_args["tags"] == expected_tags

    def test_register_model_invalid_tag_format(self):
        """Test handling of invalid tag format."""
        # Args with invalid tag (no equals sign)
        args = argparse.Namespace(
            model_name="TestModel",
            run_id="test_run",
            description=None,
            val_acc=None,
            train_loss=None,
            seeds_activated=None,
            tags=["env=prod", "invalid_tag", "version=1.0"],
        )

        with patch(
            "morphogenetic_engine.cli.model_registry_cli.ModelRegistry"
        ) as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry

            mock_model_version = Mock()
            mock_model_version.version = "1"
            mock_registry.register_best_model.return_value = mock_model_version

            register_model(args)

            # Check that only valid tags were parsed
            call_args = mock_registry.register_best_model.call_args[1]
            expected_tags = {"env": "prod", "version": "1.0"}
            assert call_args["tags"] == expected_tags
