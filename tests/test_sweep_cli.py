"""
Comprehensive tests for the sweep CLI module.

This test suite covers argument parsing, error handling, user experience,
and integration testing with proper mocking strategies.
"""

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import argparse
import os
import stat
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from morphogenetic_engine.cli.sweep import SweepCLI
from morphogenetic_engine.sweeps.config import SweepConfig


# Test Data Builders
class ConfigBuilder:
    """Builder for creating test configurations."""

    def __init__(self):
        self.config = {
            "sweep_type": "grid",
            "parameters": {"lr": [0.01]},
            "execution": {"max_parallel": 1},
        }

    def with_sweep_type(self, sweep_type: str):
        self.config["sweep_type"] = sweep_type
        return self

    def with_parameters(self, parameters: dict):
        self.config["parameters"] = parameters
        return self

    def with_execution(self, execution: dict):
        self.config["execution"] = execution
        return self

    def build(self) -> dict:
        return self.config.copy()


# Test Fixtures
@pytest.fixture
def cli():
    """Provide a SweepCLI instance for testing."""
    return SweepCLI()


@pytest.fixture
def tmp_config_file(tmp_path):
    """Create a temporary valid config file."""
    config_data = ConfigBuilder().build()
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump(config_data))
    return config_file


@pytest.fixture
def invalid_yaml_file(tmp_path):
    """Create a temporary file with invalid YAML."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content: [")
    return config_file


@pytest.fixture
def readonly_config_file(tmp_path):
    """Create a read-only config file to test permission errors."""
    config_data = ConfigBuilder().build()
    config_file = tmp_path / "readonly_config.yaml"
    config_file.write_text(yaml.dump(config_data))
    config_file.chmod(stat.S_IRUSR)  # Read-only
    return config_file


@pytest.fixture
def mock_runner():
    """Provide a mocked GridSearchRunner for testing."""
    with patch("morphogenetic_engine.cli.sweep.GridSearchRunner") as mock_class:
        mock_instance = Mock()
        mock_instance.run_sweep.return_value = {"status": "completed"}
        mock_class.return_value = mock_instance
        yield mock_instance


# Argument Parsing Tests
class TestArgumentParsing:
    """Test argument parsing functionality."""

    def test_parser_creation_contains_all_subcommands(self, cli):
        """Test that the argument parser contains all expected subcommands."""
        parser = cli.create_parser()
        help_text = parser.format_help()

        expected_commands = ["grid", "bayesian", "quick", "resume"]
        for command in expected_commands:
            assert command in help_text, f"Missing {command} subcommand in help"

    def test_grid_command_required_arguments(self, cli):
        """Test grid command requires config argument."""
        parser = cli.create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["grid"])  # Missing required --config

    def test_grid_command_with_all_options(self, cli):
        """Test grid command with all optional arguments."""
        parser = cli.create_parser()

        args = parser.parse_args(
            [
                "grid",
                "--config",
                "test.yaml",
                "--parallel",
                "4",
                "--timeout",
                "1800",
                "--seed",
                "123",
            ]
        )

        assert args.command == "grid"
        assert args.config == Path("test.yaml")
        assert args.parallel == 4
        assert args.timeout == 1800
        assert args.seed == 123

    def test_bayesian_command_with_trials(self, cli):
        """Test Bayesian command argument parsing."""
        parser = cli.create_parser()

        args = parser.parse_args(["bayesian", "--config", "test.yaml", "--trials", "100", "--timeout", "7200"])

        assert args.command == "bayesian"
        assert args.trials == 100
        assert args.timeout == 7200

    def test_quick_command_problem_choices(self, cli):
        """Test quick command validates problem choices."""
        parser = cli.create_parser()

        # Valid problem
        args = parser.parse_args(["quick", "--problem", "moons", "--trials", "5"])
        assert args.problem == "moons"

        # Invalid problem should cause SystemExit
        with pytest.raises(SystemExit):
            parser.parse_args(["quick", "--problem", "invalid_problem"])


# Error Handling Tests
class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    def test_nonexistent_config_file_shows_helpful_error(self, cli, capsys):
        """Test handling of non-existent configuration files."""
        args = argparse.Namespace(config=Path("nonexistent.yaml"), parallel=1, timeout=3600, seed=None)

        result = cli.run_grid_search(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.out or "Unexpected error:" in captured.out
        assert "nonexistent.yaml" in captured.out

    def test_invalid_yaml_syntax_error_handling(self, cli, invalid_yaml_file, capsys):
        """Test handling of malformed YAML configuration."""
        args = argparse.Namespace(config=invalid_yaml_file, parallel=1, timeout=3600, seed=None)

        result = cli.run_grid_search(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.out

    @pytest.mark.skipif(os.name == "nt", reason="Permission handling differs on Windows")
    def test_permission_denied_error_handling(self, cli, readonly_config_file, capsys):
        """Test handling of permission denied errors."""
        # Make the file unreadable
        readonly_config_file.chmod(0o000)

        try:
            args = argparse.Namespace(config=readonly_config_file, parallel=1, timeout=3600, seed=None)

            result = cli.run_grid_search(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Error:" in captured.out
        finally:
            # Restore permissions for cleanup
            readonly_config_file.chmod(0o644)

    def test_configuration_validation_error_handling(self, cli, tmp_path, capsys):
        """Test handling of invalid configuration content."""
        # Create config with missing required fields
        invalid_config = tmp_path / "invalid_config.yaml"
        invalid_config.write_text(
            yaml.dump(
                {
                    "sweep_type": "invalid_type",  # Invalid sweep type
                    "parameters": {},  # Empty parameters
                }
            )
        )

        args = argparse.Namespace(config=invalid_config, parallel=1, timeout=3600, seed=None)

        result = cli.run_grid_search(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.out

    def test_main_function_no_command_shows_help(self, cli, capsys):
        """Test main function shows help when no command provided."""
        result = cli.main([])

        assert result == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out

    def test_main_function_unknown_command_error(self, cli, capsys):
        """Test main function handles unknown commands gracefully."""
        # Create a mock args object with unknown command
        with patch.object(cli, "create_parser") as mock_parser:
            mock_args = Mock()
            mock_args.command = "unknown_command"
            mock_parser.return_value.parse_args.return_value = mock_args

            result = cli.main(["unknown_command"])

            assert result == 1
            captured = capsys.readouterr()
            assert "Unknown command" in captured.out


# Integration Tests
class TestGridSearchIntegration:
    """Test grid search integration with real configurations."""

    def test_successful_grid_search_execution(self, cli, tmp_config_file, mock_runner, capsys):
        """Test successful grid search execution with real config file."""
        args = argparse.Namespace(config=tmp_config_file, parallel=2, timeout=1800, seed=42)

        result = cli.run_grid_search(args)

        assert result == 0
        mock_runner.run_sweep.assert_called_once()

        # Verify console output shows progress
        captured = capsys.readouterr()
        assert "Loading" in captured.out
        assert "completed" in captured.out

    def test_grid_search_with_multiple_configs(self, cli, tmp_path, mock_runner, capsys):
        """Test grid search with multiple configuration files in directory."""
        # Create multiple config files
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        for i in range(3):
            config_file = config_dir / f"config_{i}.yaml"
            config_data = ConfigBuilder().with_parameters({"lr": [0.01 * (i + 1)]}).build()
            config_file.write_text(yaml.dump(config_data))

        args = argparse.Namespace(config=config_dir, parallel=1, timeout=3600, seed=None)

        with patch("morphogenetic_engine.cli.sweep.load_sweep_configs") as mock_load_configs:
            # Mock loading multiple configs
            configs = [SweepConfig(ConfigBuilder().build()) for _ in range(3)]
            mock_load_configs.return_value = configs

            result = cli.run_grid_search(args)

            assert result == 0
            assert mock_runner.run_sweep.call_count == 3

            captured = capsys.readouterr()
            assert "3 sweep configuration(s)" in captured.out

    def test_cli_arguments_override_config_settings(self, cli, tmp_config_file, mock_runner):
        """Test that CLI arguments properly override configuration settings."""
        args = argparse.Namespace(
            config=tmp_config_file,
            parallel=8,  # Override default
            timeout=900,  # Override default
            seed=123,
        )

        with patch("morphogenetic_engine.cli.sweep.load_sweep_config") as mock_load_config:
            config = SweepConfig(ConfigBuilder().build())
            mock_load_config.return_value = config

            result = cli.run_grid_search(args)

            assert result == 0
            # Verify config was modified with CLI args
            assert config.execution["max_parallel"] == 8
            assert config.execution["timeout_per_trial"] == 900


# Bayesian Optimization Tests
class TestBayesianOptimization:
    """Test Bayesian optimization functionality."""

    def test_bayesian_search_not_implemented_gracefully(self, cli, tmp_config_file, capsys):
        """Test Bayesian search handles missing implementation gracefully."""
        args = argparse.Namespace(config=tmp_config_file, trials=50, timeout=3600)

        # Mock the import error at the module level
        with patch.dict("sys.modules", {"morphogenetic_engine.sweeps.bayesian": None}):
            result = cli.run_bayesian_search(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Error:" in captured.out


# Quick Test Functionality
class TestQuickTestFunctionality:
    """Test quick test sweep functionality."""

    def test_quick_test_creates_temporary_config(self, cli, mock_runner, capsys):
        """Test that quick test creates appropriate temporary configuration."""
        args = argparse.Namespace(problem="spirals", trials=4)

        result = cli.run_quick_test(args)

        assert result == 0
        mock_runner.run_sweep.assert_called_once()

        captured = capsys.readouterr()
        assert "quick test" in captured.out.lower()

    def test_quick_test_limits_trials_when_requested(self, cli, mock_runner, capsys):
        """Test that quick test properly limits trials when there are many combinations."""
        args = argparse.Namespace(problem="moons", trials=2)  # Small number to force limiting

        result = cli.run_quick_test(args)

        assert result == 0
        captured = capsys.readouterr()
        # Should show limiting message since we have more combinations than trials
        assert "Limiting to 2 trials" in captured.out


# User Experience Tests
class TestUserExperience:
    """Test user experience elements like console output and help."""

    def test_help_command_displays_useful_content(self, cli):
        """Test that help commands display useful content."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--help"])

        assert exc_info.value.code == 0

    def test_subcommand_help_displays_options(self, cli):
        """Test that subcommand help displays available options."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["grid", "--help"])

        assert exc_info.value.code == 0

    def test_console_output_formatting_with_rich(self, cli, tmp_config_file, mock_runner, capsys):
        """Test that console output uses Rich formatting."""
        args = argparse.Namespace(config=tmp_config_file, parallel=1, timeout=3600, seed=None)

        result = cli.run_grid_search(args)

        assert result == 0
        captured = capsys.readouterr()
        # Rich formatting should be present (even if colors are stripped in test output)
        assert "Loading" in captured.out
        assert "completed" in captured.out

    def test_error_messages_are_user_friendly(self, cli, capsys):
        """Test that error messages are clear and helpful."""
        args = argparse.Namespace(config=Path("nonexistent_file.yaml"), parallel=1, timeout=3600, seed=None)

        result = cli.run_grid_search(args)

        assert result == 1
        captured = capsys.readouterr()
        error_output = captured.out

        # Error should be clearly marked and helpful
        assert "Error:" in error_output or "Unexpected error:" in error_output
        assert "nonexistent_file.yaml" in error_output


# Edge Cases and Performance
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_config_file_handling(self, cli, tmp_path, capsys):
        """Test handling of empty configuration files."""
        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")

        args = argparse.Namespace(config=empty_config, parallel=1, timeout=3600, seed=None)

        result = cli.run_grid_search(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.out or "Unexpected error:" in captured.out

    def test_very_large_timeout_value(self, cli, tmp_config_file, mock_runner):
        """Test handling of very large timeout values."""
        args = argparse.Namespace(config=tmp_config_file, parallel=1, timeout=999999999, seed=None)  # Very large timeout

        result = cli.run_grid_search(args)

        assert result == 0  # Should handle gracefully

    def test_cli_startup_performance(self, cli):
        """Test that CLI startup is reasonably fast."""
        start_time = time.time()

        # Create parser (main startup cost)
        parser = cli.create_parser()

        end_time = time.time()
        startup_time = end_time - start_time

        # Should start up in less than 1 second
        assert startup_time < 1.0
        assert parser is not None


# Resume Functionality Tests
class TestResumeFunctionality:
    """Test resume functionality (currently not implemented)."""

    def test_resume_returns_not_implemented_with_message(self, cli, capsys):
        """Test that resume command clearly indicates it's not implemented."""
        args = argparse.Namespace(sweep_id="test_sweep_123")

        result = cli.resume_sweep(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not yet implemented" in captured.out.lower()


# Integration Test Class
class TestFullWorkflowIntegration:
    """Test complete workflows from CLI to execution."""

    def test_complete_grid_search_workflow(self, cli, tmp_path, mock_runner, capsys):
        """Test complete grid search workflow from argument parsing to execution."""
        # Create a comprehensive config
        config_data = (
            ConfigBuilder()
            .with_parameters({"lr": [0.001, 0.01], "hidden_dim": [64, 128], "num_layers": [2, 4]})
            .with_execution({"max_parallel": 2, "timeout_per_trial": 1800})
            .build()
        )

        config_file = tmp_path / "comprehensive_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        # Test the complete workflow
        result = cli.main(
            [
                "grid",
                "--config",
                str(config_file),
                "--parallel",
                "4",
                "--timeout",
                "900",
                "--seed",
                "42",
            ]
        )

        assert result == 0
        mock_runner.run_sweep.assert_called_once()

        captured = capsys.readouterr()
        assert "Loading" in captured.out
        assert "completed" in captured.out

    def test_quick_test_complete_workflow(self, cli, mock_runner, capsys):
        """Test complete quick test workflow."""
        result = cli.main(["quick", "--problem", "clusters", "--trials", "3"])

        assert result == 0
        mock_runner.run_sweep.assert_called_once()

        captured = capsys.readouterr()
        assert "quick test" in captured.out.lower()
