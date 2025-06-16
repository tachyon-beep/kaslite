"""
Tests for the sweep CLI module.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from morphogenetic_engine.cli.sweep import SweepCLI


class TestSweepCLI:
    """Tests for SweepCLI class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = SweepCLI()

    def test_create_parser(self):
        """Test that the argument parser is created correctly."""
        parser = self.cli.create_parser()

        # Test that all subcommands are present
        help_text = parser.format_help()
        assert "grid" in help_text
        assert "bayesian" in help_text
        assert "quick" in help_text
        assert "resume" in help_text

    def test_grid_command_args(self):
        """Test grid command argument parsing."""
        parser = self.cli.create_parser()

        # Test basic grid command
        args = parser.parse_args(["grid", "--config", "test.yaml"])
        assert args.command == "grid"
        assert args.config == Path("test.yaml")
        assert args.parallel == 1  # default

        # Test with parallel option
        args = parser.parse_args(["grid", "--config", "test.yaml", "--parallel", "4"])
        assert args.parallel == 4

    def test_bayesian_command_args(self):
        """Test Bayesian command argument parsing."""
        parser = self.cli.create_parser()

        args = parser.parse_args(["bayesian", "--config", "test.yaml", "--trials", "50"])
        assert args.command == "bayesian"
        assert args.trials == 50

    def test_quick_command_args(self):
        """Test quick command argument parsing."""
        parser = self.cli.create_parser()

        args = parser.parse_args(["quick", "--problem", "moons", "--trials", "5"])
        assert args.command == "quick"
        assert args.problem == "moons"
        assert args.trials == 5

    def test_main_no_command(self):
        """Test main function with no command."""
        with patch("sys.argv", ["sweep"]):
            result = self.cli.main([])
            assert result == 1  # Should return error code

    def test_main_unknown_command(self):
        """Test main function with unknown command."""
        # Mock the args to simulate an unknown command that somehow got through
        with patch.object(self.cli, "create_parser") as mock_parser:
            mock_args = type("Args", (), {"command": "unknown"})()
            mock_parser.return_value.parse_args.return_value = mock_args

            result = self.cli.main(["unknown"])
            assert result == 1

    @patch("morphogenetic_engine.cli.sweep.GridSearchRunner")
    @patch("morphogenetic_engine.cli.sweep.load_sweep_config")
    def test_run_grid_search_success(self, mock_load_config, mock_runner_class):
        """Test successful grid search execution."""
        # Create a temporary config file
        config_data = {
            "sweep_type": "grid",
            "parameters": {"lr": [0.01]},
            "execution": {"max_parallel": 1},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            # Mock the config loading
            from morphogenetic_engine.sweeps.config import SweepConfig

            mock_config = SweepConfig(config_data)
            mock_load_config.return_value = mock_config

            # Mock the runner
            mock_runner = mock_runner_class.return_value
            mock_runner.run_sweep.return_value = None

            # Create args
            args = type(
                "Args", (), {"config": config_path, "parallel": 2, "timeout": 3600, "seed": 42}
            )()

            result = self.cli.run_grid_search(args)
            assert result == 0

            # Verify the runner was called
            mock_runner_class.assert_called_once()
            mock_runner.run_sweep.assert_called_once()

        finally:
            config_path.unlink()

    def test_run_grid_search_file_not_found(self):
        """Test grid search with non-existent config file."""
        args = type(
            "Args",
            (),
            {"config": Path("nonexistent.yaml"), "parallel": 1, "timeout": 3600, "seed": None},
        )()

        result = self.cli.run_grid_search(args)
        assert result == 1  # Should return error code

    def test_run_quick_test(self):
        """Test quick test functionality."""
        args = type("Args", (), {"problem": "spirals", "trials": 4})()

        # This test just checks that the method doesn't crash
        # Full integration testing would require the experiment runner
        with patch("morphogenetic_engine.cli.sweep.GridSearchRunner") as mock_runner_class:
            mock_runner = mock_runner_class.return_value
            mock_runner.run_sweep.return_value = None

            result = self.cli.run_quick_test(args)
            assert result == 0

    def test_resume_sweep_not_implemented(self):
        """Test that resume sweep returns not implemented error."""
        args = type("Args", (), {"sweep_id": "test_id"})()

        result = self.cli.resume_sweep(args)
        assert result == 1  # Not yet implemented


class TestSweepCLIIntegration:
    """Integration tests for sweep CLI (require valid config files)."""

    def test_help_commands(self):
        """Test that help commands work."""
        cli = SweepCLI()

        # Test main help - should exit with code 0 (success)
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--help"])
        assert exc_info.value.code == 0

        # Test subcommand help - should also exit with code 0
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["grid", "--help"])
        assert exc_info.value.code == 0
