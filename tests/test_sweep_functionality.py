"""
Test suite for the hyperparameter sweep functionality.

This module tests all aspects of the YAML-driven hyperparameter sweep feature,
including YAML parsing, grid expansion, parameter validation, and sweep execution.
"""

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from morphogenetic_engine.cli.arguments import get_valid_argument_names
from morphogenetic_engine.cli.arguments import parse_experiment_arguments as parse_arguments
from morphogenetic_engine.sweeps.runner import (
    expand_grid,
    generate_run_slug,
    load_sweep_configs,
    merge_args_with_combo,
    parse_value_list,
    run_parameter_sweep,
    validate_sweep_config,
)


class TestParseValueList:
    """Test the parse_value_list function for different input formats."""

    def test_comma_separated_string(self):
        """Test parsing comma-separated string values."""
        result = parse_value_list("1,2,4,12,16")
        assert result == ["1", "2", "4", "12", "16"]

    def test_comma_separated_with_spaces(self):
        """Test parsing comma-separated string with spaces."""
        result = parse_value_list("moons, spirals, clusters")
        assert result == ["moons", "spirals", "clusters"]

    def test_single_string_no_comma(self):
        """Test parsing single string value without comma."""
        result = parse_value_list("spirals")
        assert result == ["spirals"]

    def test_list_input(self):
        """Test parsing list input (should return as-is)."""
        input_list = [1, 2, 4, 12, 16]
        result = parse_value_list(input_list)
        assert result == input_list

    def test_single_numeric_value(self):
        """Test parsing single numeric values."""
        assert parse_value_list(42) == [42]
        assert parse_value_list(3.14) == [3.14]

    def test_empty_string(self):
        """Test parsing empty string."""
        result = parse_value_list("")
        assert result == [""]


class TestValidateSweepConfig:
    """Test the validate_sweep_config function."""

    def test_valid_config(self):
        """Test validation with valid configuration."""
        valid_args = get_valid_argument_names()
        config = {"num_layers": [4, 8, 16], "hidden_dim": [64, 128], "lr": [0.001, 0.003]}
        # Should not raise an exception
        validate_sweep_config(config, valid_args)

    def test_invalid_parameter(self):
        """Test validation with invalid parameter name."""
        valid_args = get_valid_argument_names()
        config = {"invalid_param": [1, 2, 3], "num_layers": [4, 8]}
        with pytest.raises(ValueError, match="Unknown parameter in sweep config: 'invalid_param'"):
            validate_sweep_config(config, valid_args)

    def test_dashed_parameter_names(self):
        """Test validation handles dashed parameter names."""
        valid_args = get_valid_argument_names()
        config = {"--num_layers": [4, 8], "hidden_dim": [64, 128]}  # With dashes  # Without dashes
        # Should not raise an exception (dashes are stripped)
        validate_sweep_config(config, valid_args)

    def test_empty_config(self):
        """Test validation with empty configuration."""
        valid_args = get_valid_argument_names()
        config: dict[str, Any] = {}
        # Should not raise an exception
        validate_sweep_config(config, valid_args)


class TestLoadSweepConfigs:
    """Test the load_sweep_configs function."""

    def test_load_single_yaml_file(self):
        """Test loading a single YAML file."""
        config_data = {"num_layers": [4, 8, 16], "hidden_dim": [64, 128]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            configs = load_sweep_configs(temp_path)
            assert len(configs) == 1
            assert configs[0] == config_data
        finally:
            os.unlink(temp_path)

    def test_load_multiple_yaml_files(self):
        """Test loading multiple YAML files from a directory."""
        config1 = {"num_layers": [4, 8]}
        config2 = {"hidden_dim": [64, 128]}

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create two YAML files
            config1_path = Path(temp_dir) / "config1.yaml"
            config2_path = Path(temp_dir) / "config2.yml"

            with open(config1_path, "w", encoding="utf-8") as f:
                yaml.dump(config1, f)
            with open(config2_path, "w", encoding="utf-8") as f:
                yaml.dump(config2, f)

            configs = load_sweep_configs(temp_dir)
            assert len(configs) == 2
            # Should be sorted by filename
            assert configs[0] == config1
            assert configs[1] == config2

    def test_invalid_file_extension(self):
        """Test error handling for invalid file extension."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some text")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Sweep config file must have .yml or .yaml extension"):
                load_sweep_configs(temp_path)
        finally:
            os.unlink(temp_path)

    def test_nonexistent_path(self):
        """Test error handling for nonexistent path."""
        with pytest.raises(ValueError, match="Sweep config path does not exist"):
            load_sweep_configs("/nonexistent/path")

    def test_directory_with_no_yaml_files(self):
        """Test error handling for directory with no YAML files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a non-YAML file
            non_yaml_path = Path(temp_dir) / "config.txt"
            with open(non_yaml_path, "w", encoding="utf-8") as f:
                f.write("not yaml")

            with pytest.raises(ValueError, match="No YAML files found in directory"):
                load_sweep_configs(temp_dir)


class TestExpandGrid:
    """Test the expand_grid function."""

    def test_single_parameter_multiple_values(self):
        """Test grid expansion with single parameter having multiple values."""
        config = {"num_layers": [4, 8, 16]}
        grid = expand_grid(config)
        expected = [{"num_layers": 4}, {"num_layers": 8}, {"num_layers": 16}]
        assert grid == expected

    def test_multiple_parameters(self):
        """Test grid expansion with multiple parameters."""
        config = {"num_layers": [4, 8], "hidden_dim": [64, 128]}
        grid = expand_grid(config)
        expected = [
            {"num_layers": 4, "hidden_dim": 64},
            {"num_layers": 4, "hidden_dim": 128},
            {"num_layers": 8, "hidden_dim": 64},
            {"num_layers": 8, "hidden_dim": 128},
        ]
        assert grid == expected

    def test_mixed_single_and_multiple_values(self):
        """Test grid expansion with mix of single and multiple values."""
        config = {"num_layers": [4, 8], "hidden_dim": 128, "lr": [0.001, 0.003]}  # Single value
        grid = expand_grid(config)
        expected = [
            {"num_layers": 4, "hidden_dim": 128, "lr": 0.001},
            {"num_layers": 4, "hidden_dim": 128, "lr": 0.003},
            {"num_layers": 8, "hidden_dim": 128, "lr": 0.001},
            {"num_layers": 8, "hidden_dim": 128, "lr": 0.003},
        ]
        assert grid == expected

    def test_comma_separated_strings(self):
        """Test grid expansion with comma-separated string values."""
        config = {"num_layers": "4,8,16", "problem_type": "moons,spirals"}
        grid = expand_grid(config)
        assert len(grid) == 6  # 3 × 2 combinations
        assert {"num_layers": "4", "problem_type": "moons"} in grid
        assert {"num_layers": "16", "problem_type": "spirals"} in grid

    def test_empty_config(self):
        """Test grid expansion with empty configuration."""
        config: dict[str, Any] = {}
        grid = expand_grid(config)
        assert grid == [{}]


class TestMergeArgsWithCombo:
    """Test the merge_args_with_combo function."""

    def test_override_existing_args(self):
        """Test merging combo that overrides existing arguments."""
        base_args = argparse.Namespace(num_layers=8, hidden_dim=128, lr=0.001, problem_type="spirals")
        combo = {"num_layers": "4", "lr": "0.003"}

        merged = merge_args_with_combo(base_args, combo)

        assert merged.num_layers == 4  # Converted to int
        assert merged.lr == pytest.approx(0.003)  # Converted to float
        assert merged.hidden_dim == 128  # Unchanged
        assert merged.problem_type == "spirals"  # Unchanged

    def test_add_new_args(self):
        """Test merging combo that adds new arguments."""
        base_args = argparse.Namespace(num_layers=8, hidden_dim=128)
        combo = {"new_param": "value", "another_param": 42}

        merged = merge_args_with_combo(base_args, combo)

        assert merged.num_layers == 8  # Unchanged
        assert merged.hidden_dim == 128  # Unchanged
        assert merged.new_param == "value"  # Added
        assert merged.another_param == 42  # Added

    def test_boolean_conversion(self):
        """Test boolean value conversion."""
        base_args = argparse.Namespace(use_feature=False)

        # Test string boolean values
        for true_val in ["true", "True", "1", "yes", "on"]:
            combo = {"use_feature": true_val}
            merged = merge_args_with_combo(base_args, combo)
            assert merged.use_feature is True

        for false_val in ["false", "False", "0", "no", "off"]:
            combo = {"use_feature": false_val}
            merged = merge_args_with_combo(base_args, combo)
            assert merged.use_feature is False

    def test_type_preservation(self):
        """Test that type conversion respects original argument types."""
        base_args = argparse.Namespace(int_param=10, float_param=1.5, str_param="hello", bool_param=True)
        combo = {
            "int_param": "20",
            "float_param": "2.7",
            "str_param": "world",
            "bool_param": "false",
        }

        merged = merge_args_with_combo(base_args, combo)

        assert isinstance(merged.int_param, int) and merged.int_param == 20
        assert isinstance(merged.float_param, float) and merged.float_param == pytest.approx(2.7)
        assert isinstance(merged.str_param, str) and merged.str_param == "world"
        assert isinstance(merged.bool_param, bool) and merged.bool_param is False


class TestGenerateRunSlug:
    """Test the generate_run_slug function."""

    def test_generates_consistent_slug(self):
        """Test that the same combo generates the same slug."""
        combo = {"num_layers": 4, "hidden_dim": 128}
        slug1 = generate_run_slug(combo, 1)
        slug2 = generate_run_slug(combo, 1)
        assert slug1 == slug2

    def test_different_combos_different_slugs(self):
        """Test that different combos generate different slugs."""
        combo1 = {"num_layers": 4, "hidden_dim": 128}
        combo2 = {"num_layers": 8, "hidden_dim": 64}
        slug1 = generate_run_slug(combo1, 1)
        slug2 = generate_run_slug(combo2, 1)
        assert slug1 != slug2

    def test_different_indices_different_slugs(self):
        """Test that different indices generate different slugs."""
        combo = {"num_layers": 4, "hidden_dim": 128}
        slug1 = generate_run_slug(combo, 1)
        slug2 = generate_run_slug(combo, 2)
        assert slug1 != slug2

    def test_slug_format(self):
        """Test the format of generated slugs."""
        combo = {"num_layers": 4}
        slug = generate_run_slug(combo, 1)
        assert slug.startswith("run_001_")
        assert len(slug.split("_")) == 3


class TestGetValidArgumentNames:
    """Test the get_valid_argument_names function."""

    def test_contains_expected_args(self):
        """Test that valid args contain expected parameter names."""
        valid_args = get_valid_argument_names()

        # Check some key arguments are present
        expected_args = {
            "num_layers",
            "seeds_per_layer",
            "hidden_dim",
            "lr",
            "problem_type",
            "n_samples",
            "input_dim",
            "device",
            "sweep_config",
            "blend_steps",
            "shadow_lr",
        }

        for arg in expected_args:
            assert arg in valid_args, f"Expected argument '{arg}' not found in valid args"

    def test_returns_set(self):
        """Test that function returns a set."""
        valid_args = get_valid_argument_names()
        assert isinstance(valid_args, set)


class TestCLISweepIntegration:
    """Test integration of sweep functionality with CLI parsing."""

    def test_sweep_config_argument_parsing(self):
        """Test that --sweep_config argument is parsed correctly."""
        with patch("sys.argv", ["test", "--sweep_config", "config.yaml"]):
            args = parse_arguments()
            assert args.sweep_config == "config.yaml"

    def test_sweep_config_short_flag(self):
        """Test that -s short flag works for sweep config."""
        with patch("sys.argv", ["test", "-s", "config.yaml"]):
            args = parse_arguments()
            assert args.sweep_config == "config.yaml"

    def test_no_sweep_config_default(self):
        """Test default value when no sweep config is provided."""
        with patch("sys.argv", ["test"]):
            args = parse_arguments()
            assert args.sweep_config is None


class TestSweepExecution:
    """Test the actual sweep execution functionality."""

    @patch("morphogenetic_engine.runners.run_single_experiment")
    @patch("morphogenetic_engine.sweeps.runner.load_sweep_configs")
    @patch("morphogenetic_engine.sweeps.runner.create_sweep_results_summary")
    def test_run_parameter_sweep_basic(self, mock_summary, mock_load, mock_run):
        """Test basic parameter sweep execution."""
        # Setup mocks
        mock_load.return_value = [{"num_layers": [4, 8], "hidden_dim": [64, 128]}]
        mock_run.return_value = {"run_id": "test_run", "best_acc": 0.95, "seeds_activated": True}

        # Create mock args
        args = argparse.Namespace(
            sweep_config="test_config.yaml",
            num_layers=8,  # Base values
            hidden_dim=128,
            problem_type="spirals",
        )

        # Run the sweep
        with patch("pathlib.Path.mkdir"), patch("builtins.print"):
            run_parameter_sweep(args)

        # Verify sweep was executed
        assert mock_load.called
        assert mock_run.call_count == 4  # 2×2 grid
        assert mock_summary.called

    @patch("morphogenetic_engine.sweeps.runner.load_sweep_configs")
    def test_run_parameter_sweep_validation_error(self, mock_load):
        """Test sweep execution with validation errors."""
        # Setup mock to return invalid config
        mock_load.return_value = [{"invalid_param": [1, 2, 3]}]

        args = argparse.Namespace(sweep_config="test_config.yaml", problem_type="spirals")

        # Should handle validation error gracefully
        with patch("pathlib.Path.mkdir"), patch("builtins.print") as mock_print:
            run_parameter_sweep(args)

        # Check that error was printed
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Error in sweep config" in call for call in print_calls)

    @patch("morphogenetic_engine.sweeps.runner.load_sweep_configs")
    def test_run_parameter_sweep_load_error(self, mock_load):
        """Test sweep execution with config loading errors."""
        # Setup mock to raise a FileNotFoundError (more specific exception)
        mock_load.side_effect = FileNotFoundError("Config file not found")

        args = argparse.Namespace(sweep_config="nonexistent.yaml", problem_type="spirals")

        # Should handle loading error gracefully
        with patch("builtins.print") as mock_print:
            run_parameter_sweep(args)

        # Check that error was printed
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Error loading sweep config" in call for call in print_calls)


class TestSweepConfigExamples:
    """Test with realistic sweep configuration examples."""

    def test_hyperparameter_optimization_config(self):
        """Test a realistic hyperparameter optimization sweep config."""
        config = {
            "num_layers": [4, 8, 16],
            "seeds_per_layer": [1, 2, 4],
            "hidden_dim": [64, 128, 256],
            "lr": "0.001,0.003,0.01",
            "problem_type": ["moons", "spirals"],
        }

        # Validate and expand
        valid_args = get_valid_argument_names()
        validate_sweep_config(config, valid_args)

        grid = expand_grid(config)

        # Should generate 3×3×3×3×2 = 162 combinations
        assert len(grid) == 162

        # Check a few specific combinations
        assert {
            "num_layers": 4,
            "seeds_per_layer": 1,
            "hidden_dim": 64,
            "lr": "0.001",
            "problem_type": "moons",
        } in grid
        assert {
            "num_layers": 16,
            "seeds_per_layer": 4,
            "hidden_dim": 256,
            "lr": "0.01",
            "problem_type": "spirals",
        } in grid

    def test_architecture_search_config(self):
        """Test an architecture search focused sweep config."""
        config = {
            "num_layers": [2, 4, 6, 8, 12, 16],
            "seeds_per_layer": [1, 2, 3, 4],
            "hidden_dim": [32, 64, 128, 256],
            "blend_steps": "10,20,30,50",
        }

        valid_args = get_valid_argument_names()
        validate_sweep_config(config, valid_args)

        grid = expand_grid(config)

        # Should generate 6×4×4×4 = 384 combinations
        assert len(grid) == 384

    def test_single_parameter_sweep(self):
        """Test sweep with only one parameter varying."""
        config = {"lr": [0.0001, 0.0003, 0.001, 0.003, 0.01]}

        valid_args = get_valid_argument_names()
        validate_sweep_config(config, valid_args)

        grid = expand_grid(config)

        # Should generate 5 combinations
        assert len(grid) == 5
        assert all("lr" in combo and len(combo) == 1 for combo in grid)
