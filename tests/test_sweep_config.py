"""
Tests for the sweep configuration module.
"""

# pylint: disable=redefined-outer-name

import tempfile
from pathlib import Path

import pytest
import yaml

from morphogenetic_engine.sweeps.config import (
    SweepConfig,
    load_sweep_config,
    load_sweep_configs,
    parse_value_list,
)


# Test Fixtures
@pytest.fixture
def grid_config():
    """Standard grid search configuration for testing."""
    return {
        "sweep_type": "grid",
        "parameters": {"lr": [0.01, 0.001], "hidden_dim": [64, 128]},
        "experiment": {"problem_type": "spirals"},
    }


@pytest.fixture
def bayesian_config():
    """Standard Bayesian optimization configuration for testing."""
    return {
        "sweep_type": "bayesian",
        "parameters": {
            "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "batch_size": {"type": "int", "low": 16, "high": 256},
            "optimizer": {"type": "categorical", "choices": ["adam", "sgd"]},
        },
        "optimization": {"target_metric": "val_acc", "direction": "maximize"},
        "execution": {"max_parallel": 4, "timeout_per_trial": 1800},
    }


@pytest.fixture
def complex_grid_config():
    """Complex grid configuration with mixed types and experiment parameters."""
    return {
        "sweep_type": "grid",
        "parameters": {
            "lr": [0.1, 0.01, 0.001],
            "batch_size": [16, 32, 64],
            "optimizer": ["adam", "sgd"],
            "dropout": [0.1, 0.2, 0.5],
        },
        "experiment": {
            "problem_type": "spirals",
            "input_dim": 3,
            "n_samples": 2000,
        },
        "execution": {"max_parallel": 2},
        "optimization": {"target_metric": "loss", "direction": "minimize"},
    }


class TestSweepConfig:
    """Tests for SweepConfig class."""

    def test_grid_config_creation(self, grid_config):
        """Test creating a grid search configuration."""
        config = SweepConfig(grid_config)
        assert config.sweep_type == "grid"
        assert config.parameters["lr"] == [0.01, 0.001]
        assert config.experiment["problem_type"] == "spirals"

    def test_bayesian_config_creation(self, bayesian_config):
        """Test creating a Bayesian optimization configuration."""
        config = SweepConfig(bayesian_config)
        assert config.sweep_type == "bayesian"
        assert config.target_metric == "val_acc"
        assert config.direction == "maximize"

    def test_bayesian_parameter_types(self, bayesian_config):
        """Test various Bayesian parameter type definitions."""
        config = SweepConfig(bayesian_config)
        search_space = config.get_bayesian_search_space()

        # Check float parameter with log scale
        assert "lr" in search_space
        assert search_space["lr"]["type"] == "float"
        assert search_space["lr"]["log"] is True
        assert abs(search_space["lr"]["low"] - 1e-4) < 1e-10
        assert abs(search_space["lr"]["high"] - 1e-2) < 1e-10

        # Check integer parameter
        assert "batch_size" in search_space
        assert search_space["batch_size"]["type"] == "int"
        assert search_space["batch_size"]["low"] == 16
        assert search_space["batch_size"]["high"] == 256

        # Check categorical parameter
        assert "optimizer" in search_space
        assert search_space["optimizer"]["type"] == "categorical"
        assert search_space["optimizer"]["choices"] == ["adam", "sgd"]

    def test_property_defaults_and_overrides(self):
        """Test all property methods with defaults and custom values."""
        # Test defaults
        config_with_defaults = SweepConfig({"parameters": {"lr": [0.01]}})
        assert config_with_defaults.sweep_type == "grid"
        assert config_with_defaults.max_parallel == 1
        assert config_with_defaults.target_metric == "val_acc"
        assert config_with_defaults.direction == "maximize"
        assert config_with_defaults.timeout_per_trial == 3600

        # Test overrides
        config_with_values = SweepConfig(
            {
                "parameters": {"lr": [0.01]},
                "execution": {"max_parallel": 4, "timeout_per_trial": 1800},
                "optimization": {"target_metric": "loss", "direction": "minimize"},
            }
        )
        assert config_with_values.max_parallel == 4
        assert config_with_values.timeout_per_trial == 1800
        assert config_with_values.target_metric == "loss"
        assert config_with_values.direction == "minimize"

    def test_invalid_sweep_type(self):
        """Test that invalid sweep types raise an error."""
        config_dict = {"sweep_type": "invalid", "parameters": {"lr": [0.01]}}

        with pytest.raises(ValueError, match="Invalid sweep_type"):
            SweepConfig(config_dict)

    def test_empty_parameters(self):
        """Test that empty parameters raise an error."""
        config_dict = {"sweep_type": "grid", "parameters": {}}

        with pytest.raises(ValueError, match="Parameters section cannot be empty"):
            SweepConfig(config_dict)

    def test_invalid_bayesian_parameters(self):
        """Test that invalid Bayesian parameter definitions are stored but not validated until use."""
        # Note: SweepConfig doesn't validate Bayesian parameter structures during initialization
        # Validation happens later in the BayesianSearchRunner when creating Optuna distributions

        invalid_configs = [
            # Invalid parameter type (will be handled gracefully by runner)
            {
                "sweep_type": "bayesian",
                "parameters": {"lr": {"type": "invalid_type", "low": 1e-4, "high": 1e-2}},
            },
            # Missing fields (will get defaults in runner)
            {
                "sweep_type": "bayesian",
                "parameters": {"lr": {"type": "float", "high": 1e-2}},  # Missing 'low'
            },
            # Missing choices for categorical (will be handled by runner)
            {"sweep_type": "bayesian", "parameters": {"optimizer": {"type": "categorical"}}},
        ]

        # These should not raise errors during SweepConfig creation
        # The validation happens later in the BayesianSearchRunner
        for config_dict in invalid_configs:
            config = SweepConfig(config_dict)
            assert config.sweep_type == "bayesian"
            search_space = config.get_bayesian_search_space()
            assert isinstance(search_space, dict)

    def test_grid_combinations(self, grid_config):
        """Test generating grid combinations."""
        config = SweepConfig(grid_config)
        combinations = config.get_grid_combinations()

        assert len(combinations) == 4  # 2 * 2

        # Check that all combinations include experiment parameters
        for combo in combinations:
            assert "problem_type" in combo
            assert combo["problem_type"] == "spirals"
            assert "lr" in combo
            assert "hidden_dim" in combo

    def test_complex_grid_combinations(self, complex_grid_config):
        """Test generating combinations for complex grid configuration."""
        config = SweepConfig(complex_grid_config)
        combinations = config.get_grid_combinations()

        # 3 * 3 * 2 * 3 = 54 combinations
        assert len(combinations) == 54

        # Verify all combinations have required parameters
        for combo in combinations:
            # Grid parameters
            assert combo["lr"] in [0.1, 0.01, 0.001]
            assert combo["batch_size"] in [16, 32, 64]
            assert combo["optimizer"] in ["adam", "sgd"]
            assert combo["dropout"] in [0.1, 0.2, 0.5]

            # Experiment parameters should be included
            assert combo["problem_type"] == "spirals"
            assert combo["input_dim"] == 3
            assert combo["n_samples"] == 2000

        # Verify we have all unique combinations
        combo_tuples = {
            (combo["lr"], combo["batch_size"], combo["optimizer"], combo["dropout"])
            for combo in combinations
        }
        assert len(combo_tuples) == 54

    def test_parameter_precedence(self):
        """Test parameter precedence between experiment and parameters sections."""
        config_dict = {
            "sweep_type": "grid",
            "parameters": {"lr": [0.01, 0.001], "hidden_dim": 128},  # hidden_dim in both
            "experiment": {"hidden_dim": 64, "problem_type": "spirals"},  # Should be overridden
        }

        config = SweepConfig(config_dict)
        combinations = config.get_grid_combinations()

        # Should have 2 combinations (lr variations)
        assert len(combinations) == 2

        # All combinations should use the parameter value, not experiment value
        for combo in combinations:
            assert combo["hidden_dim"] == 128  # From parameters, not experiment
            assert combo["problem_type"] == "spirals"  # From experiment (not in parameters)

    def test_bayesian_grid_method_exclusivity(self, grid_config, bayesian_config):
        """Test that grid and Bayesian methods are mutually exclusive."""
        grid_config_obj = SweepConfig(grid_config)
        bayesian_config_obj = SweepConfig(bayesian_config)

        # Grid config should not allow Bayesian methods
        with pytest.raises(
            ValueError, match="Bayesian search space only available for bayesian sweep type"
        ):
            grid_config_obj.get_bayesian_search_space()

        # Bayesian config should not allow grid methods
        with pytest.raises(
            ValueError, match="Grid combinations only available for grid sweep type"
        ):
            bayesian_config_obj.get_grid_combinations()

    def test_default_values(self):
        """Test default configuration values."""
        config_dict = {"parameters": {"lr": [0.01]}}

        config = SweepConfig(config_dict)
        assert config.sweep_type == "grid"  # Default
        assert config.max_parallel == 1  # Default
        assert config.target_metric == "val_acc"  # Default


class TestParseValueList:
    """Tests for parse_value_list function."""

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            # List inputs
            ([1, 2, 3], [1, 2, 3]),
            (["a", "b", "c"], ["a", "b", "c"]),
            ([0.1, 0.2], [0.1, 0.2]),
            ([], []),
            # Comma-separated string inputs
            ("1,2,3", ["1", "2", "3"]),
            ("adam,sgd,rmsprop", ["adam", "sgd", "rmsprop"]),
            ("0.1,0.01,0.001", ["0.1", "0.01", "0.001"]),
            # String inputs with spaces
            ("a, b, c", ["a", "b", "c"]),
            (" 1 , 2 , 3 ", ["1", "2", "3"]),
            # Single value inputs
            ("single", ["single"]),
            (42, [42]),
            (3.14, [3.14]),
            (True, [True]),
            (None, [None]),
            # Edge cases
            ("", [""]),
            (",", ["", ""]),
            ("a,", ["a", ""]),
            (",b", ["", "b"]),
        ],
    )
    def test_parse_value_list_comprehensive(self, input_value, expected):
        """Test parse_value_list with comprehensive input scenarios."""
        result = parse_value_list(input_value)
        assert result == expected

    def test_parse_list(self):
        """Test parsing a list value."""
        result = parse_value_list([1, 2, 3])
        assert result == [1, 2, 3]

    def test_parse_comma_separated_string(self):
        """Test parsing comma-separated string."""
        result = parse_value_list("1,2,3")
        assert result == ["1", "2", "3"]

    def test_parse_single_string(self):
        """Test parsing a single string value."""
        result = parse_value_list("single")
        assert result == ["single"]

    def test_parse_single_number(self):
        """Test parsing a single number."""
        result = parse_value_list(42)
        assert result == [42]

    def test_parse_complex_objects(self):
        """Test parsing complex object types."""
        # Test with dictionary (should wrap in list)
        obj = {"key": "value"}
        result = parse_value_list(obj)
        assert result == [obj]

        # Test with custom object
        class CustomObject:
            """Test helper class for custom object parsing."""
            def __init__(self, value):
                self.value = value

        custom_obj = CustomObject("test")
        result = parse_value_list(custom_obj)
        assert result == [custom_obj]


class TestLoadSweepConfig:
    """Tests for loading sweep configurations from files."""

    def test_load_valid_config(self):
        """Test loading a valid YAML configuration."""
        config_data = {"sweep_type": "grid", "parameters": {"lr": [0.01, 0.001]}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = load_sweep_config(config_path)
            assert config.sweep_type == "grid"
            assert config.parameters["lr"] == [0.01, 0.001]
        finally:
            config_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_sweep_config(Path("nonexistent.yaml"))

    def test_load_invalid_extension(self):
        """Test loading a file with invalid extension."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="must have .yml or .yaml extension"):
                load_sweep_config(config_path)
        finally:
            config_path.unlink()

    def test_load_malformed_yaml(self):
        """Test handling of malformed YAML files."""
        malformed_content = "invalid: yaml: content: ["

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(malformed_content)
            config_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                load_sweep_config(config_path)
        finally:
            config_path.unlink()

    def test_load_empty_yaml(self):
        """Test loading an empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Write empty file
            config_path = Path(f.name)

        try:
            with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
                # Empty YAML loads as None, which causes AttributeError when SweepConfig tries to call .get()
                load_sweep_config(config_path)
        finally:
            config_path.unlink()

    def test_load_yaml_with_null_config(self):
        """Test loading YAML that results in None/null config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("null")
            config_path = Path(f.name)

        try:
            with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
                load_sweep_config(config_path)
        finally:
            config_path.unlink()

    def test_load_both_yml_and_yaml_extensions(self):
        """Test loading files with both .yml and .yaml extensions."""
        config_data = {"sweep_type": "grid", "parameters": {"lr": [0.01]}}

        # Test .yml extension
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            yml_path = Path(f.name)

        # Test .yaml extension
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = Path(f.name)

        try:
            yml_config = load_sweep_config(yml_path)
            yaml_config = load_sweep_config(yaml_path)

            assert yml_config.sweep_type == "grid"
            assert yaml_config.sweep_type == "grid"
        finally:
            yml_path.unlink()
            yaml_path.unlink()


class TestLoadSweepConfigs:
    """Tests for loading multiple sweep configurations from directories."""

    def test_load_single_config_file(self, tmp_path):
        """Test loading a single config file using load_sweep_configs."""
        config_data = {"sweep_type": "grid", "parameters": {"lr": [0.01, 0.001]}}
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        configs = load_sweep_configs(config_file)
        assert len(configs) == 1
        assert configs[0].sweep_type == "grid"
        assert configs[0].parameters["lr"] == [0.01, 0.001]

    def test_load_configs_from_directory(self, tmp_path):
        """Test loading multiple configs from directory."""
        # Create multiple config files
        grid_config_data = {"sweep_type": "grid", "parameters": {"lr": [0.01]}}
        bayesian_config_data = {
            "sweep_type": "bayesian",
            "parameters": {"lr": {"type": "float", "low": 1e-4, "high": 1e-2}},
        }

        (tmp_path / "grid_config.yaml").write_text(yaml.dump(grid_config_data))
        (tmp_path / "bayesian_config.yml").write_text(yaml.dump(bayesian_config_data))

        configs = load_sweep_configs(tmp_path)
        assert len(configs) == 2

        sweep_types = {config.sweep_type for config in configs}
        assert "grid" in sweep_types
        assert "bayesian" in sweep_types

    def test_load_configs_sorted_order(self, tmp_path):
        """Test that configs from directory are loaded in sorted order."""
        config_template = {"sweep_type": "grid", "parameters": {"lr": [0.01]}}

        # Create files in non-alphabetical order
        (tmp_path / "z_config.yaml").write_text(yaml.dump({**config_template, "test_id": "z"}))
        (tmp_path / "a_config.yaml").write_text(yaml.dump({**config_template, "test_id": "a"}))
        (tmp_path / "m_config.yaml").write_text(yaml.dump({**config_template, "test_id": "m"}))

        configs = load_sweep_configs(tmp_path)
        assert len(configs) == 3

        # Check they're loaded in alphabetical order
        test_ids = [config.raw_config.get("test_id") for config in configs]
        assert test_ids == ["a", "m", "z"]

    def test_load_configs_empty_directory(self, tmp_path):
        """Test loading configs from directory with no YAML files."""
        # Create some non-YAML files
        (tmp_path / "readme.txt").write_text("some content")
        (tmp_path / "data.json").write_text('{"key": "value"}')

        with pytest.raises(ValueError, match="No YAML files found in directory"):
            load_sweep_configs(tmp_path)

    def test_load_configs_nonexistent_path(self):
        """Test loading configs from non-existent path."""
        with pytest.raises(ValueError, match="Sweep config path does not exist"):
            load_sweep_configs(Path("nonexistent_directory"))

    def test_load_configs_mixed_extensions(self, tmp_path):
        """Test loading configs with mixed .yml and .yaml extensions."""
        grid_config_data = {"sweep_type": "grid", "parameters": {"lr": [0.01]}}
        bayesian_config_data = {
            "sweep_type": "bayesian",
            "parameters": {"lr": {"type": "float", "low": 1e-4, "high": 1e-2}},
        }

        (tmp_path / "config1.yml").write_text(yaml.dump(grid_config_data))
        (tmp_path / "config2.yaml").write_text(yaml.dump(bayesian_config_data))
        # Add non-YAML file that should be ignored
        (tmp_path / "config3.txt").write_text("ignored")

        configs = load_sweep_configs(tmp_path)
        assert len(configs) == 2
        sweep_types = {config.sweep_type for config in configs}
        assert "grid" in sweep_types
        assert "bayesian" in sweep_types
