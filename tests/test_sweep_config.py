"""
Tests for the sweep configuration module.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from morphogenetic_engine.sweeps.config import SweepConfig, load_sweep_config, parse_value_list


class TestSweepConfig:
    """Tests for SweepConfig class."""

    def test_grid_config_creation(self):
        """Test creating a grid search configuration."""
        config_dict = {
            "sweep_type": "grid",
            "parameters": {"lr": [0.01, 0.001], "hidden_dim": [64, 128]},
            "experiment": {"problem_type": "spirals"},
        }

        config = SweepConfig(config_dict)
        assert config.sweep_type == "grid"
        assert config.parameters["lr"] == [0.01, 0.001]
        assert config.experiment["problem_type"] == "spirals"

    def test_bayesian_config_creation(self):
        """Test creating a Bayesian optimization configuration."""
        config_dict = {
            "sweep_type": "bayesian",
            "parameters": {"lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
            "optimization": {"target_metric": "val_acc", "direction": "maximize"},
        }

        config = SweepConfig(config_dict)
        assert config.sweep_type == "bayesian"
        assert config.target_metric == "val_acc"
        assert config.direction == "maximize"

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

    def test_grid_combinations(self):
        """Test generating grid combinations."""
        config_dict = {
            "sweep_type": "grid",
            "parameters": {"lr": [0.01, 0.001], "hidden_dim": [64, 128]},
            "experiment": {"problem_type": "spirals"},
        }

        config = SweepConfig(config_dict)
        combinations = config.get_grid_combinations()

        assert len(combinations) == 4  # 2 * 2

        # Check that all combinations include experiment parameters
        for combo in combinations:
            assert "problem_type" in combo
            assert combo["problem_type"] == "spirals"
            assert "lr" in combo
            assert "hidden_dim" in combo

    def test_default_values(self):
        """Test default configuration values."""
        config_dict = {"parameters": {"lr": [0.01]}}

        config = SweepConfig(config_dict)
        assert config.sweep_type == "grid"  # Default
        assert config.max_parallel == 1  # Default
        assert config.target_metric == "val_acc"  # Default


class TestParseValueList:
    """Tests for parse_value_list function."""

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
