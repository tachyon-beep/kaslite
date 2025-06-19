"""
Refactored test suite for CLI argument parsing and main function.

This module contains tests for the morphogenetic architecture CLI interface,
focusing on clean separation between unit and integration tests.

Key improvements:
- Eliminated import conflicts
- Consolidated duplicate MockArgs classes into fixtures
- Separated unit tests (argument parsing) from integration tests
- Reduced overmocking by focusing tests on their core purpose
- Added comprehensive error condition testing

Test Classes:
    TestArgumentParsing: Pure unit tests for CLI argument parsing
    TestModelIntegration: Integration tests for model building with parsed arguments
    TestErrorConditions: Error handling and edge case testing
"""

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
import torch

from morphogenetic_engine.cli.arguments import parse_experiment_arguments
from morphogenetic_engine.core import KasminaMicro, SeedManager
from morphogenetic_engine.experiment import build_model_and_agents
from scripts.run_morphogenetic_experiment import main


@dataclass
class MockArgs:
    """Centralized mock arguments factory for testing."""

    # Core experiment parameters
    problem_type: str = "moons"
    hidden_dim: int = 64
    input_dim: int = 2
    num_layers: int = 8
    seeds_per_layer: int = 1

    # Training parameters
    lr: float = 1e-3
    batch_size: int = 32
    warm_up_epochs: int = 5
    adaptation_epochs: int = 10

    # Morphogenetic parameters
    blend_steps: int = 30
    shadow_lr: float = 1e-3
    progress_thresh: float = 0.6
    drift_warn: float = 0.12
    acc_threshold: float = 0.95

    # Other parameters
    device: str = "cpu"

    @classmethod
    def with_architecture(cls, num_layers: int, seeds_per_layer: int) -> "MockArgs":
        """Create mock args with specific architecture parameters."""
        return cls(num_layers=num_layers, seeds_per_layer=seeds_per_layer)

    @classmethod
    def for_dataset(cls, problem_type: str, input_dim: int = 2) -> "MockArgs":
        """Create mock args for a specific dataset type."""
        return cls(problem_type=problem_type, input_dim=input_dim)


class TestArgumentParsing:
    """Pure unit tests for CLI argument parsing without external dependencies."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        with patch("sys.argv", ["test_script"]):
            args = parse_experiment_arguments()

            assert args.num_layers == 8
            assert args.seeds_per_layer == 1
            assert args.hidden_dim == 128
            assert args.problem_type == "spirals"  # Actual default is spirals, not moons

    @pytest.mark.parametrize(
        "num_layers,expected",
        [
            ("4", 4),
            ("12", 12),
            ("1", 1),
            ("20", 20),
        ],
    )
    def test_num_layers_parsing(self, num_layers: str, expected: int):
        """Test --num_layers argument parsing with various values."""
        with patch("sys.argv", ["test", "--num_layers", num_layers]):
            args = parse_experiment_arguments()
            assert args.num_layers == expected

    @pytest.mark.parametrize(
        "seeds_per_layer,expected",
        [
            ("1", 1),
            ("3", 3),
            ("5", 5),
            ("10", 10),
        ],
    )
    def test_seeds_per_layer_parsing(self, seeds_per_layer: str, expected: int):
        """Test --seeds_per_layer argument parsing with various values."""
        with patch("sys.argv", ["test", "--seeds_per_layer", seeds_per_layer]):
            args = parse_experiment_arguments()
            assert args.seeds_per_layer == expected

    def test_combined_architecture_flags(self):
        """Test that architecture flags work correctly together."""
        with patch("sys.argv", ["test", "--num_layers", "6", "--seeds_per_layer", "2"]):
            args = parse_experiment_arguments()
            assert args.num_layers == 6
            assert args.seeds_per_layer == 2

    @pytest.mark.parametrize(
        "problem_type,n_samples",
        [
            ("spirals", 100),
            ("moons", 200),
            ("clusters", 150),
            ("spheres", 300),
            ("complex_moons", 250),
        ],
    )
    def test_problem_type_dispatch(self, problem_type: str, n_samples: int):
        """Test that different problem types are parsed correctly."""
        test_args = ["test", "--problem_type", problem_type, "--n_samples", str(n_samples)]

        with patch("sys.argv", test_args):
            args = parse_experiment_arguments()
            assert args.problem_type == problem_type
            assert args.n_samples == n_samples

    def test_irrelevant_flags_handling(self):
        """Test that irrelevant flags are parsed but don't interfere."""
        test_args = [
            "test",
            "--problem_type",
            "spirals",
            "--cluster_count",
            "5",  # Irrelevant to spirals
            "--sphere_radii",
            "1,2,3",  # Irrelevant to spirals
            "--noise",
            "0.3",  # Relevant to spirals
        ]

        with patch("sys.argv", test_args):
            args = parse_experiment_arguments()
            assert args.problem_type == "spirals"
            assert np.isclose(args.noise, 0.3)
            assert args.cluster_count == 5  # Still parsed
            assert args.sphere_radii == [1.0, 2.0, 3.0]  # Still parsed


class TestModelIntegration:
    """Integration tests for model building with parsed CLI arguments."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Provide CPU device for testing."""
        return torch.device("cpu")

    def test_build_model_with_default_args(self, device: torch.device):
        """Test model building with default arguments."""
        args = MockArgs()

        model, seed_manager, loss_fn, kasmina = build_model_and_agents(args, device)

        # Verify model architecture
        assert model.num_layers == 8
        assert model.seeds_per_layer == 1
        assert model.get_total_seeds() == 8

        # Verify components
        assert isinstance(seed_manager, SeedManager)
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
        assert isinstance(kasmina, KasminaMicro)
        assert next(model.parameters()).device == device

    def test_build_model_with_custom_architecture(self, device: torch.device):
        """Test model building with custom architecture parameters."""
        args = MockArgs.with_architecture(num_layers=5, seeds_per_layer=3)

        model, _, _, _ = build_model_and_agents(args, device)

        assert model.num_layers == 5
        assert model.seeds_per_layer == 3
        assert model.get_total_seeds() == 15
        assert len(model.layers) == 5
        assert len(model.all_seeds) == 15

    @pytest.mark.parametrize("dataset_type", ["moons", "spirals", "clusters", "complex_moons", "spheres"])
    def test_model_building_across_datasets(self, dataset_type: str, device: torch.device):
        """Test that model building works consistently across dataset types."""
        args = MockArgs.for_dataset(dataset_type)
        args.num_layers = 3
        args.seeds_per_layer = 2

        model, _, _, _ = build_model_and_agents(args, device)

        # Architecture should be consistent regardless of dataset
        assert model.num_layers == 3
        assert model.seeds_per_layer == 2
        assert model.get_total_seeds() == 6

    def test_seed_organization_in_model(self, device: torch.device):
        """Test that seeds are correctly organized by layer."""
        args = MockArgs.with_architecture(num_layers=2, seeds_per_layer=3)

        model, _, _, _ = build_model_and_agents(args, device)

        # Test layer 0 seeds
        layer_0_seeds = model.get_seeds_for_layer(0)
        assert len(layer_0_seeds) == 3
        assert [s.seed_id for s in layer_0_seeds] == ["seed1_1", "seed1_2", "seed1_3"]

        # Test layer 1 seeds
        layer_1_seeds = model.get_seeds_for_layer(1)
        assert len(layer_1_seeds) == 3
        assert [s.seed_id for s in layer_1_seeds] == ["seed2_1", "seed2_2", "seed2_3"]

    def test_backward_compatibility(self, device: torch.device):
        """Test that default values maintain backward compatibility."""
        args = MockArgs()  # Uses defaults: 8 layers, 1 seed per layer

        model, _, _, _ = build_model_and_agents(args, device)

        # Should match legacy hardcoded behavior
        assert model.num_layers == 8
        assert model.seeds_per_layer == 1
        assert model.get_total_seeds() == 8

        # Test backward compatibility property
        assert len(model.seeds) == 8  # all_seeds property
        assert len(model.get_all_seeds()) == 8
        assert list(model.seeds) == model.get_all_seeds()


class TestErrorConditions:
    """Test error handling and edge cases for CLI arguments."""

    def test_invalid_num_layers_values(self):
        """Test handling of invalid --num_layers values."""
        # Note: argparse converts string to int, so "0" becomes 0, "-1" becomes -1
        # These are technically valid integers, so let's test model building instead
        invalid_cases = [
            ("abc", ValueError),  # Non-numeric strings should cause argparse to fail
            ("", ValueError),  # Empty string should cause argparse to fail
        ]

        for invalid_value, expected_exception in invalid_cases:
            with patch("sys.argv", ["test", "--num_layers", invalid_value]):
                with pytest.raises((SystemExit, expected_exception)):
                    parse_experiment_arguments()

    def test_invalid_seeds_per_layer_values(self):
        """Test handling of invalid --seeds_per_layer values."""
        # Similar to num_layers, test non-numeric values that argparse will reject
        invalid_cases = [
            ("1.5", ValueError),  # Float strings might be rejected by argparse int type
            ("xyz", ValueError),  # Non-numeric strings
        ]

        for invalid_value, expected_exception in invalid_cases:
            with patch("sys.argv", ["test", "--seeds_per_layer", invalid_value]):
                with pytest.raises((SystemExit, expected_exception)):
                    parse_experiment_arguments()

    def test_missing_required_dependencies_for_model_building(self):
        """Test error handling when required model building dependencies are missing."""
        # Create args missing critical attributes
        incomplete_args = type(
            "Args",
            (),
            {
                "hidden_dim": 64,
                "num_layers": 3,
                # Missing 'seeds_per_layer', 'progress_thresh', etc.
            },
        )()

        device = torch.device("cpu")

        with pytest.raises(AttributeError):
            build_model_and_agents(incomplete_args, device)

    @pytest.mark.parametrize(
        "num_layers,seeds_per_layer",
        [
            (1, 1),  # Minimum valid values
            (50, 10),  # Large valid values
            (1, 20),  # Few layers, many seeds
            (20, 1),  # Many layers, few seeds
        ],
    )
    def test_extreme_but_valid_architecture_values(self, num_layers: int, seeds_per_layer: int):
        """Test that extreme but valid architecture values work correctly."""
        args = MockArgs.with_architecture(num_layers, seeds_per_layer)
        device = torch.device("cpu")

        model, _, _, _ = build_model_and_agents(args, device)

        assert model.num_layers == num_layers
        assert model.seeds_per_layer == seeds_per_layer
        assert model.get_total_seeds() == num_layers * seeds_per_layer

    def test_model_building_with_invalid_architecture_values(self):
        """Test that model building fails with invalid architecture values."""
        device = torch.device("cpu")

        # Test with zero or negative values (even if argparse accepts them)
        invalid_architectures = [
            (0, 1),  # Zero layers
            (-1, 1),  # Negative layers
            (1, 0),  # Zero seeds per layer
            (1, -1),  # Negative seeds per layer
        ]

        for num_layers, seeds_per_layer in invalid_architectures:
            args = MockArgs.with_architecture(num_layers, seeds_per_layer)

            # Model building should fail with invalid values
            with pytest.raises((ValueError, RuntimeError, IndexError)):
                build_model_and_agents(args, device)


class TestMainFunctionIntegration:
    """Focused integration tests for main function execution."""

    def test_main_with_minimal_args(self):
        """Test main function with minimal required arguments."""
        test_args = [
            "run_morphogenetic_experiment.py",
            "--warm_up_epochs",
            "1",
            "--adaptation_epochs",
            "1",
            "--hidden_dim",
            "32",
        ]

        with patch("sys.argv", test_args):
            with patch("morphogenetic_engine.runners.run_single_experiment") as mock_run:
                mock_run.return_value = {
                    "run_id": "test",
                    "best_acc": 0.8,
                    "seeds_activated": False,
                }

                try:
                    main()
                except SystemExit:
                    pass  # Normal exit from argparse

    def test_main_with_new_architecture_flags(self):
        """Test main function execution with new architecture flags."""
        test_args = [
            "run_morphogenetic_experiment.py",
            "--num_layers",
            "4",
            "--seeds_per_layer",
            "2",
            "--warm_up_epochs",
            "1",
            "--adaptation_epochs",
            "1",
        ]

        with patch("sys.argv", test_args):
            args = parse_experiment_arguments()

            # Verify new flags are parsed correctly
            assert args.num_layers == 4
            assert args.seeds_per_layer == 2

            # Verify model can be built with these arguments
            device = torch.device("cpu")
            args.input_dim = 2
            args.blend_steps = 30
            args.shadow_lr = 1e-3

            model, _, _, _ = build_model_and_agents(args, device)
            assert model.num_layers == 4
            assert model.seeds_per_layer == 2
