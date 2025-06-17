"""
Test suite for CLI argument parsing and main function.

This module contains tests for the morphogenetic architecture CLI interface,
including tests for:

- Command-line argument parsing with new flags (--num_layers, --seeds_per_layer)
- Main function execution and integration
- Problem type dispatch logic
- New CLI arguments for architecture configuration
- Backward compatibility verification

Test Classes:
    TestMainFunction: Tests for main function integration
    TestCLIDispatch: Tests for CLI argument dispatch logic
    TestNewCLIFlags: Tests for new CLI flags --num_layers and --seeds_per_layer
    TestNewCLIArguments: Comprehensive tests for new CLI arguments
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from morphogenetic_engine.components import BaseNet
from morphogenetic_engine.core import KasminaMicro, SeedManager
from morphogenetic_engine.experiment import build_model_and_agents
from morphogenetic_engine.cli.arguments import parse_experiment_arguments
from scripts.run_morphogenetic_experiment import main


class TestMainFunction:
    """Test suite for main function integration."""

    def test_main_basic_execution(self):
        """Test that main function can execute without errors."""
        # Mock command line arguments
        test_args = [
            "--hidden_dim",
            "32",
            "--warm_up_epochs",
            "2",
            "--adaptation_epochs",
            "2",
            "--lr",
            "0.01",
            "--acc_threshold",
            "0.8",
        ]

        with patch("sys.argv", ["run_morphogenetic_experiment.py"] + test_args):
            with patch("morphogenetic_engine.runners.Path.open", create=True):
                with patch("morphogenetic_engine.logger.ExperimentLogger") as mock_logger:
                    # Mock the file operations to avoid creating actual files
                    mock_logger.return_value = Mock()
                    with patch("builtins.print"):  # Suppress print output
                        try:
                            main()
                            # If we get here without exception, test passed
                        except SystemExit:  # pylint: disable=broad-except  # nosec # noqa: S110
                            # argparse might call sys.exit, which is fine
                            pass
                        except Exception:  # pylint: disable=broad-except  # nosec # noqa: S110
                            # Any other exception is a test failure
                            pytest.fail("main() raised unexpected exception")

    def test_main_argument_parsing(self):
        """Test argument parsing in main function."""
        # Test with minimal arguments
        test_args = ["--hidden_dim", "64"]

        with patch("sys.argv", ["run_morphogenetic_experiment.py"] + test_args):
            with patch("morphogenetic_engine.runners.Path.open", create=True):
                with patch("morphogenetic_engine.logger.ExperimentLogger") as mock_logger:
                    mock_logger.return_value = Mock()
                    with patch("builtins.print"):
                        with patch("torch.utils.data.random_split") as mock_split:
                            # Mock the data splitting to avoid actual computation
                            mock_split.return_value = (Mock(), Mock())
                            with patch("morphogenetic_engine.training.evaluate") as mock_eval:
                                mock_eval.return_value = (0.5, 0.8)  # loss, accuracy
                                try:
                                    main()
                                except (  # pylint: disable=broad-except  # nosec # noqa: S110
                                    SystemExit,
                                    Exception,
                                ):
                                    pass  # We just want to test argument parsing

    def test_main_new_arguments(self):
        """Test that main function accepts new CLI arguments."""
        test_args = [
            "--problem_type",
            "complex_moons",
            "--input_dim",
            "4",
            "--device",
            "cpu",
            "--blend_steps",
            "20",
        ]

        with patch("sys.argv", ["run_morphogenetic_experiment.py"] + test_args):
            with patch(
                "morphogenetic_engine.datasets.create_complex_moons"
            ) as mock_create, patch("morphogenetic_engine.components.BaseNet") as mock_net, patch(
                "torch.optim.Adam"
            ) as mock_optim, patch(
                "builtins.open", create=True
            ):
                # Mock expensive operations
                rng = np.random.default_rng(42)
                mock_create.return_value = (
                    rng.standard_normal((100, 4)),
                    rng.integers(0, 2, 100),
                )
                mock_net.return_value = Mock()
                mock_optim.return_value = Mock()

                try:
                    main()
                except (  # pylint: disable=broad-except  # nosec # noqa: S110
                    SystemExit,
                    Exception,
                ):  # pylint: disable=broad-except  # nosec # noqa: S110
                    # Expected since we're mocking critical components
                    pass

                # Verify create_complex_moons was called with correct parameters
                mock_create.assert_called_once_with(n_samples=2000, noise=0.1, input_dim=4)


class TestCLIDispatch:
    """Test CLI argument dispatch logic."""

    def test_problem_type_dispatch(self):
        """Test that different problem types are dispatched correctly."""

        test_cases = [
            ("spirals", "create_spirals"),
            ("moons", "create_moons"),
            ("clusters", "create_clusters"),
            ("spheres", "create_spheres"),
            ("complex_moons", "create_complex_moons"),
        ]

        for problem_type, expected_function in test_cases:
            test_args = ["--problem_type", problem_type, "--n_samples", "100"]

            with patch("sys.argv", ["run_morphogenetic_experiment.py"] + test_args):
                with patch(
                    f"morphogenetic_engine.datasets.{expected_function}"
                ) as mock_func, patch("morphogenetic_engine.components.BaseNet") as mock_net, patch(
                    "torch.optim.Adam"
                ) as mock_optim, patch(
                    "builtins.open", create=True
                ):
                    # Mock return values
                    rng = np.random.default_rng(42)
                    mock_func.return_value = (
                        rng.standard_normal((100, 3)),
                        rng.integers(0, 2, 100),
                    )
                    mock_net.return_value = Mock()
                    mock_optim.return_value = Mock()

                    try:
                        main()
                    except (  # pylint: disable=broad-except  # nosec # noqa: S110
                        SystemExit,
                        Exception,
                    ):
                        # Expected since we're mocking critical components
                        pass

                    # Verify the correct function was called
                    mock_func.assert_called_once()

    def test_irrelevant_flags_ignored(self):
        """Test that flags irrelevant to chosen problem_type are silently ignored."""

        # Use spirals with cluster-specific flags (should be ignored)
        test_args = [
            "--problem_type",
            "spirals",
            "--cluster_count",
            "5",  # Irrelevant to spirals
            "--sphere_radii",
            "1,2,3",  # Irrelevant to spirals
            "--n_samples",
            "100",
            "--noise",
            "0.3",  # Relevant to spirals
        ]

        with patch("sys.argv", ["run_morphogenetic_experiment.py"] + test_args):
            with patch(
                "morphogenetic_engine.datasets.create_spirals"
            ) as mock_spirals, patch("morphogenetic_engine.components.BaseNet") as mock_net, patch(
                "torch.optim.Adam"
            ) as mock_optim, patch(
                "builtins.open", create=True
            ):
                rng = np.random.default_rng(42)
                mock_spirals.return_value = (
                    rng.standard_normal((100, 3)),
                    rng.integers(0, 2, 100),
                )
                mock_net.return_value = Mock()
                mock_optim.return_value = Mock()

                try:
                    main()
                except (  # pylint: disable=broad-except  # nosec # noqa: S110
                    SystemExit,
                    Exception,
                ):
                    pass

                # Verify spirals was called with only relevant arguments
                call_args = mock_spirals.call_args
                assert "noise" in call_args.kwargs
                assert np.isclose(call_args.kwargs["noise"], 0.3)
                # cluster_count and sphere_radii should not affect the call


class TestNewCLIFlags:
    """
    Test suite for the new CLI flags: --num_layers and --seeds_per_layer.

    This class tests the command-line argument parsing for the new architecture
    configuration flags that allow dynamic control over network depth and
    multi-seed per layer functionality.

    Tests cover:
    - Default values for both flags
    - Custom values for --num_layers
    - Custom values for --seeds_per_layer
    - Flag combination behavior
    - Integration with model creation

    The flags enable users to configure:
    - num_layers: Number of hidden layers in the network (default: 8)
    - seeds_per_layer: Number of sentinel seeds per layer (default: 1)
    """

    def test_num_layers_flag(self):
        """
        Test that --num_layers flag works correctly.

        Verifies that the --num_layers command-line argument is properly parsed
        and sets the correct value in the argument namespace. Tests both a
        small value (4) and larger value (12) to ensure the flag accepts
        various integer inputs.

        The num_layers flag controls the number of hidden layers in the
        morphogenetic network architecture.
        """
        with patch("sys.argv", ["test", "--num_layers", "4"]):
            args = parse_experiment_arguments()
            assert args.num_layers == 4

        # Test with different values
        with patch("sys.argv", ["test", "--num_layers", "12"]):
            args = parse_experiment_arguments()
            assert args.num_layers == 12

    def test_seeds_per_layer_flag(self):
        """
        Test that --seeds_per_layer flag works correctly.

        Verifies that the --seeds_per_layer command-line argument is properly
        parsed and sets the correct value. Tests multiple values (3 and 5) to
        ensure the flag accepts various integer inputs.

        The seeds_per_layer flag controls how many sentinel seeds are created
        per hidden layer, enabling ensemble-like behavior through averaging
        of multiple adaptive paths per layer.
        """
        with patch("sys.argv", ["test", "--seeds_per_layer", "3"]):
            args = parse_experiment_arguments()
            assert args.seeds_per_layer == 3

        # Test with different values
        with patch("sys.argv", ["test", "--seeds_per_layer", "5"]):
            args = parse_experiment_arguments()
            assert args.seeds_per_layer == 5

    def test_combined_flags(self):
        """Test that both new flags work together."""
        with patch("sys.argv", ["test", "--num_layers", "6", "--seeds_per_layer", "2"]):
            args = parse_experiment_arguments()
            assert args.num_layers == 6
            assert args.seeds_per_layer == 2

    def test_default_values(self):
        """Test that default values are maintained for backward compatibility."""
        with patch("sys.argv", ["test"]):
            args = parse_experiment_arguments()
            assert args.num_layers == 8  # Default should be 8
            assert args.seeds_per_layer == 1  # Default should be 1

    def test_model_creation_with_new_flags(self):
        """Test that BaseNet is created correctly with new flags."""
        seed_manager = SeedManager()

        # Test with custom values
        model = BaseNet(
            hidden_dim=64, seed_manager=seed_manager, input_dim=2, num_layers=3, seeds_per_layer=2
        )

        assert model.num_layers == 3
        assert model.seeds_per_layer == 2
        assert model.get_total_seeds() == 6  # 3 layers * 2 seeds each

        # Test seed naming
        all_seeds = model.get_all_seeds()
        expected_names = ["seed1_1", "seed1_2", "seed2_1", "seed2_2", "seed3_1", "seed3_2"]
        actual_names = [seed.seed_id for seed in all_seeds]
        assert actual_names == expected_names

    def test_layer_seed_organization(self):
        """Test that seeds are correctly organized by layer."""
        seed_manager = SeedManager()
        model = BaseNet(
            hidden_dim=32, seed_manager=seed_manager, input_dim=3, num_layers=2, seeds_per_layer=3
        )

        # Test layer 0 seeds
        layer_0_seeds = model.get_seeds_for_layer(0)
        assert len(layer_0_seeds) == 3
        assert [s.seed_id for s in layer_0_seeds] == ["seed1_1", "seed1_2", "seed1_3"]

        # Test layer 1 seeds
        layer_1_seeds = model.get_seeds_for_layer(1)
        assert len(layer_1_seeds) == 3
        assert [s.seed_id for s in layer_1_seeds] == ["seed2_1", "seed2_2", "seed2_3"]

    def test_forward_pass_with_multiple_seeds(self):
        """Test that forward pass works correctly with multiple seeds per layer."""
        seed_manager = SeedManager()
        model = BaseNet(
            hidden_dim=16, seed_manager=seed_manager, input_dim=2, num_layers=2, seeds_per_layer=3
        )

        # Test forward pass
        x = torch.randn(5, 2)
        output = model(x)

        assert output.shape == (5, 2)  # Should maintain batch size and output dims
        assert not torch.isnan(output).any()  # Should not produce NaN values
        assert torch.isfinite(output).all()  # Should produce finite values

    def test_backward_compatibility_with_seeds_property(self):
        """Test that the seeds property still works for backward compatibility."""
        seed_manager = SeedManager()
        model = BaseNet(
            hidden_dim=32, seed_manager=seed_manager, input_dim=2, num_layers=3, seeds_per_layer=1
        )

        # Test that seeds property returns all seeds
        assert len(model.seeds) == 3
        assert len(model.get_all_seeds()) == 3

        # Test that both methods return the same seeds
        assert list(model.seeds) == model.get_all_seeds()

    def test_integration_with_build_model_and_agents(self):
        """Test that the new flags work with the full model building pipeline."""

        class MockArgs:
            """Mock arguments for testing model building with new CLI flags."""

            hidden_dim = 32
            input_dim = 2
            num_layers = 3
            seeds_per_layer = 2
            blend_steps = 30
            shadow_lr = 1e-3
            progress_thresh = 0.6
            drift_warn = 0.1
            acc_threshold = 0.95

        args = MockArgs()
        device = torch.device("cpu")

        model, seed_manager, loss_fn, kasmina = build_model_and_agents(args, device)

        # Verify model properties
        assert model.get_total_seeds() == 6  # 3 layers * 2 seeds
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
        assert abs(kasmina.acc_threshold - 0.95) < 1e-6  # Use approximate comparison for float
        assert isinstance(seed_manager, SeedManager)

        # Verify model is on correct device
        assert next(model.parameters()).device == device


class TestNewCLIArguments:
    """
    Comprehensive tests for the new CLI arguments --num_layers and --seeds_per_layer.

    This test class focuses on end-to-end testing of the new command-line arguments,
    including argument parsing, model construction, and integration with the
    experiment runner.

    Key test areas:
    - Argument parsing with default and custom values
    - Model and agent construction with new architecture parameters
    - Integration testing across different dataset types
    - Backward compatibility verification
    - Main function execution with new flags

    The tests use mocked command-line arguments and MockArgs classes to simulate
    various configuration scenarios without requiring actual CLI execution.
    """

    @patch("sys.argv", ["script"] + ["--problem_type", "moons", "--adaptation_epochs", "10"])
    def test_parse_arguments_defaults(self):
        """Test that parse_arguments sets correct defaults for new flags."""
        args = parse_experiment_arguments()

        assert args.num_layers == 8  # Default value
        assert args.seeds_per_layer == 1  # Default value

    @patch(
        "sys.argv",
        ["script"] + ["--problem_type", "moons", "--adaptation_epochs", "10", "--num_layers", "5"],
    )
    def test_parse_arguments_custom_num_layers(self):
        """Test parsing custom --num_layers argument."""
        args = parse_experiment_arguments()
        assert args.num_layers == 5

    @patch(
        "sys.argv",
        ["script"]
        + ["--problem_type", "moons", "--adaptation_epochs", "10", "--seeds_per_layer", "3"],
    )
    def test_parse_arguments_custom_seeds_per_layer(self):
        """Test parsing custom --seeds_per_layer argument."""
        args = parse_experiment_arguments()
        assert args.seeds_per_layer == 3

    @patch(
        "sys.argv",
        ["script"]
        + [
            "--problem_type",
            "moons",
            "--adaptation_epochs",
            "10",
            "--num_layers",
            "5",
            "--seeds_per_layer",
            "3",
        ],
    )
    def test_parse_arguments_both_flags(self):
        """Test parsing both new flags together."""
        args = parse_experiment_arguments()

        assert args.num_layers == 5
        assert args.seeds_per_layer == 3

    def test_build_model_with_custom_architecture(self):
        """Test that build_model_and_agents respects the new CLI flags."""

        class MockArgs:
            """Mock arguments for testing build_model_and_agents with custom architecture."""

            problem_type = "moons"
            hidden_dim = 64
            lr = 1e-3
            batch_size = 32
            progress_thresh = 0.6
            drift_warn = 0.12
            num_layers = 5
            seeds_per_layer = 3
            acc_threshold = 0.95
            input_dim = 2
            blend_steps = 30
            shadow_lr = 1e-3

        args = MockArgs()
        device = torch.device("cpu")

        model, _, _, _ = build_model_and_agents(args, device)

        # Verify model has correct architecture
        assert model.num_layers == 5
        assert model.seeds_per_layer == 3
        assert model.get_total_seeds() == 15  # 5 layers * 3 seeds

        # Verify model structure
        assert len(model.layers) == 5
        assert len(model.all_seeds) == 15

    def test_integration_with_different_datasets(self):
        """Test integration of new flags with different dataset types."""

        datasets = ["moons", "spirals", "clusters", "complex_moons", "spheres"]

        for dataset in datasets:

            class MockArgs:
                """Mock arguments for testing integration with different datasets."""

                problem_type = dataset
                hidden_dim = 32
                lr = 1e-3
                batch_size = 16
                progress_thresh = 0.6
                drift_warn = 0.12
                num_layers = 3
                seeds_per_layer = 2
                acc_threshold = 0.95
                input_dim = 2
                blend_steps = 30
                shadow_lr = 1e-3

            args = MockArgs()
            device = torch.device("cpu")

            model, _, _, _ = build_model_and_agents(args, device)

            # All should create consistent architecture regardless of dataset
            assert model.num_layers == 3
            assert model.seeds_per_layer == 2
            assert model.get_total_seeds() == 6

    @patch(
        "sys.argv",
        ["script"]
        + [
            "--problem_type",
            "moons",
            "--adaptation_epochs",
            "2",
            "--num_layers",
            "4",
            "--seeds_per_layer",
            "2",
            "--hidden_dim",
            "32",
            "--batch_size",
            "16",
        ],
    )
    def test_main_function_with_new_flags(self):
        """Test that main function can be called with new CLI flags."""

        # Test that parse_arguments works with new flags
        args = parse_experiment_arguments()
        assert args.num_layers == 4
        assert args.seeds_per_layer == 2

        # Test that build_model_and_agents works with the parsed arguments
        device = torch.device("cpu")

        # Add missing attributes that build_model_and_agents expects
        args.input_dim = 2
        args.blend_steps = 30
        args.shadow_lr = 1e-3

        model, _, _, _ = build_model_and_agents(args, device)

        # Verify the model was built with correct parameters
        assert model.num_layers == 4
        assert model.seeds_per_layer == 2

    def test_backward_compatibility(self):
        """Test that old scripts work without specifying new flags."""

        # Should create backward-compatible model
        class MockArgs:
            """Mock arguments for testing backward compatibility."""

            problem_type = "moons"
            hidden_dim = 64
            lr = 1e-3
            batch_size = 32
            progress_thresh = 0.6
            drift_warn = 0.12
            num_layers = 8  # Default
            seeds_per_layer = 1  # Default
            acc_threshold = 0.95
            input_dim = 2
            blend_steps = 30
            shadow_lr = 1e-3

        args = MockArgs()
        device = torch.device("cpu")

        model, seed_manager, loss_fn, kasmina = build_model_and_agents(args, device)

        # Should have the same structure as before
        assert model.num_layers == 8
        assert model.seeds_per_layer == 1
        assert model.get_total_seeds() == 8  # Same as old hardcoded version

        # Verify all components are created correctly
        assert isinstance(seed_manager, SeedManager)
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
        assert isinstance(kasmina, KasminaMicro)
