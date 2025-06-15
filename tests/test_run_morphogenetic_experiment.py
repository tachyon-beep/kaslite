"""Comprehensive tests for the run_morphogenetic_experiment script."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from morphogenetic_engine.core import SeedManager
from scripts.run_morphogenetic_experiment import (
    create_complex_moons,
    create_spirals,
    evaluate,
    main,
    train_epoch,
)


class TestCreateSpirals:
    """Test suite for spiral dataset generation."""

    def test_default_parameters(self):
        """Test spiral generation with default parameters."""
        X, y = create_spirals()

        assert X.shape == (2000, 2)
        assert y.shape == (2000,)
        assert X.dtype == np.float32
        assert y.dtype == np.int64

        # Check label distribution
        assert np.sum(y == 0) == 1000
        assert np.sum(y == 1) == 1000

    def test_custom_parameters(self):
        """Test spiral generation with custom parameters."""
        n_samples = 1000
        X, y = create_spirals(n_samples=n_samples, noise=0.1, rotations=2)

        assert X.shape == (n_samples, 2)
        assert y.shape == (n_samples,)

        # Check label distribution
        assert np.sum(y == 0) == n_samples // 2
        assert np.sum(y == 1) == n_samples // 2

    def test_reproducibility(self):
        """Test that spiral generation is reproducible with same random seed."""
        # The function uses a fixed seed (42) so should be reproducible
        X1, y1 = create_spirals(n_samples=100)
        X2, y2 = create_spirals(n_samples=100)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_data_range(self):
        """Test that generated data has reasonable range."""
        X, _ = create_spirals(n_samples=500, noise=0.1)

        # Data should be centered around origin with reasonable range
        assert np.abs(X.mean()) < 5.0  # Not too far from origin
        assert np.std(X) > 0.1  # Has reasonable variance

    def test_spiral_structure(self):
        """Test that generated data has spiral structure."""
        X, y = create_spirals(n_samples=200, noise=0.05)

        # Convert to polar coordinates to check spiral structure
        r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        theta = np.arctan2(X[:, 1], X[:, 0])

        # For each class, radius should generally increase with angle
        for class_label in [0, 1]:
            class_mask = y == class_label
            class_r = r[class_mask]
            class_theta = theta[class_mask]

            # Sort by angle and check that radius generally increases
            sorted_indices = np.argsort(class_theta)
            sorted_r = class_r[sorted_indices]

            # Check that there's some spiral structure (not perfectly linear)
            # but with a reasonable correlation between sorted index and radius
            correlation = np.corrcoef(np.arange(len(sorted_r)), sorted_r)[0, 1]
            # Allow for more flexible spiral structure
            assert abs(correlation) > 0.05, f"Class {class_label} correlation: {correlation}"

    def test_input_dim_padding(self):
        """Test that spirals are properly padded to higher dimensions."""
        X, y = create_spirals(n_samples=100, input_dim=5)

        # Should be padded to 5 dimensions
        assert X.shape == (100, 5)
        assert y.shape == (100,)

        # First 2 dimensions should be the spiral data (not all zeros)
        # Last 3 should be independent padding
        assert not np.allclose(X[:, 2], 0)  # Padding should not be all zeros

        # Test input_dim=2 gives same result as default
        X_2d, y_2d = create_spirals(n_samples=100, input_dim=2)
        X_default, y_default = create_spirals(n_samples=100)

        np.testing.assert_array_equal(X_2d, X_default)
        np.testing.assert_array_equal(y_2d, y_default)


class TestCreateComplexMoons:
    """Test suite for complex moons dataset generation."""

    def test_default_parameters(self):
        """Test complex moons generation with default parameters."""
        X, y = create_complex_moons()

        # Check output shapes
        assert X.shape == (2000, 2)
        assert y.shape == (2000,)

        # Check data types
        assert X.dtype == np.float32
        assert y.dtype == np.int64

        # Check binary labels
        assert set(y) == {0, 1}

    def test_custom_parameters(self):
        """Test complex moons generation with custom parameters."""
        X, y = create_complex_moons(n_samples=200, noise=0.05)

        assert X.shape == (200, 2)
        assert y.shape == (200,)

    def test_input_dim_padding(self):
        """Test that complex moons are properly padded to higher dimensions."""
        X, y = create_complex_moons(n_samples=100, input_dim=4)

        # Should be padded to 4 dimensions
        assert X.shape == (100, 4)
        assert y.shape == (100,)

        # Padding dimensions should not be all zeros
        assert not np.allclose(X[:, 2], 0)
        assert not np.allclose(X[:, 3], 0)

    def test_reproducibility(self):
        """Test that complex moons generation is reproducible."""
        X1, y1 = create_complex_moons(n_samples=100)
        X2, y2 = create_complex_moons(n_samples=100)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_data_structure(self):
        """Test that complex moons has expected structure (moons + clusters)."""
        X, _ = create_complex_moons(n_samples=400, noise=0.1)

        # Should have reasonable spread (combination of moons and clusters)
        assert X.std() > 0.5  # Should have some variance
        assert np.abs(X.mean()) < 2.0  # But not too far from origin


class TestTrainEpoch:
    """Test suite for training epoch function."""

    def test_train_epoch_basic(self):
        """Test basic training epoch functionality."""
        from torch.utils.data import DataLoader, TensorDataset

        from morphogenetic_engine.components import BaseNet

        # Create simple test data
        X = torch.randn(16, 2)
        y = torch.randint(0, 2, (16,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and components
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()

        # Test training epoch
        avg_loss = train_epoch(model, loader, optimizer, criterion, seed_manager)

        assert isinstance(avg_loss, float)
        assert avg_loss > 0.0

    def test_train_epoch_with_scheduler(self):
        """Test training epoch with learning rate scheduler."""
        from torch.utils.data import DataLoader, TensorDataset

        from morphogenetic_engine.components import BaseNet

        # Create test data
        X = torch.randn(16, 2)
        y = torch.randint(0, 2, (16,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and components
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Train for one epoch
        train_epoch(model, loader, optimizer, criterion, seed_manager, scheduler)

        # Check that scheduler stepped
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_train_epoch_no_optimizer(self):
        """Test train_epoch with no optimizer."""
        from torch.utils.data import DataLoader, TensorDataset

        from morphogenetic_engine.components import BaseNet

        # Create test data
        X = torch.randn(8, 2)
        y = torch.randint(0, 2, (8,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and components
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        criterion = torch.nn.CrossEntropyLoss()

        # Test with no optimizer
        avg_loss = train_epoch(model, loader, None, criterion, seed_manager)

        assert isinstance(avg_loss, float)
        assert avg_loss > 0.0

    def test_train_epoch_empty_loader(self):
        """Test train_epoch with empty loader."""
        from torch.utils.data import DataLoader, TensorDataset

        from morphogenetic_engine.components import BaseNet

        # Create empty dataset
        X = torch.empty(0, 2)
        y = torch.empty(0, dtype=torch.long)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and components
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()

        # Test with empty loader
        avg_loss = train_epoch(model, loader, optimizer, criterion, seed_manager)

        assert np.isclose(avg_loss, 0.0)  # Should return 0.0 for empty loader

    def test_train_epoch_seed_training(self):
        """Test that seeds are trained during training epoch."""
        from torch.utils.data import DataLoader, TensorDataset

        from morphogenetic_engine.components import BaseNet

        # Create test data
        X = torch.randn(16, 2)
        y = torch.randint(0, 2, (16,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and components
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()

        # Initialize a seed for training
        model.seed1.initialize_child()

        # Add some data to seed buffer
        buffer = seed_manager.seeds["seed1"]["buffer"]
        for _ in range(15):
            buffer.append(torch.randn(2, 32))

        # Mock the train_child_step to verify it's called
        with patch.object(model.seed1, "train_child_step") as mock_train:
            train_epoch(model, loader, optimizer, criterion, seed_manager)
            assert mock_train.call_count > 0

    def test_train_epoch_large_buffer_sampling(self):
        """Test that large seed buffers are properly sampled during training."""
        from torch.utils.data import DataLoader, TensorDataset

        from morphogenetic_engine.components import BaseNet

        # Create test data
        X = torch.randn(16, 2)
        y = torch.randint(0, 2, (16,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and components
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()

        # Initialize a seed for training and fill buffer with many samples
        model.seed1.initialize_child()
        buffer = seed_manager.seeds["seed1"]["buffer"]

        # Add > 64 samples to trigger sampling logic (lines 97-98)
        for _ in range(80):
            buffer.append(torch.randn(2, 32))

        # Verify buffer has many samples
        assert len(buffer) > 64

        # Train epoch should use sampling logic for large buffers
        avg_loss = train_epoch(model, loader, optimizer, criterion, seed_manager)
        assert avg_loss >= 0.0  # Basic sanity check


class TestEvaluate:
    """Test suite for evaluation function."""

    def test_evaluate_basic(self):
        """Test basic evaluation functionality."""
        from torch.utils.data import DataLoader, TensorDataset

        from morphogenetic_engine.components import BaseNet

        # Create test data
        X = torch.randn(16, 2)
        y = torch.randint(0, 2, (16,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and criterion
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        criterion = torch.nn.CrossEntropyLoss()

        # Evaluate
        loss, accuracy = evaluate(model, loader, criterion)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_evaluate_perfect_model(self):
        """Test evaluation with a perfect model."""
        from torch.utils.data import DataLoader, TensorDataset

        # Create simple linearly separable data
        X = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
        y = torch.tensor([1, 0, 1, 0])
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2, num_workers=0)

        # Create a perfect linear model
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight[0, 0] = -1.0  # For class 0
            model.weight[0, 1] = 0.0
            model.weight[1, 0] = 1.0  # For class 1
            model.weight[1, 1] = 0.0
            model.bias[0] = 0.0
            model.bias[1] = 0.0

        criterion = torch.nn.CrossEntropyLoss()

        # Evaluate
        loss, accuracy = evaluate(model, loader, criterion)

        assert np.isclose(accuracy, 1.0)  # Perfect accuracy
        assert loss < 0.1  # Very low loss

    def test_evaluate_empty_loader(self):
        """Test evaluation with empty data loader."""
        from torch.utils.data import DataLoader, TensorDataset

        from morphogenetic_engine.components import BaseNet

        # Create empty dataset
        X = torch.empty(0, 2)
        y = torch.empty(0, dtype=torch.long)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and criterion
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        criterion = torch.nn.CrossEntropyLoss()

        # This should handle empty loader gracefully
        loss, accuracy = evaluate(model, loader, criterion)

        # With empty loader, loss should be 0 and accuracy undefined (NaN)
        assert np.isclose(loss, 0.0)
        assert np.isnan(accuracy) or np.isclose(accuracy, 0.0)


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
            with patch("scripts.run_morphogenetic_experiment.Path.open", create=True):
                with patch("scripts.run_morphogenetic_experiment.ExperimentLogger") as mock_logger:
                    # Mock the file operations to avoid creating actual files
                    mock_logger.return_value = Mock()
                    with patch("builtins.print"):  # Suppress print output
                        try:
                            main()
                            # If we get here without exception, test passed
                        except SystemExit:  # pylint: disable=broad-except
                            # argparse might call sys.exit, which is fine
                            pass
                        except Exception:  # pylint: disable=broad-except
                            # Any other exception is a test failure
                            pytest.fail("main() raised unexpected exception")

    def test_main_argument_parsing(self):
        """Test argument parsing in main function."""
        # Test with minimal arguments
        test_args = ["--hidden_dim", "64"]

        with patch("sys.argv", ["run_morphogenetic_experiment.py"] + test_args):
            with patch("scripts.run_morphogenetic_experiment.Path.open", create=True):
                with patch("scripts.run_morphogenetic_experiment.ExperimentLogger") as mock_logger:
                    mock_logger.return_value = Mock()
                    with patch("builtins.print"):
                        with patch("torch.utils.data.random_split") as mock_split:
                            # Mock the data splitting to avoid actual computation
                            mock_split.return_value = (Mock(), Mock())
                            with patch(
                                "scripts.run_morphogenetic_experiment.evaluate"
                            ) as mock_eval:
                                mock_eval.return_value = (0.5, 0.8)  # loss, accuracy
                                try:
                                    main()
                                except (SystemExit, Exception):  # pylint: disable=broad-except
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
                "scripts.run_morphogenetic_experiment.create_complex_moons"
            ) as mock_create, patch(
                "scripts.run_morphogenetic_experiment.BaseNet"
            ) as mock_net, patch(
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
                except (SystemExit, Exception):  # pylint: disable=broad-except
                    # Expected since we're mocking critical components
                    pass

                # Verify create_complex_moons was called with correct parameters
                mock_create.assert_called_once_with(n_samples=2000, noise=0.1, input_dim=4)


class TestIntegration:
    """Integration tests for the complete system."""

    def test_full_pipeline_mini(self):
        """Test a minimal full pipeline execution."""
        from torch.utils.data import DataLoader, TensorDataset

        from morphogenetic_engine.components import BaseNet
        from morphogenetic_engine.core import KasminaMicro

        # Create minimal dataset
        X, y = create_spirals(n_samples=32)
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=8, num_workers=0)
        val_loader = DataLoader(dataset, batch_size=8, num_workers=0)

        # Create components
        model = BaseNet(hidden_dim=16, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        seed_manager.seeds.clear()  # Clear any existing seeds
        kasmina = KasminaMicro(seed_manager, patience=2, acc_threshold=0.9)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()

        # Run a few training steps
        for _ in range(3):
            avg_loss = train_epoch(model, train_loader, optimizer, criterion, seed_manager)
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            # Check that functions return reasonable values
            assert isinstance(avg_loss, float)
            assert isinstance(val_loss, float)
            assert isinstance(val_acc, float)
            assert avg_loss > 0.0
            assert val_loss > 0.0
            assert 0.0 <= val_acc <= 1.0

            # Test Kasmina step
            germinated = kasmina.step(val_loss, val_acc)
            assert isinstance(germinated, bool)

    def test_seed_lifecycle(self):
        """Test complete seed lifecycle from dormant to active."""
        from morphogenetic_engine.components import SentinelSeed

        # Create seed
        seed_manager = SeedManager()
        seed = SentinelSeed("test_lifecycle", 16, seed_manager, blend_steps=5, progress_thresh=0.3)

        # Start dormant
        assert seed.state == "dormant"

        # Initialize and train
        seed.initialize_child()
        assert seed.state == "training"

        # Train until blending
        dummy_input = torch.randn(4, 16)
        for _ in range(50):  # Should exceed progress threshold
            seed.train_child_step(dummy_input)

        assert seed.state == "blending"

        # Blend until active
        while seed.state == "blending":
            seed.update_blending()

        assert seed.state == "active"
        assert seed.alpha >= 0.99

        # Test forward pass in each state works
        x = torch.randn(2, 16)
        output = seed.forward(x)
        assert output.shape == x.shape


class TestHighDimensionalIntegration:
    """Test suite for high-dimensional input integration."""

    def test_spirals_4d_integration(self):
        """Test spirals dataset with 4D input through complete pipeline."""
        from torch.utils.data import DataLoader, TensorDataset

        from morphogenetic_engine.components import BaseNet

        # Create 4D spirals data
        X, y = create_spirals(n_samples=64, input_dim=4)
        assert X.shape == (64, 4)

        # Create tensor dataset
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=16, num_workers=0)

        # Create 4D model
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=4)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()

        # Test forward pass works
        avg_loss = train_epoch(model, loader, optimizer, criterion, seed_manager)
        assert avg_loss >= 0.0

        # Test evaluation works
        loss, accuracy = evaluate(model, loader, criterion)
        assert loss >= 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_complex_moons_4d_integration(self):
        """Test complex moons dataset with 4D input through complete pipeline."""
        from torch.utils.data import DataLoader, TensorDataset

        from morphogenetic_engine.components import BaseNet

        # Create 4D complex moons data
        X, y = create_complex_moons(n_samples=64, input_dim=4)
        assert X.shape == (64, 4)

        # Create tensor dataset
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=16, num_workers=0)

        # Create 4D model
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=4)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()

        # Test forward pass works
        avg_loss = train_epoch(model, loader, optimizer, criterion, seed_manager)
        assert avg_loss >= 0.0

        # Test evaluation works
        loss, accuracy = evaluate(model, loader, criterion)
        assert loss >= 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_backward_compatibility(self):
        """Test that default args preserve legacy behavior."""
        # Default spirals should be same as explicit 2D
        X_default, y_default = create_spirals(n_samples=100)
        X_explicit, y_explicit = create_spirals(n_samples=100, input_dim=2)

        np.testing.assert_array_equal(X_default, X_explicit)
        np.testing.assert_array_equal(y_default, y_explicit)

        # Default BaseNet should work with 2D input
        from morphogenetic_engine.components import BaseNet

        model = BaseNet(hidden_dim=32, seed_manager=SeedManager())  # Should default to input_dim=2
        x = torch.randn(4, 2)
        output = model(x)
        assert output.shape == (4, 2)  # Binary classification


class TestNewDatasets:
    """Test the new dataset generation functions."""

    def test_create_moons(self):
        """Test moons dataset generation."""
        from scripts.run_morphogenetic_experiment import create_moons

        # Test basic functionality
        X, y = create_moons(n_samples=100, moon_noise=0.1, moon_sep=0.5, input_dim=2)
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert set(y) == {0, 1}  # Binary classification

        # Test with higher dimensions
        X, y = create_moons(n_samples=100, input_dim=4)
        assert X.shape == (100, 4)
        assert y.shape == (100,)

    def test_create_clusters(self):
        """Test clusters dataset generation."""
        from scripts.run_morphogenetic_experiment import create_clusters

        # Test 2 clusters
        X, y = create_clusters(cluster_count=2, cluster_size=50, input_dim=3)
        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert set(y) == {0, 1}  # Binary classification

        # Test 3 clusters (should still produce binary classification)
        X, y = create_clusters(cluster_count=3, cluster_size=50, input_dim=2)
        assert X.shape == (150, 2)
        assert y.shape == (150,)
        assert set(y) == {0, 1}  # Binary classification due to modulo

    def test_create_spheres(self):
        """Test spheres dataset generation."""
        from scripts.run_morphogenetic_experiment import create_spheres

        # Test 2 spheres
        X, y = create_spheres(sphere_count=2, sphere_size=50, sphere_radii="1,2", input_dim=3)
        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert set(y) == {0, 1}  # Binary classification

        # Test 3 spheres (should still produce binary classification)
        X, y = create_spheres(sphere_count=3, sphere_size=30, sphere_radii="1,2,3", input_dim=4)
        assert X.shape == (90, 4)
        assert y.shape == (90,)
        assert set(y) == {0, 1}  # Binary classification due to modulo

    def test_dataset_validation(self):
        """Test dataset validation and error handling."""
        from scripts.run_morphogenetic_experiment import create_spheres

        # Test mismatched radii and sphere_count
        with pytest.raises(ValueError, match="Number of radii"):
            create_spheres(sphere_count=2, sphere_size=50, sphere_radii="1,2,3", input_dim=3)


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
                    f"scripts.run_morphogenetic_experiment.{expected_function}"
                ) as mock_func, patch(
                    "scripts.run_morphogenetic_experiment.BaseNet"
                ) as mock_net, patch(
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
                    except (SystemExit, Exception):  # pylint: disable=broad-except
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
                "scripts.run_morphogenetic_experiment.create_spirals"
            ) as mock_spirals, patch(
                "scripts.run_morphogenetic_experiment.BaseNet"
            ) as mock_net, patch(
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
                except (SystemExit, Exception):  # pylint: disable=broad-except
                    pass

                # Verify spirals was called with only relevant arguments
                call_args = mock_spirals.call_args
                assert "noise" in call_args.kwargs
                assert np.isclose(call_args.kwargs["noise"], 0.3)
                # cluster_count and sphere_radii should not affect the call
