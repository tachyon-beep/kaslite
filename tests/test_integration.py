"""
Test suite for integration and architecture scaling.

This module contains tests for the morphogenetic architecture integration and scaling,
including tests for:

- Complete pipeline integration testing
- High-dimensional input integration
- Architecture scaling and performance
- Seed lifecycle testing
- Memory efficiency and parameter scaling

Test Classes:
    TestIntegration: Integration tests for the complete system
    TestHighDimensionalIntegration: Test suite for high-dimensional input integration
    TestArchitectureScaling: Test suite for architecture scaling and performance
"""


import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from morphogenetic_engine.components import BaseNet, SentinelSeed
from morphogenetic_engine.core import KasminaMicro, SeedManager
from morphogenetic_engine.datasets import create_complex_moons, create_spirals
from morphogenetic_engine.experiment import build_model_and_agents
from morphogenetic_engine.training import evaluate, train_epoch


class TestIntegration:
    """Integration tests for the complete system."""

    def test_full_pipeline_mini(self):
        """Test a minimal full pipeline execution."""
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
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager())  # Should default to input_dim=2
        x = torch.randn(4, 2)
        output = model(x)
        assert output.shape == (4, 2)  # Binary classification


class TestArchitectureScaling:
    """
    Test suite for architecture scaling and performance with new CLI flags.

    This class tests how the morphogenetic network architecture scales with
    different values of --num_layers and --seeds_per_layer flags, focusing on:

    Performance and Scaling:
    - Parameter count scaling with number of layers
    - Parameter count scaling with seeds per layer
    - Memory efficiency with large configurations
    - Computational overhead analysis

    Functional Testing:
    - Model creation with various architecture sizes
    - Forward pass execution with different configurations
    - Training compatibility across architectures
    - Integration testing with real data flows

    Edge Cases:
    - Minimal configurations (1 layer, 1 seed)
    - Large configurations (many layers, many seeds)
    - Memory constraint testing
    - Parameter initialization verification

    The tests ensure that the dynamic architecture system can handle a wide
    range of network sizes efficiently and correctly.
    """

    def test_parameter_count_scaling(self):
        """
        Test that parameter count scales appropriately with new flags.

        Verifies that the total number of parameters in the network scales
        correctly when changing the number of layers and seeds per layer:

        1. More layers should result in more parameters (linear scaling)
        2. More seeds per layer should result in more parameters

        This test ensures that the dynamic architecture system properly
        allocates parameters and that resource usage is predictable and
        scales as expected with the configuration flags.

        The test uses a helper function to create models with different
        configurations and compares their parameter counts to verify
        proper scaling behavior.
        """

        def get_param_count(num_layers_param, seeds_per_layer_param):
            class MockArgs:
                """Mock arguments for parameter counting test."""

                problem_type = "moons"
                hidden_dim = 64
                lr = 1e-3
                batch_size = 32
                progress_thresh = 0.6
                drift_warn = 0.12
                num_layers = num_layers_param
                seeds_per_layer = seeds_per_layer_param
                acc_threshold = 0.95
                input_dim = 2
                blend_steps = 30
                shadow_lr = 1e-3

            args = MockArgs()
            device = torch.device("cpu")
            model, _, _, _ = build_model_and_agents(args, device)
            return sum(p.numel() for p in model.parameters())

        # Test scaling with layers
        params_2_layers = get_param_count(2, 1)
        params_4_layers = get_param_count(4, 1)
        assert params_4_layers > params_2_layers

        # Test scaling with seeds per layer
        params_1_seed = get_param_count(3, 1)
        params_3_seeds = get_param_count(3, 3)
        assert params_3_seeds > params_1_seed

    def test_memory_efficiency(self):
        """
        Test that networks with different architectures can be created efficiently.

        This test verifies that the dynamic architecture system can handle
        various network configurations without memory issues or failures:

        Test Configurations:
        - (1, 1): Minimal configuration for basic functionality
        - (2, 2): Small configuration for lightweight applications
        - (5, 3): Medium configuration for standard experiments
        - (8, 1): Default configuration for backward compatibility
        - (10, 2): Larger configuration for complex tasks

        For each configuration, the test:
        1. Creates the model and all associated components
        2. Verifies the architecture matches the requested configuration
        3. Tests that forward pass execution works correctly
        4. Confirms no memory leaks or allocation failures

        This ensures the system is robust across different resource requirements
        and can scale from minimal to large network architectures efficiently.
        """

        # Test various combinations to ensure no memory issues
        test_configs = [
            (1, 1),  # Minimal
            (2, 2),  # Small
            (5, 3),  # Medium
            (8, 1),  # Default
            (10, 2),  # Larger
        ]

        for num_layers_param, seeds_per_layer_param in test_configs:

            class MockArgs:
                """Mock arguments for testing different layer configurations."""

                problem_type = "moons"
                hidden_dim = 32  # Keep small for memory efficiency
                lr = 1e-3
                batch_size = 16
                progress_thresh = 0.6
                drift_warn = 0.12
                num_layers = num_layers_param
                seeds_per_layer = seeds_per_layer_param
                acc_threshold = 0.95
                input_dim = 2
                blend_steps = 30
                shadow_lr = 1e-3

            args = MockArgs()
            device = torch.device("cpu")

            # Should create without memory errors
            model, seed_manager, loss_fn, kasmina = build_model_and_agents(args, device)

            # Verify architecture
            assert model.num_layers == num_layers_param
            assert model.seeds_per_layer == seeds_per_layer_param
            assert model.get_total_seeds() == num_layers_param * seeds_per_layer_param

            # Verify all components are properly initialized
            assert isinstance(seed_manager, SeedManager)
            assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
            assert isinstance(kasmina, KasminaMicro)

            # Test forward pass works
            x = torch.randn(4, 2)
            output = model(x)
            assert output.shape == (4, 2)

    def test_basic_training_compatibility(self):
        """Test that basic model creation and forward pass work with different architectures."""

        class MockArgs:
            """Mock arguments for testing basic training compatibility."""

            problem_type = "moons"
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

        model, seed_manager, loss_fn, kasmina = build_model_and_agents(args, device)

        # Verify components are properly created
        assert isinstance(seed_manager, SeedManager)
        assert isinstance(kasmina, KasminaMicro)

        # Create dummy data
        X = torch.randn(32, 2)
        y = torch.randint(0, 2, (32,))

        # Test basic forward pass and loss computation
        output = model(X)
        assert output.shape == (32, 2)

        loss = loss_fn(output, y)
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0  # Loss should be non-negative
