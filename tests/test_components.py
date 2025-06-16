"""
Comprehensive test suite for the morphogenetic_engine.components module.

This module provides extensive testing for the core components of the morphogenetic
architecture system, focusing on the dynamic and adaptive neural network components.

Key Components Tested:
    SentinelSeed: Adaptive seed modules that can activate and blend into the network
    BaseNet: Dynamic trunk network with configurable layers and seeds
    SeedManager: Management system for seed lifecycle and coordination

Test Coverage Areas:

SentinelSeed Testing:
    - Initialization and configuration
    - State management (dormant, germinating, blending, active)
    - Child network creation and adaptation
    - Health signal computation and drift monitoring
    - Parameter management and gradient flow

BaseNet Testing:
    - Dynamic architecture construction with configurable layers
    - Multi-seed per layer functionality and averaging
    - Forward pass execution and gradient propagation
    - Backward compatibility with fixed architectures
    - Helper methods for seed access and management
    - Architecture scaling and parameter counting

New CLI Flags Testing:
    - --num_layers flag for dynamic layer configuration
    - --seeds_per_layer flag for multi-seed ensemble behavior
    - Edge cases and boundary conditions
    - Integration with existing functionality
    - Backward compatibility verification

The tests ensure that the morphogenetic architecture maintains its adaptive
capabilities while providing flexible configuration options for different
experimental requirements.
"""

# pylint: disable=protected-access

from typing import cast
from unittest.mock import patch

import pytest
import torch

from morphogenetic_engine.components import BaseNet, SentinelSeed
from morphogenetic_engine.core import SeedManager


class TestSentinelSeed:
    """Test suite for SentinelSeed class."""

    def _create_seed_manager(self):
        """Helper to create a seed manager for tests."""
        return SeedManager()

    def test_initialization(self):
        """Test SentinelSeed initialization."""
        seed_manager = SeedManager()
        seed = SentinelSeed(
            seed_id="test_seed",
            dim=64,
            seed_manager=seed_manager,
            blend_steps=25,
            shadow_lr=2e-3,
            progress_thresh=0.7,
            drift_warn=0.15,
        )

        assert seed.seed_id == "test_seed"
        assert seed.dim == 64
        assert seed.blend_steps == 25
        assert seed.progress_thresh == pytest.approx(0.7)
        assert seed.alpha == pytest.approx(0.0)
        assert seed.state == "dormant"
        assert seed.training_progress == pytest.approx(0.0)
        assert seed.drift_warn == pytest.approx(0.15)

        # Check child network structure
        assert isinstance(seed.child, torch.nn.Sequential)
        assert len(seed.child) == 3  # Linear, ReLU, Linear

        # Check optimizer
        assert isinstance(seed.child_optim, torch.optim.Adam)
        assert isinstance(seed.child_loss, torch.nn.MSELoss)

    def test_set_state_transitions(self):
        """Test state transition handling."""
        seed_manager = self._create_seed_manager()
        seed = SentinelSeed("test_seed", 32, seed_manager)
        manager = seed.seed_manager
        manager.germination_log.clear()

        # Test transition to training
        seed._set_state("training")
        assert seed.state == "training"
        assert manager.seeds["test_seed"]["state"] == "training"
        assert manager.seeds["test_seed"]["status"] == "pending"

        # Test transition to active
        seed._set_state("active")
        assert seed.state == "active"
        assert manager.seeds["test_seed"]["state"] == "active"
        assert manager.seeds["test_seed"]["status"] == "active"

        # Check transitions were logged
        assert len(manager.germination_log) == 2

    def test_redundant_state_transition(self):
        """Test that redundant state transitions are ignored."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())
        manager = seed.seed_manager
        manager.germination_log.clear()

        seed._set_state("training")
        log_count_after_first = len(manager.germination_log)

        # Same state transition should be ignored
        seed._set_state("training")
        log_count_after_second = len(manager.germination_log)

        assert log_count_after_first == log_count_after_second

    def test_initialize_as_identity(self):
        """Test that child network is initialized as near-identity."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())

        # Check that parameters are initially zero
        for param in seed.child.parameters():
            assert torch.allclose(param, torch.zeros_like(param))
            assert not param.requires_grad  # Should be frozen initially

    def test_initialize_child(self):
        """Test proper child network initialization."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())

        seed.initialize_child()

        assert seed.state == "training"
        # Parameters should no longer be zero after Kaiming initialization
        weight_params = [p for p in seed.child.parameters() if p.dim() > 1]
        assert len(weight_params) > 0
        for param in weight_params:
            assert not torch.allclose(param, torch.zeros_like(param))
            assert param.requires_grad  # Should be trainable now

    def test_train_child_step_dormant_state(self):
        """Test that training step is ignored in dormant state."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())
        initial_progress = seed.training_progress

        inputs = torch.randn(4, 32)
        seed.train_child_step(inputs)

        # Progress should not change in dormant state
        assert seed.training_progress == initial_progress
        assert seed.state == "dormant"

    def test_train_child_step_empty_input(self):
        """Test training step with empty input."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())
        seed.initialize_child()

        empty_input = torch.empty(0, 32)
        initial_progress = seed.training_progress

        seed.train_child_step(empty_input)

        # Progress should not change with empty input
        assert seed.training_progress == initial_progress

    def test_train_child_step_training_state(self):
        """Test training step in training state."""
        seed = SentinelSeed(
            "test_seed",
            dim=32,
            seed_manager=self._create_seed_manager(),
            progress_thresh=0.5,
        )
        seed.initialize_child()

        inputs = torch.randn(4, 32)
        initial_progress = seed.training_progress

        # Train for several steps
        for _ in range(60):  # Should exceed progress threshold
            seed.train_child_step(inputs)

        assert seed.training_progress > initial_progress
        assert seed.state == "blending"  # Should transition to blending
        assert seed.alpha == pytest.approx(0.0)  # Alpha starts at 0 in blending

    def test_update_blending(self):
        """Test blending update process."""
        seed = SentinelSeed(
            "test_seed",
            dim=32,
            seed_manager=self._create_seed_manager(),
            blend_steps=10,
        )
        seed._set_state("blending")
        seed.alpha = 0.0

        # Update blending several times
        for _ in range(5):
            seed.update_blending()

        assert seed.alpha > 0.0
        assert seed.state == "blending"

        # Complete blending
        for _ in range(10):
            seed.update_blending()

        assert seed.alpha >= 0.99
        assert seed.state == "active"

    def test_update_blending_non_blending_state(self):
        """Test that blending update is ignored in non-blending states."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())
        initial_alpha = seed.alpha

        seed.update_blending()  # Should do nothing in dormant state

        assert seed.alpha == initial_alpha
        assert seed.state == "dormant"

    def test_forward_dormant_state(self):
        """Test forward pass in dormant state."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())
        x = torch.randn(4, 32)

        output = seed.forward(x)

        # Should return input unchanged in dormant state
        assert torch.equal(output, x)

        # Should append to buffer
        seeds_dict = seed.seed_manager.seeds["test_seed"]
        buffer = seeds_dict["buffer"]  # type: ignore
        assert len(buffer) == 1
        assert torch.equal(buffer[0], x.detach())

    def test_forward_training_state(self):
        """Test forward pass in training state."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())
        seed.initialize_child()
        x = torch.randn(4, 32)

        output = seed.forward(x)

        # Should return input unchanged in training state
        assert torch.equal(output, x)

    def test_forward_blending_state(self):
        """Test forward pass in blending state."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())
        seed.initialize_child()
        seed._set_state("blending")
        seed.alpha = 0.5

        x = torch.randn(4, 32)

        with patch.object(seed.seed_manager, "record_drift") as mock_record:
            output = seed.forward(x)
            mock_record.assert_called_once()

        # Output should be blend of input and child output
        assert output.shape == x.shape
        assert not torch.equal(output, x)  # Should be different due to blending

    def test_forward_active_state(self):
        """Test forward pass in active state."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())
        seed.initialize_child()
        seed._set_state("active")

        x = torch.randn(4, 32)

        with patch.object(seed.seed_manager, "record_drift") as mock_record:
            output = seed.forward(x)
            mock_record.assert_called_once()

        # Output should be input + child output (residual connection)
        assert output.shape == x.shape

    def test_drift_warning_only_in_blending(self):
        """Test that drift warnings only occur during blending state."""
        seed = SentinelSeed(
            "test_seed",
            dim=32,
            seed_manager=self._create_seed_manager(),
            drift_warn=0.01,
        )  # Low threshold
        seed.initialize_child()

        x = torch.randn(4, 32)

        # Test active state - should not warn
        seed._set_state("active")
        with patch("morphogenetic_engine.components.logging.warning") as mock_warn:
            seed.forward(x)
            mock_warn.assert_not_called()

        # Test blending state - should warn if drift is high
        seed._set_state("blending")
        seed.alpha = 0.5

        # Create high drift by making child return very different values
        original_child = seed.child
        # Replace with a module that returns very different output
        seed.child = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU())
        # Initialize to produce high values to create drift
        with torch.no_grad():
            first_layer = seed.child[0]
            first_layer.weight.data.fill_(10.0)  # type: ignore
            if first_layer.bias is not None:
                first_layer.bias.data.fill_(100.0)  # type: ignore

        try:
            with patch("morphogenetic_engine.components.logging.warning") as mock_warn:
                seed.forward(x)
                mock_warn.assert_called_once()
                args = mock_warn.call_args[0]
                assert "(blending)" in args[0]
        finally:
            # Restore original child
            seed.child = original_child

    def test_get_health_signal_insufficient_data(self):
        """Test health signal with insufficient buffer data."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())

        # With empty or small buffer, should return infinity
        health = seed.get_health_signal()
        assert health == float("inf")

        # Add some data but not enough
        seeds_dict = seed.seed_manager.seeds["test_seed"]
        buffer = seeds_dict["buffer"]  # type: ignore
        for _ in range(5):  # Less than 10 required
            buffer.append(torch.randn(2, 32))

        health = seed.get_health_signal()
        assert health == float("inf")

    def test_get_health_signal_sufficient_data(self):
        """Test health signal with sufficient buffer data."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())

        # Add sufficient data
        seeds_dict = seed.seed_manager.seeds["test_seed"]
        buffer = seeds_dict["buffer"]  # type: ignore
        for _ in range(15):
            buffer.append(torch.randn(2, 32))

        health = seed.get_health_signal()
        assert isinstance(health, float)
        assert health != float("inf")
        assert health >= 0.0

    def test_state_transitions_complete(self):
        """Test all state transitions including dormant."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())

        # Test transition to training
        seed._set_state("training")
        assert seed.state == "training"
        assert seed.seed_manager.seeds["test_seed"]["status"] == "pending"

        # Test transition to blending
        seed._set_state("blending")
        assert seed.state == "blending"
        assert seed.seed_manager.seeds["test_seed"]["status"] == "pending"

        # Test transition to active
        seed._set_state("active")
        assert seed.state == "active"
        assert seed.seed_manager.seeds["test_seed"]["status"] == "active"

        # Test transition back to dormant (this hits line 65)
        seed._set_state("dormant")
        assert seed.state == "dormant"
        assert seed.seed_manager.seeds["test_seed"]["status"] == "dormant"


class TestBaseNet:
    """
    Comprehensive test suite for the BaseNet class.

    BaseNet is the core trunk network in the morphogenetic architecture,
    featuring dynamic layer construction and configurable sentinel seeds.

    Test Coverage:
    - Initialization with default and custom parameters
    - Dynamic architecture construction (num_layers, seeds_per_layer)
    - Layer and seed organization and access methods
    - Forward pass execution with single and multiple seeds
    - Gradient flow and backpropagation
    - Backbone freezing functionality
    - Seed integration and coordination
    - Architecture consistency validation

    The BaseNet supports:
    - Configurable number of hidden layers (default: 8)
    - Multiple sentinel seeds per layer (default: 1)
    - Dynamic parameter allocation and management
    - Backward compatibility with fixed architectures
    """

    def test_initialization_default_params(self):
        """Test BaseNet initialization with default parameters."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2)

        # Check default hidden dimension
        assert net.hidden_dim == 64
        assert net.num_layers == 8
        assert net.seeds_per_layer == 1

        # Check input layer dimensions
        assert net.input_layer.in_features == 2
        assert net.input_layer.out_features == 64

        # Check that all layers are present
        assert len(net.layers) == 8
        for layer in net.layers:
            assert layer.in_features == 64
            assert layer.out_features == 64

        # Check that all seeds are present (8 layers * 1 seed per layer = 8 seeds)
        assert len(net.all_seeds) == 8
        for i, seed in enumerate(net.all_seeds):
            assert isinstance(seed, SentinelSeed)
            assert seed.seed_id == f"seed{i+1}_1"  # seed1_1, seed2_1, etc.
            assert seed.dim == 64

    def test_initialization_custom_params(self):
        """Test BaseNet initialization with custom parameters."""
        net = BaseNet(
            hidden_dim=128,
            seed_manager=SeedManager(),
            blend_steps=20,
            shadow_lr=5e-4,
            progress_thresh=0.8,
            drift_warn=0.05,
        )

        # Check custom hidden dimension
        assert net.hidden_dim == 128

        # Check that seed parameters are passed correctly
        seed = net.all_seeds[0]  # First seed (seed1_1)
        assert seed.dim == 128
        assert seed.blend_steps == 20
        assert seed.shadow_lr == pytest.approx(5e-4)
        assert seed.progress_thresh == pytest.approx(0.8)
        assert seed.drift_warn == pytest.approx(0.05)

    def test_freeze_backbone(self):
        """Test backbone freezing functionality."""
        net = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)

        # Initialize all seeds to make their parameters trainable
        for seed_module in net.all_seeds:
            # Cast to ensure proper typing
            seed_instance = cast(SentinelSeed, seed_module)
            seed_instance.initialize_child()

        # Now all parameters should be trainable
        for name, param in net.named_parameters():
            assert param.requires_grad, f"Parameter {name} should be trainable"

        # Freeze backbone
        net.freeze_backbone()

        # Check that non-seed parameters are frozen
        for name, param in net.named_parameters():
            if "seed" not in name:
                assert not param.requires_grad, f"Backbone parameter {name} should be frozen"
            else:
                assert param.requires_grad, f"Seed parameter {name} should remain trainable"
                # Seed parameters might be frozen (if not initialized) or trainable

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct output shapes."""
        net = BaseNet(hidden_dim=64, seed_manager=SeedManager(), input_dim=2)

        # Test different batch sizes
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 2)  # Input dimension is 2
            output = net.forward(x)

            assert output.shape == (batch_size, 2)  # Output dimension is 2

    def test_forward_pass_deterministic(self):
        """Test that forward pass is deterministic given same input."""
        net = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)

        # Set eval mode for deterministic behavior
        net.eval()

        x = torch.randn(4, 2)

        with torch.no_grad():
            output1 = net.forward(x)
            output2 = net.forward(x)

        assert torch.allclose(output1, output2)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        net = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        x = torch.randn(4, 2, requires_grad=True)

        output = net.forward(x)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        # Check that non-frozen parameters have gradients
        for _, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_seed_integration(self):
        """Test that seeds are properly integrated into forward pass."""
        net = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)

        # Get first seed and verify it's a SentinelSeed
        first_seed_module = cast(SentinelSeed, net.all_seeds[0])  # Get first seed (seed1_1)

        # Test that we can initialize it
        first_seed_module.initialize_child()

        x = torch.randn(4, 2)

        # Test that forward pass works with initialized seed
        output = net.forward(x)
        assert output.shape == (4, 2), "Forward pass should work correctly"

        # Test that the seed is properly integrated
        assert first_seed_module.seed_id == "seed1_1", "Seed ID should follow naming convention"

    def test_all_seeds_different_instances(self):
        """Test that all seeds are separate instances."""
        net = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)

        seeds = net.get_all_seeds()

        # All seeds should be different instances
        for i, seed1 in enumerate(seeds):
            for j, seed2 in enumerate(seeds):
                if i != j:
                    assert seed1 is not seed2
                    assert seed1.seed_id != seed2.seed_id

    def test_network_architecture_consistency(self):
        """Test that network architecture is consistent."""
        net = BaseNet(hidden_dim=64, seed_manager=SeedManager(), input_dim=2)

        # Check input layer dimensions
        assert net.input_layer.in_features == 2
        assert net.input_layer.out_features == 64

        # Check that all hidden layers have correct dimensions
        for i, layer in enumerate(net.layers):
            assert layer.in_features == 64, f"Layer {i} input dimension mismatch"
            assert layer.out_features == 64, f"Layer {i} output dimension mismatch"

        # Check output layer dimensions
        assert net.out.in_features == 64
        assert net.out.out_features == 2

        # Check that all seeds have correct dimension
        for seed in net.get_all_seeds():
            assert seed.dim == 64

        # Check that we have the expected number of seeds (default 8 layers, 1 seed each)
        assert net.get_total_seeds() == 8


class TestMultiSeedBaseNet:
    """Test BaseNet with multiple seeds per layer functionality."""

    def test_multi_seed_initialization(self):
        """Test BaseNet initialization with multiple seeds per layer."""
        seed_manager = SeedManager()
        net = BaseNet(
            hidden_dim=32, seed_manager=seed_manager, input_dim=3, num_layers=2, seeds_per_layer=3
        )

        assert net.num_layers == 2
        assert net.seeds_per_layer == 3
        assert net.get_total_seeds() == 6

        # Verify all seeds are properly created
        all_seeds = net.get_all_seeds()
        assert len(all_seeds) == 6

        # Verify naming convention
        expected_names = ["seed1_1", "seed1_2", "seed1_3", "seed2_1", "seed2_2", "seed2_3"]
        actual_names = [seed.seed_id for seed in all_seeds]
        assert actual_names == expected_names

    def test_single_seed_compatibility(self):
        """Test that single seed per layer maintains backward compatibility."""
        seed_manager = SeedManager()
        net = BaseNet(
            hidden_dim=64, seed_manager=seed_manager, input_dim=2, num_layers=4, seeds_per_layer=1
        )

        assert net.get_total_seeds() == 4

        # Should behave like old architecture
        all_seeds = net.get_all_seeds()
        expected_names = ["seed1_1", "seed2_1", "seed3_1", "seed4_1"]
        actual_names = [seed.seed_id for seed in all_seeds]
        assert actual_names == expected_names

    def test_get_seeds_for_layer(self):
        """Test the get_seeds_for_layer method."""
        seed_manager = SeedManager()
        net = BaseNet(
            hidden_dim=32, seed_manager=seed_manager, input_dim=2, num_layers=3, seeds_per_layer=2
        )

        # Test layer 0
        layer_0_seeds = net.get_seeds_for_layer(0)
        assert len(layer_0_seeds) == 2
        assert [s.seed_id for s in layer_0_seeds] == ["seed1_1", "seed1_2"]

        # Test layer 1
        layer_1_seeds = net.get_seeds_for_layer(1)
        assert len(layer_1_seeds) == 2
        assert [s.seed_id for s in layer_1_seeds] == ["seed2_1", "seed2_2"]

        # Test layer 2
        layer_2_seeds = net.get_seeds_for_layer(2)
        assert len(layer_2_seeds) == 2
        assert [s.seed_id for s in layer_2_seeds] == ["seed3_1", "seed3_2"]

    def test_multi_seed_forward_pass(self):
        """Test forward pass with multiple seeds per layer."""
        seed_manager = SeedManager()
        net = BaseNet(
            hidden_dim=16, seed_manager=seed_manager, input_dim=2, num_layers=2, seeds_per_layer=4
        )

        # Test with various input sizes
        for batch_size in [1, 5, 10]:
            x = torch.randn(batch_size, 2)
            output = net(x)

            assert output.shape == (batch_size, 2)
            assert torch.isfinite(output).all()

    def test_multi_seed_gradient_flow(self):
        """Test that gradients flow correctly with multiple seeds."""
        seed_manager = SeedManager()
        net = BaseNet(
            hidden_dim=8, seed_manager=seed_manager, input_dim=2, num_layers=2, seeds_per_layer=2
        )

        x = torch.randn(3, 2, requires_grad=True)
        output = net(x)
        loss = output.sum()

        # Before backward pass
        for param in net.parameters():
            assert param.grad is None

        loss.backward()

        # After backward pass, check that backbone parameters have gradients
        # (Seeds start dormant so their parameters may not have gradients initially)
        backbone_params_with_grads = 0
        total_backbone_params = 0

        for name, param in net.named_parameters():
            if "seed" not in name:  # Only check backbone parameters
                total_backbone_params += 1
                if param.grad is not None:
                    backbone_params_with_grads += 1

        # All backbone parameters should have gradients
        assert backbone_params_with_grads == total_backbone_params

    def test_extreme_configuration(self):
        """Test with extreme configurations to ensure robustness."""
        seed_manager = SeedManager()

        # Test many seeds per layer
        net = BaseNet(
            hidden_dim=8, seed_manager=seed_manager, input_dim=2, num_layers=1, seeds_per_layer=8
        )

        assert net.get_total_seeds() == 8

        x = torch.randn(2, 2)
        output = net(x)
        assert output.shape == (2, 2)

        # Test many layers with few seeds
        net2 = BaseNet(
            hidden_dim=8, seed_manager=SeedManager(), input_dim=2, num_layers=10, seeds_per_layer=1
        )

        assert net2.get_total_seeds() == 10

        x = torch.randn(2, 2)
        output = net2(x)
        assert output.shape == (2, 2)

    def test_seed_averaging_behavior(self):
        """Test that multiple seeds per layer are properly averaged."""
        seed_manager = SeedManager()
        net = BaseNet(
            hidden_dim=4, seed_manager=seed_manager, input_dim=2, num_layers=1, seeds_per_layer=3
        )

        # Get all seeds for the single layer
        layer_seeds = net.get_seeds_for_layer(0)
        assert len(layer_seeds) == 3

        # Test that all seeds receive the same input
        # This is more of a behavioral test to ensure the architecture is sound
        with torch.no_grad():
            # Apply input layer
            after_input = net.input_activation(net.input_layer(torch.randn(2, 2)))
            after_layer = net.activations[0](net.layers[0](after_input))

            # All seeds should be able to process the same input
            for seed in layer_seeds:
                seed_output = seed(after_layer)
                assert seed_output.shape == after_layer.shape


class TestNewCLIFlags:
    """
    Test suite for the new CLI flags functionality in BaseNet.

    This class tests the BaseNet component's support for the new
    --num_layers and --seeds_per_layer CLI flags, ensuring that:

    Architecture Configuration:
    - Default values work correctly (8 layers, 1 seed per layer)
    - Custom num_layers values create the correct number of layers
    - Custom seeds_per_layer values create multiple seeds per layer
    - Combined flag usage works as expected

    Infrastructure Testing:
    - Seed ID naming follows the correct convention (seed1_1, seed1_2, etc.)
    - Helper methods work correctly (get_seeds_for_layer, get_total_seeds)
    - Seed organization and access patterns are consistent
    - Architecture scaling behaves predictably

    These flags enable flexible network architecture configuration while
    maintaining compatibility with existing code and experimental setups.
    """

    def test_num_layers_flag_default(self):
        """Test that num_layers defaults to 8."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2)
        assert net.num_layers == 8
        assert len(net.layers) == 8
        assert len(net.activations) == 8
        assert len(net.all_seeds) == 8  # 8 layers * 1 seed per layer

    def test_num_layers_flag_custom(self):
        """Test custom num_layers values."""
        for num_layers in [1, 3, 5, 10, 16]:
            net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=num_layers)
            assert net.num_layers == num_layers
            assert len(net.layers) == num_layers
            assert len(net.activations) == num_layers
            assert len(net.all_seeds) == num_layers  # num_layers * 1 seed per layer

    def test_seeds_per_layer_flag_default(self):
        """Test that seeds_per_layer defaults to 1."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2)
        assert net.seeds_per_layer == 1
        assert len(net.all_seeds) == 8  # 8 layers * 1 seed per layer

    def test_seeds_per_layer_flag_custom(self):
        """Test custom seeds_per_layer values."""
        for seeds_per_layer in [1, 2, 3, 5]:
            net = BaseNet(
                seed_manager=SeedManager(),
                input_dim=2,
                num_layers=4,
                seeds_per_layer=seeds_per_layer,
            )
            assert net.seeds_per_layer == seeds_per_layer
            assert len(net.all_seeds) == 4 * seeds_per_layer  # 4 layers * seeds_per_layer

    def test_combined_flags(self):
        """Test combining both num_layers and seeds_per_layer flags."""
        test_cases = [
            (2, 1, 2),  # 2 layers, 1 seed each = 2 total
            (3, 2, 6),  # 3 layers, 2 seeds each = 6 total
            (5, 3, 15),  # 5 layers, 3 seeds each = 15 total
            (1, 5, 5),  # 1 layer, 5 seeds = 5 total
        ]

        for num_layers, seeds_per_layer, expected_total_seeds in test_cases:
            net = BaseNet(
                seed_manager=SeedManager(),
                input_dim=2,
                num_layers=num_layers,
                seeds_per_layer=seeds_per_layer,
            )
            assert net.num_layers == num_layers
            assert net.seeds_per_layer == seeds_per_layer
            assert len(net.all_seeds) == expected_total_seeds

    def test_seed_id_naming_convention(self):
        """Test that seed IDs follow the correct naming convention."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=3, seeds_per_layer=2)

        expected_seed_ids = [
            "seed1_1",
            "seed1_2",  # Layer 1
            "seed2_1",
            "seed2_2",  # Layer 2
            "seed3_1",
            "seed3_2",  # Layer 3
        ]

        actual_seed_ids = [seed.seed_id for seed in net.all_seeds]
        assert actual_seed_ids == expected_seed_ids

    def test_get_seeds_for_layer_method(self):
        """Test the get_seeds_for_layer helper method."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=3, seeds_per_layer=3)

        # Test each layer
        for layer_idx in range(3):
            layer_seeds = net.get_seeds_for_layer(layer_idx)
            assert len(layer_seeds) == 3

            # Check that seed IDs are correct for this layer
            for j, seed in enumerate(layer_seeds):
                expected_id = f"seed{layer_idx+1}_{j+1}"
                assert seed.seed_id == expected_id

    def test_get_total_seeds_method(self):
        """Test the get_total_seeds helper method."""
        test_cases = [
            (2, 1, 2),
            (4, 2, 8),
            (3, 4, 12),
        ]

        for num_layers, seeds_per_layer, expected_total in test_cases:
            net = BaseNet(
                seed_manager=SeedManager(),
                input_dim=2,
                num_layers=num_layers,
                seeds_per_layer=seeds_per_layer,
            )
            assert net.get_total_seeds() == expected_total

    def test_get_all_seeds_method(self):
        """Test the get_all_seeds helper method."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=2, seeds_per_layer=3)

        all_seeds = net.get_all_seeds()
        assert len(all_seeds) == 6
        assert all_seeds == list(net.all_seeds)  # Should return the same objects


class TestMultiSeedArchitecture:
    """
    Test suite for multi-seed per layer architecture functionality.

    This class focuses specifically on testing the behavior when multiple
    sentinel seeds are assigned to each layer, enabling ensemble-like
    behavior through seed output averaging.

    Core Functionality Tests:
    - Single seed per layer behavior (backward compatibility)
    - Multiple seeds per layer forward pass execution
    - Seed output averaging mechanics
    - Gradient flow through multiple parallel paths
    - Seed independence and parameter isolation

    Architecture Validation:
    - Proper seed initialization across multiple seeds
    - Parameter allocation and management
    - Memory efficiency with increased seed counts
    - Computational overhead analysis

    The multi-seed architecture enables adaptive capacity scaling by
    allowing multiple adaptive paths per layer, each with independent
    parameters that are averaged to produce the layer output.
    """

    def test_single_seed_forward_pass(self):
        """Test forward pass with single seed per layer (backward compatibility)."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=2, seeds_per_layer=1)

        x = torch.randn(4, 2)
        output = net(x)
        assert output.shape == (4, 2)

    def test_multiple_seeds_forward_pass(self):
        """Test forward pass with multiple seeds per layer."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=2, seeds_per_layer=3)

        x = torch.randn(4, 2)
        output = net(x)
        assert output.shape == (4, 2)

    def test_seed_averaging_behavior(self):
        """Test that multiple seeds per layer are properly averaged."""
        net = BaseNet(
            seed_manager=SeedManager(), input_dim=2, num_layers=1, seeds_per_layer=3, hidden_dim=4
        )

        # Test that we can get the seeds for the layer
        layer_seeds = net.get_seeds_for_layer(0)
        assert len(layer_seeds) == 3

        # Initialize seeds to make them trainable
        for seed in layer_seeds:
            seed.initialize_child()

        x = torch.randn(2, 2)
        output = net(x)
        assert output.shape == (2, 2)

    def test_gradient_flow_multiple_seeds(self):
        """Test that gradients flow properly through multiple seeds."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=2, seeds_per_layer=2)

        # Initialize all seeds
        for seed_module in net.all_seeds:
            seed_instance = cast(SentinelSeed, seed_module)
            seed_instance.initialize_child()

        x = torch.randn(4, 2, requires_grad=True)
        output = net(x)
        loss = output.sum()
        loss.backward()

        # Check gradients for input
        assert x.grad is not None

        # Check gradients for all trainable parameters
        has_gradients = False
        for name, param in net.named_parameters():
            if param.requires_grad and param.grad is not None and not torch.allclose(
                param.grad, torch.zeros_like(param.grad)
            ):
                has_gradients = True
                break
            # Use the name variable to avoid unused variable warning
            assert isinstance(name, str), "Parameter name should be a string"

        # At least some parameters should have non-zero gradients
        assert has_gradients, "No parameters received gradients"

    def test_seed_independence(self):
        """Test that seeds in the same layer are independent."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=1, seeds_per_layer=3)

        layer_seeds = net.get_seeds_for_layer(0)

        # Initialize seeds and verify they have different parameters
        for seed in layer_seeds:
            seed.initialize_child()

        # Check that all seeds have different parameter instances
        for i, seed1 in enumerate(layer_seeds):
            for j, seed2 in enumerate(layer_seeds):
                if i != j:
                    # Different seed objects
                    assert seed1 is not seed2
                    assert seed1.seed_id != seed2.seed_id

                    # Different parameter instances
                    for (name1, param1), (name2, param2) in zip(
                        seed1.named_parameters(), seed2.named_parameters()
                    ):
                        assert param1 is not param2
                        # Use the name variables to avoid unused variable warning
                        assert isinstance(name1, str) and isinstance(name2, str)


class TestBackwardCompatibility:
    """
    Test suite ensuring backward compatibility with the old hardcoded architecture.

    This class verifies that the new dynamic architecture system maintains
    full compatibility with existing code and experiments that relied on
    the previous hardcoded BaseNet implementation.

    Compatibility Areas:
    - Default parameter values match the old hardcoded values
    - Network structure and layer organization remain consistent
    - Forward pass behavior produces identical results
    - API compatibility through the 'seeds' property
    - Parameter initialization and naming conventions

    Verification Methods:
    - Architecture equivalence testing (8 layers, 1 seed per layer)
    - Output determinism validation with same random seeds
    - Property access patterns (maintaining seeds attribute)
    - Method signature compatibility

    This ensures that existing experiments and code can migrate to the
    new system without modifications while gaining access to the new
    dynamic architecture capabilities when needed.
    """

    def test_default_architecture_matches_old(self):
        """Test that default parameters create the same architecture as before."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2)

        # Should have same structure as old hardcoded version
        assert net.num_layers == 8
        assert net.seeds_per_layer == 1
        assert len(net.all_seeds) == 8
        assert net.hidden_dim == 64

    def test_seeds_property_compatibility(self):
        """Test that the 'seeds' property provides backward compatibility."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2)

        # Test that seeds property works
        seeds = net.seeds
        assert len(seeds) == 8
        assert seeds == net.all_seeds

    def test_forward_pass_equivalence(self):
        """Test that forward pass produces consistent results."""
        # Create two networks with same seed
        torch.manual_seed(42)
        net1 = BaseNet(seed_manager=SeedManager(), input_dim=2)

        torch.manual_seed(42)
        net2 = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=8, seeds_per_layer=1)

        # Both should produce the same output for same input
        x = torch.randn(4, 2)

        net1.eval()
        net2.eval()

        with torch.no_grad():
            output1 = net1(x)
            output2 = net2(x)

        # Outputs should be identical (within floating point precision)
        assert torch.allclose(output1, output2, atol=1e-6)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_layer_network(self):
        """Test network with only one layer."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=1, seeds_per_layer=1)

        x = torch.randn(2, 2)
        output = net(x)
        assert output.shape == (2, 2)

    def test_many_seeds_per_layer(self):
        """Test network with many seeds per layer."""
        net = BaseNet(
            seed_manager=SeedManager(), input_dim=2, num_layers=2, seeds_per_layer=10  # Many seeds
        )

        assert len(net.all_seeds) == 20

        x = torch.randn(2, 2)
        output = net(x)
        assert output.shape == (2, 2)

    def test_large_network(self):
        """Test large network configuration."""
        net = BaseNet(
            seed_manager=SeedManager(),
            input_dim=3,
            num_layers=20,
            seeds_per_layer=5,
            hidden_dim=128,
        )

        assert len(net.all_seeds) == 100  # 20 * 5

        x = torch.randn(2, 3)
        output = net(x)
        assert output.shape == (2, 2)

    def test_parameter_count_scaling(self):
        """Test that parameter count scales appropriately."""

        def count_parameters(net):
            return sum(p.numel() for p in net.parameters())

        # Network with 1 seed per layer
        net1 = BaseNet(
            seed_manager=SeedManager(), input_dim=2, num_layers=2, seeds_per_layer=1, hidden_dim=4
        )

        # Network with 3 seeds per layer
        net2 = BaseNet(
            seed_manager=SeedManager(), input_dim=2, num_layers=2, seeds_per_layer=3, hidden_dim=4
        )

        params1 = count_parameters(net1)
        params2 = count_parameters(net2)

        # net2 should have more parameters due to additional seeds
        assert params2 > params1
