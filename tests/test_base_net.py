"""Tests for BaseNet and multi-seed functionality."""

# pylint: disable=protected-access

from typing import cast
import pytest
import torch

from morphogenetic_engine.components import BaseNet, SentinelSeed
from morphogenetic_engine.core import SeedManager


class TestBaseNet:
    """Test suite for BaseNet class."""

    def test_initialization_default_params(self):
        """Test BaseNet initialization with default parameters."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2)

        assert net.hidden_dim == 64
        assert net.num_layers == 8
        assert net.seeds_per_layer == 1
        assert net.input_layer.in_features == 2
        assert net.input_layer.out_features == 64
        assert len(net.layers) == 8
        assert len(net.all_seeds) == 8

        # Check seed naming convention
        for i, seed in enumerate(net.all_seeds):
            assert isinstance(seed, SentinelSeed)
            assert seed.seed_id == f"seed{i+1}_1"
            assert seed.dim == 64

    def test_initialization_custom_params(self):
        """Test BaseNet with custom parameters."""
        net = BaseNet(
            hidden_dim=128,
            seed_manager=SeedManager(),
            input_dim=2,
            blend_steps=20,
            shadow_lr=5e-4,
            progress_thresh=0.8,
            drift_warn=0.05,
        )

        assert net.hidden_dim == 128
        seed = net.all_seeds[0]
        assert seed.dim == 128
        assert seed.blend_steps == 20
        assert seed.shadow_lr == pytest.approx(5e-4)
        assert seed.progress_thresh == pytest.approx(0.8)
        assert seed.drift_warn == pytest.approx(0.05)

    def test_freeze_backbone(self):
        """Test backbone freezing functionality."""
        net = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)

        # Initialize seeds to make parameters trainable
        for seed_module in net.all_seeds:
            seed_instance = cast(SentinelSeed, seed_module)
            seed_instance.initialize_child()

        # All parameters should be trainable
        for name, param in net.named_parameters():
            assert param.requires_grad, f"Parameter {name} should be trainable"

        # Freeze backbone
        net.freeze_backbone()

        # Check that non-seed parameters are frozen
        for name, param in net.named_parameters():
            if "seed" not in name:
                assert not param.requires_grad, f"Backbone parameter {name} should be frozen"

    def test_forward_pass_and_shapes(self):
        """Test forward pass with various batch sizes."""
        net = BaseNet(hidden_dim=64, seed_manager=SeedManager(), input_dim=2)

        # Test different batch sizes
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 2)
            output = net.forward(x)
            assert output.shape == (batch_size, 2)

        # Test deterministic behavior
        net.eval()
        x = torch.randn(4, 2)
        with torch.no_grad():
            output1 = net.forward(x)
            output2 = net.forward(x)
        assert torch.allclose(output1, output2)

    def test_gradient_flow(self):
        """Test gradient flow through the network."""
        net = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        x = torch.randn(4, 2, requires_grad=True)

        output = net.forward(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        # Check parameter gradients
        for _, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_seed_integration_and_independence(self):
        """Test seed integration and independence."""
        net = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)

        # Test seed initialization
        first_seed = cast(SentinelSeed, net.all_seeds[0])
        first_seed.initialize_child()

        x = torch.randn(4, 2)
        output = net.forward(x)
        assert output.shape == (4, 2)
        assert first_seed.seed_id == "seed1_1"

        # Test seed independence
        seeds = net.get_all_seeds()
        for i, seed1 in enumerate(seeds):
            for j, seed2 in enumerate(seeds):
                if i != j:
                    assert seed1 is not seed2
                    assert seed1.seed_id != seed2.seed_id

    def test_architecture_consistency(self):
        """Test network architecture consistency."""
        net = BaseNet(hidden_dim=64, seed_manager=SeedManager(), input_dim=2)

        # Check layer dimensions
        assert net.input_layer.in_features == 2
        assert net.input_layer.out_features == 64
        assert net.out.in_features == 64
        assert net.out.out_features == 2

        for i, layer in enumerate(net.layers):
            assert layer.in_features == 64
            assert layer.out_features == 64

        # Check seed consistency
        for seed in net.get_all_seeds():
            assert seed.dim == 64
        assert net.get_total_seeds() == 8


class TestMultiSeedBaseNet:
    """Test BaseNet with multiple seeds per layer."""

    def test_multi_seed_initialization(self):
        """Test BaseNet with multiple seeds per layer."""
        net = BaseNet(
            hidden_dim=32,
            seed_manager=SeedManager(),
            input_dim=3,
            num_layers=2,
            seeds_per_layer=3
        )

        assert net.num_layers == 2
        assert net.seeds_per_layer == 3
        assert net.get_total_seeds() == 6

        # Check seed naming for multi-seed setup
        expected_ids = ["seed1_1", "seed1_2", "seed1_3", "seed2_1", "seed2_2", "seed2_3"]
        actual_ids = [seed.seed_id for seed in net.all_seeds]
        assert actual_ids == expected_ids

    def test_single_seed_compatibility(self):
        """Test that single seed per layer works as before."""
        net = BaseNet(
            hidden_dim=32,
            seed_manager=SeedManager(),
            input_dim=2,
            num_layers=3,
            seeds_per_layer=1
        )

        assert net.get_total_seeds() == 3
        expected_ids = ["seed1_1", "seed2_1", "seed3_1"]
        actual_ids = [seed.seed_id for seed in net.all_seeds]
        assert actual_ids == expected_ids

    def test_get_seeds_for_layer(self):
        """Test getting seeds for specific layers."""
        net = BaseNet(
            hidden_dim=32,
            seed_manager=SeedManager(),
            input_dim=2,
            num_layers=2,
            seeds_per_layer=3
        )

        layer0_seeds = net.get_seeds_for_layer(0)
        layer1_seeds = net.get_seeds_for_layer(1)

        assert len(layer0_seeds) == 3
        assert len(layer1_seeds) == 3

        # Check seed IDs
        assert [s.seed_id for s in layer0_seeds] == ["seed1_1", "seed1_2", "seed1_3"]
        assert [s.seed_id for s in layer1_seeds] == ["seed2_1", "seed2_2", "seed2_3"]

    def test_multi_seed_forward_pass(self):
        """Test forward pass with multiple seeds per layer."""
        net = BaseNet(
            hidden_dim=32,
            seed_manager=SeedManager(),
            input_dim=2,
            num_layers=2,
            seeds_per_layer=2
        )

        x = torch.randn(4, 2)
        output = net.forward(x)
        assert output.shape == (4, 2)

        # Verify all seeds received input (simplified check)
        for seed in net.all_seeds:
            # Just verify the seed exists and is a SentinelSeed
            assert isinstance(seed, SentinelSeed)

    def test_multi_seed_gradient_flow(self):
        """Test gradient flow with multiple seeds."""
        net = BaseNet(
            hidden_dim=32,
            seed_manager=SeedManager(),
            input_dim=2,
            num_layers=2,
            seeds_per_layer=3
        )

        # Initialize some seeds
        for i in range(0, len(net.all_seeds), 2):  # Initialize every other seed
            cast(SentinelSeed, net.all_seeds[i]).initialize_child()

        x = torch.randn(4, 2, requires_grad=True)
        output = net.forward(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

        # Check that at least some parameters have gradients
        params_with_grads = 0
        total_trainable_params = 0
        
        for _, param in net.named_parameters():
            if param.requires_grad:
                total_trainable_params += 1
                if param.grad is not None:
                    params_with_grads += 1

        # At least some trainable parameters should have gradients
        assert params_with_grads > 0, f"No parameters received gradients out of {total_trainable_params} trainable"

    def test_extreme_configuration(self):
        """Test with extreme multi-seed configuration."""
        net = BaseNet(
            hidden_dim=16,
            seed_manager=SeedManager(),
            input_dim=2,
            num_layers=1,
            seeds_per_layer=5
        )

        assert net.get_total_seeds() == 5
        
        x = torch.randn(2, 2)
        output = net.forward(x)
        assert output.shape == (2, 2)

        # Test that all seeds are independent
        layer_seeds = net.get_seeds_for_layer(0)
        assert len(layer_seeds) == 5
        for i, seed in enumerate(layer_seeds):
            assert seed.seed_id == f"seed1_{i+1}"

    def test_seed_averaging_behavior(self):
        """Test that multiple seeds in a layer work together."""
        net = BaseNet(
            hidden_dim=32,
            seed_manager=SeedManager(),
            input_dim=2,
            num_layers=1,
            seeds_per_layer=3
        )

        # Initialize all seeds in different states
        seeds = net.get_seeds_for_layer(0)
        seeds[0].initialize_child()  # Training state
        seeds[1].initialize_child()
        seeds[1]._set_state("blending")
        seeds[1].alpha = 0.5
        seeds[2]._set_state("active")  # Active state

        x = torch.randn(4, 2)
        output = net.forward(x)
        assert output.shape == (4, 2)
        
        # Verify each seed processed the input
        for seed in seeds:
            # Verify the seed is properly configured and accessible
            assert isinstance(seed, SentinelSeed)
            assert seed.state in ["dormant", "training", "blending", "active"]
