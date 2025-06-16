"""Tests for CLI flags, compatibility, and edge cases."""

from typing import cast

import torch

from morphogenetic_engine.components import BaseNet, SentinelSeed
from morphogenetic_engine.core import SeedManager


class TestCLIFlags:
    """Test CLI flags functionality for dynamic network configuration."""

    def test_num_layers_configuration(self):
        """Test num_layers flag with various values."""
        # Test default
        net = BaseNet(seed_manager=SeedManager(), input_dim=2)
        assert net.num_layers == 8
        assert len(net.layers) == 8
        assert len(net.all_seeds) == 8

        # Test custom values
        for num_layers in [1, 3, 5, 10, 16]:
            net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=num_layers)
            assert net.num_layers == num_layers
            assert len(net.layers) == num_layers
            assert len(net.all_seeds) == num_layers

    def test_seeds_per_layer_configuration(self):
        """Test seeds_per_layer flag with various values."""
        # Test default
        net = BaseNet(seed_manager=SeedManager(), input_dim=2)
        assert net.seeds_per_layer == 1
        assert len(net.all_seeds) == 8

        # Test custom values
        for seeds_per_layer in [1, 2, 3, 5]:
            net = BaseNet(
                seed_manager=SeedManager(),
                input_dim=2,
                num_layers=4,
                seeds_per_layer=seeds_per_layer,
            )
            assert net.seeds_per_layer == seeds_per_layer
            assert len(net.all_seeds) == 4 * seeds_per_layer

    def test_combined_flags_and_naming(self):
        """Test combined flags with proper seed ID naming."""
        test_cases = [
            (2, 1, 2, ["seed1_1", "seed2_1"]),
            (3, 2, 6, ["seed1_1", "seed1_2", "seed2_1", "seed2_2", "seed3_1", "seed3_2"]),
            (1, 3, 3, ["seed1_1", "seed1_2", "seed1_3"]),
        ]

        for num_layers, seeds_per_layer, expected_total, expected_ids in test_cases:
            net = BaseNet(
                seed_manager=SeedManager(),
                input_dim=2,
                num_layers=num_layers,
                seeds_per_layer=seeds_per_layer,
            )
            assert net.num_layers == num_layers
            assert net.seeds_per_layer == seeds_per_layer
            assert len(net.all_seeds) == expected_total

            actual_ids = [seed.seed_id for seed in net.all_seeds]
            assert actual_ids == expected_ids

    def test_helper_methods(self):
        """Test helper methods for multi-seed architecture."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=3, seeds_per_layer=2)

        # Test get_seeds_for_layer
        for layer_idx in range(3):
            layer_seeds = net.get_seeds_for_layer(layer_idx)
            assert len(layer_seeds) == 2
            for j, seed in enumerate(layer_seeds):
                expected_id = f"seed{layer_idx+1}_{j+1}"
                assert seed.seed_id == expected_id

        # Test get_total_seeds
        assert net.get_total_seeds() == 6

        # Test get_all_seeds
        all_seeds = net.get_all_seeds()
        assert len(all_seeds) == 6
        assert all_seeds == list(net.all_seeds)


class TestMultiSeedArchitecture:
    """Test multi-seed architecture behavior and averaging."""

    def test_forward_pass_variations(self):
        """Test forward pass with different seed configurations."""
        # Single seed (backward compatibility)
        net1 = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=2, seeds_per_layer=1)

        # Multiple seeds
        net2 = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=2, seeds_per_layer=3)

        x = torch.randn(4, 2)

        output1 = net1(x)
        output2 = net2(x)

        assert output1.shape == (4, 2)
        assert output2.shape == (4, 2)

    def test_seed_averaging_and_independence(self):
        """Test seed averaging behavior and independence."""
        net = BaseNet(
            seed_manager=SeedManager(), input_dim=2, num_layers=1, seeds_per_layer=3, hidden_dim=4
        )

        layer_seeds = net.get_seeds_for_layer(0)
        assert len(layer_seeds) == 3

        # Initialize seeds
        for seed in layer_seeds:
            seed.initialize_child()

        # Test independence
        for i, seed1 in enumerate(layer_seeds):
            for j, seed2 in enumerate(layer_seeds):
                if i != j:
                    assert seed1 is not seed2
                    assert seed1.seed_id != seed2.seed_id

        x = torch.randn(2, 2)
        output = net(x)
        assert output.shape == (2, 2)

    def test_gradient_flow_multi_seed(self):
        """Test gradient flow through multiple seeds."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=2, seeds_per_layer=2)

        # Initialize seeds
        for seed_module in net.all_seeds:
            seed_instance = cast(SentinelSeed, seed_module)
            seed_instance.initialize_child()

        x = torch.randn(4, 2, requires_grad=True)
        output = net(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

        # Check that some parameters have gradients
        has_gradients = any(
            param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad))
            for param in net.parameters()
            if param.requires_grad
        )
        assert has_gradients


class TestBackwardCompatibility:
    """Test backward compatibility with previous hardcoded architecture."""

    def test_default_architecture_equivalence(self):
        """Test that defaults match old hardcoded version."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2)

        assert net.num_layers == 8
        assert net.seeds_per_layer == 1
        assert len(net.all_seeds) == 8
        assert net.hidden_dim == 64

    def test_seeds_property_compatibility(self):
        """Test seeds property backward compatibility."""
        net = BaseNet(seed_manager=SeedManager(), input_dim=2)

        seeds = net.seeds
        assert len(seeds) == 8
        assert seeds == net.all_seeds

    def test_output_determinism(self):
        """Test that outputs are consistent and deterministic."""
        torch.manual_seed(42)
        net1 = BaseNet(seed_manager=SeedManager(), input_dim=2)

        torch.manual_seed(42)
        net2 = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=8, seeds_per_layer=1)

        x = torch.randn(4, 2)
        net1.eval()
        net2.eval()

        with torch.no_grad():
            output1 = net1(x)
            output2 = net2(x)

        assert torch.allclose(output1, output2, atol=1e-6)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_configurations(self):
        """Test minimal network configurations."""
        # Single layer, single seed
        net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=1, seeds_per_layer=1)
        x = torch.randn(2, 2)
        output = net(x)
        assert output.shape == (2, 2)

        # Single layer, many seeds
        net = BaseNet(seed_manager=SeedManager(), input_dim=2, num_layers=1, seeds_per_layer=10)
        assert len(net.all_seeds) == 10
        output = net(x)
        assert output.shape == (2, 2)

    def test_large_configurations(self):
        """Test large network configurations."""
        net = BaseNet(
            seed_manager=SeedManager(),
            input_dim=3,
            num_layers=20,
            seeds_per_layer=5,
            hidden_dim=128,
        )

        assert len(net.all_seeds) == 100

        x = torch.randn(2, 3)
        output = net(x)
        assert output.shape == (2, 2)  # Output is always 2D

    def test_parameter_scaling(self):
        """Test parameter count scaling with architecture size."""

        def count_parameters(net):
            return sum(p.numel() for p in net.parameters())

        # Compare networks with different seed counts
        net1 = BaseNet(
            seed_manager=SeedManager(), input_dim=2, num_layers=2, seeds_per_layer=1, hidden_dim=4
        )

        net2 = BaseNet(
            seed_manager=SeedManager(), input_dim=2, num_layers=2, seeds_per_layer=3, hidden_dim=4
        )

        params1 = count_parameters(net1)
        params2 = count_parameters(net2)

        # More seeds should mean more parameters
        assert params2 > params1

    def test_various_input_dimensions(self):
        """Test networks with different input dimensions."""
        for input_dim in [1, 2, 3, 5, 10]:
            net = BaseNet(
                seed_manager=SeedManager(), input_dim=input_dim, num_layers=2, seeds_per_layer=2
            )

            x = torch.randn(4, input_dim)
            output = net(x)
            assert output.shape == (4, 2)  # Output is always 2D
