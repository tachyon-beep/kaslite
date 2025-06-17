"""Integration tests for cross-component interactions and system-level behavior."""

from typing import cast

import pytest
import torch

from morphogenetic_engine.components import BaseNet, SentinelSeed
from morphogenetic_engine.core import SeedManager


@pytest.fixture
def fresh_seed_manager() -> SeedManager:
    """Fixture for creating a fresh SeedManager instance for each test."""
    # Reset singleton state before each test
    SeedManager.reset_singleton()
    return SeedManager()


class TestSystemIntegration:
    """Test system-level integration between components."""

    def test_seed_manager_integration(self, fresh_seed_manager: SeedManager):
        """Test integration between BaseNet and SeedManager for state tracking."""
        # Arrange
        seed_manager = fresh_seed_manager
        net = BaseNet(seed_manager=seed_manager, input_dim=2, num_layers=2, seeds_per_layer=2)

        # Act - Trigger seed state changes
        for seed_module in net.all_seeds:
            seed = cast(SentinelSeed, seed_module)
            seed.initialize_child()  # Should register with SeedManager

        # Assert - Check SeedManager has tracked the changes
        assert len(seed_manager.seeds) == 4  # 2 layers × 2 seeds
        for seed_id, seed_data in seed_manager.seeds.items():
            assert seed_data["state"] == "training"
            assert "buffer" in seed_data
            assert "telemetry" in seed_data

    def test_cross_component_gradient_flow(self, fresh_seed_manager: SeedManager):
        """Test gradient flow across BaseNet, SentinelSeed, and SeedManager integration."""
        # Arrange
        seed_manager = fresh_seed_manager
        net = BaseNet(seed_manager=seed_manager, input_dim=2, num_layers=2, seeds_per_layer=1)

        # Initialize one seed to active state
        seed = cast(SentinelSeed, net.all_seeds[0])
        seed.initialize_child()
        seed._set_state("active")

        x = torch.randn(4, 2, requires_grad=True)

        # Act
        output = net(x)
        loss = output.sum()
        loss.backward()

        # Assert - Check integration worked
        assert x.grad is not None
        assert seed_manager.seeds[seed.seed_id]["state"] == "active"

        # Check that active seed child parameters got gradients
        found_active_seed_grad = False
        for name, param in net.named_parameters():
            if "seed" in name and "child" in name and param.requires_grad:
                if param.grad is not None and not torch.allclose(
                    param.grad, torch.zeros_like(param.grad)
                ):
                    found_active_seed_grad = True
                    break
        assert (
            found_active_seed_grad
        ), "Active seed should have received gradients through integration"

    def test_multi_component_state_consistency(self, fresh_seed_manager: SeedManager):
        """Test that state remains consistent across BaseNet and SeedManager."""
        # Arrange
        seed_manager = fresh_seed_manager
        net = BaseNet(seed_manager=seed_manager, input_dim=2, num_layers=1, seeds_per_layer=3)

        seeds = [cast(SentinelSeed, net.all_seeds[i]) for i in range(3)]

        # Act - Change states through different pathways
        seeds[0].initialize_child()  # training
        seeds[1].initialize_child()  # training
        seeds[1]._set_state("blending")  # blending
        seeds[2]._set_state("active")  # active

        # Assert - Check consistency
        assert seeds[0].state == seed_manager.seeds[seeds[0].seed_id]["state"] == "training"
        assert seeds[1].state == seed_manager.seeds[seeds[1].seed_id]["state"] == "blending"
        assert seeds[2].state == seed_manager.seeds[seeds[2].seed_id]["state"] == "active"


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""

    def test_large_scale_integration(self, fresh_seed_manager: SeedManager):
        """Test system performance with large configurations."""
        # Arrange
        seed_manager = fresh_seed_manager
        net = BaseNet(
            seed_manager=seed_manager,
            input_dim=10,
            num_layers=15,
            seeds_per_layer=4,
            hidden_dim=64,
        )

        # Act
        x = torch.randn(8, 10)
        output = net(x)

        # Assert - System should handle large configurations
        assert output.shape == (8, 2)
        assert len(seed_manager.seeds) == 60  # 15 × 4

        # All seeds should be tracked
        for seed_module in net.all_seeds:
            seed = cast(SentinelSeed, seed_module)
            assert seed.seed_id in seed_manager.seeds

    def test_memory_efficiency_with_dormant_seeds(self, fresh_seed_manager: SeedManager):
        """Test memory efficiency when most seeds are dormant."""
        # Arrange
        seed_manager = fresh_seed_manager
        net = BaseNet(
            seed_manager=seed_manager,
            input_dim=5,
            num_layers=10,
            seeds_per_layer=3,
            hidden_dim=32,
        )

        # Only activate a few seeds
        cast(SentinelSeed, net.all_seeds[0]).initialize_child()
        cast(SentinelSeed, net.all_seeds[1]).initialize_child()
        cast(SentinelSeed, net.all_seeds[1])._set_state("active")

        # Act
        x = torch.randn(4, 5)
        output = net(x)

        # Assert
        assert output.shape == (4, 2)

        # Most seeds should remain dormant (not consuming training resources)
        active_seeds = sum(
            1 for seed_module in net.all_seeds if cast(SentinelSeed, seed_module).state != "dormant"
        )
        total_seeds = len(net.all_seeds)
        assert active_seeds < total_seeds * 0.1, "Most seeds should remain dormant for efficiency"
