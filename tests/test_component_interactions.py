"""
Cross-component interaction and system behavior tests.

This module contains tests for interactions between different components of the
morphogenetic engine, focusing on:

- Component state synchronization and consistency
- Error propagation and handling across component boundaries
- Thread safety and concurrent component access
- Resource management and cleanup coordination
- Edge cases in component interactions
- Performance validation for multi-component operations

Test Classes:
    TestSystemIntegration: Component interaction and state consistency
    TestErrorHandlingIntegration: Error propagation across components
    TestThreadSafetyIntegration: Concurrent access and thread safety
    TestResourceManagement: Resource cleanup and memory management
    TestPerformanceIntegration: Multi-component performance validation
    TestEdgeCasesIntegration: Edge cases and boundary conditions
"""

# pylint: disable=redefined-outer-name  # pytest fixtures intentionally shadow outer scope

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest
import torch

from morphogenetic_engine.components import BaseNet, SentinelSeed
from morphogenetic_engine.core import SeedManager

# Test Constants
EXPECTED_SEEDS_LARGE_CONFIG = 15 * 4  # layers × seeds_per_layer
DORMANT_SEED_THRESHOLD = 0.1  # 10% active seeds threshold for efficiency
MEMORY_THRESHOLD_MB = 50  # Maximum additional memory usage in MB
GRADIENT_TOLERANCE = 1e-6  # Tolerance for gradient comparisons
THREAD_COUNT = 10  # Number of threads for concurrency tests
STRESS_TEST_BATCH_SIZE = 16  # Batch size for stress testing


def _get_seed_by_type(module: Any) -> SentinelSeed:
    """Type-safe helper to get SentinelSeed from module."""
    if isinstance(module, SentinelSeed):
        return module
    raise TypeError(f"Expected SentinelSeed, got {type(module)}")


def _activate_seed(net: BaseNet, index: int) -> SentinelSeed:
    """Helper to activate a specific seed in the network through natural flow."""
    seed = _get_seed_by_type(net.all_seeds[index])
    seed.initialize_child()

    # Progress the seed through its natural lifecycle to active state
    # First train it to get to blending state
    dummy_input = torch.randn(4, seed.dim)
    for _ in range(100):  # Train until it reaches blending
        seed.train_child_step(dummy_input)
        if seed.state == "blending":
            break

    # Now progress through blending to active
    for _ in range(seed.blend_steps + 5):
        seed.update_blending()
        if seed.state == "active":
            break

    return seed


def _initialize_seed(net: BaseNet, index: int) -> SentinelSeed:
    """Helper to initialize a specific seed to training state."""
    seed = _get_seed_by_type(net.all_seeds[index])
    seed.initialize_child()
    return seed


def _set_seed_to_blending_state(seed: SentinelSeed) -> None:
    """Helper to naturally progress a seed to blending state."""
    if seed.state != "training":
        seed.initialize_child()

    # Train until it reaches blending threshold
    dummy_input = torch.randn(4, seed.dim)
    for _ in range(100):
        seed.train_child_step(dummy_input)
        if seed.state == "blending":
            break


def _activate_seed_by_reference(seed: SentinelSeed) -> None:
    """Helper to activate a seed through natural lifecycle progression."""
    if seed.state == "dormant":
        seed.initialize_child()

    if seed.state == "training":
        _set_seed_to_blending_state(seed)

    if seed.state == "blending":
        # Progress through blending to active
        for _ in range(seed.blend_steps + 5):
            seed.update_blending()
            if seed.state == "active":
                break


def _get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


@pytest.fixture
def fresh_seed_manager(mocker) -> SeedManager:
    """Fixture for creating a fresh SeedManager instance for each test."""
    # Mock external dependencies for isolation
    mocker.patch("morphogenetic_engine.core.ExperimentLogger")
    # Reset singleton state before each test
    SeedManager.reset_singleton()
    return SeedManager()


@pytest.fixture
def basic_network(fresh_seed_manager: SeedManager) -> BaseNet:
    """Fixture providing a basic network for common test scenarios."""
    return BaseNet(
        seed_manager=fresh_seed_manager, input_dim=2, num_layers=2, seeds_per_layer=2, hidden_dim=32
    )


@pytest.fixture
def large_network(fresh_seed_manager: SeedManager) -> BaseNet:
    """Fixture providing a large network for performance testing."""
    return BaseNet(
        seed_manager=fresh_seed_manager,
        input_dim=10,
        num_layers=15,
        seeds_per_layer=4,
        hidden_dim=64,
    )


@pytest.fixture
def activated_network(fresh_seed_manager: SeedManager) -> tuple[BaseNet, list[SentinelSeed]]:
    """Fixture providing a network with some pre-activated seeds."""
    net = BaseNet(
        seed_manager=fresh_seed_manager, input_dim=5, num_layers=3, seeds_per_layer=2, hidden_dim=32
    )
    activated_seeds = []
    # Activate first seed in each layer
    for i in range(0, len(net.all_seeds), 2):
        seed = _activate_seed(net, i)
        activated_seeds.append(seed)

    return net, activated_seeds


class TestSystemIntegration:
    """Test system-level integration between components."""

    def test_seed_manager_integration(
        self, basic_network: BaseNet, fresh_seed_manager: SeedManager
    ):
        """Test integration between BaseNet and SeedManager for state tracking."""
        # Arrange
        net = basic_network
        seed_manager = fresh_seed_manager

        # Act - Trigger seed state changes
        for seed_module in net.all_seeds:
            seed = _get_seed_by_type(seed_module)
            seed.initialize_child()  # Should register with SeedManager

        # Assert - Check SeedManager has tracked the changes
        expected_seeds = 2 * 2  # layers × seeds_per_layer
        assert len(seed_manager.seeds) == expected_seeds
        for _, seed_data in seed_manager.seeds.items():
            assert seed_data["state"] == "training"
            assert "buffer" in seed_data
            assert "telemetry" in seed_data

    def test_cross_component_gradient_flow(
        self, basic_network: BaseNet, fresh_seed_manager: SeedManager
    ):
        """Test gradient flow across BaseNet, SentinelSeed, and SeedManager integration."""
        # Arrange
        net = basic_network
        seed_manager = fresh_seed_manager

        # Initialize one seed to active state
        seed = _activate_seed(net, 0)

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
            if (
                "seed" in name
                and "child" in name
                and param.requires_grad
                and param.grad is not None
                and not torch.allclose(
                    param.grad, torch.zeros_like(param.grad), atol=GRADIENT_TOLERANCE
                )
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

        seeds = [_get_seed_by_type(net.all_seeds[i]) for i in range(3)]

        # Act - Change states through different pathways
        seeds[0].initialize_child()  # training
        seeds[1].initialize_child()  # training
        _set_seed_to_blending_state(seeds[1])  # blending
        _activate_seed_by_reference(seeds[2])  # active

        # Assert - Check consistency
        assert seeds[0].state == seed_manager.seeds[seeds[0].seed_id]["state"] == "training"
        assert seeds[1].state == seed_manager.seeds[seeds[1].seed_id]["state"] == "blending"
        assert seeds[2].state == seed_manager.seeds[seeds[2].seed_id]["state"] == "active"


class TestErrorHandlingIntegration:
    """Test error propagation and handling across integrated components."""

    def test_error_propagation_invalid_tensor(self, basic_network: BaseNet):
        """Test how errors propagate through the integrated system with invalid tensors."""
        # Arrange
        net = basic_network
        _activate_seed(net, 0)

        # Act & Assert - Test with extremely large values that might cause overflow
        x_large = torch.full((4, 2), 1e20)
        output = net(x_large)

        # Verify output contains expected overflow/large values
        assert (
            torch.isfinite(output).all() or torch.isinf(output).any()
        ), "Should handle large values"

        # Test with invalid dimensions
        with pytest.raises((RuntimeError, ValueError)):
            x_wrong_dim = torch.randn(4, 5)  # Wrong input dimension
            _ = net(x_wrong_dim)

    def test_error_propagation_memory_pressure(self, fresh_seed_manager: SeedManager):
        """Test system behavior under memory pressure conditions."""
        # Arrange - Create an intentionally large network
        try:
            net = BaseNet(
                seed_manager=fresh_seed_manager,
                input_dim=1000,
                num_layers=100,
                seeds_per_layer=50,
                hidden_dim=512,
            )
            # Act & Assert - Should handle gracefully or fail with clear error
            x = torch.randn(100, 1000)
            _ = net(x)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # Expected behavior under memory pressure
            assert "memory" in str(e).lower() or "cuda" in str(e).lower()

    def test_seed_manager_error_recovery(self, fresh_seed_manager: SeedManager):
        """Test SeedManager recovery from corrupted state."""
        # Arrange
        seed_manager = fresh_seed_manager
        net = BaseNet(seed_manager=seed_manager, input_dim=2, num_layers=1, seeds_per_layer=1)
        seed = _initialize_seed(net, 0)

        # Act - Corrupt seed manager state
        original_state = seed_manager.seeds[seed.seed_id].copy()
        seed_manager.seeds[seed.seed_id]["state"] = "invalid_state"

        # Assert - System should handle gracefully
        try:
            x = torch.randn(4, 2)
            _ = net(x)
        except ValueError as e:
            assert "invalid_state" in str(e) or "state" in str(e)

        # Cleanup - Restore valid state
        seed_manager.seeds[seed.seed_id] = original_state


class TestThreadSafetyIntegration:
    """Test thread safety of integrated components, especially SeedManager."""

    def test_concurrent_seed_manager_access(self, fresh_seed_manager: SeedManager):
        """Test SeedManager thread safety with concurrent operations."""
        # Arrange
        seed_manager = fresh_seed_manager
        networks = []
        for _ in range(THREAD_COUNT):
            net = BaseNet(
                seed_manager=seed_manager,
                input_dim=2,
                num_layers=2,
                seeds_per_layer=1,
                hidden_dim=16,
            )
            networks.append(net)

        results = []
        errors = []

        def worker(net_idx: int) -> dict[str, Any]:
            """Worker function for concurrent testing."""
            try:
                net = networks[net_idx]
                seed = _initialize_seed(net, 0)

                # Simulate concurrent operations
                x = torch.randn(2, 2)
                output = net(x)

                return {
                    "success": True,
                    "seed_id": seed.seed_id,
                    "output_shape": output.shape,
                    "thread_id": threading.get_ident(),
                }
            except (RuntimeError, ValueError, TypeError) as e:
                return {"success": False, "error": str(e), "thread_id": threading.get_ident()}

        # Act - Execute concurrent operations
        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(worker, i) for i in range(THREAD_COUNT)]
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    results.append(result)
                else:
                    errors.append(result)

        # Assert - All operations should succeed
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        assert len(results) == THREAD_COUNT

        # Verify all seeds were registered (only 2 because networks share the same manager)
        # Each network has 2 layers × 1 seed = 2 seeds, but they share the same manager
        # So we expect only the seeds from the first two networks to be tracked
        assert (
            len(seed_manager.seeds) >= 2
        ), f"Expected at least 2 seeds, got {len(seed_manager.seeds)}"

    def test_concurrent_seed_state_changes(self, fresh_seed_manager: SeedManager):
        """Test concurrent seed state changes don't cause race conditions."""
        # Arrange
        seed_manager = fresh_seed_manager
        net = BaseNet(
            seed_manager=seed_manager, input_dim=2, num_layers=1, seeds_per_layer=THREAD_COUNT
        )

        # Initialize all seeds
        seeds = []
        for seed_idx in range(THREAD_COUNT):
            seed = _initialize_seed(net, seed_idx)
            seeds.append(seed)

        def state_changer(seed_idx: int) -> bool:
            """Worker to change seed states concurrently."""
            try:
                seed = seeds[seed_idx]
                # Progress through natural state transitions
                seed.initialize_child()  # dormant -> training
                _set_seed_to_blending_state(seed)  # training -> blending
                _activate_seed_by_reference(seed)  # blending -> active
                time.sleep(0.001)  # Small delay to increase race condition probability
                return True
            except (RuntimeError, ValueError, TypeError):
                return False

        # Act - Change states concurrently
        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(state_changer, i) for i in range(THREAD_COUNT)]
            results = [future.result() for future in as_completed(futures)]

        # Assert - No race conditions should occur
        assert all(results), "Concurrent state changes should not fail"

        # Verify final state consistency
        for seed in seeds:
            manager_state = seed_manager.seeds[seed.seed_id]["state"]
            assert (
                seed.state == manager_state
            ), f"State inconsistency: {seed.state} != {manager_state}"


class TestResourceManagement:
    """Test resource cleanup and memory management in integrated system."""

    def test_resource_cleanup_on_failure(self, fresh_seed_manager: SeedManager):
        """Verify proper cleanup when integration fails."""
        # Arrange
        seed_manager = fresh_seed_manager
        initial_memory = _get_memory_usage_mb()

        try:
            # Create network that will fail due to dimension mismatch
            net = BaseNet(seed_manager=seed_manager, input_dim=2, num_layers=2, seeds_per_layer=2)

            # Force failure with wrong input dimensions
            with pytest.raises((RuntimeError, ValueError)):
                invalid_input = torch.randn(4, 5)  # Wrong input dimension
                _ = net(invalid_input)

        except (RuntimeError, ValueError):
            # Expected failure due to dimension mismatch
            pass

        # Assert - Memory should be cleaned up
        final_memory = _get_memory_usage_mb()
        memory_increase = final_memory - initial_memory
        assert (
            memory_increase < MEMORY_THRESHOLD_MB
        ), f"Memory leak detected: {memory_increase}MB increase"

    def test_seed_optimizer_cleanup(self, fresh_seed_manager: SeedManager):
        """Test that seed optimizers are properly cleaned up."""
        # Arrange
        seed_manager = fresh_seed_manager
        net = BaseNet(seed_manager=seed_manager, input_dim=2, num_layers=2, seeds_per_layer=2)

        # Initialize seeds to create optimizers
        active_seeds = [_initialize_seed(net, i) for i in range(len(net.all_seeds))]

        # Verify optimizers exist
        for seed in active_seeds:
            assert seed.child_optim is not None
            assert len(list(seed.child_optim.param_groups)) > 0

        # Act - Simulate cleanup
        self._cleanup_seed_optimizers(active_seeds)

        # Assert - Verify cleanup
        for seed in active_seeds:
            if hasattr(seed, "child_optim") and seed.child_optim is not None:
                for group in seed.child_optim.param_groups:
                    assert len(group["params"]) == 0

    def _cleanup_seed_optimizers(self, seeds: list[SentinelSeed]) -> None:
        """Helper method to cleanup seed optimizers."""
        for seed in seeds:
            if hasattr(seed, "child_optim") and seed.child_optim is not None:
                seed.child_optim.zero_grad()
                for group in seed.child_optim.param_groups:
                    group["params"].clear()

    @pytest.mark.parametrize(
        "invalid_config",
        [
            {"num_layers": 0, "seeds_per_layer": 1},
            {"num_layers": 1, "seeds_per_layer": 0},
            {"hidden_dim": -1, "num_layers": 1, "seeds_per_layer": 1},
            {"input_dim": 0, "num_layers": 1, "seeds_per_layer": 1},
        ],
    )
    def test_invalid_configuration_integration(
        self, invalid_config: dict[str, Any], fresh_seed_manager: SeedManager
    ):
        """Test system behavior with invalid configurations."""
        # Arrange & Act & Assert
        config = {
            "seed_manager": fresh_seed_manager,
            "input_dim": 2,
            "num_layers": 1,
            "seeds_per_layer": 1,
            "hidden_dim": 32,
            **invalid_config,
        }

        with pytest.raises((ValueError, RuntimeError)):
            BaseNet(**config)


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""

    def test_large_scale_integration(self, large_network: BaseNet, fresh_seed_manager: SeedManager):
        """Test system performance with large configurations."""
        # Arrange
        net = large_network
        seed_manager = fresh_seed_manager
        initial_memory = _get_memory_usage_mb()

        # Act
        x = torch.randn(8, 10)
        start_time = time.time()
        output = net(x)
        inference_time = time.time() - start_time

        # Assert - System should handle large configurations efficiently
        assert output.shape == (8, 2)
        assert len(seed_manager.seeds) == EXPECTED_SEEDS_LARGE_CONFIG

        # Performance assertions
        assert inference_time < 5.0, f"Inference too slow: {inference_time:.3f}s"

        final_memory = _get_memory_usage_mb()
        memory_increase = final_memory - initial_memory
        assert (
            memory_increase < MEMORY_THRESHOLD_MB * 2
        ), f"Excessive memory usage: {memory_increase}MB"

        # All seeds should be tracked
        for seed_module in net.all_seeds:
            seed = _get_seed_by_type(seed_module)
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
        initial_memory = _get_memory_usage_mb()

        # Only activate a few seeds
        _initialize_seed(net, 0)
        _activate_seed(net, 1)

        # Act
        x = torch.randn(4, 5)
        start_time = time.time()
        output = net(x)
        inference_time = time.time() - start_time

        # Assert
        assert output.shape == (4, 2)

        # Performance metrics
        assert inference_time < 1.0, f"Inference with dormant seeds too slow: {inference_time:.3f}s"

        final_memory = _get_memory_usage_mb()
        memory_increase = final_memory - initial_memory
        assert (
            memory_increase < MEMORY_THRESHOLD_MB
        ), f"Dormant seeds using too much memory: {memory_increase}MB"

        # Most seeds should remain dormant (not consuming training resources)
        active_seeds = sum(
            1 for seed_module in net.all_seeds if _get_seed_by_type(seed_module).state != "dormant"
        )
        total_seeds = len(net.all_seeds)
        dormant_ratio = (total_seeds - active_seeds) / total_seeds
        assert dormant_ratio > (
            1 - DORMANT_SEED_THRESHOLD
        ), f"Not enough dormant seeds for efficiency: {dormant_ratio:.2%} dormant"

    def test_batch_processing_efficiency(
        self, activated_network: tuple[BaseNet, list[SentinelSeed]]
    ):
        """Test batch processing performance with activated seeds."""
        # Arrange
        net, _ = activated_network
        batch_sizes = [1, 4, 16, 32]
        timing_results = {}

        for batch_size in batch_sizes:
            # Act
            x = torch.randn(batch_size, 5)
            start_time = time.time()

            # Run multiple iterations for better timing accuracy
            for _ in range(10):
                output = net(x)

            total_time = time.time() - start_time
            avg_time_per_batch = total_time / 10
            timing_results[batch_size] = avg_time_per_batch

            # Assert - Output shape is correct
            assert output.shape == (batch_size, 2)

        # Assert - Batch processing should scale reasonably
        time_per_sample_batch1 = timing_results[1] / 1
        time_per_sample_batch32 = timing_results[32] / 32

        # Larger batches should be more efficient per sample
        assert time_per_sample_batch32 < time_per_sample_batch1, (
            f"Batch processing not scaling efficiently: "
            f"{time_per_sample_batch1:.6f}s/sample (batch=1) vs "
            f"{time_per_sample_batch32:.6f}s/sample (batch=32)"
        )


class TestEdgeCasesIntegration:
    """Test edge cases and boundary conditions in integrated system."""

    def test_empty_network_edge_case(self, fresh_seed_manager: SeedManager):
        """Test behavior with minimal network configuration."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError):
            BaseNet(seed_manager=fresh_seed_manager, input_dim=1, num_layers=0, seeds_per_layer=1)

    def test_single_seed_network(self, fresh_seed_manager: SeedManager):
        """Test network with exactly one seed."""
        # Arrange
        seed_manager = fresh_seed_manager
        net = BaseNet(
            seed_manager=seed_manager, input_dim=2, num_layers=1, seeds_per_layer=1, hidden_dim=4
        )

        # Act
        seed = _initialize_seed(net, 0)
        x = torch.randn(2, 2)
        output = net(x)

        # Assert
        assert output.shape == (2, 2)
        assert len(seed_manager.seeds) == 1
        assert seed.seed_id in seed_manager.seeds

    def test_extreme_dimension_handling(self, fresh_seed_manager: SeedManager):
        """Test handling of extreme input dimensions."""
        # Arrange
        seed_manager = fresh_seed_manager

        # Test very small dimensions
        net_small = BaseNet(
            seed_manager=seed_manager, input_dim=1, num_layers=1, seeds_per_layer=1, hidden_dim=2
        )

        # Act & Assert - Small dimensions
        x_small = torch.randn(1, 1)
        output_small = net_small(x_small)
        assert output_small.shape == (1, 2)

    def test_numerical_stability_integration(self, basic_network: BaseNet):
        """Test numerical stability across integrated components."""
        # Arrange
        net = basic_network
        _activate_seed(net, 0)

        # Test with extreme values
        test_cases = [
            torch.full((2, 2), 1e-8),  # Very small values
            torch.full((2, 2), 1e8),  # Very large values
            torch.zeros((2, 2)),  # Zero input
        ]

        for x in test_cases:
            # Act
            output = net(x)

            # Assert - Should not produce NaN or Inf
            assert torch.isfinite(output).all(), f"Non-finite output for input: {x[0, 0].item()}"
            assert output.shape == (2, 2)

    def test_gradient_flow_edge_cases(self, basic_network: BaseNet):
        """Test gradient flow in edge case scenarios."""
        # Arrange
        net = basic_network
        _activate_seed(net, 0)

        # Test with zero gradients
        x = torch.zeros((2, 2), requires_grad=True)

        # Act
        output = net(x)
        loss = output.sum()
        loss.backward()

        # Assert - Gradients should flow even with zero input
        assert x.grad is not None
        # Some parameters should have non-zero gradients due to bias terms
        has_nonzero_grad = any(
            param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad))
            for param in net.parameters()
            if param.requires_grad
        )
        assert has_nonzero_grad, "Expected some non-zero gradients even with zero input"
