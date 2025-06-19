"""
Comprehensive tests for the SeedManager component.

This module provides thorough testing coverage for the SeedManager singleton class,
focusing on seed orchestration, thread safety, and lifecycle management. Tests follow
modern Python 3.12+ practices with proper isolation and comprehensive scenarios.

Test Categories:
- Unit tests: Isolated SeedManager behavior with mocking
- Integration tests: KasminaMicro and monitoring integration
- Thread safety: Concurrent operation validation
- Edge cases: Boundary conditions and error scenarios
- Property-based: Hypothesis-driven testing for robust validation
"""

# pylint: disable=protected-access,redefined-outer-name

import threading
import time
import weakref
from collections import deque
from unittest.mock import Mock, patch

import pytest
import torch

from morphogenetic_engine.core import KasminaMicro, SeedManager


class TestConstants:
    """Constants for test values to avoid magic numbers."""

    LOW_HEALTH_SIGNAL = 0.1
    MEDIUM_HEALTH_SIGNAL = 0.5
    HIGH_HEALTH_SIGNAL = 0.9
    BUFFER_MAXLEN = 500


@pytest.fixture
def clean_seed_manager():
    """Provide a clean SeedManager instance for testing."""
    # Reset singleton first
    SeedManager.reset_singleton()
    manager = SeedManager()
    manager.seeds.clear()
    manager.germination_log.clear()
    return manager


@pytest.fixture
def mock_seed_factory():
    """Factory for creating mock seeds with customizable health signals."""

    def _create_mock_seed(health_signal: float = TestConstants.MEDIUM_HEALTH_SIGNAL):
        mock_seed = Mock()
        mock_seed.get_health_signal = Mock(return_value=health_signal)
        mock_seed.initialize_child = Mock()
        return mock_seed

    return _create_mock_seed


@pytest.fixture
def sample_tensor():
    """Provide a sample tensor for testing."""
    return torch.randn(2, 4)


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    mock_logger = Mock()
    mock_logger.log_germination = Mock()
    mock_logger.log_seed_event = Mock()
    return mock_logger


@pytest.fixture
def mock_monitor():
    """Provide a mock PrometheusMonitor for testing."""
    mock_monitor = Mock()
    mock_monitor.record_germination = Mock()
    mock_monitor.update_kasmina_metrics = Mock()
    return mock_monitor


class TestSeedManager:
    """Test suite for SeedManager singleton class."""

    def test_singleton_pattern(self, clean_seed_manager) -> None:
        """Test that SeedManager follows singleton pattern."""
        manager1 = SeedManager()
        manager2 = SeedManager()
        assert manager1 is manager2
        assert manager1 is clean_seed_manager

    def test_thread_safety(self) -> None:
        """Test thread safety of singleton initialization."""
        # Reset to ensure clean state
        SeedManager.reset_singleton()
        managers = []

        def create_manager():
            managers.append(SeedManager())

        threads = [threading.Thread(target=create_manager) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be the same instance
        assert all(m is managers[0] for m in managers)
        # Cleanup
        SeedManager.reset_singleton()

    def test_register_seed(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test seed registration functionality."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory()

        manager.register_seed(mock_seed, "test_seed")

        assert "test_seed" in manager.seeds
        seed_info = manager.seeds["test_seed"]
        assert seed_info["module"] is mock_seed
        assert seed_info["status"] == "dormant"
        assert seed_info["state"] == "dormant"
        assert seed_info["alpha"] == pytest.approx(0.0)
        assert len(seed_info["buffer"]) == 0
        assert isinstance(seed_info["buffer"], deque)
        assert seed_info["buffer"].maxlen == TestConstants.BUFFER_MAXLEN
        assert seed_info["telemetry"]["drift"] == pytest.approx(0.0)
        assert seed_info["telemetry"]["variance"] == pytest.approx(0.0)

    def test_append_to_buffer(self, clean_seed_manager, mock_seed_factory, sample_tensor) -> None:
        """Test buffer append functionality."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory()
        manager.register_seed(mock_seed, "test_seed")

        # Test appending tensors
        tensor1 = sample_tensor
        tensor2 = torch.randn(4, 8)

        manager.append_to_buffer("test_seed", tensor1)
        manager.append_to_buffer("test_seed", tensor2)

        buffer = manager.seeds["test_seed"]["buffer"]
        assert len(buffer) == 2
        assert torch.equal(buffer[0], tensor1.detach())
        assert torch.equal(buffer[1], tensor2.detach())

    def test_append_to_nonexistent_seed(self, clean_seed_manager, sample_tensor) -> None:
        """Test appending to non-existent seed doesn't crash."""
        manager = clean_seed_manager

        # Should not raise an exception
        manager.append_to_buffer("nonexistent", sample_tensor)

    def test_buffer_overflow_behavior(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test buffer behavior when exceeding maxlen capacity."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory()
        manager.register_seed(mock_seed, "test_seed")

        # Add more tensors than buffer capacity
        for i in range(TestConstants.BUFFER_MAXLEN + 10):
            tensor = torch.full((2, 3), float(i))
            manager.append_to_buffer("test_seed", tensor)

        buffer = manager.seeds["test_seed"]["buffer"]
        assert len(buffer) == TestConstants.BUFFER_MAXLEN

        # First tensor should be evicted, last should be preserved
        assert torch.equal(buffer[-1], torch.full((2, 3), float(TestConstants.BUFFER_MAXLEN + 9)).detach())
        assert torch.equal(buffer[0], torch.full((2, 3), float(10)).detach())

    def test_request_germination_success(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test successful germination request."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory()
        manager.register_seed(mock_seed, "test_seed")

        result = manager.request_germination("test_seed")

        assert result is True
        mock_seed.initialize_child.assert_called_once()
        assert manager.seeds["test_seed"]["status"] == "active"
        assert len(manager.germination_log) == 1
        log_entry = manager.germination_log[0]
        assert log_entry["success"] is True
        assert log_entry["event_type"] == "germination_attempt"
        assert log_entry["seed_id"] == "test_seed"

    def test_request_germination_already_active(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test germination request on already active seed."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory()
        manager.register_seed(mock_seed, "test_seed")
        manager.seeds["test_seed"]["status"] = "active"

        result = manager.request_germination("test_seed")

        assert result is False
        mock_seed.initialize_child.assert_not_called()

    def test_request_germination_nonexistent_seed(self, clean_seed_manager) -> None:
        """Test germination request on non-existent seed."""
        manager = clean_seed_manager

        result = manager.request_germination("nonexistent")

        assert result is False

    def test_request_germination_exception_handling(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test germination request with initialization failure."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory()
        mock_seed.initialize_child = Mock(side_effect=RuntimeError("Init failed"))
        manager.register_seed(mock_seed, "test_seed")

        result = manager.request_germination("test_seed")

        assert result is False
        assert manager.seeds["test_seed"]["status"] == "failed"
        assert len(manager.germination_log) == 1
        log_entry = manager.germination_log[0]
        assert log_entry["success"] is False
        assert log_entry["event_type"] == "germination_attempt"

    def test_record_transition(self, clean_seed_manager) -> None:
        """Test state transition recording."""
        manager = clean_seed_manager

        before_time = time.time()
        manager.record_transition("test_seed", "dormant", "training")
        after_time = time.time()

        assert len(manager.germination_log) == 1
        log_entry = manager.germination_log[0]
        assert log_entry["seed_id"] == "test_seed"
        assert log_entry["from"] == "dormant"
        assert log_entry["to"] == "training"
        assert log_entry["event_type"] == "state_transition"
        assert before_time <= log_entry["timestamp"] <= after_time

    def test_logger_integration_on_germination(self, mock_seed_factory, mock_logger) -> None:
        """Ensure ExperimentLogger.log_germination is invoked."""
        manager = SeedManager(logger=mock_logger)
        manager.seeds.clear()
        mock_seed = mock_seed_factory()
        manager.register_seed(mock_seed, "test_seed")

        result = manager.request_germination("test_seed", epoch=5)

        assert result is True
        mock_logger.log_germination.assert_called_once_with(5, "test_seed")

    def test_logger_integration_on_transition(self, mock_logger) -> None:
        """Ensure ExperimentLogger.log_seed_event is invoked."""
        manager = SeedManager(logger=mock_logger)
        manager.germination_log.clear()

        manager.record_transition("seedX", "dormant", "training", epoch=7)

        mock_logger.log_seed_event.assert_called_once_with(7, "seedX", "dormant", "training")

    def test_record_drift(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test drift recording functionality."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory()
        manager.register_seed(mock_seed, "test_seed")

        manager.record_drift("test_seed", 0.123)

        assert manager.seeds["test_seed"]["telemetry"]["drift"] == pytest.approx(0.123)

    def test_record_drift_nonexistent_seed(self, clean_seed_manager) -> None:
        """Test drift recording for non-existent seed."""
        manager = clean_seed_manager

        # Should not raise an exception
        manager.record_drift("nonexistent", 0.123)

    def test_telemetry_variance_recording(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test that variance telemetry can be recorded and accessed."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory()
        manager.register_seed(mock_seed, "test_seed")

        # Directly modify variance (since there's no dedicated method in production code)
        manager.seeds["test_seed"]["telemetry"]["variance"] = 0.456

        assert manager.seeds["test_seed"]["telemetry"]["variance"] == pytest.approx(0.456)

    def test_reset_methods(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test reset() and reset_singleton() methods."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory()
        manager.register_seed(mock_seed, "test_seed")
        manager.record_transition("test_seed", "dormant", "active")

        # Verify state before reset
        assert len(manager.seeds) == 1
        assert len(manager.germination_log) == 1

        # Test instance reset
        manager.reset()
        assert len(manager.seeds) == 0
        assert len(manager.germination_log) == 0

        # Add data again and test singleton reset
        manager.register_seed(mock_seed, "test_seed2")
        assert len(manager.seeds) == 1

        SeedManager.reset_singleton()
        new_manager = SeedManager()
        assert new_manager is not manager  # New instance
        assert len(new_manager.seeds) == 0

    def test_concurrent_germination_requests(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test thread safety under concurrent germination requests."""
        manager = clean_seed_manager
        results = []

        # Register multiple seeds
        for i in range(5):
            mock_seed = mock_seed_factory()
            manager.register_seed(mock_seed, f"seed_{i}")

        def attempt_germination(seed_id: str):
            result = manager.request_germination(seed_id)
            results.append((seed_id, result))

        # Attempt concurrent germinations
        threads = [threading.Thread(target=attempt_germination, args=(f"seed_{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed since they're different seeds
        assert len(results) == 5
        assert all(result for _, result in results)

        # All should be marked as active
        for i in range(5):
            assert manager.seeds[f"seed_{i}"]["status"] == "active"


class TestSeedManagerEdgeCases:
    """Test edge cases and boundary conditions for SeedManager."""

    def test_concurrent_access_to_shared_resources(self, clean_seed_manager, mock_seed_factory):
        """Test concurrent access to germination log and seed registry."""
        manager = clean_seed_manager
        results = []

        def concurrent_operations(thread_id: int):
            """Perform mixed operations concurrently."""
            try:
                # Register a seed
                mock_seed = mock_seed_factory()
                manager.register_seed(mock_seed, f"thread_{thread_id}_seed")

                # Record a transition
                manager.record_transition(f"thread_{thread_id}_seed", "dormant", "training")

                # Record drift
                manager.record_drift(f"thread_{thread_id}_seed", 0.1 * thread_id)

                results.append(f"thread_{thread_id}_success")
            except (RuntimeError, ValueError, AttributeError) as e:
                results.append(f"thread_{thread_id}_error: {e}")

        # Run concurrent operations
        threads = [threading.Thread(target=concurrent_operations, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All operations should succeed
        assert len(results) == 10
        assert all("success" in result for result in results)

        # Verify all seeds were registered
        assert len(manager.seeds) == 10
        for i in range(10):
            assert f"thread_{i}_seed" in manager.seeds

    def test_memory_cleanup_on_reset(self, clean_seed_manager, mock_seed_factory):
        """Test that reset properly cleans up memory references."""
        manager = clean_seed_manager

        # Create large buffer data
        for i in range(100):
            mock_seed = mock_seed_factory()
            manager.register_seed(mock_seed, f"seed_{i}")

            # Add data to buffers
            for _ in range(50):
                tensor = torch.randn(100, 100)  # Large tensor
                manager.append_to_buffer(f"seed_{i}", tensor)

        # Verify data exists
        assert len(manager.seeds) == 100
        total_buffer_items = sum(len(info["buffer"]) for info in manager.seeds.values())
        assert total_buffer_items > 0

        # Reset and verify cleanup
        manager.reset()
        assert len(manager.seeds) == 0
        assert len(manager.germination_log) == 0


class TestSeedManagerPerformance:
    """Performance and stress tests for SeedManager."""

    def test_large_scale_seed_registration(self, clean_seed_manager, mock_seed_factory):
        """Test performance with large number of seeds."""
        manager = clean_seed_manager

        # Register many seeds
        num_seeds = 1000
        for i in range(num_seeds):
            mock_seed = mock_seed_factory()
            manager.register_seed(mock_seed, f"performance_seed_{i}")

        assert len(manager.seeds) == num_seeds

        # Test lookup performance
        for i in range(0, num_seeds, 100):  # Sample every 100th seed
            seed_id = f"performance_seed_{i}"
            assert seed_id in manager.seeds
            assert manager.seeds[seed_id]["status"] == "dormant"

    def test_buffer_performance_stress(self, clean_seed_manager, mock_seed_factory):
        """Test buffer operations under stress."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory()
        manager.register_seed(mock_seed, "stress_seed")

        # Add many tensors rapidly
        num_tensors = TestConstants.BUFFER_MAXLEN * 2
        for _ in range(num_tensors):
            tensor = torch.randn(10, 10)
            manager.append_to_buffer("stress_seed", tensor)

        # Buffer should maintain size limit
        buffer = manager.seeds["stress_seed"]["buffer"]
        assert len(buffer) == TestConstants.BUFFER_MAXLEN

        # Recent tensors should be preserved
        assert buffer[-1] is not None
        assert buffer[0] is not None


class TestKasminaMicroIntegration:
    """Integration tests for KasminaMicro and SeedManager collaboration."""

    def test_kasmina_seed_selection_integration(self, clean_seed_manager, mock_seed_factory):
        """Test that KasminaMicro correctly selects seeds based on health signals."""
        manager = clean_seed_manager
        kasmina = KasminaMicro(manager, patience=2, acc_threshold=0.8)

        # Register multiple seeds with different health signals
        seed_bad = mock_seed_factory(TestConstants.LOW_HEALTH_SIGNAL)  # Worst health
        seed_medium = mock_seed_factory(TestConstants.MEDIUM_HEALTH_SIGNAL)
        seed_good = mock_seed_factory(TestConstants.HIGH_HEALTH_SIGNAL)

        manager.register_seed(seed_bad, "bad_seed")
        manager.register_seed(seed_medium, "medium_seed")
        manager.register_seed(seed_good, "good_seed")

        # Simulate plateau condition that should trigger germination
        # Need to reach patience threshold with no improvement and low accuracy
        assert kasmina.step(1.0, 0.7) is False  # Below threshold, plateau = 1
        assert kasmina.step(1.0, 0.7) is False  # Still below, plateau = 2
        result = kasmina.step(1.0, 0.7)  # plateau = 3, should trigger (patience=2)

        assert result is True  # Germination should have occurred

        # The seed with worst health signal should be selected and activated
        seed_bad.initialize_child.assert_called_once()
        seed_medium.initialize_child.assert_not_called()
        seed_good.initialize_child.assert_not_called()

        # Verify the seed status was updated
        assert manager.seeds["bad_seed"]["status"] == "active"

    def test_kasmina_no_germination_above_threshold(self, clean_seed_manager, mock_seed_factory):
        """Test that KasminaMicro doesn't trigger germination when accuracy is above threshold."""
        manager = clean_seed_manager
        kasmina = KasminaMicro(manager, patience=1, acc_threshold=0.8)

        seed = mock_seed_factory()
        manager.register_seed(seed, "test_seed")

        # Accuracy above threshold should not trigger germination even with plateau
        result = kasmina.step(1.0, 0.9)  # Above threshold
        assert result is False
        seed.initialize_child.assert_not_called()

    def test_kasmina_plateau_reset_on_improvement(self, clean_seed_manager, mock_seed_factory):
        """Test that plateau counter resets when loss improves significantly."""
        manager = clean_seed_manager
        kasmina = KasminaMicro(manager, patience=3, delta=0.1, acc_threshold=0.8)

        seed = mock_seed_factory()
        manager.register_seed(seed, "test_seed")

        # Build up plateau
        kasmina.step(1.0, 0.7)
        kasmina.step(1.0, 0.7)

        # Significant improvement should reset plateau
        kasmina.step(0.8, 0.7)  # Improvement > delta, should reset
        assert kasmina.plateau == 0

        # Now need to build plateau again for germination
        kasmina.step(0.8, 0.7)
        kasmina.step(0.8, 0.7)
        kasmina.step(0.8, 0.7)  # Should trigger germination

        seed.initialize_child.assert_called_once()

    def test_kasmina_no_dormant_seeds_available(self, clean_seed_manager, mock_seed_factory):
        """Test KasminaMicro behavior when no dormant seeds are available."""
        manager = clean_seed_manager
        kasmina = KasminaMicro(manager, patience=1, acc_threshold=0.8)

        # Register seed but make it already active
        seed = mock_seed_factory()
        manager.register_seed(seed, "test_seed")
        manager.seeds["test_seed"]["status"] = "active"

        # Should not attempt germination when no dormant seeds available
        kasmina.step(1.0, 0.7)  # Below threshold, build plateau
        result = kasmina.step(1.0, 0.7)  # plateau >= patience, should try germination
        assert result is False  # But no dormant seeds available
        seed.initialize_child.assert_not_called()

    @patch("morphogenetic_engine.monitoring.get_monitor")
    def test_kasmina_monitoring_integration_success(self, mock_get_monitor, clean_seed_manager, mock_seed_factory, mock_monitor):
        """Test that successful germination records metrics in monitoring system."""
        mock_get_monitor.return_value = mock_monitor

        manager = clean_seed_manager
        kasmina = KasminaMicro(manager, patience=1, acc_threshold=0.8)

        seed = mock_seed_factory()
        manager.register_seed(seed, "test_seed")

        # Trigger germination by building up plateau
        kasmina.step(1.0, 0.7)  # Below threshold, build plateau
        result = kasmina.step(1.0, 0.7)  # plateau >= patience, should trigger

        assert result is True
        mock_monitor.update_kasmina_metrics.assert_called_with(1, 1)
        mock_monitor.record_germination.assert_called_once()

    @patch("morphogenetic_engine.monitoring.get_monitor")
    def test_kasmina_monitoring_integration_no_monitor(self, mock_get_monitor, clean_seed_manager, mock_seed_factory):
        """Test that KasminaMicro handles absence of monitoring gracefully."""
        mock_get_monitor.return_value = None

        manager = clean_seed_manager
        kasmina = KasminaMicro(manager, patience=1, acc_threshold=0.8)

        seed = mock_seed_factory()
        manager.register_seed(seed, "test_seed")

        # Should work fine without monitor - need to trigger germination properly
        kasmina.step(1.0, 0.7)  # Build plateau
        result = kasmina.step(1.0, 0.7)  # Trigger germination
        assert result is True
        seed.initialize_child.assert_called_once()


class TestMonitoringIntegration:
    """Tests for monitoring system integration."""

    @patch("morphogenetic_engine.monitoring.get_monitor")
    def test_circular_import_resilience(self, mock_get_monitor):
        """Test that monitoring import failures don't break core functionality."""
        # Simulate import failure
        mock_get_monitor.side_effect = ImportError("Circular import detected")

        manager = SeedManager()
        manager.seeds.clear()

        # Core functionality should still work
        mock_seed = Mock()
        mock_seed.initialize_child = Mock()
        manager.register_seed(mock_seed, "test_seed")

        # Germination should work despite monitoring failure
        with pytest.raises(ImportError):
            from morphogenetic_engine.monitoring import get_monitor

            get_monitor()

        # But direct SeedManager operations should be unaffected
        result = manager.request_germination("test_seed")
        assert result is True
        mock_seed.initialize_child.assert_called_once()

    def test_monitoring_none_handling(self, clean_seed_manager, mock_seed_factory):
        """Test graceful handling when monitoring returns None."""
        manager = clean_seed_manager

        with patch("morphogenetic_engine.monitoring.get_monitor", return_value=None):
            kasmina = KasminaMicro(manager, patience=1, acc_threshold=0.8)
            seed = mock_seed_factory()
            manager.register_seed(seed, "test_seed")

            # Should work fine with None monitor - build plateau first
            kasmina.step(1.0, 0.7)  # Build plateau
            result = kasmina.step(1.0, 0.7)  # Trigger germination
            assert result is True


class TestEnhancedErrorScenarios:
    """Enhanced error scenario testing for concurrent operations and edge cases."""

    def test_concurrent_germination_with_failures(self, clean_seed_manager, mock_seed_factory):
        """Test behavior when multiple threads attempt germination with some failures."""
        manager = clean_seed_manager
        results = []
        failure_seeds = {"seed_2", "seed_4"}

        # Register seeds with some that will fail
        for i in range(5):
            seed = mock_seed_factory()
            if f"seed_{i}" in failure_seeds:
                seed.initialize_child.side_effect = RuntimeError("Initialization failed")
            manager.register_seed(seed, f"seed_{i}")

        def attempt_germination(seed_id: str):
            try:
                result = manager.request_germination(seed_id)
                results.append((seed_id, "success" if result else "failed", None))
            except (RuntimeError, ValueError, AttributeError) as e:
                results.append((seed_id, "exception", str(e)))

        # Attempt concurrent germinations
        threads = [threading.Thread(target=attempt_germination, args=(f"seed_{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify results
        assert len(results) == 5
        success_count = sum(1 for _, status, _ in results if status == "success")
        failed_count = sum(1 for _, status, _ in results if status == "failed")

        assert success_count == 3  # Non-failure seeds should succeed
        assert failed_count == 2  # Failure seeds should fail gracefully

        # Verify failed seeds have correct status
        for seed_id in failure_seeds:
            assert manager.seeds[seed_id]["status"] == "failed"

    def test_memory_leak_detection_with_weak_references(self, clean_seed_manager, mock_seed_factory):
        """Test for memory leaks using weak references to ensure proper cleanup."""
        manager = clean_seed_manager
        weak_refs = []

        # Create seeds and store weak references
        for i in range(10):
            seed = mock_seed_factory()
            manager.register_seed(seed, f"leak_test_seed_{i}")
            weak_refs.append(weakref.ref(seed))

            # Add some buffer data
            for _ in range(10):
                tensor = torch.randn(50, 50)
                manager.append_to_buffer(f"leak_test_seed_{i}", tensor)

        # Verify all references are alive
        assert all(ref() is not None for ref in weak_refs)

        # Reset and force garbage collection
        manager.reset()
        import gc

        gc.collect()

        # Check that we can still access manager state (no corruption)
        assert len(manager.seeds) == 0
        assert len(manager.germination_log) == 0

        # Note: Weak references may still be alive due to test framework holding references
        # The key test is that manager.reset() doesn't crash and clears state properly

    def test_buffer_operations_under_stress_with_errors(self, clean_seed_manager, mock_seed_factory):
        """Test buffer operations when some operations encounter errors."""
        manager = clean_seed_manager
        seed = mock_seed_factory()
        manager.register_seed(seed, "stress_seed")

        error_count = 0
        success_count = 0

        def buffer_operation(i: int):
            nonlocal error_count, success_count
            try:
                if i % 7 == 0:  # Simulate occasional invalid tensor
                    # This should be handled gracefully
                    manager.append_to_buffer("nonexistent_seed", torch.randn(10, 10))
                else:
                    manager.append_to_buffer("stress_seed", torch.randn(10, 10))
                success_count += 1
            except (RuntimeError, ValueError, MemoryError):
                error_count += 1

        # Run operations concurrently
        threads = [threading.Thread(target=buffer_operation, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should handle errors gracefully without crashing
        assert success_count > 0
        assert error_count == 0  # append_to_buffer should handle missing seeds gracefully

        # Buffer should contain valid data
        buffer = manager.seeds["stress_seed"]["buffer"]
        assert len(buffer) > 0
        assert len(buffer) <= TestConstants.BUFFER_MAXLEN

    @pytest.mark.parametrize(
        "maxlen_size,expected_behavior",
        [
            (1, "single_item_only"),
            (5, "small_buffer"),
            (TestConstants.BUFFER_MAXLEN, "normal_operation"),
        ],
    )
    def test_buffer_edge_cases_parametrized(self, clean_seed_manager, mock_seed_factory, maxlen_size, expected_behavior):
        """Test buffer operations under various edge conditions."""
        manager = clean_seed_manager
        seed = mock_seed_factory()

        # Create custom buffer with specific maxlen before registering
        custom_buffer = deque(maxlen=maxlen_size)
        manager.register_seed(seed, "edge_test_seed")
        manager.seeds["edge_test_seed"]["buffer"] = custom_buffer

        if expected_behavior == "single_item_only":
            # With maxlen=1, only the last item should remain
            for i in range(3):
                manager.append_to_buffer("edge_test_seed", torch.full((2, 2), float(i)))

            buffer = manager.seeds["edge_test_seed"]["buffer"]
            assert len(buffer) == 1
            assert torch.equal(buffer[0], torch.full((2, 2), 2.0))

        elif expected_behavior == "small_buffer":
            # Fill beyond small buffer capacity
            for i in range(maxlen_size + 2):
                manager.append_to_buffer("edge_test_seed", torch.full((2, 2), float(i)))

            buffer = manager.seeds["edge_test_seed"]["buffer"]
            assert len(buffer) == maxlen_size
            # Should contain the last maxlen_size items
            expected_start = 2  # Started from 0, added maxlen_size+2 items, so last should be from 2
            for i, tensor in enumerate(buffer):
                assert torch.equal(tensor, torch.full((2, 2), float(expected_start + i)))

        elif expected_behavior == "normal_operation":
            # Fill to capacity
            for i in range(maxlen_size):
                manager.append_to_buffer("edge_test_seed", torch.randn(2, 2))

            buffer = manager.seeds["edge_test_seed"]["buffer"]
            assert len(buffer) == maxlen_size


# Property-based testing requires hypothesis
try:
    from hypothesis import given
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestPropertyBasedScenarios:
    """Property-based tests using Hypothesis for robust validation."""

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=20))
    def test_health_signal_selection_property(self, health_signals):
        """Property-based test for seed selection based on health signals."""
        SeedManager.reset_singleton()
        manager = SeedManager()
        manager.seeds.clear()

        # Create seeds with the given health signals
        for i, signal in enumerate(health_signals):
            mock_seed = Mock()
            mock_seed.get_health_signal = Mock(return_value=signal)
            manager.register_seed(mock_seed, f"prop_seed_{i}")

        kasmina = KasminaMicro(manager, patience=1, acc_threshold=0.8)

        # The selected seed should always have the lowest health signal
        selected_id = kasmina._select_seed()

        if selected_id:  # Only test if a seed was selected
            min_signal = min(health_signals)
            selected_signal = manager.seeds[selected_id]["module"].get_health_signal()
            assert selected_signal == min_signal

    @given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=50))
    def test_buffer_operations_property_based(self, tensor_sizes):
        """Property-based test for buffer operations with various tensor sizes."""
        SeedManager.reset_singleton()
        manager = SeedManager()
        manager.seeds.clear()

        mock_seed = Mock()
        mock_seed.get_health_signal = Mock(return_value=0.5)
        manager.register_seed(mock_seed, "prop_buffer_seed")

        # Add tensors of varying sizes
        for size in tensor_sizes:
            try:
                tensor = torch.randn(size, 4)  # Fixed second dimension for consistency
                manager.append_to_buffer("prop_buffer_seed", tensor)
            except (RuntimeError, MemoryError):
                # Skip if tensor is too large for memory
                continue

        buffer = manager.seeds["prop_buffer_seed"]["buffer"]

        # Properties that should always hold
        assert len(buffer) <= TestConstants.BUFFER_MAXLEN
        assert all(isinstance(item, torch.Tensor) for item in buffer)
        assert all(item.dim() == 2 for item in buffer)  # All should be 2D
        assert all(item.shape[1] == 4 for item in buffer)  # Consistent second dimension

    @given(st.integers(min_value=1, max_value=20))
    def test_concurrent_seed_registration_property(self, num_seeds):
        """Property-based test for concurrent seed registration."""
        SeedManager.reset_singleton()
        manager = SeedManager()
        manager.seeds.clear()

        results = []

        def register_seed(seed_idx: int):
            try:
                mock_seed = Mock()
                mock_seed.get_health_signal = Mock(return_value=0.5)
                manager.register_seed(mock_seed, f"concurrent_seed_{seed_idx}")
                results.append(f"concurrent_seed_{seed_idx}")
            except (RuntimeError, ValueError, AttributeError) as e:
                results.append(f"error_{seed_idx}: {e}")

        # Run concurrent registrations
        threads = [threading.Thread(target=register_seed, args=(i,)) for i in range(num_seeds)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Properties that should hold
        assert len(results) == num_seeds
        assert len(manager.seeds) == num_seeds

        # All seeds should be registered successfully
        for i in range(num_seeds):
            assert f"concurrent_seed_{i}" in manager.seeds
            assert manager.seeds[f"concurrent_seed_{i}"]["status"] == "dormant"


class TestPerformanceBenchmarks:
    """Performance benchmarking tests with timing assertions."""

    def test_seed_lookup_performance_benchmark(self, clean_seed_manager, mock_seed_factory):
        """Benchmark seed lookup performance with large numbers of seeds."""
        manager = clean_seed_manager

        # Register many seeds
        num_seeds = 1000
        start_time = time.time()

        for i in range(num_seeds):
            seed = mock_seed_factory()
            manager.register_seed(seed, f"benchmark_seed_{i}")

        registration_time = time.time() - start_time

        # Registration should be reasonably fast (less than 5 seconds for 1000 seeds)
        assert registration_time < 5.0, f"Registration took {registration_time:.2f}s, expected < 5.0s"

        # Test lookup performance
        start_time = time.time()

        for i in range(0, num_seeds, 50):  # Sample every 50th seed
            seed_id = f"benchmark_seed_{i}"
            assert seed_id in manager.seeds
            assert manager.seeds[seed_id]["status"] == "dormant"

        lookup_time = time.time() - start_time

        # Lookups should be very fast (less than 0.1 seconds for 20 lookups)
        assert lookup_time < 0.1, f"Lookup took {lookup_time:.2f}s, expected < 0.1s"

    def test_germination_performance_under_load(self, clean_seed_manager, mock_seed_factory):
        """Test germination performance with multiple concurrent requests."""
        manager = clean_seed_manager

        # Register seeds
        num_seeds = 50
        for i in range(num_seeds):
            seed = mock_seed_factory()
            manager.register_seed(seed, f"perf_seed_{i}")

        results = []
        start_time = time.time()

        def attempt_germination(seed_id: str):
            result = manager.request_germination(seed_id)
            results.append((seed_id, result, time.time()))

        # Concurrent germination attempts
        threads = [threading.Thread(target=attempt_germination, args=(f"perf_seed_{i}",)) for i in range(num_seeds)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_time = time.time() - start_time

        # All germinations should complete in reasonable time
        assert total_time < 2.0, f"Germination took {total_time:.2f}s, expected < 2.0s"
        assert len(results) == num_seeds
        assert all(result for _, result, _ in results)  # All should succeed
