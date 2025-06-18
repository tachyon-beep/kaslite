"""
Comprehensive tests for the SeedManager component.

This module provides thorough testing coverage for the SeedManager singleton class,
focusing on seed orchestration, thread safety, and lifecycle management. Tests follow
modern Python 3.12+ practices with proper isolation and comprehensive scenarios.

Test Categories:
- Unit tests: Isolated SeedManager behavior with mocking
- Thread safety: Concurrent operation validation  
- Edge cases: Boundary conditions and error scenarios
- Integration: Logger and monitoring integration
"""

# pylint: disable=protected-access,redefined-outer-name

import threading
import time
from collections import deque
from unittest.mock import Mock

import pytest
import torch

from morphogenetic_engine.core import SeedManager


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
        assert torch.equal(
            buffer[-1], torch.full((2, 3), float(TestConstants.BUFFER_MAXLEN + 9)).detach()
        )
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

    def test_request_germination_already_active(
        self, clean_seed_manager, mock_seed_factory
    ) -> None:
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

    def test_request_germination_exception_handling(
        self, clean_seed_manager, mock_seed_factory
    ) -> None:
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
        threads = [
            threading.Thread(target=attempt_germination, args=(f"seed_{i}",)) for i in range(5)
        ]
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
        threads = [
            threading.Thread(target=concurrent_operations, args=(i,)) for i in range(10)
        ]
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
