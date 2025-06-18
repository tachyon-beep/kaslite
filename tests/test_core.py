"""
Comprehensive tests for the core module.

This module provides thorough testing coverage for the core morphogenetic engine
components including SeedManager (singleton seed orchestrator) and KasminaMicro
(germination controller). Tests are organized by component and follow modern
Python 3.12+ practices with proper isolation and integration testing.

Test Categories:
- Unit tests: Isolated component behavior with mocking
- Integration tests: Component collaboration scenarios
- Thread safety: Concurrent operation validation
- Edge cases: Boundary conditions and error scenarios
"""

# pylint: disable=protected-access,redefined-outer-name

import threading
import time
from collections import deque
from unittest.mock import Mock, patch

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from morphogenetic_engine.core import KasminaMicro, SeedManager


class TestConstants:
    """Constants for test values to avoid magic numbers."""

    LOW_HEALTH_SIGNAL = 0.1
    MEDIUM_HEALTH_SIGNAL = 0.5
    HIGH_HEALTH_SIGNAL = 0.9
    HIGH_ACCURACY = 0.95
    LOW_ACCURACY = 0.5
    PLATEAU_THRESHOLD = 1e-3
    SMALL_LOSS_DELTA = 1e-4
    BUFFER_MAXLEN = 500
    PATIENCE_SHORT = 2
    PATIENCE_MEDIUM = 5
    PATIENCE_LONG = 15


@pytest.fixture
def clean_seed_manager():
    """Provide a clean SeedManager instance for testing."""
    # Reset singleton first
    SeedManager.reset_singleton()
    manager = SeedManager()
    manager.seeds.clear()
    manager.germination_log.clear()
    yield manager
    # Cleanup after test
    SeedManager.reset_singleton()


@pytest.fixture
def mock_seed_factory():
    """Factory for creating mock seeds with customizable health signals."""

    def _create_mock_seed(health_signal: float = TestConstants.MEDIUM_HEALTH_SIGNAL) -> Mock:
        mock_seed = Mock()
        mock_seed.get_health_signal = Mock(return_value=health_signal)
        mock_seed.initialize_child = Mock()
        return mock_seed

    return _create_mock_seed


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    logger = Mock()
    logger.log_germination = Mock()
    logger.log_seed_event = Mock()
    return logger


@pytest.fixture
def sample_tensor():
    """Provide a sample tensor for buffer testing."""
    return torch.randn(4, 8)


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


class TestKasminaMicro:
    """Test suite for KasminaMicro germination controller."""

    def test_initialization(self, clean_seed_manager) -> None:
        """Test KasminaMicro initialization."""
        manager = clean_seed_manager
        km = KasminaMicro(
            manager,
            patience=TestConstants.PATIENCE_LONG,
            delta=TestConstants.PLATEAU_THRESHOLD,
            acc_threshold=TestConstants.HIGH_ACCURACY,
        )

        assert km.seed_manager is manager
        assert km.patience == TestConstants.PATIENCE_LONG
        assert km.delta == pytest.approx(TestConstants.PLATEAU_THRESHOLD)
        assert km.acc_threshold == pytest.approx(TestConstants.HIGH_ACCURACY)
        assert km.plateau == 0
        assert km.prev_loss == float("inf")

    def test_step_loss_improvement_resets_plateau(self, clean_seed_manager) -> None:
        """Test step with improving loss resets plateau counter."""
        manager = clean_seed_manager
        km = KasminaMicro(
            manager, patience=TestConstants.PATIENCE_SHORT, delta=TestConstants.PLATEAU_THRESHOLD
        )

        # First step should not trigger germination
        result = km.step(1.0, TestConstants.LOW_ACCURACY)
        assert result is False
        assert km.plateau == 0
        assert km.prev_loss == pytest.approx(1.0)

        # Small improvement should reset plateau
        result = km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
        assert result is False
        assert km.plateau == 1  # No improvement, plateau increases

        # Significant improvement should reset plateau
        result = km.step(0.8, TestConstants.LOW_ACCURACY)
        assert result is False
        assert km.plateau == 0  # Reset due to improvement
        assert km.prev_loss == pytest.approx(0.8)

    def test_plateau_detection_high_accuracy_blocks_germination(self, clean_seed_manager) -> None:
        """Test plateau detection with high accuracy blocks germination."""
        manager = clean_seed_manager
        km = KasminaMicro(
            manager,
            patience=TestConstants.PATIENCE_SHORT,
            delta=TestConstants.PLATEAU_THRESHOLD,
            acc_threshold=TestConstants.HIGH_ACCURACY,
        )

        # Set initial loss
        km.step(1.0, TestConstants.HIGH_ACCURACY)

        # Create plateau with high accuracy (shouldn't germinate)
        for _ in range(TestConstants.PATIENCE_SHORT + 2):
            result = km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, TestConstants.HIGH_ACCURACY)
            assert result is False  # No germination due to high accuracy

    def test_germination_triggered_by_plateau_and_low_accuracy(
        self, clean_seed_manager, mock_seed_factory
    ) -> None:
        """Test germination triggering under correct conditions."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)
        manager.register_seed(mock_seed, "test_seed")

        km = KasminaMicro(
            manager,
            patience=TestConstants.PATIENCE_SHORT,
            delta=TestConstants.PLATEAU_THRESHOLD,
            acc_threshold=TestConstants.HIGH_ACCURACY,
        )

        # Set initial loss with low accuracy
        result1 = km.step(1.0, TestConstants.LOW_ACCURACY)
        assert result1 is False
        assert km.plateau == 0

        # Create plateau with low accuracy
        result2 = km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
        assert result2 is False
        assert km.plateau == 1

        with patch.object(manager, "request_germination", return_value=True) as mock_germ:
            result3 = km.step(1.0 + 2 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
            assert result3 is True  # Should trigger germination
            mock_germ.assert_called_once_with("test_seed")

    def test_germination_resets_plateau_counter(
        self, clean_seed_manager, mock_seed_factory
    ) -> None:
        """Test that successful germination resets plateau counter."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)
        manager.register_seed(mock_seed, "test_seed")

        km = KasminaMicro(
            manager, patience=TestConstants.PATIENCE_SHORT, delta=TestConstants.PLATEAU_THRESHOLD
        )

        # Build up plateau
        km.step(1.0, TestConstants.LOW_ACCURACY)
        km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
        assert km.plateau == 1

        # Trigger germination
        with patch.object(manager, "request_germination", return_value=True):
            result = km.step(1.0 + 2 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
            assert result is True
            assert km.plateau == 0  # Reset after germination

    def test_select_seed_no_dormant_seeds(self, clean_seed_manager) -> None:
        """Test seed selection when no dormant seeds available."""
        manager = clean_seed_manager
        km = KasminaMicro(manager)

        result = km._select_seed()
        assert result is None

    def test_select_seed_chooses_worst_health(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test seed selection chooses seed with worst health signal."""
        manager = clean_seed_manager

        # Create mock seeds with different health signals
        mock_seed1 = mock_seed_factory(health_signal=TestConstants.MEDIUM_HEALTH_SIGNAL)
        mock_seed2 = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)  # Worst
        mock_seed3 = mock_seed_factory(health_signal=0.3)

        manager.register_seed(mock_seed1, "seed1")
        manager.register_seed(mock_seed2, "seed2")
        manager.register_seed(mock_seed3, "seed3")

        km = KasminaMicro(manager)
        result = km._select_seed()

        assert result == "seed2"  # Should select seed with lowest health signal

    def test_select_seed_ignores_non_dormant(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test seed selection ignores non-dormant seeds."""
        manager = clean_seed_manager

        mock_seed1 = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)
        mock_seed2 = mock_seed_factory(health_signal=0.2)

        manager.register_seed(mock_seed1, "seed1")
        manager.register_seed(mock_seed2, "seed2")

        # Make seed1 non-dormant (should be ignored despite better health signal)
        manager.seeds["seed1"]["status"] = "active"

        km = KasminaMicro(manager)
        result = km._select_seed()

        assert result == "seed2"  # Should select the dormant seed

    @given(accuracy=st.floats(min_value=0.85, max_value=1.0))
    @pytest.mark.property
    def test_accuracy_threshold_boundary_conditions(self, accuracy: float) -> None:
        """Test accuracy threshold boundary conditions with property-based testing."""
        # Create manager directly to avoid fixture issues with Hypothesis
        SeedManager.reset_singleton()
        manager = SeedManager()
        manager.seeds.clear()
        km = KasminaMicro(manager, patience=1, acc_threshold=0.8)

        # High accuracy should prevent germination regardless of plateau
        km.step(1.0, accuracy)
        result = km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, accuracy)

        if accuracy >= 0.8:
            assert result is False  # No germination due to high accuracy
        # Note: For accuracy < 0.8, we'd need seeds registered to test germination

        # Cleanup
        SeedManager.reset_singleton()

    def test_monitoring_integration(self, clean_seed_manager) -> None:
        """Test integration with monitoring system."""
        manager = clean_seed_manager
        km = KasminaMicro(manager, patience=TestConstants.PATIENCE_SHORT)

        # Mock the monitoring module's get_monitor function
        with patch("morphogenetic_engine.monitoring.get_monitor") as mock_get_monitor:
            mock_monitor = Mock()
            mock_get_monitor.return_value = mock_monitor

            # Test that monitoring methods are called
            km.step(1.0, TestConstants.LOW_ACCURACY)

            mock_monitor.update_kasmina_metrics.assert_called_once_with(
                0, TestConstants.PATIENCE_SHORT
            )

            # Test germination recording
            mock_seed = Mock()
            mock_seed.get_health_signal = Mock(return_value=TestConstants.LOW_HEALTH_SIGNAL)
            manager.register_seed(mock_seed, "test_seed")

            # Build up plateau and trigger germination
            km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
            km.step(1.0 + 2 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)

            mock_monitor.record_germination.assert_called_once()


class TestSeedManagerKasminaMicroIntegration:
    """Integration tests between SeedManager and KasminaMicro."""

    def test_end_to_end_germination_workflow(self, mock_seed_factory, mock_logger) -> None:
        """Test complete workflow from plateau detection to successful germination."""
        manager = SeedManager(logger=mock_logger)
        manager.seeds.clear()

        # Register seeds with different health signals
        good_seed = mock_seed_factory(health_signal=TestConstants.HIGH_HEALTH_SIGNAL)
        bad_seed = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)

        manager.register_seed(good_seed, "good_seed")
        manager.register_seed(bad_seed, "bad_seed")

        km = KasminaMicro(
            manager,
            patience=TestConstants.PATIENCE_SHORT,
            delta=TestConstants.PLATEAU_THRESHOLD,
            acc_threshold=TestConstants.HIGH_ACCURACY,
        )

        # Simulate training with plateau and low accuracy
        assert km.step(1.0, TestConstants.LOW_ACCURACY) is False
        assert km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY) is False
        assert km.step(1.0 + 2 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY) is True

        # Verify worst seed was selected and germinated
        assert manager.seeds["bad_seed"]["status"] == "active"
        assert manager.seeds["good_seed"]["status"] == "dormant"

        # Verify logging
        mock_logger.log_germination.assert_called_once_with(0, "bad_seed")
        bad_seed.initialize_child.assert_called_once()

    def test_multiple_germination_attempts_different_seeds(
        self, clean_seed_manager, mock_seed_factory
    ) -> None:
        """Test multiple germination events select different seeds appropriately."""
        manager = clean_seed_manager

        # Register multiple seeds with ascending health signals
        seeds = []
        for i in range(3):
            seed = mock_seed_factory(health_signal=0.1 + i * 0.2)  # 0.1, 0.3, 0.5
            seeds.append(seed)
            manager.register_seed(seed, f"seed_{i}")

        km = KasminaMicro(manager, patience=2, delta=TestConstants.PLATEAU_THRESHOLD)

        # First germination - 2 steps to reach plateau=2 and trigger
        base_loss = 1.0
        km.step(base_loss, TestConstants.LOW_ACCURACY)
        km.step(base_loss + TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
        result1 = km.step(
            base_loss + 2 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY
        )
        assert result1 is True
        assert manager.seeds["seed_0"]["status"] == "active"

        # Second germination - Step 4 builds plateau, Step 5 triggers
        km.step(base_loss + 3 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
        result2 = km.step(
            base_loss + 4 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY
        )
        assert result2 is True
        assert manager.seeds["seed_1"]["status"] == "active"

        # Third germination - Step 6 builds plateau, Step 7 triggers
        km.step(base_loss + 5 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
        result3 = km.step(
            base_loss + 6 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY
        )
        assert result3 is True
        assert manager.seeds["seed_2"]["status"] == "active"

        # Fourth attempt should fail (no dormant seeds left)
        km.step(base_loss + 7 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
        result4 = km.step(
            base_loss + 8 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY
        )
        assert result4 is False

    def test_germination_with_initialization_failures(
        self, clean_seed_manager, mock_seed_factory
    ) -> None:
        """Test handling of germination when seed initialization fails."""
        manager = clean_seed_manager

        # Create seed that will fail initialization
        failing_seed = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)
        failing_seed.initialize_child = Mock(side_effect=RuntimeError("Initialization failed"))

        # Create backup seed
        working_seed = mock_seed_factory(health_signal=0.2)

        manager.register_seed(failing_seed, "failing_seed")
        manager.register_seed(working_seed, "working_seed")

        km = KasminaMicro(manager, patience=1, delta=TestConstants.PLATEAU_THRESHOLD)

        # First germination should fail but not crash
        km.step(1.0, TestConstants.LOW_ACCURACY)
        result1 = km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
        assert result1 is False
        assert manager.seeds["failing_seed"]["status"] == "failed"

        # Second germination should succeed immediately with backup seed
        # Since failed germination doesn't reset plateau, we can try again right away
        result2 = km.step(1.0 + 2 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
        assert result2 is True
        assert manager.seeds["working_seed"]["status"] == "active"


# =============================================================================
# Advanced Testing Patterns
# =============================================================================


class TestPerformanceBenchmarking:
    """Performance benchmarking tests to catch regressions and validate scalability."""

    def test_seed_manager_scales_linearly(self, mock_seed_factory) -> None:
        """Verify SeedManager performance scales acceptably with seed count."""
        SeedManager.reset_singleton()
        seed_counts = [10, 100, 500]  # Reasonable sizes for testing
        times = []

        for count in seed_counts:
            SeedManager.reset_singleton()
            manager = SeedManager()
            seeds = [mock_seed_factory() for _ in range(count)]

            start = time.time()
            for i, seed in enumerate(seeds):
                manager.register_seed(seed, f"seed_{i}")
            # Trigger some germination requests
            for i in range(min(10, count)):
                manager.request_germination(f"seed_{i}")
            times.append(time.time() - start)

        # Verify roughly linear scaling (not exponential)
        # Allow for some overhead but ensure it's not wildly non-linear
        if len(times) >= 2:
            # 100 seeds should be less than 20x slower than 10 seeds
            assert times[1] < times[0] * 20, f"Scaling issue: {times[0]:.4f}s -> {times[1]:.4f}s"
        if len(times) >= 3:
            # 500 seeds should be less than 10x slower than 100 seeds
            assert times[2] < times[1] * 10, f"Scaling issue: {times[1]:.4f}s -> {times[2]:.4f}s"

    def test_concurrent_germination_performance(self, mock_seed_factory) -> None:
        """Test germination under concurrent load without deadlocks."""
        SeedManager.reset_singleton()
        manager = SeedManager()
        errors = []

        def add_and_germinate(thread_id: int):
            """Worker function for concurrent testing."""
            try:
                for i in range(20):  # Reduced for faster testing
                    seed = mock_seed_factory()
                    manager.register_seed(seed, f"thread_{thread_id}_seed_{i}")
                    manager.request_germination(f"thread_{thread_id}_seed_{i}")
            except (RuntimeError, ValueError, AttributeError) as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Test with moderate concurrency
        threads = [threading.Thread(target=add_and_germinate, args=(i,)) for i in range(3)]
        start = time.time()

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        duration = time.time() - start

        assert not errors, f"Concurrent errors: {errors}"
        assert duration < 10.0, f"Concurrent germination too slow: {duration}s"
        # Verify no corruption of shared state
        assert len(manager.seeds) > 0
        assert all(isinstance(seed_id, str) for seed_id in manager.seeds.keys())

    def test_plateau_detection_performance(self, mock_seed_factory) -> None:
        """Test that plateau detection remains performant with many steps."""
        SeedManager.reset_singleton()
        manager = SeedManager()
        seed = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)
        manager.register_seed(seed, "test_seed")

        km = KasminaMicro(
            manager, patience=TestConstants.PATIENCE_MEDIUM, delta=TestConstants.PLATEAU_THRESHOLD
        )

        # Simulate many training steps
        start = time.time()
        for i in range(1000):
            # Slowly decreasing loss to avoid plateau
            loss = 1.0 - (i * 0.0001)
            km.step(loss, TestConstants.MEDIUM_HEALTH_SIGNAL)

        duration = time.time() - start
        assert duration < 1.0, f"Plateau detection too slow for 1000 steps: {duration}s"


class TestSeedContract:
    """Contract testing to verify seed implementations conform to expected interface."""

    def test_mock_seed_interface_compliance(self, mock_seed_factory) -> None:
        """Test that mock seeds implement required interface correctly."""
        seed = mock_seed_factory()

        # Verify required methods exist and are callable
        assert hasattr(seed, "get_health_signal")
        assert hasattr(seed, "initialize_child")
        assert callable(seed.get_health_signal)
        assert callable(seed.initialize_child)

        # Verify method signatures and return types
        health = seed.get_health_signal()
        assert isinstance(
            health, (int, float)
        ), f"Health signal should be numeric, got {type(health)}"
        assert 0.0 <= health <= 1.0, f"Health signal should be in [0,1], got {health}"

        # Verify behavior contracts
        seed.initialize_child()  # Should not raise

        # Verify initialize_child can be called multiple times
        seed.initialize_child()
        seed.initialize_child()

    def test_seed_health_signal_consistency(self, mock_seed_factory) -> None:
        """Test that health signal is consistent across multiple calls."""
        seed = mock_seed_factory(health_signal=TestConstants.HIGH_HEALTH_SIGNAL)

        # Health signal should be stable
        health1 = seed.get_health_signal()
        health2 = seed.get_health_signal()
        health3 = seed.get_health_signal()

        assert health1 == health2 == health3, "Health signal should be consistent"
        assert health1 == TestConstants.HIGH_HEALTH_SIGNAL

    def test_seed_initialization_contract(self, mock_seed_factory) -> None:
        """Test that seed initialization follows expected contract."""
        seed = mock_seed_factory()

        # Should be able to call initialize_child without arguments
        try:
            seed.initialize_child()
        except TypeError as e:
            if "required positional argument" in str(e):
                pytest.fail("initialize_child should not require arguments")
            raise

        # Should be idempotent (safe to call multiple times)
        seed.initialize_child()
        seed.initialize_child()

        # Should track that it was called
        assert seed.initialize_child.called, "Mock should track that initialize_child was called"


class TestAdvancedPropertyBased:
    """Enhanced property-based testing for edge cases and algorithmic correctness."""

    def _create_mock_seed(self, health_signal: float = TestConstants.MEDIUM_HEALTH_SIGNAL) -> Mock:
        """Create a mock seed with customizable health signal."""
        mock_seed = Mock()
        mock_seed.get_health_signal = Mock(return_value=health_signal)
        mock_seed.initialize_child = Mock()
        return mock_seed

    @given(capacity=st.integers(min_value=1, max_value=100))
    def test_seed_manager_capacity_invariant(self, capacity) -> None:
        """SeedManager should handle various capacities gracefully."""
        SeedManager.reset_singleton()
        manager = SeedManager()

        # Try to add more seeds than might be reasonable
        for i in range(min(capacity * 2, 50)):  # Cap at 50 for test performance
            seed = self._create_mock_seed()
            manager.register_seed(seed, f"seed_{i}")

        # Should not crash or corrupt state
        assert isinstance(manager.seeds, dict)
        assert len(manager.seeds) > 0  # At least some seeds should be registered

        # All registered seeds should have valid IDs
        for seed_id in manager.seeds.keys():
            assert isinstance(seed_id, str)
            assert len(seed_id) > 0

    @given(losses=st.lists(st.floats(min_value=0.0, max_value=10.0), min_size=1, max_size=20))
    def test_loss_plateau_detection_correctness(self, losses) -> None:
        """Plateau detection should work correctly for any valid loss sequence."""
        # Skip invalid sequences
        if not losses or any(not (0.0 <= loss <= 10.0) for loss in losses):
            return

        SeedManager.reset_singleton()
        manager = SeedManager()
        seed = self._create_mock_seed(health_signal=TestConstants.LOW_HEALTH_SIGNAL)
        manager.register_seed(seed, "test_seed")

        kasmina = KasminaMicro(manager, patience=3, delta=0.1)

        for loss in losses:
            kasmina.step(loss, TestConstants.MEDIUM_HEALTH_SIGNAL)

        # Invariant: plateau counter should never exceed patience + some reasonable buffer
        # (since germination resets it)
        assert kasmina.plateau >= 0, "Plateau counter should never be negative"
        assert isinstance(kasmina.plateau, int), "Plateau counter should be an integer"

    @given(
        patience=st.integers(min_value=1, max_value=20),
        delta=st.floats(min_value=0.001, max_value=1.0),
    )
    def test_patience_parameter_correctness(self, patience, delta) -> None:
        """Patience and delta parameters should work correctly across ranges."""
        SeedManager.reset_singleton()
        manager = SeedManager()
        seed = self._create_mock_seed(health_signal=TestConstants.LOW_HEALTH_SIGNAL)
        manager.register_seed(seed, "test_seed")

        kasmina = KasminaMicro(manager, patience=patience, delta=delta)

        # Verify parameters are set correctly
        assert kasmina.patience == patience
        assert kasmina.delta == delta

        # Should handle edge case of very small delta
        for i in range(patience + 2):
            loss = 1.0 + i * (delta / 2)  # Small increments
            kasmina.step(loss, TestConstants.MEDIUM_HEALTH_SIGNAL)

        # Should not crash with small deltas and various patience values
        assert isinstance(kasmina.plateau, int)
        assert kasmina.plateau >= 0

    @given(num_threads=st.integers(min_value=1, max_value=10))
    def test_concurrent_safety_invariants(self, num_threads) -> None:
        """Test that concurrent operations maintain data structure invariants."""
        SeedManager.reset_singleton()
        manager = SeedManager()
        errors = []

        def worker(worker_id: int):
            """Worker thread function."""
            try:
                for i in range(5):
                    seed = self._create_mock_seed()
                    manager.register_seed(seed, f"worker_{worker_id}_seed_{i}")
                    manager.request_germination(f"worker_{worker_id}_seed_{i}")
            except (RuntimeError, ValueError, AttributeError) as e:
                errors.append(f"Worker {worker_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify no errors and data structure integrity
        assert not errors, f"Concurrent errors: {errors}"
        assert isinstance(manager.seeds, dict)
        assert len(manager.seeds) <= num_threads * 5  # At most this many seeds

        # All seed IDs should be unique and valid
        seed_ids = list(manager.seeds.keys())
        assert len(seed_ids) == len(set(seed_ids)), "Duplicate seed IDs found"
        assert all(isinstance(sid, str) and len(sid) > 0 for sid in seed_ids)


class TestEdgeCaseScenarios:
    """Test edge cases and boundary conditions for robustness."""

    def test_very_large_loss_values(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test handling of very large loss values."""
        manager = clean_seed_manager
        seed = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)
        manager.register_seed(seed, "test_seed")

        km = KasminaMicro(manager, patience=2, delta=TestConstants.PLATEAU_THRESHOLD)

        # Test with very large loss values
        large_losses = [1e6, 1e7, 1e8, 1e9]
        for loss in large_losses:
            result = km.step(loss, TestConstants.MEDIUM_HEALTH_SIGNAL)
            # Should not crash or produce invalid state
            assert isinstance(result, bool)
            assert isinstance(km.prev_loss, (int, float))

    def test_very_small_loss_deltas(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test plateau detection with very small loss changes."""
        manager = clean_seed_manager
        # Use high health signal to prevent germination that would reset plateau
        seed = mock_seed_factory(health_signal=TestConstants.HIGH_HEALTH_SIGNAL)
        manager.register_seed(seed, "test_seed")

        # Use very small delta for sensitive plateau detection
        km = KasminaMicro(manager, patience=3, delta=1e-10)

        base_loss = 1.0
        # Make tiny changes that should trigger plateau - use high accuracy to prevent germination
        for i in range(km.patience + 1):  # Ensure we exceed patience
            loss = base_loss + i * 1e-12  # Extremely small changes
            km.step(loss, TestConstants.HIGH_ACCURACY)  # High accuracy prevents germination

        # Should detect plateau with such small changes
        assert km.plateau >= km.patience

    def test_rapid_seed_registration_and_removal(
        self, clean_seed_manager, mock_seed_factory
    ) -> None:
        """Test rapid registration and germination of seeds."""
        manager = clean_seed_manager

        # Rapidly register and germinate many seeds
        for i in range(100):
            seed = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)
            manager.register_seed(seed, f"rapid_seed_{i}")

            if i % 10 == 0:  # Periodically trigger germination
                manager.request_germination(f"rapid_seed_{i}")

        # Final germination attempts
        for i in range(0, 100, 20):
            manager.request_germination(f"rapid_seed_{i}")

        # Verify state integrity
        assert isinstance(manager.seeds, dict)
        assert isinstance(manager.germination_log, list)

        # Check that some seeds were processed
        active_seeds = [s for s in manager.seeds.values() if s["status"] == "active"]
        dormant_seeds = [s for s in manager.seeds.values() if s["status"] == "dormant"]

        # Should have processed some seeds given the low health signals
        assert len(active_seeds) + len(dormant_seeds) <= 100
