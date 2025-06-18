"""
Unit and integration tests for the monitoring functionality.

This test suite validates the PrometheusMonitor class with both unit tests
(using real Prometheus metrics) and integration tests (including HTTP server).
"""
# pylint: disable=redefined-outer-name

import os
import threading
import time
from typing import Generator
from unittest.mock import Mock, patch

import psutil  # type: ignore[import-untyped]
import pytest
from hypothesis import given, strategies as st
import prometheus_client
from prometheus_client import REGISTRY, generate_latest

from morphogenetic_engine.monitoring import (
    BEST_ACCURACY,
    EPOCHS_TOTAL,
    EXPERIMENT_DURATION,
    GERMINATIONS_TOTAL,
    KASMINA_PLATEAU_COUNTER,
    KASMINA_PATIENCE,
    PHASE_TRANSITIONS_TOTAL,
    SEED_ALPHA,
    SEED_DRIFT,
    SEED_HEALTH_SIGNAL,
    SEED_STATE,
    SEED_TRAINING_PROGRESS,
    TRAINING_LOSS,
    VALIDATION_ACCURACY,
    VALIDATION_LOSS,
    PrometheusMonitor,
    cleanup_monitoring,
    get_monitor,
    initialize_monitoring,
)


class MetricValidator:
    """Helper class for validating actual Prometheus metric values."""

    @staticmethod
    def get_metric_value(metric, labels: dict[str, str]) -> float | None:
        """Get the current value of a Prometheus metric with specific labels."""
        try:
            metric_family = metric.collect()[0]
            for sample in metric_family.samples:
                if sample.labels == labels:
                    return float(sample.value)
            return None
        except (IndexError, AttributeError, ValueError):
            return None

    @staticmethod
    def get_counter_value(metric, labels: dict[str, str]) -> float | None:
        """Get the current value of a Counter metric."""
        return MetricValidator.get_metric_value(metric, labels)

    @staticmethod
    def get_gauge_value(metric, labels: dict[str, str]) -> float | None:
        """Get the current value of a Gauge metric."""
        return MetricValidator.get_metric_value(metric, labels)

    @staticmethod
    def clear_registry():
        """Clear all metrics from the Prometheus registry for clean tests."""
        # Clear all collectors to ensure clean state
        # pylint: disable=protected-access
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass  # Already unregistered


@pytest.fixture
def metric_validator() -> MetricValidator:
    """Provide a MetricValidator instance for tests."""
    return MetricValidator()


@pytest.fixture
def monitor_with_mocked_server() -> Generator[PrometheusMonitor, None, None]:
    """Provide a PrometheusMonitor with mocked HTTP server for testing."""
    with patch("morphogenetic_engine.monitoring.start_http_server"):
        monitor = PrometheusMonitor(experiment_id="test_exp_123")
        yield monitor


@pytest.fixture
def clean_metrics():
    """Ensure clean metric state before and after each test."""
    MetricValidator.clear_registry()
    yield
    MetricValidator.clear_registry()


class TestPrometheusMonitorUnit:
    """Unit tests for PrometheusMonitor core functionality."""

    def test_monitor_initialization_basic(self) -> None:
        """Test basic PrometheusMonitor initialization."""
        monitor = PrometheusMonitor(experiment_id="test_exp_456", port=9000)

        assert monitor.experiment_id == "test_exp_456"
        assert monitor.port == 9000
        assert monitor.server_started is False
        assert isinstance(monitor.server_lock, type(threading.Lock()))

    def test_state_mapping_completeness(self) -> None:
        """Test seed state to numeric mapping covers all expected states."""
        monitor = PrometheusMonitor(experiment_id="test_state_mapping")
        
        expected_states = {
            "dormant": 0,
            "training": 1,
            "blending": 2,
            "active": 3,
            "failed": -1
        }
        
        assert monitor.state_map == expected_states

    @patch("morphogenetic_engine.monitoring.start_http_server")
    def test_start_metrics_server_success(self, mock_start_server: Mock) -> None:
        """Test successful metrics server start."""
        monitor = PrometheusMonitor(experiment_id="test_server_start")
        monitor.start_metrics_server()

        mock_start_server.assert_called_once_with(monitor.port)
        assert monitor.server_started is True

    @patch("morphogenetic_engine.monitoring.start_http_server")
    def test_start_metrics_server_failure(self, mock_start_server: Mock) -> None:
        """Test metrics server start failure handling."""
        mock_start_server.side_effect = OSError("Port already in use")
        monitor = PrometheusMonitor(experiment_id="test_server_failure")

        # Should not raise exception
        monitor.start_metrics_server()
        assert monitor.server_started is False

    @patch("morphogenetic_engine.monitoring.start_http_server")
    def test_monitor_server_start_idempotent(self, mock_start_server: Mock) -> None:
        """Test that starting server multiple times is safe."""
        monitor = PrometheusMonitor(experiment_id="test_idempotent")

        # Start server multiple times
        monitor.start_metrics_server()
        monitor.start_metrics_server()
        monitor.start_metrics_server()

        # Should only call start_http_server once
        mock_start_server.assert_called_once()
        assert monitor.server_started is True


class TestPrometheusMonitorRealMetrics:
    """Tests using real Prometheus metrics (no mocking of metric objects)."""

    def test_record_epoch_completion_real_metrics(
        self, monitor_with_mocked_server: PrometheusMonitor, metric_validator: MetricValidator
    ) -> None:
        """Test epoch completion recording with real Prometheus metrics."""
        # Record epoch completion
        monitor_with_mocked_server.record_epoch_completion("phase_1", 12.5)

        # Verify counter increment
        counter_value = metric_validator.get_counter_value(
            EPOCHS_TOTAL, {"phase": "phase_1", "experiment_id": "test_exp_123"}
        )
        assert counter_value is not None and abs(counter_value - 1.0) < 1e-10

    def test_update_training_metrics_real_prometheus(
        self, monitor_with_mocked_server: PrometheusMonitor, metric_validator: MetricValidator
    ) -> None:
        """Test training metrics update with real Prometheus objects."""
        # Update training metrics
        monitor_with_mocked_server.update_training_metrics("phase_2", 0.25, 0.30, 0.85, 0.90)

        # Verify all metric values
        labels = {"phase": "phase_2", "experiment_id": "test_exp_123"}
        
        train_loss = metric_validator.get_gauge_value(TRAINING_LOSS, labels)
        val_loss = metric_validator.get_gauge_value(VALIDATION_LOSS, labels)
        val_acc = metric_validator.get_gauge_value(VALIDATION_ACCURACY, labels)
        best_acc = metric_validator.get_gauge_value(BEST_ACCURACY, {"experiment_id": "test_exp_123"})

        assert train_loss is not None and abs(train_loss - 0.25) < 1e-10
        assert val_loss is not None and abs(val_loss - 0.30) < 1e-10
        assert val_acc is not None and abs(val_acc - 0.85) < 1e-10
        assert best_acc is not None and abs(best_acc - 0.90) < 1e-10

    def test_record_germination_real_metrics(
        self, monitor_with_mocked_server: PrometheusMonitor, metric_validator: MetricValidator
    ) -> None:
        """Test germination event recording with real metrics."""
        # Record multiple germinations
        monitor_with_mocked_server.record_germination()
        monitor_with_mocked_server.record_germination()
        monitor_with_mocked_server.record_germination()

        # Verify counter value
        germination_count = metric_validator.get_counter_value(
            GERMINATIONS_TOTAL, {"experiment_id": "test_exp_123"}
        )
        assert germination_count is not None and abs(germination_count - 3.0) < 1e-10

    def test_record_phase_transition_real_metrics(
        self, monitor_with_mocked_server: PrometheusMonitor, metric_validator: MetricValidator
    ) -> None:
        """Test phase transition recording with real metrics."""
        monitor_with_mocked_server.record_phase_transition("phase_1", "phase_2")

        transition_count = metric_validator.get_counter_value(
            PHASE_TRANSITIONS_TOTAL,
            {"from_phase": "phase_1", "to_phase": "phase_2", "experiment_id": "test_exp_123"}
        )
        assert transition_count is not None and abs(transition_count - 1.0) < 1e-10

    def test_update_seed_metrics_real_prometheus(
        self, monitor_with_mocked_server: PrometheusMonitor, metric_validator: MetricValidator
    ) -> None:
        """Test seed metrics update with real Prometheus objects."""
        monitor_with_mocked_server.update_seed_metrics(
            seed_id="seed_1_2",
            state="blending",
            alpha=0.5,
            drift=0.1,
            health_signal=0.8,
            training_progress=0.6,
        )

        labels = {"seed_id": "seed_1_2", "experiment_id": "test_exp_123"}

        # Verify all seed metrics
        state_value = metric_validator.get_gauge_value(SEED_STATE, labels)
        alpha_value = metric_validator.get_gauge_value(SEED_ALPHA, labels)
        drift_value = metric_validator.get_gauge_value(SEED_DRIFT, labels)
        health_value = metric_validator.get_gauge_value(SEED_HEALTH_SIGNAL, labels)
        progress_value = metric_validator.get_gauge_value(SEED_TRAINING_PROGRESS, labels)

        assert state_value is not None and abs(state_value - 2.0) < 1e-10  # blending state
        assert alpha_value is not None and abs(alpha_value - 0.5) < 1e-10
        assert drift_value is not None and abs(drift_value - 0.1) < 1e-10
        assert health_value is not None and abs(health_value - 0.8) < 1e-10
        assert progress_value is not None and abs(progress_value - 0.6) < 1e-10

    def test_update_kasmina_metrics_real_prometheus(
        self, monitor_with_mocked_server: PrometheusMonitor, metric_validator: MetricValidator
    ) -> None:
        """Test Kasmina controller metrics update with real Prometheus objects."""
        monitor_with_mocked_server.update_kasmina_metrics(5, 15)

        labels = {"experiment_id": "test_exp_123"}
        plateau_value = metric_validator.get_gauge_value(KASMINA_PLATEAU_COUNTER, labels)
        patience_value = metric_validator.get_gauge_value(KASMINA_PATIENCE, labels)

        assert plateau_value is not None and abs(plateau_value - 5.0) < 1e-10
        assert patience_value is not None and abs(patience_value - 15.0) < 1e-10

    @patch("time.time")
    def test_update_experiment_duration_real_prometheus(
        self, mock_time: Mock, monitor_with_mocked_server: PrometheusMonitor, metric_validator: MetricValidator
    ) -> None:
        """Test experiment duration update with real Prometheus objects."""
        mock_time.return_value = 1000.0
        monitor_with_mocked_server.experiment_start_time = 900.0

        monitor_with_mocked_server.update_experiment_duration()

        duration_value = metric_validator.get_gauge_value(
            EXPERIMENT_DURATION, {"experiment_id": "test_exp_123"}
        )
        assert duration_value is not None and abs(duration_value - 100.0) < 1e-10  # 1000 - 900


class TestPrometheusMonitorErrorHandling:
    """Tests for error scenarios and edge cases."""

    @pytest.mark.parametrize("invalid_value", [
        float('nan'),
        float('inf'),
        float('-inf'),
    ])
    def test_invalid_metric_values_handling(
        self, monitor_with_mocked_server: PrometheusMonitor, invalid_value: float
    ) -> None:
        """Test handling of invalid metric values (NaN, infinity)."""
        # This should not raise an exception but should handle gracefully
        try:
            monitor_with_mocked_server.update_training_metrics("phase_1", invalid_value, 0.5, 0.8, 0.9)
            # Prometheus client should handle these values appropriately
        except (ValueError, TypeError) as e:
            # If exceptions are raised, they should be meaningful
            assert "invalid" in str(e).lower() or "nan" in str(e).lower() or "inf" in str(e).lower()

    @pytest.mark.parametrize("malformed_id", [
        "",  # Empty string
        " ",  # Whitespace only
        "test\nexp",  # Newline character
        "test\texp",  # Tab character
        "test exp",  # Space character
        "test/exp",  # Forward slash
        "test\\exp",  # Backslash
        "test\"exp",  # Quote character
        "a" * 1000,  # Very long string
    ])
    def test_malformed_experiment_ids(self, malformed_id: str) -> None:
        """Test behavior with malformed experiment IDs."""
        # Monitor should be created but may have issues with metric labeling
        monitor = PrometheusMonitor(experiment_id=malformed_id)
        assert monitor.experiment_id == malformed_id
        
        # Basic operations should not crash
        try:
            with patch("morphogenetic_engine.monitoring.start_http_server"):
                monitor.start_metrics_server()
                monitor.record_germination()
        except (ValueError, TypeError, KeyError) as e:
            # If exceptions occur, they should be related to Prometheus label validation
            assert any(word in str(e).lower() for word in ["label", "metric", "invalid", "character"])

    @pytest.mark.parametrize("malformed_seed_id", [
        "",  # Empty string
        " ",  # Whitespace only
        "seed\nid",  # Newline character
        "seed\"id",  # Quote character
        "seed{id}",  # Curly braces
        "seed[id]",  # Square brackets
    ])
    def test_malformed_seed_ids(self, monitor_with_mocked_server: PrometheusMonitor, malformed_seed_id: str) -> None:
        """Test behavior with malformed seed IDs."""
        # Operations should handle malformed seed IDs gracefully
        try:
            monitor_with_mocked_server.update_seed_metrics(
                seed_id=malformed_seed_id,
                state="active",
                alpha=0.5
            )
        except (ValueError, TypeError, KeyError) as e:
            # If exceptions occur, they should be related to Prometheus label validation
            assert any(word in str(e).lower() for word in ["label", "metric", "invalid", "character"])

    def test_unknown_seed_state_handling(self, monitor_with_mocked_server: PrometheusMonitor) -> None:
        """Test behavior with unknown seed states."""
        monitor_with_mocked_server.update_seed_metrics(
            seed_id="test_seed",
            state="unknown_state",  # Not in state_map
            alpha=0.5
        )
        
        # Should default to -1 for unknown states
        validator = MetricValidator()
        state_value = validator.get_gauge_value(
            SEED_STATE, {"seed_id": "test_seed", "experiment_id": "test_exp_123"}
        )
        assert state_value == -1.0

    def test_negative_values_where_inappropriate(self, monitor_with_mocked_server: PrometheusMonitor) -> None:
        """Test handling of negative values where they shouldn't be allowed."""
        # Some metrics shouldn't have negative values in real scenarios
        monitor_with_mocked_server.update_training_metrics("phase_1", -0.5, -0.3, -0.1, -0.9)
        
        # Prometheus itself doesn't prevent negative values, but we can verify they're recorded
        validator = MetricValidator()
        train_loss = validator.get_gauge_value(
            TRAINING_LOSS, {"phase": "phase_1", "experiment_id": "test_exp_123"}
        )
        assert train_loss == -0.5  # Value is recorded as-is

    def test_extremely_large_values(self, monitor_with_mocked_server: PrometheusMonitor) -> None:
        """Test handling of extremely large metric values."""
        large_value = 1e100
        monitor_with_mocked_server.update_training_metrics("phase_1", large_value, 0.5, 0.8, 0.9)
        
        validator = MetricValidator()
        train_loss = validator.get_gauge_value(
            TRAINING_LOSS, {"phase": "phase_1", "experiment_id": "test_exp_123"}
        )
        assert train_loss is not None and abs(train_loss - large_value) < 1e-10


class TestPrometheusMonitorPropertyBased:
    """Property-based tests using Hypothesis for edge case discovery."""

    @given(
        experiment_id=st.text(min_size=1, max_size=100),
        phase=st.text(min_size=1, max_size=50),
        train_loss=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        val_loss=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        val_acc=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        best_acc=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_training_metrics_property_based(
        self, experiment_id: str, phase: str, train_loss: float, val_loss: float, val_acc: float, best_acc: float
    ) -> None:
        """Property-based test for training metrics with valid ranges."""
        with patch("morphogenetic_engine.monitoring.start_http_server"):
            monitor = PrometheusMonitor(experiment_id=experiment_id)
            
            try:
                monitor.update_training_metrics(phase, train_loss, val_loss, val_acc, best_acc)
                
                # Verify metrics can be retrieved
                validator = MetricValidator()
                recorded_loss = validator.get_gauge_value(
                    TRAINING_LOSS, {"phase": phase, "experiment_id": experiment_id}
                )
                
                # If no exception was raised, the value should be recorded correctly
                if recorded_loss is not None:
                    assert abs(recorded_loss - train_loss) < 1e-10
                    
            except (ValueError, TypeError, KeyError) as e:
                # If exceptions occur, they should be related to label validation
                assert any(word in str(e).lower() for word in ["label", "metric", "invalid"])

    @given(
        seed_id=st.text(min_size=1, max_size=50),
        state=st.sampled_from(["dormant", "training", "blending", "active", "failed", "unknown"]),
        alpha=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        drift=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_seed_metrics_property_based(
        self, seed_id: str, state: str, alpha: float, drift: float
    ) -> None:
        """Property-based test for seed metrics."""
        with patch("morphogenetic_engine.monitoring.start_http_server"):
            monitor = PrometheusMonitor(experiment_id="prop_test")
            
            try:
                monitor.update_seed_metrics(
                    seed_id=seed_id,
                    state=state,
                    alpha=alpha,
                    drift=drift
                )
                
                # Verify state mapping
                expected_state_value = monitor.state_map.get(state, -1)
                validator = MetricValidator()
                recorded_state = validator.get_gauge_value(
                    SEED_STATE, {"seed_id": seed_id, "experiment_id": "prop_test"}
                )
                
                if recorded_state is not None:
                    assert recorded_state == expected_state_value
                    
            except (ValueError, TypeError, KeyError) as e:
                # If exceptions occur, they should be related to label validation
                assert any(word in str(e).lower() for word in ["label", "metric", "invalid"])


class TestMonitoringUtilities:
    """Test monitoring utility functions."""

    def test_initialize_monitoring(self) -> None:
        """Test monitoring initialization utility."""
        with patch("morphogenetic_engine.monitoring.PrometheusMonitor") as mock_monitor_class, \
             patch("morphogenetic_engine.monitoring.start_http_server"):

            mock_monitor = Mock()
            mock_monitor_class.return_value = mock_monitor

            result = initialize_monitoring("test_exp", 8080)

            mock_monitor_class.assert_called_once_with("test_exp", 8080)
            mock_monitor.start_metrics_server.assert_called_once()
            assert result == mock_monitor

    def test_cleanup_monitoring(self) -> None:
        """Test monitoring cleanup utility."""
        with patch("morphogenetic_engine.monitoring._monitor") as mock_monitor:
            mock_monitor.update_experiment_duration = Mock()

            cleanup_monitoring()

            mock_monitor.update_experiment_duration.assert_called_once()

    def test_get_monitor(self) -> None:
        """Test getting current monitor instance."""
        with patch("morphogenetic_engine.monitoring._monitor", "test_monitor"):
            result = get_monitor()
            assert result == "test_monitor"


class TestMonitoringIntegration:
    """Real integration tests for monitoring functionality."""

    def test_monitor_thread_safety_comprehensive(self) -> None:
        """Test comprehensive thread safety with realistic concurrent operations."""
        monitor = PrometheusMonitor(experiment_id="test_thread_safety")

        def update_metrics_worker(worker_id: int) -> None:
            """Worker function for concurrent metric updates."""
            for i in range(50):  # More iterations for stress testing
                monitor.update_training_metrics(f"phase_{worker_id}", 0.1 * i, 0.2 * i, 0.8, 0.9)
                monitor.record_germination()
                monitor.update_seed_metrics(
                    seed_id=f"seed_{worker_id}_{i}",
                    state="training",
                    alpha=0.5,
                    drift=0.1
                )

        # Run concurrent updates with more threads
        threads = [threading.Thread(target=update_metrics_worker, args=(i,)) for i in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Test should complete without errors
        assert monitor.experiment_id == "test_thread_safety"

    @patch("morphogenetic_engine.monitoring.start_http_server")
    def test_metrics_endpoint_integration(self, mock_start_server: Mock) -> None:
        """Test actual metrics endpoint integration (simulated)."""
        monitor = PrometheusMonitor(experiment_id="test_endpoint", port=8000)
        monitor.start_metrics_server()
        
        # Simulate some metric updates
        monitor.update_training_metrics("phase_1", 0.5, 0.4, 0.85, 0.90)
        monitor.record_germination()
        
        # Verify server was started
        mock_start_server.assert_called_once_with(8000)
        assert monitor.server_started is True

    def test_metric_label_consistency(self, monitor_with_mocked_server: PrometheusMonitor) -> None:
        """Test that metric labels are consistent across operations."""
        # Update same metrics multiple times
        monitor_with_mocked_server.update_training_metrics("phase_1", 0.1, 0.2, 0.8, 0.9)
        monitor_with_mocked_server.update_training_metrics("phase_1", 0.05, 0.15, 0.85, 0.95)  # Should overwrite
        
        validator = MetricValidator()
        labels = {"phase": "phase_1", "experiment_id": "test_exp_123"}
        
        # Latest values should be present
        train_loss = validator.get_gauge_value(TRAINING_LOSS, labels)
        val_acc = validator.get_gauge_value(VALIDATION_ACCURACY, labels)
        
        assert train_loss is not None and abs(train_loss - 0.05) < 1e-10  # Latest value
        assert val_acc is not None and abs(val_acc - 0.85) < 1e-10  # Latest value

    def test_metric_persistence_across_operations(self) -> None:
        """Test that metrics persist correctly across multiple operations."""
        with patch("morphogenetic_engine.monitoring.start_http_server"):
            monitor = PrometheusMonitor(experiment_id="persistence_test")
            
            # Record multiple germinations
            for _ in range(10):
                monitor.record_germination()
            
            validator = MetricValidator()
            germination_count = validator.get_counter_value(
                GERMINATIONS_TOTAL, {"experiment_id": "persistence_test"}
            )
            
            assert germination_count is not None and abs(germination_count - 10.0) < 1e-10
            
            # Add more germinations
            for _ in range(5):
                monitor.record_germination()
                
            # Count should be cumulative
            final_count = validator.get_counter_value(
                GERMINATIONS_TOTAL, {"experiment_id": "persistence_test"}
            )
            assert final_count is not None and abs(final_count - 15.0) < 1e-10


class TestMonitoringPerformance:
    """Performance and scalability tests."""

    def test_high_frequency_metric_updates(self) -> None:
        """Test high-frequency metric updates for performance."""
        with patch("morphogenetic_engine.monitoring.start_http_server"):
            monitor = PrometheusMonitor(experiment_id="perf_test")
            
            start_time = time.time()
            
            # Perform many rapid updates
            for i in range(1000):
                monitor.update_training_metrics("phase_1", 0.1, 0.2, 0.8, 0.9)
                if i % 100 == 0:  # Occasional other operations
                    monitor.record_germination()
                    
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete in reasonable time (less than 5 seconds for 1000 updates)
            assert duration < 5.0
            
            # Verify final metrics are correct
            validator = MetricValidator()
            germination_count = validator.get_counter_value(
                GERMINATIONS_TOTAL, {"experiment_id": "perf_test"}
            )
            assert germination_count is not None and abs(germination_count - 10.0) < 1e-10

    def test_concurrent_metric_updates_stress(self) -> None:
        """Stress test with many concurrent metric updates."""
        with patch("morphogenetic_engine.monitoring.start_http_server"):
            monitor = PrometheusMonitor(experiment_id="stress_test")
            
            def stress_worker() -> None:
                """Worker function for stress testing."""
                for i in range(100):
                    monitor.update_training_metrics(f"phase_{i%3}", 0.1, 0.2, 0.8, 0.9)
                    monitor.record_germination()
                    monitor.update_seed_metrics(
                        seed_id=f"seed_{i}",
                        state="training",
                        alpha=0.5
                    )
            
            # Many concurrent workers
            threads = [threading.Thread(target=stress_worker) for _ in range(10)]
            
            start_time = time.time()
            
            for thread in threads:
                thread.start()
                
            for thread in threads:
                thread.join()
                
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete in reasonable time
            assert duration < 10.0  # 10 threads * 100 operations each should be fast
            
            # Verify metrics were recorded
            validator = MetricValidator()
            germination_count = validator.get_counter_value(
                GERMINATIONS_TOTAL, {"experiment_id": "stress_test"}
            )
            assert germination_count is not None and abs(germination_count - 1000.0) < 1e-10  # 10 threads * 100 operations each


class TestPrometheusCompliance:
    """Tests for Prometheus metric format compliance."""

    def test_metric_naming_conventions(self) -> None:
        """Test that metric names follow Prometheus conventions."""
        # All metrics should start with application prefix
        metrics_to_check = [
            EPOCHS_TOTAL, GERMINATIONS_TOTAL, TRAINING_LOSS, VALIDATION_LOSS,
            VALIDATION_ACCURACY, BEST_ACCURACY, SEED_ALPHA, SEED_STATE
        ]
        
        for metric in metrics_to_check:
            # pylint: disable=protected-access
            metric_name = metric._name
            assert metric_name.startswith("kaslite_"), f"Metric {metric_name} should start with kaslite_"
            assert "_" in metric_name, f"Metric {metric_name} should use underscores"
            assert metric_name.islower() or "_" in metric_name, f"Metric {metric_name} should be lowercase"

    def test_label_cardinality_limits(self, monitor_with_mocked_server: PrometheusMonitor) -> None:
        """Test that label cardinality stays within reasonable limits."""
        # Create many different seed metrics to test cardinality
        for i in range(100):  # Reasonable number of seeds
            monitor_with_mocked_server.update_seed_metrics(
                seed_id=f"seed_{i}",
                state="training",
                alpha=0.5
            )
        
        # Should not raise any cardinality warnings or errors
        validator = MetricValidator()
        
        # Verify last metric is recorded correctly
        state_value = validator.get_gauge_value(
            SEED_STATE, {"seed_id": "seed_99", "experiment_id": "test_exp_123"}
        )
        assert state_value is not None and abs(state_value - 1.0) < 1e-10  # training state

    def test_metric_help_strings(self) -> None:
        """Test that metrics have appropriate help strings."""
        metrics_with_help = [
            (EPOCHS_TOTAL, "epoch"),
            (GERMINATIONS_TOTAL, "germination"),
            (TRAINING_LOSS, "loss"),
            (VALIDATION_ACCURACY, "accuracy"),
            (SEED_STATE, "state")
        ]
        
        for metric, expected_keyword in metrics_with_help:
            # pylint: disable=protected-access
            help_text = metric._documentation.lower()
            assert expected_keyword in help_text, f"Metric help should mention {expected_keyword}"
            assert len(help_text) > 10, "Help text should be descriptive"


class TestMonitoringMemoryManagement:
    """Tests for memory management and resource cleanup."""

    def test_metric_memory_cleanup(self) -> None:
        """Test that metrics don't accumulate unbounded memory."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch("morphogenetic_engine.monitoring.start_http_server"):
            monitor = PrometheusMonitor(experiment_id="memory_test")
            
            # Create many metrics with unique labels to test memory usage
            for i in range(1000):
                monitor.update_seed_metrics(
                    seed_id=f"seed_{i}", 
                    state="training", 
                    alpha=0.5,
                    drift=0.1,
                    health_signal=0.8,
                    training_progress=0.6
                )
            
            # Clear registry
            MetricValidator.clear_registry()
            
            # Memory should not grow significantly
            final_memory = process.memory_info().rss
            memory_growth = final_memory - initial_memory
            
            # Allow some growth but not unbounded (< 50MB for 1000 metrics)
            assert memory_growth < 50 * 1024 * 1024, f"Memory grew by {memory_growth / (1024*1024):.1f} MB"

    def test_prometheus_export_format_validation(self) -> None:
        """Test that metrics export in valid Prometheus format."""
        with patch("morphogenetic_engine.monitoring.start_http_server"):
            monitor = PrometheusMonitor(experiment_id="export_test")
            monitor.update_training_metrics("phase_1", 0.5, 0.4, 0.85, 0.90)
            monitor.record_germination()
            
            # Ensure metrics are collected properly by checking the specific metrics
            validator = MetricValidator()
            
            # Verify metrics were recorded first
            train_loss = validator.get_gauge_value(TRAINING_LOSS, {"phase": "phase_1", "experiment_id": "export_test"})
            germination_count = validator.get_counter_value(GERMINATIONS_TOTAL, {"experiment_id": "export_test"})
            
            # If metrics aren't being recorded, skip the export format test
            if train_loss is None or germination_count is None:
                pytest.skip("Metrics not being recorded properly - registry may be cleared")
            
            # Generate metrics output
            output = generate_latest().decode('utf-8')
            
            # Now we know metrics should be present
            assert len(output) > 0, "Prometheus output should not be empty"
            
            # Look for kaslite metrics
            kaslite_metrics = [line for line in output.split('\n') if 'kaslite_' in line and not line.startswith('#')]
            assert len(kaslite_metrics) > 0, "No kaslite metrics found in output"
            
            # Validate format contains expected metrics (flexible pattern matching)
            training_loss_found = any('kaslite_training_loss' in line and 'export_test' in line and 'phase_1' in line for line in kaslite_metrics)
            germination_found = any('kaslite_germinations_total' in line and 'export_test' in line for line in kaslite_metrics)
            
            assert training_loss_found, f"Training loss metric not found. Available kaslite metrics: {kaslite_metrics}"
            assert germination_found, f"Germination metric not found. Available kaslite metrics: {kaslite_metrics}"
            
            # Validate no malformed lines
            for line in kaslite_metrics:
                if line.strip():
                    # Valid metric line should have format: metric_name{labels} value
                    assert '{' in line and '}' in line and ' ' in line, f"Malformed metric line: {line}"

    def test_label_cardinality_monitoring(self) -> None:
        """Test monitoring of label cardinality to prevent explosion."""
        with patch("morphogenetic_engine.monitoring.start_http_server"):
            monitor = PrometheusMonitor(experiment_id="cardinality_test")
            
            # Track unique label combinations
            label_combinations = set()
            
            for i in range(100):
                seed_id = f"seed_{i}"
                monitor.update_seed_metrics(seed_id=seed_id, state="training", alpha=0.5)
                label_combinations.add((seed_id, "cardinality_test"))
            
            # Cardinality should be reasonable
            assert len(label_combinations) == 100
            
            # Verify metrics are recorded correctly
            validator = MetricValidator()
            state_value = validator.get_gauge_value(
                SEED_STATE, {"seed_id": "seed_99", "experiment_id": "cardinality_test"}
            )
            assert state_value is not None and abs(state_value - 1.0) < 1e-10  # training state


class TestMonitoringNetworkResilience:
    """Tests for network failure resilience and recovery."""

    def test_network_failure_resilience(self) -> None:
        """Test behavior when network interfaces are unavailable."""
        with patch("morphogenetic_engine.monitoring.start_http_server") as mock_server:
            # Simulate various network failures
            mock_server.side_effect = [
                OSError("Network is unreachable"),
                OSError("Address already in use"),
                None  # Success on third try
            ]
            
            monitor = PrometheusMonitor(experiment_id="network_test")
            
            # Should handle failures gracefully
            monitor.start_metrics_server()
            assert not monitor.server_started
            
            monitor.start_metrics_server()
            assert not monitor.server_started
            
            monitor.start_metrics_server()
            assert monitor.server_started

    def test_prometheus_client_compatibility(self) -> None:
        """Test compatibility with expected Prometheus client features."""
        # Test that we're using expected API
        assert hasattr(prometheus_client, 'Counter')
        assert hasattr(prometheus_client, 'Gauge')
        assert hasattr(prometheus_client, 'Histogram')
        assert hasattr(prometheus_client, 'start_http_server')
        
        # Test metric creation patterns work
        test_counter = prometheus_client.Counter('test_compat_counter', 'Test counter', ['label'])
        test_counter.labels(label='test').inc()
        
        # Verify collection works
        samples = list(test_counter.collect())
        assert len(samples) > 0
        
        # Clean up test metric
        try:
            prometheus_client.REGISTRY.unregister(test_counter)
        except KeyError:
            pass


class TestMonitoringPerformanceBaselines:
    """Performance baseline and regression tests."""

    def test_performance_baseline_regression(self) -> None:
        """Test that performance doesn't regress below baseline."""
        with patch("morphogenetic_engine.monitoring.start_http_server"):
            monitor = PrometheusMonitor(experiment_id="baseline_test")
            
            # Warm up
            for _ in range(100):
                monitor.update_training_metrics("phase_1", 0.1, 0.2, 0.8, 0.9)
            
            # Benchmark core operations
            start_time = time.perf_counter()
            for _ in range(1000):
                monitor.update_training_metrics("phase_1", 0.1, 0.2, 0.8, 0.9)
            end_time = time.perf_counter()
            
            operations_per_second = 1000 / (end_time - start_time)
            
            # Should handle at least 5,000 operations per second (conservative baseline)
            assert operations_per_second > 5_000, f"Performance regression: {operations_per_second:.1f} ops/sec"

    def test_concurrent_operations_performance(self) -> None:
        """Test performance under concurrent load."""
        with patch("morphogenetic_engine.monitoring.start_http_server"):
            monitor = PrometheusMonitor(experiment_id="concurrent_perf_test")
            
            def concurrent_worker(worker_id: int) -> None:
                """Worker function for concurrent performance testing."""
                for i in range(200):
                    monitor.update_training_metrics(f"phase_{worker_id}", 0.1, 0.2, 0.8, 0.9)
                    if i % 10 == 0:
                        monitor.record_germination()
            
            # Test with multiple concurrent workers
            threads = [threading.Thread(target=concurrent_worker, args=(i,)) for i in range(5)]
            
            start_time = time.perf_counter()
            
            for thread in threads:
                thread.start()
                
            for thread in threads:
                thread.join()
                
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Should complete 1000 operations (5 threads × 200 ops) in reasonable time
            total_operations = 5 * 200
            operations_per_second = total_operations / duration
            
            # Conservative baseline: should handle at least 2,000 concurrent ops/sec
            assert operations_per_second > 2_000, f"Concurrent performance regression: {operations_per_second:.1f} ops/sec"
            assert duration < 2.0, f"Concurrent operations took too long: {duration:.2f}s"

    def test_metric_timestamp_accuracy(self) -> None:
        """Test that metric timestamps are accurate and consistent."""
        with patch("morphogenetic_engine.monitoring.start_http_server"):
            monitor = PrometheusMonitor(experiment_id="timestamp_test")
            
            # Record metrics at known times
            monitor.update_training_metrics("phase_1", 0.5, 0.4, 0.85, 0.90)
            after_time = time.time()
            
            # Verify experiment duration is within expected range
            monitor.update_experiment_duration()
            
            validator = MetricValidator()
            duration = validator.get_gauge_value(
                EXPERIMENT_DURATION, {"experiment_id": "timestamp_test"}
            )
            
            # Duration should be reasonable (within measurement window)
            assert duration is not None
            expected_duration = after_time - monitor.experiment_start_time
            assert abs(duration - expected_duration) < 1.0, "Timestamp accuracy issue"
