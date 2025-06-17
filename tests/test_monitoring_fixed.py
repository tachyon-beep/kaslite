"""
Unit tests for the monitoring functionality.

Tests the PrometheusMonitor class with the actual implementation interface.
"""

import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from morphogenetic_engine.monitoring import (
    BEST_ACCURACY,
    EPOCHS_TOTAL,
    GERMINATIONS_TOTAL,
    KASMINA_PLATEAU_COUNTER,
    SEED_ALPHA,
    SEED_STATE,
    TRAINING_LOSS,
    VALIDATION_ACCURACY,
    VALIDATION_LOSS,
    PrometheusMonitor,
    cleanup_monitoring,
    get_monitor,
    initialize_monitoring,
)


class TestPrometheusMonitor:
    """Test suite for PrometheusMonitor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = PrometheusMonitor(experiment_id="test_exp_123")

    def test_monitor_initialization(self):
        """Test PrometheusMonitor initialization."""
        monitor = PrometheusMonitor(experiment_id="test_exp_456", port=9000)

        assert monitor.experiment_id == "test_exp_456"
        assert monitor.port == 9000
        assert monitor.server_started is False
        assert isinstance(monitor.server_lock, type(threading.Lock()))
        assert "dormant" in monitor.state_map
        assert "active" in monitor.state_map

    @patch("morphogenetic_engine.monitoring.start_http_server")
    def test_start_metrics_server_success(self, mock_start_server):
        """Test successful metrics server start."""
        self.monitor.start_metrics_server()

        mock_start_server.assert_called_once_with(self.monitor.port)
        assert self.monitor.server_started is True

    @patch("morphogenetic_engine.monitoring.start_http_server")
    def test_start_metrics_server_failure(self, mock_start_server):
        """Test metrics server start failure handling."""
        mock_start_server.side_effect = OSError("Port already in use")

        # Should not raise exception
        self.monitor.start_metrics_server()
        assert self.monitor.server_started is False

    @patch("morphogenetic_engine.monitoring.EPOCHS_TOTAL")
    @patch("morphogenetic_engine.monitoring.EPOCH_DURATION")
    def test_record_epoch_completion(self, mock_duration, mock_epochs):
        """Test epoch completion recording."""
        mock_epochs_labels = Mock()
        mock_epochs.labels.return_value = mock_epochs_labels
        mock_duration_labels = Mock()
        mock_duration.labels.return_value = mock_duration_labels

        self.monitor.record_epoch_completion("phase_1", 12.5)

        mock_epochs.labels.assert_called_once_with(phase="phase_1", experiment_id="test_exp_123")
        mock_epochs_labels.inc.assert_called_once()
        mock_duration.labels.assert_called_once_with(phase="phase_1", experiment_id="test_exp_123")
        mock_duration_labels.observe.assert_called_once_with(12.5)

    @patch("morphogenetic_engine.monitoring.TRAINING_LOSS")
    @patch("morphogenetic_engine.monitoring.VALIDATION_LOSS")
    @patch("morphogenetic_engine.monitoring.VALIDATION_ACCURACY")
    @patch("morphogenetic_engine.monitoring.BEST_ACCURACY")
    def test_update_training_metrics(
        self, mock_best_acc, mock_val_acc, mock_val_loss, mock_train_loss
    ):
        """Test training metrics update."""
        # Setup mocks
        for mock in [mock_train_loss, mock_val_loss, mock_val_acc]:
            mock_labels = Mock()
            mock.labels.return_value = mock_labels

        mock_best_labels = Mock()
        mock_best_acc.labels.return_value = mock_best_labels

        self.monitor.update_training_metrics("phase_2", 0.25, 0.30, 0.85, 0.90)

        # Verify all metrics were updated
        mock_train_loss.labels.assert_called_once_with(
            phase="phase_2", experiment_id="test_exp_123"
        )
        mock_val_loss.labels.assert_called_once_with(phase="phase_2", experiment_id="test_exp_123")
        mock_val_acc.labels.assert_called_once_with(phase="phase_2", experiment_id="test_exp_123")
        mock_best_acc.labels.assert_called_once_with(experiment_id="test_exp_123")

    @patch("morphogenetic_engine.monitoring.GERMINATIONS_TOTAL")
    def test_record_germination(self, mock_germinations):
        """Test germination event recording."""
        mock_labels = Mock()
        mock_germinations.labels.return_value = mock_labels

        self.monitor.record_germination()

        mock_germinations.labels.assert_called_once_with(experiment_id="test_exp_123")
        mock_labels.inc.assert_called_once()

    @patch("morphogenetic_engine.monitoring.PHASE_TRANSITIONS_TOTAL")
    def test_record_phase_transition(self, mock_transitions):
        """Test phase transition recording."""
        mock_labels = Mock()
        mock_transitions.labels.return_value = mock_labels

        self.monitor.record_phase_transition("phase_1", "phase_2")

        mock_transitions.labels.assert_called_once_with(
            from_phase="phase_1", to_phase="phase_2", experiment_id="test_exp_123"
        )
        mock_labels.inc.assert_called_once()

    @patch("morphogenetic_engine.monitoring.SEED_STATE")
    @patch("morphogenetic_engine.monitoring.SEED_ALPHA")
    @patch("morphogenetic_engine.monitoring.SEED_DRIFT")
    @patch("morphogenetic_engine.monitoring.SEED_HEALTH_SIGNAL")
    @patch("morphogenetic_engine.monitoring.SEED_TRAINING_PROGRESS")
    def test_update_seed_metrics(
        self, mock_progress, mock_health, mock_drift, mock_alpha, mock_state
    ):
        """Test seed metrics update."""
        # Setup mocks
        mocks = [mock_state, mock_alpha, mock_drift, mock_health, mock_progress]
        for mock in mocks:
            mock_labels = Mock()
            mock.labels.return_value = mock_labels

        self.monitor.update_seed_metrics(
            seed_id="seed_1_2",
            state="blending",
            alpha=0.5,
            drift=0.1,
            health_signal=0.8,
            training_progress=0.6,
        )

        # Verify state mapping
        mock_state.labels.assert_called_once_with(seed_id="seed_1_2", experiment_id="test_exp_123")
        mock_state.labels.return_value.set.assert_called_once_with(2)  # "blending" = 2

        # Verify other metrics
        mock_alpha.labels.assert_called_once_with(seed_id="seed_1_2", experiment_id="test_exp_123")
        mock_alpha.labels.return_value.set.assert_called_once_with(0.5)

    @patch("morphogenetic_engine.monitoring.KASMINA_PLATEAU_COUNTER")
    @patch("morphogenetic_engine.monitoring.KASMINA_PATIENCE")
    def test_update_kasmina_metrics(self, mock_patience, mock_plateau):
        """Test Kasmina controller metrics update."""
        mock_plateau_labels = Mock()
        mock_plateau.labels.return_value = mock_plateau_labels
        mock_patience_labels = Mock()
        mock_patience.labels.return_value = mock_patience_labels

        self.monitor.update_kasmina_metrics(5, 15)

        mock_plateau.labels.assert_called_once_with(experiment_id="test_exp_123")
        mock_plateau_labels.set.assert_called_once_with(5)
        mock_patience.labels.assert_called_once_with(experiment_id="test_exp_123")
        mock_patience_labels.set.assert_called_once_with(15)

    @patch("morphogenetic_engine.monitoring.EXPERIMENT_DURATION")
    @patch("time.time")
    def test_update_experiment_duration(self, mock_time, mock_duration):
        """Test experiment duration update."""
        mock_time.return_value = 1000.0
        self.monitor.experiment_start_time = 900.0

        mock_labels = Mock()
        mock_duration.labels.return_value = mock_labels

        self.monitor.update_experiment_duration()

        mock_duration.labels.assert_called_once_with(experiment_id="test_exp_123")
        mock_labels.set.assert_called_once_with(100.0)  # 1000 - 900

    def test_state_mapping(self):
        """Test seed state to numeric mapping."""
        assert self.monitor.state_map["dormant"] == 0
        assert self.monitor.state_map["training"] == 1
        assert self.monitor.state_map["blending"] == 2
        assert self.monitor.state_map["active"] == 3
        assert self.monitor.state_map["failed"] == -1


class TestMonitoringUtilities:
    """Test monitoring utility functions."""

    def test_initialize_monitoring(self):
        """Test monitoring initialization."""
        with patch(
            "morphogenetic_engine.monitoring.PrometheusMonitor"
        ) as mock_monitor_class, patch("morphogenetic_engine.monitoring.start_http_server"):

            mock_monitor = Mock()
            mock_monitor_class.return_value = mock_monitor

            result = initialize_monitoring("test_exp", 8080)

            mock_monitor_class.assert_called_once_with("test_exp", 8080)
            mock_monitor.start_metrics_server.assert_called_once()
            assert result == mock_monitor

    def test_cleanup_monitoring(self):
        """Test monitoring cleanup."""
        with patch("morphogenetic_engine.monitoring._monitor") as mock_monitor:
            mock_monitor.update_experiment_duration = Mock()

            cleanup_monitoring()

            mock_monitor.update_experiment_duration.assert_called_once()

    def test_get_monitor(self):
        """Test getting current monitor instance."""
        with patch("morphogenetic_engine.monitoring._monitor", "test_monitor"):
            result = get_monitor()
            assert result == "test_monitor"


class TestMonitoringIntegration:
    """Integration tests for monitoring functionality."""

    def test_monitor_thread_safety(self):
        """Test that monitor can handle concurrent operations."""
        monitor = PrometheusMonitor(experiment_id="test_thread_safety")

        def update_metrics():
            for i in range(10):
                monitor.update_training_metrics(f"phase_{i%2}", 0.1 * i, 0.2 * i, 0.8, 0.9)
                monitor.record_germination()

        # Run concurrent updates
        threads = [threading.Thread(target=update_metrics) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Test should complete without errors
        assert monitor.experiment_id == "test_thread_safety"

    @patch("morphogenetic_engine.monitoring.start_http_server")
    def test_monitor_server_start_idempotent(self, mock_start_server):
        """Test that starting server multiple times is safe."""
        monitor = PrometheusMonitor(experiment_id="test_idempotent")

        # Start server multiple times
        monitor.start_metrics_server()
        monitor.start_metrics_server()
        monitor.start_metrics_server()

        # Should only call start_http_server once
        mock_start_server.assert_called_once()
        assert monitor.server_started is True
