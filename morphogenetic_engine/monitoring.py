"""
Prometheus monitoring instrumentation for the morphogenetic engine.

This module provides Prometheus metrics collection for training phases,
seed states, germination events, and real-time telemetry.
"""

import logging
import threading
import time
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from .events import SeedState

# Training Metrics
EPOCHS_TOTAL = Counter("kaslite_epochs_total", "Number of training epochs completed", ["phase", "experiment_id"])

GERMINATIONS_TOTAL = Counter("kaslite_germinations_total", "Total seed germinations", ["experiment_id"])

PHASE_TRANSITIONS_TOTAL = Counter(
    "kaslite_phase_transitions_total",
    "Phase transitions in experiment",
    ["from_phase", "to_phase", "experiment_id"],
)

# Performance Metrics
VALIDATION_LOSS = Gauge("kaslite_validation_loss", "Current validation loss", ["phase", "experiment_id"])

VALIDATION_ACCURACY = Gauge("kaslite_validation_accuracy", "Current validation accuracy", ["phase", "experiment_id"])

TRAINING_LOSS = Gauge("kaslite_training_loss", "Current training loss", ["phase", "experiment_id"])

BEST_ACCURACY = Gauge("kaslite_best_accuracy", "Best validation accuracy achieved", ["experiment_id"])

# Seed-specific Metrics
SEED_ALPHA = Gauge("kaslite_seed_alpha", "Current alpha blending value for each seed", ["seed_id", "experiment_id"])

SEED_DRIFT = Gauge("kaslite_seed_drift", "Interface drift per seed", ["seed_id", "experiment_id"])

SEED_HEALTH_SIGNAL = Gauge(
    "kaslite_seed_health_signal",
    "Health signal (activation variance) per seed",
    ["seed_id", "experiment_id"],
)

SEED_STATE = Gauge(
    "kaslite_seed_state",
    "Current state of each seed (encoded as enum value)",
    ["seed_id", "experiment_id"],
)

SEED_TRAINING_PROGRESS = Gauge(
    "kaslite_seed_training_progress",
    "Training progress for seeds in training state",
    ["seed_id", "experiment_id"],
)

# Kasmina Controller Metrics
KASMINA_PLATEAU_COUNTER = Gauge(
    "kaslite_kasmina_plateau_counter",
    "Current plateau counter in Kasmina controller",
    ["experiment_id"],
)

KASMINA_PATIENCE = Gauge("kaslite_kasmina_patience", "Patience threshold for Kasmina controller", ["experiment_id"])

# Training Duration Metrics
EPOCH_DURATION = Histogram("kaslite_epoch_duration_seconds", "Time taken to complete an epoch", ["phase", "experiment_id"])

EXPERIMENT_DURATION = Gauge("kaslite_experiment_duration_seconds", "Total experiment duration", ["experiment_id"])


class PrometheusMonitor:
    """
    Centralized Prometheus monitoring for morphogenetic experiments.

    This class provides a clean interface for updating metrics and manages
    the Prometheus HTTP server lifecycle.
    """

    def __init__(self, experiment_id: str, port: int = 8000):
        self.experiment_id = experiment_id
        self.port = port
        self.server_started = False
        self.server_lock = threading.Lock()
        self.experiment_start_time = time.time()

        # State mapping for seed states - built from SeedState enum for consistency
        self.state_map = {state.value: idx for idx, state in enumerate(SeedState)}
        self.state_map["failed"] = -1  # Add special failed state

    def start_metrics_server(self):
        """Start the Prometheus metrics HTTP server."""
        with self.server_lock:
            if not self.server_started:
                try:
                    start_http_server(self.port)
                    self.server_started = True
                    logging.info("Prometheus metrics server started on port %d", self.port)
                    logging.info("Metrics available at http://localhost:%d/metrics", self.port)
                except OSError as e:
                    logging.error("Failed to start Prometheus server: %s", e)
                    # Don't fail the experiment if monitoring fails

    def record_epoch_completion(self, phase: str, epoch_duration: float):
        """Record completion of a training epoch."""
        EPOCHS_TOTAL.labels(phase=phase, experiment_id=self.experiment_id).inc()
        EPOCH_DURATION.labels(phase=phase, experiment_id=self.experiment_id).observe(epoch_duration)

    def update_training_metrics(self, phase: str, train_loss: float, val_loss: float, val_acc: float, best_acc: float):
        """Update training performance metrics."""
        TRAINING_LOSS.labels(phase=phase, experiment_id=self.experiment_id).set(train_loss)
        VALIDATION_LOSS.labels(phase=phase, experiment_id=self.experiment_id).set(val_loss)
        VALIDATION_ACCURACY.labels(phase=phase, experiment_id=self.experiment_id).set(val_acc)
        BEST_ACCURACY.labels(experiment_id=self.experiment_id).set(best_acc)

    def record_germination(self):
        """Record a seed germination event."""
        GERMINATIONS_TOTAL.labels(experiment_id=self.experiment_id).inc()

    def record_phase_transition(self, from_phase: str, to_phase: str):
        """Record a phase transition."""
        PHASE_TRANSITIONS_TOTAL.labels(from_phase=from_phase, to_phase=to_phase, experiment_id=self.experiment_id).inc()

    def update_seed_metrics(
        self,
        seed_id: tuple[int, int] | str,
        state: str,
        alpha: float = 0.0,
        drift: float = 0.0,
        health_signal: float = 0.0,
        training_progress: float = 0.0,
    ):
        """Update all metrics for a specific seed."""
        # Convert tuple seed_id to string for Prometheus labels
        if isinstance(seed_id, tuple):
            seed_id_str = f"L{seed_id[0]}_S{seed_id[1]}"
        else:
            seed_id_str = seed_id

        # Convert state to numeric value
        state_value = self.state_map.get(state, -1)

        SEED_STATE.labels(seed_id=seed_id_str, experiment_id=self.experiment_id).set(state_value)
        SEED_ALPHA.labels(seed_id=seed_id_str, experiment_id=self.experiment_id).set(alpha)
        SEED_DRIFT.labels(seed_id=seed_id_str, experiment_id=self.experiment_id).set(drift)
        SEED_HEALTH_SIGNAL.labels(seed_id=seed_id_str, experiment_id=self.experiment_id).set(health_signal)
        SEED_TRAINING_PROGRESS.labels(seed_id=seed_id_str, experiment_id=self.experiment_id).set(training_progress)

    def update_kasmina_metrics(self, plateau_counter: int, patience: int):
        """Update Kasmina controller metrics."""
        KASMINA_PLATEAU_COUNTER.labels(experiment_id=self.experiment_id).set(plateau_counter)
        KASMINA_PATIENCE.labels(experiment_id=self.experiment_id).set(patience)

    def update_experiment_duration(self):
        """Update total experiment duration."""
        duration = time.time() - self.experiment_start_time
        EXPERIMENT_DURATION.labels(experiment_id=self.experiment_id).set(duration)


# Global monitor instance (will be initialized per experiment)
_monitor: Optional[PrometheusMonitor] = None


def get_monitor() -> Optional[PrometheusMonitor]:
    """Get the current monitor instance."""
    return _monitor


def initialize_monitoring(experiment_id: str, port: int = 8000) -> PrometheusMonitor:
    """Initialize monitoring for an experiment."""
    global _monitor  # pylint: disable=global-statement
    _monitor = PrometheusMonitor(experiment_id, port)
    _monitor.start_metrics_server()
    return _monitor


def cleanup_monitoring():
    """Cleanup monitoring resources."""
    global _monitor  # pylint: disable=global-statement
    if _monitor:
        _monitor.update_experiment_duration()
    _monitor = None
