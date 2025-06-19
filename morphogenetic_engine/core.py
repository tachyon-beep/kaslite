"""Core module for morphogenetic engine containing seed management and germination control."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import torch

from .cli_dashboard import RichDashboard
from .logger import ExperimentLogger
from .monitoring import get_monitor


class SeedManager:
    """
    Singleton manager for sentinel seeds with thread-safe operations.
    """

    _instance: Optional["SeedManager"] = None
    _singleton_lock = threading.Lock()

    # Declare instance attributes for type checking
    seeds: Dict[tuple[int, int], Dict]
    germination_log: List[Dict[str, Any]]
    lock: threading.RLock
    logger: Optional[ExperimentLogger]
    _initialized: bool

    def __new__(cls, *args, **kwargs) -> "SeedManager":
        with cls._singleton_lock:
            if cls._instance is None:
                # Create the instance
                instance = super().__new__(cls)
                # Initialize its state directly
                instance.seeds = {}
                instance.germination_log = []
                instance.lock = threading.RLock()
                instance.logger = kwargs.get("logger")
                instance._initialized = True  # Optional: for clarity
                cls._instance = instance
        return cls._instance

    def __init__(self, logger: Optional[ExperimentLogger] = None) -> None:
        """Initialize SeedManager with optional logger and empty state."""
        # The __init__ can now be used for lighter-weight tasks,
        # like injecting dependencies on an existing instance.
        if logger is not None:
            self.logger = logger

    def register_seed(self, seed_module, seed_id: tuple[int, int]) -> None:
        """Register a new seed module with the manager."""
        with self.lock:
            self.seeds[seed_id] = {
                "module": seed_module,
                "status": "dormant",
                "state": "dormant",
                "alpha": 0.0,
                "buffer": deque(maxlen=500),
                "telemetry": {"drift": 0.0, "variance": 0.0},
            }

    def append_to_buffer(self, seed_id: str, x: torch.Tensor) -> None:
        """Append tensor data to the specified seed's buffer."""
        with self.lock:
            if seed_id in self.seeds:
                self.seeds[seed_id]["buffer"].append(x.detach().clone())

    def request_germination(self, seed_id: tuple[int, int], epoch: int = 0) -> bool:
        """Request germination for a specific seed. Returns True if successful."""
        with self.lock:
            seed_info = self.seeds.get(seed_id)
            if not seed_info or seed_info["status"] != "dormant":
                return False

            try:
                seed_info["module"].initialize_child()
                seed_info["status"] = "active"
                self._log_event(seed_id, True)
                if self.logger is not None:
                    self.logger.log_germination(epoch, seed_id)
                return True
            except (RuntimeError, ValueError) as e:
                logging.exception("Germination failed for '%s': %s", seed_id, e)
                seed_info["status"] = "failed"
                self._log_event(seed_id, False)
                return False

    def _log_event(self, seed_id: tuple[int, int], success: bool) -> None:
        """Log a germination event with timestamp."""
        self.germination_log.append(
            {
                "event_type": "germination_attempt",
                "seed_id": f"L{seed_id[0]}_S{seed_id[1]}",  # Convert tuple to string for logging
                "success": success,
                "timestamp": time.time(),
            }
        )

    def record_transition(self, seed_id: tuple[int, int], old_state: str, new_state: str, epoch: int = 0) -> None:
        """Record a state change for analytics and log the event."""
        self.germination_log.append(
            {
                "event_type": "state_transition",
                "seed_id": f"L{seed_id[0]}_S{seed_id[1]}",  # Convert tuple to string for logging
                "from": old_state,
                "to": new_state,
                "timestamp": time.time(),
            }
        )
        if self.logger is not None:
            self.logger.log_seed_event(epoch, seed_id, old_state, new_state)

    def record_drift(self, seed_id: tuple[int, int], drift: float) -> None:
        """Record drift telemetry for a specific seed."""
        with self.lock:
            if seed_id in self.seeds:
                self.seeds[seed_id]["telemetry"]["drift"] = drift

    def reset(self) -> None:
        """Reset the SeedManager to initial state. Primarily for testing."""
        with self.lock:
            self.seeds.clear()
            self.germination_log.clear()

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance. Primarily for testing."""
        with cls._singleton_lock:
            if cls._instance is not None:
                cls._instance.reset()
                cls._instance = None


class KasminaMicro:
    """
    Micro-germination controller that monitors training progress and triggers seed germination.

    This class watches for training plateaus and low accuracy to decide when to activate
    dormant seeds to increase model capacity.
    """

    def __init__(
        self,
        seed_manager: SeedManager,
        patience: int = 15,
        delta: float = 1e-4,
        acc_threshold: float = 0.95,
        logger: ExperimentLogger | None = None,
        dashboard: RichDashboard | None = None,
    ) -> None:
        self.seed_manager = seed_manager
        self.patience = patience
        self.delta = delta
        self.acc_threshold = acc_threshold  # Accuracy threshold for germination
        self.plateau = 0
        self.prev_loss = float("inf")
        self.logger = logger
        self.dashboard = dashboard

    def step(self, epoch: int, val_loss: float, val_acc: float) -> bool:
        """
        Process a training step and determine if germination should occur.

        Args:
            epoch: The current epoch number.
            val_loss: Current validation loss
            val_acc: Current validation accuracy

        Returns:
            True if germination occurred, False otherwise
        """
        # Check if loss has not improved by at least delta
        if self.prev_loss - val_loss < self.delta:
            self.plateau += 1
        else:
            self.plateau = 0  # Reset if we see improvement

        self.prev_loss = val_loss

        # Update Prometheus metrics for Kasmina controller
        monitor = get_monitor()
        if monitor:
            monitor.update_kasmina_metrics(self.plateau, self.patience)

        # Only trigger germination if:
        # 1. Accuracy is below threshold (problem not solved)
        # 2. Loss plateau persists beyond patience
        if val_acc < self.acc_threshold and self.plateau >= self.patience:
            seed_id = self._select_seed()
            if seed_id and self.seed_manager.request_germination(seed_id):
                self.plateau = 0  # Reset plateau counter only on successful germination
                # Record germination in monitoring
                if monitor:
                    monitor.record_germination()
                # Log the phase transition for germination
                if self.logger:
                    self.logger.log_phase_update(
                        epoch=epoch,
                        from_phase="Monitoring",
                        to_phase="Germination",
                    )
                return True  # Signal germination occurred
        return False

    def _select_seed(self) -> Optional[tuple[int, int]]:
        candidate_id: Optional[tuple[int, int]] = None
        worst_signal = float("inf")  # Start with worst possible value

        with self.seed_manager.lock:
            for sid, info in self.seed_manager.seeds.items():
                if info["status"] == "dormant":
                    # Calculate health signal (lower = worse bottleneck)
                    signal = info["module"].get_health_signal()
                    if signal < worst_signal:
                        worst_signal = signal
                        candidate_id = sid
        return candidate_id
