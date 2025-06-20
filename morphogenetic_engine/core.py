"""Core module for morphogenetic engine containing seed management and germination control."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import torch

from .events import SeedState
from .logger import ExperimentLogger
from .monitoring import get_monitor
from .ui_dashboard import RichDashboard


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
                # Germinate seed but don't start training yet - it goes to the "parking lot"
                seed_info["module"].initialize_child()
                seed_info["status"] = "germinated"  # Parking lot state
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

    def get_currently_training_seed(self) -> tuple[int, int] | None:
        """Get the seed that is currently training (active state), if any."""
        with self.lock:
            for seed_id, seed_info in self.seeds.items():
                if seed_info["status"] == "active":
                    return seed_id
            return None

    def get_next_germinated_seed(self) -> tuple[int, int] | None:
        """Get the next seed in the germinated queue waiting to start training."""
        with self.lock:
            # Find the oldest germinated seed (first one added)
            oldest_time = float('inf')
            oldest_seed = None
            for seed_id, seed_info in self.seeds.items():
                if seed_info["status"] == "germinated":
                    # Use germination log to find oldest
                    for log_entry in self.germination_log:
                        if (log_entry.get("seed_id") == f"L{seed_id[0]}_S{seed_id[1]}" and 
                            log_entry.get("success", False) and
                            log_entry.get("timestamp", float('inf')) < oldest_time):
                            oldest_time = log_entry["timestamp"]
                            oldest_seed = seed_id
            return oldest_seed

    def start_training_next_seed(self) -> tuple[int, int] | None:
        """Start training the next seed in the germinated queue, if no seed is currently training."""
        with self.lock:
            # Check if any seed is currently training
            if self.get_currently_training_seed() is not None:
                return None  # Another seed is already training
            
            # Get the next germinated seed
            next_seed = self.get_next_germinated_seed()
            if next_seed is None:
                return None  # No seeds waiting
            
            # Start training this seed
            seed_info = self.seeds[next_seed]
            seed_info["status"] = "active"
            seed_info["module"]._set_state("active")
            
            # Log the training start
            if self.logger is not None:
                self.logger.log_seed_event_detailed(
                    epoch=0,  # We don't have epoch context here
                    event_type="TRAINING_START",
                    message=f"Seed L{next_seed[0]}_S{next_seed[1]} started training",
                    data={"seed_id": f"L{next_seed[0]}_S{next_seed[1]}"}
                )
            
            return next_seed


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
    ) -> None:
        self.seed_manager = seed_manager
        self.patience = patience
        self.delta = delta
        self.acc_threshold = acc_threshold  # Accuracy threshold for germination
        self.plateau = 0
        self.prev_loss = float("inf")
        self.logger = logger

    def assess_and_update_seeds(self, epoch: int) -> None:
        """Assess all seeds, update their metrics, and log changes."""
        if not self.logger:
            return

        # Collect all seed information for batch logging
        from .events import SeedInfo, SeedState

        seed_infos = []

        with self.seed_manager.lock:
            for seed_id, seed_info in self.seed_manager.seeds.items():
                module = seed_info["module"]

                # Store previous state for comparison
                prev_state = module.state

                # This method should update the seed's internal state
                module.update_state()

                # Convert string state to SeedState enum if needed
                if isinstance(module.state, str):
                    try:
                        state_enum = SeedState[module.state.upper()]
                    except KeyError:
                        state_enum = SeedState.DORMANT  # Default fallback
                else:
                    state_enum = module.state

                # Collect seed info for batch logging
                layer_idx, seed_idx = seed_id
                seed_info_obj = SeedInfo(
                    id=seed_id,
                    state=state_enum,
                    layer=layer_idx,
                    index_in_layer=seed_idx,
                    metrics={
                        "alpha": module.alpha,
                        "grad_norm": module.get_gradient_norm(),
                        "patience": getattr(module, "patience_counter", None),
                    },
                )
                seed_infos.append(seed_info_obj)

                # Log a specific event only if the state has changed
                if module.state != prev_state:
                    self.logger.log_seed_event_detailed(
                        epoch=epoch,
                        event_type="STATE_CHANGE",
                        message=f"Seed L{layer_idx}_S{seed_idx} changed from {prev_state} to {module.state}",
                        data={
                            "seed_id": f"L{layer_idx}_S{seed_idx}",
                            "from_state": prev_state,
                            "to_state": module.state,
                            "epoch": epoch,
                        },
                    )

        # Log all seed states in one batch
        self.logger.log_seed_state_update(epoch, seed_infos)
        
        # Check if we need to start training the next seed in the queue
        self.start_training_next_seed()

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
            if seed_id and self.seed_manager.request_germination(seed_id, epoch=epoch):  # Pass epoch
                self.plateau = 0  # Reset plateau counter only on successful germination
                # Record germination in monitoring
                if monitor:
                    monitor.record_germination()

                # Log the germination event via the logger
                if self.logger:
                    self.logger.log_seed_event_detailed(
                        epoch=epoch,
                        event_type="GERMINATION",
                        message=f"Seed L{seed_id[0]}_S{seed_id[1]} germinated!",
                        data={"seed_id": f"L{seed_id[0]}_S{seed_id[1]}", "epoch": epoch},
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

    def is_any_seed_training(self) -> bool:
        """Check if any seed is currently in ACTIVE state (training)."""
        with self.seed_manager.lock:
            for seed_info in self.seed_manager.seeds.values():
                if seed_info.get("state") == SeedState.ACTIVE.value:
                    return True
        return False

    def start_training_next_seed(self) -> bool:
        """Start training the next germinated seed if no seed is currently training."""
        if self.is_any_seed_training():
            return False  # Another seed is already training
        
        # Find the first seed in GERMINATED state (waiting in parking lot)
        with self.seed_manager.lock:
            for seed_id, seed_info in self.seed_manager.seeds.items():
                if seed_info.get("state") == SeedState.GERMINATED.value:
                    # Start training this seed
                    if "module" in seed_info:
                        seed_info["module"]._set_state(SeedState.ACTIVE)
                        
                        # Log the training start
                        if self.logger:
                            self.logger.log_seed_event_detailed(
                                epoch=0,  # We don't have epoch context here
                                event_type="TRAINING_START", 
                                message=f"Seed L{seed_id[0]}_S{seed_id[1]} started training!",
                                data={"seed_id": f"L{seed_id[0]}_S{seed_id[1]}"},
                            )
                        return True
        return False  # No seeds waiting to train
