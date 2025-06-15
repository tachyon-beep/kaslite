"""Core module for morphogenetic engine containing seed management and germination control."""
import logging
import threading
import time
from collections import deque
from typing import Dict, Optional, List, Any
import torch


class SeedManager:
    """
    Singleton manager for sentinel seeds with thread-safe operations.
    """
    _instance: Optional['SeedManager'] = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize SeedManager with empty state."""
        # Only initialize once, even if __init__ is called multiple times
        if not hasattr(self, '_initialized'):
            self.seeds: Dict[str, Dict] = {}
            self.germination_log: List[Dict[str, Any]] = []
            self.lock = threading.RLock()
            self._initialized = True

    def __new__(cls) -> 'SeedManager':
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def register_seed(self, seed_module, seed_id: str) -> None:
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

    def request_germination(self, seed_id: str) -> bool:
        """Request germination for a specific seed. Returns True if successful."""
        with self.lock:
            seed_info = self.seeds.get(seed_id)
            if not seed_info or seed_info["status"] != "dormant":
                return False

            try:
                seed_info["module"].initialize_child()
                seed_info["status"] = "active"
                self._log_event(seed_id, True)
                return True
            except (RuntimeError, ValueError) as e:
                logging.exception("Germination failed for '%s': %s", seed_id, e)
                seed_info["status"] = "failed"
                self._log_event(seed_id, False)
                return False

    def _log_event(self, seed_id: str, success: bool) -> None:
        """Log a germination event with timestamp."""
        self.germination_log.append(
            {
                "seed_id": seed_id,
                "success": success,
                "timestamp": time.time(),
            }
        )

    def record_transition(self, seed_id: str, old_state: str, new_state: str) -> None:
        """Record a state change for analytics."""
        self.germination_log.append(
            {
                "seed_id": seed_id,
                "from": old_state,
                "to": new_state,
                "timestamp": time.time(),
            }
        )

    def record_drift(self, seed_id: str, drift: float) -> None:
        """Record drift telemetry for a specific seed."""
        with self.lock:
            if seed_id in self.seeds:
                self.seeds[seed_id]["telemetry"]["drift"] = drift


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
    ) -> None:
        self.seed_manager = seed_manager
        self.patience = patience
        self.delta = delta
        self.acc_threshold = acc_threshold  # Accuracy threshold for germination
        self.plateau = 0
        self.prev_loss = float("inf")

    def step(self, val_loss: float, val_acc: float) -> bool:
        """
        Process a training step and determine if germination should occur.
        
        Args:
            val_loss: Current validation loss
            val_acc: Current validation accuracy
            
        Returns:
            True if germination occurred, False otherwise
        """
        # Calculate absolute difference
        loss_diff = abs(val_loss - self.prev_loss)

        # Check if loss has plateaued (minimal improvement)
        if loss_diff < self.delta:
            self.plateau += 1
        else:
            self.plateau = 0  # Reset if we see improvement

        self.prev_loss = val_loss

        # Only trigger germination if:
        # 1. Accuracy is below threshold (problem not solved)
        # 2. Loss plateau persists beyond patience
        if val_acc < self.acc_threshold and self.plateau >= self.patience:
            self.plateau = 0  # Reset plateau counter
            seed_id = self._select_seed()
            if seed_id and self.seed_manager.request_germination(seed_id):
                return True  # Signal germination occurred
        return False

    def _select_seed(self) -> Optional[str]:
        candidate_id = None
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
