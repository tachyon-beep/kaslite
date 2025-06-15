import logging
import threading
import time
from collections import deque
from typing import Dict, Optional
import torch

class SeedManager:
    _instance = None
    _singleton_lock = threading.Lock()
    
    def __new__(cls):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.seeds: Dict[str, Dict] = {}
                cls._instance.germination_log = []
                cls._instance.lock = threading.RLock()
            return cls._instance

    def register_seed(self, seed_module, seed_id: str):
        with self.lock:
            self.seeds[seed_id] = {
                "module": seed_module,
                "status": "dormant",
                "buffer": deque(maxlen=500),
                "telemetry": {"drift": 0.0, "variance": 0.0},
            }

    def append_to_buffer(self, seed_id: str, x: torch.Tensor):
        with self.lock:
            if seed_id in self.seeds:
                self.seeds[seed_id]["buffer"].append(x.detach().clone())

    def request_germination(self, seed_id: str) -> bool:
        with self.lock:
            seed_info = self.seeds.get(seed_id)
            if not seed_info or seed_info["status"] != "dormant":
                return False
            
            try:
                seed_info["module"].initialize_child()
                seed_info["status"] = "active"
                self._log_event(seed_id, True)
                return True
            except Exception as e:
                logging.exception(f"Germination failed for '{seed_id}': {e}")
                seed_info["status"] = "failed"
                self._log_event(seed_id, False)
                return False

    def _log_event(self, seed_id: str, success: bool):
        self.germination_log.append({
            "seed_id": seed_id,
            "success": success,
            "timestamp": time.time(),
        })

    def record_drift(self, seed_id: str, drift: float):
        with self.lock:
            if seed_id in self.seeds:
                self.seeds[seed_id]["telemetry"]["drift"] = drift


class KasminaMicro:
    def __init__(self, seed_manager: SeedManager, patience: int = 15, 
                 delta: float = 1e-4, acc_threshold: float = 0.95):
        self.seed_manager = seed_manager
        self.patience = patience
        self.delta = delta
        self.acc_threshold = acc_threshold  # Accuracy threshold for germination
        self.plateau = 0
        self.prev_loss = float('inf')

    def step(self, val_loss: float, val_acc: float) -> bool:
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
        worst_signal = float('inf')  # Start with worst possible value
        
        with self.seed_manager.lock:
            for sid, info in self.seed_manager.seeds.items():
                if info["status"] == "dormant":
                    # Calculate health signal (lower = worse bottleneck)
                    signal = info["module"].get_health_signal()
                    if signal < worst_signal:
                        worst_signal = signal
                        candidate_id = sid
        return candidate_id

