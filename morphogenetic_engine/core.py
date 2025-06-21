"""Core module for morphogenetic engine containing seed management and germination control."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .events import SeedState
from .logger import ExperimentLogger
from .monitoring import get_monitor
from .ui_dashboard import RichDashboard


@dataclass(frozen=True)
class BlendingConfig:
    """Configuration for blending strategies."""

    fixed_steps: int = 30
    stabilization_epochs: int = 5  # Number of epochs to hold alpha=1.0 during stabilization
    high_drift_threshold: float = 0.12
    low_health_threshold: float = 1e-3
    performance_loss_factor: float = 0.8
    grad_norm_lower: float = 0.1
    grad_norm_upper: float = 1.0
    """Configuration for blending strategies."""

class SeedManager:
    """
    Singleton manager for sentinel seeds with thread-safe operations.
    """

    _instance: Optional["SeedManager"] = None
    _singleton_lock = threading.Lock()

    # Declare instance attributes for type checking
    seeds: Dict[tuple[int, int], Dict]
    germination_log: List[Dict[str, Any]]
    germinated_queue: deque[tuple[int, int]]
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
                instance.germinated_queue = deque()
                instance.lock = threading.RLock()
                instance.logger = kwargs.get("logger")
                instance._initialized = True  # Optional: for clarity
                cls._instance = instance
            else:
                # If instance exists and a new logger is provided, update it
                new_logger = kwargs.get("logger")
                if new_logger is not None:
                    cls._instance.logger = new_logger
        return cls._instance

    def __init__(self, logger: Optional[ExperimentLogger] = None) -> None:
        """Only do one-time initialization (including wiring the logger)."""
        # If we've already done __init__, bail out immediately.
        if getattr(self, "_init_done", False):
            return

        # First-ever init: wire up the logger if provided
        if logger is not None:
            self.logger = logger

        # Mark that we've initialized already
        self._init_done = True

    def register_seed(self, seed_module, seed_id: tuple[int, int]) -> None:
        """Register a new seed module with the manager."""
        with self.lock:
            self.seeds[seed_id] = {
                "module": seed_module,
                "state": SeedState.DORMANT.value,
                "alpha": 0.0,
                "gradient_norm": 0.0,
                "buffer": deque(maxlen=2000),  # Increased from 500 to handle more training data
                "telemetry": {"drift": 0.0, "variance": 0.0},
            }

    def append_to_buffer(self, seed_id: tuple[int, int], x: torch.Tensor) -> None:
        """Append tensor data to the specified seed's buffer."""
        with self.lock:
            if seed_id in self.seeds:
                self.seeds[seed_id]["buffer"].append(x.detach().clone())

    def request_germination(self, seed_id: tuple[int, int], epoch: int = 0) -> bool:
        """Request germination for a specific seed. Returns True if successful."""
        with self.lock:
            seed_info = self.seeds.get(seed_id)
            if not seed_info or seed_info.get("state") != SeedState.DORMANT.value:
                return False

            try:
                # Germinate seed but don't start training yet - it goes to the "parking lot"
                seed_info["module"].initialize_child(epoch=epoch)
                seed_info["germination_epoch"] = epoch  # Store germination epoch for training start
                self._log_event(seed_id, True, epoch)
                # State transition from DORMANT → GERMINATED is already logged by initialize_child()
                # No need for separate GERMINATION event
                
                # Add to the germinated queue for efficient FIFO processing
                self.germinated_queue.append(seed_id)
                
                return True
            except (RuntimeError, ValueError) as e:
                logging.exception("Germination failed for '%s': %s", seed_id, e)
                # Update state to indicate failure
                seed_info["state"] = "failed"
                self._log_event(seed_id, False, epoch)
                return False

    def _log_event(self, seed_id: tuple[int, int], success: bool, epoch: int = 0) -> None:
        """Log a germination event with timestamp."""
        self.germination_log.append(
            {
                "event_type": "germination_attempt",
                "seed_id": f"L{seed_id[0]}_S{seed_id[1]}",  # Convert tuple to string for logging
                "success": success,
                "timestamp": time.time(),
            }
        )
        
        # Dispatch to external logger if available
        if self.logger:
            self.logger.log_seed_event_detailed(
                epoch=epoch,
                event_type="GERMINATION_ATTEMPT",
                message=f"Seed L{seed_id[0]}_S{seed_id[1]} germination {'succeeded' if success else 'failed'}",
                data={"seed_id": f"L{seed_id[0]}_S{seed_id[1]}", "success": success},
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
            self.germinated_queue.clear()

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance. Primarily for testing."""
        with cls._singleton_lock:
            if cls._instance is not None:
                cls._instance.reset()
                cls._instance = None

    def get_currently_training_seed(self) -> tuple[int, int] | None:
        """Get the seed that is currently training (TRAINING state), if any."""
        with self.lock:
            for seed_id, seed_info in self.seeds.items():
                if seed_info.get("state") == SeedState.TRAINING.value:
                    return seed_id
            return None


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
        blending_config: BlendingConfig | None = None,  # Add new config
    ) -> None:
        self.seed_manager = seed_manager
        self.patience = patience
        self.delta = delta
        self.acc_threshold = acc_threshold  # Accuracy threshold for germination
        self.plateau = 0
        self.prev_loss = float("inf")
        self.logger = logger
        self.cull_embargo_epochs = 50  # Number of epochs to wait before replacing a culled seed
        self.graft_cfg = blending_config or BlendingConfig()

    def _handle_culled_seeds(self, epoch: int):
        """Check for culled seeds whose embargo period has expired and replace them."""
        parent_net = None
        # This is a bit of a hack to get the parent_net. A better solution would be to pass it in.
        with self.seed_manager.lock:
            for seed_info in self.seed_manager.seeds.values():
                if hasattr(seed_info["module"], "parent_net_ref"):
                    parent_net = seed_info["module"].parent_net_ref()
                    if parent_net:
                        break
        
        if not parent_net:
            return # Cannot proceed without a reference to the parent network

        with self.seed_manager.lock:
            for seed_id, seed_info in self.seed_manager.seeds.items():
                if seed_info.get("state") == SeedState.CULLED.value:
                    culling_epoch = seed_info.get("culling_epoch")
                    if culling_epoch is not None and epoch > culling_epoch + self.cull_embargo_epochs:
                        logging.info(f"Embargo lifted for culled seed {seed_id}. Replacing.")
                        parent_net.reset_culled_seed(seed_id)

    def assess_and_update_seeds(self, epoch: int) -> None:
        """Assess all seeds, apply state transitions, and handle culled seed replacement."""
        
        # --- FIX: ADVANCE THE QUEUE FIRST ---
        # Promote a seed that has been waiting since the last epoch.
        self._advance_training_queue(epoch)

        with self.seed_manager.lock:
            seed_ids = list(self.seed_manager.seeds.keys())
            for seed_id in seed_ids:
                # It's possible a seed is removed during iteration, so check existence
                if seed_id not in self.seed_manager.seeds:
                    continue

                info = self.seed_manager.seeds[seed_id]
                module = info["module"]

                # --- STRATEGY SELECTION ---
                if module.state == SeedState.GRAFTING.value and "graft_strategy" not in info:
                    strategy_name = self._choose_graft_strategy(seed_id)
                    info["graft_strategy"] = strategy_name
                    
                    # Capture comprehensive telemetry for logging
                    health_signal = module.get_health_signal()
                    drift = info.get("telemetry", {}).get("drift", 0.0)
                    current_loss = info.get("current_loss", float('inf'))
                    baseline_loss = info.get("baseline_loss", float('inf'))
                    grad_norm = info.get("avg_grad_norm", 0.0)
                    
                    # Create rich telemetry dict
                    telemetry_data = {
                        "health_signal": health_signal,
                        "drift": drift,
                        "baseline_loss": baseline_loss,
                        "current_loss": current_loss,
                        "grad_norm": grad_norm
                    }
                    
                    # Log the strategy choice with rich telemetry
                    if self.logger:
                        from .events import EventType, BlendStrategyChosenPayload, LogEvent
                        import time
                        payload = BlendStrategyChosenPayload(
                            seed_id=seed_id,
                            epoch=epoch,
                            strategy_name=strategy_name,
                            telemetry=telemetry_data,
                            timestamp=time.time()
                        )
                        event = LogEvent(EventType.BLEND_STRATEGY_CHOSEN, payload)
                        self.logger.log_event(epoch, event)
                        
                        # Record in analytics for Phase 2 dashboard
                        try:
                            from .grafting_analytics import get_grafting_analytics
                            analytics = get_grafting_analytics()
                            analytics.record_strategy_chosen(payload)
                        except ImportError:
                            pass  # Analytics not available
                        
                        # Also log detailed telemetry
                        self.logger.log_seed_event_detailed(
                            epoch=epoch,
                            event_type="GRAFT_STRATEGY_CHOSEN",
                            message=f"Seed L{seed_id[0]}_S{seed_id[1]} selected {strategy_name} strategy",
                            data={
                                "seed_id": f"L{seed_id[0]}_S{seed_id[1]}",
                                "strategy_name": strategy_name,
                                "telemetry": {
                                    "health_signal": health_signal,
                                    "drift": drift,
                                    "baseline_loss": baseline_loss,
                                    "current_loss": current_loss
                                }
                            }
                        )

                module.assess_and_transition_state(epoch)

        # After handling state transitions, check for any culled seeds that need replacing.
        self._handle_culled_seeds(epoch)

        # The rest of the logging can be done here if needed, or moved to a separate method.
        if self.logger:
            self.log_all_seed_states(epoch)

    def log_all_seed_states(self, epoch: int):
        """Logs the current state of all seeds."""
        if not self.logger:
            logging.warning(f"No logger available to log seed states at epoch {epoch}")
            return

        logging.info(f"Logging seed states for epoch {epoch}")
        from .events import SeedInfo
        seed_infos = []
        with self.seed_manager.lock:
            logging.info(f"Found {len(self.seed_manager.seeds)} seeds to log")
            for seed_id, seed_info in self.seed_manager.seeds.items():
                module = seed_info["module"]
                try:
                    state_enum = SeedState(module.state)
                    logging.info(f"Seed {seed_id}: state={state_enum.value}")
                except (ValueError, KeyError) as e:
                    logging.error(f"Failed to read state for seed {seed_id}: {e}, module.state={getattr(module, 'state', 'MISSING')}")
                    state_enum = SeedState.DORMANT # Default fallback

                layer_idx, seed_idx = seed_id
                seed_infos.append(SeedInfo(
                    id=seed_id,
                    state=state_enum,
                    layer=layer_idx,
                    index_in_layer=seed_idx,
                    metrics={
                        "alpha": module.alpha,
                        "grad_norm": 0.0, # Placeholder, consider adding get_gradient_norm back if needed
                        "patience": seed_info.get("val_patience_counter", 0),
                        "current_loss": seed_info.get("current_loss", 0.0),
                    },
                ))
        
        logging.info(f"Calling logger.log_seed_state_update with {len(seed_infos)} seeds")
        self.logger.log_seed_state_update(epoch, seed_infos)

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
                self.plateau = 0 # Reset plateau counter only on successful germination
                
                # IMMEDIATELY update UI to show GERMINATED state before it gets promoted
                if self.logger:
                    self.log_all_seed_states(epoch)
                
                # Record germination in monitoring
                if monitor:
                    monitor.record_germination()
                return True  # Signal germination occurred
        return False

    def _choose_graft_strategy(self, seed_id: tuple[int, int]) -> str:
        """Dynamically selects a grafting strategy based on real-time telemetry."""
        info = self.seed_manager.seeds[seed_id]
        module = info["module"]
        
        # Gather real-time telemetry
        health_signal = module.get_health_signal()
        drift = info.get("telemetry", {}).get("drift", 0.0)
        current_loss = info.get("current_loss", float('inf'))
        baseline_loss = info.get("baseline_loss", float('inf'))
        grad_norm = info.get("avg_grad_norm", 0.0)
        
        cfg = self.graft_cfg
        
        # Dynamic strategy selection based on current conditions
        if drift > cfg.high_drift_threshold:
            # High drift - need controlled, adaptive grafting
            return "DRIFT_CONTROLLED"
        elif health_signal < cfg.low_health_threshold:
            # Severe bottleneck - performance-driven grafting
            return "PERFORMANCE_LINKED"
        elif (current_loss < baseline_loss * cfg.performance_loss_factor and 
              cfg.grad_norm_lower < grad_norm < cfg.grad_norm_upper):
            # Good performance and stable gradients - use gradient-aware grafting
            return "GRAD_NORM_GATED"
        else:
            # Default fallback
            return "FIXED_RAMP"

    def _select_seed(self) -> Optional[tuple[int, int]]:
        """Selects the best DORMANT seed to germinate based on the worst health signal."""
        candidate_id: Optional[tuple[int, int]] = None
        worst_signal = float("inf")  # Start with worst possible value

        with self.seed_manager.lock:
            for sid, info in self.seed_manager.seeds.items():
                # Only consider dormant seeds that are not under a culling embargo
                if info.get("state") == SeedState.DORMANT.value:
                    # Calculate health signal (lower = worse bottleneck)
                    # This requires the module to have a get_health_signal method.
                    if hasattr(info["module"], "get_health_signal"):
                        signal = info["module"].get_health_signal()
                        if signal < worst_signal:
                            worst_signal = signal
                            candidate_id = sid
                    else:
                        # Fallback if method doesn't exist, though it should
                        logging.warning(f"Seed {sid} has no get_health_signal method.")

        if candidate_id:
            logging.info(f"Selected seed {candidate_id} for germination with health signal: {worst_signal:.4f}")
        else:
            logging.info("No suitable dormant seeds available for germination.")

        return candidate_id

    def _is_training_slot_occupied(self) -> bool:
        """
        Check if the training slot is currently occupied by any seed.
        
        The training slot is considered occupied if any seed is in a state that
        requires exclusive access to training resources:
        - TRAINING: Actively training
        - GRAFTING: Integrating into the network
        - STABILIZATION: Stage 1 validation
        - FINE_TUNING: Stage 2 validation
        
        Returns:
            bool: True if training slot is occupied, False if available
        """
        occupied_states = {
            SeedState.TRAINING.value,
            SeedState.GRAFTING.value, 
            SeedState.STABILIZATION.value,
            SeedState.FINE_TUNING.value
        }
        
        with self.seed_manager.lock:
            for seed_info in self.seed_manager.seeds.values():
                if seed_info.get("state") in occupied_states:
                    return True
        return False

    def is_any_seed_training(self) -> bool:
        """Check if any seed is currently in TRAINING state."""
        with self.seed_manager.lock:
            for seed_info in self.seed_manager.seeds.values():
                if seed_info.get("state") == SeedState.TRAINING.value:
                    return True
        return False

    def start_training_next_seed(self, epoch: int | None = None) -> bool:
        """Public method to advance the training queue. Called from the training loop."""
        return self._advance_training_queue(epoch)

    def _advance_training_queue(self, epoch: int | None = None) -> bool:
        """Start training the next germinated seed if no seed is currently training.
        
        Seeds can only start training if they were germinated in a PREVIOUS epoch
        to ensure the UI shows the GERMINATED state for at least one full epoch.
        """
        if self._is_training_slot_occupied():
            return False  # Training slot is occupied by a seed in an active state
        
        # Check if there are any seeds ready to be promoted
        with self.seed_manager.lock:
            if not self.seed_manager.germinated_queue:
                return False  # No seeds waiting to train
            
            # Peek at the oldest seed in the queue (don't remove yet)
            seed_id = self.seed_manager.germinated_queue[0]
            seed_info = self.seed_manager.seeds.get(seed_id)
            
            if not seed_info or seed_info.get("state") != SeedState.GERMINATED.value:
                # Seed was removed or changed state, remove from queue and try next one
                self.seed_manager.germinated_queue.popleft()
                return self._advance_training_queue(epoch)
            
            # CHECK: Only allow training if seed was germinated in a PREVIOUS epoch
            germination_epoch = seed_info.get('germination_epoch', 0)
            current_epoch = epoch if epoch is not None else 0
            
            if germination_epoch >= current_epoch:
                # Seed was germinated this epoch or later - must wait until next epoch
                return False
            
            # Safe to start training - remove from queue and promote
            seed_id = self.seed_manager.germinated_queue.popleft()
            
            # Start training this seed
            if "module" in seed_info:
                seed_info["module"]._set_state(SeedState.TRAINING, epoch=current_epoch)
                
                # Log the training start
                if self.logger:
                    self.logger.log_seed_event_detailed(
                        epoch=current_epoch,
                        event_type="TRAINING_START", 
                        message=f"Seed L{seed_id[0]}_S{seed_id[1]} started training!",
                        data={"seed_id": f"L{seed_id[0]}_S{seed_id[1]}"},
                    )
                return True
        return False
