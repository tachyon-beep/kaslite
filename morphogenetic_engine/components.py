"""Components module for morphogenetic engine."""

import logging
import weakref
from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch import nn

from morphogenetic_engine.core import SeedManager
from morphogenetic_engine.events import SeedState

if TYPE_CHECKING:
    from morphogenetic_engine.core import BlendingConfig


class SentinelSeed(nn.Module):
    """
    A sentinel seed that monitors activations and can evolve when needed.

    This module starts as dormant, monitors activation patterns, and can be
    germinated to learn adaptive transformations when bottlenecks are detected.
    """

    def __init__(
        self,
        seed_id: tuple[int, int],
        dim: int,
        seed_manager: SeedManager,
        parent_net: "BaseNet",  # Add parent_net reference
        probationary_steps: int = 50,
        shadow_lr: float = 1e-3,
        drift_warn: float = 0.12,
        stability_threshold: float = 0.01,
        improvement_threshold: float = 0.95,
        blend_cfg: "BlendingConfig | None" = None,  # Add blending config
    ):
        super().__init__()

        # Parameter validation
        if dim <= 0:
            raise ValueError(f"Invalid dimension: {dim}. Must be positive.")

        self.seed_id = seed_id
        self.dim = dim
        self.parent_net_ref = weakref.ref(parent_net)  # Use a weak reference to avoid recursion
        self.probationary_steps = probationary_steps
        self.probationary_counter = 0
        self.shadow_lr = shadow_lr
        self.alpha = 0.0
        self.state = SeedState.DORMANT.value
        self.drift_warn = drift_warn
        self.stability_threshold = stability_threshold
        self.improvement_threshold = improvement_threshold
        
        # Store blending configuration
        if blend_cfg is None:
            # Import here to avoid circular imports
            from morphogenetic_engine.core import BlendingConfig
            blend_cfg = BlendingConfig()
        self.blend_cfg = blend_cfg
        
        # Convergence-based training completion
        self.loss_history: list[float] = []
        self.convergence_threshold = 1e-3  # Realistic threshold for neural network training
        self.convergence_window = 20  # Reasonable window for stability assessment

        # Create child network with residual connection
        self.child = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
        )
        self._initialize_as_identity()

        # Optimizer will be created when child is initialized (germinated)
        self.child_optim: torch.optim.Optimizer | None = None
        self.child_loss = nn.MSELoss()

        # Register with seed manager
        self.seed_manager = seed_manager
        self.seed_manager.register_seed(self, seed_id)

    def get_health_signal(self) -> float:
        """
        Calculate a health signal for this seed to determine germination priority.
        
        A lower signal indicates a worse bottleneck (less variance in activations)
        and thus a higher priority for germination.
        
        Returns:
            float: Health signal, where signal ≈ variance. Lower is higher priority.
        """
        with self.seed_manager.lock:
            seed_info = self.seed_manager.seeds.get(self.seed_id, {})
            buffer = seed_info.get("buffer")
            
            # Require a minimum number of data points to make a meaningful decision.
            if not buffer or len(buffer) < 20:
                return float('inf') # Return a high value to deprioritize this seed until it has enough data.
            
            # --- FIX 1: DEFINE buffer_tensor ---
            # Concatenate the list of tensors in the buffer into a single tensor.
            try:
                buffer_tensor = torch.cat(list(buffer), dim=0)
            except (RuntimeError, ValueError) as e:
                 # This can happen if buffers contain tensors of different shapes, indicating an upstream issue.
                 logging.error(f"Error concatenating buffer for seed {self.seed_id}: {e}")
                 return float('inf') # Deprioritize if buffer is corrupt.

            # --- FIX 2: CORRECT THE LOGIC ---
            # The health signal IS the variance. _select_seed will find the minimum.
            # Low variance is a sign of a bottleneck and is what we want to select.
            variance = buffer_tensor.var().item()
            
            # Ensure the signal is a small positive number to avoid issues with zero.
            health_signal = max(variance, 1e-9)
            
            return health_signal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the sentinel seed, with behavior determined by its current state.
        """

        # In DORMANT state, the primary job is to buffer activations for health signal calculation.
        if self.state == SeedState.DORMANT.value:
            self.seed_manager.append_to_buffer(self.seed_id, x)
            return x

        # In several states, the seed should act as a pass-through identity function
        # and not interfere with the parent network's forward pass.
        # TRAINING: The child network trains on its buffer in the background, not in the forward pass.
        # STABILIZATION: The child is validated in the background; the forward pass remains inert.
        if self.state in [
            SeedState.CULLED.value,
            SeedState.GERMINATED.value,
            SeedState.TRAINING.value,
        ]:
            return x

        # For the active states, we compute the child's output.
        child_out = self.child(x)

        # In GRAFTING state, we smoothly interpolate between the original path and the child path.
        if self.state == SeedState.GRAFTING.value:
            output = (1 - self.alpha) * x + self.alpha * child_out
        
        # In STABILIZATION, FINE_TUNING and FOSSILIZED states, the child is fully active as a residual connection.
        elif self.state in [SeedState.STABILIZATION.value, SeedState.FINE_TUNING.value, SeedState.FOSSILIZED.value]:
            output = x + child_out
        else:
            # Default fallback case, should not be reached in normal operation.
            return x

        # Monitor interface drift for any state where the output is being modified.
        if self.state in [SeedState.GRAFTING.value, SeedState.STABILIZATION.value, SeedState.FINE_TUNING.value]:
            with torch.no_grad():
                cos_sim = torch.cosine_similarity(x, output, dim=-1).mean()
                drift = 1.0 - cos_sim.item()
                if self.drift_warn > 0 and drift > self.drift_warn:
                    logging.warning("High drift %.4f at %s (%s)", drift, self.seed_id, self.state)
                self.seed_manager.record_drift(self.seed_id, drift)
                
        return output

    # ------------------------------------------------------------------
    def _set_state(self, new_state: str | SeedState, epoch: int | None = None):
        """Set the seed state, validating against allowed transitions."""

        # Convert string to enum if needed
        if isinstance(new_state, str):
            try:
                new_state = SeedState(new_state)
            except ValueError:
                raise ValueError(f"Invalid state: {new_state}. Must be one of {[s.value for s in SeedState]}")
        
        if self.state == new_state.value:
            return  # redundant transition guard
        
        old_state = self.state
        self.state = new_state.value
        info = self.seed_manager.seeds[self.seed_id]
        info["state"] = new_state.value

        # Use provided epoch or get the last known one for the record.
        current_epoch = epoch if epoch is not None else info.get('last_epoch', 0)
        
        # Specifically record the culling epoch when a seed is culled.
        if new_state == SeedState.CULLED:
            info["culling_epoch"] = current_epoch
        
        # Track when grafting starts for duration calculation
        if new_state == SeedState.GRAFTING:
            info["graft_start_epoch"] = current_epoch
            # Capture initial grafting metrics for dynamic strategies
            info["graft_initial_loss"] = self.validate_on_holdout() if hasattr(self, 'validate_on_holdout') else 0.0
            # Capture current drift measurement if available
            current_drift = info.get("telemetry", {}).get("drift", 0.0)
            info["graft_initial_drift"] = current_drift
        
        # Set up training infrastructure when transitioning TO TRAINING state
        if new_state == SeedState.TRAINING:
            # Create optimizer if not already created
            if self.child_optim is None:
                self.child_optim = torch.optim.Adam(
                    self.child.parameters(), 
                    lr=self.shadow_lr,
                    weight_decay=1e-4  # L2 regularization
                )
            
            # Create train/validation split from buffer data
            self.create_train_val_split()
            
            # Initialize training metrics
            info["training_steps"] = 0
            info["baseline_loss"] = None
            info["current_loss"] = 0.0
            
            logging.info(f"Seed {self.seed_id} training setup complete at epoch {current_epoch}")
            
        self.seed_manager.record_transition(self.seed_id, old_state, new_state.value, epoch=current_epoch)

    def _initialize_as_identity(self):
        """Initialize to near-zero output (identity function)"""
        for m in self.child.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Keep parameters frozen initially
        for p in self.child.parameters():
            p.requires_grad = False

    def initialize_child(self, epoch: int | None = None):
        """Proper initialization when germinating - goes to parking lot (GERMINATED state)"""
        for m in self.child.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Make parameters trainable
        for p in self.child.parameters():
            p.requires_grad = True
        
        # NOW create the optimizer with fresh, trainable parameters and regularization
        self.child_optim = torch.optim.Adam(
            self.child.parameters(), 
            lr=self.shadow_lr, 
            weight_decay=1e-4  # L2 regularization to prevent overfitting
        )
        
        # Capture baseline performance for later evaluation
        self.seed_manager.seeds[self.seed_id]["baseline_loss"] = None
        self.seed_manager.seeds[self.seed_id]["current_loss"] = None
        self.seed_manager.seeds[self.seed_id]["training_steps"] = 0
        
        # Create a local train/val split from the seed's buffer for early stopping
        self.create_train_val_split()

        # Only transition to GERMINATED state (parking lot)
        # The seed will transition to TRAINING when training starts
        self._set_state(SeedState.GERMINATED, epoch=epoch)
        
        # Explicit fallback logging for germination to ensure it appears in timeline
        if self.seed_manager.logger:
            self.seed_manager.logger.log_seed_event_detailed(
                epoch=epoch or 0,
                event_type="GERMINATED",
                message=f"Seed L{self.seed_id[0]}_S{self.seed_id[1]} entered GERMINATED state",
                data={"seed_id": f"L{self.seed_id[0]}_S{self.seed_id[1]}", "epoch": epoch or 0},
            )

    def train_child_step(self, inputs: torch.Tensor, epoch: int | None = None):
        """
        Per-step: Train the child network for one step and record metrics.
        DOES NOT change the state.
        """
        if self.state != SeedState.TRAINING.value or inputs.numel() == 0:
            return
        if self.child_optim is None:
            raise RuntimeError("Optimizer not initialized.")
        inputs = inputs.detach()
        self.child_optim.zero_grad(set_to_none=True)
        outputs = self.child(inputs)
        loss = self.child_loss(outputs, inputs)
        loss.backward()
        
        # Calculate gradient norm for GradNormGatedBlending strategy
        total_grad_norm = 0.0
        param_count = 0
        for param in self.child.parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2)
                total_grad_norm += param_grad_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            avg_grad_norm = (total_grad_norm / param_count) ** 0.5
        else:
            avg_grad_norm = 0.0
        
        self.child_optim.step()
        current_loss = loss.item()
        seed_info = self.seed_manager.seeds[self.seed_id]
        if seed_info["baseline_loss"] is None:
            seed_info["baseline_loss"] = current_loss
        seed_info["current_loss"] = current_loss
        seed_info["training_steps"] += 1
        seed_info["avg_grad_norm"] = avg_grad_norm  # Store for GradNormGatedBlending
        self.loss_history.append(current_loss)
        # All state transition logic has been REMOVED from this method.

    def create_train_val_split(self, split_ratio: float = 0.8):
        """Create train/validation split from seed's data buffer to prevent overfitting."""
        seed_info = self.seed_manager.seeds[self.seed_id]
        buffer = seed_info.get("buffer")
        if not buffer or len(buffer) < 10:
            # Not enough data to create a meaningful split
            seed_info["train_indices"] = None
            seed_info["val_indices"] = None
            seed_info["best_val_loss"] = float('inf')
            seed_info["val_patience_counter"] = 0
            seed_info["val_patience_limit"] = 10 # Reasonable patience
            return

        buffer_data = torch.cat(list(buffer), dim=0)
        buffer_size = len(buffer_data)
        indices = torch.randperm(buffer_size)
        train_size = int(buffer_size * split_ratio)

        seed_info["train_indices"] = indices[:train_size]
        seed_info["val_indices"] = indices[train_size:]
        seed_info["best_val_loss"] = float('inf')
        seed_info["val_patience_counter"] = 0
        seed_info["val_patience_limit"] = 10  # Reasonable patience for stable training

    def validate_on_holdout(self) -> float:
        """Evaluate the child network on its local validation data to check for overfitting."""
        seed_info = self.seed_manager.seeds[self.seed_id]
        val_indices = seed_info.get("val_indices")
        buffer = seed_info.get("buffer")

        # If there's no validation set, return the last known training loss to avoid breaking logic.
        if val_indices is None or buffer is None or len(val_indices) == 0 or len(buffer) == 0:
            return seed_info.get("current_loss", float('inf'))

        buffer_data = torch.cat(list(buffer), dim=0)
        
        # Ensure indices are within the current buffer's bounds
        valid_val_indices = val_indices[val_indices < len(buffer_data)]
        if len(valid_val_indices) == 0:
            return seed_info.get("current_loss", float('inf'))

        # Use a small, random sample from the validation set for efficiency
        val_batch_indices = valid_val_indices[torch.randperm(len(valid_val_indices))[:min(32, len(valid_val_indices))]]
        val_batch = buffer_data[val_batch_indices]
        
        # Use the main evaluation method to get the loss on this validation batch
        return self.evaluate_loss(val_batch)

    def evaluate_loss(self, inputs: torch.Tensor) -> float:
        """Evaluate current loss without training (for validation or stability checks)."""
        if inputs.numel() == 0:
            return 0.0
            
        inputs = inputs.detach()
        with torch.no_grad():
            outputs = self.child(inputs)
            loss = self.child_loss(outputs, inputs)
            current_loss = loss.item()
            
        # Update metrics for validation logic
        seed_info = self.seed_manager.seeds[self.seed_id]
        seed_info["current_loss"] = current_loss
        
        return current_loss

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def update_blending(self, epoch: int | None = None):
        """Update the blending alpha value during blending phase - only once per epoch."""
        if self.state != SeedState.BLENDING.value:
            return
            
        seed_info = self.seed_manager.seeds[self.seed_id]
        
        # Only increment alpha once per epoch
        last_blend_epoch = seed_info.get("last_blend_epoch", -1)
        current_epoch = epoch if epoch is not None else 0
        
        if current_epoch == last_blend_epoch:
            return  # Already updated this epoch
            
        # Record that we've updated this epoch
        seed_info["last_blend_epoch"] = current_epoch
        
        # Use the new strategy system if available
        strategy_name = seed_info.get("blend_strategy")
        if strategy_name:
            # Import here to avoid circular imports
            from .blending import get_strategy
            
            # Create or reuse the strategy instance
            strategy_obj = seed_info.get("blend_strategy_obj")
            if strategy_obj is None:
                strategy_obj = get_strategy(strategy_name, self, self.blend_cfg)
                seed_info["blend_strategy_obj"] = strategy_obj
            
            # Use the strategy to calculate the new alpha
            new_alpha = strategy_obj.update()
            self.alpha = new_alpha
            seed_info["alpha"] = self.alpha
        else:
            # Fallback to old logic for backward compatibility
            self.alpha = min(1.0, self.alpha + 1 / self.blend_cfg.fixed_steps)
            seed_info["alpha"] = self.alpha

    def update_shadowing(self, epoch: int | None = None, inputs: torch.Tensor | None = None):
        """Stage 1 validation - monitor for internal stability during shadowing phase."""
        if self.state != SeedState.SHADOWING.value:
            return

        seed_info = self.seed_manager.seeds[self.seed_id]

        # Initialize shadowing counters if they don't exist
        seed_info.setdefault("stability_history", [])
        seed_info.setdefault("shadowing_steps", 0)
        # Now safely bump the step count
        seed_info["shadowing_steps"] += 1
        
        # Record one more loss into the history
        if inputs is not None:
            current_loss = self.evaluate_loss(inputs)
        else:
            current_loss = seed_info.get("current_loss", 0.0)
        seed_info["stability_history"].append(current_loss)

        # Keep history bounded
        if len(seed_info["stability_history"]) > 20:
            seed_info["stability_history"].pop(0)
            
    def update_probationary(self, epoch: int | None = None, inputs: torch.Tensor | None = None):
        """Stage 2 validation - monitor for systemic impact during probationary period."""
        if self.state == SeedState.PROBATIONARY.value:
            seed_info = self.seed_manager.seeds[self.seed_id]
            
            # Initialize probationary tracking if not present
            if "probationary_steps" not in seed_info:
                seed_info["probationary_steps"] = 0
                seed_info["baseline_performance"] = None
                
            seed_info["probationary_steps"] += 1
            
            # Update loss metrics for final evaluation
            if inputs is not None:
                self.evaluate_loss(inputs)

    def _perform_training_steps(self, seed_info: dict, epoch: int | None):
        """Perform actual training work for the current epoch."""
        train_indices = seed_info.get("train_indices")
        if train_indices is None or len(train_indices) == 0:
            return
            
        # Get training data from buffer
        buffer = seed_info.get("buffer")
        if not buffer or len(buffer) == 0:
            return
            
        buffer_data = torch.cat(list(buffer), dim=0)
        train_data = buffer_data[train_indices]
        
        # Perform multiple training steps this epoch
        batch_size = min(32, len(train_data))
        num_batches = max(1, len(train_data) // batch_size)
        
        for batch_idx in range(min(num_batches, 3)):  # Reduced to 3 batches per epoch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_data))
            batch_data = train_data[start_idx:end_idx]
            
            if len(batch_data) > 0:
                self.train_child_step(batch_data, epoch)

    def _check_training_completion(self, seed_info: dict, epoch: int | None) -> bool:
        """Check if training should be completed based on convergence or early stopping."""
        # Track training epochs (not just steps)
        training_epochs = seed_info.get("training_epochs", 0)
        seed_info["training_epochs"] = training_epochs + 1
        
        # Early Stopping Check
        current_val_loss = self.validate_on_holdout()
        if current_val_loss < seed_info.get("best_val_loss", float('inf')):
            seed_info["best_val_loss"] = current_val_loss
            seed_info["val_patience_counter"] = 0
        else:
            seed_info["val_patience_counter"] += 1

        # Check for early stopping
        val_patience_limit = seed_info.get("val_patience_limit", 25)
        if seed_info.get("val_patience_counter", 0) >= val_patience_limit:
            logging.info(f"Early stopping for {self.seed_id} at epoch {epoch} after {training_epochs} epochs due to validation plateau.")
            return True
            
        # Check for convergence
        if self.check_convergence():
            logging.info(f"Convergence detected for {self.seed_id} at epoch {epoch} after {training_epochs} epochs.")
            return True
            
        return False

    def _handle_training_transition(self, seed_info: dict, epoch: int | None):
        """Handles the transition logic for a seed in the TRAINING state."""
        
        # FIRST: Do actual training work if we have training data
        self._perform_training_steps(seed_info, epoch)
        
        # THEN: Check if training should continue or transition
        if self._check_training_completion(seed_info, epoch):
            self.alpha = 0.0  # Reset alpha for blending
            seed_info["last_blend_epoch"] = -1  # Reset epoch tracking
            self._set_state(SeedState.BLENDING, epoch=epoch)

    def _handle_blending_transition(self, epoch: int | None):
        """Handles the transition logic for a seed in the BLENDING state."""
        
        # FIRST: Do actual blending work (update alpha)
        self.update_blending(epoch)
        
        # THEN: Check if blending is complete based on alpha reaching 1.0
        if self.alpha >= 0.99:
            # Capture final metrics and emit rich blend completed event
            seed_info = self.seed_manager.seeds[self.seed_id]
            strategy_name = seed_info.get("blend_strategy", "FIXED_RAMP")
            
            # Calculate comprehensive blending metrics
            blend_start_epoch = seed_info.get("blend_start_epoch", 0)
            current_epoch = epoch if epoch is not None else 0
            duration_epochs = max(0, current_epoch - blend_start_epoch)
            
            initial_loss = seed_info.get("blend_initial_loss", 0.0)
            final_loss = self.validate_on_holdout() if hasattr(self, 'validate_on_holdout') else 0.0
            initial_drift = seed_info.get("blend_initial_drift", 0.0)
            final_drift = seed_info.get("telemetry", {}).get("drift", 0.0)
            
            if self.seed_manager.logger:
                from .events import EventType, BlendCompletedPayload, LogEvent
                import time
                
                payload = BlendCompletedPayload(
                    seed_id=self.seed_id,
                    epoch=epoch or 0,
                    strategy_name=strategy_name,
                    duration_epochs=duration_epochs,
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    initial_drift=initial_drift,
                    final_drift=final_drift,
                    timestamp=time.time()
                )
                event = LogEvent(EventType.BLEND_COMPLETED, payload)
                self.seed_manager.logger.log_event(epoch or 0, event)
                
                # Record in analytics for Phase 2 dashboard
                try:
                    from .blending_analytics import get_blending_analytics
                    analytics = get_blending_analytics()
                    analytics.record_blend_completed(payload)
                except ImportError:
                    pass  # Analytics not available
                
                # Log detailed completion metrics
                self.seed_manager.logger.log_seed_event_detailed(
                    epoch=epoch or 0,
                    event_type=EventType.BLEND_COMPLETED.value,  # Use EventType enum value
                    message=f"Seed L{self.seed_id[0]}_S{self.seed_id[1]} completed {strategy_name} blending",
                    data={
                        "seed_id": f"L{self.seed_id[0]}_S{self.seed_id[1]}",
                        "strategy_name": strategy_name,
                        "duration_epochs": duration_epochs,
                        "initial_loss": initial_loss,
                        "final_loss": final_loss,
                        "initial_drift": initial_drift,
                        "final_drift": final_drift,
                        "loss_improvement": initial_loss - final_loss,
                        "drift_change": final_drift - initial_drift
                    }
                )
            
            self._set_state(SeedState.SHADOWING, epoch=epoch)

    def _handle_shadowing_transition(self, seed_info: dict, epoch: int | None):
        """Handles the transition logic for a seed in the SHADOWING state."""
        
        # FIRST: Do actual shadowing monitoring work
        buffer = seed_info.get("buffer")
        if buffer and len(buffer) > 0:
            recent_data = list(buffer)[-5:]  # Get last 5 activations
            if recent_data:
                monitoring_data = torch.cat(recent_data, dim=0)
                self.update_shadowing(epoch, monitoring_data)
        else:
            # Update without input data (will use stored loss)
            self.update_shadowing(epoch, None)
        
        # THEN: Check if shadowing period is complete based on stability
        stability_history = seed_info.get("stability_history", [])
        if len(stability_history) < 10:  # Need some history to assess
            return

        recent_losses = stability_history[-10:]
        loss_variance = torch.tensor(recent_losses).var().item()
        
        # Only advance if loss is stable (low variance)
        if loss_variance < self.stability_threshold:
            self._set_state(SeedState.PROBATIONARY, epoch=epoch)

    def _handle_probationary_transition(self, seed_info: dict, epoch: int | None):
        """Handles the transition logic for a seed in the PROBATIONARY state."""
        
        # FIRST: Do actual probationary monitoring work
        # Get some recent buffer data for monitoring
        buffer = seed_info.get("buffer")
        if buffer and len(buffer) > 0:
            recent_data = list(buffer)[-5:]  # Get last 5 activations
            if recent_data:
                monitoring_data = torch.cat(recent_data, dim=0)
                self.update_probationary(epoch, monitoring_data)
        else:
            # Update without input data
            self.update_probationary(epoch, None)
        
        # THEN: Check if probationary period is complete
        if seed_info.get("probationary_steps", 0) >= self.probationary_steps:
            self._evaluate_and_complete(epoch)

    def assess_and_transition_state(self, epoch: int | None = None):
        """
        Checks if the seed should transition to a new state.
        This method contains all primary lifecycle transition logic and should be
        called once per epoch, after per-step updates are complete.
        It ensures at most one transition occurs per invocation.
        """
        seed_info = self.seed_manager.seeds[self.seed_id]

        match self.state:
            case SeedState.TRAINING.value:
                self._handle_training_transition(seed_info, epoch)
            case SeedState.BLENDING.value:
                self._handle_blending_transition(epoch)
            case SeedState.SHADOWING.value:
                self._handle_shadowing_transition(seed_info, epoch)
            case SeedState.PROBATIONARY.value:
                self._handle_probationary_transition(seed_info, epoch)
            case _:
                # No transitions for other states (DORMANT, FOSSILIZED, CULLED)
                pass

    def check_convergence(self) -> bool:
        """Checks if the training loss has converged based on actual learning patterns."""
        if len(self.loss_history) < self.convergence_window:
            return False
        
        recent_losses = self.loss_history[-self.convergence_window:]
        loss_tensor = torch.tensor(recent_losses)
        
        # Check for very low variance (stability)
        variance = loss_tensor.var()
        if variance > self.convergence_threshold:
            return False
        
        # Ensure the average loss is reasonable (not stuck at high values)
        mean_loss = loss_tensor.mean()
        if mean_loss > 1.0:  # Lowered threshold for better convergence detection
            return False
        
        # Check that we're not oscillating wildly
        max_loss = loss_tensor.max()
        min_loss = loss_tensor.min()
        loss_range = max_loss - min_loss
        
        # Loss should be stable (small range relative to mean)
        relative_stability = loss_range / (mean_loss + 1e-8)
        
        return relative_stability < 0.1  # Loss range should be < 10% of mean

    def _cull_seed(self, epoch: int | None = None):
        """Marks the seed as CULLED, records the epoch, and freezes the child."""
        self._set_state(SeedState.CULLED, epoch=epoch)
        
        # Freeze parameters to save computation
        for p in self.child.parameters():
            p.requires_grad = False
        
        # Log the culling event for the embargo mechanism
        self.seed_manager.seeds[self.seed_id]["culling_epoch"] = epoch
        
        logging.info(f"Seed {self.seed_id} culled at epoch {epoch}. Embargo started.")

    def _evaluate_and_complete(self, epoch: int | None = None):
        """Final evaluation to decide between Fossilization and Culling."""
        seed_info = self.seed_manager.seeds[self.seed_id]
        
        # Use the holdout validation loss for the final decision
        final_val_loss = self.validate_on_holdout()
        baseline_loss = seed_info.get("baseline_loss")

        # Ensure baseline_loss is valid before comparison
        if baseline_loss is None or not isinstance(baseline_loss, (int, float)):
            logging.warning(f"Seed {self.seed_id}: No valid baseline loss recorded. Defaulting to CULL.")
            self._cull_seed(epoch=epoch)
            return

        # Fossilize if the new component provides a significant improvement
        # Use a relative improvement threshold
        if final_val_loss < baseline_loss * self.improvement_threshold:
            self._set_state(SeedState.FOSSILIZED, epoch=epoch)
            
            # Explicit fallback logging for fossilization to ensure it appears in timeline
            if self.seed_manager.logger:
                self.seed_manager.logger.log_seed_event_detailed(
                    epoch=epoch or 0,
                    event_type="FOSSILIZED",
                    message=f"Seed L{self.seed_id[0]}_S{self.seed_id[1]} fossilized!",
                    data={"seed_id": f"L{self.seed_id[0]}_S{self.seed_id[1]}", "epoch": epoch or 0},
                )
        else:
            self._cull_seed(epoch=epoch)


class GraftedChild(nn.Module):
    """
    A wrapper for a trained child network that is permanently grafted.
    This module ensures the residual connection (x + child_out) is preserved
    after the original SentinelSeed is removed from the graph.
    """
    def __init__(self, child_network: nn.Module):
        super().__init__()
        self.child_network = child_network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the grafted child network as a residual connection."""
        return x + self.child_network(x)


class BaseNet(nn.Module):
    """
    Configurable trunk network with dynamic number of hidden layers and seeds.
    Each hidden layer can have multiple sentinel seeds for adaptive capacity.
    Default configuration creates 8 layers with 1 seed each for backward compatibility.
    Multiple seeds per layer are averaged to provide ensemble-like behavior.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        *,
        seed_manager: SeedManager,
        input_dim: int = 2,
        output_dim: int = 2,
        num_layers: int = 8,
        seeds_per_layer: int = 1,
        probationary_steps: int = 50,
        shadow_lr: float = 1e-3,
        drift_warn: float = 0.1,
        blend_cfg: "BlendingConfig | None" = None,  # Add blending config
    ):
        super().__init__()

        # Parameter validation - fail fast with clear error messages
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if seeds_per_layer <= 0:
            raise ValueError("seeds_per_layer must be positive")

        self.num_layers = num_layers
        self.seeds_per_layer = seeds_per_layer
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Store blending configuration
        if blend_cfg is None:
            # Import here to avoid circular imports
            from morphogenetic_engine.core import BlendingConfig
            blend_cfg = BlendingConfig()
        self.blend_cfg = blend_cfg

        # Store seed creation parameters for replacing culled seeds
        self.seed_manager_ref = seed_manager
        self.seed_probationary_steps = probationary_steps
        self.seed_shadow_lr = shadow_lr
        self.seed_drift_warn = drift_warn

        # Create input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_activation = nn.ReLU()

        # Dynamically create hidden layers and seeds
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        # Use a flat list for all seeds and organize by layer in forward()
        self.all_seeds = nn.ModuleList()

        for i in range(num_layers):
            # Hidden layer
            layer = nn.Linear(hidden_dim, hidden_dim)
            activation = nn.ReLU()

            # Multiple seeds per layer
            for j in range(seeds_per_layer):
                seed = SentinelSeed(
                    (i, j),  # Use a tuple (layer_idx, seed_idx) as the ID
                    hidden_dim,
                    seed_manager,
                    parent_net=self,  # Pass a reference to the parent network
                    probationary_steps=probationary_steps,
                    shadow_lr=shadow_lr,
                    drift_warn=drift_warn,
                    blend_cfg=self.blend_cfg,  # Pass the blending configuration
                )
                self.all_seeds.append(seed)

            self.layers.append(layer)
            self.activations.append(activation)

        # Output layer
        self.out = nn.Linear(hidden_dim, output_dim)

    def replace_seed(self, seed_id: tuple[int, int], new_module: nn.Module):
        """
        Physically replaces a seed in the ModuleList with a new module.
        """
        layer_idx, seed_idx_in_layer = seed_id
        # Calculate the flat index in the all_seeds list
        flat_idx = layer_idx * self.seeds_per_layer + seed_idx_in_layer

        if flat_idx < 0 or flat_idx >= len(self.all_seeds):
            logging.error(f"Cannot replace seed: index {flat_idx} is out of bounds.")
            return

        # Perform the physical replacement in the ModuleList
        self.all_seeds[flat_idx] = new_module
        logging.info(f"Seed {seed_id} physically replaced in the network graph.")

    def reset_culled_seed(self, seed_id: tuple[int, int]):
        """
        Replaces a culled seed at a specific ID with a new, dormant one.
        """
        layer_idx, seed_idx_in_layer = seed_id
        flat_idx = layer_idx * self.seeds_per_layer + seed_idx_in_layer

        if flat_idx < 0 or flat_idx >= len(self.all_seeds):
            logging.error(f"Cannot reset seed: index {flat_idx} for ID {seed_id} is out of bounds.")
            return

        # Create a new seed with the original configuration
        new_seed = SentinelSeed(
            seed_id=seed_id,
            dim=self.hidden_dim,
            seed_manager=self.seed_manager_ref,
            parent_net=self,
            probationary_steps=self.seed_probationary_steps,
            shadow_lr=self.seed_shadow_lr,
            drift_warn=self.seed_drift_warn,
            blend_cfg=self.blend_cfg,  # Pass the blending configuration
        )

        # The old seed module is replaced in the network's ModuleList.
        # The new seed's __init__ will handle re-registering with the SeedManager,
        # overwriting the old entry for that seed_id.
        self.all_seeds[flat_idx] = new_seed
        logging.info(f"Culled seed {seed_id} has been replaced with a new dormant seed.")

    def save_grafted_model(self, path: str = "grafted_model.pth"):
        """Saves the state dictionary of the modified network."""
        logging.info(f"Saving grafted model state_dict to {path}")
        torch.save(self.state_dict(), path)

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        """Freeze every parameter that doesn’t belong to a seed module."""
        for name, p in self.named_parameters():
            if "seed" not in name:
                p.requires_grad = False

    # ------------------------------------------------------------------
    def get_total_seeds(self) -> int:
        """Get total number of seeds in the network."""
        return len(self.all_seeds)

    def get_all_seeds(self) -> list:
        """Get a flat list of all seeds for compatibility."""
        return list(self.all_seeds)

    def get_seeds_for_layer(self, layer_idx: int) -> list:
        """Get all seeds for a specific layer."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")

        start_idx = layer_idx * self.seeds_per_layer
        end_idx = start_idx + self.seeds_per_layer
        return list(self.all_seeds[start_idx:end_idx])

    @property
    def seeds(self):
        """Backward compatibility property - returns all seeds."""
        return self.all_seeds

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the complete network."""
        # Input layer
        x = self.input_activation(self.input_layer(x))

        # Dynamic hidden layers with multiple seeds per layer
        for i in range(self.num_layers):
            # Apply linear layer and activation
            x = self.activations[i](self.layers[i](x))

            # Apply seeds for this layer
            layer_seeds = self.get_seeds_for_layer(i)

            if self.seeds_per_layer == 1:
                # Single seed case (backward compatible)
                x = layer_seeds[0](x)
            else:
                # Multiple seeds case - apply all and average
                seed_outputs = []
                for seed in layer_seeds:
                    seed_output = seed(x)
                    seed_outputs.append(seed_output)

                # Average the outputs from all seeds in this layer
                x = torch.stack(seed_outputs, dim=0).mean(dim=0)

        return self.out(x)

    # ------------------------------------------------------------------
