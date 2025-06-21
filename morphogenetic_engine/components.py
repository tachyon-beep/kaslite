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
    from morphogenetic_engine.core import GraftingConfig


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
        shadow_lr: float = 1e-3,
        drift_warn: float = 0.12,
        stability_threshold: float = 0.01,
        improvement_threshold: float = 0.95,
        graft_cfg: "GraftingConfig | None" = None,  # Add grafting config
    ):
        super().__init__()

        # Parameter validation
        if dim <= 0:
            raise ValueError(f"Invalid dimension: {dim}. Must be positive.")

        self.seed_id = seed_id
        self.dim = dim
        self.parent_net_ref = weakref.ref(parent_net)  # Use a weak reference to avoid recursion
        self.shadow_lr = shadow_lr
        self.alpha = 0.0
        self.state = SeedState.DORMANT.value
        self.drift_warn = drift_warn
        self.stability_threshold = stability_threshold
        self.improvement_threshold = improvement_threshold
        
        # Store grafting configuration
        if graft_cfg is None:
            # Import here to avoid circular imports
            from morphogenetic_engine.core import GraftingConfig
            graft_cfg = GraftingConfig()
        self.graft_cfg = graft_cfg
        
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
        
        # Task projection layer for fine-tuning (created when needed)
        self.task_projection_layer: torch.nn.Module | None = None
        self.task_loss_criterion: torch.nn.Module | None = None
        self.finetune_buffer_activations: list[torch.Tensor] = []
        self.finetune_buffer_labels: list[torch.Tensor] = []

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

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through the sentinel seed, with behavior determined by its current state.
        """

        # In DORMANT state, the primary job is to buffer activations for health signal calculation.
        if self.state == SeedState.DORMANT.value:
            self.seed_manager.append_to_buffer(self.seed_id, x)
            return x

        # If labels are provided, buffer them for fine-tuning.
        if y is not None and self.state in [SeedState.STABILIZATION.value, SeedState.FINE_TUNING.value]:
            self.append_to_buffer_with_labels(x, y)

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
    def _initialize_state_metrics(self, new_state: SeedState, epoch: int | None):
        """Helper to initialize metrics and settings for a new state."""
        info = self.seed_manager.seeds[self.seed_id]
        current_epoch = epoch if epoch is not None else info.get('last_epoch', 0)

        if new_state == SeedState.CULLED:
            info["culling_epoch"] = current_epoch
        elif new_state == SeedState.GRAFTING:
            info["graft_start_epoch"] = current_epoch
            info["graft_initial_loss"] = self.validate_on_holdout() if hasattr(self, 'validate_on_holdout') else 0.0
            info["graft_initial_drift"] = info.get("telemetry", {}).get("drift", 0.0)
        elif new_state == SeedState.STABILIZATION:
            info["stabilization_start_epoch"] = current_epoch
            info["stabilization_epochs_remaining"] = self.graft_cfg.stabilization_epochs
        elif new_state == SeedState.TRAINING:
            if self.child_optim is None:
                self.child_optim = torch.optim.Adam(
                    self.child.parameters(), 
                    lr=self.shadow_lr,
                    weight_decay=1e-4
                )
            self.create_train_val_split()
            info["training_steps"] = 0
            info["baseline_loss"] = None
            info["current_loss"] = 0.0
            logging.info(f"Seed {self.seed_id} training setup complete at epoch {current_epoch}")
        elif new_state == SeedState.FINE_TUNING:
            info["fine_tuning_start_epoch"] = current_epoch
            info["fine_tuning_steps"] = 0
            info["best_task_loss"] = float('inf')
            info["task_patience_counter"] = 0
            info["task_loss_history"] = []
            for param in self.child.parameters():
                param.requires_grad = True
            parent_net = self.parent_net_ref()
            if parent_net:
                self.setup_task_projection(parent_net.output_dim, nn.CrossEntropyLoss())

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
        self.seed_manager.seeds[self.seed_id]["state"] = new_state.value

        # Initialize metrics for the new state
        self._initialize_state_metrics(new_state, epoch)

        current_epoch = epoch if epoch is not None else self.seed_manager.seeds[self.seed_id].get('last_epoch', 0)
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
        
        # Calculate gradient norm for GradNormGatedGrafting strategy
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
        seed_info["avg_grad_norm"] = avg_grad_norm  # Store for GradNormGatedGrafting
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
    # Fine-Tuning Infrastructure (Phase 2.3)
    # ------------------------------------------------------------------

    def setup_task_projection(self, num_classes: int, task_loss_criterion: nn.Module):
        """Initializes the task-specific projection layer and loss function for fine-tuning."""
        if num_classes <= 0:
            raise ValueError("Number of classes for task projection must be positive.")
        
        self.task_projection_layer = nn.Linear(self.dim, num_classes)
        self.task_loss_criterion = task_loss_criterion
        
        # Add new parameters to the existing optimizer
        if self.child_optim and self.task_projection_layer:
            # Move new layer to the same device as the child network
            device = next(self.child.parameters()).device
            self.task_projection_layer.to(device)
            
            self.child_optim.add_param_group(
                {'params': self.task_projection_layer.parameters()}
            )
            logging.info(f"Seed {self.seed_id} task projection layer created and optimizer updated.")
        else:
            # This path is a fallback, optimizer should exist from TRAINING state
            params = list(self.child.parameters())
            if self.task_projection_layer:
                params.extend(list(self.task_projection_layer.parameters()))
            self.child_optim = torch.optim.Adam(params, lr=self.shadow_lr, weight_decay=1e-4)
            logging.warning(f"Seed {self.seed_id} optimizer re-created for fine-tuning.")

    def append_to_buffer_with_labels(self, inputs: torch.Tensor, labels: torch.Tensor):
        """Appends activations and corresponding labels to local buffers for fine-tuning."""
        self.finetune_buffer_activations.append(inputs.detach().cpu())
        self.finetune_buffer_labels.append(labels.detach().cpu())

        # Keep buffer size bounded to prevent memory issues
        max_buffer_size = 256  # Configurable: number of batches
        while len(self.finetune_buffer_activations) > max_buffer_size:
            self.finetune_buffer_activations.pop(0)
            self.finetune_buffer_labels.pop(0)

    def get_training_batch_with_labels(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Gets a random batch of activations and labels from the fine-tuning buffer."""
        if not self.finetune_buffer_activations or not self.finetune_buffer_labels:
            return None
        
        # Concatenate all buffered tensors
        try:
            all_activations = torch.cat(self.finetune_buffer_activations, dim=0)
            all_labels = torch.cat(self.finetune_buffer_labels, dim=0)
        except RuntimeError:
            logging.error(f"Seed {self.seed_id}: Could not concatenate fine-tuning buffers. Clearing.")
            self.finetune_buffer_activations.clear()
            self.finetune_buffer_labels.clear()
            return None

        if len(all_activations) == 0:
            return None

        # Create a random permutation and select a batch
        indices = torch.randperm(len(all_activations))[:batch_size]
        return all_activations[indices], all_labels[indices]

    def perform_fine_tuning_step(self, epoch: int | None = None):
        """
        Performs one training step using task loss.
        This is called by the main training loop for seeds in the FINE_TUNING state.
        """
        if self.state != SeedState.FINE_TUNING.value:
            return

        if not self.child_optim or not self.task_projection_layer or not self.task_loss_criterion:
            logging.warning(f"Seed {self.seed_id} cannot fine-tune: missing optimizer, projection, or criterion.")
            return

        batch = self.get_training_batch_with_labels(batch_size=32)
        if batch is None:
            return # Not enough data to train yet

        inputs, labels = batch
        
        # Move data to the correct device (assuming child is on one device)
        device = next(self.child.parameters()).device
        inputs, labels = inputs.to(device), labels.to(device)

        self.child_optim.zero_grad(set_to_none=True)
        
        # Forward pass through child and projection layer
        child_output = self.child(inputs)
        task_logits = self.task_projection_layer(child_output)
        
        loss = self.task_loss_criterion(task_logits, labels)
        loss.backward()
        self.child_optim.step()

        # Record metrics for this fine-tuning step
        task_loss = loss.item()
        seed_info = self.seed_manager.seeds[self.seed_id]
        seed_info.setdefault("task_loss_history", []).append(task_loss)
        seed_info["fine_tuning_steps"] = seed_info.get("fine_tuning_steps", 0) + 1

    def evaluate_task_performance(self, data_loader) -> float:
        """Evaluates the network's performance on the main task, with this seed active."""
        parent_net = self.parent_net_ref()
        if not parent_net:
            return float('inf')

        device = next(parent_net.parameters()).device
        parent_net.eval() # Set the whole network to evaluation mode
        total_loss = 0
        total_count = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # We need to get the output from the final layer of the parent network
                outputs = parent_net(inputs)
                
                # Assuming the parent network's loss is what we measure
                # This requires the parent to have a loss function.
                if hasattr(parent_net, 'loss_function'):
                    loss = parent_net.loss_function(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)
                    total_count += inputs.size(0)

        parent_net.train() # Return to training mode
        return total_loss / total_count if total_count > 0 else float('inf')

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def update_grafting(self, epoch: int | None = None):
        """Update the grafting alpha value during grafting phase - only once per epoch."""
        if self.state != SeedState.GRAFTING.value:
            return
            
        seed_info = self.seed_manager.seeds[self.seed_id]
        
        # Only increment alpha once per epoch
        last_graft_epoch = seed_info.get("last_graft_epoch", -1)
        current_epoch = epoch if epoch is not None else 0
        
        if current_epoch == last_graft_epoch:
            return  # Already updated this epoch
            
        # Record that we've updated this epoch
        seed_info["last_graft_epoch"] = current_epoch
        
        # Use the new strategy system if available
        strategy_name = seed_info.get("graft_strategy")
        if strategy_name:
            # Import here to avoid circular imports
            from .grafting import get_strategy
            
            # Create or reuse the strategy instance
            strategy_obj = seed_info.get("graft_strategy_obj")
            if strategy_obj is None:
                strategy_obj = get_strategy(strategy_name, self, self.graft_cfg)
                seed_info["graft_strategy_obj"] = strategy_obj
            
            # Use the strategy to calculate the new alpha
            new_alpha = strategy_obj.update()
            self.alpha = new_alpha
            seed_info["alpha"] = self.alpha
        else:
            # Fallback to old logic for backward compatibility
            self.alpha = min(1.0, self.alpha + 1 / self.graft_cfg.fixed_steps)
            seed_info["alpha"] = self.alpha

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
            self.alpha = 0.0  # Reset alpha for grafting
            seed_info["last_graft_epoch"] = -1  # Reset epoch tracking
            self._set_state(SeedState.GRAFTING, epoch=epoch)

    def _handle_grafting_transition(self, epoch: int | None):
        """Handles the transition logic for a seed in the GRAFTING state."""
        
        # FIRST: Do actual grafting work (update alpha)
        self.update_grafting(epoch)
        
        # THEN: Check if grafting is complete based on alpha reaching 1.0
        if self.alpha >= 0.99:
            # Capture final metrics and emit rich graft completed event
            seed_info = self.seed_manager.seeds[self.seed_id]
            strategy_name = seed_info.get("graft_strategy", "FIXED_RAMP")
            
            # Calculate comprehensive grafting metrics
            graft_start_epoch = seed_info.get("graft_start_epoch", 0)
            current_epoch = epoch if epoch is not None else 0
            duration_epochs = max(0, current_epoch - graft_start_epoch)
            
            initial_loss = seed_info.get("graft_initial_loss", 0.0)
            final_loss = self.validate_on_holdout() if hasattr(self, 'validate_on_holdout') else 0.0
            initial_drift = seed_info.get("graft_initial_drift", 0.0)
            final_drift = seed_info.get("telemetry", {}).get("drift", 0.0)
            
            if self.seed_manager.logger:
                from .events import EventType, GraftCompletedPayload, LogEvent
                import time
                
                payload = GraftCompletedPayload(
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
                event = LogEvent(EventType.GRAFT_COMPLETED, payload)
                self.seed_manager.logger.log_event(epoch or 0, event)
                
                # Record in analytics for Phase 2 dashboard
                try:
                    from .grafting_analytics import get_grafting_analytics
                    analytics = get_grafting_analytics()
                    analytics.record_graft_completed(payload)
                except ImportError:
                    pass  # Analytics not available
                
                # Log detailed completion metrics
                self.seed_manager.logger.log_seed_event_detailed(
                    epoch=epoch or 0,
                    event_type=EventType.GRAFT_COMPLETED.value,  # Use EventType enum value
                    message=f"Seed L{self.seed_id[0]}_S{self.seed_id[1]} completed {strategy_name} grafting",
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
            
            self._set_state(SeedState.STABILIZATION, epoch=epoch)

    def _handle_stabilization_transition(self, seed_info: dict, epoch: int | None):
        """Handles the transition logic for a seed in the STABILIZATION state."""
        
        # Initialize stabilization tracking if not present
        if "stabilization_epochs_remaining" not in seed_info:
            seed_info["stabilization_epochs_remaining"] = self.graft_cfg.stabilization_epochs
            # Lock alpha at 1.0 and freeze child parameters during stabilization
            self.alpha = 1.0
            seed_info["alpha"] = self.alpha
            for param in self.child.parameters():
                param.requires_grad = False
                
        # Decrement remaining stabilization epochs
        remaining = seed_info.get("stabilization_epochs_remaining", 0) - 1
        seed_info["stabilization_epochs_remaining"] = remaining
        
        # Check if stabilization period is complete
        if remaining <= 0:
            # Before transitioning, capture baseline task performance.
            # This requires access to a validation loader.
            parent_net = self.parent_net_ref()
            if parent_net and hasattr(parent_net, 'val_loader'):
                baseline = self.evaluate_task_performance(parent_net.val_loader)
                seed_info["task_performance_baseline"] = baseline
                logging.info(f"Seed {self.seed_id} captured baseline task performance: {baseline:.4f}")
            else:
                seed_info["task_performance_baseline"] = float('inf') # Fallback
                logging.warning(f"Seed {self.seed_id}: Could not find validation loader to capture task baseline.")

            # Re-enable child parameter training for fine-tuning
            for param in self.child.parameters():
                param.requires_grad = True
            
            # Transition to FINE_TUNING state
            self._set_state(SeedState.FINE_TUNING, epoch=epoch)

    def _handle_fine_tuning_transition(self, seed_info: dict, epoch: int | None):
        """Handles the transition logic for a seed in the FINE_TUNING state."""
        
        # The actual training work is done by the training loop calling `perform_fine_tuning_step`.
        # This handler's job is just to check for completion based on local task loss stability.
        
        task_loss_history = seed_info.get("task_loss_history", [])
        if len(task_loss_history) > 0:  # Check if any fine-tuning has happened
            # Use patience to check for plateau
            patience_limit = seed_info.get("task_patience_limit", 20)
            max_steps = 200  # Max fine-tuning steps to prevent running forever

            # Early stopping logic based on local task loss
            current_task_loss = task_loss_history[-1]
            if current_task_loss < seed_info.get("best_task_loss", float('inf')):
                seed_info["best_task_loss"] = current_task_loss
                seed_info["task_patience_counter"] = 0
            else:
                seed_info["task_patience_counter"] = seed_info.get("task_patience_counter", 0) + 1

            if (seed_info.get("task_patience_counter", 0) >= patience_limit or 
                seed_info.get("fine_tuning_steps", 0) >= max_steps):
                logging.info(f"Seed {self.seed_id} completing fine-tuning phase. Evaluating...")
                self._evaluate_fine_tuning_and_complete(epoch)
    
    def _evaluate_fine_tuning_and_complete(self, epoch: int | None = None):
        """Final evaluation after fine-tuning to decide between FOSSILIZED and CULLED."""
        seed_info = self.seed_manager.seeds[self.seed_id]
        task_baseline = seed_info.get("task_performance_baseline", float('inf'))

        # Trigger a final evaluation of the whole network's task performance
        parent_net = self.parent_net_ref()
        if parent_net and hasattr(parent_net, 'val_loader'):
            final_task_loss = self.evaluate_task_performance(parent_net.val_loader)
        else:
            final_task_loss = float('inf') # Cannot evaluate
            logging.warning(f"Seed {self.seed_id}: Could not find validation loader to evaluate final task performance.")

        logging.info(f"Seed {self.seed_id} fine-tuning evaluation: Baseline Task Loss={task_baseline:.4f}, Final Task Loss={final_task_loss:.4f}")

        # Compare final global task loss with the baseline global task loss
        improvement_threshold = 0.95 # Require 5% improvement
        if final_task_loss < task_baseline * improvement_threshold:
            self._set_state(SeedState.FOSSILIZED, epoch=epoch)
            logging.info("Seed %s fossilized - task performance improved.", self.seed_id)
        else:
            self._cull_seed(epoch=epoch)
            logging.info("Seed %s culled - insufficient task performance improvement.", self.seed_id)

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
            case SeedState.GRAFTING.value:
                self._handle_grafting_transition(epoch)
            case SeedState.STABILIZATION.value:
                self._handle_stabilization_transition(seed_info, epoch)
            case SeedState.FINE_TUNING.value:
                self._handle_fine_tuning_transition(seed_info, epoch)
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
        shadow_lr: float = 1e-3,
        drift_warn: float = 0.1,
        graft_cfg: "GraftingConfig | None" = None,  # Add grafting config
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

        # Store grafting configuration
        if graft_cfg is None:
            # Import here to avoid circular imports
            from morphogenetic_engine.core import GraftingConfig
            graft_cfg = GraftingConfig()
        self.graft_cfg = graft_cfg

        # Store seed creation parameters for replacing culled seeds
        self.seed_manager_ref = seed_manager
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
                    shadow_lr=shadow_lr,
                    drift_warn=drift_warn,
                    graft_cfg=self.graft_cfg,  # Pass the grafting configuration
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
            shadow_lr=self.seed_shadow_lr,
            drift_warn=self.seed_drift_warn,
            graft_cfg=self.graft_cfg,  # Pass the grafting configuration
        )

        # The old seed is still in the seed_manager, but the new one will be registered
        # when the old seed is garbage collected.
        # This ensures we don't disrupt the network graph or lose references unexpectedly.
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

    def get_seeds_in_state(self, state: SeedState) -> list:
        """Get all seeds currently in a specific state."""
        return [seed for seed in self.all_seeds if hasattr(seed, 'state') and seed.state == state.value]

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
    def _apply_layer_seeds(self, x: torch.Tensor, y: torch.Tensor | None, layer_idx: int) -> torch.Tensor:
        """Helper function to apply all seeds for a given layer."""
        layer_seeds = self.get_seeds_for_layer(layer_idx)

        if not layer_seeds:
            return x

        # Define a helper to call a seed with optional labels
        def call_seed(seed, data, labels):
            if isinstance(seed, SentinelSeed):
                return seed(data, labels)
            return seed(data)

        if self.seeds_per_layer == 1:
            return call_seed(layer_seeds[0], x, y)
        else:
            seed_outputs = [call_seed(seed, x, y) for seed in layer_seeds]
            return torch.stack(seed_outputs, dim=0).mean(dim=0)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the complete network."""
        # Input layer
        x = self.input_activation(self.input_layer(x))

        # Dynamic hidden layers with multiple seeds per layer
        for i in range(self.num_layers):
            # Apply linear layer and activation
            x = self.activations[i](self.layers[i](x))
            # Apply seeds for this layer using the helper
            x = self._apply_layer_seeds(x, y, i)

        return self.out(x)
