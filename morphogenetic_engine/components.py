"""Components module for morphogenetic engine."""

import logging
import weakref
from enum import Enum

import torch
from torch import nn

from morphogenetic_engine.core import SeedManager
from morphogenetic_engine.events import SeedState


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
        blend_steps: int = 30,
        probationary_steps: int = 50,
        shadow_lr: float = 1e-3,
        drift_warn: float = 0.12,
    ):
        super().__init__()

        # Parameter validation
        if dim <= 0:
            raise ValueError(f"Invalid dimension: {dim}. Must be positive.")
        if blend_steps <= 0:
            raise ValueError(f"Invalid blend_steps: {blend_steps}. Must be positive.")

        self.seed_id = seed_id
        self.dim = dim
        self.parent_net_ref = weakref.ref(parent_net)  # Use a weak reference to avoid recursion
        self.blend_steps = blend_steps
        self.probationary_steps = probationary_steps
        self.probationary_counter = 0
        self.shadow_lr = shadow_lr
        self.alpha = 0.0
        self.state = SeedState.DORMANT.value
        self.drift_warn = drift_warn
        
        # Convergence-based training completion
        self.loss_history: list[float] = []
        self.convergence_threshold = 1e-3  # More realistic threshold for neural network training
        self.convergence_window = 15  # Slightly larger window for better stability

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
        
        # Update status mapping for backward compatibility  
        if new_state in [SeedState.TRAINING, SeedState.GERMINATED, SeedState.BLENDING, SeedState.SHADOWING, SeedState.PROBATIONARY]:
            info["status"] = "active"
        elif new_state == SeedState.DORMANT:
            info["status"] = "dormant"
        elif new_state == SeedState.FOSSILIZED:
            info["status"] = "fossilized"
        elif new_state == SeedState.CULLED:
            info["status"] = "culled"

        # If epoch is not provided, try to get it from the seed's own record
        if epoch is None:
            epoch = info.get('last_epoch', 0)
            
        self.seed_manager.record_transition(self.seed_id, old_state, new_state.value, epoch=epoch)

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
        
        # Only transition to GERMINATED state (parking lot)
        # The seed will transition to TRAINING when training starts
        self._set_state(SeedState.GERMINATED, epoch=epoch)

    # ------------------------------------------------------------------
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
        self.child_optim.step()
        current_loss = loss.item()
        seed_info = self.seed_manager.seeds[self.seed_id]
        if seed_info["baseline_loss"] is None:
            seed_info["baseline_loss"] = current_loss
        seed_info["current_loss"] = current_loss
        seed_info["training_steps"] += 1
        self.loss_history.append(current_loss)
        # All state transition logic has been REMOVED from this method.

    def evaluate_loss(self, inputs: torch.Tensor) -> float:
        """Evaluate current loss without training (for validation phases)."""
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
        """Per-step: update the blending alpha value during blending phase (no state transition)."""
        if self.state == SeedState.BLENDING.value:
            self.alpha = min(1.0, self.alpha + 1 / self.blend_steps)
            self.seed_manager.seeds[self.seed_id]["alpha"] = self.alpha

    def update_shadowing(self, epoch: int | None = None, inputs: torch.Tensor | None = None):
        """Per-step: accumulate stability metrics during shadowing phase (no state transition)."""
        if self.state == SeedState.SHADOWING.value:
            seed_info = self.seed_manager.seeds[self.seed_id]
            if "stability_history" not in seed_info:
                seed_info["stability_history"] = []
                seed_info["shadowing_steps"] = 0
            seed_info["shadowing_steps"] += 1
            if inputs is not None:
                current_loss = self.evaluate_loss(inputs)
            else:
                current_loss = seed_info.get("current_loss", 0.0)
            seed_info["stability_history"].append(current_loss)
            if len(seed_info["stability_history"]) > 20:
                seed_info["stability_history"].pop(0)

    def update_probationary(self, epoch: int | None = None, inputs: torch.Tensor | None = None):
        """Per-step: accumulate probationary metrics (no state transition)."""
        if self.state == SeedState.PROBATIONARY.value:
            seed_info = self.seed_manager.seeds[self.seed_id]
            if "probationary_steps" not in seed_info:
                seed_info["probationary_steps"] = 0
                seed_info["baseline_performance"] = None
            seed_info["probationary_steps"] += 1
            if inputs is not None:
                self.evaluate_loss(inputs)

    def check_blending_transition(self, epoch: int | None = None):
        """Per-epoch: promote BLENDING -> SHADOWING if blending complete."""
        if self.state == SeedState.BLENDING.value and self.alpha >= 0.99:
            self._set_state(SeedState.SHADOWING, epoch=epoch)

    def check_shadowing_transition(self, epoch: int | None = None):
        """Per-epoch: promote SHADOWING -> PROBATIONARY if stability/step criteria met."""
        if self.state == SeedState.SHADOWING.value:
            seed_info = self.seed_manager.seeds[self.seed_id]
            min_shadowing_steps = self.probationary_steps // 2
            if seed_info.get("shadowing_steps", 0) >= min_shadowing_steps:
                if len(seed_info.get("stability_history", [])) >= 10:
                    recent_losses = seed_info["stability_history"][-10:]
                    mean = sum(recent_losses) / len(recent_losses)
                    loss_variance = sum((x - mean) ** 2 for x in recent_losses) / len(recent_losses)
                    if loss_variance < 0.01:
                        self._set_state(SeedState.PROBATIONARY, epoch=epoch)
                        seed_info["probationary_steps"] = 0

    def check_probationary_transition(self, epoch: int | None = None):
        """Per-epoch: complete probationary if enough steps have passed."""
        if self.state == SeedState.PROBATIONARY.value:
            seed_info = self.seed_manager.seeds[self.seed_id]
            min_probationary_steps = self.probationary_steps
            if seed_info.get("probationary_steps", 0) >= min_probationary_steps:
                self._evaluate_and_complete(epoch)

    def check_training_transition(self, epoch: int | None = None):
        """Per-epoch: promote TRAINING -> BLENDING if converged."""
        if self.state == SeedState.TRAINING.value and self.check_convergence():
            self._set_state(SeedState.BLENDING, epoch=epoch)
            self.alpha = 0.0

    def apply_epoch_transitions(self, epoch: int | None = None):
        """
        Evaluates and applies at most one state transition at the end of an epoch.
        The if/elif structure ensures atomicity.
        """
        if self.state == SeedState.TRAINING.value:
            self.check_training_transition(epoch)
        elif self.state == SeedState.BLENDING.value:
            self.check_blending_transition(epoch)
        elif self.state == SeedState.SHADOWING.value:
            self.check_shadowing_transition(epoch)
        elif self.state == SeedState.PROBATIONARY.value:
            self.check_probationary_transition(epoch)