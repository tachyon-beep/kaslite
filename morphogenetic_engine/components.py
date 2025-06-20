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
        progress_thresh: float = 0.6,
        drift_warn: float = 0.12,
    ):
        super().__init__()

        # Parameter validation
        if dim <= 0:
            raise ValueError(f"Invalid dimension: {dim}. Must be positive.")
        if blend_steps <= 0:
            raise ValueError(f"Invalid blend_steps: {blend_steps}. Must be positive.")
        if not (0.0 < progress_thresh < 1.0):
            raise ValueError(f"Invalid progress_thresh: {progress_thresh}. Must be between 0 and 1.")

        self.seed_id = seed_id
        self.dim = dim
        self.parent_net_ref = weakref.ref(parent_net)  # Use a weak reference to avoid recursion
        self.blend_steps = blend_steps
        self.probationary_steps = probationary_steps
        self.probationary_counter = 0
        self.shadow_lr = shadow_lr
        self.progress_thresh = progress_thresh
        self.alpha = 0.0
        self.state = SeedState.DORMANT.value
        self.training_progress = 0.0
        self.drift_warn = drift_warn

        # Create child network with residual connection
        self.child = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
        )
        self._initialize_as_identity()

        self.child_optim = torch.optim.Adam(self.child.parameters(), lr=shadow_lr, weight_decay=0.0)
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
        
        # Capture baseline performance for later evaluation
        self.seed_manager.seeds[self.seed_id]["baseline_loss"] = None
        self.seed_manager.seeds[self.seed_id]["current_loss"] = None
        self.seed_manager.seeds[self.seed_id]["training_steps"] = 0
        
        # Only transition to GERMINATED state (parking lot)
        # The seed will transition to TRAINING when training starts
        self._set_state(SeedState.GERMINATED, epoch=epoch)

    # ------------------------------------------------------------------
    def train_child_step(self, inputs: torch.Tensor, epoch: int | None = None):
        """Train the child network on input data when in training state."""
        if self.state != SeedState.TRAINING.value or inputs.numel() == 0:
            return
        inputs = inputs.detach()  # block trunk grads

        self.child_optim.zero_grad(set_to_none=True)
        outputs = self.child(inputs)
        loss = self.child_loss(outputs, inputs)
        loss.backward()
        
        # Calculate and store gradient norm before optimizer step
        total_norm = 0.0
        param_count = 0
        for p in self.child.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                param_count += 1
        current_grad_norm = (total_norm**0.5) if param_count > 0 else 0.0
        
        self.child_optim.step()

        # Track loss and gradient norm for performance evaluation
        current_loss = loss.item()
        seed_info = self.seed_manager.seeds[self.seed_id]
        
        # Set baseline on first training step
        if seed_info["baseline_loss"] is None:
            seed_info["baseline_loss"] = current_loss
        
        seed_info["current_loss"] = current_loss
        seed_info["gradient_norm"] = current_grad_norm
        seed_info["training_steps"] += 1

        # Update training progress and alpha during training
        self.training_progress = min(1.0, self.training_progress + 0.01)
        
        # Alpha remains 0 during training - only increases during blending
        if self.state == SeedState.TRAINING.value:
            self.alpha = 0.0  # Keep at 0 during training
        
        if self.training_progress > self.progress_thresh:
            self._set_state(SeedState.BLENDING, epoch=epoch)
            # Start blending with alpha at 0
            self.alpha = 0.0
        
        # Always sync alpha and training_steps to seed manager dict
        seed_info["alpha"] = self.alpha
        seed_info["training_steps"] = seed_info.get("training_steps", 0)  # Ensure key exists

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def update_blending(self, epoch: int | None = None):
        """Update the blending alpha value during blending phase."""
        if self.state == SeedState.BLENDING.value:
            self.alpha = min(1.0, self.alpha + 1 / self.blend_steps)
            self.seed_manager.seeds[self.seed_id]["alpha"] = self.alpha
                
            if self.alpha >= 0.99:
                # Blending complete - transition to shadowing phase
                self._set_state(SeedState.SHADOWING, epoch=epoch)

    def update_shadowing(self, epoch: int | None = None):
        """Stage 1 validation - monitor for internal stability during shadowing phase."""
        if self.state == SeedState.SHADOWING.value:
            # During shadowing, monitor internal stability metrics
            seed_info = self.seed_manager.seeds[self.seed_id]
            
            # Initialize stability tracking if not present
            if "stability_history" not in seed_info:
                seed_info["stability_history"] = []
                seed_info["shadowing_steps"] = 0
            
            seed_info["shadowing_steps"] += 1
            
            # Record current loss for stability analysis
            current_loss = seed_info.get("current_loss", 0.0)
            seed_info["stability_history"].append(current_loss)
            
            # Keep only recent history
            if len(seed_info["stability_history"]) > 20:
                seed_info["stability_history"].pop(0)
            
            # Require minimum steps AND stability before advancing
            min_shadowing_steps = self.probationary_steps // 2  # e.g., 25 steps
            if seed_info["shadowing_steps"] >= min_shadowing_steps:
                # Check stability: loss should be relatively stable
                if len(seed_info["stability_history"]) >= 10:
                    recent_losses = seed_info["stability_history"][-10:]
                    loss_variance = sum((x - sum(recent_losses)/len(recent_losses))**2 for x in recent_losses) / len(recent_losses)
                    
                    # Only advance if loss is stable (low variance)
                    if loss_variance < 0.01:  # Configurable stability threshold
                        self._set_state(SeedState.PROBATIONARY, epoch=epoch)
                        # Reset counter for probationary phase  
                        seed_info["probationary_steps"] = 0

    def update_probationary(self, epoch: int | None = None):
        """Stage 2 validation - monitor for systemic impact during probationary period."""
        if self.state == SeedState.PROBATIONARY.value:
            seed_info = self.seed_manager.seeds[self.seed_id]
            
            # Initialize probationary tracking if not present
            if "probationary_steps" not in seed_info:
                seed_info["probationary_steps"] = 0
                seed_info["baseline_performance"] = None
                
            seed_info["probationary_steps"] += 1
            
            # Require minimum observation period for systemic impact assessment
            min_probationary_steps = self.probationary_steps  # e.g., 50 steps
            if seed_info["probationary_steps"] >= min_probationary_steps:
                # In a real implementation, this would check global performance metrics
                # For now, we'll use a simple success criteria
                self._evaluate_and_complete(epoch)

    def update_state(self, epoch: int | None = None):
        """Update the seed's state based on its current metrics and progress."""
        # This method is called by KasminaMicro to assess and potentially update state
        # For now, we'll implement basic state transitions based on existing logic

        # Update metrics in seed manager
        self.seed_manager.seeds[self.seed_id]["alpha"] = self.alpha

        # State transition logic (simplified for now)
        if self.state == SeedState.TRAINING.value and self.training_progress > self.progress_thresh:
            self._set_state(SeedState.BLENDING, epoch=epoch)
            self.alpha = 0.0
        elif self.state == SeedState.BLENDING.value and self.alpha >= 0.99:
            self._set_state(SeedState.SHADOWING, epoch=epoch)

    def get_gradient_norm(self) -> float:
        """Get the gradient norm for this seed's parameters."""
        if self.state in [SeedState.DORMANT.value]:
            return 0.0

        # Return the stored gradient norm from the last training step
        seed_info = self.seed_manager.seeds[self.seed_id]
        return seed_info.get("gradient_norm", 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sentinel seed."""

        # Buffer activations for dormant seeds to gauge health
        if self.state == SeedState.DORMANT.value:
            self.seed_manager.append_to_buffer(self.seed_id, x)
            return x

        # For culled or pre-training seeds, act as an identity function.
        # TRAINING seeds train on the buffer, not in the forward pass.
        if self.state in [
            SeedState.CULLED.value,
            SeedState.GERMINATED.value,
            SeedState.TRAINING.value,
        ]:
            return x

        # For blending state, gradually mix input with child output
        if self.state == SeedState.BLENDING.value:
            child_out = self.child(x)
            output = (1 - self.alpha) * x + self.alpha * child_out
            
        # For shadowing state, return input unmodified (inert validation)
        elif self.state == SeedState.SHADOWING.value:
            return x
            
        # For probationary and fossilized states, apply child network additively
        elif self.state in [SeedState.PROBATIONARY.value, SeedState.FOSSILIZED.value]:
            child_out = self.child(x)
            output = x + child_out
        else:
            # Default case - should not reach here
            return x

        # Monitor drift for active states only
        if self.state in [SeedState.BLENDING.value, SeedState.PROBATIONARY.value]:
            with torch.no_grad():
                cos_sim = torch.cosine_similarity(x, output, dim=-1).mean()
                drift = 1.0 - cos_sim.item()
                # only warn during blending and probationary periods
                if drift > self.drift_warn > 0:
                    logging.warning("High drift %.4f at %s (%s)", drift, self.seed_id, self.state)
            # always record drift for telemetry
            self.seed_manager.record_drift(self.seed_id, drift)
            
        return output

    def get_health_signal(self) -> float:
        """Health signal = activation variance (LOWER = worse bottleneck)"""
        buffer = self.seed_manager.seeds[self.seed_id]["buffer"]
        if len(buffer) < 10:  # Need sufficient samples
            return float("inf")  # Return worst possible signal if insufficient data

        # Calculate variance across all buffered activations
        return torch.var(torch.cat(list(buffer)), dim=0).mean().item()

    def _evaluate_and_complete(self, epoch: int | None = None):
        """Evaluate seed performance and transition to final state (fossilized or culled)."""
        # Get performance metrics
        baseline_loss = self.seed_manager.seeds[self.seed_id].get("baseline_loss")
        current_loss = self.seed_manager.seeds[self.seed_id].get("current_loss") 
        
        # If we don't have baseline, assume improvement and fossilize
        if baseline_loss is None:
            self._fossilize_seed(epoch)
            return
            
        # Calculate improvement
        improvement = baseline_loss - current_loss if current_loss else 0
        improvement_threshold = 0.01  # Minimum improvement required
        
        if improvement > improvement_threshold:
            self._fossilize_seed(epoch)
        else:
            self._cull_seed(epoch)
    
    def _fossilize_seed(self, epoch: int | None = None):
        """Permanently integrate the seed into the parent network (grafting)."""
        self._set_state(SeedState.FOSSILIZED, epoch=epoch)
        
        # Log the fossilization event
        if hasattr(self.seed_manager, 'logger') and self.seed_manager.logger:
            self.seed_manager.logger.log_seed_event_detailed(
                epoch=epoch or 0,
                event_type="FOSSILIZATION",
                message=f"Seed L{self.seed_id[0]}_S{self.seed_id[1]} fossilized!",
                data={"seed_id": f"L{self.seed_id[0]}_S{self.seed_id[1]}", "improvement": True}
            )
        
        # Graft the trained network into the parent permanently
        # In a full implementation, this would modify the parent network architecture
        # to replace the sentinel seed with the trained child network
        self._graft_into_parent()
        
        # Save the modified model
        parent = self.parent_net_ref()
        if parent:
            parent.save_grafted_model()
        
        # Start training the next seed in the queue
        self.seed_manager.start_training_next_seed()
        
    def _graft_into_parent(self):
        """Physically replaces this seed with its trained child in the parent network."""
        parent = self.parent_net_ref()
        if not parent:
            logging.warning("Cannot graft seed: parent_net reference not found.")
            return

        # 1. Create the permanent, grafted module
        # The child network is already trained and will be frozen.
        for p in self.child.parameters():
            p.requires_grad = False
        grafted_module = GraftedChild(self.child)

        # 2. Instruct the parent network to replace this seed
        parent.replace_seed(self.seed_id, grafted_module)
        
    def _cull_seed(self, epoch: int | None = None):
        """Remove the seed due to poor performance."""
        self._set_state(SeedState.CULLED, epoch=epoch)
        
        # Log the culling event
        if hasattr(self.seed_manager, 'logger') and self.seed_manager.logger:
            self.seed_manager.logger.log_seed_event_detailed(
                epoch=epoch or 0,
                event_type="CULLING", 
                message=f"Seed L{self.seed_id[0]}_S{self.seed_id[1]} culled!",
                data={"seed_id": f"L{self.seed_id[0]}_S{self.seed_id[1]}", "improvement": False}
            )
        
        # Deactivate the seed - set alpha to 0 so it has no effect
        self.alpha = 0.0
        self.seed_manager.seeds[self.seed_id]["alpha"] = 0.0
        
        # Freeze parameters to save computation
        for p in self.child.parameters():
            p.requires_grad = False
            
        # Start training the next seed in the queue
        self.seed_manager.start_training_next_seed()


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
        blend_steps: int = 30,
        shadow_lr: float = 1e-3,
        progress_thresh: float = 0.6,
        drift_warn: float = 0.1,
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
                    blend_steps=blend_steps,
                    shadow_lr=shadow_lr,
                    progress_thresh=progress_thresh,
                    drift_warn=drift_warn,
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
