"""Components module for morphogenetic engine."""

import logging

import torch
from torch import nn

from morphogenetic_engine.core import SeedManager


class SentinelSeed(nn.Module):
    """
    A sentinel seed that monitors activations and can evolve when needed.

    This module starts as dormant, monitors activation patterns, and can be
    germinated to learn adaptive transformations when bottlenecks are detected.
    """

    def __init__(
        self,
        seed_id: str,
        dim: int,
        seed_manager: SeedManager,
        blend_steps: int = 30,
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
            raise ValueError(
                f"Invalid progress_thresh: {progress_thresh}. Must be between 0 and 1."
            )

        self.seed_id = seed_id
        self.dim = dim
        self.blend_steps = blend_steps
        self.shadow_lr = shadow_lr
        self.progress_thresh = progress_thresh
        self.alpha = 0.0
        self.state = "dormant"
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
    def _set_state(self, new_state: str):
        # Validate state
        valid_states = {"dormant", "training", "blending", "active"}
        if new_state not in valid_states:
            raise ValueError(f"Invalid state: {new_state}. Must be one of {valid_states}")

        if self.state == new_state:
            return  # redundant transition guard
        old_state = self.state
        self.state = new_state
        info = self.seed_manager.seeds[self.seed_id]
        info["state"] = new_state
        if new_state == "active":
            info["status"] = "active"
        elif new_state == "dormant":
            info["status"] = "dormant"
        else:  # training or blending
            info["status"] = "pending"
        self.seed_manager.record_transition(self.seed_id, old_state, new_state)

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

    def initialize_child(self):
        """Proper initialization when germinating"""
        for m in self.child.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Make parameters trainable
        for p in self.child.parameters():
            p.requires_grad = True
        self._set_state("training")

    # ------------------------------------------------------------------
    def train_child_step(self, inputs: torch.Tensor):
        """Train the child network on input data when in training state."""
        if self.state != "training" or inputs.numel() == 0:
            return
        inputs = inputs.detach()  # block trunk grads

        self.child_optim.zero_grad(set_to_none=True)
        outputs = self.child(inputs)
        loss = self.child_loss(outputs, inputs)
        loss.backward()
        self.child_optim.step()

        self.training_progress = min(1.0, self.training_progress + 0.01)
        if self.training_progress > self.progress_thresh:
            self._set_state("blending")
            self.alpha = 0.0
        self.seed_manager.seeds[self.seed_id]["alpha"] = self.alpha

    # ------------------------------------------------------------------
    def update_blending(self):
        """Update the blending alpha value during blending phase."""
        if self.state == "blending":
            self.alpha = min(1.0, self.alpha + 1 / self.blend_steps)
            self.seed_manager.seeds[self.seed_id]["alpha"] = self.alpha
            if self.alpha >= 0.99:
                self._set_state("active")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sentinel seed."""

        if self.state != "active":
            self.seed_manager.append_to_buffer(self.seed_id, x)

        if self.state == "dormant":
            return x
        if self.state == "training":
            return x

        child_out = self.child(x)

        if self.state == "blending":
            output = (1 - self.alpha) * x + self.alpha * child_out
        else:  # active
            output = x + child_out

        with torch.no_grad():
            cos_sim = torch.cosine_similarity(x, output, dim=-1).mean()
            drift = 1.0 - cos_sim.item()
            # only warn during blending
            if self.state == "blending" and drift > self.drift_warn > 0:
                logging.warning("High drift %.4f at %s (blending)", drift, self.seed_id)
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
                    f"seed{i+1}_{j+1}",  # e.g., seed1_1, seed1_2, seed2_1, etc.
                    hidden_dim,
                    seed_manager,
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
