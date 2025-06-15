"""Components module for morphogenetic engine."""
import logging

import torch
from torch import nn

from .core import SeedManager


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
        blend_steps: int = 30,
        shadow_lr: float = 1e-3,
        progress_thresh: float = 0.6,
        drift_warn: float = 0.12,
    ):
        super().__init__()
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

        self.child_optim = torch.optim.Adam(
            self.child.parameters(), lr=shadow_lr, weight_decay=0.0
        )
        self.child_loss = nn.MSELoss()

        # Register with seed manager
        self.seed_manager = SeedManager()  # type: SeedManager
        self.seed_manager.register_seed(self, seed_id)

    # ------------------------------------------------------------------
    def _set_state(self, new_state: str):
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
        inputs = inputs.detach()       # block trunk grads

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
    Trunk: 3 hidden linear blocks, each followed by a sentinel seed.
    Then two extra seed–linear pairs so we end up with 8 seeds in total.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        *,
        input_dim: int = 2,
        blend_steps: int = 30,
        shadow_lr: float = 1e-3,
        progress_thresh: float = 0.6,
        drift_warn: float = 0.1
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.seed1 = SentinelSeed(
            "seed1",
            hidden_dim,
            blend_steps=blend_steps,
            shadow_lr=shadow_lr,
            progress_thresh=progress_thresh,
            drift_warn=drift_warn,
        )

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.seed2 = SentinelSeed(
            "seed2",
            hidden_dim,
            blend_steps=blend_steps,
            shadow_lr=shadow_lr,
            progress_thresh=progress_thresh,
            drift_warn=drift_warn,
        )

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.act3 = nn.ReLU()
        self.seed3 = SentinelSeed(
            "seed3",
            hidden_dim,
            blend_steps=blend_steps,
            shadow_lr=shadow_lr,
            progress_thresh=progress_thresh,
            drift_warn=drift_warn,
        )

        # extra capacity layers + seeds --------------------
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.act4 = nn.ReLU()
        self.seed4 = SentinelSeed(
            "seed4",
            hidden_dim,
            blend_steps=blend_steps,
            shadow_lr=shadow_lr,
            progress_thresh=progress_thresh,
            drift_warn=drift_warn,
        )

        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.act5 = nn.ReLU()
        self.seed5 = SentinelSeed(
            "seed5",
            hidden_dim,
            blend_steps=blend_steps,
            shadow_lr=shadow_lr,
            progress_thresh=progress_thresh,
            drift_warn=drift_warn,
        )

        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.act6 = nn.ReLU()
        self.seed6 = SentinelSeed(
            "seed6",
            hidden_dim,
            blend_steps=blend_steps,
            shadow_lr=shadow_lr,
            progress_thresh=progress_thresh,
            drift_warn=drift_warn,
        )

        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.act7 = nn.ReLU()
        self.seed7 = SentinelSeed(
            "seed7",
            hidden_dim,
            blend_steps=blend_steps,
            shadow_lr=shadow_lr,
            progress_thresh=progress_thresh,
            drift_warn=drift_warn,
        )

        self.fc8 = nn.Linear(hidden_dim, hidden_dim)
        self.act8 = nn.ReLU()
        self.seed8 = SentinelSeed(
            "seed8",
            hidden_dim,
            blend_steps=blend_steps,
            shadow_lr=shadow_lr,
            progress_thresh=progress_thresh,
            drift_warn=drift_warn,
        )
        # ---------------------------------------------------

        self.out = nn.Linear(hidden_dim, 2)

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        """Freeze every parameter that doesn’t belong to a seed module."""
        for name, p in self.named_parameters():
            if "seed" not in name:
                p.requires_grad = False

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the complete network."""
        x = self.act1(self.fc1(x))
        x = self.seed1(x)
        x = self.act2(self.fc2(x))
        x = self.seed2(x)
        x = self.act3(self.fc3(x))
        x = self.seed3(x)

        x = self.act4(self.fc4(x))
        x = self.seed4(x)
        x = self.act5(self.fc5(x))
        x = self.seed5(x)
        x = self.act6(self.fc6(x))
        x = self.seed6(x)
        x = self.act7(self.fc7(x))
        x = self.seed7(x)
        x = self.act8(self.fc8(x))
        x = self.seed8(x)

        return self.out(x)

    # ------------------------------------------------------------------
