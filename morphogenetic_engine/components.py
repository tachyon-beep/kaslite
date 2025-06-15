import torch
from torch import nn

from .core import SeedManager


class SentinelSeed(nn.Module):
    def __init__(self, seed_id: str, dim: int):
        super().__init__()
        self.seed_id = seed_id
        self.dim = dim

        # Create child network with residual connection
        self.child = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
        )
        self._initialize_as_identity()

        # Register with seed manager
        self.seed_manager = SeedManager()
        self.seed_manager.register_seed(self, seed_id)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        info = self.seed_manager.seeds[self.seed_id]

        if info["status"] != "active":
            # Monitor mode: collect activations without changing them
            self.seed_manager.append_to_buffer(self.seed_id, x)
            return x  # Identity pass-through

        # Active mode: compute residual connection
        residual = self.child(x)
        output = x + residual

        # Monitor interface drift (critical safety metric)
        with torch.no_grad():
            # Cosine similarity measures how much the output has changed
            cos_sim = torch.cosine_similarity(x, output, dim=-1).mean()
            drift = 1.0 - cos_sim.item()
        self.seed_manager.record_drift(self.seed_id, drift)

        return output

    def get_health_signal(self) -> float:
        """Health signal = activation variance (LOWER = worse bottleneck)"""
        buffer = self.seed_manager.seeds[self.seed_id]["buffer"]
        if len(buffer) < 10:  # Need sufficient samples
            return float("inf")  # Return worst possible signal if insufficient data

        # Calculate variance across all buffered activations
        return torch.var(torch.cat(list(buffer))).item()

class BaseNet(nn.Module):
    """
    Trunk: 3 hidden linear blocks, each followed by a sentinel seed.
    Then two extra seed–linear pairs so we end up with 8 seeds in total.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.fc1  = nn.Linear(2, hidden_dim)
        self.act1 = nn.ReLU()
        self.seed1 = SentinelSeed("seed1", hidden_dim)

        self.fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.seed2 = SentinelSeed("seed2", hidden_dim)

        self.fc3  = nn.Linear(hidden_dim, hidden_dim)
        self.act3 = nn.ReLU()
        self.seed3 = SentinelSeed("seed3", hidden_dim)

        # extra capacity layers + seeds --------------------
        self.fc4  = nn.Linear(hidden_dim, hidden_dim)
        self.act4 = nn.ReLU()
        self.seed4 = SentinelSeed("seed4", hidden_dim)

        self.fc5  = nn.Linear(hidden_dim, hidden_dim)
        self.act5 = nn.ReLU()
        self.seed5 = SentinelSeed("seed5", hidden_dim)

        self.fc6  = nn.Linear(hidden_dim, hidden_dim)
        self.act6 = nn.ReLU()
        self.seed6 = SentinelSeed("seed6", hidden_dim)

        self.fc7  = nn.Linear(hidden_dim, hidden_dim)
        self.act7 = nn.ReLU()
        self.seed7 = SentinelSeed("seed7", hidden_dim)

        self.fc8  = nn.Linear(hidden_dim, hidden_dim)
        self.act8 = nn.ReLU()
        self.seed8 = SentinelSeed("seed8", hidden_dim)
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
        x = self.act1(self.fc1(x)); x = self.seed1(x)
        x = self.act2(self.fc2(x)); x = self.seed2(x)
        x = self.act3(self.fc3(x)); x = self.seed3(x)

        x = self.act4(self.fc4(x)); x = self.seed4(x)
        x = self.act5(self.fc5(x)); x = self.seed5(x)
        x = self.act6(self.fc6(x)); x = self.seed6(x)
        x = self.act7(self.fc7(x)); x = self.seed7(x)
        x = self.act8(self.fc8(x)); x = self.seed8(x)

        return self.out(x)
    # ------------------------------------------------------------------

