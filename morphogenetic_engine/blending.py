"""
Defines a pluggable system for blending strategies in the morphogenetic engine.

This module provides an abstract base class for all blending strategies and includes
concrete implementations that can be selected and configured at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .components import SentinelSeed
    from .core import BlendingConfig


class BlendingStrategy(ABC):
    """
    Abstract base class for all blending strategies.
    Each strategy is instantiated for a specific seed and manages its blending process.
    """

    def __init__(self, seed: SentinelSeed, config: BlendingConfig):
        self.seed = seed
        self.config = config

    @abstractmethod
    def update(self) -> float:
        """
        Calculates the next alpha value for the seed.

        Returns:
            The new alpha value, capped at 1.0. The caller is responsible
            for updating the seed's state and handling the transition
            once blending is complete (i.e., alpha >= 1.0).
        """
        raise NotImplementedError


class FixedRampBlending(BlendingStrategy):
    """
    A simple, time-based blending strategy that increases the blend factor (`alpha`)
    linearly over a fixed number of steps.
    """

    def update(self) -> float:
        """Increments alpha linearly based on the configured number of fixed_steps."""
        # Aligns with the plan by using the central BlendingConfig.
        new_alpha = self.seed.alpha + (1.0 / self.config.fixed_steps)
        return min(1.0, new_alpha)


class PerformanceLinkedBlending(BlendingStrategy):
    """
    Dynamically adjusts alpha based on ACTUAL performance improvement.
    Uses gradual progression with performance-based acceleration for stability.
    """

    def update(self) -> float:
        seed_info = self.seed.seed_manager.seeds[self.seed.seed_id]
        start_loss = seed_info.get("blend_initial_loss")
        if start_loss is None:
            # First call - capture baseline
            start_loss = self.seed.validate_on_holdout()
            seed_info["blend_initial_loss"] = start_loss
            return self.seed.alpha  # No progress until we have baseline
        
        current_loss = self.seed.validate_on_holdout()
        loss_reduction = max(0.0, start_loss - current_loss)
        
        # Alpha scales with actual performance improvement, but progresses gradually
        if start_loss > 1e-12:  # Avoid division by zero
            improvement_ratio = loss_reduction / start_loss
            # Scale the improvement to a reasonable step size, with minimum progress
            base_step = 1.0 / self.config.fixed_steps
            performance_multiplier = max(0.5, min(3.0, improvement_ratio * 2.0))  # 0.5x to 3x speed
            new_alpha = self.seed.alpha + (base_step * performance_multiplier)
        else:
            # If baseline is zero, use normal fixed progression
            new_alpha = self.seed.alpha + (1.0 / self.config.fixed_steps)
            
        return min(1.0, new_alpha)  # Cap at 1.0


class DriftControlledBlending(BlendingStrategy):
    """
    Dynamically controls blending rate based on measured weight drift.
    Speeds up when stable, slows down when drifting, holds when unstable.
    """

    def update(self) -> float:
        info = self.seed.seed_manager.seeds[self.seed.seed_id]
        drift_window = info.setdefault("drift_window", deque(maxlen=5))
        
        # Get current drift measurement (this should be computed by the system)
        current_drift = info.get("telemetry", {}).get("drift", 0.0)
        drift_window.append(current_drift)
        
        # Calculate average drift over the window
        avg_drift = sum(drift_window) / len(drift_window)
        
        # Dynamic step size based on stability
        if avg_drift < 0.5 * self.config.high_drift_threshold:
            # Very stable - accelerate (2x speed)
            step_size = 2.0 / self.config.fixed_steps
        elif avg_drift < self.config.high_drift_threshold:
            # Moderately stable - normal speed
            step_size = 1.0 / self.config.fixed_steps
        else:
            # Unstable - hold position
            step_size = 0.0
            
        new_alpha = self.seed.alpha + step_size
        return min(1.0, new_alpha)


class GradNormGatedBlending(BlendingStrategy):
    """
    Only allows blending progress when gradient norms are in stable range.
    Prevents blending during gradient instability.
    """

    def update(self) -> float:
        grad_norm = self.seed.seed_manager.seeds[self.seed.seed_id].get("avg_grad_norm", 0.0)
        
        # Only progress if gradients are stable
        if self.config.grad_norm_lower <= grad_norm <= self.config.grad_norm_upper:
            # Gradients stable - normal progress
            step_size = 1.0 / self.config.fixed_steps
            new_alpha = self.seed.alpha + step_size
            return min(1.0, new_alpha)
        else:
            # Gradients unstable - hold position
            return self.seed.alpha


# =============================================================================
# Strategy Factory
# =============================================================================

STRATEGIES = {
    "FIXED_RAMP": FixedRampBlending,
    "PERFORMANCE_LINKED": PerformanceLinkedBlending,
    "DRIFT_CONTROLLED": DriftControlledBlending,
    "GRAD_NORM_GATED": GradNormGatedBlending,
}


def get_strategy(
    strategy_name: str, seed: SentinelSeed, config: BlendingConfig
) -> BlendingStrategy:
    """
    Factory function to create a blending strategy instance.

    If the requested strategy name is not found, it defaults to FixedRampBlending
    to ensure backward compatibility and robustness.
    """
    strategy_class = STRATEGIES.get(strategy_name, FixedRampBlending)
    return strategy_class(seed, config)
