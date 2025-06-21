"""
Blending analytics aggregator for Phase 2 dashboard integration.

This module provides the foundation for collecting and aggregating
blending strategy performance metrics that can be consumed by
dashboard components in Phase 2.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .events import BlendCompletedPayload, BlendStrategyChosenPayload


@dataclass
class BlendingStrategyStats:
    """Statistics for a single blending strategy."""
    
    strategy_name: str
    usage_count: int = 0
    total_duration: int = 0
    successful_completions: int = 0
    avg_loss_improvement: float = 0.0
    avg_drift_change: float = 0.0
    recent_performances: List[float] = field(default_factory=list)
    
    @property
    def avg_duration(self) -> float:
        """Average duration in epochs."""
        return self.total_duration / max(1, self.usage_count)
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        return (self.successful_completions / max(1, self.usage_count)) * 100


class BlendingAnalytics:
    """
    Aggregates blending strategy performance data for analytics and dashboard display.
    
    This class provides the foundation for Phase 2 dashboard components to
    query strategy performance metrics and trends.
    """
    
    def __init__(self):
        self.strategy_stats: Dict[str, BlendingStrategyStats] = defaultdict(
            lambda: BlendingStrategyStats("UNKNOWN")
        )
        self.recent_events: List[Dict[str, Any]] = []
        self.max_recent_events = 100
    
    def record_strategy_chosen(self, payload: BlendStrategyChosenPayload):
        """Record when a strategy is chosen for a seed."""
        strategy_name = payload["strategy_name"]
        stats = self.strategy_stats[strategy_name]
        stats.strategy_name = strategy_name
        stats.usage_count += 1
        
        # Store event for recent activity
        self.recent_events.append({
            "type": "strategy_chosen",
            "seed_id": payload["seed_id"],
            "strategy": strategy_name,
            "epoch": payload["epoch"],
            "telemetry": payload["telemetry"],
            "timestamp": payload["timestamp"]
        })
        self._trim_recent_events()
    
    def record_blend_completed(self, payload: BlendCompletedPayload):
        """Record when a blending phase completes."""
        strategy_name = payload["strategy_name"]
        stats = self.strategy_stats[strategy_name]
        
        # Update aggregated statistics
        stats.total_duration += payload["duration_epochs"]
        stats.successful_completions += 1
        
        # Calculate performance metrics
        loss_improvement = payload["initial_loss"] - payload["final_loss"]
        drift_change = payload["final_drift"] - payload["initial_drift"]
        
        # Update running averages
        total_completions = stats.successful_completions
        stats.avg_loss_improvement = (
            (stats.avg_loss_improvement * (total_completions - 1) + loss_improvement) / 
            total_completions
        )
        stats.avg_drift_change = (
            (stats.avg_drift_change * (total_completions - 1) + drift_change) / 
            total_completions
        )
        
        # Track recent performance
        stats.recent_performances.append(loss_improvement)
        if len(stats.recent_performances) > 10:
            stats.recent_performances.pop(0)
        
        # Store event for recent activity
        self.recent_events.append({
            "type": "blend_completed",
            "seed_id": payload["seed_id"],
            "strategy": strategy_name,
            "epoch": payload["epoch"],
            "duration": payload["duration_epochs"],
            "loss_improvement": loss_improvement,
            "timestamp": payload["timestamp"]
        })
        self._trim_recent_events()
    
    def get_strategy_summary(self) -> Dict[str, BlendingStrategyStats]:
        """Get summary statistics for all strategies."""
        return dict(self.strategy_stats)
    
    def get_top_performing_strategies(self, limit: int = 3) -> List[BlendingStrategyStats]:
        """Get the top performing strategies by average loss improvement."""
        strategies = list(self.strategy_stats.values())
        return sorted(
            strategies, 
            key=lambda s: s.avg_loss_improvement, 
            reverse=True
        )[:limit]
    
    def get_recent_activity(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent blending activity for timeline display."""
        return self.recent_events[-limit:]
    
    def _trim_recent_events(self):
        """Keep recent events list at manageable size."""
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events = self.recent_events[-self.max_recent_events:]


# Global analytics instance for Phase 2 integration
_analytics_instance: BlendingAnalytics | None = None


def get_blending_analytics() -> BlendingAnalytics:
    """Get the global blending analytics instance."""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = BlendingAnalytics()
    return _analytics_instance


def reset_blending_analytics():
    """Reset analytics for testing purposes."""
    global _analytics_instance
    _analytics_instance = None
