"""
Legacy ui_dashboard module for backward compatibility.

This module now re-exports the RichDashboard class from the refactored ui module.
The original 1158-line file has been broken down into focused components:

- ui/config.py - Constants, emoji mappings, and configuration
- ui/metrics.py - Metrics tracking, sparklines, and calculations
- ui/events.py - Event logging and timeline management
- ui/grids.py - Grid visualizations for seeds and network strain
- ui/panels.py - Panel factory methods for various UI components
- ui/layout.py - Layout management and structure
- ui/dashboard.py - Main dashboard coordination class

For new code, prefer importing from morphogenetic_engine.ui directly.
"""

from __future__ import annotations

# Re-export for backward compatibility
from morphogenetic_engine.ui import RichDashboard

__all__ = ["RichDashboard"]
