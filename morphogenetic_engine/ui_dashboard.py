"""
Backward compatibility module for the UI dashboard.

This module provides backward compatibility by re-exporting the RichDashboard
from the new modular UI structure.
"""

from __future__ import annotations

from morphogenetic_engine.ui import RichDashboard

__all__ = ["RichDashboard"]
