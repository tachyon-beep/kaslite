"""
Event handling components for the UI dashboard.

This module manages event logging, seed event tracking, and timeline management
for the Rich dashboard interface.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from rich.panel import Panel
from rich.text import Text

from morphogenetic_engine.events import LogPayload, SeedLogPayload

from .config import MAX_EVENTS, MAX_SEED_EVENTS


class EventManager:
    """Manages event logging and display for the dashboard."""

    def __init__(self):
        self.last_events: deque[str] = deque(maxlen=MAX_EVENTS)
        self.seed_log_events: deque[str] = deque(maxlen=MAX_SEED_EVENTS)

    def log_event(self, payload: LogPayload) -> None:
        """Log a generic event to the timeline."""
        # Use the 'level' from the payload, not 'event_type' which does not exist
        level = payload.get("level", "INFO").upper()
        message = payload.get("message", "")
        event_str = f"[{level}] {message}"
        self.last_events.append(event_str)

    def log_seed_event(self, payload: SeedLogPayload) -> None:
        """Add a new event to the dedicated seed log."""
        event_str = f"[{payload['event_type'].upper()}] {payload['message']}"
        data = payload.get("data")
        if data:
            data_str = ", ".join([f"{k}={v}" for k, v in data.items() if v is not None])
            if data_str:
                event_str += f" ({data_str})"
        self.seed_log_events.append(event_str)

    def create_event_log_panel(self) -> Panel:
        """Generate the panel for experiment event log."""
        event_text = "\n".join(self.last_events)
        content = Text.from_markup(event_text)
        return Panel(content, title="Event Log", border_style="blue")

    def create_seed_timeline_panel(self) -> Panel:
        """Generate the panel for the seed event log."""
        if not self.seed_log_events:
            content = Text("No seed events yet...", style="dim")
        else:
            # Show only the most recent events (last 20 to fit in panel without scrolling issues)
            recent_events = list(self.seed_log_events)[-20:]
            
            # Add header showing total events if we're truncating
            lines = []
            if len(self.seed_log_events) > 20:
                hidden_count = len(self.seed_log_events) - 20
                lines.append(f"[dim]... {hidden_count} earlier events[/]")
                lines.append("")  # Empty line separator
            
            # Add all recent events 
            lines.extend(recent_events)
            
            # Join all lines into single text content (same as event log panel)
            event_text = "\n".join(lines)
            content = Text.from_markup(event_text)
        
        # Show event count in title
        title = f"Seed Timeline ({len(self.seed_log_events)} events)"
        return Panel(content, title=title, border_style="red")

    def log_metrics_update(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Log a simplified metrics update to the event log."""
        simple_metrics = {
            "loss": f"{metrics.get('train_loss', 0.0):.4f}",
            "acc": f"{metrics.get('val_acc', 0.0):.4f}",
        }
        self.log_event({"event_type": "epoch", "message": f"Epoch {epoch} complete", "data": simple_metrics})

    def log_phase_transition(self, from_phase: str, to_phase: str, epoch: int) -> None:
        """Log a phase transition event."""
        self.log_event(
            {"event_type": "phase_transition", "message": f"Moving to {to_phase}", "data": {"from": from_phase, "epoch": epoch}}
        )
