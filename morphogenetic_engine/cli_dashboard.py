"""
Rich-powered CLI dashboard for real-time progress and metrics during morphogenetic experiments.

This is a simplified version, designed to be rebuilt.
"""
from __future__ import annotations
import time
from collections import deque
from typing import Any, Deque, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class RichDashboard:
    """A simplified Rich CLI dashboard for experiment monitoring."""

    def __init__(self, console: Optional[Console] = None, experiment_params: Optional[dict[str, Any]] = None):
        self.console = console or Console()
        self.experiment_params = experiment_params or {}
        self.layout: Optional[Layout] = None
        self.live: Optional[Live] = None
        self._layout_initialized = False
        self.last_events: Deque[str] = deque(maxlen=10)

    def _setup_layout(self):
        """Initialize the dashboard layout (only once)."""
        if self._layout_initialized:
            return
        
        self.layout = Layout(name="root")
        self.layout.split(
            Layout(name="main", ratio=1)
        )
        self.layout["main"].update(self.get_content())
        self._layout_initialized = True

    def get_content(self) -> Panel:
        """Generate the content for the dashboard."""
        event_text = "\n".join(self.last_events)
        content = Text.from_markup(
            f"[bold cyan]Morphogenetic Engine Dashboard[/bold cyan]\n\n"
            f"This is a placeholder UI. Waiting for events...\n\n"
            f"[bold]Last Events:[/bold]\n{event_text}"
        )
        return Panel(content, title="Status", border_style="blue")

    def add_live_event(self, event_type: str, message: str, data: dict[str, Any]):
        """Add a new event to be displayed."""
        event_str = f"[{event_type.upper()}] {message}"
        if data:
            data_str = ", ".join([f"{k}={v}" for k, v in data.items() if v is not None])
            if data_str:
                event_str += f" ({data_str})"
        self.last_events.append(event_str)
        if self.layout:
            self.layout["main"].update(self.get_content())

    def start_phase(self, phase_name: str, total_epochs: int, description: str = ""):
        """Handle phase start event."""
        self.add_live_event(
            "phase_start",
            description or phase_name,
            {"epochs": total_epochs}
        )

    def update_progress(self, epoch: int, metrics: dict[str, Any]):
        """Handle epoch progress event."""
        # To avoid clutter, only show a few key metrics
        simple_metrics = {
            "loss": f"{metrics.get('train_loss', 0.0):.4f}",
            "acc": f"{metrics.get('val_acc', 0.0):.4f}",
        }
        self.add_live_event("epoch", f"Epoch {epoch} complete", simple_metrics)

    def update_seed(self, seed_id: str, state: str, alpha: float = 0.0, from_state: str = "unknown"):
        """Handle seed state change event."""
        self.add_live_event("seed", f"Seed {seed_id}: {from_state} -> {state}", {"alpha": f"{alpha:.2f}"})

    def show_phase_transition(self, to_phase: str, epoch: int, from_phase: str = ""):
        """Handle phase transition event."""
        self.add_live_event("phase_transition", f"Moving to {to_phase}", {"from": from_phase, "epoch": epoch})

    def show_germination_event(self, seed_id: str, epoch: int):
        """Handle seed germination event."""
        self.add_live_event("germination", f"Seed {seed_id} germinated!", {"epoch": epoch})

    def start(self):
        """Start the live dashboard display."""
        if not self.live:
            self._setup_layout()
            self.live = Live(self.layout, console=self.console, screen=True, auto_refresh=True)
            self.live.start()

    def stop(self):
        """Stop the live dashboard display."""
        if self.live:
            self.live.stop()
            self.live = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def demo_dashboard():
    """Demo function to showcase the simplified dashboard."""
    console = Console()
    print("Starting dashboard demo...")
    with RichDashboard(console) as dashboard:
        dashboard.add_live_event("INFO", "Dashboard Initialized", {})
        time.sleep(2)
        dashboard.add_live_event("METRICS", "Epoch 1/100", {"loss": 0.9, "acc": 0.5})
        time.sleep(2)
        dashboard.add_live_event("SEED", "Seed 'abc' is now active", {})
        time.sleep(2)
        dashboard.add_live_event("PHASE", "Transitioning to Phase 2", {})
        time.sleep(2)

    console.print("\n✅ Demo completed!")


if __name__ == "__main__":
    demo_dashboard()
