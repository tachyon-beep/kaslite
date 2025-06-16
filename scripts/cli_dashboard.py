"""
Rich-powered CLI dashboard for real-time progress and metrics during morphogenetic experiments.

This module provides a live dashboard that displays:
- Progress bars for training epochs
- Live metrics table showing validation loss, accuracy, best accuracy, and active seeds
- Seed states panel showing the current status of each seed
- Phase transition banners
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


class SeedState:
    """Represents the state of a single seed."""

    def __init__(self, seed_id: str, state: str = "dormant", alpha: float = 0.0):
        self.seed_id = seed_id
        self.state = state
        self.alpha = alpha

    def update(self, state: str, alpha: float = 0.0):
        """Update the seed's state and alpha value."""
        self.state = state
        self.alpha = alpha

    def get_styled_status(self) -> Text:
        """Get a styled representation of the seed's status."""
        if self.state == "dormant":
            return Text(f"{self.seed_id}: {self.state}", style="dim white")
        elif self.state == "blending":
            return Text(f"{self.seed_id}: {self.state} α={self.alpha:.3f}", style="yellow")
        elif self.state == "active":
            return Text(f"{self.seed_id}: {self.state} α={self.alpha:.3f}", style="green bold")
        else:
            return Text(f"{self.seed_id}: {self.state}", style="cyan")


class RichDashboard:
    """Rich-powered CLI dashboard for experiment monitoring."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        )
        
        # Tracking variables
        self.current_task: Optional[TaskID] = None
        self.current_phase = "init"
        self.seeds: Dict[str, SeedState] = {}
        self.metrics = {
            "epoch": 0,
            "val_loss": 0.0,
            "val_acc": 0.0,
            "best_acc": 0.0,
            "train_loss": 0.0,
            "seeds_active": 0,
        }
        
        # Layout for the dashboard
        self.layout = Layout()
        self.live = Live(self.layout, console=self.console, refresh_per_second=4)

    def _create_metrics_table(self) -> Table:
        """Create the metrics table showing current experiment progress."""
        table = Table(title="📊 Experiment Metrics", title_style="bold blue")
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Epoch", str(self.metrics["epoch"]))
        table.add_row("Phase", self.current_phase)
        table.add_row("Train Loss", f"{self.metrics['train_loss']:.4f}")
        table.add_row("Val Loss", f"{self.metrics['val_loss']:.4f}")
        table.add_row("Val Accuracy", f"{self.metrics['val_acc']:.4f}")
        table.add_row("Best Accuracy", f"{self.metrics['best_acc']:.4f}")
        table.add_row("Seeds Active", f"{self.metrics['seeds_active']}/{len(self.seeds)}")
        
        return table

    def _create_seeds_panel(self) -> Panel:
        """Create the seeds status panel."""
        if not self.seeds:
            content = Text("No seeds initialized yet", style="dim white")
        else:
            content = Text()
            for i, seed in enumerate(self.seeds.values()):
                if i > 0:
                    content.append("\n")
                content.append(seed.get_styled_status())
        
        return Panel(content, title="🌱 Seed States", title_align="left")

    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.layout.split_column(
            Layout(self.progress, name="progress", size=3),
            Layout(self._create_metrics_table(), name="metrics", size=12),
            Layout(self._create_seeds_panel(), name="seeds"),
        )

    def start_phase(self, phase_name: str, total_epochs: int, description: str = ""):
        """Start a new training phase with a progress bar."""
        self.current_phase = phase_name
        
        # Stop any existing task
        if self.current_task is not None:
            self.progress.stop_task(self.current_task)
        
        # Create new progress task
        task_description = description or f"Phase {phase_name}"
        self.current_task = self.progress.add_task(
            task_description, total=total_epochs
        )
        
        # Update layout
        self._setup_layout()

    def update_progress(self, epoch: int, metrics: Dict[str, Any]):
        """Update the progress bar and metrics."""
        self.metrics.update({
            "epoch": epoch,
            "val_loss": metrics.get("val_loss", 0.0),
            "val_acc": metrics.get("val_acc", 0.0),
            "best_acc": metrics.get("best_acc", 0.0),
            "train_loss": metrics.get("train_loss", 0.0),
        })
        
        # Update active seeds count
        self.metrics["seeds_active"] = sum(
            1 for seed in self.seeds.values() if seed.state == "active"
        )
        
        # Update progress bar
        if self.current_task is not None:
            self.progress.update(self.current_task, completed=epoch)
        
        # Update layout with new metrics
        self.layout["metrics"].update(self._create_metrics_table())
        self.layout["seeds"].update(self._create_seeds_panel())

    def update_seed(self, seed_id: str, state: str, alpha: float = 0.0):
        """Update a seed's state and alpha value."""
        if seed_id not in self.seeds:
            self.seeds[seed_id] = SeedState(seed_id, state, alpha)
        else:
            self.seeds[seed_id].update(state, alpha)
        
        # Update seeds panel
        self.layout["seeds"].update(self._create_seeds_panel())

    def show_phase_transition(self, to_phase: str, epoch: int):
        """Display a highlighted phase transition banner."""
        banner = f"➡️  Entering {to_phase} at epoch {epoch}"
        
        # Print the banner above the live display
        self.console.print()
        self.console.print(Panel(
            Text(banner, style="bold green"),
            title="Phase Transition",
            title_align="center",
            style="green"
        ))
        self.console.print()
        
        self.current_phase = to_phase

    def show_germination_event(self, seed_id: str, epoch: int):
        """Display a germination event notification."""
        message = f"🌱 Seed '{seed_id}' germinated at epoch {epoch}"
        
        self.console.print()
        self.console.print(Panel(
            Text(message, style="bold yellow"),
            title="Germination Event",
            title_align="center",
            style="yellow"
        ))
        self.console.print()

    def __enter__(self):
        """Enter the live display context."""
        self._setup_layout()
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the live display context."""
        # Complete any remaining tasks
        if self.current_task is not None:
            self.progress.stop_task(self.current_task)
        
        self.live.stop()

    def start(self):
        """Start the live dashboard."""
        self._setup_layout()
        self.live.start()

    def stop(self):
        """Stop the live dashboard."""
        if self.current_task is not None:
            self.progress.stop_task(self.current_task)
        self.live.stop()


def demo_dashboard():
    """Demo function to showcase the dashboard capabilities."""
    import random
    import time
    
    console = Console()
    
    with RichDashboard(console) as dashboard:
        # Initialize some seeds
        for i in range(3):
            dashboard.update_seed(f"seed_{i}", "dormant")
        
        # Phase 1: Warm-up
        dashboard.start_phase("phase_1", 10, "Warm-up Training")
        dashboard.show_phase_transition("phase_1", 0)
        
        for epoch in range(1, 11):
            # Simulate training metrics
            metrics = {
                "train_loss": 1.0 - epoch * 0.08 + random.uniform(-0.1, 0.1),
                "val_loss": 1.0 - epoch * 0.07 + random.uniform(-0.1, 0.1),
                "val_acc": epoch * 0.08 + random.uniform(-0.05, 0.05),
                "best_acc": max(epoch * 0.08, 0.6),
            }
            
            dashboard.update_progress(epoch, metrics)
            time.sleep(0.5)
        
        # Phase transition
        dashboard.show_phase_transition("phase_2", 10)
        
        # Phase 2: Adaptation
        dashboard.start_phase("phase_2", 20, "Adaptation Phase")
        
        for epoch in range(11, 31):
            # Simulate seed germination
            if epoch == 15:
                dashboard.update_seed("seed_0", "blending", 0.2)
                dashboard.show_germination_event("seed_0", epoch)
            elif epoch == 18:
                dashboard.update_seed("seed_0", "active", 0.8)
                dashboard.update_seed("seed_1", "blending", 0.1)
            elif epoch == 22:
                dashboard.update_seed("seed_1", "active", 0.6)
            
            # Simulate training metrics
            metrics = {
                "train_loss": 0.3 - (epoch - 10) * 0.01 + random.uniform(-0.05, 0.05),
                "val_loss": 0.4 - (epoch - 10) * 0.015 + random.uniform(-0.05, 0.05),
                "val_acc": 0.7 + (epoch - 10) * 0.01 + random.uniform(-0.02, 0.02),
                "best_acc": max(0.7 + (epoch - 10) * 0.01, 0.8),
            }
            
            dashboard.update_progress(epoch - 10, metrics)
            time.sleep(0.3)
    
    console.print("\n✅ Demo completed!")


if __name__ == "__main__":
    demo_dashboard()
