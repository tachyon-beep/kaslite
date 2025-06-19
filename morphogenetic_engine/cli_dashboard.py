"""
Rich-powered CLI dashboard for real-time progress and metrics during morphogenetic experiments.

This is a simplified version, designed to be rebuilt.
"""
from __future__ import annotations
import time
from collections import deque
from typing import Any, Deque, Optional

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box


class RichDashboard:
    """A Rich CLI dashboard for experiment monitoring with progress bars."""

    def __init__(self, console: Optional[Console] = None, experiment_params: Optional[dict[str, Any]] = None):
        self.console = console or Console()
        self.experiment_params = experiment_params or {}
        self.layout: Optional[Layout] = None
        self.live: Optional[Live] = None
        self._layout_initialized = False
        self.last_events: Deque[str] = deque(maxlen=20)
        self.seed_log_events: Deque[str] = deque(maxlen=40) # Larger buffer for seed events
        self.seed_states: dict[str, dict[str, Any]] = {}
        self.latest_metrics: dict[str, Any] = {}
        self.previous_metrics: dict[str, Any] = {}

        # Progress Bars
        self.total_progress = Progress(
            TextColumn("[bold blue]Overall Progress"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            expand=True,
        )
        self.phase_progress = Progress(
            TextColumn("[bold green]Phase Progress"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            expand=True,
        )

        self.total_epochs = self.experiment_params.get("epochs", 100)
        self.total_task = self.total_progress.add_task("Overall", total=self.total_epochs)
        self.phase_task = self.phase_progress.add_task("Phase", total=100)  # Default, will be updated

        self.phase_start_epoch = 0
        self.current_phase_epochs = 0

    def _setup_layout(self):
        """Initialize the dashboard layout."""
        if self._layout_initialized:
            return

        self.layout = Layout(name="root")

        # Create a header for progress bars
        header = Layout(name="header", size=3)
        progress_layout = Layout()
        progress_layout.split_row(
            Layout(self.total_progress, name="total", ratio=2),
            Layout(name="spacer", ratio=1),  # This is the gap
            Layout(self.phase_progress, name="phase", ratio=2),
        )
        header.update(progress_layout)

        # --- Main Area Setup ---
        main_area = Layout(name="main")
        left_column = Layout(name="left_column", ratio=2)    # 40%
        right_column = Layout(name="right_column", ratio=3) # 60%

        # Split the left column (top/bottom)
        top_left_area = Layout(name="top_left_area")
        top_left_area.split_row(
            Layout(name="info_panel"), Layout(name="metrics_panel")
        )
        left_column.split_column(
            top_left_area,
            Layout(name="event_log_panel", ratio=1),
        )

        # Split the right column (top/bottom)
        top_right_area = Layout(name="top_right_area")
        top_right_area.split_row(
            Layout(name="seed_metrics_panel", ratio=1),  # Swapped and renamed
            Layout(name="seed_box_panel", ratio=2),      # Swapped and renamed
        )
        right_column.split_column(
            top_right_area,
            Layout(name="seed_log_panel")
        )

        main_area.split_row(left_column, right_column)
        self.layout.split(header, main_area)

        # --- Populate Panels ---
        self.layout["info_panel"].update(self._create_info_panel())
        self.layout["metrics_panel"].update(self._create_metrics_panel())
        self.layout["event_log_panel"].update(self._create_event_log_panel())
        self.layout["seed_box_panel"].update(self._create_seed_box_panel())
        self.layout["seed_metrics_panel"].update(self._create_seed_metrics_panel())
        self.layout["seed_log_panel"].update(self._create_seed_log_panel())

        self._layout_initialized = True

    def _create_info_panel(self) -> Panel:
        """Generate the panel for experiment parameters."""
        info_table = Table(
            show_header=False, expand=True, box=box.MINIMAL, padding=(0, 1)
        )
        info_table.add_column("Param", style="bold yellow")
        info_table.add_column("Value")

        # Add a few key parameters
        params_to_show = {
            "problem_type": "Problem",
            "n_samples": "Samples",
            "input_dim": "Input Dim",
            "learning_rate": "LR",
            "seed": "Seed",
        }
        for key, label in params_to_show.items():
            value = self.experiment_params.get(key)
            if value is not None:
                info_table.add_row(f"{label}:", str(value))

        return Panel(info_table, title="Info", border_style="yellow")

    def _create_metrics_panel(self) -> Panel:
        """Generate the panel for the live metrics table."""
        metrics_table = Table(show_header=True, header_style="bold magenta", expand=True)
        metrics_table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
        metrics_table.add_column("Value", justify="center", style="green")
        metrics_table.add_column("Trend", justify="center")

        # The table is left blank for now, as requested.

        return Panel(metrics_table, title="Live Metrics", border_style="cyan")

    def _create_event_log_panel(self) -> Panel:
        """Generate the panel for the scrolling event log."""
        event_text = "\n".join(self.last_events)
        content = Text.from_markup(f"[bold]Experiment Log:[/bold]\n{event_text}")
        return Panel(content, title="Event Log", border_style="blue")

    def _get_seed_states_by_layer(
        self, num_layers: int, seeds_per_layer: int
    ) -> dict[int, list[str | None]]:
        """Parse seed_states and organize them by layer and seed index."""
        layer_seeds: dict[int, list[str | None]] = {
            i: [None] * seeds_per_layer for i in range(num_layers)
        }
        for seed_id, data in self.seed_states.items():
            if not seed_id.startswith("L") or "_" not in seed_id:
                continue
            try:
                parts = seed_id.split("_")
                layer_idx = int(parts[0][1:])
                seed_idx = int(parts[1])
                if layer_idx < num_layers and seed_idx < seeds_per_layer:
                    layer_seeds[layer_idx][seed_idx] = data.get("state")
            except (ValueError, IndexError):
                continue  # Ignore malformed IDs
        return layer_seeds

    def _create_seed_box_panel(self) -> Panel:
        """Generate the panel for the seed status grid."""
        num_layers = self.experiment_params.get("num_layers", 0)
        seeds_per_layer = self.experiment_params.get("seeds_per_layer", 0)

        emoji_map = {
            "active": "🟢",
            "dormant": "⚪",
            "blending": "🟡",
            "germinated": "🌱",
        }
        empty_emoji = "⚫"

        grid_table = Table(
            show_header=True,
            header_style="bold magenta",
            expand=False,
            box=box.SIMPLE_HEAD,  # A box style without internal vertical lines
        )
        # L# column
        grid_table.add_column("L#", style="dim", width=3, justify="center")
        # Visual separator column
        grid_table.add_column("│", width=1, no_wrap=True, style="dim")
        # Seed columns
        for i in range(16):
            grid_table.add_column(
                str(i + 1), justify="center", no_wrap=True, style="bold"
            )

        if num_layers == 0 or seeds_per_layer == 0:
            return Panel(
                Align.center("Seed box data unavailable.", vertical="middle"),
                title="Seed Box",
                border_style="magenta",
            )

        layer_seeds = self._get_seed_states_by_layer(num_layers, seeds_per_layer)

        # Always display 16 layer rows
        for i in range(16):
            # Start row with layer number (1-indexed) and the separator
            row = [f"{i + 1}", "│"]

            # Check if the current layer index is valid for the experiment
            if i < num_layers:
                states = layer_seeds.get(i, [])
                for j in range(16):  # Populate 16 seed columns
                    emoji = empty_emoji
                    if j < seeds_per_layer and j < len(states):
                        state = states[j]
                        emoji = emoji_map.get(state, empty_emoji)
                    row.append(emoji)
            else:
                # If layer does not exist, fill row with empty emojis
                row.extend([empty_emoji] * 16)

            grid_table.add_row(*row)

        return Panel(grid_table, title="Seed Box", border_style="magenta")

    def _create_seed_metrics_panel(self) -> Panel:
        """Generate the panel for displaying detailed seed statuses."""
        if not self.seed_states:
            content = Align.center("Waiting for seed data...", vertical="middle")
        else:
            status_text = ""
            for seed_id, data in sorted(self.seed_states.items()):
                state = data.get("state", "unknown")
                alpha = data.get("alpha")
                if state == "blending":
                    status_text += f"• {seed_id[:12]}...: [yellow]{state}[/yellow] (α={alpha:.2f})\n"
                elif state == "active":
                    status_text += f"• {seed_id[:12]}...: [bold green]{state}[/bold green]\n"
                else:
                    status_text += f"• {seed_id[:12]}...: {state}\n"
            content = Text.from_markup(status_text)

        return Panel(content, title="Seed Metrics", border_style="green")

    def _create_seed_log_panel(self) -> Panel:
        """Generate the panel for the scrolling seed-specific event log."""
        event_text = "\n".join(self.seed_log_events)
        content = Text.from_markup(f"[bold]Seed Events:[/bold]\n{event_text}")
        return Panel(content, title="Seed Log", border_style="red")

    def add_live_event(self, event_type: str, message: str, data: dict[str, Any]):
        """Add a new event to be displayed in the main experiment log."""
        event_str = f"[{event_type.upper()}] {message}"
        if data:
            data_str = ", ".join([f"{k}={v}" for k, v in data.items() if v is not None])
            if data_str:
                event_str += f" ({data_str})"
        self.last_events.append(event_str)
        if self.layout:
            self.layout["event_log_panel"].update(self._create_event_log_panel())

    def add_seed_log_event(self, event_type: str, message: str, data: dict[str, Any]):
        """Add a new event to the dedicated seed log."""
        event_str = f"[{event_type.upper()}] {message}"
        if data:
            data_str = ", ".join([f"{k}={v}" for k, v in data.items() if v is not None])
            if data_str:
                event_str += f" ({data_str})"
        self.seed_log_events.append(event_str)
        if self.layout:
            self.layout["seed_log_panel"].update(self._create_seed_log_panel())

    def update_progress(self, epoch: int, metrics: dict[str, Any]):
        """Handle epoch progress event and update progress bars."""
        # Update total progress
        self.total_progress.update(self.total_task, completed=epoch + 1)

        # Update phase progress
        current_phase_epoch = epoch - self.phase_start_epoch
        self.phase_progress.update(self.phase_task, completed=current_phase_epoch + 1)

        # Store metrics for later use
        self.previous_metrics = self.latest_metrics.copy()
        self.latest_metrics = metrics

        # To avoid clutter, only show a few key metrics in the event log
        simple_metrics = {
            "loss": f"{metrics.get('train_loss', 0.0):.4f}",
            "acc": f"{metrics.get('val_acc', 0.0):.4f}",
        }
        self.add_live_event("epoch", f"Epoch {epoch} complete", simple_metrics)

        # The metrics panel is not updated here yet, per user request.

    def update_seed(self, seed_id: str, state: str, alpha: float = 0.0, from_state: str = "unknown"):
        """Handle seed state change event and update the status panel."""
        # Log the event to the dedicated seed log
        self.add_seed_log_event(
            "seed", f"Seed {seed_id[:12]}...: {from_state} -> {state}", {"alpha": f"{alpha:.2f}"}
        )

        # Update the internal state for the right panels
        self.seed_states[seed_id] = {"state": state, "alpha": alpha}
        if self.layout:
            self.layout["seed_metrics_panel"].update(self._create_seed_metrics_panel())
            self.layout["seed_box_panel"].update(self._create_seed_box_panel())

    def show_phase_transition(
        self, to_phase: str, epoch: int, from_phase: str = "", total_epochs: int | None = None
    ):
        """Handle phase transition event and reset phase progress bar."""
        self.add_live_event("phase_transition", f"Moving to {to_phase}", {"from": from_phase, "epoch": epoch})

        if total_epochs is not None:
            self.phase_start_epoch = epoch
            self.current_phase_epochs = total_epochs
            self.phase_progress.reset(self.phase_task)
            self.phase_progress.update(
                self.phase_task,
                total=total_epochs,
                completed=0,
                description=to_phase.replace("_", " ").title(),
            )

    def show_germination_event(self, seed_id: str, epoch: int):
        """Handle seed germination event."""
        self.add_seed_log_event("germination", f"Seed {seed_id[:12]}... germinated!", {"epoch": epoch})

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
    """Demo function to showcase the dashboard's new layout."""
    console = Console()
    params = {
        "epochs": 200,
        "num_layers": 5,
        "seeds_per_layer": 8,  # Match the grid columns
        "problem_type": "spirals",
        "n_samples": 1000,
        "input_dim": 3,
        "learning_rate": 0.001,
        "seed": 42,
    }
    print("Starting dashboard demo...")
    with RichDashboard(console, experiment_params=params) as dashboard:
        dashboard.add_live_event("INFO", "Dashboard Initialized", {"run_id": "demo_run_123"})
        time.sleep(1)

        # --- Phase 1: Seeding and Initial Training ---
        dashboard.show_phase_transition("SEEDING", 0, total_epochs=50)
        for i in range(50):
            dashboard.update_progress(i, {"train_loss": 1.0 - i * 0.01, "val_acc": 0.5 + i * 0.005})
            if i < params["num_layers"] * params["seeds_per_layer"]:
                layer = i // params["seeds_per_layer"]
                seed_idx = i % params["seeds_per_layer"]
                if layer < params["num_layers"]:
                    seed_id = f"L{layer}_S{seed_idx}"
                    dashboard.update_seed(seed_id, "dormant")
            time.sleep(0.02)
        dashboard.add_live_event("INFO", "Seeding complete", {"seeded_count": 40})

        # --- Phase 2: Activation and Blending ---
        dashboard.show_phase_transition("ACTIVATION", 50, from_phase="SEEDING", total_epochs=50)
        time.sleep(1)
        # Activate some seeds
        for i in range(4):
            dashboard.update_seed(f"L{i}_S{i}", "active", from_state="dormant")
            dashboard.add_seed_log_event("activation", f"Seed L{i}_S{i} activated.", {"epoch": 50 + i})
            time.sleep(0.2)

        # Blend a seed
        dashboard.update_seed("L1_S1", "blending", from_state="active", alpha=0.3)
        dashboard.add_seed_log_event("blending", "Seed L1_S1 starts blending.", {"alpha": 0.3})
        time.sleep(0.5)

        for i in range(50, 100):
            dashboard.update_progress(
                i, {"train_loss": 0.5 - (i - 50) * 0.008, "val_acc": 0.7 + (i - 50) * 0.003}
            )
            if i == 75:
                dashboard.update_seed("L1_S1", "blending", from_state="blending", alpha=0.8)
                dashboard.add_seed_log_event("blending", "Seed L1_S1 blend factor increased.", {"alpha": 0.8})
            time.sleep(0.02)

        # --- Phase 3: Germination and Pruning ---
        dashboard.show_phase_transition("EVOLUTION", 100, from_phase="ACTIVATION", total_epochs=100)
        time.sleep(1)

        # Germinate a seed
        dashboard.update_seed("L1_S1", "germinated", from_state="blending")
        dashboard.show_germination_event("L1_S1", epoch=101) # Specific event for seed log
        time.sleep(0.5)

        # Prune a seed
        dashboard.update_seed("L3_S3", "pruned", from_state="active")
        dashboard.add_seed_log_event("pruning", "Seed L3_S3 pruned due to low performance.", {"reason": "low_val_acc"})
        time.sleep(0.5)

        # A seed fails
        dashboard.update_seed("L0_S0", "failed", from_state="active")
        dashboard.add_seed_log_event("error", "Seed L0_S0 failed during update.", {"details": "NaN loss detected"})
        time.sleep(0.5)


        for i in range(100, 200):
            dashboard.update_progress(
                i, {"train_loss": 0.1 - (i - 100) * 0.0005, "val_acc": 0.85 + (i - 100) * 0.0001}
            )
            time.sleep(0.02)

    console.print("\n[bold green]✅ Demo completed![/bold green]")


if __name__ == "__main__":
    demo_dashboard()
