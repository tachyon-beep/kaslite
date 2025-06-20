"""
Rich-powered CLI dashboard for real-time progress and metrics during morphogenetic experiments.

This is a simplified version, designed to be rebuilt.
"""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Mapping
from typing import Any, cast

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from morphogenetic_engine.events import (
    NetworkStrain,
    NetworkStrainGridUpdatePayload,
    SeedState,
    SeedStateUpdatePayload,
)


class RichDashboard:
    """A Rich CLI dashboard for experiment monitoring with progress bars."""

    GRID_SIZE = 16

    # EMOJI MAPS
    SEED_EMOJI_MAP = {
        SeedState.ACTIVE: "🟢",
        SeedState.DORMANT: "⚪",
        SeedState.BLENDING: "🟡",
        SeedState.GERMINATED: "🌱",
        SeedState.FOSSILIZED: "🦴",
    }

    STRAIN_EMOJI_MAP = {
        NetworkStrain.NONE: "🔵",
        NetworkStrain.LOW: "🟢",
        NetworkStrain.MEDIUM: "🟡",
        NetworkStrain.HIGH: "🔴",
        NetworkStrain.FIRED: "💥",
    }

    # Common
    EMPTY_CELL_EMOJI = "⚫"
    STYLE_BOLD_BLUE = "bold blue"

    def __init__(self, console: Console | None = None, experiment_params: dict[str, Any] | None = None):
        self.console = console or Console()
        self.experiment_params = experiment_params or {}
        self.layout: Layout | None = None
        self.live: Live | None = None
        self._layout_initialized = False
        self.last_events: deque[str] = deque(maxlen=20)
        self.seed_log_events: deque[str] = deque(maxlen=40)  # Larger buffer for seed events
        self.seed_states: dict[str, dict[str, Any]] = {}
        self.latest_metrics: dict[str, Any] = {}
        self.previous_metrics: dict[str, Any] = {}

        # Progress Bars
        self.total_progress = Progress(
            TextColumn("[bold blue]Overall Progress"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            expand=True,
        )
        self.phase_progress = Progress(
            TextColumn("[bold green]Phase Progress"),
            BarColumn(bar_width=None),
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

        # --- Title Header ---
        title_header = Layout(
            Panel(Align.center("[bold]QUICKSILVER[/bold]"), box=box.MINIMAL),
            name="title",
            size=3,
        )

        # --- Main Area Setup ---
        main_area = Layout(name="main")

        # --- Footer Setup for Progress Bars ---
        footer = Layout(name="footer", size=3)
        progress_layout = Layout()
        # Spacer added
        progress_layout.split_row(
            Layout(self.phase_progress, name="phase", ratio=1),
            Layout(Align.center(Text("•", style="dim")), name="spacer", size=3),
            Layout(self.total_progress, name="total", ratio=1),
        )
        footer.update(progress_layout)

        # --- Quit Message Footer ---
        quit_footer = Layout(
            Align.center(Text("Press 'q' to quit", style="dim")),
            name="quit_footer",
            size=1,
        )

        # Root layout splits into title, main content, and the footer
        self.layout.split(title_header, main_area, footer, quit_footer)

        # 3 Columns in Main
        self.layout["main"].split_row(
            Layout(name="left_column", ratio=7),
            Layout(name="center_column", ratio=5),
            Layout(name="right_column", ratio=7),
        )

        # --- Left Column ---
        bottom_left = Layout(name="bottom_left_area")
        bottom_left.split_column(
            Layout(name="event_log_panel"),
            Layout(name="seed_timeline_panel"),
        )
        self.layout["left_column"].split_column(
            Layout(name="top_left_area"),
            bottom_left,
        )
        self.layout["top_left_area"].split_column(
            Layout(name="top_row"),
            Layout(name="sparkline_panel"),
        )
        self.layout["top_row"].split_row(
            Layout(name="info_panel"),
            Layout(name="metrics_table_panel"),
        )

        # --- Center Column ---
        self.layout["center_column"].split_column(
            Layout(name="kasima_panel"),
            Layout(name="tamiyo_panel", size=13),
            Layout(name="karn_panel", size=9),
            Layout(name="crucible_panel", size=8),
        )

        # --- Right Column ---
        self.layout["right_column"].split_column(
            Layout(name="seed_box_area"),
            Layout(name="network_strain_area"),
        )
        self.layout["seed_box_area"].split_column(
            Layout(name="seed_box_panel"),
            Layout(name="seed_legend_panel", size=3),
        )
        self.layout["network_strain_area"].split_column(
            Layout(name="network_strain_panel"),
            Layout(name="strain_legend_panel", size=3),
        )

        # --- Populate Panels ---
        self.layout["info_panel"].update(self._create_info_panel())
        self.layout["metrics_table_panel"].update(self._create_metrics_table_panel())
        self.layout["sparkline_panel"].update(self._create_sparkline_panel())
        self.layout["event_log_panel"].update(self._create_event_log_panel())
        self.layout["kasima_panel"].update(self._create_kasima_panel())
        self.layout["tamiyo_panel"].update(self._create_tamiyo_panel())
        self.layout["karn_panel"].update(self._create_karn_panel())
        self.layout["crucible_panel"].update(self._create_crucible_panel())
        self.layout["seed_timeline_panel"].update(self._create_seed_timeline_panel())
        self.layout["seed_box_panel"].update(self._create_seed_box_panel())
        self.layout["seed_legend_panel"].update(self._create_seed_legend_panel())
        self.layout["network_strain_panel"].update(self._create_network_strain_panel())
        self.layout["strain_legend_panel"].update(self._create_strain_legend_panel())

        self._layout_initialized = True

    def _create_info_panel(self) -> Panel:
        """Generate the panel for experiment parameters."""
        info_table = Table(show_header=False, expand=True, box=box.MINIMAL, padding=(0, 1))
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

    def _create_metrics_table_panel(self) -> Panel:
        """Generate the panel for the live metrics table."""
        metrics_table = Table(show_header=True, header_style="bold magenta", expand=True)
        metrics_table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        metrics_table.add_column("Last", justify="center", style="green")
        metrics_table.add_column("Previous", justify="center", style="yellow")

        # Populate with latest and previous metrics
        metrics_to_show = sorted(self.latest_metrics.keys())
        for metric in metrics_to_show:
            current_val = self.latest_metrics.get(metric)
            last_val = self.previous_metrics.get(metric)

            current_str = f"{current_val:.4f}" if isinstance(current_val, float) else str(current_val)
            last_str = f"{last_val:.4f}" if isinstance(last_val, float) else "N/A"

            metrics_table.add_row(metric, current_str, last_str)

        return Panel(metrics_table, title="Live Metrics", border_style="cyan")

    def _create_sparkline_panel(self) -> Panel:
        """Generate the panel for metric sparklines."""
        sparkline_table = Table(show_header=True, header_style="bold cyan", expand=True, box=box.MINIMAL)
        sparkline_table.add_column("Metric", style="cyan", no_wrap=True)
        sparkline_table.add_column("Trend / Sparkline", style="green", ratio=2)
        sparkline_table.add_column("Average", justify="right")

        # Add rows for key metrics with placeholders
        metrics_to_track = ["train_loss", "val_acc"]
        for metric in metrics_to_track:
            latest_val = self.latest_metrics.get(metric)
            # For now, "Average" will just show the latest value.
            avg_val_str = f"{latest_val:.4f}" if isinstance(latest_val, float) else "N/A"

            sparkline_table.add_row(
                metric,
                "[dim]...plotting area...[/dim]",
                avg_val_str,
            )

        return Panel(sparkline_table, title="Trends", border_style="cyan")

    def _create_event_log_panel(self) -> Panel:
        """Generate the panel for the scrolling event log."""
        event_text = "\n".join(self.last_events)
        content = Text.from_markup(f"[bold]Experiment Log:[/bold]\n{event_text}")
        return Panel(content, title="Event Log", border_style="blue")

    def _get_seed_states_by_layer(
        self, num_layers: int, seeds_per_layer: int
    ) -> dict[int, list[SeedState | None]]:
        """Parse seed_states and organize them by layer and seed index."""
        layer_seeds: dict[int, list[SeedState | None]] = {
            i: [None] * seeds_per_layer for i in range(num_layers)
        }
        
        for seed_id, data in self.seed_states.items():
            layer_idx, seed_idx = self._parse_seed_id(seed_id)
            if (layer_idx is not None and seed_idx is not None and 
                layer_idx < num_layers and seed_idx < seeds_per_layer):
                state = data.get("state")
                if isinstance(state, SeedState):
                    layer_seeds[layer_idx][seed_idx] = state
        return layer_seeds
    
    def _parse_seed_id(self, seed_id) -> tuple[int | None, int | None]:
        """Extract layer and seed indices from a seed_id (tuple or string)."""
        try:
            # Handle tuple format: (layer_idx, seed_idx)
            if isinstance(seed_id, tuple) and len(seed_id) == 2:
                return seed_id
            
            # Handle string format: "L0_S1" or similar
            if isinstance(seed_id, str) and seed_id.startswith("L") and "_" in seed_id:
                parts = seed_id.split("_")
                layer_idx = int(parts[0][1:])
                seed_idx_str = parts[1]
                if seed_idx_str.startswith("S"):
                    seed_idx_str = seed_idx_str[1:]
                seed_idx = int(seed_idx_str)
                return layer_idx, seed_idx
        except (ValueError, IndexError, TypeError):
            pass
        
        return None, None

    def _get_network_strain_states_by_layer(
        self, num_layers: int, seeds_per_layer: int
    ) -> dict[int, list[NetworkStrain | None]]:
        """
        Parse and organize network strain data by layer and seed index.
        NOTE: This is a placeholder. It currently returns random data for visual demonstration.
        """
        layer_strains: dict[int, list[NetworkStrain | None]] = {
            i: [None] * seeds_per_layer for i in range(num_layers)
        }
        strain_choices = list(NetworkStrain)
        for i in range(num_layers):
            for j in range(seeds_per_layer):
                # Create some dynamic-looking random data
                if (i * seeds_per_layer + j + int(time.time() * 2)) % 7 < 2:
                    layer_strains[i][j] = random.choice(strain_choices)
                else:
                    layer_strains[i][j] = NetworkStrain.NONE
        return layer_strains

    def _create_grid_row(
        self,
        layer_index: int,
        num_layers: int,
        seeds_per_layer: int,
        layer_data: Mapping[int, list[SeedState | NetworkStrain | None]],
        emoji_map: Mapping[SeedState | NetworkStrain, str],
        empty_emoji: str,
    ) -> list[str]:
        """Helper to create a single row for a grid table."""
        row = [f"{layer_index + 1}"]
        if layer_index < num_layers:
            states = layer_data.get(layer_index, [])
            for j in range(seeds_per_layer):  # Use seeds_per_layer instead of GRID_SIZE
                emoji = empty_emoji
                if j < len(states):
                    state = states[j]
                    if state:
                        emoji = emoji_map.get(state, empty_emoji)
                row.append(emoji)
        else:
            row.extend([empty_emoji] * seeds_per_layer)  # Use seeds_per_layer instead of GRID_SIZE
        return row

    def _create_grid_table(
        self,
        grid: Mapping[int, list[SeedState | NetworkStrain | None]],
        emoji_map: Mapping[SeedState | NetworkStrain, str],
        num_layers: int,
        seeds_per_layer: int,
    ) -> Table:
        """Creates a Rich Table for a grid."""
        table = Table(
            title="", 
            show_header=True, 
            header_style="bold magenta", 
            box=box.ROUNDED,
            expand=False
        )
        table.add_column("L", justify="center")
        #table.add_column("", justify="center")  # Separator
        for i in range(seeds_per_layer):  # Use actual seeds_per_layer
            table.add_column(f"{i}", justify="center", width=3)

        for i in range(num_layers):  # Use actual num_layers
            row = self._create_grid_row(
                i, num_layers, seeds_per_layer, grid, emoji_map, self.EMPTY_CELL_EMOJI
            )
            table.add_row(*row)

        return table

    def _create_seed_box_panel(self) -> Panel:
        """Generate the main panel for visualizing seed states."""
        num_layers = self.experiment_params.get("num_layers", 8)
        seeds_per_layer = self.experiment_params.get("seeds_per_layer", 1)

        layer_data = self._get_seed_states_by_layer(num_layers, seeds_per_layer)

        grid_table = self._create_grid_table(
            cast(Any, layer_data), cast(Any, self.SEED_EMOJI_MAP), num_layers, seeds_per_layer
        )
        return Panel(
            Align.center(grid_table, vertical="middle"),
            title="Seed States",
            border_style="green",
        )

    def _create_seeds_training_table(self) -> Table:
        """Generate the table for detailed seed training metrics."""
        table = Table(show_header=True, header_style="bold yellow", expand=True)
        table.add_column("ID", style="cyan")
        table.add_column("State", style="green")
        table.add_column("α", justify="right", style="magenta")
        table.add_column("∇ Norm", justify="right", style="yellow")
        table.add_column("Pat.", justify="right", style="dim")

        # Sort seeds for stable display order
        sorted_seed_ids = sorted(self.seed_states.keys())

        for seed_id in sorted_seed_ids:
            data = self.seed_states[seed_id]
            state = data.get("state")
            if isinstance(state, SeedState):
                state_str = state.name.capitalize()
            else:
                state_str = str(state)

            alpha = data.get("alpha", 0.0)
            grad_norm = data.get("grad_norm")
            patience = data.get("patience")

            # Convert tuple seed_id back to string for display
            seed_id_str = f"L{seed_id[0]}_S{seed_id[1]}"

            table.add_row(
                seed_id_str,
                state_str,
                f"{alpha:.3f}",
                f"{grad_norm:.2e}" if grad_norm is not None else "N/A",
                str(patience) if patience is not None else "N/A",
            )

        return table

    def _create_kasima_panel(self) -> Panel:
        """Generate the panel containing the seeds training table."""
        if not self.seed_states:
            return Panel(
                Align.center("Waiting for seed data...", vertical="middle"),
                title="Kasima",
                border_style="green",
            )

        seeds_training_table = self._create_seeds_training_table()

        return Panel(seeds_training_table, title="Kasima", border_style="green")

    def _create_tamiyo_panel(self) -> Panel:
        """Generate the panel for Tamiyo's status."""
        tamiyo_table = Table(show_header=False, expand=True, box=box.MINIMAL, padding=(0, 1))
        tamiyo_table.add_column("Metric", style=self.STYLE_BOLD_BLUE)
        tamiyo_table.add_column("Value")

        tamiyo_table.add_row("Status", "Selecting Blueprint")
        tamiyo_table.add_row("Target Seed", "seed_4_layer_2")
        tamiyo_table.add_row("Strain Signature", "[High Loss, Low Grad Norm]")
        tamiyo_table.add_row("Candidate Blueprint", "[ID: 7b3d_v2] (Gated-SSM)")
        tamiyo_table.add_row("Predicted Utility", "Δ-Accuracy: +2.1%")
        tamiyo_table.add_row("Policy Loss", "0.198")
        tamiyo_table.add_row("Critic Loss", "0.051")
        tamiyo_table.add_row("Mean Reward (last 100)", "88.1")
        tamiyo_table.add_row("Action Entropy", "0.55")
        tamiyo_table.add_row()  # Whitespace

        return Panel(tamiyo_table, title="Tamiyo", border_style="blue")

    def _create_karn_panel(self) -> Panel:
        """Generate the panel for Karn's status."""
        karn_table = Table(show_header=False, expand=True, box=box.MINIMAL, padding=(0, 1))
        karn_table.add_column("Metric", style=self.STYLE_BOLD_BLUE)
        karn_table.add_column("Value")

        karn_table.add_row("Status", "Exploiting Keystone Lineage")
        karn_table.add_row("Archive Size", "21,249 Blueprints (81.2%)")
        karn_table.add_row("Archive Quality (Mean)", "74.5 ELO")
        karn_table.add_row("Critic Loss", "0.051")
        karn_table.add_row("Exploration/Exploitation", "Exploiting (80%)")
        karn_table.add_row()  # Whitespace

        return Panel(karn_table, title="Karn", border_style="blue")

    def _create_crucible_table(self) -> Table:
        """Generate the table for the Crucible panel with specific formatting."""
        table = Table(show_header=False, expand=True, box=box.MINIMAL, padding=(0, 1))
        table.add_column("Metric", style=self.STYLE_BOLD_BLUE, ratio=1)
        table.add_column("Value", ratio=1)

        # Static key-value pairs as requested, formatted like Tamiyo/Karn panels.
        table.add_row(Text("ID", style="blue"), Text("---", style="dim"))
        table.add_row(Text("State", style="blue"), Text("---", style="dim"))
        table.add_row(Text("α", style="blue"), Text("---", style="dim"))
        table.add_row(Text("∇ Norm", style="blue"), Text("---", style="dim"))
        table.add_row(Text("Pat.", style="blue"), Text("---", style="dim"))

        return table

    def _create_crucible_panel(self) -> Panel:
        """Generate the panel for detailed seed training data."""
        return Panel(
            self._create_crucible_table(),
            title="Crucible",
            border_style="yellow",
        )

    def _create_network_strain_panel(self) -> Panel:
        """Generate the panel for visualizing network strain."""
        num_layers = self.experiment_params.get("num_layers", 8)
        seeds_per_layer = self.experiment_params.get("seeds_per_layer", 1)

        layer_data = self._get_network_strain_states_by_layer(
            num_layers, seeds_per_layer
        )

        grid_table = self._create_grid_table(
            cast(Any, layer_data), cast(Any, self.STRAIN_EMOJI_MAP), num_layers, seeds_per_layer
        )
        return Panel(
            Align.center(grid_table, vertical="middle"),
            title="Network Strain",
            border_style="red",
        )

    def _create_seed_legend_panel(self) -> Panel:
        """Generate the legend for the seed box."""
        legend_text = Text.from_markup(
            f"{self.SEED_EMOJI_MAP[SeedState.ACTIVE]} Active  "
            f"{self.SEED_EMOJI_MAP[SeedState.BLENDING]} Blending  "
            f"{self.SEED_EMOJI_MAP[SeedState.GERMINATED]} Germinated  "
            f"{self.SEED_EMOJI_MAP[SeedState.DORMANT]} Dormant  "
            f"{self.SEED_EMOJI_MAP[SeedState.FOSSILIZED]} Fossilized  "
            f"{self.EMPTY_CELL_EMOJI} Empty"
        )
        return Panel(
            Align.center(legend_text),
            box=box.MINIMAL,
            style="dim",
            border_style="green",
            padding=(0, 1),
        )

    def _create_strain_legend_panel(self) -> Panel:
        """Generate the legend for the network strain."""
        legend_text = Text.from_markup(
            f"{self.STRAIN_EMOJI_MAP[NetworkStrain.NONE]} None  "
            f"{self.STRAIN_EMOJI_MAP[NetworkStrain.LOW]} Low  "
            f"{self.STRAIN_EMOJI_MAP[NetworkStrain.MEDIUM]} Medium  "
            f"{self.STRAIN_EMOJI_MAP[NetworkStrain.HIGH]} High  "
            f"{self.STRAIN_EMOJI_MAP[NetworkStrain.FIRED]} Fired"
        )
        return Panel(
            Align.center(legend_text),
            box=box.MINIMAL,
            style="dim",
            border_style="red",
            padding=(0, 1),
        )

    def _create_seed_timeline_panel(self) -> Panel:
        """Generate the panel for the seed event log."""
        event_text = "\n".join(self.seed_log_events)
        content = Text.from_markup(f"[bold]Seed Events:[/bold]\n{event_text}")
        return Panel(content, title="Seed Timeline", border_style="red")

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
            self.layout["seed_timeline_panel"].update(self._create_seed_timeline_panel())

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

        # Update the metrics panels with the latest data
        if self.layout:
            self.layout["metrics_table_panel"].update(self._create_metrics_table_panel())
            self.layout["sparkline_panel"].update(self._create_sparkline_panel())

    def update_seed(
        self,
        seed_id: str,
        state: SeedState,
        alpha: float = 0.0,
        from_state: str = "unknown",
        activation_epoch: int | None = None,
        grad_norm: float | None = None,
        weight_norm: float | None = None,
        patience: int | None = None,
    ):
        """Update the state and metrics of a single seed."""
        if seed_id not in self.seed_states:
            self.seed_states[seed_id] = {}

        # Store the raw enum member
        self.seed_states[seed_id]["state"] = state
        self.seed_states[seed_id]["alpha"] = alpha
        if activation_epoch is not None:
            self.seed_states[seed_id]["activation_epoch"] = activation_epoch
        if grad_norm is not None:
            self.seed_states[seed_id]["grad_norm"] = grad_norm
        if weight_norm is not None:
            self.seed_states[seed_id]["weight_norm"] = weight_norm
        if patience is not None:
            self.seed_states[seed_id]["patience"] = patience

        # The from_state is for logging, can remain a string
        message = f"Seed {seed_id} changed from {from_state} to {state.name}"
        self.add_seed_log_event(
            "SEED_STATE_CHANGE", message, {"seed_id": seed_id, "to_state": state.name}
        )

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

    def update_grid_view(self, payload: NetworkStrainGridUpdatePayload) -> None:
        """Receives a grid update and refreshes the view."""
        num_layers = self.experiment_params.get("num_layers", 8)
        seeds_per_layer = self.experiment_params.get("seeds_per_layer", 1)
        grid_table = self._create_grid_table(
            cast(Any, payload["grid"]), cast(Any, self.STRAIN_EMOJI_MAP), num_layers, seeds_per_layer
        )
        panel = Panel(
            Align.center(grid_table, vertical="middle"),
            title="Network Strain",
            border_style="red",
        )
        self.layout["network_strain_panel"].update(panel)

    def update_seeds_view(self, payload: SeedStateUpdatePayload) -> None:
        """Receives a seed state update and refreshes the view."""
        num_layers = self.experiment_params.get("num_layers", 8)
        seeds_per_layer = self.experiment_params.get("seeds_per_layer", 1)
        grid_table = self._create_grid_table(
            cast(Any, payload["grid"]), cast(Any, self.SEED_EMOJI_MAP), num_layers, seeds_per_layer
        )
        panel = Panel(
            Align.center(grid_table, vertical="middle"),
            title="Seed States",
            border_style="green",
        )
        self.layout["seed_box_panel"].update(panel)


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
    num_layers: int = cast(int, params.get("num_layers", 0))
    seeds_per_layer: int = cast(int, params.get("seeds_per_layer", 0))
    print("Starting dashboard demo...")
    with RichDashboard(console, experiment_params=params) as dashboard:
        dashboard.add_live_event("INFO", "Dashboard Initialized", {"run_id": "demo_run_123"})
        time.sleep(1)

        # --- Phase 1: Seeding and Initial Training ---
        dashboard.show_phase_transition("SEEDING", 0, total_epochs=50)
        for i in range(50):
            dashboard.update_progress(i, {"train_loss": 1.0 - i * 0.01, "val_acc": 0.5 + i * 0.005})
            if i < num_layers * seeds_per_layer:
                layer = i // seeds_per_layer
                seed_idx = i % seeds_per_layer
                if layer < num_layers:
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
            dashboard.update_progress(i, {"train_loss": 0.5 - (i - 50) * 0.008, "val_acc": 0.7 + (i - 50) * 0.003})
            if i == 75:
                dashboard.update_seed("L1_S1", "blending", from_state="blending", alpha=0.8)
                dashboard.add_seed_log_event("blending", "Seed L1_S1 blend factor increased.", {"alpha": 0.8})
            time.sleep(0.02)

        # --- Phase 3: Germination and Pruning ---
        dashboard.show_phase_transition("EVOLUTION", 100, from_phase="ACTIVATION", total_epochs=100)
        time.sleep(1)

        # Germinate a seed
        dashboard.update_seed("L1_S1", "germinated", from_state="blending")
        dashboard.show_germination_event("L1_S1", epoch=101)  # Specific event for seed log
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
            dashboard.update_progress(i, {"train_loss": 0.1 - (i - 100) * 0.0005, "val_acc": 0.85 + (i - 100) * 0.0001})
            time.sleep(0.02)

    console.print("\n[bold green]✅ Demo completed![/bold green]")


if __name__ == "__main__":
    demo_dashboard()
