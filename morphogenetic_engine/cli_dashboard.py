"""
Rich-powered CLI dashboard for real-time progress and metrics during morphogenetic experiments.

This is a simplified version, designed to be rebuilt.
"""
from __future__ import annotations
import time
from collections import deque
from typing import Any

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
    GRID_SIZE = 16

    # Seed Emojis
    SEED_ACTIVE_EMOJI = "🟢"
    SEED_DORMANT_EMOJI = "⚪"
    SEED_BLENDING_EMOJI = "🟡"
    SEED_GERMINATED_EMOJI = "🌱"
    SEED_FOSSILIZED_EMOJI = "🦴"

    # Strain Emojis
    STRAIN_NONE_EMOJI = "🔵"
    STRAIN_LOW_EMOJI = "🟢"
    STRAIN_MEDIUM_EMOJI = "🟡"
    STRAIN_HIGH_EMOJI = "🔴"
    STRAIN_FIRED_EMOJI = "💥"

    # Common
    EMPTY_CELL_EMOJI = "⚫"

    def __init__(self, console: Console | None = None, experiment_params: dict[str, Any] | None = None):
        self.console = console or Console()
        self.experiment_params = experiment_params or {}
        self.layout: Layout | None = None
        self.live: Live | None = None
        self._layout_initialized = False
        self.last_events: deque[str] = deque(maxlen=20)
        self.seed_log_events: deque[str] = deque(maxlen=40) # Larger buffer for seed events
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
        # Spacer removed, bars take equal space
        progress_layout.split_row(
            Layout(self.phase_progress, name="phase", ratio=1),
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
            Layout(name="right_column", ratio=8),
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
            Layout(name="kasima_panel", ratio=1),
            Layout(name="tamiyo_panel", ratio=1),
            Layout(name="karn_panel", ratio=1),
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
        self.layout["seed_timeline_panel"].update(self._create_seed_timeline_panel())
        self.layout["seed_box_panel"].update(self._create_seed_box_panel())
        self.layout["seed_legend_panel"].update(self._create_seed_legend_panel())
        self.layout["network_strain_panel"].update(self._create_network_strain_panel())
        self.layout["strain_legend_panel"].update(self._create_strain_legend_panel())

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

            current_str = (
                f"{current_val:.4f}" if isinstance(current_val, float) else str(current_val)
            )
            last_str = f"{last_val:.4f}" if isinstance(last_val, float) else "N/A"

            metrics_table.add_row(metric, current_str, last_str)

        return Panel(metrics_table, title="Live Metrics", border_style="cyan")

    def _create_sparkline_panel(self) -> Panel:
        """Generate the panel for metric sparklines."""
        sparkline_table = Table(
            show_header=True, header_style="bold cyan", expand=True, box=box.MINIMAL
        )
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

    def _get_network_strain_states_by_layer(
        self, num_layers: int, seeds_per_layer: int
    ) -> dict[int, list[str | None]]:
        """
        Parse and organize network strain data by layer and seed index.
        NOTE: This is a placeholder. It currently returns the same data as
        the seed states for visual demonstration.
        """
        # For now, just reuse the seed state logic.
        # In the future, this should query a different data source for strain.
        return self._get_seed_states_by_layer(num_layers, seeds_per_layer)

    def _create_grid_row(
        self,
        layer_index: int,
        num_layers: int,
        seeds_per_layer: int,
        layer_data: dict[int, list[str | None]],
        emoji_map: dict[str, str],
        empty_emoji: str,
    ) -> list[str]:
        """Helper to create a single row for a grid table."""
        row = [f"{layer_index + 1}", "│"]
        if layer_index < num_layers:
            states = layer_data.get(layer_index, [])
            for j in range(self.GRID_SIZE):
                emoji = empty_emoji
                if j < seeds_per_layer and j < len(states):
                    state = states[j]
                    if state:
                        emoji = emoji_map.get(state, empty_emoji)
                row.append(emoji)
        else:
            row.extend([empty_emoji] * self.GRID_SIZE)
        return row

    def _create_grid_table(
        self,
        num_layers: int,
        seeds_per_layer: int,
        layer_data: dict[int, list[str | None]],
        emoji_map: dict[str, str],
        empty_emoji: str,
    ) -> Table:
        """Creates the Rich Table for a grid display."""
        grid_table = Table(
            show_header=True,
            header_style="bold magenta",
            expand=False,
            box=box.SIMPLE_HEAD,
        )
        grid_table.add_column("L#", style="dim", width=3, justify="center")
        grid_table.add_column("│", width=1, no_wrap=True, style="dim")
        for i in range(self.GRID_SIZE):
            grid_table.add_column(
                str(i + 1), justify="center", no_wrap=True, style="bold"
            )

        for i in range(self.GRID_SIZE):
            row = self._create_grid_row(
                i, num_layers, seeds_per_layer, layer_data, emoji_map, empty_emoji
            )
            grid_table.add_row(*row)
        return grid_table

    def _create_seed_box_panel(self) -> Panel:
        """Generate the panel for the seed status grid."""
        num_layers = self.experiment_params.get("num_layers", 0)
        seeds_per_layer = self.experiment_params.get("seeds_per_layer", 0)

        if num_layers == 0 or seeds_per_layer == 0:
            return Panel(
                Align.center("Seed box data unavailable.", vertical="middle"),
                title="Seed Box",
                border_style="magenta",
            )

        emoji_map = {
            "active": self.SEED_ACTIVE_EMOJI,
            "dormant": self.SEED_DORMANT_EMOJI,
            "blending": self.SEED_BLENDING_EMOJI,
            "germinated": self.SEED_GERMINATED_EMOJI,
            "fossilized": self.SEED_FOSSILIZED_EMOJI,
        }
        empty_emoji = self.EMPTY_CELL_EMOJI
        layer_seeds = self._get_seed_states_by_layer(num_layers, seeds_per_layer)

        grid_table = self._create_grid_table(
            num_layers, seeds_per_layer, layer_seeds, emoji_map, empty_emoji
        )

        return Panel(
            Align.center(grid_table, vertical="middle"),
            title="Seed Box",
            border_style="magenta",
        )

    def _create_high_perf_seeds_panel(self) -> Panel:
        """Generate the panel for high-performing seeds."""
        table = Table(
            show_header=True, header_style="bold green", expand=True, box=box.MINIMAL
        )
        table.add_column("Seed ID", style="cyan", no_wrap=True, ratio=4)
        table.add_column("Act. Epoch", justify="center", ratio=5)
        table.add_column("Grad Norm", justify="center", ratio=5)
        table.add_column("Weight Norm", justify="center", ratio=5)
        table.add_column("Patience", justify="center", ratio=5)

        high_perf_seeds = {
            sid: d
            for sid, d in self.seed_states.items()
            if d.get("state") in ["active", "blending"]
        }

        for seed_id, data in sorted(high_perf_seeds.items()):
            table.add_row(
                seed_id[:12] + "...",
                str(data.get("activation_epoch", "N/A")),
                (f"{data.get('grad_norm'):.2e}" if data.get('grad_norm') is not None else "N/A"),
                (f"{data.get('weight_norm'):.2f}" if data.get('weight_norm') is not None else "N/A"),
                str(data.get("patience", "N/A")),
            )

        return Panel(table, title="High Performing Seeds", border_style="green")

    def _create_low_perf_seeds_panel(self) -> Panel:
        """Generate the panel for poorly-performing seeds."""
        table = Table(
            show_header=True, header_style="bold red", expand=True, box=box.MINIMAL
        )
        table.add_column("Seed ID", style="cyan", no_wrap=True, ratio=4)
        table.add_column("Act. Epoch", justify="center", ratio=5)
        table.add_column("Grad Norm", justify="center", ratio=5)
        table.add_column("Weight Norm", justify="center", ratio=5)
        table.add_column("Patience", justify="center", ratio=5)

        low_perf_seeds = {
            sid: d
            for sid, d in self.seed_states.items()
            if d.get("state") not in ["active", "blending"]
        }

        for seed_id, data in sorted(low_perf_seeds.items()):
            table.add_row(
                seed_id[:12] + "...",
                str(data.get("activation_epoch", "N/A")),
                (f"{data.get('grad_norm'):.2e}" if data.get('grad_norm') is not None else "N/A"),
                (f"{data.get('weight_norm'):.2f}" if data.get('weight_norm') is not None else "N/A"),
                str(data.get("patience", "N/A")),
            )

        return Panel(table, title="Poorly Performing Seeds", border_style="red")

    def _create_kasima_panel(self) -> Panel:
        """Generate the panel containing high and poor performing seed tables."""
        if not self.seed_states:
            return Panel(
                Align.center("Waiting for seed data...", vertical="middle"),
                title="Kasima",
                border_style="green",
            )

        metrics_layout = Layout(name="seed_metrics_split")
        metrics_layout.split_column(
            Layout(self._create_high_perf_seeds_panel()),
            Layout(self._create_low_perf_seeds_panel()),
        )
        return Panel(metrics_layout, title="Kasima", border_style="green")

    def _create_tamiyo_panel(self) -> Panel:
        """Generate the panel for Tamiyo's status."""
        tamiyo_table = Table(
            show_header=False, expand=True, box=box.MINIMAL, padding=(0, 1)
        )
        tamiyo_table.add_column("Metric", style="bold blue")
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

        return Panel(tamiyo_table, title="Tamiyo", border_style="blue")

    def _create_karn_panel(self) -> Panel:
        """Generate the panel for Karn's status."""
        karn_table = Table(
            show_header=False, expand=True, box=box.MINIMAL, padding=(0, 1)
        )
        karn_table.add_column("Metric", style="bold blue")
        karn_table.add_column("Value")

        karn_table.add_row("Status", "Exploiting Keystone Lineage")
        karn_table.add_row("Archive Health", "81.2% Filled (21k blueprints)")
        karn_table.add_row("Archive Quality (Mean)", "74.5 ELO")
        karn_table.add_row("Critic Loss", "0.051")
        karn_table.add_row("Exploration/Exploitation", "Exploiting (80%)")

        return Panel(karn_table, title="Karn", border_style="blue")

    def _create_network_strain_panel(self) -> Panel:
        """Generate the panel for the network strain visualization."""
        num_layers = self.experiment_params.get("num_layers", 0)
        seeds_per_layer = self.experiment_params.get("seeds_per_layer", 0)

        if num_layers == 0 or seeds_per_layer == 0:
            return Panel(
                Align.center("Network strain data unavailable.", vertical="middle"),
                title="Network Strain",
                border_style="yellow",
            )

        # Emoji map corresponding to the strain legend
        emoji_map = {
            "none": self.STRAIN_NONE_EMOJI,
            "low": self.STRAIN_LOW_EMOJI,
            "medium": self.STRAIN_MEDIUM_EMOJI,
            "high": self.STRAIN_HIGH_EMOJI,
            "fired": self.STRAIN_FIRED_EMOJI,
            # Mapping seed states to strain for placeholder visualization
            "active": self.STRAIN_LOW_EMOJI,  # Low strain
            "blending": self.STRAIN_MEDIUM_EMOJI, # Medium strain
            "germinated": self.STRAIN_NONE_EMOJI, # No strain
            "dormant": self.STRAIN_NONE_EMOJI, # No strain
        }
        empty_emoji = self.EMPTY_CELL_EMOJI

        # NOTE: This currently uses seed data as a placeholder for strain data.
        layer_data = self._get_network_strain_states_by_layer(
            num_layers, seeds_per_layer
        )

        grid_table = self._create_grid_table(
            num_layers, seeds_per_layer, layer_data, emoji_map, empty_emoji
        )

        return Panel(
            Align.center(grid_table, vertical="middle"),
            title="Network Strain",
            border_style="yellow",
        )

    def _create_seed_legend_panel(self) -> Panel:
        """Generate the legend for the seed box."""
        legend_text = Text.from_markup(
            f"{self.SEED_ACTIVE_EMOJI} Active  {self.SEED_BLENDING_EMOJI} Blending  "
            f"{self.SEED_GERMINATED_EMOJI} Germinated  {self.SEED_DORMANT_EMOJI} Dormant  "
            f"{self.SEED_FOSSILIZED_EMOJI} Fossilized  {self.EMPTY_CELL_EMOJI} Empty"
        )
        return Panel(
            Align.center(legend_text),
            box=box.MINIMAL,
            style="dim",
            border_style="magenta",
            padding=(0, 1),
        )

    def _create_strain_legend_panel(self) -> Panel:
        """Generate the legend for the network strain box."""
        legend_text = Text.from_markup(
            f"{self.STRAIN_NONE_EMOJI} None  {self.STRAIN_LOW_EMOJI} Low  "
            f"{self.STRAIN_MEDIUM_EMOJI} Medium  {self.STRAIN_HIGH_EMOJI} High  "
            f"{self.STRAIN_FIRED_EMOJI} Fired  {self.EMPTY_CELL_EMOJI} Empty"
        )
        return Panel(
            Align.center(legend_text),
            box=box.MINIMAL,
            style="dim",
            border_style="yellow",
            padding=(0, 1),
        )

    def _create_seed_timeline_panel(self) -> Panel:
        """Generate the panel for the scrolling seed-specific event log."""
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
        state: str,
        alpha: float = 0.0,
        from_state: str = "unknown",
        activation_epoch: int | None = None,
        grad_norm: float | None = None,
        weight_norm: float | None = None,
        patience: int | None = None,
    ):
        """Handle seed state change event and update the status panel."""
        # Log the event to the dedicated seed log
        log_data = {"alpha": f"{alpha:.2f}"}
        if grad_norm is not None:
            log_data["grad"] = f"{grad_norm:.2e}"
        self.add_seed_log_event(
            "seed", f"Seed {seed_id[:12]}...: {from_state} -> {state}", log_data
        )

        # Update the internal state for the right panels
        new_data = {
            "state": state,
            "alpha": alpha,
            "activation_epoch": activation_epoch,
            "grad_norm": grad_norm,
            "weight_norm": weight_norm,
            "patience": patience,
        }
        # Merge with existing data to preserve fields that aren't updated in this event
        self.seed_states.setdefault(seed_id, {}).update(
            {k: v for k, v in new_data.items() if v is not None}
        )

        if self.layout:
            self.layout["kasima_panel"].update(self._create_kasima_panel())
            self.layout["seed_box_panel"].update(self._create_seed_box_panel())
            self.layout["network_strain_panel"].update(self._create_network_strain_panel())

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
    num_layers: int = params.get("num_layers", 0)
    seeds_per_layer: int = params.get("seeds_per_layer", 0)
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
