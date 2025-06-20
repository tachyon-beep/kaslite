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
    MetricsUpdatePayload,
    NetworkStrain,
    NetworkStrainGridUpdatePayload,
    PhaseUpdatePayload,
    SeedLogPayload,
    SeedMetricsUpdatePayload,
    SeedState,
    SeedStateUpdatePayload,
    LogPayload,
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
        
        # Enhanced tracking for metrics and timing
        self.metrics_history: list[dict[str, Any]] = []
        self.epoch_start_time: float | None = None
        self.epoch_times: list[float] = []
        self.experiment_start_time: float = time.time()
        self.total_planned_epochs: int | None = None

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

        # Calculate total epochs from warm_up_epochs + adaptation_epochs
        warm_up_epochs = self.experiment_params.get("warm_up_epochs", 50)
        adaptation_epochs = self.experiment_params.get("adaptation_epochs", 50)
        self.total_epochs = warm_up_epochs + adaptation_epochs
        
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
            Layout(name="event_log_panel", ratio=1),
            Layout(name="seed_timeline_panel", ratio=1),
        )
        self.layout["left_column"].split_column(
            Layout(name="top_left_area", size=34),  # 20 + 14 = fixed total (was 31)
            bottom_left,
        )
        self.layout["top_left_area"].split_column(
            Layout(name="top_row", size=20),
            Layout(name="sparkline_panel", size=14),  # increased from 11 to 14
        )
        self.layout["top_row"].split_row(
            Layout(name="info_panel", ratio=2),
            Layout(name="metrics_table_panel", ratio=3),
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
        """Generate the panel for comprehensive experiment parameters."""
        info_table = Table(show_header=False, expand=True, box=box.MINIMAL, padding=(0, 0))
        info_table.add_column("Param", style="cyan", ratio=1)
        info_table.add_column("Value", style="white", ratio=1)

        # Architecture section
        arch_params = [
            ("hidden_dim", "Hidden Dim"),
            ("num_layers", "Layers"),
            ("seeds_per_layer", "Seeds/Layer"),
            ("input_dim", "Input Dim"),
        ]
        self._add_info_section(info_table, arch_params)

        # Training section
        train_params = [
            ("warm_up_epochs", "Warmup Epochs"),
            ("adaptation_epochs", "Adapt Epochs"),
            ("lr", "Learning Rate"),
            ("batch_size", "Batch Size"),
        ]
        self._add_info_section(info_table, train_params)

        # Data section
        data_params = [
            ("problem_type", "Problem"),
            ("n_samples", "Samples"),
            ("train_frac", "Train Fraction"),
        ]
        self._add_info_section(info_table, data_params)

        # System section
        sys_params = [
            ("device", "Device"),
            ("seed", "Random Seed"),
        ]
        self._add_info_section(info_table, sys_params)

        # Experiment section
        experiment_info = [
            ("start_time", "Started", self._format_start_time()),
            ("planned_epochs", "Total Epochs", str(self.total_planned_epochs) if self.total_planned_epochs else "Unknown"),
            ("current_epoch", "Current Epoch", str(self.latest_metrics.get("epoch_num", 0))),
        ]
        
        for i, (key, label, value) in enumerate(experiment_info):
            info_table.add_row(f"{label}:", value)

        return Panel(info_table, title="Experiment Info", border_style="yellow")

    def _add_info_section(self, table: Table, params: list[tuple[str, str]]) -> None:
        """Add a section of parameters to the info table."""
        for i, (key, label) in enumerate(params):
            value = self.experiment_params.get(key)
            value_str = str(value) if value is not None else "N/A"
            table.add_row(f"{label}:", value_str)

    def _format_start_time(self) -> str:
        """Format the experiment start time for display."""
        from datetime import datetime
        start_dt = datetime.fromtimestamp(self.experiment_start_time)
        return start_dt.strftime("%H:%M:%S")

    def _create_metrics_table_panel(self) -> Panel:
        """Generate the panel for the live metrics table with comprehensive training data."""
        metrics_table = Table(show_header=True, header_style="bold magenta", expand=True, box=box.MINIMAL, padding=(0, 0))
        metrics_table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        metrics_table.add_column("Curr", justify="center", style="green")
        metrics_table.add_column("Prev", justify="center", style="yellow")
        metrics_table.add_column("Chg", justify="center", style="red")

        # Define metrics to display with their formatting and priority
        metrics_config = self._get_metrics_display_config()

        for metric_key, display_name, fmt in metrics_config:
            current_str, previous_str, change_str = self._format_metric_row(metric_key, fmt)
            metrics_table.add_row(display_name, current_str, previous_str, change_str)

        return Panel(metrics_table, title="Live Training Metrics", border_style="cyan")

    def _get_metrics_display_config(self) -> list[tuple[str, str, str]]:
        """Get the configuration for metrics display."""
        return [
            # Core training metrics
            ("train_loss", "Train Loss", ".4f"),
            ("val_loss", "Val Loss", ".4f"),
            ("val_acc", "Val Accuracy", ".4f"),
            ("best_acc", "Best Accuracy", ".4f"),
            
            # Training dynamics
            ("loss_change", "Loss Δ", "+.4f"),
            ("acc_change", "Acc Δ", "+.4f"),
            ("loss_improvement", "Loss ↗", "+.4f"),
            
            # Optimization metrics
            ("learning_rate", "Learning Rate", ".2e"),
            
            # Timing and performance
            ("avg_epoch_time", "Epoch Time (s)", ".2f"),
            ("total_time", "Total Time (s)", ".0f"),
            ("samples_per_sec", "Samples/sec", ".1f"),
            ("eta_seconds", "ETA (s)", ".0f"),
        ]

    def _format_metric_row(self, metric_key: str, fmt: str) -> tuple[str, str, str]:
        """Format a single metric row for display."""
        current_val = self.latest_metrics.get(metric_key)
        previous_val = self.previous_metrics.get(metric_key)
        
        # Format current value
        current_str = self._format_metric_value(current_val, fmt)
        
        # Format previous value
        previous_str = self._format_metric_value(previous_val, fmt)
        
        # Calculate and format change
        change_str = self._calculate_metric_change(current_val, previous_val, fmt, metric_key)
        
        return current_str, previous_str, change_str

    def _format_metric_value(self, value: Any, fmt: str) -> str:
        """Format a single metric value."""
        if value is not None:
            if isinstance(value, float):
                return f"{value:{fmt}}"
            return str(value)
        return "N/A"

    def _calculate_metric_change(self, current_val: Any, previous_val: Any, fmt: str, metric_key: str = "") -> str:
        """Calculate and format the change between two metric values."""
        if current_val is not None and previous_val is not None and isinstance(current_val, (int, float)) and isinstance(previous_val, (int, float)):
            change = current_val - previous_val
            if abs(change) > 1e-6:  # Only show meaningful changes
                # Special handling for learning rate - show in exponential format
                if metric_key == "learning_rate":
                    return f"{change:+.2e}"
                # Use the same format as the metric, but with + sign for positive changes
                elif fmt.startswith("+"):
                    return f"{change:{fmt}}"
                else:
                    return f"{change:+{fmt}}"
            # Return zero with appropriate precision
            if ".3f" in fmt:
                return "0.000"
            elif ".2f" in fmt:
                return "0.00"
            elif ".1f" in fmt:
                return "0.0"
            elif ".0f" in fmt:
                return "0"
            elif ".2e" in fmt or metric_key == "learning_rate":
                return "0.00e+00"  # Show in exponential format for learning rate
            else:
                return "0.000"  # Default fallback
        return "N/A"

    def _create_sparkline_panel(self) -> Panel:
        """Generate the panel for metric trends and sparklines."""
        sparkline_table = Table(show_header=True, header_style="bold cyan", expand=True, box=box.MINIMAL)
        sparkline_table.add_column("Metric", style="cyan", no_wrap=True)
        sparkline_table.add_column("Current", justify="center", style="green")
        sparkline_table.add_column("Change", justify="center", style="yellow")
        sparkline_table.add_column("Trend", style="dim", ratio=4)

        # Define metrics useful for tracking trends
        trend_metrics = [
            ("train_loss", "Train Loss", ".3f"),
            ("val_loss", "Val Loss", ".3f"),
            ("val_acc", "Val Accuracy", ".3f"),
            ("loss_change", "Loss Δ", "+.3f"),
            ("acc_change", "Acc Δ", "+.3f"),
            ("loss_improvement", "Loss ↗", "+.3f"),
            ("learning_rate", "Learning Rate", ".2e"),
        ]

        for metric_key, display_name, fmt in trend_metrics:
            current_val = self.latest_metrics.get(metric_key)
            previous_val = self.previous_metrics.get(metric_key)
            
            # Format current value
            if current_val is not None:
                if isinstance(current_val, float):
                    current_str = f"{current_val:{fmt}}"
                else:
                    current_str = str(current_val)
            else:
                current_str = "N/A"
            
            # Calculate change
            change_str = self._calculate_metric_change(current_val, previous_val, fmt, metric_key)
            
            # Trend placeholder - longer sparkline to fill the expanded column
            trend_indicator = "[dim]···○···○···○···○···○···○···○···○···○···[/dim]" if current_val is not None else "[dim]----------------------------------------[/dim]"

            sparkline_table.add_row(display_name, current_str, change_str, trend_indicator)

        return Panel(sparkline_table, title="Trends", border_style="cyan")

    def _create_event_log_panel(self) -> Panel:
        """Generate the panel for the scrolling event log."""
        event_text = "\n".join(self.last_events)
        content = Text.from_markup(event_text)
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

    def _sort_seed_id_key(self, seed_id: str | tuple) -> tuple[int, int]:
        """Convert a seed ID (string like 'L1_S2' or tuple (1, 2)) into a sortable tuple."""
        parsed = self._parse_seed_id(seed_id)
        if parsed[0] is not None and parsed[1] is not None:
            return (parsed[0], parsed[1])
        return (9999, 9999)  # Place unparseable IDs at the end

    def _create_seeds_training_table(self) -> Table:
        """Generate the table for detailed seed training metrics."""
        table = Table(show_header=True, header_style="bold yellow", expand=True)
        table.add_column("ID", style="cyan")
        table.add_column("State", style="green")
        table.add_column("α", justify="right", style="magenta")
        table.add_column("∇ Norm", justify="right", style="yellow")
        table.add_column("Pat.", justify="right", style="dim")

        # Sort seeds for stable display order, handling string keys
        sorted_seed_ids = sorted(self.seed_states.keys(), key=self._sort_seed_id_key)

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

            table.add_row(
                seed_id,
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
        legend_parts = [
            f"{self.SEED_EMOJI_MAP[SeedState.ACTIVE]} Active",
            f"{self.SEED_EMOJI_MAP[SeedState.BLENDING]} Blending",
            f"{self.SEED_EMOJI_MAP[SeedState.GERMINATED]} Germinated",
            f"{self.SEED_EMOJI_MAP[SeedState.DORMANT]} Dormant",
            f"{self.SEED_EMOJI_MAP[SeedState.FOSSILIZED]} Fossilized",
            f"{self.EMPTY_CELL_EMOJI} Empty",
        ]
        legend_text = Text("  ".join(legend_parts))
        return Panel(
            Align.center(legend_text),
            box=box.MINIMAL,
            style="dim",
            border_style="green",
            padding=(0, 1),
        )

    def _create_strain_legend_panel(self) -> Panel:
        """Generate the legend for the network strain."""
        legend_parts = [
            f"{self.STRAIN_EMOJI_MAP[NetworkStrain.NONE]} None",
            f"{self.STRAIN_EMOJI_MAP[NetworkStrain.LOW]} Low",
            f"{self.STRAIN_EMOJI_MAP[NetworkStrain.MEDIUM]} Medium",
            f"{self.STRAIN_EMOJI_MAP[NetworkStrain.HIGH]} High",
            f"{self.STRAIN_EMOJI_MAP[NetworkStrain.FIRED]} Fired",
        ]
        legend_text = Text("  ".join(legend_parts))
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
        content = Text.from_markup(event_text)
        return Panel(content, title="Seed Timeline", border_style="red")

    def _add_derived_metrics(self, epoch: int, timestamp: float) -> None:
        """Calculate and add derived metrics to latest_metrics."""
        # Timing metrics
        if self.epoch_times:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            self.latest_metrics["avg_epoch_time"] = avg_epoch_time
            
            # Estimate remaining time
            if self.total_planned_epochs and epoch > 0:
                remaining_epochs = self.total_planned_epochs - epoch
                eta_seconds = remaining_epochs * avg_epoch_time
                self.latest_metrics["eta_seconds"] = eta_seconds
        
        # Training progress metrics
        total_time = timestamp - self.experiment_start_time
        self.latest_metrics["total_time"] = total_time
        self.latest_metrics["epoch_num"] = epoch
        
        # Loss and accuracy derivatives if we have previous metrics
        if self.previous_metrics:
            # Loss change
            prev_loss = self.previous_metrics.get("train_loss")
            curr_loss = self.latest_metrics.get("train_loss")
            if prev_loss is not None and curr_loss is not None:
                loss_change = curr_loss - prev_loss
                self.latest_metrics["loss_change"] = loss_change
                self.latest_metrics["loss_improvement"] = -loss_change  # Negative change is improvement
            
            # Accuracy change
            prev_acc = self.previous_metrics.get("val_acc")
            curr_acc = self.latest_metrics.get("val_acc")
            if prev_acc is not None and curr_acc is not None:
                acc_change = curr_acc - prev_acc
                self.latest_metrics["acc_change"] = acc_change
        
        # Calculate learning rate if available in config
        base_lr = self.experiment_params.get("lr", 0.001)
        # For now, assume constant LR unless we get scheduler info
        self.latest_metrics["learning_rate"] = base_lr
        
        # Performance metrics (samples per second)
        if self.epoch_times:
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            if avg_time > 0:
                # Rough estimate - assumes one pass through data per epoch
                n_samples = self.experiment_params.get("n_samples", 1000)
                samples_per_sec = n_samples / avg_time
                self.latest_metrics["samples_per_sec"] = samples_per_sec

    def update_metrics(self, payload: MetricsUpdatePayload) -> None:
        """Receives a metrics update and refreshes the view."""
        epoch = payload["epoch"]
        metrics = payload["metrics"]
        timestamp = payload["timestamp"]

        if epoch < 0:
            self.log_event({"event_type": "error", "message": "Invalid epoch value", "data": {"epoch": epoch}})
            return

        # Track timing information
        current_time = time.time()
        if self.epoch_start_time is not None:
            epoch_duration = current_time - self.epoch_start_time
            self.epoch_times.append(epoch_duration)
            # Keep only recent epoch times for averaging
            if len(self.epoch_times) > 50:
                self.epoch_times = self.epoch_times[-50:]
        self.epoch_start_time = current_time

        # Update phase progress (left bar) - progress within current phase
        current_phase_epoch = epoch - self.phase_start_epoch
        self.phase_progress.update(self.phase_task, completed=current_phase_epoch + 1)

        # Update total progress (right bar) - overall experiment progress
        self.total_progress.update(self.total_task, completed=epoch + 1)

        # Store metrics for later use
        self.previous_metrics = self.latest_metrics.copy()
        self.latest_metrics = metrics.copy()
        
        # Add derived metrics
        self._add_derived_metrics(epoch, timestamp)
        
        # Store in history for trend analysis
        metrics_with_metadata = {
            "epoch": epoch,
            "timestamp": timestamp,
            **metrics
        }
        self.metrics_history.append(metrics_with_metadata)
        # Keep only recent history
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

        # To avoid clutter, only show a few key metrics in the event log
        simple_metrics = {
            "loss": f"{metrics.get('train_loss', 0.0):.4f}",
            "acc": f"{metrics.get('val_acc', 0.0):.4f}",
        }
        self.log_event({"event_type": "epoch", "message": f"Epoch {epoch} complete", "data": simple_metrics})

        # Update the metrics panels with the latest data
        if self.layout:
            self.layout["metrics_table_panel"].update(self._create_metrics_table_panel())
            self.layout["sparkline_panel"].update(self._create_sparkline_panel())

    def update_seed_metrics(self, payload: SeedMetricsUpdatePayload) -> None:
        """Update the state and metrics of a single seed from a payload."""
        seed_id = payload["seed_id"]
        
        # Normalize seed_id to string representation
        if isinstance(seed_id, tuple):
            seed_id = f"L{seed_id[0]}_S{seed_id[1]}"

        if not isinstance(seed_id, str) or not seed_id:
            self.log_event({"event_type": "error", "message": "Invalid seed_id", "data": {"seed_id": seed_id}})
            return

        if seed_id not in self.seed_states:
            self.seed_states[seed_id] = {}

        # Update state if present
        if "state" in payload:
            state = payload["state"]
            if isinstance(state, str):
                try:
                    state = SeedState[state.upper()]
                except KeyError:
                    self.log_event({
                        "event_type": "error", 
                        "message": f"Invalid seed state value: '{state}'", 
                        "data": {"seed_id": seed_id}
                    })
                    return
            self.seed_states[seed_id]["state"] = state

        # Update other metrics from the payload
        for key, value in payload.items():
            if key not in ["seed_id", "state"] and value is not None:
                self.seed_states[seed_id][key] = value
        
        # Refresh the Kasima panel to show the updated data
        if self.layout:
            self.layout["kasima_panel"].update(self._create_kasima_panel())

    def transition_phase(self, payload: PhaseUpdatePayload) -> None:
        """Handle phase transition event and reset phase progress bar."""
        self.log_event({
            "event_type": "phase_transition", 
            "message": f"Moving to {payload['to_phase']}", 
            "data": {"from": payload['from_phase'], "epoch": payload['epoch']}
        })

        if payload.get("total_epochs_in_phase") is not None:
            self.phase_start_epoch = payload['epoch']
            self.current_phase_epochs = payload['total_epochs_in_phase']
            self.phase_progress.reset(self.phase_task)
            self.phase_progress.update(
                self.phase_task,
                total=payload['total_epochs_in_phase'],
                completed=0,
                description=payload['to_phase'].replace("_", " ").title(),
            )

    def log_event(self, payload: LogPayload) -> None:
        """Add a new event to be displayed in the main experiment log."""
        event_str = f"[{payload['event_type'].upper()}] {payload['message']}"
        data = payload.get("data")
        if data:
            data_str = ", ".join([f"{k}={v}" for k, v in data.items() if v is not None])
            if data_str:
                event_str += f" ({data_str})"
        self.last_events.append(event_str)
        if self.layout:
            self.layout["event_log_panel"].update(self._create_event_log_panel())

    def log_seed_event(self, payload: SeedLogPayload) -> None:
        """Add a new event to the dedicated seed log."""
        event_str = f"[{payload['event_type'].upper()}] {payload['message']}"
        data = payload.get("data")
        if data:
            data_str = ", ".join([f"{k}={v}" for k, v in data.items() if v is not None])
            if data_str:
                event_str += f" ({data_str})"
        self.seed_log_events.append(event_str)
        if self.layout:
            self.layout["seed_timeline_panel"].update(self._create_seed_timeline_panel())

    def add_live_event(self, event_type: str, message: str, data: dict[str, Any]):
        """DEPRECATED: Use log_event with LogPayload instead."""
        self.log_event({"event_type": event_type, "message": message, "data": data})

    def add_seed_log_event(self, event_type: str, message: str, data: dict[str, Any]):
        """DEPRECATED: Use log_seed_event with SeedLogPayload instead."""
        self.log_seed_event({"event_type": event_type, "message": message, "data": data})
        
    def update_progress(self, epoch: int, metrics: dict[str, Any]):
        """DEPRECATED: Use update_metrics with MetricsUpdatePayload instead."""
        import time
        self.update_metrics({"epoch": epoch, "metrics": metrics, "timestamp": time.time()})

    def update_seed(
        self,
        seed_id: str | tuple,
        state: SeedState | str,
        alpha: float = 0.0,
        from_state: str = "unknown",
        grad_norm: float | None = None,
        patience: int | None = None,
    ):
        """DEPRECATED: Use update_seed_metrics with SeedMetricsUpdatePayload instead."""
        payload = SeedMetricsUpdatePayload(
            seed_id=seed_id,
            state=state,
            alpha=alpha,
            grad_norm=grad_norm,
            patience=patience,
        )
        self.update_seed_metrics(payload)
        
        # Log the state change event separately
        if "state" in payload:
            message = f"Seed {seed_id} changed from {from_state} to {state.name if isinstance(state, SeedState) else state}"
            self.log_seed_event({
                "event_type": "SEED_STATE_CHANGE", 
                "message": message, 
                "data": {"seed_id": seed_id, "to_state": state.name if isinstance(state, SeedState) else state}
            })

    def show_phase_transition(
        self, to_phase: str, epoch: int, from_phase: str = "", total_epochs: int | None = None
    ):
        """DEPRECATED: Use transition_phase with PhaseTransitionPayload instead."""
        import time
        self.transition_phase({
            "to_phase": to_phase, 
            "epoch": epoch, 
            "from_phase": from_phase, 
            "total_epochs_in_phase": total_epochs,
            "timestamp": time.time()
        })

    def show_germination_event(self, seed_id: str, epoch: int):
        """DEPRECATED: Use log_seed_event instead."""
        self.log_seed_event({
            "event_type": "germination", 
            "message": f"Seed {seed_id[:12]}... germinated!", 
            "data": {"epoch": epoch}
        })

    def start(self):
        """Start the live dashboard display, with error handling."""
        if self.live:
            return
        try:
            self._setup_layout()
            if self.layout:
                self.live = Live(
                    self.layout,
                    console=self.console,
                    screen=True,
                    auto_refresh=True,
                )
                self.live.start()
        except (Exception, KeyboardInterrupt):
            # If start fails, ensure we stop to restore the terminal.
            if self.live:
                self.live.stop()
            # Re-raise the exception to not swallow it.
            raise

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

    def update_network_strain_grid(self, payload: NetworkStrainGridUpdatePayload) -> None:
        """Receives a grid update and refreshes the view."""
        if not self.layout:
            return
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

    def update_seed_states_grid(self, payload: SeedStateUpdatePayload) -> None:
        """Receives a seed state update and refreshes the view."""
        if not self.layout:
            return
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

    def initialize_experiment(self, config: dict[str, Any]) -> None:
        """Initialize experiment with configuration data."""
        self.experiment_params.update(config)
        
        # Calculate total planned epochs from config
        warm_up = config.get("warm_up_epochs", 0)
        adaptation = config.get("adaptation_epochs", 0)
        self.total_planned_epochs = warm_up + adaptation
        
        # Update total progress bar with planned epochs
        if self.total_planned_epochs > 0:
            self.total_progress.update(self.total_task, total=self.total_planned_epochs)
