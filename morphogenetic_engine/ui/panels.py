"""
Panel factories for the UI dashboard.

This module contains factory methods for creating various Rich panels
used throughout the dashboard interface, including static panels for
agents and information display.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from morphogenetic_engine.events import SeedState

from .config import STYLE_BOLD_BLUE


class PanelFactory:
    """Factory class for creating various dashboard panels."""

    def __init__(self, seed_states: dict[str, dict[str, Any]], experiment_params: dict[str, Any]):
        self.seed_states = seed_states
        self.experiment_params = experiment_params
        self.experiment_start_time: float = 0.0
        self.total_planned_epochs: int | None = None
        self.latest_metrics: dict[str, Any] = {}

    def create_info_panel(self) -> Panel:
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

        for _, label, value in experiment_info:
            info_table.add_row(f"{label}:", value)

        return Panel(info_table, title="Experiment Info", border_style="yellow")

    def create_kasima_panel(self) -> Panel:
        """Generate the panel containing the seeds training table."""
        if not self.seed_states:
            return Panel(
                Align.center("Waiting for seed data...", vertical="middle"),
                title="Kasima",
                border_style="green",
            )

        seeds_training_table = self._create_seeds_training_table()
        return Panel(seeds_training_table, title="Kasima", border_style="green")

    def create_tamiyo_panel(self) -> Panel:
        """Generate the panel for Tamiyo's status."""
        tamiyo_table = Table(show_header=False, expand=True, box=box.MINIMAL, padding=(0, 1))
        tamiyo_table.add_column("Metric", style=STYLE_BOLD_BLUE)
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

    def create_karn_panel(self) -> Panel:
        """Generate the panel for Karn's status."""
        karn_table = Table(show_header=False, expand=True, box=box.MINIMAL, padding=(0, 1))
        karn_table.add_column("Metric", style=STYLE_BOLD_BLUE)
        karn_table.add_column("Value")

        karn_table.add_row("Status", "Exploiting Keystone Lineage")
        karn_table.add_row("Archive Size", "21,249 Blueprints (81.2%)")
        karn_table.add_row("Archive Quality (Mean)", "74.5 ELO")
        karn_table.add_row("Critic Loss", "0.051")
        karn_table.add_row("Exploration/Exploitation", "Exploiting (80%)")
        karn_table.add_row()  # Whitespace

        return Panel(karn_table, title="Karn", border_style="blue")

    def create_crucible_panel(self) -> Panel:
        """Generate the panel for detailed seed training data."""
        crucible_table = Table(show_header=False, expand=True, box=box.MINIMAL, padding=(0, 1))
        crucible_table.add_column("Metric", style=STYLE_BOLD_BLUE, ratio=1)
        crucible_table.add_column("Value", ratio=1)

        # Static key-value pairs as requested, formatted like Tamiyo/Karn panels.
        crucible_table.add_row(Text("ID", style="blue"), Text("---", style="dim"))
        crucible_table.add_row(Text("State", style="blue"), Text("---", style="dim"))
        crucible_table.add_row(Text("α", style="blue"), Text("---", style="dim"))
        crucible_table.add_row(Text("∇ Norm", style="blue"), Text("---", style="dim"))
        crucible_table.add_row(Text("Pat.", style="blue"), Text("---", style="dim"))

        return Panel(crucible_table, title="Crucible", border_style="yellow")

    def _add_info_section(self, table: Table, params: list[tuple[str, str]]) -> None:
        """Add a section of parameters to the info table."""
        for _, (key, label) in enumerate(params):
            value = self.experiment_params.get(key)
            value_str = str(value) if value is not None else "N/A"
            table.add_row(f"{label}:", value_str)

    def _format_start_time(self) -> str:
        """Format the experiment start time for display."""
        start_dt = datetime.fromtimestamp(self.experiment_start_time)
        return start_dt.strftime("%H:%M:%S")

    def _sort_seed_id_key(self, seed_id: str | tuple) -> tuple[int, int]:
        """Convert a seed ID (string like 'L1_S2' or tuple (1, 2)) into a sortable tuple."""
        parsed = self._parse_seed_id(seed_id)
        if parsed[0] is not None and parsed[1] is not None:
            return (parsed[0], parsed[1])
        return (9999, 9999)  # Place unparseable IDs at the end

    def _parse_seed_id(self, seed_id: str | tuple) -> tuple[int | None, int | None]:
        """Parse seed ID to extract layer and seed indices (simplified version)."""
        if isinstance(seed_id, tuple) and len(seed_id) == 2:
            try:
                return (int(seed_id[0]), int(seed_id[1]))
            except (ValueError, TypeError):
                return (None, None)

        if isinstance(seed_id, str):
            seed_id = seed_id.upper()
            # Handle "L1_S2" format
            if "L" in seed_id and "S" in seed_id:
                try:
                    parts = seed_id.replace("L", "").replace("S", "_").split("_")
                    if len(parts) >= 2:
                        return (int(parts[0]), int(parts[1]))
                except (ValueError, IndexError):
                    pass

        return (None, None)

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
