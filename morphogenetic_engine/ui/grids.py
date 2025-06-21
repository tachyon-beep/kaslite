"""
Grid visualization components for the UI dashboard.

This module handles the grid-based visualizations for seed states
and network strain, including the creation of emoji-based grids
and their associated legends.
"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from typing import Any, cast

from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import wcwidth

from morphogenetic_engine.events import NetworkStrain, SeedState

from .config import EMPTY_CELL_EMOJI, SEED_EMOJI_MAP, STRAIN_EMOJI_MAP


class GridManager:
    """Manages grid-based visualizations for seeds and network strain."""

    def __init__(self, seed_states: dict[str, dict[str, Any]]):
        self.seed_states = seed_states

    def create_seed_box_panel(self, experiment_params: dict[str, Any]) -> Panel:
        """Generate the main panel for visualizing seed states."""
        num_layers = experiment_params.get("num_layers", 8)
        seeds_per_layer = experiment_params.get("seeds_per_layer", 1)

        layer_data = self._get_seed_states_by_layer(num_layers, seeds_per_layer)

        grid_table = self._create_grid_table(cast(Any, layer_data), cast(Any, SEED_EMOJI_MAP), num_layers, seeds_per_layer)
        return Panel(
            Align.center(grid_table, vertical="middle"),
            title="Seed States",
            border_style="green",
        )

    def create_network_strain_panel(self, experiment_params: dict[str, Any]) -> Panel:
        """Generate the panel for visualizing network strain."""
        num_layers = experiment_params.get("num_layers", 8)
        seeds_per_layer = experiment_params.get("seeds_per_layer", 1)

        layer_data = self._get_network_strain_states_by_layer(num_layers, seeds_per_layer)

        grid_table = self._create_grid_table(cast(Any, layer_data), cast(Any, STRAIN_EMOJI_MAP), num_layers, seeds_per_layer)
        return Panel(
            Align.center(grid_table, vertical="middle"),
            title="Network Strain",
            border_style="red",
        )

    def pad_cell(self,text: str, width: int) -> str:
        """Right-pad `text` so it occupies exactly `width` terminal cells."""
        w = wcwidth.wcswidth(text)
        if w < 0:
            w = len(text)
        return text + " " * max(0, width - w)


    def create_seed_legend_panel(self) -> Panel:
        # 1) Define your two rows of (emoji, label) pairs
        row1 = [
            ("⚪", "Dormant"),
            ("🌱", "Germinated"),
            ("🟢", "Training"),
            ("🟡", "Blending"),
            ("👻", "Shadowing"),
        ]
        row2 = [
            ("🧑‍⚖️", "Probationary"),
            ("🦴", "Fossilized"),
            ("🥀", "Culled"),
            (EMPTY_CELL_EMOJI, "Empty slot"),
            ("", ""),  # blank filler
        ]

        # 2) Compute a uniform cell width for the widest entry
        all_texts = [f"{e} {l}" for e, l in row1 + row2]
        max_w = max(
            (wcwidth.wcswidth(t) if wcwidth.wcswidth(t) > 0 else len(t))
            for t in all_texts
        )
        CELL_WIDTH = max_w + 2  # add a little breathing room

        # 3) Build the two padded lines
        line1 = "".join(self.pad_cell(f"{e} {l}", CELL_WIDTH) for e, l in row1)
        line2 = "".join(self.pad_cell(f"{e} {l}", CELL_WIDTH) for e, l in row2)

        # 4) Render as a single Text block (monospaced!)
        text = Text(f"{line1}\n{line2}", style="dim")

        # 5) Wrap in a non‐expanding Panel and centre it
        panel = Panel(
            text,
            box=box.MINIMAL,
            padding=0,
            border_style="green",
            expand=False,
        )
        return Align.center(panel)

    def create_strain_legend_panel(self) -> Panel:
        """Generate the legend for the network strain."""
        legend_parts = [
            f"{STRAIN_EMOJI_MAP[NetworkStrain.NONE]} None",
            f"{STRAIN_EMOJI_MAP[NetworkStrain.LOW]} Low",
            f"{STRAIN_EMOJI_MAP[NetworkStrain.MEDIUM]} Medium",
            f"{STRAIN_EMOJI_MAP[NetworkStrain.HIGH]} High",
            f"{STRAIN_EMOJI_MAP[NetworkStrain.FIRED]} Fired",
        ]
        legend_text = Text("  ".join(legend_parts))
        return Panel(
            Align.center(legend_text),
            box=box.MINIMAL,
            style="dim",
            border_style="red",
            padding=(0, 1),
        )

    def _get_seed_states_by_layer(self, num_layers: int, seeds_per_layer: int) -> dict[int, list[SeedState | None]]:
        """Organize seed states by layer for grid display."""
        layer_states: dict[int, list[SeedState | None]] = {i: [None] * seeds_per_layer for i in range(num_layers)}

        for seed_id, seed_data in self.seed_states.items():
            parsed = self._parse_seed_id(seed_id)
            layer, seed_idx = parsed

            if layer is not None and seed_idx is not None and 0 <= layer < num_layers and 0 <= seed_idx < seeds_per_layer:
                state = seed_data.get("state")
                if isinstance(state, SeedState):
                    layer_states[layer][seed_idx] = state

        return layer_states

    def _get_network_strain_states_by_layer(self, num_layers: int, seeds_per_layer: int) -> dict[int, list[NetworkStrain | None]]:
        """Generate placeholder network strain data by layer (currently random)."""
        layer_strains: dict[int, list[NetworkStrain | None]] = {i: [None] * seeds_per_layer for i in range(num_layers)}
        strain_choices = list(NetworkStrain)
        for i in range(num_layers):
            for j in range(seeds_per_layer):
                # Create some dynamic-looking random data
                if (i * seeds_per_layer + j + int(time.time() * 2)) % 7 < 2:
                    layer_strains[i][j] = random.choice(strain_choices)
                else:
                    layer_strains[i][j] = NetworkStrain.NONE
        return layer_strains

    def _parse_seed_id(self, seed_id: str | tuple) -> tuple[int | None, int | None]:
        """Parse seed ID to extract layer and seed indices."""
        if isinstance(seed_id, tuple):
            return self._parse_tuple_seed_id(seed_id)

        # seed_id must be str at this point due to type annotation
        return self._parse_string_seed_id(seed_id)

    def _parse_tuple_seed_id(self, seed_id: tuple) -> tuple[int | None, int | None]:
        """Parse tuple format seed ID."""
        if len(seed_id) >= 2:
            try:
                return (int(seed_id[0]), int(seed_id[1]))
            except (ValueError, TypeError):
                pass
        return (None, None)

    def _parse_string_seed_id(self, seed_id: str) -> tuple[int | None, int | None]:
        """Parse string format seed ID."""
        seed_id = seed_id.upper()

        # Try "L1_S2" format
        if "L" in seed_id and "S" in seed_id:
            result = self._parse_ls_format(seed_id)
            if result != (None, None):
                return result

        # Try "layer_1_seed_2" format
        if "LAYER" in seed_id and "SEED" in seed_id:
            result = self._parse_layer_seed_format(seed_id)
            if result != (None, None):
                return result

        # Try "1_2" format
        return self._parse_simple_format(seed_id)

    def _parse_ls_format(self, seed_id: str) -> tuple[int | None, int | None]:
        """Parse L1_S2 format."""
        try:
            parts = seed_id.replace("L", "").replace("S", "_").split("_")
            if len(parts) >= 2:
                return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            pass
        return (None, None)

    def _parse_layer_seed_format(self, seed_id: str) -> tuple[int | None, int | None]:
        """Parse layer_1_seed_2 format."""
        try:
            parts = seed_id.split("_")
            layer_idx = None
            seed_idx = None
            for i, part in enumerate(parts):
                if part == "LAYER" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                elif part == "SEED" and i + 1 < len(parts):
                    seed_idx = int(parts[i + 1])
            return (layer_idx, seed_idx)
        except (ValueError, IndexError):
            pass
        return (None, None)

    def _parse_simple_format(self, seed_id: str) -> tuple[int | None, int | None]:
        """Parse 1_2 format."""
        parts = seed_id.split("_")
        if len(parts) == 2:
            try:
                return (int(parts[0]), int(parts[1]))
            except ValueError:
                pass
        return (None, None)

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
        row = [f"{layer_index}"]  # Use 0-based indexing to match logs
        if layer_index < num_layers:
            states = layer_data.get(layer_index, [])
            for j in range(seeds_per_layer):
                emoji = empty_emoji
                if j < len(states):
                    state = states[j]
                    if state:
                        emoji = emoji_map.get(state, empty_emoji)
                row.append(emoji)
        else:
            row.extend([empty_emoji] * seeds_per_layer)
        return row

    def _create_grid_table(
        self,
        grid: Mapping[int, list[SeedState | NetworkStrain | None]],
        emoji_map: Mapping[SeedState | NetworkStrain, str],
        num_layers: int,
        seeds_per_layer: int,
    ) -> Table:
        """Creates a Rich Table for a grid."""
        table = Table(title="", show_header=True, header_style="bold magenta", box=box.ROUNDED, expand=False)
        table.add_column("L", justify="center")
        for i in range(seeds_per_layer):
            table.add_column(f"{i}", justify="center", width=3)

        for i in range(num_layers):
            row = self._create_grid_row(i, num_layers, seeds_per_layer, grid, emoji_map, EMPTY_CELL_EMOJI)
            table.add_row(*row)

        return table

    def update_seed_grid_from_payload(
        self, grid_data: dict[int, list[SeedState | None]], experiment_params: dict[str, Any]
    ) -> Panel:
        """Update seed grid visualization from external grid data."""
        num_layers = experiment_params.get("num_layers", 8)
        seeds_per_layer = experiment_params.get("seeds_per_layer", 1)

        grid_table = self._create_grid_table(cast(Any, grid_data), cast(Any, SEED_EMOJI_MAP), num_layers, seeds_per_layer)
        return Panel(
            Align.center(grid_table, vertical="middle"),
            title="Seed States",
            border_style="green",
        )

    def update_strain_grid_from_payload(
        self, grid_data: dict[int, list[NetworkStrain | None]], experiment_params: dict[str, Any]
    ) -> Panel:
        """Update network strain grid visualization from external grid data."""
        num_layers = experiment_params.get("num_layers", 8)
        seeds_per_layer = experiment_params.get("seeds_per_layer", 1)

        grid_table = self._create_grid_table(cast(Any, grid_data), cast(Any, STRAIN_EMOJI_MAP), num_layers, seeds_per_layer)
        return Panel(
            Align.center(grid_table, vertical="middle"),
            title="Network Strain",
            border_style="red",
        )
