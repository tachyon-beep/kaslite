"""
Layout management for the UI dashboard.

This module handles the layout setup and management for the Rich dashboard,
including the organization of panels and the overall dashboard structure.
"""

from __future__ import annotations

from rich import box
from rich.align import Align
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.text import Text

from .config import COLUMN_RATIOS, LAYOUT_SIZES


class LayoutManager:
    """Manages the layout structure and organization of the dashboard."""

    def __init__(self):
        self.layout: Layout | None = None
        self._layout_initialized = False

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

        # Progress tasks
        self.total_task = None
        self.phase_task = None

    def setup_layout(self, total_epochs: int = 100) -> Layout:
        """Initialize the dashboard layout."""
        if self._layout_initialized and self.layout:
            return self.layout

        self.layout = Layout(name="root")

        # Initialize progress bars
        self.total_task = self.total_progress.add_task("Overall", total=total_epochs)
        self.phase_task = self.phase_progress.add_task("Phase", total=100)

        # --- Title Header ---
        title_header = Layout(
            Panel(Align.center("[bold]QUICKSILVER[/bold]"), box=box.MINIMAL),
            name="title",
            size=LAYOUT_SIZES["title_header"],
        )

        # --- Main Area Setup ---
        main_area = Layout(name="main")

        # --- Footer Setup for Progress Bars ---
        footer = Layout(name="footer", size=LAYOUT_SIZES["footer"])
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
            size=LAYOUT_SIZES["quit_footer"],
        )

        # Root layout splits into title, main content, and the footer
        self.layout.split(title_header, main_area, footer, quit_footer)

        # Setup main area columns and panels
        self._setup_main_area()

        self._layout_initialized = True
        return self.layout

    def _setup_main_area(self) -> None:
        """Setup the main area layout with columns and panels."""
        if not self.layout:
            return

        # 3 Columns in Main
        self.layout["main"].split_row(
            Layout(name="left_column", ratio=COLUMN_RATIOS["left_column"]),
            Layout(name="center_column", ratio=COLUMN_RATIOS["center_column"]),
            Layout(name="right_column", ratio=COLUMN_RATIOS["right_column"]),
        )

        self._setup_left_column()
        self._setup_center_column()
        self._setup_right_column()

    def _setup_left_column(self) -> None:
        """Setup the left column layout."""
        if not self.layout:
            return

        bottom_left = Layout(name="bottom_left_area")
        bottom_left.split_column(
            Layout(name="event_log_panel", ratio=1),
            Layout(name="seed_timeline_panel", ratio=1),
        )
        self.layout["left_column"].split_column(
            Layout(name="top_left_area", size=LAYOUT_SIZES["top_left_area"]),
            bottom_left,
        )
        self.layout["top_left_area"].split_column(
            Layout(name="top_row", size=LAYOUT_SIZES["top_row"]),
            Layout(name="sparkline_panel", size=LAYOUT_SIZES["sparkline_panel"]),
        )
        self.layout["top_row"].split_row(
            Layout(name="info_panel", ratio=COLUMN_RATIOS["info_panel"]),
            Layout(name="metrics_table_panel", ratio=COLUMN_RATIOS["metrics_table_panel"]),
        )

    def _setup_center_column(self) -> None:
        """Setup the center column layout."""
        if not self.layout:
            return

        self.layout["center_column"].split_column(
            Layout(name="kasima_panel"),
            Layout(name="tamiyo_panel", size=LAYOUT_SIZES["tamiyo_panel"]),
            Layout(name="karn_panel", size=LAYOUT_SIZES["karn_panel"]),
            Layout(name="crucible_panel", size=LAYOUT_SIZES["crucible_panel"]),
        )

    def _setup_right_column(self) -> None:
        """Setup the right column layout."""
        if not self.layout:
            return

        self.layout["right_column"].split_column(
            Layout(name="seed_box_area"),
            Layout(name="network_strain_area"),
        )
        self.layout["seed_box_area"].split_column(
            Layout(name="seed_box_panel"),
            Layout(name="seed_legend_panel", size=LAYOUT_SIZES["legend_panel"]),
        )
        self.layout["network_strain_area"].split_column(
            Layout(name="network_strain_panel"),
            Layout(name="strain_legend_panel", size=LAYOUT_SIZES["legend_panel"]),
        )

    def update_panel(self, panel_name: str, panel: Panel) -> None:
        """Update a specific panel in the layout."""
        if self.layout:
            try:
                self.layout[panel_name].update(panel)
            except KeyError:
                # Panel name not found in layout - silently ignore for now
                pass

    def reset_phase_progress(self, total_epochs: int, description: str = "Phase") -> None:
        """Reset the phase progress bar."""
        if self.phase_task is not None:
            self.phase_progress.reset(self.phase_task)
            self.phase_progress.update(
                self.phase_task,
                total=total_epochs,
                completed=0,
                description=description,
            )

    def update_progress(self, total_completed: int, phase_completed: int) -> None:
        """Update both progress bars."""
        if self.total_task is not None:
            self.total_progress.update(self.total_task, completed=total_completed)
        if self.phase_task is not None:
            self.phase_progress.update(self.phase_task, completed=phase_completed)

    def get_layout(self) -> Layout | None:
        """Get the current layout."""
        return self.layout
