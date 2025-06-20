"""
Main dashboard class for the UI module.

This module contains the RichDashboard class that coordinates all UI components
and provides the main interface for the morphogenetic experiment monitoring.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.live import Live

from morphogenetic_engine.events import (
    LogPayload,
    MetricsUpdatePayload,
    NetworkStrainGridUpdatePayload,
    PhaseUpdatePayload,
    SeedLogPayload,
    SeedMetricsUpdatePayload,
    SeedState,
    SeedStateUpdatePayload,
)

from .config import GRID_SIZE
from .events import EventManager
from .grids import GridManager
from .layout import LayoutManager
from .metrics import MetricsManager
from .panels import PanelFactory


class RichDashboard:
    """A Rich CLI dashboard for experiment monitoring with progress bars."""

    GRID_SIZE = GRID_SIZE

    def __init__(self, console: Console | None = None, experiment_params: dict[str, Any] | None = None):
        self.console = console or Console()
        self.experiment_params = experiment_params or {}
        self.live: Live | None = None

        # Component managers
        self.event_manager = EventManager()
        self.metrics_manager = MetricsManager()
        self.layout_manager = LayoutManager()

        # Seed state tracking
        self.seed_states: dict[str, dict[str, Any]] = {}

        # Initialize dependent managers
        self.grid_manager = GridManager(self.seed_states)
        self.panel_factory = PanelFactory(self.seed_states, self.experiment_params)

        # Timing and epoch tracking
        self.phase_start_epoch = 0
        self.current_phase_epochs = 0

        # Calculate total epochs from warm_up_epochs + adaptation_epochs
        warm_up_epochs = self.experiment_params.get("warm_up_epochs", 50)
        adaptation_epochs = self.experiment_params.get("adaptation_epochs", 50)
        self.total_epochs = warm_up_epochs + adaptation_epochs

        # Initialize metrics manager timing
        self.metrics_manager.experiment_start_time = time.time()
        self.metrics_manager.total_planned_epochs = self.total_epochs

        # Initialize panel factory timing
        self.panel_factory.experiment_start_time = self.metrics_manager.experiment_start_time
        self.panel_factory.total_planned_epochs = self.total_epochs

    def _setup_layout(self):
        """Initialize the dashboard layout and populate initial panels."""
        self.layout_manager.setup_layout(self.total_epochs)

        # Populate initial panels
        self._populate_initial_panels()

    def _populate_initial_panels(self) -> None:
        """Populate all panels with initial content."""
        self.layout_manager.update_panel("info_panel", self.panel_factory.create_info_panel())
        self.layout_manager.update_panel(
            "metrics_table_panel", self.metrics_manager.create_metrics_table_panel(self.experiment_params)
        )
        self.layout_manager.update_panel("sparkline_panel", self.metrics_manager.create_sparkline_panel())
        self.layout_manager.update_panel("event_log_panel", self.event_manager.create_event_log_panel())
        self.layout_manager.update_panel("kasima_panel", self.panel_factory.create_kasima_panel())
        self.layout_manager.update_panel("tamiyo_panel", self.panel_factory.create_tamiyo_panel())
        self.layout_manager.update_panel("karn_panel", self.panel_factory.create_karn_panel())
        self.layout_manager.update_panel("crucible_panel", self.panel_factory.create_crucible_panel())
        self.layout_manager.update_panel("seed_timeline_panel", self.event_manager.create_seed_timeline_panel())
        self.layout_manager.update_panel("seed_box_panel", self.grid_manager.create_seed_box_panel(self.experiment_params))
        self.layout_manager.update_panel("seed_legend_panel", self.grid_manager.create_seed_legend_panel())
        self.layout_manager.update_panel(
            "network_strain_panel", self.grid_manager.create_network_strain_panel(self.experiment_params)
        )
        self.layout_manager.update_panel("strain_legend_panel", self.grid_manager.create_strain_legend_panel())

    def update_metrics(self, payload: MetricsUpdatePayload) -> None:
        """Receives a metrics update and refreshes the view."""
        epoch = payload["epoch"]
        metrics = payload["metrics"]
        timestamp = payload["timestamp"]

        if epoch < 0:
            self.event_manager.log_event({"event_type": "error", "message": "Invalid epoch value", "data": {"epoch": epoch}})
            return

        # Track timing information
        current_time = time.time()
        if self.metrics_manager.epoch_start_time is not None:
            epoch_duration = current_time - self.metrics_manager.epoch_start_time
            self.metrics_manager.epoch_times.append(epoch_duration)
            # Keep only recent epoch times for averaging
            if len(self.metrics_manager.epoch_times) > 50:
                self.metrics_manager.epoch_times = self.metrics_manager.epoch_times[-50:]
        self.metrics_manager.epoch_start_time = current_time

        # Update progress bars
        current_phase_epoch = epoch - self.phase_start_epoch
        self.layout_manager.update_progress(epoch + 1, current_phase_epoch + 1)

        # Update metrics
        self.metrics_manager.update_metrics(metrics, epoch, timestamp)

        # Sync latest metrics to panel factory
        self.panel_factory.latest_metrics = self.metrics_manager.latest_metrics

        # Log the metrics update
        self.event_manager.log_metrics_update(epoch, metrics)

        # Update the metrics panels with the latest data
        self.layout_manager.update_panel(
            "metrics_table_panel", self.metrics_manager.create_metrics_table_panel(self.experiment_params)
        )
        self.layout_manager.update_panel("sparkline_panel", self.metrics_manager.create_sparkline_panel())

    def update_seed_metrics(self, payload: SeedMetricsUpdatePayload) -> None:
        """Update the state and metrics of a single seed from a payload."""
        seed_id = payload["seed_id"]

        # Normalize seed_id to string representation
        if isinstance(seed_id, tuple):
            seed_id = f"L{seed_id[0]}_S{seed_id[1]}"

        if not isinstance(seed_id, str) or not seed_id:
            self.event_manager.log_event({"event_type": "error", "message": "Invalid seed_id", "data": {"seed_id": seed_id}})
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
                    self.event_manager.log_event(
                        {"event_type": "error", "message": f"Invalid seed state value: '{state}'", "data": {"seed_id": seed_id}}
                    )
                    return
            self.seed_states[seed_id]["state"] = state

        # Update other metrics from the payload
        for key, value in payload.items():
            if key not in ["seed_id", "state"] and value is not None:
                self.seed_states[seed_id][key] = value

        # Refresh the Kasima panel to show the updated data
        self.layout_manager.update_panel("kasima_panel", self.panel_factory.create_kasima_panel())

    def transition_phase(self, payload: PhaseUpdatePayload) -> None:
        """Handle phase transition event and reset phase progress bar."""
        self.event_manager.log_phase_transition(payload["from_phase"], payload["to_phase"], payload["epoch"])

        if payload.get("total_epochs_in_phase") is not None:
            total_epochs_in_phase = payload["total_epochs_in_phase"]
            if total_epochs_in_phase is not None:
                self.phase_start_epoch = payload["epoch"]
                self.current_phase_epochs = total_epochs_in_phase
                phase_description = payload["to_phase"].replace("_", " ").title()
                self.layout_manager.reset_phase_progress(total_epochs_in_phase, phase_description)

    def log_event(self, payload: LogPayload) -> None:
        """Add a new event to be displayed in the main experiment log."""
        self.event_manager.log_event(payload)
        self.layout_manager.update_panel("event_log_panel", self.event_manager.create_event_log_panel())

    def log_seed_event(self, payload: SeedLogPayload) -> None:
        """Add a new event to the dedicated seed log."""
        self.event_manager.log_seed_event(payload)
        self.layout_manager.update_panel("seed_timeline_panel", self.event_manager.create_seed_timeline_panel())

    def start(self):
        """Start the live dashboard display, with error handling."""
        if self.live:
            return
        try:
            self._setup_layout()
            layout = self.layout_manager.get_layout()
            if layout:
                self.live = Live(
                    layout,
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
        panel = self.grid_manager.update_strain_grid_from_payload(payload["grid"], self.experiment_params)
        self.layout_manager.update_panel("network_strain_panel", panel)

    def update_seed_states_grid(self, payload: SeedStateUpdatePayload) -> None:
        """Receives a seed state update and refreshes the view."""
        panel = self.grid_manager.update_seed_grid_from_payload(payload["grid"], self.experiment_params)
        self.layout_manager.update_panel("seed_box_panel", panel)

    def initialize_experiment(self, config: dict[str, Any]) -> None:
        """Initialize experiment with configuration data."""
        self.experiment_params.update(config)

        # Calculate total planned epochs from config
        warm_up = config.get("warm_up_epochs", 0)
        adaptation = config.get("adaptation_epochs", 0)
        total_planned_epochs = warm_up + adaptation

        # Update managers with new config
        self.metrics_manager.total_planned_epochs = total_planned_epochs
        self.panel_factory.total_planned_epochs = total_planned_epochs
        self.panel_factory.experiment_params = self.experiment_params

        # Update total progress bar with planned epochs
        if total_planned_epochs > 0 and self.layout_manager.total_task is not None:
            self.layout_manager.total_progress.update(self.layout_manager.total_task, total=total_planned_epochs)

    # Configuration methods for metrics
    def configure_sparklines(self, **kwargs) -> None:
        """Allow runtime configuration of sparkline behavior."""
        self.metrics_manager.configure_sparklines(**kwargs)

    def add_metric_type(self, metric_key: str, inverted: bool = False, color: str = "white") -> None:
        """Register a new metric type for sparkline rendering."""
        self.metrics_manager.add_metric_type(metric_key, inverted, color)
