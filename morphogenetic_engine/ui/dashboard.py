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
    NetworkStrain,
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

    # EMOJI MAPS
    SEED_EMOJI_MAP = {
        SeedState.TRAINING: "🟢",
        SeedState.DORMANT: "⚪",
        SeedState.BLENDING: "🟡",
        SeedState.GERMINATED: "🌱",
        SeedState.FOSSILIZED: "🦴",
        SeedState.CULLED: "🥀",
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
        self.layout_manager.update_panel("metrics_table_panel", self.metrics_manager.create_metrics_table_panel())
        self.layout_manager.update_panel("sparkline_panel", self.metrics_manager.create_sparkline_panel())
        self.layout_manager.update_panel("event_log_panel", self.event_manager.create_event_log_panel())
        self.layout_manager.update_panel("kasima_panel", self.panel_factory.create_kasima_panel())
        self.layout_manager.update_panel("tamiyo_panel", self.panel_factory.create_tamiyo_panel())
        self.layout_manager.update_panel("karn_panel", self.panel_factory.create_karn_panel())
        self.layout_manager.update_panel("crucible_panel", self.panel_factory.create_crucible_panel())
        
        # Add a test seed event to verify the seed timeline panel works
        self.event_manager.log_seed_event({
            "event_type": "initialization", 
            "message": "Seed timeline initialized", 
            "data": {"status": "ready"}
        })
        
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
        self.layout_manager.update_panel("metrics_table_panel", self.metrics_manager.create_metrics_table_panel())
        self.layout_manager.update_panel("sparkline_panel", self.metrics_manager.create_sparkline_panel())
        
        # Update the info panel to show current epoch
        self.layout_manager.update_panel("info_panel", self.panel_factory.create_info_panel())
        
        # Update the event log panel to show the new metrics event
        self.layout_manager.update_panel("event_log_panel", self.event_manager.create_event_log_panel())
        
        # Collect and update real seed metrics for the Kasima table
        self._collect_seed_metrics_from_manager()
        self.layout_manager.update_panel("kasima_panel", self.panel_factory.create_kasima_panel())

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

    def update_network_strain_grid(self, payload: NetworkStrainGridUpdatePayload) -> None:
        """Receives a grid update and refreshes the view."""
        panel = self.grid_manager.update_strain_grid_from_payload(payload["grid"], self.experiment_params)
        self.layout_manager.update_panel("network_strain_panel", panel)

    def update_seed_states_grid(self, payload: SeedStateUpdatePayload) -> None:
        """Receives a seed state update and refreshes the view."""
        # Update the internal seed_states dictionary for the Kasima table
        self._update_seed_states_from_grid(payload["grid"])
        
        # Collect real metrics from seed manager if available
        self._collect_seed_metrics_from_manager()
        
        # Update the grid panel
        panel = self.grid_manager.update_seed_grid_from_payload(payload["grid"], self.experiment_params)
        self.layout_manager.update_panel("seed_box_panel", panel)
        
        # Update the Kasima table panel
        kasima_panel = self.panel_factory.create_kasima_panel()
        self.layout_manager.update_panel("kasima_panel", kasima_panel)

    def _update_seed_states_from_grid(self, grid_data: dict[int, list[SeedState | None]]) -> None:
        """Update the internal seed_states dictionary from grid data."""
        for layer_idx, states in grid_data.items():
            for seed_idx, state in enumerate(states):
                if state is not None:
                    self._update_seed_entry(layer_idx, seed_idx, state)

    def _update_seed_entry(self, layer_idx: int, seed_idx: int, state: SeedState) -> None:
        """Update a single seed entry in the seed_states dictionary."""
        # Create seed ID string matching log format (L3_S0, etc.)
        seed_id = f"L{layer_idx}_S{seed_idx}"
        
        # Update or create seed state entry
        if seed_id not in self.seed_states:
            self.seed_states[seed_id] = {}
        
        self.seed_states[seed_id]["state"] = state
        # Initialize other fields if not present
        self._initialize_seed_defaults(seed_id)

    def _initialize_seed_defaults(self, seed_id: str) -> None:
        """Initialize default values for a seed entry."""
        defaults = {
            "alpha": 0.0,
            "grad_norm": None,
            "patience": None,
        }
        for key, value in defaults.items():
            if key not in self.seed_states[seed_id]:
                self.seed_states[seed_id][key] = value

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
        
        # Update the event log panel to show the phase transition
        self.layout_manager.update_panel("event_log_panel", self.event_manager.create_event_log_panel())

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

    # Configuration methods for metrics
    def configure_sparklines(self, **kwargs) -> None:
        """Allow runtime configuration of sparkline behavior."""
        self.metrics_manager.configure_sparklines(**kwargs)

    def add_metric_type(self, metric_key: str, inverted: bool = False, color: str = "white") -> None:
        """Register a new metric type for sparkline rendering."""
        self.metrics_manager.add_metric_type(metric_key, inverted, color)

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

    def _collect_seed_metrics_from_manager(self) -> None:
        """Collect real metrics from the seed manager if available."""
        try:
            from morphogenetic_engine.core import SeedManager
            seed_manager = SeedManager()
            
            if hasattr(seed_manager, 'seeds') and seed_manager.seeds:
                with seed_manager.lock:
                    for seed_id, seed_info in seed_manager.seeds.items():
                        self._update_seed_metrics_from_manager(seed_id, seed_info)
                                
        except Exception as e:
            # Log the specific error instead of swallowing it silently
            import logging
            logging.warning(f"Failed to collect seed metrics from manager: {e}")

    def _update_seed_metrics_from_manager(self, seed_id: tuple[int, int], seed_info: dict) -> None:
        """Update metrics for a single seed from the seed manager."""
        seed_id_str = f"L{seed_id[0]}_S{seed_id[1]}"
        
        # Ensure seed exists in dashboard state
        if seed_id_str not in self.seed_states:
            self.seed_states[seed_id_str] = {}
        
        self._update_alpha_metric(seed_id_str, seed_info)
        self._update_gradient_norm_metric(seed_id_str, seed_info)
        self._update_patience_metric(seed_id_str, seed_info)
        self._update_state_from_status(seed_id_str, seed_info)

    def _update_alpha_metric(self, seed_id_str: str, seed_info: dict) -> None:
        """Update alpha metric for a seed."""
        # Always prefer the value from the manager dict
        if "alpha" in seed_info and seed_info["alpha"] is not None:
            self.seed_states[seed_id_str]["alpha"] = seed_info["alpha"]
        elif "module" in seed_info and hasattr(seed_info["module"], "alpha"):
            self.seed_states[seed_id_str]["alpha"] = seed_info["module"].alpha

    def _update_gradient_norm_metric(self, seed_id_str: str, seed_info: dict) -> None:
        """Update gradient norm metric for a seed."""
        if "module" in seed_info and hasattr(seed_info["module"], "get_gradient_norm"):
            try:
                grad_norm = seed_info["module"].get_gradient_norm()
                self.seed_states[seed_id_str]["grad_norm"] = grad_norm
            except (AttributeError, TypeError) as e:
                import logging
                logging.warning(f"Failed to get gradient norm for {seed_id_str}: {e}")

    def _update_patience_metric(self, seed_id_str: str, seed_info: dict) -> None:
        """Update patience metric for a seed."""
        # Always prefer the value from the manager dict
        if "training_steps" in seed_info and seed_info["training_steps"] is not None:
            self.seed_states[seed_id_str]["patience"] = seed_info["training_steps"]
        elif "module" in seed_info and hasattr(seed_info["module"], "training_progress"):
            # Fallback to old calculation if training_steps not available
            training_progress = seed_info["module"].training_progress
            patience = max(0, int((1.0 - training_progress) * 100))
            self.seed_states[seed_id_str]["patience"] = patience

    def _update_state_from_status(self, seed_id_str: str, seed_info: dict) -> None:
        """Update seed state from status in seed info."""
        if "status" in seed_info:
            status = seed_info["status"]
            if status == "active":
                self.seed_states[seed_id_str]["state"] = SeedState.TRAINING
            elif status == "dormant":
                self.seed_states[seed_id_str]["state"] = SeedState.DORMANT
            elif status == "failed":
                self.seed_states[seed_id_str]["state"] = SeedState.CULLED
