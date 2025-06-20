"""Experiment logging utilities for the morphogenetic engine."""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Any, Optional, cast

from .events import (
    EventPayload,
    EventType,
    GerminationPayload,
    LogEvent,
    LogPayload,
    MetricsUpdatePayload,
    NetworkStrainGridUpdatePayload,
    PhaseUpdatePayload,
    SeedInfo,
    SeedLogPayload,
    SeedState,
    SeedStateUpdatePayload,
    SystemInitPayload,
    SystemShutdownPayload,
)
from .ui import RichDashboard


class ExperimentLogger:
    """Lightweight experiment logger with optional dashboard dispatching.

    This logger keeps an in-memory list of :class:`LogEvent` instances and
    writes each event to a file on disk. If a dashboard is provided, it also
    dispatches events for real-time visualization.
    """

    def __init__(self, log_dir: str | Path, log_file: str) -> None:
        """Initialize the logger and ensure the results directory exists."""
        self.log_dir = Path(log_dir)
        self.log_file_path = self.log_dir / log_file
        self.events: list[LogEvent] = []
        self.dashboard: Optional[RichDashboard] = None

        # Ensure the log directory exists and truncate the log file
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path.write_text("", encoding="utf-8")

        # Map EventType to the corresponding RichDashboard method
        self.dashboard_dispatch_map = {
            EventType.SYSTEM_INIT: lambda p: (
                self.dashboard.initialize_experiment(cast(SystemInitPayload, p)["config"]) if self.dashboard else None
            ),
            EventType.METRICS_UPDATE: lambda p: (
                self.dashboard.update_metrics(cast(MetricsUpdatePayload, p)) if self.dashboard else None
            ),
            EventType.PHASE_UPDATE: lambda p: (
                self.dashboard.transition_phase(cast(PhaseUpdatePayload, p)) if self.dashboard else None
            ),
            EventType.SEED_STATE_UPDATE: lambda p: (
                self.dashboard.update_seed_states_grid(cast(SeedStateUpdatePayload, p)) if self.dashboard else None
            ),
            EventType.NETWORK_STRAIN_UPDATE: lambda p: (
                self.dashboard.update_network_strain_grid(cast(NetworkStrainGridUpdatePayload, p)) if self.dashboard else None
            ),
            EventType.GERMINATION: lambda p: (
                self.dashboard.log_seed_event(
                    {
                        "event_type": "germination",
                        "message": f"Seed {cast(GerminationPayload, p)['seed_id']} germinated!",
                        "data": {"epoch": cast(GerminationPayload, p)["epoch"]},
                    }
                )
                if self.dashboard
                else None
            ),
            # General logging events can be mapped to the dashboard's log methods
            EventType.LOG_EVENT: lambda p: self.dashboard.log_event(cast(LogPayload, p)) if self.dashboard else None,
            EventType.SEED_LOG_EVENT: lambda p: (
                self.dashboard.log_seed_event(cast(SeedLogPayload, p)) if self.dashboard else None
            ),
        }

    def _log_event(self, event_type: EventType, payload: EventPayload) -> None:
        """Create, store, write, and dispatch a LogEvent."""
        event = LogEvent(event_type=event_type, payload=payload)
        self.events.append(event)

        # Dispatch to dashboard if present and a handler exists
        if self.dashboard and event.event_type in self.dashboard_dispatch_map:
            handler = self.dashboard_dispatch_map[event.event_type]
            handler(payload)

        try:
            with self.log_file_path.open("a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")
        except OSError as e:
            # Add context to I/O errors before propagating
            raise OSError(f"Failed to write to log file: {e}") from e
        except TypeError as e:
            # Add context to serialization errors before propagating
            raise TypeError(f"Failed to serialize log event: {e}") from e

    def log_system_init(self, config: dict) -> None:
        """Log a system initialization event."""
        payload = SystemInitPayload(config=config, timestamp=time.time())
        self._log_event(event_type=EventType.SYSTEM_INIT, payload=payload)

    def log_metrics_update(self, epoch: int, metrics: dict) -> None:
        """Log a metrics update event."""
        payload = MetricsUpdatePayload(epoch=epoch, metrics=metrics, timestamp=time.time())
        self._log_event(event_type=EventType.METRICS_UPDATE, payload=payload)

    def log_phase_update(
        self,
        epoch: int,
        from_phase: str,
        to_phase: str,
        total_epochs_in_phase: int | None = None,
    ) -> None:
        """Log a phase change event."""
        payload = PhaseUpdatePayload(
            epoch=epoch,
            from_phase=from_phase,
            to_phase=to_phase,
            total_epochs_in_phase=total_epochs_in_phase,
            timestamp=time.time(),
        )
        self._log_event(event_type=EventType.PHASE_UPDATE, payload=payload)

    def log_seed_state_update(self, epoch: int, seeds: list[SeedInfo]) -> None:
        """Log a seed state update event."""
        # Convert seed data to grid format for payload
        grid: dict[int, list[SeedState | None]] = {}

        if not seeds:
            # Empty payload for empty seeds list
            payload = SeedStateUpdatePayload(epoch=epoch, grid=grid, timestamp=time.time())
            self._log_event(event_type=EventType.SEED_STATE_UPDATE, payload=payload)
            return

        # Determine grid dimensions
        max_layer = max(seed["layer"] for seed in seeds)
        max_seeds_per_layer = 0
        for layer in range(max_layer + 1):
            layer_seeds = [seed["index_in_layer"] for seed in seeds if seed["layer"] == layer]
            if layer_seeds:
                max_seeds_per_layer = max(max_seeds_per_layer, *layer_seeds)

        # Initialize grid
        for layer in range(max_layer + 1):
            grid[layer] = [None] * (max_seeds_per_layer + 1)

        # Populate grid with seed states
        for seed in seeds:
            layer = seed["layer"]
            index = seed["index_in_layer"]
            grid[layer][index] = seed["state"]

        payload = SeedStateUpdatePayload(epoch=epoch, grid=grid, timestamp=time.time())
        self._log_event(event_type=EventType.SEED_STATE_UPDATE, payload=payload)

    def log_system_shutdown(self, final_stats: dict) -> None:
        """Log a system shutdown event."""
        payload = SystemShutdownPayload(final_stats=final_stats, timestamp=time.time())
        self._log_event(event_type=EventType.SYSTEM_SHUTDOWN, payload=payload)

    def log_germination(self, epoch: int, seed_id: tuple[int, int]) -> None:
        """Log a seed germination event with dedicated event type."""
        payload = GerminationPayload(epoch=epoch, seed_id=seed_id, timestamp=time.time())
        self._log_event(event_type=EventType.GERMINATION, payload=payload)

    def log_seed_event_detailed(self, epoch: int, event_type: str, message: str, data: dict[str, Any] | None = None) -> None:
        """Log a detailed seed event for the timeline."""
        # Include epoch in the data payload
        event_data = data or {}
        event_data["epoch"] = epoch

        payload = SeedLogPayload(
            event_type=event_type,
            message=message,
            data=event_data,
        )
        self._log_event(event_type=EventType.SEED_LOG_EVENT, payload=payload)

    def log_seed_event(self, epoch: int, seed_id: tuple[int, int], old_state: str, new_state: str) -> None:
        """Log a seed state transition event (convenience method)."""
        self.log_seed_event_detailed(
            epoch=epoch,
            event_type="STATE_TRANSITION",
            message=f"Seed L{seed_id[0]}_S{seed_id[1]}: {old_state} → {new_state}",
            data={
                "seed_id": f"L{seed_id[0]}_S{seed_id[1]}",
                "from_state": old_state,
                "to_state": new_state,
                "epoch": epoch,
            }
        )

    def generate_final_report(self) -> dict[str, int]:
        """Generate a summary report of events from the in-memory list."""
        if not self.events:
            return {}
        # Use collections.Counter for a more direct and readable implementation
        return collections.Counter(event.event_type.value for event in self.events)
