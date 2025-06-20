"""Experiment logging utilities for the morphogenetic engine."""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Optional, cast

from .cli_dashboard import RichDashboard
from .events import (
    EventPayload,
    EventType,
    GerminationPayload,
    LogEvent,
    MetricsUpdatePayload,
    PhaseUpdatePayload,
    SeedInfo,
    SeedState,
    SeedStateUpdatePayload,
    SystemInitPayload,
    SystemShutdownPayload,
)


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

    def _log_event(self, event_type: EventType, payload: EventPayload) -> None:
        """Create, store, write, and dispatch a LogEvent."""
        event = LogEvent(event_type=event_type, payload=payload)
        self.events.append(event)

        # Dispatch to dashboard if present
        if self.dashboard:
            if event.event_type == EventType.METRICS_UPDATE:
                # The dashboard's update_progress method handles metrics
                metrics_payload = cast(MetricsUpdatePayload, payload)
                self.dashboard.update_progress(
                    metrics_payload["epoch"], metrics_payload["metrics"]
                )
            elif event.event_type == EventType.PHASE_UPDATE:
                phase_payload = cast(PhaseUpdatePayload, payload)
                self.dashboard.show_phase_transition(
                    to_phase=phase_payload["to_phase"],
                    epoch=phase_payload["epoch"],
                    from_phase=phase_payload["from_phase"],
                    total_epochs=phase_payload.get("total_epochs_in_phase"),
                )
            elif event.event_type == EventType.SEED_STATE_UPDATE:
                # Use the proper dashboard method for seed updates
                seed_payload = cast(SeedStateUpdatePayload, payload)
                self.dashboard.update_seeds_view(seed_payload)
            elif event.event_type == EventType.GERMINATION:
                # Log germination events
                germination_payload = cast(GerminationPayload, payload)
                seed_id = germination_payload["seed_id"]
                self.dashboard.add_live_event(
                    "GERMINATION",
                    f"Seed L{seed_id[0]}_S{seed_id[1]} germinated at epoch {germination_payload['epoch']}",
                    {},
                )

        try:
            with self.log_file_path.open("a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")
        except OSError:
            # Propagate I/O errors to the caller for specific handling
            raise
        except TypeError as e:
            # Add context to serialization errors before propagating
            raise TypeError(f"Failed to serialize log event: {e}") from e

    def log_system_init(self, config: dict) -> None:
        """Log a system initialization event."""
        payload = SystemInitPayload(
            config=config, timestamp=time.time()
        )
        self._log_event(event_type=EventType.SYSTEM_INIT, payload=payload)

    def log_metrics_update(self, epoch: int, metrics: dict) -> None:
        """Log a metrics update event."""
        payload = MetricsUpdatePayload(
            epoch=epoch, metrics=metrics, timestamp=time.time()
        )
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
            payload = SeedStateUpdatePayload(
                epoch=epoch, grid=grid, timestamp=time.time()
            )
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
        
        payload = SeedStateUpdatePayload(
            epoch=epoch, grid=grid, timestamp=time.time()
        )
        self._log_event(event_type=EventType.SEED_STATE_UPDATE, payload=payload)

    def log_system_shutdown(self, final_stats: dict) -> None:
        """Log a system shutdown event."""
        payload = SystemShutdownPayload(
            final_stats=final_stats, timestamp=time.time()
        )
        self._log_event(event_type=EventType.SYSTEM_SHUTDOWN, payload=payload)

    def log_germination(self, epoch: int, seed_id: tuple[int, int]) -> None:
        """Log a seed germination event with dedicated event type."""
        payload = GerminationPayload(
            epoch=epoch, 
            seed_id=seed_id,
            timestamp=time.time()
        )
        self._log_event(event_type=EventType.GERMINATION, payload=payload)

    def log_seed_event(self, epoch: int, _seed_id: tuple[int, int], old_state: str, new_state: str) -> None:
        """Log a seed state transition event (convenience method)."""
        # For now, just log the phase transition
        self.log_phase_update(
            epoch=epoch,
            from_phase=old_state,
            to_phase=new_state,
        )

    def generate_final_report(self) -> dict[str, int]:
        """Generate a summary report of events from the in-memory list."""
        if not self.events:
            return {}
        # Use collections.Counter for a more direct and readable implementation
        return collections.Counter(event.event_type.value for event in self.events)
