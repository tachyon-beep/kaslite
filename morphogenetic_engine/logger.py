"""Experiment logging utilities for the morphogenetic engine."""

from __future__ import annotations

import collections
import time
from pathlib import Path

from .events import (
    EventType,
    LogEvent,
    MetricsUpdatePayload,
    PhaseUpdatePayload,
    SeedStateUpdatePayload,
    SystemInitPayload,
    SystemShutdownPayload,
)


class ExperimentLogger:
    """Lightweight experiment logger.

    This logger keeps an in-memory list of :class:`LogEvent` instances and
    writes each event to a file on disk. It is fully decoupled from any UI or
    dashboard implementation.
    """

    def __init__(self, log_dir: str | Path, log_file: str) -> None:
        """Initialize the logger and ensure the results directory exists."""
        self.log_dir = Path(log_dir)
        self.log_file_path = self.log_dir / log_file
        self.events: list[LogEvent] = []

        # Ensure the log directory exists and truncate the log file
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path.write_text("", encoding="utf-8")

    def _log_event(self, event_type: EventType, payload: dict) -> None:
        """Create, store, and write a LogEvent."""
        event = LogEvent(
            timestamp=time.time(),
            event_type=event_type,
            payload=payload,
        )
        self.events.append(event)
        try:
            with self.log_file_path.open("a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")
        except (OSError, PermissionError) as e:
            # Propagate I/O errors to the caller for specific handling
            raise e
        except TypeError as e:
            # Propagate serialization errors
            raise e

    def log_system_init(self, message: str, details: dict | None = None) -> None:
        """Log a system initialization event."""
        payload = SystemInitPayload(message=message, details=details)
        self._log_event(event_type=EventType.SYSTEM_INIT, payload=payload)

    def log_metrics_update(self, epoch: int, metrics: dict) -> None:
        """Log a metrics update event."""
        payload = MetricsUpdatePayload(epoch=epoch, metrics=metrics)
        self._log_event(event_type=EventType.METRICS_UPDATE, payload=payload)

    def log_phase_update(
        self, phase_name: str, epoch: int, details: dict | None = None
    ) -> None:
        """Log a phase change event."""
        payload = PhaseUpdatePayload(
            phase_name=phase_name, epoch=epoch, details=details
        )
        self._log_event(event_type=EventType.PHASE_UPDATE, payload=payload)

    def log_seed_state_update(self, epoch: int, seed_updates: list[dict]) -> None:
        """Log a seed state update event."""
        payload = SeedStateUpdatePayload(epoch=epoch, seed_updates=seed_updates)
        self._log_event(event_type=EventType.SEED_STATE_UPDATE, payload=payload)

    def log_system_shutdown(self, message: str, details: dict | None = None) -> None:
        """Log a system shutdown event."""
        payload = SystemShutdownPayload(message=message, details=details)
        self._log_event(event_type=EventType.SYSTEM_SHUTDOWN, payload=payload)

    def generate_final_report(self) -> dict[str, int]:
        """Generate a summary report of events from the in-memory list."""
        if not self.events:
            return {}
        # Use collections.Counter for a more direct and readable implementation
        return collections.Counter(event.event_type.value for event in self.events)
