"""Experiment logging utilities for the morphogenetic engine."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List


class EventType(Enum):
    """Enumeration of supported log event types."""

    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_END = "experiment_end"
    EPOCH_PROGRESS = "epoch_progress"
    PHASE_TRANSITION = "phase_transition"
    SEED_STATE_CHANGE = "seed_state_change"
    GERMINATION = "germination"
    BLENDING_PROGRESS = "blending_progress"
    ACCURACY_DIP = "accuracy_dip"
    SCHEDULER_STEP = "scheduler_step"


@dataclass
class LogEvent:
    """Container for a single log event."""

    timestamp: float
    epoch: int
    event_type: EventType
    message: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON serialisable representation of the event."""

        return {
            "timestamp": self.timestamp,
            "epoch": self.epoch,
            "event_type": self.event_type.value,
            "message": self.message,
            "data": self.data,
        }


class ExperimentLogger:
    """Lightweight experiment logger.

    This logger keeps an in-memory list of :class:`LogEvent` instances and
    writes each event to both the console and a file on disk. It can also
    integrate with a dashboard for live updates.
    """

    def __init__(self, log_file_path: str | Path, config: Dict[str, Any], dashboard=None) -> None:
        """Initialize the logger and ensure the results directory exists."""
        # Always place logs inside the project ``results`` directory
        results_dir = Path(__file__).resolve().parents[1] / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = results_dir / Path(log_file_path).name
        self.config = config
        self.events: List[LogEvent] = []
        self.dashboard = dashboard  # Optional dashboard integration
        # Truncate any existing log file
        self.log_file_path.write_text("")

    # ------------------------------------------------------------------
    def _record_event(self, event: LogEvent) -> None:
        self.events.append(event)
        self.write_to_file(event)

        # If we have a dashboard, send the event there instead of printing
        if self.dashboard is not None:
            self._send_to_dashboard(event)
        else:
            self.print_real_time_update(event)

    def _send_to_dashboard(self, event: LogEvent) -> None:
        """Send event to dashboard for live display."""
        if not self.dashboard:
            return

        # Use a dispatcher to call the appropriate method on the dashboard
        # based on the event type, passing the specific data it needs.
        event_map = {
            EventType.EXPERIMENT_START: (
                self.dashboard.add_live_event,
                {"event_type": "INFO", "message": "Experiment Started", "data": {}},
            ),
            EventType.EPOCH_PROGRESS: (
                self.dashboard.update_progress,
                {"epoch": event.epoch, "metrics": event.data},
            ),
            EventType.PHASE_TRANSITION: (
                self.dashboard.show_phase_transition,
                {
                    "to_phase": event.data.get("to"),
                    "epoch": event.epoch,
                    "from_phase": event.data.get("from"),
                    "total_epochs": event.data.get("total_epochs"),
                },
            ),
            EventType.SEED_STATE_CHANGE: (
                self.dashboard.update_seed,
                {
                    "seed_id": event.data.get("seed_id"),
                    "state": event.data.get("to"),
                    "from_state": event.data.get("from"),
                },
            ),
            EventType.GERMINATION: (
                self.dashboard.show_germination_event,
                {"seed_id": event.data.get("seed_id"), "epoch": event.epoch},
            ),
            EventType.BLENDING_PROGRESS: (
                self.dashboard.update_seed,  # Can reuse update_seed for this
                {
                    "seed_id": event.data.get("seed_id"),
                    "state": "blending",
                    "alpha": event.data.get("alpha", 0.0),
                    "from_state": "germinated",
                },
            ),
            EventType.EXPERIMENT_END: (
                self.dashboard.add_live_event,
                {"event_type": "INFO", "message": "Experiment Finished", "data": {}},
            ),
        }

        handler, kwargs = event_map.get(event.event_type, (None, None))

        if handler and kwargs and isinstance(kwargs, dict):
            # Filter out None values from kwargs to avoid passing them
            handler(**{k: v for k, v in kwargs.items() if v is not None})

        # Also send the raw event for historical logging in the panel
        if self.dashboard:
            self.dashboard.add_live_event(event_type=event.event_type.value.upper(), message=event.message, data=event.data)

    # ------------------------------------------------------------------
    def log_experiment_start(self) -> None:
        """Record the start of an experiment."""

        event = LogEvent(
            timestamp=time.time(),
            epoch=0,
            event_type=EventType.EXPERIMENT_START,
            message="Experiment started",
            data={"config": self.config},
        )
        self._record_event(event)

    def log_epoch_progress(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Log metrics for a training epoch."""

        event = LogEvent(
            timestamp=time.time(),
            epoch=epoch,
            event_type=EventType.EPOCH_PROGRESS,
            message="Epoch progress",
            data={**metrics},
        )
        self._record_event(event)

    def log_seed_event(
        self,
        epoch: int,
        seed_id: str,
        from_state: str,
        to_state: str,
        description: str | None = None,
    ) -> None:
        """Record a seed state transition."""

        msg = description or f"Seed {seed_id}: {from_state} -> {to_state}"
        event = LogEvent(
            timestamp=time.time(),
            epoch=epoch,
            event_type=EventType.SEED_STATE_CHANGE,
            message=msg,
            data={"seed_id": seed_id, "from": from_state, "to": to_state},
        )
        self._record_event(event)

    def log_germination(self, epoch: int, seed_id: str) -> None:
        """Record successful germination of a seed."""

        event = LogEvent(
            timestamp=time.time(),
            epoch=epoch,
            event_type=EventType.GERMINATION,
            message=f"Seed {seed_id} germinated",
            data={"seed_id": seed_id},
        )
        self._record_event(event)

    def log_blending_progress(self, epoch: int, seed_id: str, alpha: float) -> None:
        """Log soft-landing blending progress for a seed."""

        event = LogEvent(
            timestamp=time.time(),
            epoch=epoch,
            event_type=EventType.BLENDING_PROGRESS,
            message=f"Blending {seed_id} alpha={alpha:.3f}",
            data={"seed_id": seed_id, "alpha": alpha},
        )
        self._record_event(event)

    def log_phase_transition(
        self, epoch: int, from_phase: str, to_phase: str, description: str = "", total_epochs: int | None = None
    ) -> None:
        """Record a phase transition of the experiment."""

        event = LogEvent(
            timestamp=time.time(),
            epoch=epoch,
            event_type=EventType.PHASE_TRANSITION,
            message=description or f"Phase transition {from_phase} -> {to_phase}",
            data={
                "from": from_phase,
                "to": to_phase,
                "total_epochs": total_epochs,
            },
        )
        self._record_event(event)

    def log_accuracy_dip(self, epoch: int, accuracy: float) -> None:
        """Record a significant dip in accuracy."""

        event = LogEvent(
            timestamp=time.time(),
            epoch=epoch,
            event_type=EventType.ACCURACY_DIP,
            message="Accuracy dip detected",
            data={"accuracy": accuracy},
        )
        self._record_event(event)

    def log_experiment_end(self, epoch: int) -> None:
        """Record the end of an experiment and report a summary."""

        summary = self.generate_final_report()
        # Include this experiment_end event in the summary
        summary["experiment_end"] = summary.get("experiment_end", 0) + 1

        event = LogEvent(
            timestamp=time.time(),
            epoch=epoch,
            event_type=EventType.EXPERIMENT_END,
            message="Experiment finished",
            data={"summary": summary},
        )
        self._record_event(event)

    # ------------------------------------------------------------------
    def write_to_file(self, event: LogEvent) -> None:
        """Append the event to the log file in JSON format."""

        with open(self.log_file_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

    def print_real_time_update(self, event: LogEvent) -> None:
        """Print a human readable version of the event."""

        ts = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        event_name = event.event_type.value.upper()
        message = f"[{ts}] {event_name}: {event.message} - {event.data}"

        # If dashboard is available, send there instead of printing
        if hasattr(self, "dashboard") and self.dashboard is not None:
            self.dashboard.add_log_message(message)
        else:
            print(message)

    # ------------------------------------------------------------------
    def generate_final_report(self) -> Dict[str, int]:
        """Generate a simple summary of event counts by type."""

        summary: Dict[str, int] = {}
        for event in self.events:
            event_name = event.event_type.value
            summary[event_name] = summary.get(event_name, 0) + 1
        return dict(sorted(summary.items()))
