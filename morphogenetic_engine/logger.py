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
    writes each event to both the console and a file on disk.
    """

    def __init__(self, log_file_path: str | Path, config: Dict[str, Any]) -> None:
        self.log_file_path = Path(log_file_path)
        self.config = config
        self.events: List[LogEvent] = []

        # Ensure log directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate existing file
        self.log_file_path.write_text("")

    # ------------------------------------------------------------------
    def _record_event(self, event: LogEvent) -> None:
        self.events.append(event)
        self.write_to_file(event)
        self.print_real_time_update(event)

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

    def log_seed_event(self, epoch: int, seed_id: str, description: str) -> None:
        """Record an event related to a sentinel seed."""

        event = LogEvent(
            timestamp=time.time(),
            epoch=epoch,
            event_type=EventType.SEED_STATE_CHANGE,
            message=description,
            data={"seed_id": seed_id},
        )
        self._record_event(event)

    def log_phase_transition(self, epoch: int, from_phase: str, to_phase: str) -> None:
        """Record a phase transition of the experiment."""

        event = LogEvent(
            timestamp=time.time(),
            epoch=epoch,
            event_type=EventType.PHASE_TRANSITION,
            message=f"Phase transition {from_phase} -> {to_phase}",
            data={
                "from": from_phase,
                "to": to_phase,
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

    # ------------------------------------------------------------------
    def write_to_file(self, event: LogEvent) -> None:
        """Append the event to the log file in JSON format."""

        with open(self.log_file_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict()) + "\n")

    def print_real_time_update(self, event: LogEvent) -> None:
        """Print a human readable version of the event."""

        ts = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {event.message} | {event.data}")

    # ------------------------------------------------------------------
    def generate_final_report(self) -> Dict[str, int]:
        """Generate a simple summary of event counts by type."""

        summary: Dict[str, int] = {}
        for event in self.events:
            event_name = event.event_type.value
            summary[event_name] = summary.get(event_name, 0) + 1
        return summary

