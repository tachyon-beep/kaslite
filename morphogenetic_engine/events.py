"""
Defines the standardized, type-safe data contracts for all events within the morphogenetic engine.

This module serves as the single source of truth for the "API" between the backend logic
and any consumers of event data, such as the UI, loggers, or analysis tools.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, TypedDict, Union


# =============================================================================
# Event Type Enumeration
# =============================================================================


class EventType(Enum):
    """Enumeration of all verifiable event types."""

    SYSTEM_INIT = "system_init"
    SYSTEM_SHUTDOWN = "system_shutdown"
    METRICS_UPDATE = "metrics_update"
    PHASE_UPDATE = "phase_update"
    SEED_STATE_UPDATE = "seed_state_update"


# =============================================================================
# TypedDict Payloads for Each Event
# =============================================================================


class SystemInitPayload(TypedDict):
    """Payload for when the system starts."""

    message: str
    details: dict[str, Any] | None


class SystemShutdownPayload(TypedDict):
    """Payload for when the system stops."""

    message: str
    details: dict[str, Any] | None


class MetricsUpdatePayload(TypedDict):
    """Payload for sending all metrics after an epoch."""

    epoch: int
    metrics: dict[str, float | int]


class PhaseUpdatePayload(TypedDict):
    """Payload for announcing a change in the experiment phase."""

    phase_name: str
    epoch: int
    details: dict[str, Any] | None


class SeedStateUpdatePayload(TypedDict):
    """Payload for providing a complete update of all tracked seeds."""

    epoch: int
    seed_updates: list[dict[str, Any]]


# A union of all possible event payloads for type-safe handling
EventPayload = Union[
    SystemInitPayload,
    SystemShutdownPayload,
    MetricsUpdatePayload,
    PhaseUpdatePayload,
    SeedStateUpdatePayload,
]


# =============================================================================
# Core LogEvent Dataclass
# =============================================================================


@dataclass
class LogEvent:
    """A container for a single, fully type-safe log event."""

    timestamp: float
    event_type: EventType
    payload: EventPayload

    def to_json(self) -> str:
        """Serialize the event to a JSON string."""

        # Custom JSON encoder to handle Enum
        class EventEncoder(json.JSONEncoder):
            def default(self, o: Any) -> Any:
                if isinstance(o, EventType):
                    return o.value
                return super().default(o)

        return json.dumps(asdict(self), cls=EventEncoder)
