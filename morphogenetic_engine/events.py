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
    NETWORK_STRAIN_UPDATE = "network_strain_update"
    GERMINATION = "germination"
    LOG_EVENT = "log_event"
    SEED_LOG_EVENT = "seed_log_event"


class SeedState(Enum):
    """Enumeration of possible states for a morphogenetic seed."""

    ACTIVE = "active"
    DORMANT = "dormant"
    BLENDING = "blending"
    GERMINATED = "germinated"
    FOSSILIZED = "fossilized"


class NetworkStrain(Enum):
    """Enumeration of network strain levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    FIRED = "fired"


# =============================================================================
# TypedDict Payloads for Each Event
# =============================================================================


class SystemInitPayload(TypedDict):
    """Payload for when the system starts."""

    config: dict[str, Any]
    timestamp: float


class SystemShutdownPayload(TypedDict):
    """Payload for when the system stops."""

    final_stats: dict[str, Any]
    timestamp: float


class MetricsUpdatePayload(TypedDict):
    """Payload for sending all metrics after an epoch."""

    epoch: int
    metrics: dict[str, float | int]
    timestamp: float


class PhaseUpdatePayload(TypedDict):
    """Payload for announcing a change in the experiment phase."""

    epoch: int
    from_phase: str
    to_phase: str
    total_epochs_in_phase: int | None
    timestamp: float


class SeedInfo(TypedDict):
    """A snapshot of a single seed's state and metrics."""

    id: tuple[int, int]  # (layer_index, seed_index_in_layer)
    state: SeedState
    layer: int
    index_in_layer: int
    metrics: dict[str, float | int | None]  # e.g., train_loss, val_acc, alpha


class SeedStateUpdatePayload(TypedDict):
    """Payload for providing a complete update of all tracked seeds."""

    epoch: int
    grid: dict[int, list[SeedState | None]]
    timestamp: float


class NetworkStrainGridUpdatePayload(TypedDict):
    """Payload for providing a complete update of network strain."""

    epoch: int
    grid: dict[int, list[NetworkStrain | None]]


class GerminationPayload(TypedDict):
    """Payload for seed germination events."""

    epoch: int
    seed_id: tuple[int, int]
    timestamp: float


class LogPayload(TypedDict):
    """Payload for a generic log event."""

    event_type: str
    message: str
    data: dict[str, Any] | None


class SeedLogPayload(TypedDict):
    """Payload for a seed-specific log event."""

    event_type: str
    message: str
    data: dict[str, Any] | None


class SeedMetricsUpdatePayload(TypedDict, total=False):
    """Payload for updating a single seed's metrics. All fields are optional except seed_id."""

    seed_id: str | tuple[int, int]
    state: SeedState | str | None
    alpha: float | None
    grad_norm: float | None
    patience: int | None


# A union of all possible event payloads for type-safe handling
EventPayload = Union[
    SystemInitPayload,
    SystemShutdownPayload,
    MetricsUpdatePayload,
    PhaseUpdatePayload,
    SeedStateUpdatePayload,
    GerminationPayload,
    LogPayload,
    SeedLogPayload,
    SeedMetricsUpdatePayload,
]


# =============================================================================
# Core LogEvent Dataclass
# =============================================================================


@dataclass
class LogEvent:
    """A container for a single, fully type-safe log event."""

    event_type: EventType
    payload: EventPayload

    def to_json(self) -> str:
        """Serialize the event to a JSON string."""

        # Custom JSON encoder to handle Enum
        class EventEncoder(json.JSONEncoder):
            def default(self, o: Any) -> Any:
                if isinstance(o, (EventType, SeedState, NetworkStrain)):
                    return o.value
                if isinstance(o, float):
                    # Optional: Round floats for cleaner logs
                    return round(o, 6)
                return super().default(o)

        return json.dumps(asdict(self), cls=EventEncoder)
