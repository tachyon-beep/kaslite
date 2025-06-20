"""Tests for the standardized event data contracts."""

import json
from typing import Any

import pytest

from morphogenetic_engine.events import (
    EventType,
    LogEvent,
    MetricsUpdatePayload,
    PhaseUpdatePayload,
    SeedInfo,
    SeedStateUpdatePayload,
    SystemInitPayload,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fixed_timestamp() -> float:
    """Provide a fixed timestamp for deterministic testing."""
    return 1640995200.0


@pytest.fixture
def sample_metrics() -> dict[str, float | int]:
    """Provide a sample metrics dictionary."""
    return {"accuracy": 0.95, "loss": 0.12, "epoch_duration_sec": 30}


@pytest.fixture
def sample_seed_info() -> list[SeedInfo]:
    """Provide a list of sample SeedInfo TypedDicts."""
    return [
        {
            "id": "L0_S1",
            "state": "active",
            "layer": 0,
            "index_in_layer": 1,
            "metrics": {"train_loss": 0.2, "val_acc": 0.91},
        },
        {
            "id": "L2_S5",
            "state": "blending",
            "layer": 2,
            "index_in_layer": 5,
            "metrics": {"alpha": 0.5, "val_acc": 0.88},
        },
    ]


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.unit
class TestEventType:
    """Test suite for the EventType enum."""

    def test_all_event_types_have_unique_string_values(self):
        """Ensures all EventType members have unique, non-empty string values."""
        values = {event_type.value for event_type in EventType}
        assert len(values) == len(EventType)
        for value in values:
            assert isinstance(value, str)
            assert len(value) > 0

    @pytest.mark.parametrize("event_type", list(EventType))
    def test_event_type_string_conversion(self, event_type: EventType):
        """Verifies that each EventType converts to a lowercase string correctly."""
        assert isinstance(event_type.value, str)
        assert event_type.value.islower()
        assert "_" in event_type.value


@pytest.mark.unit
class TestLogEventAndPayloads:
    """Test suite for the LogEvent dataclass and its TypedDict payloads."""

    def test_system_init_event_creation(self, fixed_timestamp: float):
        """Tests creation of a SYSTEM_INIT event."""
        config = {"lr": 0.01, "model": "test_net"}
        payload = SystemInitPayload(config=config, timestamp=fixed_timestamp)
        event = LogEvent(
            timestamp=fixed_timestamp,
            epoch=0,
            event_type=EventType.SYSTEM_INIT,
            message="Init",
            data=payload,
        )

        assert event.data["config"]["lr"] == 0.01
        assert event.event_type == EventType.SYSTEM_INIT

    def test_metrics_update_event_creation(self, fixed_timestamp: float, sample_metrics: dict[str, Any]):
        """Tests creation of a METRICS_UPDATE event."""
        payload = MetricsUpdatePayload(epoch=10, metrics=sample_metrics, timestamp=fixed_timestamp)
        event = LogEvent(
            timestamp=fixed_timestamp,
            epoch=10,
            event_type=EventType.METRICS_UPDATE,
            message="Metrics",
            data=payload,
        )

        assert event.data["epoch"] == 10
        assert event.data["metrics"]["accuracy"] == 0.95
        assert event.event_type == EventType.METRICS_UPDATE

    def test_phase_update_event_creation(self, fixed_timestamp: float):
        """Tests creation of a PHASE_UPDATE event."""
        payload = PhaseUpdatePayload(
            epoch=50,
            from_phase="seeding",
            to_phase="activation",
            total_epochs_in_phase=100,
            timestamp=fixed_timestamp,
        )
        event = LogEvent(
            timestamp=fixed_timestamp,
            epoch=50,
            event_type=EventType.PHASE_UPDATE,
            message="Phase change",
            data=payload,
        )

        assert event.data["to_phase"] == "activation"
        assert event.data["total_epochs_in_phase"] == 100
        assert event.event_type == EventType.PHASE_UPDATE

    def test_seed_state_update_event_creation(self, fixed_timestamp: float, sample_seed_info: list[SeedInfo]):
        """Tests creation of a SEED_STATE_UPDATE event."""
        payload = SeedStateUpdatePayload(epoch=25, seeds=sample_seed_info, timestamp=fixed_timestamp)
        event = LogEvent(
            timestamp=fixed_timestamp,
            epoch=25,
            event_type=EventType.SEED_STATE_UPDATE,
            message="Seeds updated",
            data=payload,
        )

        assert len(event.data["seeds"]) == 2
        assert event.data["seeds"][0]["state"] == "active"
        assert event.event_type == EventType.SEED_STATE_UPDATE

    def test_to_dict_serialization(self, fixed_timestamp: float, sample_metrics: dict[str, Any]):
        """Tests that LogEvent.to_dict() produces a correct, JSON-serializable format."""
        payload = MetricsUpdatePayload(epoch=5, metrics=sample_metrics, timestamp=fixed_timestamp)
        event = LogEvent(
            timestamp=fixed_timestamp,
            epoch=5,
            event_type=EventType.METRICS_UPDATE,
            message="Test message",
            data=payload,
        )

        result = event.to_dict()
        expected_data = {
            "timestamp": fixed_timestamp,
            "epoch": 5,
            "event_type": "metrics_update",
            "message": "Test message",
            "data": payload,
        }

        assert result == expected_data
        # Ensure it's JSON serializable
        json_str = json.dumps(result)
        assert '"accuracy": 0.95' in json_str
