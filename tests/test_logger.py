"""Tests for the experiment logging system."""

# pylint: disable=redefined-outer-name,unused-argument,protected-access

import json
import collections
from unittest.mock import patch

import pytest

from morphogenetic_engine.events import (
    EventType,
    LogEvent,
    MetricsUpdatePayload,
    PhaseUpdatePayload,
    SeedStateUpdatePayload,
    SystemInitPayload,
    SystemShutdownPayload,
)
from morphogenetic_engine.logger import ExperimentLogger

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fixed_timestamp() -> float:
    """Provide a fixed timestamp for deterministic testing."""
    return 1640995200.0


@pytest.fixture
def sample_config() -> dict:
    """Provide sample configuration data for testing."""
    return {"dataset": "test_dataset", "epochs": 100, "lr": 0.001}


@pytest.fixture
def logger_fixture(tmp_path):
    """Fixture to create an ExperimentLogger instance for integration tests."""
    log_dir = tmp_path / "results"
    log_file = "integration_test.log"
    logger = ExperimentLogger(log_dir=log_dir, log_file=log_file)
    yield logger, log_dir / log_file
    # Teardown: clean up the created log file and directory if they exist
    log_path = log_dir / log_file
    if log_path.exists():
        log_path.unlink()
    if log_dir.exists() and not any(log_dir.iterdir()):
        log_dir.rmdir()


# =============================================================================
# Helper Functions
# =============================================================================


def assert_log_event_structure(
    event: LogEvent,
    expected_type: EventType,
    expected_payload_type: type,
):
    """Helper function to assert LogEvent structure."""
    assert isinstance(event, LogEvent)
    assert event.event_type == expected_type
    assert isinstance(event.payload, expected_payload_type)
    assert isinstance(event.timestamp, float)


def get_events_from_file(file_path) -> list[dict]:
    """Read all log events from a log file."""
    if not file_path.exists():
        return []
    content = file_path.read_text(encoding="utf-8")
    if not content:
        return []
    return [json.loads(line) for line in content.strip().split("\n")]


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.unit
class TestExperimentLoggerUnit:
    """Unit tests for the ExperimentLogger, focusing on in-memory event handling."""

    @pytest.fixture
    def mock_logger(self, mocker):
        """Fixture to create a mocked ExperimentLogger that doesn't write to disk."""
        mocker.patch("pathlib.Path.mkdir")
        mocker.patch("pathlib.Path.write_text")
        mocker.patch("pathlib.Path.open")
        logger = ExperimentLogger(log_dir="mock_dir", log_file="mock_file.log")
        return logger

    def test_log_system_init(self, mock_logger):
        """Verify that log_system_init creates and appends the correct event."""
        message = "System starting"
        details = {"pid": 123}
        mock_logger.log_system_init(message=message, details=details)

        assert len(mock_logger.events) == 1
        event = mock_logger.events[0]
        assert event.event_type == EventType.SYSTEM_INIT
        assert event.payload == {"message": message, "details": details}
        assert event.timestamp is not None

    def test_log_metrics_update(self, mock_logger):
        """Verify that log_metrics_update creates and appends the correct event."""
        epoch = 1
        metrics = {"accuracy": 0.95, "loss": 0.1}
        mock_logger.log_metrics_update(epoch=epoch, metrics=metrics)

        assert len(mock_logger.events) == 1
        event = mock_logger.events[0]
        assert event.event_type == EventType.METRICS_UPDATE
        assert event.payload == {"epoch": epoch, "metrics": metrics}

    def test_log_phase_update(self, mock_logger):
        """Verify that log_phase_update creates and appends the correct event."""
        phase_name = "TRAINING"
        epoch = 10
        details = {"reason": "Reached convergence"}
        mock_logger.log_phase_update(
            phase_name=phase_name, epoch=epoch, details=details
        )

        assert len(mock_logger.events) == 1
        event = mock_logger.events[0]
        assert event.event_type == EventType.PHASE_UPDATE
        assert event.payload == {
            "phase_name": phase_name,
            "epoch": epoch,
            "details": details,
        }

    def test_log_seed_state_update(self, mock_logger):
        """Verify that log_seed_state_update creates and appends the correct event."""
        epoch = 5
        seed_updates = [
            {"id": "seed_a", "state": "active"},
            {"id": "seed_b", "state": "inactive", "reason": "pruned"},
        ]
        mock_logger.log_seed_state_update(epoch=epoch, seed_updates=seed_updates)

        assert len(mock_logger.events) == 1
        event = mock_logger.events[0]
        assert event.event_type == EventType.SEED_STATE_UPDATE
        assert event.payload == {"epoch": epoch, "seed_updates": seed_updates}

    def test_log_system_shutdown(self, mock_logger):
        """Verify that log_system_shutdown creates and appends the correct event."""
        message = "System shutting down gracefully"
        details = {"exit_code": 0}
        mock_logger.log_system_shutdown(message=message, details=details)

        assert len(mock_logger.events) == 1
        event = mock_logger.events[0]
        assert event.event_type == EventType.SYSTEM_SHUTDOWN
        assert event.payload == {"message": message, "details": details}

    def test_generate_final_report_empty(self, mock_logger):
        """Test report generation with no events."""
        report = mock_logger.generate_final_report()
        assert report == {}

    def test_generate_final_report(self, mock_logger):
        """Test that the final report correctly summarizes events."""
        # Manually add events to the logger's list
        mock_logger.events.extend(
            [
                LogEvent(
                    timestamp=0,
                    event_type=EventType.SYSTEM_INIT,
                    payload=SystemInitPayload(message="", details=None),
                ),
                LogEvent(
                    timestamp=1,
                    event_type=EventType.METRICS_UPDATE,
                    payload=MetricsUpdatePayload(epoch=1, metrics={}),
                ),
                LogEvent(
                    timestamp=2,
                    event_type=EventType.METRICS_UPDATE,
                    payload=MetricsUpdatePayload(epoch=2, metrics={}),
                ),
                LogEvent(
                    timestamp=3,
                    event_type=EventType.PHASE_UPDATE,
                    payload=PhaseUpdatePayload(
                        phase_name="a", epoch=1, details=None
                    ),
                ),
            ]
        )

        report = mock_logger.generate_final_report()
        expected_report = {
            "system_init": 1,
            "metrics_update": 2,
            "phase_update": 1,
        }
        assert report == expected_report

    def test_multiple_events_maintain_order(self, mock_logger):
        """Ensure events are stored in the order they are logged."""
        mock_logger.log_system_init(message="1")
        mock_logger.log_metrics_update(epoch=1, metrics={})
        mock_logger.log_system_shutdown(message="3")

        assert len(mock_logger.events) == 3
        assert mock_logger.events[0].event_type == EventType.SYSTEM_INIT
        assert mock_logger.events[1].event_type == EventType.METRICS_UPDATE
        assert mock_logger.events[2].event_type == EventType.SYSTEM_SHUTDOWN

    def test_log_raises_permission_error(self, mocker):
        """Test that a PermissionError during file I/O is raised."""
        mocker.patch("pathlib.Path.mkdir")
        mocker.patch("pathlib.Path.write_text")  # For init
        mocker.patch("pathlib.Path.open", side_effect=PermissionError("Permission denied"))

        logger = ExperimentLogger(log_dir="any", log_file="any.log")
        with pytest.raises(PermissionError, match="Permission denied"):
            logger.log_system_init(message="test")

    def test_log_raises_os_error(self, mocker):
        """Test that an OSError during file I/O is raised."""
        mocker.patch("pathlib.Path.mkdir")
        mocker.patch("pathlib.Path.write_text")  # For init
        mocker.patch("pathlib.Path.open", side_effect=OSError("OS error"))

        logger = ExperimentLogger(log_dir="any", log_file="any.log")
        with pytest.raises(OSError, match="OS error"):
            logger.log_metrics_update(epoch=1, metrics={})

    def test_json_serialization_error(self, mock_logger, mocker):
        """Verify that a TypeError from JSON serialization is handled."""
        unserializable_details = {"unserializable": object()}
        # We need to patch the json.dumps call inside the to_json method
        mocker.patch("morphogenetic_engine.events.json.dumps", side_effect=TypeError("Not serializable"))

        with pytest.raises(TypeError, match="Not serializable"):
            mock_logger.log_system_init(
                message="test", details=unserializable_details
            )


@pytest.mark.integration
class TestExperimentLoggerIntegration:
    """Integration tests for the ExperimentLogger, verifying file I/O."""

    def test_directory_and_file_creation(self, logger_fixture):
        """Test that the logger creates the directory and log file on init."""
        _, log_path = logger_fixture
        assert log_path.parent.exists()
        assert log_path.parent.is_dir()
        assert log_path.exists()  # File is created and truncated on init
        assert log_path.read_text() == ""

    def test_file_writing_and_content(self, logger_fixture):
        """Test that events are correctly written to the log file as JSON lines."""
        logger, log_path = logger_fixture
        logger.log_system_init(message="System initialized")
        logger.log_metrics_update(epoch=1, metrics={"acc": 0.9})

        events = get_events_from_file(log_path)

        assert len(events) == 2
        event1 = events[0]
        event2 = events[1]

        assert event1["event_type"] == "system_init"
        assert event1["payload"]["message"] == "System initialized"
        assert "timestamp" in event1

        assert event2["event_type"] == "metrics_update"
        assert event2["payload"]["epoch"] == 1

    def test_complete_experiment_lifecycle(self, logger_fixture):
        """Simulate a full experiment, verifying the final log file."""
        logger, log_path = logger_fixture

        # 1. System Init
        logger.log_system_init(
            message="Starting full experiment",
            details={"config": "test_config"},
        )

        # 2. Metrics Updates
        logger.log_metrics_update(epoch=1, metrics={"loss": 0.5, "acc": 0.8})
        logger.log_metrics_update(epoch=2, metrics={"loss": 0.4, "acc": 0.85})

        # 3. Seed State Update
        seed_updates = [
            {"id": "s1", "state": "active", "details": {"fitness": 0.9}},
            {
                "id": "s2",
                "state": "pruned",
                "details": {"reason": "low_fitness"},
            },
        ]
        logger.log_seed_state_update(epoch=2, seed_updates=seed_updates)

        # 4. Phase Update
        logger.log_phase_update(
            phase_name="ANNEALING", epoch=3, details={"temp": 0.5}
        )

        # 5. System Shutdown
        logger.log_system_shutdown(
            message="Experiment finished", details={"exit_code": 0}
        )

        # Verification
        events = get_events_from_file(log_path)
        assert len(events) == 6

        # Check event types are in order
        expected_types = [
            "system_init",
            "metrics_update",
            "metrics_update",
            "seed_state_update",
            "phase_update",
            "system_shutdown",
        ]
        actual_types = [e["event_type"] for e in events]
        assert actual_types == expected_types

        # Spot-check a few payloads
        assert events[0]["payload"]["message"] == "Starting full experiment"
        assert events[2]["payload"]["metrics"]["acc"] == pytest.approx(0.85)
        assert events[3]["payload"]["seed_updates"][1]["id"] == "s2"
        assert events[5]["payload"]["details"]["exit_code"] == 0
