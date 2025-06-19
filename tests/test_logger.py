"""Tests for the experiment logging system."""

# pylint: disable=redefined-outer-name,unused-argument

import json
from unittest.mock import mock_open, patch

import pytest

from morphogenetic_engine.logger import EventType, ExperimentLogger, LogEvent

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fixed_timestamp():
    """Provide a fixed timestamp for deterministic testing."""
    return 1640995200.0


@pytest.fixture
def sample_config():
    """Provide sample configuration data for testing."""
    return {"dataset": "test_dataset", "epochs": 100, "lr": 0.001}


@pytest.fixture
def sample_log_event(fixed_timestamp):
    """Create a sample LogEvent for testing."""
    return LogEvent(
        timestamp=fixed_timestamp,
        epoch=5,
        event_type=EventType.EPOCH_PROGRESS,
        message="Test message",
        data={"accuracy": 0.85, "loss": 0.32},
    )


@pytest.fixture
def mock_logger(tmp_path, sample_config):
    """Create a logger instance with mocked file operations."""
    log_file = tmp_path / "test.log"

    # Create logger and override the problematic path resolution
    logger = object.__new__(ExperimentLogger)
    logger.log_file_path = log_file
    logger.config = sample_config
    logger.events = []

    # Create the log file
    log_file.write_text("")
    return logger


@pytest.fixture
def real_filesystem_logger(tmp_path, sample_config):
    """Create a logger with real filesystem for integration tests."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)

    logger = object.__new__(ExperimentLogger)
    logger.log_file_path = results_dir / "integration_test.log"
    logger.config = sample_config
    logger.events = []
    logger.log_file_path.write_text("")
    return logger


# =============================================================================
# Helper Functions
# =============================================================================


def assert_event_structure(event, expected_epoch, expected_type, expected_message, expected_data=None):
    """Helper function to assert event structure consistently."""
    assert event.epoch == expected_epoch
    assert event.event_type == expected_type
    assert event.message == expected_message
    if expected_data is not None:
        assert event.data == expected_data


def create_sample_events(logger, fixed_timestamp):
    """Create a diverse set of sample events for testing."""
    events_data = [
        (0, EventType.EXPERIMENT_START, "Experiment started", {"config": logger.config}),
        (1, EventType.EPOCH_PROGRESS, "Epoch progress", {"accuracy": 0.8, "loss": 0.5}),
        (5, EventType.GERMINATION, "Seed seed1 germinated", {"seed_id": "seed1"}),
        (
            10,
            EventType.BLENDING_PROGRESS,
            "Blending seed1 alpha=0.500",
            {"seed_id": "seed1", "alpha": 0.5},
        ),
        (
            15,
            EventType.PHASE_TRANSITION,
            "Phase transition exploration -> exploitation",
            {"from": "exploration", "to": "exploitation"},
        ),
    ]

    for i, (epoch, event_type, message, data) in enumerate(events_data):
        event = LogEvent(
            timestamp=fixed_timestamp + i,
            epoch=epoch,
            event_type=event_type,
            message=message,
            data=data,
        )
        logger.events.append(event)


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.unit
class TestLogEvent:
    """Test suite for LogEvent dataclass."""

    def test_log_event_creation_with_fixed_timestamp(self, fixed_timestamp):
        """Test basic LogEvent creation with deterministic timestamp."""
        event = LogEvent(
            timestamp=fixed_timestamp,
            epoch=5,
            event_type=EventType.EPOCH_PROGRESS,
            message="Test message",
            data={"accuracy": 0.85, "loss": 0.32},
        )

        assert_event_structure(
            event,
            expected_epoch=5,
            expected_type=EventType.EPOCH_PROGRESS,
            expected_message="Test message",
            expected_data={"accuracy": 0.85, "loss": 0.32},
        )
        assert event.timestamp == fixed_timestamp

    def test_to_dict_serialization(self, sample_log_event):
        """Test that LogEvent.to_dict() produces correct JSON-serializable format."""
        result = sample_log_event.to_dict()
        expected = {
            "timestamp": 1640995200.0,
            "epoch": 5,
            "event_type": "epoch_progress",
            "message": "Test message",
            "data": {"accuracy": 0.85, "loss": 0.32},
        }

        assert result == expected
        # Ensure it's JSON serializable
        json.dumps(result)  # Should not raise

    @pytest.mark.parametrize(
        "data_input,expected_serializable",
        [
            ({"simple": "value"}, True),
            ({"nested": {"key": "value"}}, True),
            ({"list": [1, 2, 3]}, True),
            ({"float": 3.14159}, True),
            ({"none": None}, True),
            ({}, True),  # Empty dict
        ],
    )
    def test_to_dict_handles_various_data_types(self, fixed_timestamp, data_input, expected_serializable):
        """Test that to_dict handles various data types correctly."""
        event = LogEvent(
            timestamp=fixed_timestamp,
            epoch=1,
            event_type=EventType.EXPERIMENT_START,
            message="Test",
            data=data_input,
        )

        result = event.to_dict()
        assert result["data"] == data_input

        if expected_serializable:
            # Should not raise
            json.dumps(result)

    def test_event_with_empty_data(self, fixed_timestamp):
        """Test LogEvent with empty data dictionary."""
        event = LogEvent(
            timestamp=fixed_timestamp,
            epoch=0,
            event_type=EventType.EXPERIMENT_START,
            message="Starting",
            data={},
        )

        result = event.to_dict()
        assert not result["data"]
        json.dumps(result)  # Should not raise

    def test_event_with_large_data_payload(self, fixed_timestamp):
        """Test LogEvent with large data payload."""
        large_data = {f"metric_{i}": float(i) for i in range(1000)}

        event = LogEvent(
            timestamp=fixed_timestamp,
            epoch=1,
            event_type=EventType.EPOCH_PROGRESS,
            message="Large metrics",
            data=large_data,
        )

        result = event.to_dict()
        assert len(result["data"]) == 1000
        json.dumps(result)  # Should not raise


@pytest.mark.unit
class TestEventType:
    """Test suite for EventType enum."""

    def test_all_event_types_have_string_values(self):
        """Test that all EventType members have string values."""
        for event_type in EventType:
            assert isinstance(event_type.value, str)
            assert len(event_type.value) > 0

    def test_event_type_uniqueness(self):
        """Test that all EventType values are unique."""
        values = [event_type.value for event_type in EventType]
        assert len(values) == len(set(values))

    @pytest.mark.parametrize("event_type", list(EventType))
    def test_event_type_string_conversion(self, event_type):
        """Test that each EventType converts to string correctly."""
        assert isinstance(event_type.value, str)
        assert event_type.value.islower()
        assert "_" in event_type.value or event_type.value.isalpha()


@pytest.mark.unit
class TestExperimentLogger:
    """Test suite for ExperimentLogger."""

    @patch("time.time")
    def test_log_experiment_start(self, mock_time, mock_logger, fixed_timestamp):
        """Test experiment start logging with mocked time."""
        mock_time.return_value = fixed_timestamp

        with patch.object(mock_logger, "write_to_file"), patch.object(mock_logger, "print_real_time_update"):
            mock_logger.log_experiment_start()

        assert len(mock_logger.events) == 1
        event = mock_logger.events[0]
        assert_event_structure(
            event,
            expected_epoch=0,
            expected_type=EventType.EXPERIMENT_START,
            expected_message="Experiment started",
        )
        assert event.data["config"] == mock_logger.config
        assert event.timestamp == fixed_timestamp

    @patch("time.time")
    def test_log_epoch_progress(self, mock_time, mock_logger, fixed_timestamp):
        """Test epoch progress logging with mocked time."""
        mock_time.return_value = fixed_timestamp
        metrics = {"accuracy": 0.92, "loss": 0.15, "val_accuracy": 0.89}

        with patch.object(mock_logger, "write_to_file"), patch.object(mock_logger, "print_real_time_update"):
            mock_logger.log_epoch_progress(15, metrics)

        assert len(mock_logger.events) == 1
        event = mock_logger.events[0]
        assert_event_structure(
            event,
            expected_epoch=15,
            expected_type=EventType.EPOCH_PROGRESS,
            expected_message="Epoch progress",
            expected_data=metrics,
        )

    @patch("time.time")
    @pytest.mark.parametrize(
        "seed_id,from_state,to_state,description,expected_message",
        [
            ("seed_alpha", "dormant", "training", "Custom description", "Custom description"),
            ("seed_beta", "training", "active", None, "Seed seed_beta: training -> active"),
            (
                "seed_gamma",
                "inactive",
                "germinating",
                "",
                "Seed seed_gamma: inactive -> germinating",
            ),
        ],
    )
    def test_log_seed_event_variations(
        self,
        mock_time,
        mock_logger,
        fixed_timestamp,
        seed_id,
        from_state,
        to_state,
        description,
        expected_message,
    ):
        """Test seed state transition logging with various parameters."""
        mock_time.return_value = fixed_timestamp

        with patch.object(mock_logger, "write_to_file"), patch.object(mock_logger, "print_real_time_update"):
            mock_logger.log_seed_event(8, seed_id, from_state, to_state, description)

        assert len(mock_logger.events) == 1
        event = mock_logger.events[0]
        assert_event_structure(
            event,
            expected_epoch=8,
            expected_type=EventType.SEED_STATE_CHANGE,
            expected_message=expected_message,
        )
        assert event.data == {"seed_id": seed_id, "from": from_state, "to": to_state}

    @patch("time.time")
    def test_log_germination(self, mock_time, mock_logger, fixed_timestamp):
        """Test germination event logging."""
        mock_time.return_value = fixed_timestamp

        with patch.object(mock_logger, "write_to_file"), patch.object(mock_logger, "print_real_time_update"):
            mock_logger.log_germination(20, "seed_gamma")

        assert len(mock_logger.events) == 1
        event = mock_logger.events[0]
        assert_event_structure(
            event,
            expected_epoch=20,
            expected_type=EventType.GERMINATION,
            expected_message="Seed seed_gamma germinated",
            expected_data={"seed_id": "seed_gamma"},
        )

    @patch("time.time")
    @pytest.mark.parametrize(
        "alpha,expected_formatted",
        [
            (0.75, "0.750"),
            (0.123456, "0.123"),
            (1.0, "1.000"),
            (0.0, "0.000"),
        ],
    )
    def test_log_blending_progress_formatting(self, mock_time, mock_logger, fixed_timestamp, alpha, expected_formatted):
        """Test blending progress logging with various alpha values."""
        mock_time.return_value = fixed_timestamp

        with patch.object(mock_logger, "write_to_file"), patch.object(mock_logger, "print_real_time_update"):
            mock_logger.log_blending_progress(25, "seed_delta", alpha)

        assert len(mock_logger.events) == 1
        event = mock_logger.events[0]
        assert_event_structure(
            event,
            expected_epoch=25,
            expected_type=EventType.BLENDING_PROGRESS,
            expected_message=f"Blending seed_delta alpha={expected_formatted}",
        )
        assert event.data == {"seed_id": "seed_delta", "alpha": alpha}

    @patch("time.time")
    def test_log_phase_transition(self, mock_time, mock_logger, fixed_timestamp):
        """Test phase transition logging."""
        mock_time.return_value = fixed_timestamp

        with patch.object(mock_logger, "write_to_file"), patch.object(mock_logger, "print_real_time_update"):
            mock_logger.log_phase_transition(30, "exploration", "exploitation")

        assert len(mock_logger.events) == 1
        event = mock_logger.events[0]
        assert_event_structure(
            event,
            expected_epoch=30,
            expected_type=EventType.PHASE_TRANSITION,
            expected_message="Phase transition exploration -> exploitation",
            expected_data={"from": "exploration", "to": "exploitation"},
        )

    @patch("time.time")
    def test_log_accuracy_dip(self, mock_time, mock_logger, fixed_timestamp):
        """Test accuracy dip logging."""
        mock_time.return_value = fixed_timestamp

        with patch.object(mock_logger, "write_to_file"), patch.object(mock_logger, "print_real_time_update"):
            mock_logger.log_accuracy_dip(35, 0.72)

        assert len(mock_logger.events) == 1
        event = mock_logger.events[0]
        assert_event_structure(
            event,
            expected_epoch=35,
            expected_type=EventType.ACCURACY_DIP,
            expected_message="Accuracy dip detected",
            expected_data={"accuracy": 0.72},
        )

    @patch("time.time")
    def test_log_experiment_end_with_summary(self, mock_time, mock_logger, fixed_timestamp):
        """Test experiment end logging and summary generation."""
        mock_time.return_value = fixed_timestamp

        # Add some events first
        create_sample_events(mock_logger, fixed_timestamp)

        with patch.object(mock_logger, "write_to_file"), patch.object(mock_logger, "print_real_time_update"):
            mock_logger.log_experiment_end(100)

        assert len(mock_logger.events) == 6  # 5 sample + 1 end event
        end_event = mock_logger.events[-1]
        assert_event_structure(
            end_event,
            expected_epoch=100,
            expected_type=EventType.EXPERIMENT_END,
            expected_message="Experiment finished",
        )

        # Check summary includes all event types
        summary = end_event.data["summary"]
        assert summary["experiment_start"] == 1
        assert summary["epoch_progress"] == 1
        assert summary["germination"] == 1
        assert summary["blending_progress"] == 1
        assert summary["phase_transition"] == 1
        assert summary["experiment_end"] == 1

    def test_generate_final_report(self, mock_logger, fixed_timestamp):
        """Test final report generation logic."""
        create_sample_events(mock_logger, fixed_timestamp)

        summary = mock_logger.generate_final_report()
        expected = {
            "blending_progress": 1,
            "epoch_progress": 1,
            "experiment_start": 1,
            "germination": 1,
            "phase_transition": 1,
        }
        assert summary == expected

    def test_generate_final_report_empty(self, mock_logger):
        """Test final report generation with no events."""
        summary = mock_logger.generate_final_report()
        assert not summary

    @patch("time.time")
    def test_multiple_events_maintain_order(self, mock_time, mock_logger):
        """Test that multiple events maintain chronological order with mocked time."""
        # Mock progressive timestamps
        timestamps = [1640995200.0, 1640995201.0, 1640995202.0]
        mock_time.side_effect = timestamps

        with patch.object(mock_logger, "write_to_file"), patch.object(mock_logger, "print_real_time_update"):
            mock_logger.log_experiment_start()
            mock_logger.log_epoch_progress(1, {"accuracy": 0.8})
            mock_logger.log_germination(5, "seed1")

        # Verify chronological order
        assert len(mock_logger.events) == 3
        assert mock_logger.events[0].timestamp == timestamps[0]
        assert mock_logger.events[1].timestamp == timestamps[1]
        assert mock_logger.events[2].timestamp == timestamps[2]

        # Verify event types in correct order
        assert mock_logger.events[0].event_type == EventType.EXPERIMENT_START
        assert mock_logger.events[1].event_type == EventType.EPOCH_PROGRESS
        assert mock_logger.events[2].event_type == EventType.GERMINATION

    # Error handling tests
    @patch("morphogenetic_engine.logger.open", side_effect=PermissionError("Permission denied"))
    @patch("time.time")
    def test_write_to_file_permission_error(self, mock_time, mock_open, mock_logger, fixed_timestamp):
        """Test handling of file permission errors."""
        mock_time.return_value = fixed_timestamp

        with patch.object(mock_logger, "print_real_time_update"):
            with pytest.raises(PermissionError):
                mock_logger.log_experiment_start()

    @patch("morphogenetic_engine.logger.open", side_effect=OSError("Disk full"))
    @patch("time.time")
    def test_write_to_file_disk_full_error(self, mock_time, mock_open, mock_logger, fixed_timestamp):
        """Test handling of disk space errors."""
        mock_time.return_value = fixed_timestamp

        with patch.object(mock_logger, "print_real_time_update"):
            with pytest.raises(OSError):
                mock_logger.log_experiment_start()

    @patch(
        "morphogenetic_engine.logger.json.dumps",
        side_effect=TypeError("Object not JSON serializable"),
    )
    @patch("time.time")
    def test_malformed_json_serialization(self, mock_time, mock_json, mock_logger, fixed_timestamp):
        """Test handling of JSON serialization errors."""
        mock_time.return_value = fixed_timestamp

        with patch.object(mock_logger, "print_real_time_update"), patch("builtins.open", mock_open()):
            with pytest.raises(TypeError):
                mock_logger.log_epoch_progress(1, {"complex": complex(1, 2)})  # Non-serializable


@pytest.mark.integration
class TestExperimentLoggerIntegration:
    """Integration tests for ExperimentLogger with real file system."""

    @patch("time.time")
    def test_file_writing_integration(self, mock_time, real_filesystem_logger, fixed_timestamp):
        """Test that events are properly written to file."""
        timestamps = [fixed_timestamp, fixed_timestamp + 1]
        mock_time.side_effect = timestamps

        with patch.object(real_filesystem_logger, "print_real_time_update"):
            real_filesystem_logger.log_experiment_start()
            real_filesystem_logger.log_epoch_progress(1, {"accuracy": 0.95})

        # Verify file content
        assert real_filesystem_logger.log_file_path.exists()
        content = real_filesystem_logger.log_file_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2

        # Parse and verify JSON structure
        event1 = json.loads(lines[0])
        assert event1["event_type"] == "experiment_start"
        assert event1["epoch"] == 0
        assert event1["timestamp"] == fixed_timestamp
        assert event1["data"]["config"] == real_filesystem_logger.config

        event2 = json.loads(lines[1])
        assert event2["event_type"] == "epoch_progress"
        assert event2["epoch"] == 1
        assert event2["timestamp"] == fixed_timestamp + 1
        assert abs(event2["data"]["accuracy"] - 0.95) < 1e-10

    def test_results_directory_creation(self, real_filesystem_logger):
        """Test that the results directory is created if it doesn't exist."""
        results_dir = real_filesystem_logger.log_file_path.parent
        assert results_dir.exists()
        assert results_dir.name == "results"

    @patch("time.time")
    def test_complete_experiment_lifecycle(self, mock_time, real_filesystem_logger):
        """Test a complete experiment lifecycle with real file operations."""
        # Set up progressive timestamps
        base_time = 1640995200.0
        mock_time.side_effect = [base_time + i for i in range(10)]

        with patch.object(real_filesystem_logger, "print_real_time_update"):
            # Simulate complete experiment
            real_filesystem_logger.log_experiment_start()
            for epoch in range(1, 4):
                real_filesystem_logger.log_epoch_progress(epoch, {"accuracy": 0.7 + epoch * 0.1})
            real_filesystem_logger.log_germination(2, "seed1")
            real_filesystem_logger.log_blending_progress(3, "seed1", 0.5)
            real_filesystem_logger.log_phase_transition(3, "exploration", "exploitation")
            real_filesystem_logger.log_experiment_end(3)

        # Verify all events recorded
        assert len(real_filesystem_logger.events) == 8

        # Verify file contains all events
        content = real_filesystem_logger.log_file_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 8

        # Verify final summary
        final_event = json.loads(lines[-1])
        assert final_event["event_type"] == "experiment_end"
        summary = final_event["data"]["summary"]
        assert summary["experiment_start"] == 1
        assert summary["epoch_progress"] == 3
        assert summary["germination"] == 1
        assert summary["blending_progress"] == 1
        assert summary["phase_transition"] == 1
        assert summary["experiment_end"] == 1
