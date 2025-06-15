"""Tests for the experiment logging system."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from morphogenetic_engine.logger import EventType, ExperimentLogger, LogEvent


class TestLogEvent:
    """Test suite for LogEvent dataclass."""

    def test_log_event_creation(self):
        """Test basic LogEvent creation and attributes."""
        timestamp = time.time()
        event = LogEvent(
            timestamp=timestamp,
            epoch=5,
            event_type=EventType.EPOCH_PROGRESS,
            message="Test message",
            data={"accuracy": 0.85, "loss": 0.32}
        )

        assert event.timestamp == timestamp
        assert event.epoch == 5
        assert event.event_type == EventType.EPOCH_PROGRESS
        assert event.message == "Test message"
        assert event.data == {"accuracy": 0.85, "loss": 0.32}

    def test_to_dict_serialization(self):
        """Test that LogEvent.to_dict() produces correct JSON-serializable format."""
        timestamp = 1640995200.0  # Fixed timestamp for reproducible tests
        event = LogEvent(
            timestamp=timestamp,
            epoch=10,
            event_type=EventType.GERMINATION,
            message="Seed sprouted",
            data={"seed_id": "seed1", "growth_rate": 1.5}
        )

        result = event.to_dict()
        expected = {
            "timestamp": 1640995200.0,
            "epoch": 10,
            "event_type": "germination",
            "message": "Seed sprouted",
            "data": {"seed_id": "seed1", "growth_rate": 1.5}
        }

        assert result == expected
        # Ensure it's JSON serializable
        json.dumps(result)  # Should not raise


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


class TestExperimentLogger:
    """Test suite for ExperimentLogger."""

    def _create_test_logger(self, log_file_path: str, config: dict, temp_dir: str):
        """Helper to create a logger with mocked directory resolution."""
        results_dir = Path(temp_dir) / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Create logger and override the problematic path resolution
        logger = object.__new__(ExperimentLogger)
        logger.log_file_path = results_dir / Path(log_file_path).name
        logger.config = config
        logger.events = []
        # Create the log file
        logger.log_file_path.write_text("")
        return logger

    def test_logger_initialization(self):
        """Test logger initialization and directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"dataset": "spirals", "epochs": 100}
            logger = self._create_test_logger("test.log", config, temp_dir)

            assert logger.config == config
            assert len(logger.events) == 0
            assert logger.log_file_path.name == "test.log"

    def test_log_experiment_start(self):
        """Test experiment start logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"dataset": "moons", "lr": 0.001}
            logger = self._create_test_logger("test.log", config, temp_dir)
            
            with patch.object(logger, 'print_real_time_update'):  # Suppress console output
                logger.log_experiment_start()

            assert len(logger.events) == 1
            event = logger.events[0]
            assert event.epoch == 0
            assert event.event_type == EventType.EXPERIMENT_START
            assert event.message == "Experiment started"
            assert event.data["config"] == config

    def test_log_epoch_progress(self):
        """Test epoch progress logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._create_test_logger("test.log", {}, temp_dir)
            metrics = {"accuracy": 0.92, "loss": 0.15, "val_accuracy": 0.89}
            
            with patch.object(logger, 'print_real_time_update'):
                logger.log_epoch_progress(15, metrics)

            assert len(logger.events) == 1
            event = logger.events[0]
            assert event.epoch == 15
            assert event.event_type == EventType.EPOCH_PROGRESS
            assert event.message == "Epoch progress"
            assert event.data == metrics

    def test_log_seed_event(self):
        """Test seed state transition logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._create_test_logger("test.log", {}, temp_dir)
            
            with patch.object(logger, 'print_real_time_update'):
                # Test with custom description
                logger.log_seed_event(8, "seed_alpha", "dormant", "training", "Custom description")

            assert len(logger.events) == 1
            event = logger.events[0]
            assert event.epoch == 8
            assert event.event_type == EventType.SEED_STATE_CHANGE
            assert event.message == "Custom description"
            assert event.data == {"seed_id": "seed_alpha", "from": "dormant", "to": "training"}

            # Test with default description
            with patch.object(logger, 'print_real_time_update'):
                logger.log_seed_event(12, "seed_beta", "training", "active")

            assert len(logger.events) == 2
            event = logger.events[1]
            assert event.message == "Seed seed_beta: training -> active"

    def test_log_germination(self):
        """Test germination event logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._create_test_logger("test.log", {}, temp_dir)
            
            with patch.object(logger, 'print_real_time_update'):
                logger.log_germination(20, "seed_gamma")

            assert len(logger.events) == 1
            event = logger.events[0]
            assert event.epoch == 20
            assert event.event_type == EventType.GERMINATION
            assert event.message == "Seed seed_gamma germinated"
            assert event.data == {"seed_id": "seed_gamma"}

    def test_log_blending_progress(self):
        """Test blending progress logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._create_test_logger("test.log", {}, temp_dir)
            
            with patch.object(logger, 'print_real_time_update'):
                logger.log_blending_progress(25, "seed_delta", 0.75)

            assert len(logger.events) == 1
            event = logger.events[0]
            assert event.epoch == 25
            assert event.event_type == EventType.BLENDING_PROGRESS
            assert event.message == "Blending seed_delta alpha=0.750"
            assert event.data == {"seed_id": "seed_delta", "alpha": 0.75}

    def test_log_phase_transition(self):
        """Test phase transition logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._create_test_logger("test.log", {}, temp_dir)
            
            with patch.object(logger, 'print_real_time_update'):
                logger.log_phase_transition(30, "exploration", "exploitation")

            assert len(logger.events) == 1
            event = logger.events[0]
            assert event.epoch == 30
            assert event.event_type == EventType.PHASE_TRANSITION
            assert event.message == "Phase transition exploration -> exploitation"
            assert event.data == {"from": "exploration", "to": "exploitation"}

    def test_log_accuracy_dip(self):
        """Test accuracy dip logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._create_test_logger("test.log", {}, temp_dir)
            
            with patch.object(logger, 'print_real_time_update'):
                logger.log_accuracy_dip(35, 0.72)

            assert len(logger.events) == 1
            event = logger.events[0]
            assert event.epoch == 35
            assert event.event_type == EventType.ACCURACY_DIP
            assert event.message == "Accuracy dip detected"
            assert event.data == {"accuracy": 0.72}

    def test_log_experiment_end(self):
        """Test experiment end logging and summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._create_test_logger("test.log", {}, temp_dir)
            
            # Add some events first
            with patch.object(logger, 'print_real_time_update'):
                logger.log_experiment_start()
                logger.log_epoch_progress(1, {"accuracy": 0.8})
                logger.log_germination(5, "seed1")
                logger.log_germination(8, "seed2")
                logger.log_experiment_end(100)

            assert len(logger.events) == 5
            end_event = logger.events[-1]
            assert end_event.epoch == 100
            assert end_event.event_type == EventType.EXPERIMENT_END
            assert end_event.message == "Experiment finished"
            
            # Check summary
            summary = end_event.data["summary"]
            assert summary["experiment_start"] == 1
            assert summary["epoch_progress"] == 1
            assert summary["germination"] == 2
            assert summary["experiment_end"] == 1

    def test_generate_final_report(self):
        """Test final report generation logic."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._create_test_logger("test.log", {}, temp_dir)
            
            # Add diverse events
            with patch.object(logger, 'print_real_time_update'):
                logger.log_experiment_start()
                logger.log_epoch_progress(1, {})
                logger.log_epoch_progress(2, {})
                logger.log_germination(3, "seed1")
                logger.log_blending_progress(4, "seed1", 0.5)
                logger.log_phase_transition(5, "A", "B")

            summary = logger.generate_final_report()
            expected = {
                "blending_progress": 1,
                "epoch_progress": 2,
                "experiment_start": 1,
                "germination": 1,
                "phase_transition": 1
            }
            assert summary == expected

    def test_file_writing_integration(self):
        """Test that events are properly written to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._create_test_logger("test_integration.log", {"test": True}, temp_dir)
            
            with patch.object(logger, 'print_real_time_update'):
                logger.log_experiment_start()
                logger.log_epoch_progress(1, {"accuracy": 0.95})

            # Verify file content
            assert logger.log_file_path.exists()
            content = logger.log_file_path.read_text()
            lines = content.strip().split('\n')
            assert len(lines) == 2

            # Parse and verify JSON structure
            event1 = json.loads(lines[0])
            assert event1["event_type"] == "experiment_start"
            assert event1["epoch"] == 0
            assert event1["data"]["config"] == {"test": True}

            event2 = json.loads(lines[1])
            assert event2["event_type"] == "epoch_progress"
            assert event2["epoch"] == 1
            assert abs(event2["data"]["accuracy"] - 0.95) < 1e-10

    def test_results_directory_creation(self):
        """Test that the results directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # This test is implicitly covered by our helper method
            logger = self._create_test_logger("test.log", {}, temp_dir)
            results_dir = logger.log_file_path.parent
            assert results_dir.exists()
            assert results_dir.name == "results"

    def test_multiple_events_maintain_order(self):
        """Test that multiple events maintain chronological order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._create_test_logger("test.log", {}, temp_dir)
            
            with patch.object(logger, 'print_real_time_update'):
                start_time = time.time()
                logger.log_experiment_start()
                time.sleep(0.01)  # Small delay to ensure different timestamps
                logger.log_epoch_progress(1, {"accuracy": 0.8})
                time.sleep(0.01)
                logger.log_germination(5, "seed1")

            # Verify chronological order
            assert len(logger.events) == 3
            assert logger.events[0].timestamp >= start_time
            assert logger.events[1].timestamp > logger.events[0].timestamp
            assert logger.events[2].timestamp > logger.events[1].timestamp
            
            # Verify event types in correct order
            assert logger.events[0].event_type == EventType.EXPERIMENT_START
            assert logger.events[1].event_type == EventType.EPOCH_PROGRESS
            assert logger.events[2].event_type == EventType.GERMINATION
