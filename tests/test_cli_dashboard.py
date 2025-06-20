"""
Refactored tests for the CLI dashboard module, updated for the new event-driven architecture.
"""

# pylint: disable=protected-access,missing-function-docstring,redefined-outer-name
import re
from dataclasses import dataclass
from io import StringIO
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from rich.console import Console
from rich.layout import Layout

from morphogenetic_engine.core import Population
from morphogenetic_engine.events import (
    EventType,
    ExperimentEndPayload,
    ExperimentStartPayload,
    LogEvent,
    MessagePayload,
    MetricsPayload,
    PopulationState,
    SeedState,
)
from morphogenetic_engine.experiment import Experiment
from morphogenetic_engine.ui_dashboard import RichDashboard

# Test Utilities and Fixtures


@dataclass
class TestMetrics:
    """Simple test metrics factory."""

    epoch: int = 0
    val_loss: float = 0.0
    val_acc: float = 0.0
    train_loss: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by dashboard."""
        return {
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "train_loss": self.train_loss,
        }


def create_test_console(width: int = 80) -> tuple[Console, StringIO]:
    """Factory for creating test consoles with captured output."""
    string_io = StringIO()
    console = Console(file=string_io, force_terminal=True, width=width, no_color=True, legacy_windows=False)
    return console, string_io


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


# Pytest Configuration

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*use pytest.approx.*:UserWarning"),
]


class TestRichDashboardUnit:
    """Unit tests for RichDashboard core functionality using the new event system."""

    @pytest.fixture
    def mock_console(self) -> Mock:
        """Create a mock console for testing."""
        return Mock(spec=Console)

    @pytest.fixture
    def dashboard(self, mock_console: Mock) -> RichDashboard:
        """Create a dashboard instance with mocked dependencies."""
        with patch("morphogenetic_engine.ui_dashboard.Live"):
            dashboard = RichDashboard(console=mock_console)
            dashboard.live = Mock()
            return dashboard

    def test_dashboard_initialization(self):
        """Test dashboard initialization with and without a console."""
        # Test with provided console
        mock_console = Mock(spec=Console)
        dashboard = RichDashboard(console=mock_console)
        assert dashboard.console == mock_console

        # Test without console (should create default)
        with patch("morphogenetic_engine.ui_dashboard.Console") as mock_console_class:
            mock_instance = Mock()
            mock_console_class.return_value = mock_instance
            dashboard = RichDashboard()
            assert dashboard.console == mock_instance

    def test_handle_system_init_event(self, dashboard: RichDashboard):
        """Test that the dashboard correctly processes the SYSTEM_INIT event."""
        params = {"epochs": 200, "learning_rate": 0.01}
        payload = SystemInitPayload(timestamp=123, experiment_params=params)
        event = LogEvent(event_type=EventType.SYSTEM_INIT, payload=payload)

        dashboard.handle_event(event)

        assert dashboard.experiment_params == params
        assert dashboard.total_epochs == 200
        dashboard.total_progress.update.assert_called_with(dashboard.total_task, total=200)
        assert "Experiment initialized" in dashboard.last_events[0]

    def test_handle_phase_update_event(self, dashboard: RichDashboard):
        """Test that the dashboard correctly processes the PHASE_UPDATE event."""
        dashboard.handle_event(
            LogEvent(
                event_type=EventType.SYSTEM_INIT,
                payload=SystemInitPayload(timestamp=0, experiment_params={"epochs": 100}),
            )
        )

        payload = PhaseUpdatePayload(
            timestamp=123,
            phase_name="TRAINING",
            epoch=10,
            total_epochs_in_phase=50,
            details="Starting training phase",
        )
        event = LogEvent(event_type=EventType.PHASE_UPDATE, payload=payload)

        dashboard.handle_event(event)

        assert dashboard.phase_start_epoch == 10
        assert dashboard.current_phase_epochs == 50
        dashboard.phase_progress.update.assert_called_with(dashboard.phase_task, completed=0, total=50, description="TRAINING")
        assert "Phase changed to TRAINING" in dashboard.last_events[-1]

    def test_handle_metrics_update_event(self, dashboard: RichDashboard):
        """Test that the dashboard correctly processes the METRICS_UPDATE event."""
        # Set up initial state
        dashboard.handle_event(
            LogEvent(
                event_type=EventType.SYSTEM_INIT,
                payload=SystemInitPayload(timestamp=0, experiment_params={"epochs": 100}),
            )
        )
        dashboard.handle_event(
            LogEvent(
                event_type=EventType.PHASE_UPDATE,
                payload=PhaseUpdatePayload(timestamp=1, phase_name="TRAIN", epoch=0, total_epochs_in_phase=50),
            )
        )

        # First metrics update
        metrics1 = {"val_acc": 0.80, "val_loss": 0.25}
        payload1 = MetricsUpdatePayload(timestamp=123, epoch=5, metrics=metrics1)
        event1 = LogEvent(event_type=EventType.METRICS_UPDATE, payload=payload1)
        dashboard.handle_event(event1)

        assert dashboard.latest_metrics == metrics1
        assert dashboard.previous_metrics == {}
        dashboard.total_progress.update.assert_called_with(dashboard.total_task, completed=5)
        dashboard.phase_progress.update.assert_called_with(dashboard.phase_task, completed=5)

        # Second metrics update
        metrics2 = {"val_acc": 0.85, "val_loss": 0.20}
        payload2 = MetricsUpdatePayload(timestamp=124, epoch=6, metrics=metrics2)
        event2 = LogEvent(event_type=EventType.METRICS_UPDATE, payload=payload2)
        dashboard.handle_event(event2)

        assert dashboard.latest_metrics == metrics2
        assert dashboard.previous_metrics == metrics1
        dashboard.total_progress.update.assert_called_with(dashboard.total_task, completed=6)
        dashboard.phase_progress.update.assert_called_with(dashboard.phase_task, completed=6)

    def test_handle_seed_state_update_event(self, dashboard: RichDashboard):
        """Test that the dashboard correctly processes the SEED_STATE_UPDATE event."""
        payload = SeedStateUpdatePayload(
            timestamp=123,
            seed_id="L0_42",
            state="active",
            phenotype={"val_acc": 0.99},
            details="Seed became active",
        )
        event = LogEvent(event_type=EventType.SEED_STATE_UPDATE, payload=payload)

        dashboard.handle_event(event)

        assert "L0_42" in dashboard.seed_states
        assert dashboard.seed_states["L0_42"]["state"] == "active"
        assert dashboard.seed_states["L0_42"]["val_acc"] == pytest.approx(0.99)
        assert "L0_42" in dashboard.seed_log_events[0]

    def test_context_manager_protocol(self, dashboard: RichDashboard):
        """Test the dashboard's context manager functionality."""
        with dashboard as db:
            assert db is dashboard
            dashboard.live.start.assert_called_once()
        dashboard.live.stop.assert_called_once()


class TestRichDashboardIntegration:
    """Integration tests with real Rich components."""

    @pytest.fixture
    def console_with_output(self) -> tuple[Console, StringIO]:
        """Provides a real Console instance with captured output."""
        return create_test_console(width=120)

    def test_full_lifecycle_rendering(self, console_with_output: tuple[Console, StringIO]):
        """Test a typical sequence of events and verify the rendered output."""
        console, string_io = console_with_output

        with RichDashboard(console=console, experiment_params={"num_layers": 2, "seeds_per_layer": 2}) as dashboard:
            # System Init
            dashboard.handle_event(
                LogEvent(
                    event_type=EventType.SYSTEM_INIT,
                    payload=SystemInitPayload(timestamp=0, experiment_params={"epochs": 10}),
                )
            )

            # Phase Update
            dashboard.handle_event(
                LogEvent(
                    event_type=EventType.PHASE_UPDATE,
                    payload=PhaseUpdatePayload(timestamp=1, phase_name="GROW", epoch=0, total_epochs_in_phase=5),
                )
            )

            # Seed Update
            dashboard.handle_event(
                LogEvent(
                    event_type=EventType.SEED_STATE_UPDATE,
                    payload=SeedStateUpdatePayload(timestamp=2, seed_id="L0_0", state="active"),
                )
            )

            # Metrics Update
            dashboard.handle_event(
                LogEvent(
                    event_type=EventType.METRICS_UPDATE,
                    payload=MetricsUpdatePayload(timestamp=3, epoch=1, metrics={"val_acc": 0.5}),
                )
            )

            # Force a refresh to capture output
            dashboard.refresh()

        output = strip_ansi_codes(string_io.getvalue())

        # Verify key pieces of information are in the final output
        assert "QUICKSILVER" in output
        assert "Overall Progress" in output
        assert "GROW" in output  # Phase name
        assert "Live Metrics" in output
        assert "val_acc" in output
        assert "0.5000" in output
        assert "Seed Box" in output
        assert "L0_0" in dashboard.seed_states  # Check internal state
        assert "Event Log" in output
        assert "Experiment initialized" in output
        assert "Phase changed to GROW" in output

    def test_get_seed_states_by_layer(self, console_with_output: tuple[Console, StringIO]):
        """Test the logic for organizing seed states into layers for grid rendering."""
        console, _ = console_with_output
        dashboard = RichDashboard(console=console)
        dashboard.seed_states = {
            "L0_0": {"state": "active"},
            "L1_1": {"state": "dormant"},
            "L0_1": {"state": "blending"},
            "malformed": {"state": "active"},  # Should be ignored
            "L99_99": {"state": "active"},  # Should be ignored if out of bounds
        }

        # Test with valid dimensions
        result = dashboard._get_seed_states_by_layer(num_layers=2, seeds_per_layer=2)
        expected = {
            0: ["active", "blending"],
            1: [None, "dormant"],
        }
        assert result == expected

        # Test with dimensions that exclude some seeds
        result_small = dashboard._get_seed_states_by_layer(num_layers=1, seeds_per_layer=1)
        expected_small = {0: ["active"]}
        assert result_small == expected_small

    def test_grid_rendering_with_seed_data(self, console_with_output: tuple[Console, StringIO]):
        """Verify that the seed box and network strain grids render correctly."""
        console, string_io = console_with_output
        params = {"num_layers": 2, "seeds_per_layer": 2}
        dashboard = RichDashboard(console=console, experiment_params=params)
        dashboard._setup_layout()

        # Handle a seed update event
        dashboard.handle_event(
            LogEvent(
                event_type=EventType.SEED_STATE_UPDATE,
                payload=SeedStateUpdatePayload(timestamp=1, seed_id="L0_1", state="germinated"),
            )
        )
        dashboard.handle_event(
            LogEvent(
                event_type=EventType.SEED_STATE_UPDATE,
                payload=SeedStateUpdatePayload(timestamp=2, seed_id="L1_0", state="fossilized"),
            )
        )

        # Render the specific panel
        panel = dashboard._create_seed_box_panel()
        console.print(panel)
        output = strip_ansi_codes(string_io.getvalue())

        # Check for emojis in the output
        assert dashboard.SEED_GERMINATED_EMOJI in output
        assert dashboard.SEED_FOSSILIZED_EMOJI in output
        assert dashboard.EMPTY_CELL_EMOJI in output
