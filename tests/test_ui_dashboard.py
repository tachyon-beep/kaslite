"""
Refactored tests for the CLI dashboard module, updated for the new event-driven architecture.
"""

# pylint: disable=protected-access,missing-function-docstring,redefined-outer-name
import re
from dataclasses import dataclass
from io import StringIO
from typing import Any
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from morphogenetic_engine.events import MetricsUpdatePayload, PhaseUpdatePayload, SeedMetricsUpdatePayload, SeedState
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
        with patch("rich.live.Live"):
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
        dashboard = RichDashboard()
        assert dashboard.console is not None
        assert isinstance(dashboard.console, Console)

    def test_update_metrics(self, dashboard: RichDashboard):
        """Test that the dashboard correctly processes metrics updates."""
        metrics = {"val_loss": 0.25, "val_acc": 0.80, "train_loss": 0.30}
        payload: MetricsUpdatePayload = {"epoch": 5, "metrics": metrics, "timestamp": 123.0}

        dashboard.update_metrics(payload)

        # Verify the original metrics were stored (the metrics manager adds derived metrics)
        latest = dashboard.metrics_manager.latest_metrics
        assert latest["val_loss"] == pytest.approx(0.25)
        assert latest["val_acc"] == pytest.approx(0.80)
        assert latest["train_loss"] == pytest.approx(0.30)

    def test_transition_phase(self, dashboard: RichDashboard):
        """Test that the dashboard correctly processes phase transitions."""
        payload: PhaseUpdatePayload = {
            "epoch": 10,
            "from_phase": "warm_up",
            "to_phase": "adaptation",
            "total_epochs_in_phase": 50,
            "timestamp": 123.0,
        }

        dashboard.transition_phase(payload)

        assert dashboard.phase_start_epoch == 10
        assert dashboard.current_phase_epochs == 50

    def test_update_metrics_sequence(self, dashboard: RichDashboard):
        """Test that the dashboard correctly processes sequential metrics updates."""
        # First metrics update
        metrics1 = {"val_acc": 0.80, "val_loss": 0.25}
        payload1: MetricsUpdatePayload = {"epoch": 5, "metrics": metrics1, "timestamp": 123.0}
        dashboard.update_metrics(payload1)

        latest = dashboard.metrics_manager.latest_metrics
        assert latest["val_acc"] == pytest.approx(0.80)
        assert latest["val_loss"] == pytest.approx(0.25)

        # Second metrics update
        metrics2 = {"val_acc": 0.85, "val_loss": 0.20}
        payload2: MetricsUpdatePayload = {"epoch": 6, "metrics": metrics2, "timestamp": 124.0}
        dashboard.update_metrics(payload2)

        latest = dashboard.metrics_manager.latest_metrics
        assert latest["val_acc"] == pytest.approx(0.85)
        assert latest["val_loss"] == pytest.approx(0.20)

    def test_update_seed_metrics(self, dashboard: RichDashboard):
        """Test that the dashboard correctly processes seed metrics updates."""
        payload: SeedMetricsUpdatePayload = {"seed_id": (0, 42), "state": SeedState.ACTIVE, "alpha": 0.99}

        dashboard.update_seed_metrics(payload)

        assert "L0_S42" in dashboard.seed_states
        assert dashboard.seed_states["L0_S42"]["state"] == SeedState.ACTIVE
        assert dashboard.seed_states["L0_S42"]["alpha"] == pytest.approx(0.99)

    def test_context_manager_protocol(self, dashboard: RichDashboard):
        """Test the dashboard's context manager functionality."""
        # Mock the start/stop methods instead of live directly
        dashboard.start = Mock()
        dashboard.stop = Mock()

        with dashboard as db:
            assert db is dashboard
            dashboard.start.assert_called_once()
        dashboard.stop.assert_called_once()


class TestRichDashboardIntegration:
    """Integration tests with real Rich components."""

    @pytest.fixture
    def console_with_output(self) -> tuple[Console, StringIO]:
        """Provides a real Console instance with captured output."""
        return create_test_console(width=120)

    def test_basic_dashboard_functionality(self, console_with_output: tuple[Console, StringIO]):
        """Test basic dashboard operations without the full lifecycle."""
        console, _ = console_with_output
        params = {"num_layers": 2, "seeds_per_layer": 2, "warm_up_epochs": 5, "adaptation_epochs": 5}
        dashboard = RichDashboard(console=console, experiment_params=params)

        # Initialize experiment
        dashboard.initialize_experiment(params)

        # Update metrics
        metrics_payload: MetricsUpdatePayload = {"epoch": 1, "metrics": {"val_acc": 0.5, "val_loss": 0.3}, "timestamp": 123.0}
        dashboard.update_metrics(metrics_payload)

        # Update seed metrics
        seed_payload: SeedMetricsUpdatePayload = {"seed_id": (0, 0), "state": SeedState.GERMINATED, "alpha": 0.8}
        dashboard.update_seed_metrics(seed_payload)

        # Verify internal state
        assert dashboard.metrics_manager.latest_metrics["val_acc"] == pytest.approx(0.5)
        assert "L0_S0" in dashboard.seed_states
        assert dashboard.seed_states["L0_S0"]["state"] == SeedState.GERMINATED

    def test_grid_manager_functionality(self, console_with_output: tuple[Console, StringIO]):
        """Test the grid manager functionality through the dashboard."""
        console, _ = console_with_output
        dashboard = RichDashboard(console=console)

        # Manually set some seed states for testing
        dashboard.seed_states = {
            "L0_S0": {"state": SeedState.ACTIVE},
            "L1_S1": {"state": SeedState.DORMANT},
            "L0_S1": {"state": SeedState.BLENDING},
        }

        # Test the grid manager's method through the dashboard
        result = dashboard.grid_manager._get_seed_states_by_layer(num_layers=2, seeds_per_layer=2)

        # The result should be organized by layer
        assert 0 in result
        assert 1 in result
        assert len(result[0]) == 2  # 2 seeds per layer
        assert len(result[1]) == 2

    def test_panel_creation(self, console_with_output: tuple[Console, StringIO]):
        """Test that panels can be created without errors."""
        console, _ = console_with_output
        params = {"num_layers": 2, "seeds_per_layer": 2}
        dashboard = RichDashboard(console=console, experiment_params=params)

        # Test panel creation (these should not raise exceptions)
        info_panel = dashboard.panel_factory.create_info_panel()
        metrics_panel = dashboard.metrics_manager.create_metrics_table_panel(params)
        seed_panel = dashboard.grid_manager.create_seed_box_panel(params)

        # Basic assertions - panels should be created
        assert info_panel is not None
        assert metrics_panel is not None
        assert seed_panel is not None
