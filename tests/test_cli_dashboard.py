"""
Refactored tests for the CLI dashboard module.

Key improvements from original test_cli_dashboard.py:
- Removed duplicate context manager tests (kept most comprehensive one)
- Consolidated initialization tests into parametrized version
- Completed incomplete test implementations
- Simplified overengineered DashboardTestBuilder
- Separated unit tests from integration tests more clearly
- Reduced code duplication and improved maintainability

Test Classes:
    TestSeedState: Unit tests for SeedState data class
    TestRichDashboardUnit: Unit tests for RichDashboard core functionality
    TestRichDashboardIntegration: Integration tests with real Rich components
"""

# pylint: disable=protected-access,missing-function-docstring,redefined-outer-name
import re
from dataclasses import dataclass
from io import StringIO
from typing import Dict, Tuple
from unittest.mock import Mock, patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from rich.console import Console
from rich.layout import Layout
from rich.text import Text

from morphogenetic_engine.cli_dashboard import RichDashboard, SeedState

# Test Utilities and Fixtures


@dataclass
class TestMetrics:
    """Simple test metrics factory to replace complex builders."""

    epoch: int = 0
    val_loss: float = 0.0
    val_acc: float = 0.0
    best_acc: float = 0.0
    train_loss: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format expected by dashboard."""
        return {
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "best_acc": self.best_acc,
            "train_loss": self.train_loss,
        }


def create_test_console(width: int = 80, no_color: bool = False) -> Tuple[Console, StringIO]:
    """Factory for creating test consoles with captured output."""
    string_io = StringIO()
    console = Console(file=string_io, force_terminal=True, width=width, no_color=no_color, legacy_windows=False)
    return console, string_io


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    ansi_escape = re.compile(
        r"""
        \x1B  # ESC
        (?:     # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |       # or [ for CSI sequence
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    """,
        re.VERBOSE,
    )
    return ansi_escape.sub("", text)


# Hypothesis Strategies for Property-Based Testing

seed_id_strategy = st.text(min_size=1, max_size=50).filter(lambda s: s.strip() and not any(ord(c) < 32 for c in s))

seed_state_strategy = st.sampled_from(["dormant", "active", "blending"]) | st.text(min_size=1, max_size=20)

alpha_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# Pytest Configuration

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*use pytest.approx.*:UserWarning"),
]


class TestSeedState:
    """Unit tests for SeedState class."""

    @pytest.mark.parametrize(
        "seed_id,state,alpha,expected_content",
        [
            ("test_seed", "dormant", 0.0, "test_seed: dormant"),
            ("test_seed", "blending", 0.3, "test_seed: blending α=0.300"),
            ("test_seed", "active", 0.85, "test_seed: active α=0.850"),
            ("test_seed", "unknown", 0.0, "test_seed: unknown"),
        ],
    )
    def test_styled_status_output(self, seed_id: str, state: str, alpha: float, expected_content: str):
        """Test styled status output for various states."""
        seed = SeedState(seed_id, state, alpha)
        styled_text = seed.get_styled_status()

        assert isinstance(styled_text, Text)
        assert expected_content in str(styled_text)

    @given(seed_id_strategy, seed_state_strategy, alpha_strategy)
    @settings(max_examples=50, deadline=None)
    def test_seed_state_properties_robust(self, seed_id: str, state: str, alpha: float):
        """Property-based test ensuring SeedState robustness with any valid inputs."""
        assume(len(seed_id.strip()) > 0)

        seed = SeedState(seed_id, state, alpha)

        # Basic properties
        assert seed.seed_id == seed_id
        assert seed.state == state
        assert seed.alpha == pytest.approx(alpha)

        # Styled output always returns Text object
        styled_status = seed.get_styled_status()
        assert isinstance(styled_status, Text)
        assert seed_id in str(styled_status)

    @given(seed_id_strategy, alpha_strategy)
    @settings(max_examples=30, deadline=None)
    def test_seed_state_update_properties(self, seed_id: str, alpha: float):
        """Property-based test for seed state updates."""
        assume(len(seed_id.strip()) > 0)

        seed = SeedState(seed_id, "dormant", 0.0)
        original_id = seed.seed_id

        # Updating preserves seed_id
        seed.update("active", alpha)
        assert seed.seed_id == original_id
        assert seed.state == "active"
        assert seed.alpha == pytest.approx(alpha)

        # Multiple updates work correctly
        seed.update("blending", alpha / 2)
        assert seed.seed_id == original_id
        assert seed.state == "blending"
        assert seed.alpha == pytest.approx(alpha / 2)


class TestRichDashboardUnit:
    """Unit tests for RichDashboard core functionality."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console for testing."""
        return Mock(spec=Console)

    @pytest.fixture
    def dashboard(self, mock_console):
        """Create a dashboard instance with mocked dependencies."""
        with patch("morphogenetic_engine.cli_dashboard.Live") as mock_live:
            mock_live_instance = Mock()
            mock_live.return_value = mock_live_instance

            dashboard = RichDashboard(console=mock_console)
            dashboard.live = mock_live_instance
            return dashboard

    @pytest.mark.parametrize(
        "console_provided,console_created",
        [
            (True, False),  # Console provided, shouldn't create new one
            (False, True),  # No console provided, should create one
        ],
    )
    def test_dashboard_initialization(self, console_provided: bool, console_created: bool):  # pylint: disable=unused-argument
        """Test dashboard initialization with and without console parameter."""
        with patch("morphogenetic_engine.cli_dashboard.Live"):
            if console_provided:
                # Test with provided console
                mock_console = Mock(spec=Console)
                dashboard = RichDashboard(console=mock_console)
                assert dashboard.console == mock_console
            else:
                # Test without console (should create default)
                with patch("morphogenetic_engine.cli_dashboard.Console") as mock_console_class:
                    mock_console_instance = Mock()
                    mock_console_class.return_value = mock_console_instance

                    dashboard = RichDashboard()
                    assert dashboard.console == mock_console_instance

            # Verify initial state
            assert dashboard.current_task is None
            assert dashboard.current_phase == "init"
            assert not dashboard.seeds
            assert dashboard.metrics["epoch"] == 0
            assert dashboard.metrics["seeds_active"] == 0

    def test_seed_management(self, dashboard):
        """Test seed addition and updates."""
        # Add new seed
        dashboard.update_seed("seed_1", "blending", 0.4)

        assert "seed_1" in dashboard.seeds
        seed = dashboard.seeds["seed_1"]
        assert seed.seed_id == "seed_1"
        assert seed.state == "blending"
        assert seed.alpha == pytest.approx(0.4)

        # Update existing seed
        dashboard.update_seed("seed_1", "active", 0.8)
        assert dashboard.seeds["seed_1"].state == "active"
        assert dashboard.seeds["seed_1"].alpha == pytest.approx(0.8)

    def test_progress_updates(self, dashboard):
        """Test progress and metrics updates."""
        test_metrics = TestMetrics(val_loss=0.1, val_acc=0.8, train_loss=0.15)

        dashboard.update_progress(5, test_metrics.to_dict())

        assert dashboard.metrics["epoch"] == 5
        assert dashboard.metrics["val_loss"] == pytest.approx(0.1)
        assert dashboard.metrics["val_acc"] == pytest.approx(0.8)
        assert dashboard.metrics["train_loss"] == pytest.approx(0.15)

    def test_active_seeds_count_calculation(self, dashboard):
        """Test that active seeds count is computed correctly."""
        dashboard.update_seed("seed_1", "dormant")
        dashboard.update_seed("seed_2", "active", 0.8)
        dashboard.update_seed("seed_3", "blending", 0.3)
        dashboard.update_seed("seed_4", "active", 0.9)

        # Trigger metrics update
        dashboard.update_progress(1, {})
        assert dashboard.metrics["seeds_active"] == 2

        # Change seed state and verify count updates
        dashboard.update_seed("seed_1", "active", 0.5)
        dashboard.update_progress(2, {})
        assert dashboard.metrics["seeds_active"] == 3

    def test_phase_management(self, dashboard):
        """Test phase transitions and task management."""
        with (
            patch.object(dashboard.progress, "add_task") as mock_add_task,
            patch.object(dashboard.progress, "stop_task") as mock_stop_task,
        ):

            mock_add_task.return_value = "task_123"

            # Start new phase
            dashboard.start_phase("phase_1", 20, "Test Phase")
            assert dashboard.current_phase == "phase_1"
            assert dashboard.current_task == "task_123"
            mock_add_task.assert_called_once_with("Test Phase", total=20)

            # Start another phase (should stop previous task)
            mock_add_task.return_value = "task_456"
            dashboard.start_phase("phase_2", 30)

            mock_stop_task.assert_called_once_with("task_123")
            assert dashboard.current_task == "task_456"
            assert dashboard.current_phase == "phase_2"

    def test_context_manager_protocol(self, dashboard):
        """Comprehensive test of context manager functionality."""
        # Test entering context
        with dashboard as ctx_dashboard:
            assert ctx_dashboard is dashboard
            dashboard.live.start.assert_called_once()

        # Test exiting context
        dashboard.live.stop.assert_called_once()

        # Test with active task
        dashboard.current_task = "test_task"
        with patch.object(dashboard.progress, "stop_task") as mock_stop_task:
            dashboard.__exit__(None, None, None)
            mock_stop_task.assert_called_once_with("test_task")


class TestRichDashboardIntegration:
    """Integration tests with real Rich components."""

    @pytest.fixture
    def console_with_output(self):
        """Provides a real Console instance with captured output."""
        console, string_io = create_test_console(width=100, no_color=False)
        return console, string_io

    def test_complete_dashboard_lifecycle(self, console_with_output):
        """Integration test for complete dashboard usage lifecycle."""
        console, _ = console_with_output

        with RichDashboard(console=console) as dashboard:
            # Phase management
            dashboard.start_phase("test_phase", 10, "Integration Test")
            assert dashboard.current_phase == "test_phase"

            # Seed management
            dashboard.update_seed("seed_alpha", "active", 0.8)
            dashboard.update_seed("seed_beta", "blending", 0.3)
            assert "seed_alpha" in dashboard.seeds
            assert "seed_beta" in dashboard.seeds

            # Metrics updates
            test_metrics = TestMetrics(epoch=5, val_loss=0.25, val_acc=0.85, best_acc=0.90, train_loss=0.20)
            dashboard.update_progress(5, test_metrics.to_dict())

            # Verify final state
            expected_metrics = {
                "epoch": 5,
                "val_loss": 0.25,
                "val_acc": 0.85,
                "best_acc": 0.90,
                "train_loss": 0.20,
                "seeds_active": 1,  # Only seed_alpha is active
            }

            for key, expected_value in expected_metrics.items():
                assert dashboard.metrics[key] == pytest.approx(expected_value)

    def test_seeds_panel_rendering_and_ordering(self, console_with_output):
        """Test that seeds panel renders correctly with proper ordering."""
        console, string_io = console_with_output

        dashboard = RichDashboard(console=console)
        dashboard._setup_layout()

        # Add seeds in non-alphabetical order
        dashboard.update_seed("seed_gamma", "dormant")
        dashboard.update_seed("seed_alpha", "active", 0.8)
        dashboard.update_seed("seed_beta", "blending", 0.4)

        # Render seeds panel and capture output
        seeds_panel = dashboard._create_seeds_panel()
        dashboard.console.print(seeds_panel)
        output = strip_ansi_codes(string_io.getvalue())

        # Verify alphabetical ordering
        expected_order = [
            "seed_alpha: active α=0.800",
            "seed_beta: blending α=0.400",
            "seed_gamma: dormant",
        ]

        positions = [output.find(line) for line in expected_order]
        assert -1 not in positions, f"Not all expected seed lines found in output:\n{output}"
        assert positions == sorted(positions), f"Seeds not in alphabetical order:\n{output}"

    def test_layout_structure_creation(self, console_with_output):
        """Test that Rich layout components are created correctly."""
        console, _ = console_with_output

        dashboard = RichDashboard(console=console)
        dashboard._setup_layout()

        # Verify layout structure
        assert isinstance(dashboard.layout, Layout)

        # Test layout section access based on actual structure (progress, metrics, seeds)
        try:
            progress_section = dashboard.layout["progress"]
            metrics_section = dashboard.layout["metrics"]
            seeds_section = dashboard.layout["seeds"]
            # Verify sections exist by accessing them
            assert progress_section is not None
            assert metrics_section is not None
            assert seeds_section is not None
        except KeyError as e:
            pytest.fail(f"Layout missing expected sections: {e}")

    def test_error_resilience_with_malformed_inputs(self, console_with_output):
        """Test dashboard handles malformed inputs gracefully."""
        console, _ = console_with_output

        dashboard = RichDashboard(console=console)
        dashboard._setup_layout()

        # Test with empty metrics
        dashboard.update_progress(1, {})
        assert dashboard.metrics["epoch"] == 1
        assert dashboard.metrics["val_loss"] == pytest.approx(0.0)

        # Test with malformed metrics
        malformed_metrics = {"invalid_key": "invalid_value"}
        dashboard.update_progress(2, malformed_metrics)
        assert dashboard.metrics["epoch"] == 2

        # Test with edge case seed parameters
        dashboard.update_seed("", "active", 0.5)  # Empty seed ID
        assert "" in dashboard.seeds

    @pytest.mark.parametrize("epoch_val", [-1, 0, 999999])
    def test_edge_case_epoch_values(self, epoch_val: int, console_with_output):
        """Test dashboard handles edge case epoch values correctly."""
        console, _ = console_with_output
        dashboard = RichDashboard(console=console)
        dashboard._setup_layout()

        # Should not raise errors
        dashboard.update_progress(epoch_val, {})
        assert dashboard.metrics["epoch"] == epoch_val

    def test_rapid_concurrent_updates_handling(self, console_with_output):
        """Test dashboard handles rapid successive updates correctly."""
        console, _ = console_with_output

        with RichDashboard(console=console) as dashboard:
            # Simulate rapid updates that might occur in training
            for i in range(10):
                dashboard.update_seed(f"seed_{i}", "active", 0.1 * i)
                test_metrics = TestMetrics(epoch=i, val_acc=0.1 * i)
                dashboard.update_progress(i, test_metrics.to_dict())

            # Verify final state is consistent
            assert len(dashboard.seeds) == 10
            assert dashboard.metrics["epoch"] == 9
            assert dashboard.metrics["seeds_active"] == 10  # All seeds are active
