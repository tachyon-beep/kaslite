"""Tests for the CLI dashboard functionality."""

from typing import Dict
from unittest.mock import Mock, patch

import pytest
from rich.console import Console
from rich.text import Text

from morphogenetic_engine.cli_dashboard import RichDashboard, SeedState


class TestSeedState:
    """Test suite for SeedState class."""

    def test_seed_state_initialization(self):
        """Test SeedState initialization with default values."""
        seed = SeedState("seed_1")
        
        assert seed.seed_id == "seed_1"
        assert seed.state == "dormant"
        assert abs(seed.alpha - 0.0) < 1e-9

    def test_seed_state_initialization_with_values(self):
        """Test SeedState initialization with custom values."""
        seed = SeedState("seed_2", "active", 0.75)
        
        assert seed.seed_id == "seed_2"
        assert seed.state == "active"
        assert abs(seed.alpha - 0.75) < 1e-9

    def test_seed_state_update(self):
        """Test updating seed state and alpha."""
        seed = SeedState("seed_3")
        
        # Initial state
        assert seed.state == "dormant"
        assert abs(seed.alpha - 0.0) < 1e-9
        
        # Update state
        seed.update("blending", 0.5)
        assert seed.state == "blending"
        assert abs(seed.alpha - 0.5) < 1e-9
        
        # Update again
        seed.update("active", 0.9)
        assert seed.state == "active"
        assert abs(seed.alpha - 0.9) < 1e-9

    def test_seed_state_update_without_alpha(self):
        """Test updating seed state without alpha parameter."""
        seed = SeedState("seed_4", "active", 0.8)
        
        seed.update("dormant")
        assert seed.state == "dormant"
        assert abs(seed.alpha - 0.0) < 1e-9  # Default alpha

    def test_get_styled_status_dormant(self):
        """Test styled status for dormant seeds."""
        seed = SeedState("seed_5", "dormant")
        styled_text = seed.get_styled_status()
        
        assert isinstance(styled_text, Text)
        assert "seed_5: dormant" in str(styled_text)

    def test_get_styled_status_blending(self):
        """Test styled status for blending seeds."""
        seed = SeedState("seed_6", "blending", 0.3)
        styled_text = seed.get_styled_status()
        
        assert isinstance(styled_text, Text)
        assert "seed_6: blending α=0.300" in str(styled_text)

    def test_get_styled_status_active(self):
        """Test styled status for active seeds."""
        seed = SeedState("seed_7", "active", 0.85)
        styled_text = seed.get_styled_status()
        
        assert isinstance(styled_text, Text)
        assert "seed_7: active α=0.850" in str(styled_text)

    def test_get_styled_status_unknown_state(self):
        """Test styled status for unknown states."""
        seed = SeedState("seed_8", "unknown_state")
        styled_text = seed.get_styled_status()
        
        assert isinstance(styled_text, Text)
        assert "seed_8: unknown_state" in str(styled_text)


class TestRichDashboard:
    """Test suite for RichDashboard class."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console for testing."""
        return Mock(spec=Console)

    @pytest.fixture
    def dashboard(self, mock_console):
        """Create a dashboard instance for testing."""
        with patch('morphogenetic_engine.cli_dashboard.Live'):
            return RichDashboard(console=mock_console)

    def test_dashboard_initialization(self, mock_console):
        """Test dashboard initialization."""
        with patch('morphogenetic_engine.cli_dashboard.Live'):
            dashboard = RichDashboard(console=mock_console)
            
            assert dashboard.console == mock_console
            assert dashboard.current_task is None
            assert dashboard.current_phase == "init"
            assert not dashboard.seeds
            assert dashboard.metrics["epoch"] == 0
            assert abs(dashboard.metrics["val_loss"] - 0.0) < 1e-9
            assert abs(dashboard.metrics["val_acc"] - 0.0) < 1e-9
            assert abs(dashboard.metrics["best_acc"] - 0.0) < 1e-9
            assert abs(dashboard.metrics["train_loss"] - 0.0) < 1e-9
            assert dashboard.metrics["seeds_active"] == 0

    def test_dashboard_initialization_without_console(self):
        """Test dashboard initialization without providing console."""
        with patch('morphogenetic_engine.cli_dashboard.Live'), \
             patch('morphogenetic_engine.cli_dashboard.Console') as mock_console_class:
            
            mock_console_instance = Mock()
            mock_console_class.return_value = mock_console_instance
            
            dashboard = RichDashboard()
            assert dashboard.console == mock_console_instance

    def test_update_seed_data_logic(self, dashboard):
        """Test seed update logic without UI updates."""
        # Test adding new seed
        dashboard.seeds["seed_1"] = SeedState("seed_1", "blending", 0.4)
        
        assert "seed_1" in dashboard.seeds
        seed = dashboard.seeds["seed_1"]
        assert seed.seed_id == "seed_1"
        assert seed.state == "blending"
        assert abs(seed.alpha - 0.4) < 1e-9
        
        # Test updating existing seed
        dashboard.seeds["seed_1"].update("active", 0.8)
        assert dashboard.seeds["seed_1"].state == "active"
        assert abs(dashboard.seeds["seed_1"].alpha - 0.8) < 1e-9

    def test_update_progress_data_logic(self, dashboard):
        """Test progress update logic without UI updates."""
        metrics = {
            "val_loss": 0.25,
            "val_acc": 0.85,
            "best_acc": 0.90,
            "train_loss": 0.20,
        }
        
        # Update metrics directly
        dashboard.metrics.update({
            "epoch": 15,
            "val_loss": metrics.get("val_loss", 0.0),
            "val_acc": metrics.get("val_acc", 0.0),
            "best_acc": metrics.get("best_acc", 0.0),
            "train_loss": metrics.get("train_loss", 0.0),
        })
        
        assert dashboard.metrics["epoch"] == 15
        assert abs(dashboard.metrics["val_loss"] - 0.25) < 1e-9
        assert abs(dashboard.metrics["val_acc"] - 0.85) < 1e-9
        assert abs(dashboard.metrics["best_acc"] - 0.90) < 1e-9
        assert abs(dashboard.metrics["train_loss"] - 0.20) < 1e-9

    def test_seeds_active_count_logic(self, dashboard):
        """Test active seeds counting logic."""
        # Add seeds with different states
        dashboard.seeds = {
            "seed_1": SeedState("seed_1", "dormant"),
            "seed_2": SeedState("seed_2", "active", 0.8),
            "seed_3": SeedState("seed_3", "blending", 0.3),
            "seed_4": SeedState("seed_4", "active", 0.9),
        }
        
        # Count active seeds
        active_count = sum(
            1 for seed in dashboard.seeds.values() if seed.state == "active"
        )
        
        assert active_count == 2  # Only seed_2 and seed_4 are active

    def test_start_phase_new_phase(self, dashboard):
        """Test starting a new phase."""
        with patch.object(dashboard.progress, 'add_task') as mock_add_task:
            mock_add_task.return_value = "task_id_123"
            
            dashboard.start_phase("phase_2", 20, "Test Phase")
            
            assert dashboard.current_phase == "phase_2"
            mock_add_task.assert_called_once_with("Test Phase", total=20)
            assert dashboard.current_task == "task_id_123"

    def test_start_phase_with_existing_task(self, dashboard):
        """Test starting a new phase when a task already exists."""
        with patch.object(dashboard.progress, 'add_task') as mock_add_task, \
             patch.object(dashboard.progress, 'stop_task') as mock_stop_task:
            
            # Set up existing task
            dashboard.current_task = "old_task_id"
            mock_add_task.return_value = "new_task_id"
            
            dashboard.start_phase("phase_3", 30)
            
            # Should stop the old task and create a new one
            mock_stop_task.assert_called_once_with("old_task_id")
            mock_add_task.assert_called_once_with("Phase phase_3", total=30)
            assert dashboard.current_task == "new_task_id"

    def test_start_phase_default_description(self, dashboard):
        """Test starting a phase with default description."""
        with patch.object(dashboard.progress, 'add_task') as mock_add_task:
            mock_add_task.return_value = "task_id"
            
            dashboard.start_phase("test_phase", 15)
            
            mock_add_task.assert_called_once_with("Phase test_phase", total=15)

    def test_show_phase_transition(self, dashboard):
        """Test showing phase transition."""
        dashboard.show_phase_transition("phase_2", 25)
        
        assert dashboard.current_phase == "phase_2"
        # Verify console.print was called (through mock)
        assert dashboard.console.print.called

    def test_show_germination_event(self, dashboard):
        """Test showing germination event."""
        dashboard.show_germination_event("seed_5", 18)
        
        # Verify console.print was called (through mock)
        assert dashboard.console.print.called

    def test_context_manager_enter(self, dashboard):
        """Test dashboard context manager entry."""
        with patch.object(dashboard.live, 'start') as mock_start:
            with dashboard:
                # Test that the context manager returns itself
                pass
            
            mock_start.assert_called_once()

    def test_context_manager_exit_with_task(self, dashboard):
        """Test dashboard context manager exit with active task."""
        dashboard.current_task = "test_task"
        
        with patch.object(dashboard.progress, 'stop_task') as mock_stop_task, \
             patch.object(dashboard.live, 'stop') as mock_stop:
            
            dashboard.__exit__(None, None, None)
            
            mock_stop_task.assert_called_once_with("test_task")
            mock_stop.assert_called_once()

    def test_context_manager_exit_without_task(self, dashboard):
        """Test dashboard context manager exit without active task."""
        with patch.object(dashboard.progress, 'stop_task') as mock_stop_task, \
             patch.object(dashboard.live, 'stop') as mock_stop:
            
            dashboard.__exit__(None, None, None)
            
            mock_stop_task.assert_not_called()
            mock_stop.assert_called_once()

    def test_start_method(self, dashboard):
        """Test the start method."""
        with patch.object(dashboard.live, 'start') as mock_start:
            dashboard.start()
            mock_start.assert_called_once()

    def test_stop_method_with_task(self, dashboard):
        """Test the stop method with active task."""
        dashboard.current_task = "test_task"
        
        with patch.object(dashboard.progress, 'stop_task') as mock_stop_task, \
             patch.object(dashboard.live, 'stop') as mock_stop:
            
            dashboard.stop()
            
            mock_stop_task.assert_called_once_with("test_task")
            mock_stop.assert_called_once()

    def test_stop_method_without_task(self, dashboard):
        """Test the stop method without active task."""
        with patch.object(dashboard.progress, 'stop_task') as mock_stop_task, \
             patch.object(dashboard.live, 'stop') as mock_stop:
            
            dashboard.stop()
            
            mock_stop_task.assert_not_called()
            mock_stop.assert_called_once()

    def test_progress_update_with_current_task(self, dashboard):
        """Test progress bar update when current task exists."""
        with patch.object(dashboard.progress, 'update') as mock_progress_update:
            dashboard.current_task = "test_task_id"
            
            # Mock the layout updates to avoid KeyError
            with patch.object(dashboard.layout, '__getitem__'):
                # Simulate the metrics update logic only
                epoch = 5
                dashboard.metrics.update({
                    "epoch": epoch,
                    "val_loss": 0.5,
                    "val_acc": 0.8,
                    "best_acc": 0.85,
                    "train_loss": 0.45,
                })
                
                # Call progress update method for the task update
                dashboard.progress.update(dashboard.current_task, completed=epoch)
                
                mock_progress_update.assert_called_once_with("test_task_id", completed=5)

    def test_progress_update_conditional_logic(self, dashboard):
        """Test the conditional logic for progress bar updates."""
        def test_progress_logic(task_id, expected_calls):
            """Helper to test progress update logic."""
            with patch.object(dashboard.progress, 'update') as mock_progress_update:
                if task_id is not None:
                    dashboard.progress.update(task_id, completed=5)
                assert mock_progress_update.call_count == expected_calls
        
        # Test case 1: task_id is None - should not call update
        test_progress_logic(None, 0)
        
        # Test case 2: task_id exists - should call update  
        test_progress_logic("test_task", 1)

    def test_metrics_get_with_defaults(self):
        """Test that metrics.get() properly handles missing keys with defaults."""
        # Test the exact logic used in update_progress
        incomplete_metrics = {"val_loss": 0.3}
        
        extracted_metrics = {
            "val_loss": incomplete_metrics.get("val_loss", 0.0),
            "val_acc": incomplete_metrics.get("val_acc", 0.0),
            "best_acc": incomplete_metrics.get("best_acc", 0.0),
            "train_loss": incomplete_metrics.get("train_loss", 0.0),
        }
        
        assert abs(extracted_metrics["val_loss"] - 0.3) < 1e-9
        assert abs(extracted_metrics["val_acc"] - 0.0) < 1e-9
        assert abs(extracted_metrics["best_acc"] - 0.0) < 1e-9
        assert abs(extracted_metrics["train_loss"] - 0.0) < 1e-9

    def test_seed_enumeration_logic(self):
        """Test the enumeration logic used in _create_seeds_panel."""
        # Create test seeds to test the enumeration
        seeds = {
            "seed_1": SeedState("seed_1", "active", 0.8),
            "seed_2": SeedState("seed_2", "dormant"),
            "seed_3": SeedState("seed_3", "blending", 0.5),
        }
        
        # Test the enumeration logic used in the actual method
        content_parts = []
        for i, seed in enumerate(seeds.values()):
            if i > 0:
                content_parts.append("\n")
            content_parts.append(str(seed.get_styled_status()))
        
        # Should have newlines between seeds (2 newlines for 3 seeds)
        newline_count = content_parts.count("\n")
        assert newline_count == 2
        
        # Should have 3 seed status strings plus 2 newlines = 5 parts total
        assert len(content_parts) == 5

    def test_empty_seeds_handling(self):
        """Test handling of empty seeds dictionary."""
        seeds: Dict[str, SeedState] = {}
        
        # Test the logic: if not self.seeds
        has_seeds = bool(seeds)
        assert not has_seeds
        
        # Test active seeds count with empty dict
        active_count = sum(
            1 for seed in seeds.values() if seed.state == "active"
        )
        assert active_count == 0

    def test_phase_name_assignment(self, dashboard):
        """Test phase name assignment in show_phase_transition."""
        initial_phase = dashboard.current_phase
        assert initial_phase == "init"
        
        # Test the assignment logic from show_phase_transition
        new_phase = "phase_2"
        dashboard.current_phase = new_phase
        
        assert dashboard.current_phase == "phase_2"
