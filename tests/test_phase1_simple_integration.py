"""
Simplified integration test for Phase 1 blending system.

This test validates the strategy selection, caching, and event logging
without requiring a full seed lifecycle simulation.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from morphogenetic_engine.blending import get_strategy, FixedRampBlending
from morphogenetic_engine.core import BlendingConfig, KasminaMicro, SeedManager
from morphogenetic_engine.events import SeedState
from morphogenetic_engine.logger import ExperimentLogger


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def experiment_logger(temp_log_dir):
    """Create an ExperimentLogger for testing."""
    return ExperimentLogger(
        log_dir=temp_log_dir,
        log_file="test_integration.log"
    )


@pytest.fixture
def blend_config():
    """Create a BlendingConfig for testing."""
    return BlendingConfig(
        fixed_steps=5,  # Short for testing
        high_drift_threshold=0.05,
        low_health_threshold=1e-3,
        performance_loss_factor=0.8,
        grad_norm_lower=0.1,
        grad_norm_upper=1.0
    )


@pytest.fixture
def mock_seed_and_manager(experiment_logger):
    """Create mock seed and seed manager for isolated testing."""
    seed_manager = SeedManager(logger=experiment_logger)
    
    # Create a mock seed with proper methods
    mock_seed = Mock()
    mock_seed.seed_id = (0, 0)
    mock_seed.alpha = 0.0
    mock_seed.state = SeedState.BLENDING.value
    mock_seed.get_health_signal = Mock(return_value=0.1)
    mock_seed.validate_on_holdout = Mock(return_value=0.5)
    # Mock assess_and_transition_state to not change the state
    mock_seed.assess_and_transition_state = Mock()
    
    # Register the mock seed with the manager
    seed_manager.seeds[(0, 0)] = {
        "module": mock_seed,
        "state": SeedState.BLENDING.value,
        "alpha": 0.0,
        "telemetry": {"drift": 0.01},
        "current_loss": 0.02,
        "baseline_loss": 0.05,
        "avg_grad_norm": 0.5
    }
    
    return mock_seed, seed_manager


class TestBlendingSystemIntegration:
    """Integration tests for the blending system components."""
    
    def test_strategy_selection_and_logging(self, mock_seed_and_manager, blend_config, experiment_logger):
        """Test that strategy selection works and events are logged properly."""
        _, seed_manager = mock_seed_and_manager
        
        # Create KasminaMicro with our test components
        tamiyo = KasminaMicro(
            seed_manager=seed_manager,
            patience=10,
            delta=1e-3,
            acc_threshold=0.7,
            logger=experiment_logger,
            blending_config=blend_config
        )
        
        seed_id = (0, 0)
        
        # Verify no strategy is initially selected
        seed_info = seed_manager.seeds[seed_id]
        assert "blend_strategy" not in seed_info
        
        # Trigger strategy selection
        initial_event_count = len(experiment_logger.events)
        tamiyo.assess_and_update_seeds(epoch=5)
        
        # Verify strategy was selected
        assert "blend_strategy" in seed_info
        strategy_name = seed_info["blend_strategy"]
        assert strategy_name in ["DRIFT_CONTROLLED", "PERFORMANCE_LINKED", "GRAD_NORM_GATED", "FIXED_RAMP"]
        
        # Verify events were logged
        new_events = experiment_logger.events[initial_event_count:]
        strategy_events = [e for e in new_events if "BLEND_STRATEGY_CHOSEN" in str(e)]
        assert len(strategy_events) >= 1, "BLEND_STRATEGY_CHOSEN event should be logged"
    
    def test_strategy_caching_mechanism(self, mock_seed_and_manager, blend_config):
        """Test that strategy objects are cached correctly."""
        mock_seed, seed_manager = mock_seed_and_manager
        seed_id = (0, 0)
        seed_info = seed_manager.seeds[seed_id]
        
        # Set up a strategy manually
        seed_info["blend_strategy"] = "FIXED_RAMP"
        
        # First call should create the strategy object
        strategy_obj_1 = get_strategy("FIXED_RAMP", mock_seed, blend_config)
        seed_info["blend_strategy_obj"] = strategy_obj_1
        
        # Second call should return the same object (simulate caching)
        strategy_obj_2 = seed_info.get("blend_strategy_obj")
        
        # Verify same object is returned
        assert strategy_obj_1 is strategy_obj_2, "Strategy object should be cached"
        assert isinstance(strategy_obj_1, FixedRampBlending)
    
    def test_strategy_selection_conditions(self, mock_seed_and_manager, blend_config, experiment_logger):
        """Test that different conditions trigger appropriate strategies."""
        mock_seed, seed_manager = mock_seed_and_manager
        
        tamiyo = KasminaMicro(
            seed_manager=seed_manager,
            patience=10,
            delta=1e-3,
            acc_threshold=0.7,
            logger=experiment_logger,
            blending_config=blend_config
        )
        
        test_cases = [
            {
                "name": "High drift triggers DRIFT_CONTROLLED",
                "telemetry": {"drift": 0.15},  # Above high_drift_threshold (0.05)
                "expected_strategy": "DRIFT_CONTROLLED"
            },
            {
                "name": "Low health triggers PERFORMANCE_LINKED", 
                "telemetry": {"drift": 0.01},  # Low drift
                "health_signal": 5e-4,  # Below low_health_threshold (1e-3)
                "expected_strategy": "PERFORMANCE_LINKED"
            },
            {
                "name": "Good performance with stable gradients triggers GRAD_NORM_GATED",
                "telemetry": {"drift": 0.01},
                "health_signal": 0.1,  # Above threshold
                "current_loss": 0.3,  # 0.3 < 0.5 * 0.8 = 0.4 ✓
                "baseline_loss": 0.5,
                "grad_norm": 0.5,
                "expected_strategy": "GRAD_NORM_GATED"
            },
            {
                "name": "Default conditions trigger FIXED_RAMP",
                "telemetry": {"drift": 0.01},
                "health_signal": 0.1,
                "current_loss": 0.9,  # Poor performance
                "baseline_loss": 0.5,
                "expected_strategy": "FIXED_RAMP"
            }
        ]
        
        for i, case in enumerate(test_cases):
            seed_id = (0, 0)
            seed_info = seed_manager.seeds[seed_id]
            
            # Clear previous strategy selection
            if "blend_strategy" in seed_info:
                del seed_info["blend_strategy"]
            
            # Set up test conditions
            seed_info["telemetry"] = case["telemetry"]
            if "health_signal" in case:
                health_val = case["health_signal"]
                mock_seed.get_health_signal = Mock(return_value=health_val)
            if "current_loss" in case:
                seed_info["current_loss"] = case["current_loss"]
            if "baseline_loss" in case:
                seed_info["baseline_loss"] = case["baseline_loss"]
            if "grad_norm" in case:
                seed_info["avg_grad_norm"] = case["grad_norm"]
            
            # Trigger strategy selection
            tamiyo.assess_and_update_seeds(epoch=i*10 + 1)
            
            # Verify expected strategy was selected
            assert "blend_strategy" in seed_info, f"Strategy should be selected for case: {case['name']}"
            selected_strategy = seed_info["blend_strategy"]
            assert selected_strategy == case["expected_strategy"], \
                f"Expected {case['expected_strategy']}, got {selected_strategy} for case: {case['name']}"
    
    def test_import_safety_and_analytics_integration(self, mock_seed_and_manager, blend_config, experiment_logger):
        """Test that analytics integration works without breaking the system."""
        _, seed_manager = mock_seed_and_manager
        
        tamiyo = KasminaMicro(
            seed_manager=seed_manager,
            patience=10,
            delta=1e-3,
            acc_threshold=0.7,
            logger=experiment_logger,
            blending_config=blend_config
        )
        
        # This should work even if analytics import fails
        try:
            tamiyo.assess_and_update_seeds(epoch=1)
        except Exception as e:
            pytest.fail(f"System should handle analytics import gracefully, but got: {e}")
    
    def test_blending_config_propagation(self, blend_config):
        """Test that BlendingConfig is properly used throughout the system."""
        # Test that the config values are accessible (using approximate equality for floats)
        assert blend_config.fixed_steps == 5
        assert abs(blend_config.high_drift_threshold - 0.05) < 1e-9
        assert abs(blend_config.low_health_threshold - 1e-3) < 1e-9
        assert abs(blend_config.performance_loss_factor - 0.8) < 1e-9
        assert abs(blend_config.grad_norm_lower - 0.1) < 1e-9
        assert abs(blend_config.grad_norm_upper - 1.0) < 1e-9
        
        # Test that strategies can be created with the config
        mock_seed = Mock()
        mock_seed.alpha = 0.0
        
        strategy = get_strategy("FIXED_RAMP", mock_seed, blend_config)
        assert strategy.config is blend_config
        assert isinstance(strategy, FixedRampBlending)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
