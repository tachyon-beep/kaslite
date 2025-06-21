"""
Comprehensive integration test for Phase 1 blending system.

This test validates the complete seed lifecycle with dynamic strategy selection,
ensuring that seeds properly transition through all states with correct
strategy application and metrics collection.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from morphogenetic_engine.components import BaseNet, SentinelSeed
from morphogenetic_engine.core import BlendingConfig, KasminaMicro, SeedManager
from morphogenetic_engine.events import SeedState, EventType
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


@pytest.fixture(autouse=True)
def setup_deterministic_environment():
    """Set up deterministic environment for consistent test results."""
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield
    # Cleanup is automatic with autouse


@pytest.fixture
def test_system(experiment_logger, blend_config):
    """Create a fresh test system for each test method."""
    # Fresh seed manager and system for each test to avoid state bleed
    seed_manager = SeedManager(logger=experiment_logger)
    
    model = BaseNet(
        hidden_dim=16,
        seed_manager=seed_manager,
        input_dim=4,
        output_dim=2,
        num_layers=2,
        seeds_per_layer=1,
        blend_cfg=blend_config
    )
    
    tamiyo = KasminaMicro(
        seed_manager=seed_manager,
        patience=10,
        delta=1e-3,
        acc_threshold=0.7,
        logger=experiment_logger,
        blending_config=blend_config
    )
    
    return model, seed_manager, tamiyo


class TestCompleteBlendingLifecycle:
    """Test the complete seed lifecycle with blending strategies."""
    
    def _force_training_completion(self, seed, converged_loss=0.01):
        """Helper to force training completion via early stopping."""
        seed_info = seed.seed_manager.seeds[seed.seed_id]
        
        # Mock validate_on_holdout to return converged loss
        seed.validate_on_holdout = MagicMock(return_value=converged_loss)
        
        # Set up early stopping trigger (cleaner than manipulating loss_history)
        seed_info["best_val_loss"] = converged_loss
        seed_info["val_patience_counter"] = 26  # Above default patience limit of 25
        seed_info["val_patience_limit"] = 25
        
        # Also set convergence as backup
        seed.loss_history = [converged_loss] * 25
    
    def _setup_seed_for_blending(self, seed, strategy_conditions=None):
        """Helper to set up a seed in BLENDING state with optional strategy conditions."""
        seed_info = seed.seed_manager.seeds[seed.seed_id]
        
        # Reset any previous strategy
        seed_info.pop("blend_strategy", None)
        seed_info.pop("blend_strategy_obj", None)
        
        # Set state to BLENDING
        seed._set_state(SeedState.BLENDING, epoch=1)
        
        # Verify state is properly set
        assert seed.state == SeedState.BLENDING.value
        assert seed_info["state"] == SeedState.BLENDING.value
        
        # Set up strategy selection conditions if provided
        if strategy_conditions:
            for key, value in strategy_conditions.items():
                if key == "health_signal":
                    seed.get_health_signal = MagicMock(return_value=value)
                else:
                    seed_info[key] = value
    
    def test_full_seed_lifecycle_with_strategy_selection(self, test_system):
        """
        Test that a seed can progress through the complete lifecycle:
        DORMANT → GERMINATED → TRAINING → BLENDING → SHADOWING → PROBATIONARY → FOSSILIZED
        
        Verifies:
        1. Strategy is chosen once based on telemetry
        2. Alpha ramps correctly according to strategy
        3. Proper state transitions occur
        4. Events are logged with correct payloads
        """
        model, seed_manager, tamiyo = test_system
        seed_id = (0, 0)  # First seed in first layer
        
        # Get the seed instance and ensure proper device setup
        seed = model.get_seeds_for_layer(0)[0]
        device = next(seed.child.parameters()).device
        assert seed.seed_id == seed_id
        assert seed.state == SeedState.DORMANT.value
        
        # === PHASE 1: DORMANT → GERMINATED ===
        # Fill the seed's buffer with some data to make it germinable
        dummy_data = torch.randn(50, 16, device=device, dtype=torch.float32)
        for i in range(50):
            seed_manager.append_to_buffer(seed_id, dummy_data[i:i+1])
        
        # Trigger germination
        seed.initialize_child(epoch=1)
        assert seed.state == SeedState.GERMINATED.value
        
        # === PHASE 2: GERMINATED → TRAINING ===
        # Manually transition to training (normally done by KasminaMicro)
        seed._set_state(SeedState.TRAINING, epoch=2)
        assert seed.state == SeedState.TRAINING.value
        
        # Verify training infrastructure is set up
        seed_info = seed_manager.seeds[seed_id]
        assert seed_info["training_steps"] == 0
        assert seed_info["baseline_loss"] is None
        assert seed.child_optim is not None
        
        # === PHASE 3: TRAINING → BLENDING ===
        # Use helper to force training completion
        self._force_training_completion(seed)
        
        # Trigger transition check
        seed.assess_and_transition_state(epoch=10)
        assert seed.state == SeedState.BLENDING.value
        assert abs(seed.alpha) < 1e-6  # Alpha should be reset for blending
        
        # === PHASE 4: BLENDING WITH STRATEGY SELECTION ===
        # Set up telemetry to trigger a *progressing* strategy (GRAD_NORM_GATED)
        seed_info["telemetry"] = {"drift": 0.01}   # low drift
        seed_info["current_loss"] = 0.02
        seed_info["baseline_loss"] = 0.05
        seed_info["avg_grad_norm"] = 0.5               # in [0.1,1.0]
        
        # Verify no strategy selected yet
        assert "blend_strategy" not in seed_info
        
        # Run Tamiyo assessment to trigger strategy selection
        tamiyo.assess_and_update_seeds(epoch=11)
        
        # Verify strategy was selected
        assert "blend_strategy" in seed_info
        strategy_name = seed_info["blend_strategy"]
        assert strategy_name in ["GRAD_NORM_GATED", "FIXED_RAMP", "PERFORMANCE_LINKED"], \
               f"Expected a progressing strategy, got {strategy_name}"
        
        # Verify alpha progressed during strategy selection (assess_and_update_seeds calls update_blending)
        initial_alpha = seed.alpha
        assert initial_alpha > 0.0, "Alpha should have increased during strategy selection"
        
        # Update blending again to verify strategy object caching
        seed.update_blending(epoch=12)
        first_strategy_obj = seed_info.get("blend_strategy_obj")
        assert first_strategy_obj is not None
        
        # Update blending again with different epoch and verify same strategy object is reused
        seed.update_blending(epoch=13)
        second_strategy_obj = seed_info.get("blend_strategy_obj")
        assert first_strategy_obj is second_strategy_obj, "Strategy object should be cached"
        
        # Now α *must* go up
        assert seed.alpha > initial_alpha, f"Alpha should increase under {strategy_name}"
        
        # === PHASE 5: BLENDING COMPLETION ===
        # Manually set alpha close to completion to trigger transition
        seed.alpha = 0.99
        seed_info["alpha"] = 0.99
        
        # Trigger transition
        seed.assess_and_transition_state(epoch=15)
        assert seed.state == SeedState.SHADOWING.value
        
        # === PHASE 6: SHADOWING → PROBATIONARY ===
        # Mock evaluate_loss to return stable values for shadowing
        stable_loss = 0.01
        seed.evaluate_loss = MagicMock(return_value=stable_loss)
        
        # Simulate stable shadowing by setting up stability history and buffer
        seed_info["stability_history"] = [stable_loss] * 15  # Stable low losses
        
        # Trigger transition
        seed.assess_and_transition_state(epoch=20)
        assert seed.state == SeedState.PROBATIONARY.value
        
        # === PHASE 7: PROBATIONARY → FOSSILIZED ===
        # Set up probationary completion
        seed_info["probationary_steps"] = seed.probationary_steps
        
        # Ensure fossilization by setting good performance
        seed_info["baseline_loss"] = 1.0  # High baseline
        # validate_on_holdout will return current_loss which is 0.02 (good improvement)
        
        # Trigger final evaluation
        seed.assess_and_transition_state(epoch=25)
        assert seed.state == SeedState.FOSSILIZED.value
        
        # === VERIFICATION OF FOSSILIZED STATE ===
        # Verify fossilized seed is live in forward pass with proper device handling
        test_input = torch.randn(5, seed.dim, device=device, dtype=torch.float32)
        with torch.no_grad():
            output = seed(test_input)
            # In fossilized state, output should be x + child_out (residual connection)
            child_output = seed.child(test_input)
            expected_output = test_input + child_output
            assert torch.allclose(output, expected_output, atol=1e-6)
        
        # Verify parameters are still trainable (not frozen)
        for param in seed.child.parameters():
            assert param.requires_grad, "Fossilized seed parameters should remain trainable"
    
    def test_strategy_selection_conditions(self, test_system):
        """Test that different telemetry conditions trigger appropriate strategies."""
        model, seed_manager, tamiyo = test_system
        
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
                "current_loss": 0.3,  # Good performance: 0.3 < 0.5 * 0.8 = 0.4
                "baseline_loss": 0.5,
                "avg_grad_norm": 0.5,  # Within bounds [0.1, 1.0]
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
            # Get a fresh seed instance to avoid state bleed
            seed = model.get_seeds_for_layer(0)[0]
            seed_id = seed.seed_id
            
            # Set up seed in BLENDING state with test conditions
            self._setup_seed_for_blending(seed, case)
            
            # Verify preconditions
            seed_info = seed_manager.seeds[seed_id]
            assert seed.state == SeedState.BLENDING.value
            assert seed_info["state"] == SeedState.BLENDING.value
            assert "blend_strategy" not in seed_info
            
            # Trigger strategy selection
            tamiyo.assess_and_update_seeds(epoch=i*10 + 1)
            
            # Verify expected strategy was selected
            assert "blend_strategy" in seed_info, f"Strategy should be selected for case: {case['name']}"
            selected_strategy = seed_info["blend_strategy"]
            assert selected_strategy == case["expected_strategy"], \
                f"Expected {case['expected_strategy']}, got {selected_strategy} for case: {case['name']}"
    
    def test_event_logging_during_lifecycle(self, test_system, temp_log_dir):
        """Test that proper events are logged during the blending lifecycle."""
        model, seed_manager, tamiyo = test_system
        seed = model.get_seeds_for_layer(0)[0]
        seed_id = seed.seed_id
        device = next(seed.child.parameters()).device
        
        # Prepare seed for blending
        dummy_data = torch.randn(30, 16, device=device, dtype=torch.float32)
        for i in range(30):
            seed_manager.append_to_buffer(seed_id, dummy_data[i:i+1])
        
        seed.initialize_child(epoch=1)
        
        # Set up seed in BLENDING state with proper conditions
        strategy_conditions = {
            "telemetry": {"drift": 0.01},
            "current_loss": 0.02,
            "baseline_loss": 0.05,
            "avg_grad_norm": 0.5
        }
        self._setup_seed_for_blending(seed, strategy_conditions)
        
        # Trigger strategy selection (should log BLEND_STRATEGY_CHOSEN)
        initial_event_count = len(seed_manager.logger.events)
        tamiyo.assess_and_update_seeds(epoch=6)
        
        # Verify strategy selection event was logged
        new_events = seed_manager.logger.events[initial_event_count:]
        strategy_events = [e for e in new_events if e.event_type == EventType.BLEND_STRATEGY_CHOSEN]
        assert len(strategy_events) >= 1, "BLEND_STRATEGY_CHOSEN event should be logged"
        
        # Validate strategy event payload
        strategy_event = strategy_events[0]
        strategy_payload = strategy_event.payload
        assert "seed_id" in strategy_payload
        assert "strategy_name" in strategy_payload
        assert "telemetry" in strategy_payload
        assert abs(strategy_payload["telemetry"]["drift"] - 0.01) < 1e-6
        
        # Verify strategy was actually selected
        seed_info = seed_manager.seeds[seed_id]
        assert "blend_strategy" in seed_info
        assert strategy_payload["strategy_name"] == seed_info["blend_strategy"]
        
        # Complete blending and trigger BLEND_COMPLETED event
        seed.alpha = 0.99
        seed_info["alpha"] = 0.99
        
        # Set up blend start metrics for proper completion event
        seed_info["blend_start_epoch"] = 1
        seed_info["blend_initial_loss"] = 0.05
        seed_info["blend_initial_drift"] = 0.01
        
        pre_completion_event_count = len(seed_manager.logger.events)
        seed.assess_and_transition_state(epoch=10)
        
        # Verify blend completion was logged
        completion_events = seed_manager.logger.events[pre_completion_event_count:]
        blend_completed_events = [e for e in completion_events if e.event_type == EventType.BLEND_COMPLETED]
        assert len(blend_completed_events) >= 1, "BLEND_COMPLETED event should be logged"
        
        # Validate completion event payload
        completion_event = blend_completed_events[0]
        completion_payload = completion_event.payload
        assert "seed_id" in completion_payload
        assert "strategy_name" in completion_payload
        assert "duration_epochs" in completion_payload
        assert "initial_loss" in completion_payload
        assert "final_loss" in completion_payload
        assert completion_payload["strategy_name"] == seed_info["blend_strategy"]
        
        # Verify log files were created
        log_files = list(temp_log_dir.glob("*.log"))
        assert len(log_files) > 0, "Log files should be created"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
