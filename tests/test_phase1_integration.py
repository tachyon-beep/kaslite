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

from morphogenetic_engine.components import BaseNet, SentinelSeed
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
def test_system(experiment_logger, blend_config):
    """Create a complete test system with BaseNet, SeedManager, and KasminaMicro."""
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
        
        # Get the seed instance
        seed = model.get_seeds_for_layer(0)[0]
        assert seed.seed_id == seed_id
        assert seed.state == SeedState.DORMANT.value
        
        # === PHASE 1: DORMANT → GERMINATED ===
        # Fill the seed's buffer with some data to make it germinable
        dummy_data = torch.randn(50, 16)  # 50 samples of 16-dim data
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
        # Simulate training completion by manually triggering convergence
        # Fill loss history with converged values
        converged_loss = 0.01
        seed.loss_history = [converged_loss] * 25  # Enough for convergence window
        
        # Trigger transition check
        seed.assess_and_transition_state(epoch=10)
        assert seed.state == SeedState.BLENDING.value
        assert abs(seed.alpha) < 1e-6  # Alpha should be reset for blending
        
        # === PHASE 4: BLENDING WITH STRATEGY SELECTION ===
        # Verify that no strategy has been selected yet
        assert "blend_strategy" not in seed_info
        
        # Set up telemetry to trigger specific strategy selection
        seed_info["telemetry"] = {"drift": 0.08}  # High drift for DRIFT_CONTROLLED
        seed_info["current_loss"] = 0.02
        seed_info["baseline_loss"] = 0.05
        seed_info["avg_grad_norm"] = 0.5
        
        # Run Tamiyo assessment to trigger strategy selection
        tamiyo.assess_and_update_seeds(epoch=11)
        
        # Verify strategy was selected
        assert "blend_strategy" in seed_info
        strategy_name = seed_info["blend_strategy"]
        assert strategy_name in ["DRIFT_CONTROLLED", "PERFORMANCE_LINKED", "GRAD_NORM_GATED", "FIXED_RAMP"]
        
        # Verify strategy object is cached
        initial_alpha = seed.alpha
        seed.update_blending(epoch=11)
        first_strategy_obj = seed_info.get("blend_strategy_obj")
        assert first_strategy_obj is not None
        
        # Update blending again and verify same strategy object is reused
        seed.update_blending(epoch=12)
        second_strategy_obj = seed_info.get("blend_strategy_obj")
        assert first_strategy_obj is second_strategy_obj, "Strategy object should be cached"
        
        # Verify alpha is progressing
        assert seed.alpha > initial_alpha, "Alpha should increase during blending"
        
        # === PHASE 5: BLENDING COMPLETION ===
        # Manually set alpha close to completion to trigger transition
        seed.alpha = 0.99
        seed_info["alpha"] = 0.99
        
        # Trigger transition
        seed.assess_and_transition_state(epoch=15)
        assert seed.state == SeedState.SHADOWING.value
        
        # === PHASE 6: SHADOWING → PROBATIONARY ===
        # Simulate stable shadowing by setting up stability history
        seed_info["stability_history"] = [0.01] * 15  # Stable low losses
        
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
        # Verify fossilized seed is live in forward pass
        test_input = torch.randn(5, 16)
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
                "current_loss": 0.4,
                "baseline_loss": 0.5,
                "grad_norm": 0.5,  # Within bounds [0.1, 1.0]
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
            seed_id = (0, 0)  # Use same seed, reset for each test
            seed = model.get_seeds_for_layer(0)[0]
            
            # Reset seed state and prepare for blending
            seed._set_state(SeedState.BLENDING, epoch=i*10)
            seed_info = seed_manager.seeds[seed_id]
            
            # Clear any previous strategy selection
            if "blend_strategy" in seed_info:
                del seed_info["blend_strategy"]
            
            # Set up the test conditions
            seed_info["telemetry"] = case["telemetry"]
            if "health_signal" in case:
                # Mock the health signal method with proper closure
                health_val = case["health_signal"]
                seed.get_health_signal = lambda h=health_val: h
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
    
    def test_event_logging_during_lifecycle(self, test_system, temp_log_dir):
        """Test that proper events are logged during the blending lifecycle."""
        model, seed_manager, tamiyo = test_system
        seed = model.get_seeds_for_layer(0)[0]
        seed_id = seed.seed_id
        
        # Prepare seed for blending
        dummy_data = torch.randn(30, 16)
        for i in range(30):
            seed_manager.append_to_buffer(seed_id, dummy_data[i:i+1])
        
        seed.initialize_child(epoch=1)
        seed._set_state(SeedState.BLENDING, epoch=5)
        
        # Set up telemetry for strategy selection
        seed_info = seed_manager.seeds[seed_id]
        seed_info["telemetry"] = {"drift": 0.01}
        seed_info["current_loss"] = 0.02
        seed_info["baseline_loss"] = 0.05
        
        # Trigger strategy selection (should log BLEND_STRATEGY_CHOSEN)
        initial_log_size = len(seed_manager.logger.events)
        tamiyo.assess_and_update_seeds(epoch=6)
        
        # Verify strategy selection was logged
        new_events = seed_manager.logger.events[initial_log_size:]
        strategy_events = [e for e in new_events if e.event_type == "BLEND_STRATEGY_CHOSEN"]
        assert len(strategy_events) >= 1, "BLEND_STRATEGY_CHOSEN event should be logged"
        
        # Complete blending and trigger BLEND_COMPLETED event
        seed.alpha = 0.99
        seed_info["alpha"] = 0.99
        
        pre_completion_log_size = len(seed_manager.logger.events)
        seed.assess_and_transition_state(epoch=10)
        
        # Verify blend completion was logged
        completion_events = seed_manager.logger.events[pre_completion_log_size:]
        blend_completed_events = [e for e in completion_events if e.event_type == "BLEND_COMPLETED"]
        assert len(blend_completed_events) >= 1, "BLEND_COMPLETED event should be logged"
        
        # Verify log files were created
        log_files = list(temp_log_dir.glob("*.log"))
        assert len(log_files) > 0, "Log files should be created"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
