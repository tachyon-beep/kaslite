"""
Unit tests for blending strategies in the morphogenetic engine.

This module tests the new strategy pattern for blending phase management,
ensuring each strategy correctly calculates alpha values and integrates
with the overall system.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock

from morphogenetic_engine.blending import (
    BlendingStrategy,
    FixedRampBlending,
    PerformanceLinkedBlending,
    DriftControlledBlending,
    GradNormGatedBlending,
    get_strategy
)
from morphogenetic_engine.core import BlendingConfig


@pytest.fixture
def mock_seed():
    """Create a mock SentinelSeed for testing strategies."""
    seed = Mock()
    seed.alpha = 0.0
    seed.seed_info = {}
    seed.validate_on_holdout = Mock(return_value=0.5)
    return seed


@pytest.fixture
def blend_config():
    """Create a BlendingConfig for testing."""
    return BlendingConfig(
        fixed_steps=10,
        high_drift_threshold=0.12,
        grad_norm_lower=0.1,
        grad_norm_upper=1.0
    )


class TestFixedRampBlending:
    """Test suite for FixedRampBlending strategy."""

    def test_fixed_ramp_progression(self, mock_seed, blend_config):
        """Test that FixedRampBlending increments alpha linearly."""
        strategy = FixedRampBlending(mock_seed, blend_config)
        
        # First update
        new_alpha = strategy.update()
        expected_alpha = 1.0 / blend_config.fixed_steps
        assert new_alpha == expected_alpha
        
        # Verify it doesn't modify the seed's alpha directly
        assert mock_seed.alpha == 0.0

    def test_fixed_ramp_completion(self, mock_seed, blend_config):
        """Test that FixedRampBlending caps at 1.0."""
        strategy = FixedRampBlending(mock_seed, blend_config)
        mock_seed.alpha = 0.95  # Close to completion
        
        new_alpha = strategy.update()
        assert new_alpha == 1.0  # Should be capped

    def test_fixed_ramp_exact_steps(self, mock_seed, blend_config):
        """Test that FixedRampBlending reaches 1.0 in exactly fixed_steps."""
        strategy = FixedRampBlending(mock_seed, blend_config)
        
        for i in range(blend_config.fixed_steps):
            mock_seed.alpha = i / blend_config.fixed_steps
            new_alpha = strategy.update()
            
        assert new_alpha == 1.0


class TestPerformanceLinkedBlending:
    """Test suite for PerformanceLinkedBlending strategy."""

    def test_performance_linked_with_improvement(self, mock_seed, blend_config):
        """Test PerformanceLinkedBlending with loss improvement."""
        strategy = PerformanceLinkedBlending(mock_seed, blend_config)
        
        # Set up loss improvement scenario
        mock_seed.seed_info["blend_initial_loss"] = 1.0
        mock_seed.validate_on_holdout.return_value = 0.5  # 50% improvement
        
        new_alpha = strategy.update()
        
        # Alpha should be based on relative improvement
        expected_alpha = 0.5 / 1.0  # (1.0 - 0.5) / 1.0
        assert abs(new_alpha - expected_alpha) < 1e-6

    def test_performance_linked_no_improvement(self, mock_seed, blend_config):
        """Test PerformanceLinkedBlending with no improvement."""
        strategy = PerformanceLinkedBlending(mock_seed, blend_config)
        
        # Set up no improvement scenario
        mock_seed.seed_info["blend_initial_loss"] = 1.0
        mock_seed.validate_on_holdout.return_value = 1.2  # Worse performance
        
        new_alpha = strategy.update()
        
        # Alpha should be 0 when performance gets worse
        assert new_alpha == 0.0


class TestDriftControlledBlending:
    """Test suite for DriftControlledBlending strategy."""

    def test_drift_controlled_low_drift(self, mock_seed, blend_config):
        """Test DriftControlledBlending with low drift (fast progression)."""
        strategy = DriftControlledBlending(mock_seed, blend_config)
        
        # Set up low drift scenario
        mock_seed.seed_info["drift_window"] = [0.01, 0.02, 0.01, 0.02, 0.01]  # Low drift
        
        new_alpha = strategy.update()
        
        # Should use 2x step size for low drift
        expected_step = 2.0 / blend_config.fixed_steps
        assert abs(new_alpha - expected_step) < 1e-6

    def test_drift_controlled_high_drift(self, mock_seed, blend_config):
        """Test DriftControlledBlending with high drift (hold)."""
        strategy = DriftControlledBlending(mock_seed, blend_config)
        
        # Set up high drift scenario
        mock_seed.seed_info["drift_window"] = [0.15, 0.16, 0.14, 0.15, 0.17]  # High drift
        
        new_alpha = strategy.update()
        
        # Should hold (no progress) for high drift
        assert new_alpha == 0.0


class TestGradNormGatedBlending:
    """Test suite for GradNormGatedBlending strategy."""

    def test_grad_norm_in_range(self, mock_seed, blend_config):
        """Test GradNormGatedBlending with gradient norm in acceptable range."""
        strategy = GradNormGatedBlending(mock_seed, blend_config)
        
        # Set gradient norm in acceptable range
        mock_seed.seed_info["avg_grad_norm"] = 0.5  # Between 0.1 and 1.0
        
        new_alpha = strategy.update()
        
        # Should proceed with normal step
        expected_step = 1.0 / blend_config.fixed_steps
        assert abs(new_alpha - expected_step) < 1e-6

    def test_grad_norm_out_of_range(self, mock_seed, blend_config):
        """Test GradNormGatedBlending with gradient norm out of range."""
        strategy = GradNormGatedBlending(mock_seed, blend_config)
        
        # Set gradient norm too high
        mock_seed.seed_info["avg_grad_norm"] = 2.0  # Above upper bound
        
        new_alpha = strategy.update()
        
        # Should hold (no progress) for unstable gradients
        assert new_alpha == 0.0


class TestStrategyFactory:
    """Test suite for the strategy factory function."""

    def test_get_strategy_fixed_ramp(self, mock_seed, blend_config):
        """Test factory creates FixedRampBlending strategy."""
        strategy = get_strategy("FIXED_RAMP", mock_seed, blend_config)
        assert isinstance(strategy, FixedRampBlending)

    def test_get_strategy_performance_linked(self, mock_seed, blend_config):
        """Test factory creates PerformanceLinkedBlending strategy."""
        strategy = get_strategy("PERFORMANCE_LINKED", mock_seed, blend_config)
        assert isinstance(strategy, PerformanceLinkedBlending)

    def test_get_strategy_drift_controlled(self, mock_seed, blend_config):
        """Test factory creates DriftControlledBlending strategy."""
        strategy = get_strategy("DRIFT_CONTROLLED", mock_seed, blend_config)
        assert isinstance(strategy, DriftControlledBlending)

    def test_get_strategy_grad_norm_gated(self, mock_seed, blend_config):
        """Test factory creates GradNormGatedBlending strategy."""
        strategy = get_strategy("GRAD_NORM_GATED", mock_seed, blend_config)
        assert isinstance(strategy, GradNormGatedBlending)

    def test_get_strategy_unknown_defaults_to_fixed_ramp(self, mock_seed, blend_config):
        """Test factory defaults to FixedRampBlending for unknown strategies."""
        strategy = get_strategy("UNKNOWN_STRATEGY", mock_seed, blend_config)
        assert isinstance(strategy, FixedRampBlending)

    def test_get_strategy_empty_string_defaults_to_fixed_ramp(self, mock_seed, blend_config):
        """Test factory defaults to FixedRampBlending for empty string strategy."""
        strategy = get_strategy("", mock_seed, blend_config)
        assert isinstance(strategy, FixedRampBlending)


class TestBlendingIntegration:
    """Integration tests for blending strategies with the broader system."""

    def test_strategy_maintains_alpha_bounds(self, mock_seed, blend_config):
        """Test that all strategies respect alpha bounds [0, 1]."""
        strategies = [
            FixedRampBlending(mock_seed, blend_config),
            PerformanceLinkedBlending(mock_seed, blend_config),
            DriftControlledBlending(mock_seed, blend_config),
            GradNormGatedBlending(mock_seed, blend_config),
        ]
        
        # Set up seed state for all strategies
        mock_seed.seed_info.update({
            "blend_initial_loss": 1.0,
            "drift_window": [0.05, 0.06, 0.05],
            "avg_grad_norm": 0.5
        })
        
        for strategy in strategies:
            mock_seed.alpha = 0.0
            new_alpha = strategy.update()
            assert 0.0 <= new_alpha <= 1.0, f"Alpha out of bounds for {type(strategy).__name__}"
            
            # Test at boundary
            mock_seed.alpha = 0.95
            new_alpha = strategy.update()
            assert 0.0 <= new_alpha <= 1.0, f"Alpha out of bounds for {type(strategy).__name__} at boundary"

    def test_strategy_convergence_properties(self, mock_seed, blend_config):
        """Test that strategies can reach completion."""
        strategy = FixedRampBlending(mock_seed, blend_config)
        
        # The issue is floating point precision. With fixed_steps=10, each step is 0.1
        # But 0.1 cannot be represented exactly in floating point
        # Let's test the actual behavior instead of expecting exact step counts
        
        current_alpha = 0.0
        steps = 0
        
        while current_alpha < 0.999 and steps < blend_config.fixed_steps + 2:  # More lenient
            mock_seed.alpha = current_alpha
            current_alpha = strategy.update()
            steps += 1
            
        # The key test: strategy should reach near 1.0 in reasonable time
        assert abs(current_alpha - 1.0) < 1e-6, "Strategy should reach completion"
        assert steps <= blend_config.fixed_steps + 1, f"Should take at most {blend_config.fixed_steps + 1} steps due to floating point, took {steps}"

    def test_full_seed_lifecycle_with_blending_strategy(self, mock_seed, blend_config):
        """Integration test: seed progresses through GERMINATED → TRAINING → BLENDING → SHADOWING."""
        from morphogenetic_engine.events import SeedState
        from morphogenetic_engine.core import SeedManager
        from morphogenetic_engine.components import SentinelSeed
        from unittest.mock import patch
        
        # Create a real SeedManager and SentinelSeed for integration testing
        seed_manager = SeedManager()
        
        # Mock parent_net with blend_cfg
        mock_parent = Mock()
        mock_parent.blend_cfg = blend_config
        
        # Create a real SentinelSeed
        real_seed = SentinelSeed(
            seed_id=(0, 0),
            dim=10,
            seed_manager=seed_manager,
            parent_net=mock_parent,
            blend_cfg=blend_config
        )
        
        # Test progression through states
        assert real_seed.state == SeedState.DORMANT.value
        
        # Initialize (DORMANT → GERMINATED)
        real_seed.initialize_child()
        assert real_seed.state == SeedState.GERMINATED.value
        
        # Start training (GERMINATED → TRAINING)
        real_seed._set_state(SeedState.TRAINING)
        assert real_seed.state == SeedState.TRAINING.value
        
        # Transition to blending (TRAINING → BLENDING)
        real_seed._set_state(SeedState.BLENDING)
        assert real_seed.state == SeedState.BLENDING.value
        
        # Verify strategy selection works
        seed_info = seed_manager.seeds[(0, 0)]
        seed_info["blend_strategy"] = "FIXED_RAMP"
        
        # Test blending progression
        initial_alpha = real_seed.alpha
        real_seed.update_blending(epoch=1)
        assert real_seed.alpha > initial_alpha, "Alpha should progress during blending"
        
        # Simulate full blending process
        for epoch in range(2, blend_config.fixed_steps + 5):  # Ensure we reach 1.0
            real_seed.update_blending(epoch=epoch)
            if real_seed.alpha >= 0.99:
                break
        
        # Test transition to shadowing (BLENDING → SHADOWING)
        real_seed._handle_blending_transition(epoch=blend_config.fixed_steps + 1)
        assert real_seed.state == SeedState.SHADOWING.value
        
        # Cleanup
        seed_manager.reset()
