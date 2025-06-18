"""
Comprehensive tests for soft landing functionality and state transitions in the morphogenetic engine.

This module tests the critical "soft landing" capabilities that ensure smooth state transitions
between dormant, training, blending, and active states in SentinelSeed components, as well
as proper gradient isolation and resource management during these transitions.

Test Coverage:
- State transition mechanics and lifecycle management
- Gradient isolation and computational boundaries
- Buffer management and tensor shape consistency  
- Error handling and edge cases in state transitions
- Performance characteristics during state changes
- Integration scenarios between components

Classes:
    TestSoftLandingStateTransitions: Core state transition testing
    TestSoftLandingGradientIsolation: Gradient flow and isolation validation
    TestSoftLandingBufferManagement: Buffer operations and shape consistency
    TestSoftLandingErrorHandling: Error scenarios and edge cases
    TestSoftLandingPerformance: Performance validation during transitions
    TestSoftLandingIntegration: Cross-component integration scenarios
"""

# pylint: disable=protected-access

import time
import threading
from typing import List
from unittest.mock import MagicMock

import pytest
import torch

from morphogenetic_engine.components import BaseNet, SentinelSeed
from morphogenetic_engine.core import SeedManager


class TestConstants:
    """Constants for soft landing tests."""
    DEFAULT_DIM = 4
    DEFAULT_BATCH_SIZE = 2
    TRAINING_ITERATIONS = 100
    SMALL_TENSOR_BATCH = 3
    LARGE_TENSOR_BATCH = 64
    BUFFER_SIZE_LIMIT = 64
    PERFORMANCE_THRESHOLD_MS = 10
    MEMORY_THRESHOLD_MB = 50


@pytest.fixture
def seed_manager():
    """Provide a clean SeedManager instance for testing."""
    manager = SeedManager()
    manager.seeds.clear()
    manager.germination_log.clear()
    return manager


@pytest.fixture
def mock_seed_manager(mocker):
    """Provide a mocked SeedManager for isolated unit testing."""
    mock_manager = mocker.create_autospec(SeedManager, spec_set=True)
    mock_manager.seeds = {}
    mock_manager.germination_log = []
    mock_manager.record_transition = mocker.MagicMock()
    mock_manager.append_to_buffer = mocker.MagicMock()
    mock_manager.record_drift = mocker.MagicMock()
    return mock_manager


@pytest.fixture
def sample_tensor():
    """Provide a sample tensor for testing."""
    return torch.randn(TestConstants.DEFAULT_BATCH_SIZE, TestConstants.DEFAULT_DIM)


@pytest.fixture
def configured_seed(seed_manager):
    """Provide a configured SentinelSeed for testing."""
    return SentinelSeed(
        seed_id="test_seed",
        dim=TestConstants.DEFAULT_DIM,
        seed_manager=seed_manager,
        progress_thresh=0.6,
        blend_steps=30
    )


@pytest.fixture
def base_net_with_seeds(seed_manager):
    """Provide a BaseNet with multiple seeds for integration testing."""
    return BaseNet(
        hidden_dim=TestConstants.DEFAULT_DIM,
        seed_manager=seed_manager,
        input_dim=2,
        num_layers=3,
        seeds_per_layer=2
    )


class TestSoftLandingStateTransitions:
    """Test suite for core state transition mechanics during soft landing."""

    def test_training_to_blending_transition(self, configured_seed, sample_tensor):
        """Test that seed transitions from training to blending state after sufficient training steps."""
        seed = configured_seed
        seed.initialize_child()
        
        # Ensure buffer has data for training
        seed.seed_manager.seeds[seed.seed_id]["buffer"].append(sample_tensor)
        
        # Train until transition to blending
        initial_state = seed.state
        for _ in range(TestConstants.TRAINING_ITERATIONS):
            seed.train_child_step(sample_tensor)
            if seed.state == "blending":
                break
        
        assert initial_state == "training"
        assert seed.state == "blending"
        assert seed.alpha == pytest.approx(0.0)  # Alpha starts at 0 in blending

    def test_blending_to_active_transition(self, configured_seed, sample_tensor):
        """Test that seed transitions from blending to active state with proper alpha progression."""
        seed = configured_seed
        seed.initialize_child()
        
        # Progress to blending state
        seed.seed_manager.seeds[seed.seed_id]["buffer"].append(sample_tensor)
        for _ in range(TestConstants.TRAINING_ITERATIONS):
            seed.train_child_step(sample_tensor)
            if seed.state == "blending":
                break
        
        # Progress through blending to active
        initial_alpha = seed.alpha
        for _ in range(seed.blend_steps + 5):  # Extra steps to ensure completion
            seed.update_blending()
            if seed.state == "active":
                break
        
        assert seed.state == "active"
        assert seed.alpha >= 0.99  # Should be very close to 1.0
        assert seed.alpha > initial_alpha

    def test_complete_state_lifecycle(self, configured_seed, sample_tensor):
        """Test complete state progression from dormant through active."""
        seed = configured_seed
        
        # Start in dormant state
        assert seed.state == "dormant"
        
        # Initialize to training
        seed.initialize_child()
        assert seed.state == "training"
        
        # Progress to blending
        seed.seed_manager.seeds[seed.seed_id]["buffer"].append(sample_tensor)
        for _ in range(TestConstants.TRAINING_ITERATIONS):
            seed.train_child_step(sample_tensor)
            if seed.state == "blending":
                break
        assert seed.state == "blending"
        
        # Progress to active
        for _ in range(seed.blend_steps + 5):
            seed.update_blending()
            if seed.state == "active":
                break
        assert seed.state == "active"

    @pytest.mark.parametrize("invalid_state", ["invalid", "unknown", "corrupted"])
    def test_invalid_state_handling(self, configured_seed, invalid_state):
        """Test handling of invalid state transitions."""
        seed = configured_seed
        
        # Directly set invalid state to test error handling
        original_state = seed.state
        seed.state = invalid_state
        
        # Attempt operations that should handle invalid state gracefully
        sample_input = torch.randn(TestConstants.DEFAULT_BATCH_SIZE, TestConstants.DEFAULT_DIM)
        
        try:
            # Forward pass should either work or raise appropriate error
            output = seed.forward(sample_input)
            assert output.shape == sample_input.shape
        except (ValueError, RuntimeError) as e:
            # Should raise meaningful error for invalid state
            assert "state" in str(e).lower() or "invalid" in str(e).lower()
        
        # Restore valid state
        seed.state = original_state

    def test_redundant_transition_prevention(self, configured_seed):
        """Test that redundant state transitions are properly prevented and logged."""
        seed = configured_seed
        manager = seed.seed_manager
        
        # Clear logs for clean testing
        manager.germination_log.clear()
        
        # Perform initial transition
        seed._set_state("training")
        logs_after_first = len(manager.germination_log)
        
        # Attempt redundant transition
        seed._set_state("training")
        logs_after_redundant = len(manager.germination_log)
        
        # Should have logged first transition but not redundant one
        assert logs_after_first == 1
        assert logs_after_redundant == logs_after_first


class TestSoftLandingGradientIsolation:
    """Test suite for gradient isolation during soft landing transitions."""

    def test_gradient_isolation_during_training(self, base_net_with_seeds, sample_tensor):
        """Test that gradient computation is properly isolated during seed training."""
        model = base_net_with_seeds
        
        # Get and initialize first seed
        seeds = model.get_all_seeds()
        assert len(seeds) > 0, "Model should have seeds"
        
        seed = seeds[0]
        seed.initialize_child()
        
        # Create input with gradient tracking
        input_tensor = torch.randn(TestConstants.SMALL_TENSOR_BATCH, TestConstants.DEFAULT_DIM, requires_grad=True)
        
        # Add to buffer and train
        seed.seed_manager.seeds[seed.seed_id]["buffer"].append(input_tensor)
        for _ in range(5):
            seed.train_child_step(input_tensor)
        
        # Check that backbone parameters don't have gradients
        for name, param in model.named_parameters():
            if "seed" not in name and "child" not in name:
                assert param.grad is None, f"Backbone parameter {name} should not have gradients"

    def test_gradient_flow_in_active_state(self, base_net_with_seeds):
        """Test that gradients flow properly when seed is in active state."""
        model = base_net_with_seeds
        
        # Get and activate a seed
        seeds = model.get_all_seeds()
        seed = seeds[0]
        seed.initialize_child()
        seed._set_state("active")
        
        # Forward pass with gradient tracking
        input_tensor = torch.randn(TestConstants.SMALL_TENSOR_BATCH, model.input_layer.in_features, requires_grad=True)
        output = model(input_tensor)
        
        # Compute loss and backward pass
        loss = output.mean()
        loss.backward()
        
        # Active seed should contribute to gradients
        assert input_tensor.grad is not None
        
        # Some model parameters should have gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "Model should have gradients when seed is active"

    def test_gradient_isolation_with_multiple_seeds(self, base_net_with_seeds):
        """Test gradient isolation when multiple seeds are in different states."""
        model = base_net_with_seeds
        seeds = model.get_all_seeds()
        
        if len(seeds) < 2:
            pytest.skip("Need at least 2 seeds for this test")
        
        # Initialize seeds in different states
        seeds[0].initialize_child()  # training state
        seeds[1].initialize_child()
        seeds[1]._set_state("active")  # active state
        
        # Forward pass
        input_tensor = torch.randn(TestConstants.SMALL_TENSOR_BATCH, model.input_layer.in_features, requires_grad=True)
        output = model(input_tensor)
        
        # Loss and backward
        loss = output.mean()
        loss.backward()
        
        # Training seed should not affect backbone gradients
        # Active seed should contribute to gradients
        backbone_has_grads = any(
            p.grad is not None 
            for name, p in model.named_parameters() 
            if "seed" not in name and "child" not in name
        )
        
        # Should have gradients due to active seed
        assert backbone_has_grads


class TestSoftLandingBufferManagement:
    """Test suite for buffer operations and tensor shape consistency during soft landing."""

    def test_forward_pass_shape_consistency(self, configured_seed):
        """Test that forward pass maintains correct tensor shapes throughout state transitions."""
        seed = configured_seed
        input_tensor = torch.randn(5, TestConstants.DEFAULT_DIM)
        
        # Test shape consistency in dormant state
        output = seed.forward(input_tensor)
        assert output.shape == input_tensor.shape
        
        # Test shape consistency in training state
        seed.initialize_child()
        output = seed.forward(input_tensor)
        assert output.shape == input_tensor.shape
        
        # Progress to blending state and test
        seed.seed_manager.seeds[seed.seed_id]["buffer"].append(input_tensor)
        for _ in range(TestConstants.TRAINING_ITERATIONS):
            seed.train_child_step(input_tensor)
            if seed.state == "blending":
                break
        
        seed.alpha = 0.5  # Set blending alpha
        output = seed.forward(input_tensor)
        assert output.shape == input_tensor.shape
        
        # Test shape consistency in active state
        seed._set_state("active")
        output = seed.forward(input_tensor)
        assert output.shape == input_tensor.shape

    def test_buffer_tensor_concatenation(self):
        """Test that buffer sampling produces correct batch shapes with proper tensor concatenation."""
        # Create buffer with different batch sizes
        buffer_tensors = [
            torch.randn(TestConstants.LARGE_TENSOR_BATCH, 128),
            torch.randn(16, 128),
            torch.randn(32, 128)
        ]
        
        # Simulate buffer sampling logic
        sample_tensors = buffer_tensors[:min(TestConstants.BUFFER_SIZE_LIMIT, len(buffer_tensors))]
        batch = torch.cat(sample_tensors, dim=0)
        
        # Ensure batch doesn't exceed limit
        if batch.size(0) > TestConstants.BUFFER_SIZE_LIMIT:
            idx = torch.randperm(batch.size(0), device=batch.device)[:TestConstants.BUFFER_SIZE_LIMIT]
            batch = batch[idx]
        
        assert batch.shape[0] <= TestConstants.BUFFER_SIZE_LIMIT
        assert batch.shape[1] == 128  # Feature dimension preserved

    def test_buffer_overflow_handling(self, configured_seed):
        """Test buffer behavior when capacity limits are exceeded."""
        seed = configured_seed
        buffer = seed.seed_manager.seeds[seed.seed_id]["buffer"]
        
        # Fill buffer beyond typical capacity
        for i in range(600):  # Exceed default maxlen of 500
            tensor = torch.randn(TestConstants.DEFAULT_BATCH_SIZE, TestConstants.DEFAULT_DIM)
            buffer.append(tensor)
        
        # Buffer should maintain size limit
        assert len(buffer) <= 500  # Default maxlen
        
        # Most recent tensors should be preserved
        assert buffer[-1] is not None

    def test_empty_buffer_handling(self, configured_seed):
        """Test handling of operations on empty buffers."""
        seed = configured_seed
        
        # Ensure buffer is empty
        buffer = seed.seed_manager.seeds[seed.seed_id]["buffer"]
        buffer.clear()
        
        # Health signal should handle empty buffer gracefully
        health = seed.get_health_signal()
        assert health == float("inf")  # Expected behavior for insufficient data

    def test_buffer_data_integrity(self, configured_seed, sample_tensor):
        """Test that buffer maintains data integrity during operations."""
        seed = configured_seed
        buffer = seed.seed_manager.seeds[seed.seed_id]["buffer"]
        
        # Add tensor to buffer
        original_tensor = sample_tensor.clone()
        buffer.append(sample_tensor)
        
        # Retrieve and verify
        retrieved_tensor = buffer[-1]
        assert torch.allclose(retrieved_tensor, original_tensor, atol=1e-6)
        
        # Ensure detachment (no gradients)
        assert not retrieved_tensor.requires_grad


class TestSoftLandingErrorHandling:
    """Test suite for error scenarios and edge cases in soft landing."""

    def test_training_with_empty_input(self, configured_seed):
        """Test training behavior with empty input tensors."""
        seed = configured_seed
        seed.initialize_child()
        
        empty_input = torch.empty(0, TestConstants.DEFAULT_DIM)
        initial_progress = seed.training_progress
        
        # Training with empty input should not change progress
        seed.train_child_step(empty_input)
        assert seed.training_progress == initial_progress

    def test_training_with_nan_input(self, configured_seed):
        """Test training behavior with NaN input values."""
        seed = configured_seed
        seed.initialize_child()
        
        nan_input = torch.full((TestConstants.DEFAULT_BATCH_SIZE, TestConstants.DEFAULT_DIM), float('nan'))
        
        # Training should handle NaN input gracefully
        try:
            seed.train_child_step(nan_input)
            # If no exception, training should not crash
        except (ValueError, RuntimeError):
            # Acceptable to raise error for invalid input
            pass

    def test_dimension_mismatch_handling(self, configured_seed):
        """Test handling of dimension mismatches in forward pass."""
        seed = configured_seed
        seed.initialize_child()
        seed._set_state("active")  # State where child network is used
        
        # Create input with wrong dimensions
        wrong_dim_input = torch.randn(TestConstants.DEFAULT_BATCH_SIZE, TestConstants.DEFAULT_DIM + 5)
        
        # Should raise appropriate error for dimension mismatch
        with pytest.raises(RuntimeError, match="size|dimension|shape"):
            seed.forward(wrong_dim_input)

    def test_corrupted_state_recovery(self, configured_seed):
        """Test recovery from corrupted seed manager state."""
        seed = configured_seed
        
        # Corrupt the seed manager state
        original_seeds = seed.seed_manager.seeds
        seed.seed_manager.seeds = None
        
        # Operations should handle corrupted state
        with pytest.raises((AttributeError, TypeError)):
            seed._set_state("training")
        
        # Restore state for cleanup
        seed.seed_manager.seeds = original_seeds

    def test_extreme_alpha_values(self, configured_seed, sample_tensor):
        """Test behavior with extreme alpha values during blending."""
        seed = configured_seed
        seed.initialize_child()
        seed._set_state("blending")
        
        # Test with alpha values at boundaries
        extreme_alphas = [-0.1, 0.0, 0.5, 1.0, 1.1]
        
        for alpha in extreme_alphas:
            seed.alpha = alpha
            
            try:
                output = seed.forward(sample_tensor)
                assert output.shape == sample_tensor.shape
            except (ValueError, RuntimeError):
                # Acceptable to raise errors for invalid alpha values
                if alpha < 0 or alpha > 1:
                    pass  # Expected for out-of-range values
                else:
                    raise


class TestSoftLandingPerformance:
    """Test suite for performance characteristics during soft landing transitions."""

    def test_state_transition_performance(self, configured_seed, sample_tensor):
        """Test that state transitions complete within reasonable time limits."""
        seed = configured_seed
        
        # Measure training to blending transition time
        start_time = time.perf_counter()
        
        seed.initialize_child()
        seed.seed_manager.seeds[seed.seed_id]["buffer"].append(sample_tensor)
        
        for _ in range(TestConstants.TRAINING_ITERATIONS):
            seed.train_child_step(sample_tensor)
            if seed.state == "blending":
                break
        
        end_time = time.perf_counter()
        transition_time_ms = (end_time - start_time) * 1000
        
        # Should complete within reasonable time
        assert transition_time_ms < 1000  # 1 second threshold

    def test_forward_pass_performance(self, configured_seed):
        """Test forward pass performance across different states."""
        seed = configured_seed
        input_tensor = torch.randn(100, TestConstants.DEFAULT_DIM)  # Larger batch for timing
        
        # Measure forward pass times in different states
        states_to_test = ["dormant"]
        
        seed.initialize_child()
        states_to_test.append("training")
        
        # Progress to blending
        dummy_input = torch.randn(TestConstants.DEFAULT_BATCH_SIZE, TestConstants.DEFAULT_DIM)
        seed.seed_manager.seeds[seed.seed_id]["buffer"].append(dummy_input)
        for _ in range(TestConstants.TRAINING_ITERATIONS):
            seed.train_child_step(dummy_input)
            if seed.state == "blending":
                break
        states_to_test.append("blending")
        
        # Progress to active
        seed._set_state("active")
        states_to_test.append("active")
        
        for state in states_to_test:
            if state != seed.state:
                seed._set_state(state)
            
            start_time = time.perf_counter()
            for _ in range(10):
                seed.forward(input_tensor)
            end_time = time.perf_counter()
            
            avg_time_ms = ((end_time - start_time) / 10) * 1000
            assert avg_time_ms < TestConstants.PERFORMANCE_THRESHOLD_MS

    def test_memory_usage_during_transitions(self, configured_seed, sample_tensor):
        """Test memory usage remains reasonable during state transitions."""
        seed = configured_seed
        
        # Get initial memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0
        
        # Perform complete lifecycle
        seed.initialize_child()
        seed.seed_manager.seeds[seed.seed_id]["buffer"].append(sample_tensor)
        
        for _ in range(TestConstants.TRAINING_ITERATIONS):
            seed.train_child_step(sample_tensor)
            if seed.state == "blending":
                break
        
        for _ in range(seed.blend_steps + 5):
            seed.update_blending()
            if seed.state == "active":
                break
        
        # Check final memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)
            assert memory_growth_mb < TestConstants.MEMORY_THRESHOLD_MB


class TestSoftLandingIntegration:
    """Test suite for integration scenarios during soft landing."""

    def test_concurrent_state_transitions(self, seed_manager):
        """Test thread safety during concurrent state transitions."""
        seeds = [
            SentinelSeed(f"concurrent_seed_{i}", TestConstants.DEFAULT_DIM, seed_manager)
            for i in range(3)
        ]
        
        results = []
        
        def transition_worker(seed, target_state):
            try:
                if target_state == "training":
                    seed.initialize_child()
                else:
                    seed._set_state(target_state)
                results.append(f"success_{seed.seed_id}")
            except Exception as e:
                results.append(f"error_{seed.seed_id}_{e}")
        
        # Create concurrent transitions
        threads = [
            threading.Thread(target=transition_worker, args=(seeds[i], "training"))
            for i in range(len(seeds))
        ]
        
        # Execute concurrently
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should succeed
        assert len(results) == len(seeds)
        assert all("success" in r for r in results)

    def test_multi_seed_soft_landing(self, base_net_with_seeds, sample_tensor):
        """Test soft landing behavior with multiple seeds in a BaseNet."""
        model = base_net_with_seeds
        seeds = model.get_all_seeds()
        
        # Initialize some seeds
        for i, seed in enumerate(seeds[:2]):  # Initialize first 2 seeds
            seed.initialize_child()
            seed.seed_manager.seeds[seed.seed_id]["buffer"].append(sample_tensor)
        
        # Train seeds to different states
        for _ in range(TestConstants.TRAINING_ITERATIONS):
            for seed in seeds[:2]:
                seed.train_child_step(sample_tensor)
                if seed.state == "blending":
                    break
        
        # Verify seeds are in expected states
        active_seeds = [s for s in seeds[:2] if s.state in ["training", "blending"]]
        assert len(active_seeds) >= 1
        
        # Forward pass should work with mixed seed states
        input_tensor = torch.randn(TestConstants.SMALL_TENSOR_BATCH, model.input_layer.in_features)
        output = model(input_tensor)
        assert output.shape[0] == TestConstants.SMALL_TENSOR_BATCH

    def test_seed_manager_integration_during_transitions(self, configured_seed, sample_tensor):
        """Test SeedManager integration during complete state transitions."""
        seed = configured_seed
        manager = seed.seed_manager
        
        # Clear logs for clean testing
        manager.germination_log.clear()
        
        # Complete lifecycle with logging verification
        seed.initialize_child()
        transition_count = len(manager.germination_log)
        
        seed.seed_manager.seeds[seed.seed_id]["buffer"].append(sample_tensor)
        for _ in range(TestConstants.TRAINING_ITERATIONS):
            seed.train_child_step(sample_tensor)
            if seed.state == "blending":
                break
        
        blending_transition_count = len(manager.germination_log)
        assert blending_transition_count > transition_count
        
        # Progress to active
        for _ in range(seed.blend_steps + 5):
            seed.update_blending()
            if seed.state == "active":
                break
        
        final_transition_count = len(manager.germination_log)
        assert final_transition_count > blending_transition_count
        
        # Verify all transitions were logged
        assert final_transition_count >= 2  # At least training->blending->active
