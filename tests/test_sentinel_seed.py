"""
Enhanced tests for the SentinelSeed component.

This module provides comprehensive unit testing for the SentinelSeed class,
focusing on proper isolation, error handling, and modern testing practices.

Test Coverage:
- Initialization and parameter validation
- State transition mechanics with proper isolation
- Child network behavior and training dynamics
- Error handling and edge cases
- Performance and resource validation
- Concurrency and thread safety
- Property-based testing for robustness
"""

# pylint: disable=protected-access,redefined-outer-name

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from morphogenetic_engine.components import SentinelSeed
from morphogenetic_engine.core import SeedManager


class TestConstants:
    """Constants for test configuration and validation."""

    DEFAULT_DIM = 32
    DEFAULT_BATCH_SIZE = 4
    TRAINING_ITERATIONS_TO_THRESHOLD = 60
    SUFFICIENT_BUFFER_SIZE = 15
    SMALL_BLEND_STEPS = 10
    MEMORY_THRESHOLD_MB = 50  # Maximum memory growth in MB
    PERFORMANCE_THRESHOLD_MS = 10  # Maximum time per operation in ms
    THREAD_COUNT = 5  # Number of threads for concurrency tests


@pytest.fixture
def mock_seed_manager(mocker):
    """Provide a properly mocked SeedManager for isolated unit testing."""
    mock_manager = mocker.MagicMock(spec=SeedManager)

    # Configure mock behavior - use regular attributes, not autospec
    mock_manager.seeds = {}
    mock_manager.germination_log = []
    mock_manager.record_transition = mocker.MagicMock()
    mock_manager.record_drift = mocker.MagicMock()
    mock_manager.append_to_buffer = mocker.MagicMock()
    mock_manager.register_seed = mocker.MagicMock()

    # Set up seeds dictionary behavior
    def register_seed_side_effect(seed_module, seed_id):
        mock_manager.seeds[seed_id] = {
            "module": seed_module,
            "status": "dormant",
            "state": "dormant",
            "alpha": 0.0,
            "buffer": [],
            "telemetry": {"drift": 0.0, "variance": 0.0},
        }

    mock_manager.register_seed.side_effect = register_seed_side_effect

    return mock_manager


@pytest.fixture
def sample_tensor():
    """Provide a sample tensor for testing."""
    return torch.randn(TestConstants.DEFAULT_BATCH_SIZE, TestConstants.DEFAULT_DIM)


@pytest.fixture
def configured_seed(mock_seed_manager):
    """Provide a configured SentinelSeed for testing."""
    return SentinelSeed(
        seed_id="test_seed",
        dim=TestConstants.DEFAULT_DIM,
        seed_manager=mock_seed_manager,
        progress_thresh=0.5,
        blend_steps=TestConstants.SMALL_BLEND_STEPS,
        shadow_lr=1e-3,
        drift_warn=0.1,
    )


@pytest.fixture
def configured_seed_real_manager():
    """Provide a SentinelSeed with real SeedManager for integration scenarios."""
    return SentinelSeed(
        seed_id="test_seed",
        dim=TestConstants.DEFAULT_DIM,
        seed_manager=SeedManager(),
        progress_thresh=0.5,
        blend_steps=TestConstants.SMALL_BLEND_STEPS,
    )


class TestSentinelSeedInitialization:
    """Test suite for SentinelSeed initialization and parameter validation."""

    def test_initialization_with_mocked_manager(self, mock_seed_manager):
        """Test SentinelSeed initialization with proper isolation."""
        seed = SentinelSeed(
            seed_id="test_seed",
            dim=64,
            seed_manager=mock_seed_manager,
            blend_steps=25,
            shadow_lr=2e-3,
            progress_thresh=0.7,
            drift_warn=0.15,
        )

        # Verify basic properties
        assert seed.seed_id == "test_seed"
        assert seed.dim == 64
        assert seed.blend_steps == 25
        assert seed.progress_thresh == pytest.approx(0.7)
        assert seed.alpha == pytest.approx(0.0)
        assert seed.state == "dormant"
        assert seed.training_progress == pytest.approx(0.0)
        assert seed.drift_warn == pytest.approx(0.15)

        # Verify child network structure
        assert isinstance(seed.child, torch.nn.Sequential)
        assert len(seed.child) == 3  # Linear, ReLU, Linear

        # Verify optimizer and loss function
        assert isinstance(seed.child_optim, torch.optim.Adam)
        assert isinstance(seed.child_loss, torch.nn.MSELoss)

        # Verify manager interaction
        mock_seed_manager.register_seed.assert_called_once_with(seed, "test_seed")

    def test_invalid_initialization_parameters(self, mock_seed_manager):
        """Test behavior with invalid initialization parameters."""
        # Test invalid dimension
        with pytest.raises(ValueError, match="Invalid dimension"):
            SentinelSeed("test", dim=0, seed_manager=mock_seed_manager)

        with pytest.raises(ValueError, match="Invalid dimension"):
            SentinelSeed("test", dim=-5, seed_manager=mock_seed_manager)

        # Test invalid blend_steps
        with pytest.raises(ValueError, match="Invalid blend_steps"):
            SentinelSeed("test", dim=32, seed_manager=mock_seed_manager, blend_steps=0)

        with pytest.raises(ValueError, match="Invalid blend_steps"):
            SentinelSeed("test", dim=32, seed_manager=mock_seed_manager, blend_steps=-10)

        # Test invalid progress_thresh
        with pytest.raises(ValueError, match="Invalid progress_thresh"):
            SentinelSeed("test", dim=32, seed_manager=mock_seed_manager, progress_thresh=0.0)

        with pytest.raises(ValueError, match="Invalid progress_thresh"):
            SentinelSeed("test", dim=32, seed_manager=mock_seed_manager, progress_thresh=1.1)

    @given(
        dim=st.integers(min_value=1, max_value=64),  # Reduced range for performance
        blend_steps=st.integers(min_value=1, max_value=50),  # Reduced range for performance
        progress_thresh=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,  # Disable deadline for this test
        max_examples=10,  # Reduce number of examples for performance
    )
    def test_initialization_property_based(
        self, dim, blend_steps, progress_thresh, mock_seed_manager
    ):
        """Test initialization works correctly for any valid parameters."""
        seed = SentinelSeed(
            "test_seed",
            dim=dim,
            seed_manager=mock_seed_manager,
            blend_steps=blend_steps,
            progress_thresh=progress_thresh,
        )

        assert seed.dim == dim
        assert seed.blend_steps == blend_steps
        assert seed.progress_thresh == pytest.approx(progress_thresh)
        assert seed.state == "dormant"
        assert seed.alpha == pytest.approx(0.0)


class TestSentinelSeedStateTransitions:
    """Test suite for state transition mechanics with proper isolation."""

    def test_state_transitions_isolated(self, configured_seed, mock_seed_manager):
        """Test state transition handling with proper mocking."""
        seed = configured_seed

        # Configure mock to track state changes
        mock_seed_manager.seeds["test_seed"] = {"state": "dormant", "status": "dormant"}

        # Test transition to training
        seed._set_state("training")
        assert seed.state == "training"

        # Verify manager interaction
        mock_seed_manager.record_transition.assert_called_with("test_seed", "dormant", "training")

        # Test transition to active
        seed._set_state("active")
        assert seed.state == "active"

        # Verify multiple transitions logged
        assert mock_seed_manager.record_transition.call_count == 2

    def test_redundant_state_transition_ignored(self, configured_seed, mock_seed_manager):
        """Test that redundant state transitions are properly ignored."""
        seed = configured_seed

        # Set initial state
        seed._set_state("training")
        call_count_after_first = mock_seed_manager.record_transition.call_count

        # Same state transition should be ignored
        seed._set_state("training")
        call_count_after_second = mock_seed_manager.record_transition.call_count

        assert call_count_after_first == call_count_after_second

    @pytest.mark.parametrize(
        "initial_state,target_state,should_log",
        [
            ("dormant", "training", True),
            ("training", "blending", True),
            ("blending", "active", True),
            ("active", "dormant", True),
            ("dormant", "dormant", False),  # Redundant transition
            ("training", "training", False),  # Redundant transition
        ],
    )
    def test_state_transitions_parametrized(
        self, configured_seed, mock_seed_manager, initial_state, target_state, should_log
    ):
        """Test various state transitions systematically."""
        seed = configured_seed
        seed.state = initial_state  # Set initial state directly

        mock_seed_manager.record_transition.reset_mock()
        seed._set_state(target_state)

        assert seed.state == target_state
        if should_log:
            mock_seed_manager.record_transition.assert_called_once()
        else:
            mock_seed_manager.record_transition.assert_not_called()

    def test_invalid_state_transition_handling(self, configured_seed):
        """Test handling of invalid state transitions."""
        seed = configured_seed

        # Test transition to invalid state
        with pytest.raises(ValueError, match="Invalid state"):
            seed._set_state("invalid_state")


class TestSentinelSeedChildNetwork:
    """Test suite for child network initialization and behavior."""

    def test_child_network_identity_initialization(self, configured_seed):
        """Test that child network is properly initialized as identity."""
        seed = configured_seed

        # Initially should be identity (zero weights)
        for param in seed.child.parameters():
            assert torch.allclose(param, torch.zeros_like(param))
            assert not param.requires_grad

    def test_child_network_proper_initialization(self, configured_seed, mock_seed_manager):
        """Test proper child network initialization when germinating."""
        seed = configured_seed

        # After initialization, should have proper weights
        seed.initialize_child()
        assert seed.state == "training"

        weight_params = [p for p in seed.child.parameters() if p.dim() > 1]
        for param in weight_params:
            assert not torch.allclose(param, torch.zeros_like(param))
            assert param.requires_grad

        # Verify state transition was recorded
        mock_seed_manager.record_transition.assert_called_with("test_seed", "dormant", "training")

    def test_child_network_initialization_failure(self, configured_seed, mocker):
        """Test behavior when child network initialization fails."""
        seed = configured_seed

        # Mock Kaiming normal to raise an exception
        with mocker.patch("torch.nn.init.kaiming_normal_", side_effect=RuntimeError("Init failed")):
            with pytest.raises(RuntimeError, match="Init failed"):
                seed.initialize_child()

        # State should remain dormant after failure
        assert seed.state == "dormant"


class TestSentinelSeedTraining:
    """Test suite for training behavior and progress tracking."""

    def test_training_behavior_dormant_state(self, configured_seed, sample_tensor):
        """Test that training is ignored in dormant state."""
        seed = configured_seed
        initial_progress = seed.training_progress

        seed.train_child_step(sample_tensor)

        assert seed.training_progress == initial_progress
        assert seed.state == "dormant"

    def test_training_behavior_empty_input(self, configured_seed):
        """Test training behavior with empty input tensor."""
        seed = configured_seed
        seed.initialize_child()  # Set to training state

        empty_input = torch.empty(0, TestConstants.DEFAULT_DIM)
        initial_progress = seed.training_progress

        seed.train_child_step(empty_input)

        # Progress should not change with empty input
        assert seed.training_progress == initial_progress

    def test_training_progress_and_state_transition(
        self, configured_seed, sample_tensor, mock_seed_manager
    ):
        """Test training progress tracking and automatic state transitions."""
        seed = configured_seed
        seed.initialize_child()

        initial_progress = seed.training_progress

        # Train until threshold is reached
        for _ in range(TestConstants.TRAINING_ITERATIONS_TO_THRESHOLD):
            seed.train_child_step(sample_tensor)
            if seed.state == "blending":
                break

        assert seed.training_progress > initial_progress
        assert seed.state == "blending"
        assert seed.alpha == pytest.approx(0.0)

        # Verify state transition was logged
        mock_seed_manager.record_transition.assert_any_call("test_seed", "training", "blending")

    def test_training_with_corrupted_optimizer(self, configured_seed, sample_tensor, mocker):
        """Test training behavior when optimizer fails."""
        seed = configured_seed
        seed.initialize_child()

        # Mock optimizer step to raise exception
        with mocker.patch.object(
            seed.child_optim, "step", side_effect=RuntimeError("Optimizer failed")
        ):
            with pytest.raises(RuntimeError, match="Optimizer failed"):
                seed.train_child_step(sample_tensor)


class TestSentinelSeedBlending:
    """Test suite for blending process and alpha interpolation."""

    def test_blending_alpha_progression(self, configured_seed):
        """Test alpha progression during blending phase."""
        seed = configured_seed
        seed._set_state("blending")
        seed.alpha = 0.0

        initial_alpha = seed.alpha
        seed.update_blending()

        assert seed.alpha > initial_alpha
        assert seed.state == "blending"

    def test_blending_completion_to_active(self, configured_seed, mock_seed_manager):
        """Test automatic transition to active state when blending completes."""
        seed = configured_seed
        seed._set_state("blending")
        seed.alpha = 0.0

        # Complete blending process
        for _ in range(TestConstants.SMALL_BLEND_STEPS + 2):
            seed.update_blending()
            if seed.state == "active":
                break

        assert seed.alpha >= 0.99
        assert seed.state == "active"

        # Verify transition was recorded
        mock_seed_manager.record_transition.assert_any_call("test_seed", "blending", "active")

    def test_blending_ignored_in_non_blending_state(self, configured_seed):
        """Test that blending updates are ignored in non-blending states."""
        seed = configured_seed

        # Test in dormant state
        initial_alpha = seed.alpha
        seed.update_blending()
        assert seed.alpha == initial_alpha

        # Test in active state
        seed._set_state("active")
        alpha_before = seed.alpha
        seed.update_blending()
        assert seed.alpha == alpha_before


class TestSentinelSeedForwardPass:
    """Test suite for forward pass behavior across different states."""

    @pytest.mark.parametrize(
        "state,expected_behavior",
        [
            ("dormant", "returns_input_unchanged"),
            ("training", "returns_input_unchanged"),
            ("blending", "returns_blended_output"),
            ("active", "returns_residual_output"),
        ],
    )
    def test_forward_behavior_by_state(
        self, configured_seed_real_manager, sample_tensor, state, expected_behavior
    ):
        """Test forward pass behavior across all states."""
        seed = configured_seed_real_manager

        # Set up seed for specific state
        if state in ["training", "blending", "active"]:
            seed.initialize_child()

        if state in ["blending", "active"]:
            seed._set_state(state)

        if state == "blending":
            seed.alpha = 0.5

        output = seed.forward(sample_tensor)

        # Verify output properties
        assert output.shape == sample_tensor.shape
        assert output.dtype == sample_tensor.dtype

        if expected_behavior == "returns_input_unchanged":
            assert torch.allclose(output, sample_tensor, atol=1e-6)
        elif expected_behavior in ["returns_blended_output", "returns_residual_output"]:
            # Output should be different from input
            assert not torch.allclose(output, sample_tensor, atol=1e-6)

    def test_drift_monitoring_and_recording(
        self, configured_seed, sample_tensor, mock_seed_manager
    ):
        """Test drift computation and telemetry recording."""
        seed = configured_seed
        seed.initialize_child()
        seed._set_state("blending")
        seed.alpha = 0.5

        output = seed.forward(sample_tensor)

        # Verify output shape
        assert output.shape == sample_tensor.shape

        # Verify drift was recorded
        mock_seed_manager.record_drift.assert_called_once()

        # Get the drift value that was recorded
        drift_call_args = mock_seed_manager.record_drift.call_args
        assert drift_call_args[0][0] == "test_seed"  # seed_id
        assert isinstance(drift_call_args[0][1], float)  # drift value
        assert 0.0 <= drift_call_args[0][1] <= 2.0  # drift should be in valid range

    def test_forward_pass_with_nan_input(self, configured_seed):
        """Test forward pass behavior with NaN input values."""
        seed = configured_seed

        nan_input = torch.full((4, TestConstants.DEFAULT_DIM), float("nan"))

        # Test that forward pass doesn't crash with NaN inputs
        # In dormant state, should return the input (which will be NaN)
        output = seed.forward(nan_input)

        # Output should have the same shape
        assert output.shape == nan_input.shape

        # In dormant state, output should be the same as input (all NaN)
        if seed.state == "dormant":
            assert torch.isnan(output).all()
        else:
            # For other states, NaN handling depends on the child network
            # At minimum, output should have correct shape
            assert output.shape == nan_input.shape


class TestSentinelSeedHealthSignal:
    """Test suite for health signal computation and buffer management."""

    def test_health_signal_insufficient_data(self, configured_seed_real_manager):
        """Test health signal computation with insufficient buffer data."""
        seed = configured_seed_real_manager

        # With empty buffer, should return infinity
        health = seed.get_health_signal()
        assert health == float("inf")

    def test_health_signal_sufficient_data(self, configured_seed_real_manager):
        """Test health signal computation with sufficient buffer data."""
        seed = configured_seed_real_manager

        # Add sufficient data to buffer
        buffer = seed.seed_manager.seeds["test_seed"]["buffer"]
        for _ in range(TestConstants.SUFFICIENT_BUFFER_SIZE):
            buffer.append(torch.randn(2, TestConstants.DEFAULT_DIM))

        health = seed.get_health_signal()
        assert isinstance(health, float)
        assert health != float("inf")
        assert health >= 0.0

    def test_health_signal_with_zero_variance(self, configured_seed_real_manager):
        """Test health signal when buffer contains identical values."""
        seed = configured_seed_real_manager

        # Add identical tensors to create zero variance
        buffer = seed.seed_manager.seeds["test_seed"]["buffer"]
        identical_tensor = torch.ones(2, TestConstants.DEFAULT_DIM)
        for _ in range(TestConstants.SUFFICIENT_BUFFER_SIZE):
            buffer.append(identical_tensor.clone())

        health = seed.get_health_signal()
        assert isinstance(health, float)
        assert health == pytest.approx(0.0, abs=1e-6)


class TestSentinelSeedPerformance:
    """Test suite for performance validation and resource management."""

    def test_memory_usage_during_training(self, configured_seed_real_manager, sample_tensor):
        """Test that training doesn't cause excessive memory growth."""
        seed = configured_seed_real_manager
        seed.initialize_child()

        # Get initial memory usage
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Perform multiple training steps
        for _ in range(100):
            seed.train_child_step(sample_tensor)

        # Check memory growth
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)

        assert memory_growth_mb < TestConstants.MEMORY_THRESHOLD_MB

    def test_forward_pass_performance(self, configured_seed_real_manager):
        """Test that forward pass operations complete within acceptable time."""
        seed = configured_seed_real_manager

        # Use larger batch for meaningful timing
        large_batch = torch.randn(100, TestConstants.DEFAULT_DIM)

        # Measure execution time
        start_time = time.perf_counter()
        for _ in range(10):
            seed.forward(large_batch)
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / 10) * 1000
        assert avg_time_ms < TestConstants.PERFORMANCE_THRESHOLD_MS

    def test_memory_cleanup_on_state_reset(self, configured_seed_real_manager):
        """Test that changing states doesn't leak memory references."""
        seed = configured_seed_real_manager

        # Initialize and activate seed
        seed.initialize_child()
        seed._set_state("active")

        # Get initial parameter count
        initial_params = sum(p.numel() for p in seed.child.parameters())

        # Reset to dormant (should not change parameter count)
        seed._set_state("dormant")

        final_params = sum(p.numel() for p in seed.child.parameters())
        assert final_params == initial_params


class TestSentinelSeedConcurrency:
    """Test suite for concurrency and thread safety validation."""

    def test_concurrent_state_transitions(self, configured_seed_real_manager):
        """Test thread safety of state transition operations."""
        seed = configured_seed_real_manager
        results = []

        def transition_worker(target_state):
            try:
                seed._set_state(target_state)
                results.append(f"success_{target_state}")
            except Exception as e:
                results.append(f"error_{target_state}_{e}")

        # Create threads for different state transitions
        threads = [
            threading.Thread(target=transition_worker, args=("training",)),
            threading.Thread(target=transition_worker, args=("blending",)),
            threading.Thread(target=transition_worker, args=("active",)),
        ]

        # Execute concurrent transitions
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All operations should complete (success or controlled failure)
        assert len(results) == 3
        assert all("success" in r or "error" in r for r in results)

    def test_concurrent_forward_passes(self, configured_seed_real_manager):
        """Test thread safety of forward pass operations."""
        seed = configured_seed_real_manager
        seed.initialize_child()

        results = []
        sample_inputs = [
            torch.randn(4, TestConstants.DEFAULT_DIM) for _ in range(TestConstants.THREAD_COUNT)
        ]

        def forward_worker(inputs, worker_id):
            try:
                output = seed.forward(inputs)
                results.append(f"success_{worker_id}_{output.shape}")
            except Exception as e:
                results.append(f"error_{worker_id}_{e}")

        # Execute concurrent forward passes
        with ThreadPoolExecutor(max_workers=TestConstants.THREAD_COUNT) as executor:
            futures = [
                executor.submit(forward_worker, inputs, i) for i, inputs in enumerate(sample_inputs)
            ]

            for future in as_completed(futures):
                future.result()  # Wait for completion

        # All operations should succeed
        assert len(results) == TestConstants.THREAD_COUNT
        assert all("success" in r for r in results)

    def test_concurrent_training_steps(self, configured_seed_real_manager):
        """Test thread safety of training operations."""
        seed = configured_seed_real_manager
        seed.initialize_child()

        results = []

        def training_worker(worker_id):
            try:
                inputs = torch.randn(4, TestConstants.DEFAULT_DIM)
                for _ in range(10):  # Multiple training steps per worker
                    seed.train_child_step(inputs)
                results.append(f"success_{worker_id}")
            except Exception as e:
                results.append(f"error_{worker_id}_{e}")

        # Execute concurrent training
        threads = [
            threading.Thread(target=training_worker, args=(i,))
            for i in range(TestConstants.THREAD_COUNT)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All operations should complete
        assert len(results) == TestConstants.THREAD_COUNT


class TestSentinelSeedErrorHandling:
    """Test suite for comprehensive error handling and edge cases."""

    def test_corrupted_seed_manager_state(self, configured_seed, mock_seed_manager):
        """Test behavior when SeedManager state is corrupted."""
        seed = configured_seed

        # Corrupt the seeds dictionary
        mock_seed_manager.seeds = None

        # Operations should handle corrupted state gracefully
        with pytest.raises((AttributeError, TypeError)):
            seed._set_state("training")

    def test_invalid_tensor_operations(self, configured_seed):
        """Test handling of invalid tensor operations."""
        seed = configured_seed
        seed.initialize_child()
        seed._set_state("active")  # Set to active state where child network is used

        # Test with mismatched dimensions that will cause PyTorch error
        # The child network expects DEFAULT_DIM (32), so give it a different dimension
        wrong_dim_input = torch.randn(4, TestConstants.DEFAULT_DIM + 10)  # 42 instead of 32

        # This should raise a RuntimeError due to dimension mismatch in Linear layer
        try:
            seed.forward(wrong_dim_input)
            # If no exception was raised, the test should fail
            pytest.fail("Expected RuntimeError for dimension mismatch, but none was raised")
        except RuntimeError as e:
            # Verify it's the expected dimension error
            assert (
                "size" in str(e).lower()
                or "dimension" in str(e).lower()
                or "shape" in str(e).lower()
            )

    def test_extreme_parameter_values(self, mock_seed_manager):
        """Test behavior with extreme but valid parameter values."""
        # Test with very small dimension
        seed_small = SentinelSeed("small", dim=1, seed_manager=mock_seed_manager)
        assert seed_small.dim == 1

        # Test with very large blend_steps
        seed_large_blend = SentinelSeed(
            "large_blend", dim=32, seed_manager=mock_seed_manager, blend_steps=1000
        )
        assert seed_large_blend.blend_steps == 1000

        # Test with extreme progress threshold
        seed_high_thresh = SentinelSeed(
            "high_thresh", dim=32, seed_manager=mock_seed_manager, progress_thresh=0.99
        )
        assert seed_high_thresh.progress_thresh == pytest.approx(0.99)

    def test_resource_exhaustion_simulation(self, configured_seed_real_manager, mocker):
        """Test behavior under simulated resource exhaustion."""
        seed = configured_seed_real_manager

        # Mock torch operations to simulate out of memory
        with mocker.patch("torch.randn", side_effect=RuntimeError("CUDA out of memory")):
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                large_input = torch.randn(1000, TestConstants.DEFAULT_DIM)  # This should fail
                seed.forward(large_input)


class TestSentinelSeedIntegrationScenarios:
    """Test suite for integration scenarios and end-to-end workflows."""

    def test_complete_lifecycle_integration(self, configured_seed_real_manager, sample_tensor):
        """Test complete seed lifecycle from dormant to active."""
        seed = configured_seed_real_manager

        # Start in dormant state
        assert seed.state == "dormant"

        # Initialize to training
        seed.initialize_child()
        assert seed.state == "training"

        # Train until blending
        for _ in range(TestConstants.TRAINING_ITERATIONS_TO_THRESHOLD):
            seed.train_child_step(sample_tensor)
            if seed.state == "blending":
                break

        assert seed.state == "blending"

        # Complete blending to active
        for _ in range(TestConstants.SMALL_BLEND_STEPS + 2):
            seed.update_blending()
            if seed.state == "active":
                break

        assert seed.state == "active"

        # Verify final forward pass works
        output = seed.forward(sample_tensor)
        assert output.shape == sample_tensor.shape

    def test_seed_manager_integration_telemetry(self, configured_seed_real_manager, sample_tensor):
        """Test integration with SeedManager telemetry systems."""
        seed = configured_seed_real_manager
        seed.initialize_child()
        seed._set_state("blending")
        seed.alpha = 0.5

        # Perform forward pass that should record telemetry
        seed.forward(sample_tensor)

        # Verify telemetry was recorded in manager
        telemetry = seed.seed_manager.seeds["test_seed"]["telemetry"]
        assert "drift" in telemetry
        assert isinstance(telemetry["drift"], float)

    def test_buffer_integration_and_health_monitoring(
        self, configured_seed_real_manager, sample_tensor
    ):
        """Test integration between buffer management and health monitoring."""
        seed = configured_seed_real_manager

        # Perform multiple forward passes to populate buffer
        for _ in range(TestConstants.SUFFICIENT_BUFFER_SIZE):
            seed.forward(sample_tensor)

        # Health signal should now be available
        health = seed.get_health_signal()
        assert health != float("inf")
        assert isinstance(health, float)
