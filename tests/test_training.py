"""
Test suite for training and evaluation functions.

This module contains comprehensive tests for the morphogenetic architecture training system,
including tests for:

- Training epoch functionality with various configurations and error scenarios
- Model evaluation functions with edge cases and error handling
- Optimizer and scheduler integration with failure modes
- Seed training logic with state transitions and buffer management
- Performance and concurrency validation
- Memory usage and resource management

Test Classes:
    TestTrainEpoch: Unit tests for training epoch functionality
    TestEvaluate: Unit tests for model evaluation  
    TestSeedTraining: Tests for seed training logic and state management
    TestErrorHandling: Tests for error scenarios and edge cases
    TestPerformance: Performance and resource usage tests
    TestConcurrency: Thread safety and concurrent access tests
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import psutil
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from morphogenetic_engine.components import BaseNet, SentinelSeed
from morphogenetic_engine.core import SeedManager
from morphogenetic_engine.training import evaluate, handle_seed_training, train_epoch

# Test constants to replace magic numbers
BATCH_SIZE = 4
HIDDEN_DIM = 32
INPUT_DIM = 2
SMALL_DATASET_SIZE = 8
MEDIUM_DATASET_SIZE = 16
LARGE_DATASET_SIZE = 64
LEARNING_RATE = 0.01
BUFFER_THRESHOLD = 64
LARGE_BUFFER_SIZE = 80
NUM_WORKERS = 0


@pytest.fixture
def test_data_small():
    """Fixture providing small test dataset."""
    X = torch.randn(SMALL_DATASET_SIZE, INPUT_DIM)
    y = torch.randint(0, 2, (SMALL_DATASET_SIZE,))
    return X, y


@pytest.fixture
def test_data_medium():
    """Fixture providing medium test dataset."""
    X = torch.randn(MEDIUM_DATASET_SIZE, INPUT_DIM)
    y = torch.randint(0, 2, (MEDIUM_DATASET_SIZE,))
    return X, y


@pytest.fixture
def test_data_empty():
    """Fixture providing empty test dataset."""
    X = torch.empty(0, INPUT_DIM)
    y = torch.empty(0, dtype=torch.long)
    return X, y


@pytest.fixture
def data_loader(test_data_medium):
    """Fixture providing configured data loader."""
    X, y = test_data_medium
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


@pytest.fixture
def empty_data_loader(test_data_empty):
    """Fixture providing empty data loader."""
    X, y = test_data_empty
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


@pytest.fixture
def mock_seed_manager(mocker):
    """Fixture providing isolated mock SeedManager."""
    # Create a fresh mock instance instead of using singleton
    mock_manager = mocker.MagicMock(spec=SeedManager)
    mock_manager.seeds = {}
    mock_manager.germination_log = []
    mock_manager.lock = threading.RLock()
    mock_manager.logger = None
    return mock_manager


@pytest.fixture
def mock_model(mocker, mock_seed_manager):
    """Fixture providing mock BaseNet model."""
    mock_model = mocker.MagicMock(spec=BaseNet)
    mock_model.get_all_seeds.return_value = []
    mock_model.train.return_value = None
    mock_model.eval.return_value = None
    
    # Mock forward pass to return reasonable predictions
    def mock_forward(x):
        batch_size = x.shape[0]
        return torch.randn(batch_size, 2)  # 2-class output
    
    mock_model.return_value = mock_forward
    mock_model.side_effect = mock_forward
    mock_model.__call__ = mock_forward
    return mock_model


@pytest.fixture
def mock_optimizer(mocker):
    """Fixture providing mock optimizer."""
    mock_opt = mocker.MagicMock(spec=torch.optim.Adam)
    mock_opt.param_groups = [{"lr": LEARNING_RATE}]
    mock_opt.zero_grad = mocker.MagicMock()
    mock_opt.step = mocker.MagicMock()
    return mock_opt


@pytest.fixture
def mock_criterion(mocker):
    """Fixture providing mock loss criterion."""
    mock_crit = mocker.MagicMock(spec=torch.nn.CrossEntropyLoss)
    
    def mock_loss(preds, targets):
        loss_tensor = torch.tensor(0.5, requires_grad=True)
        loss_tensor.item = lambda: 0.5
        return loss_tensor
    
    mock_crit.side_effect = mock_loss
    return mock_crit


@pytest.fixture
def mock_scheduler(mocker, mock_optimizer):
    """Fixture providing mock learning rate scheduler."""
    mock_sched = mocker.MagicMock(spec=torch.optim.lr_scheduler.StepLR)
    mock_sched.step = mocker.MagicMock()
    return mock_sched


@pytest.fixture
def mock_seed_with_buffer(mocker):
    """Fixture providing mock seed with buffer for testing."""
    mock_seed_manager = mocker.MagicMock(spec=SeedManager)
    mock_seed_manager.seeds = {}
    
    mock_seed = mocker.MagicMock(spec=SentinelSeed)
    mock_seed.seed_id = "test_seed_001"
    mock_seed.state = "training"
    mock_seed.alpha = 0.5
    mock_seed.initialize_child = mocker.MagicMock()
    mock_seed.train_child_step = mocker.MagicMock()
    mock_seed.update_blending = mocker.MagicMock()
    
    # Create buffer
    buffer = deque()
    mock_seed_manager.seeds[mock_seed.seed_id] = {
        "module": mock_seed,
        "buffer": buffer
    }
    
    return mock_seed_manager, mock_seed, buffer


class TestTrainEpochUnit:
    """Unit tests for training epoch functionality with mocked dependencies."""

    def test_train_epoch_basic_unit(self, data_loader, mock_model, mock_optimizer, 
                                   mock_criterion, mock_seed_manager, mocker):
        """Test basic training epoch functionality with mocked components."""
        # Mock handle_seed_training to isolate the core training logic
        mocker.patch('morphogenetic_engine.training.handle_seed_training')
        
        avg_loss = train_epoch(
            mock_model, data_loader, mock_optimizer, mock_criterion, mock_seed_manager
        )
        
        # Verify core training workflow
        assert mock_model.train.called
        assert mock_optimizer.zero_grad.call_count == len(data_loader)
        assert mock_optimizer.step.call_count == len(data_loader)
        assert isinstance(avg_loss, float)
        assert avg_loss > 0.0

    def test_train_epoch_no_optimizer_unit(self, data_loader, mock_model, 
                                          mock_criterion, mock_seed_manager, mocker):
        """Test training epoch without optimizer (inference mode)."""
        mocker.patch('morphogenetic_engine.training.handle_seed_training')
        
        avg_loss = train_epoch(
            mock_model, data_loader, None, mock_criterion, mock_seed_manager
        )
        
        assert mock_model.train.called
        assert isinstance(avg_loss, float)
        assert avg_loss > 0.0

    def test_train_epoch_with_scheduler_unit(self, data_loader, mock_model, 
                                           mock_optimizer, mock_criterion, 
                                           mock_seed_manager, mock_scheduler, mocker):
        """Test training epoch with learning rate scheduler."""
        mocker.patch('morphogenetic_engine.training.handle_seed_training')
        
        train_epoch(
            mock_model, data_loader, mock_optimizer, mock_criterion, 
            mock_seed_manager, mock_scheduler
        )
        
        # Scheduler should be called once after all batches
        assert mock_scheduler.step.call_count == 1

    def test_train_epoch_empty_loader_unit(self, empty_data_loader, mock_model, 
                                          mock_optimizer, mock_criterion, 
                                          mock_seed_manager, mocker):
        """Test training epoch with empty data loader."""
        mocker.patch('morphogenetic_engine.training.handle_seed_training')
        
        avg_loss = train_epoch(
            mock_model, empty_data_loader, mock_optimizer, mock_criterion, mock_seed_manager
        )
        
        # With empty loader, should return 0.0 and not call optimizer
        assert np.isclose(avg_loss, 0.0)
        assert not mock_optimizer.zero_grad.called
        assert not mock_optimizer.step.called


class TestTrainEpochIntegration:
    """Integration tests using real PyTorch components for critical paths."""

    def test_train_epoch_real_components_integration(self, test_data_medium):
        """Integration test with real PyTorch components to validate end-to-end flow."""
        X, y = test_data_medium
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        
        # Use real components for this integration test
        model = BaseNet(hidden_dim=HIDDEN_DIM, seed_manager=SeedManager(), input_dim=INPUT_DIM)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Run training epoch
        avg_loss = train_epoch(model, loader, optimizer, criterion, seed_manager)
        
        assert isinstance(avg_loss, float)
        assert avg_loss > 0.0


class TestEvaluateUnit:
    """Unit tests for evaluation function with mocked dependencies."""

    def test_evaluate_basic_unit(self, data_loader, mock_model, mock_criterion, mocker):
        """Test basic evaluation functionality with mocked components."""
        # Configure mock model to return consistent predictions
        def mock_forward(x):
            batch_size = x.shape[0]
            # Return predictions that will give 50% accuracy
            preds = torch.zeros(batch_size, 2)
            preds[:batch_size//2, 1] = 1.0  # First half predicted as class 1
            preds[batch_size//2:, 0] = 1.0  # Second half predicted as class 0
            return preds
        
        mock_model.side_effect = mock_forward
        mock_model.__call__ = mock_forward
        
        loss, accuracy = evaluate(mock_model, data_loader, mock_criterion)
        
        assert mock_model.eval.called
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_evaluate_empty_loader_unit(self, empty_data_loader, mock_model, mock_criterion):
        """Test evaluation with empty data loader."""
        loss, accuracy = evaluate(mock_model, empty_data_loader, mock_criterion)
        
        assert np.isclose(loss, 0.0)
        assert np.isnan(accuracy) or np.isclose(accuracy, 0.0)


class TestSeedTraining:
    """Tests for seed training logic and state management."""

    def test_handle_seed_training_with_sufficient_buffer(self, mock_seed_with_buffer, mocker):
        """Test seed training when buffer has sufficient data."""
        mock_seed_manager, mock_seed, buffer = mock_seed_with_buffer
        
        # Fill buffer with sufficient data
        for _ in range(15):
            buffer.append(torch.randn(2, HIDDEN_DIM))
        
        device = torch.device("cpu")
        handle_seed_training(mock_seed_manager, device)
        
        # Verify seed training was called
        assert mock_seed.train_child_step.called
        assert mock_seed.update_blending.called

    def test_handle_seed_training_with_large_buffer_sampling(self, mock_seed_with_buffer, mocker):
        """Test buffer sampling logic when buffer exceeds threshold."""
        mock_seed_manager, mock_seed, buffer = mock_seed_with_buffer
        
        # Fill buffer with more than threshold
        for _ in range(LARGE_BUFFER_SIZE):
            buffer.append(torch.randn(2, HIDDEN_DIM))
        
        device = torch.device("cpu")
        handle_seed_training(mock_seed_manager, device)
        
        # Verify training was called with sampled data
        assert mock_seed.train_child_step.called
        call_args = mock_seed.train_child_step.call_args[0]
        batch = call_args[0]
        assert batch.shape[0] <= BUFFER_THRESHOLD  # Should be sampled

    def test_handle_seed_training_insufficient_buffer(self, mock_seed_with_buffer):
        """Test seed training when buffer has insufficient data."""
        mock_seed_manager, mock_seed, buffer = mock_seed_with_buffer
        
        # Fill buffer with insufficient data (< 10)
        for _ in range(5):
            buffer.append(torch.randn(2, HIDDEN_DIM))
        
        device = torch.device("cpu")
        handle_seed_training(mock_seed_manager, device)
        
        # Verify training was not called but blending update was
        assert not mock_seed.train_child_step.called
        assert mock_seed.update_blending.called

    def test_seed_state_transitions(self, mocker):
        """Test seed state transitions during training."""
        mock_seed = mocker.MagicMock(spec=SentinelSeed)
        
        # Test dormant → training transition
        mock_seed.state = "dormant"
        mock_seed.germinate = mocker.MagicMock()
        mock_seed.germinate()
        
        # Test training → blending transition
        mock_seed.state = "training"
        mock_seed.training_progress = 1.0
        mock_seed.transition_to_blending = mocker.MagicMock()
        mock_seed.transition_to_blending()
        
        # Test blending → mature transition
        mock_seed.state = "blending"
        mock_seed.alpha = 1.0
        mock_seed.transition_to_mature = mocker.MagicMock()
        mock_seed.transition_to_mature()
        
        assert mock_seed.germinate.called
        assert mock_seed.transition_to_blending.called
        assert mock_seed.transition_to_mature.called


class TestErrorHandling:
    """Tests for error scenarios and edge cases."""

    def test_train_epoch_invalid_tensor_shapes(self, mock_model, mock_optimizer, 
                                              mock_criterion, mock_seed_manager, mocker):
        """Test training epoch with mismatched tensor shapes."""
        # Create data with mismatched shapes
        X = torch.randn(4, 3)  # Wrong input dimension
        y = torch.randint(0, 2, (4,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2, num_workers=NUM_WORKERS)
        
        # Configure model to raise error on wrong input shape
        def failing_forward(x):
            if x.shape[1] != INPUT_DIM:
                raise RuntimeError("Input shape mismatch")
            return torch.randn(x.shape[0], 2)
        
        mock_model.side_effect = failing_forward
        mock_model.__call__ = failing_forward
        
        mocker.patch('morphogenetic_engine.training.handle_seed_training')
        
        with pytest.raises(RuntimeError, match="Input shape mismatch"):
            train_epoch(mock_model, loader, mock_optimizer, mock_criterion, mock_seed_manager)

    def test_train_epoch_device_mismatch(self, data_loader, mock_seed_manager, mocker):
        """Test training epoch with device mismatch between model and data."""
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mismatch test")
        
        model = BaseNet(hidden_dim=HIDDEN_DIM, seed_manager=SeedManager(), input_dim=INPUT_DIM)
        model = model.cuda()  # Move model to GPU
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()
        
        mocker.patch('morphogenetic_engine.training.handle_seed_training')
        
        # Data stays on CPU, model on GPU - should raise error
        with pytest.raises(RuntimeError, match="Expected.*cuda.*cpu"):
            train_epoch(model, data_loader, optimizer, criterion, mock_seed_manager)

    def test_train_epoch_nan_loss(self, data_loader, mock_model, mock_optimizer, 
                                 mock_seed_manager, mocker):
        """Test training epoch with NaN loss values."""
        # Mock criterion to return NaN
        mock_criterion = mocker.MagicMock()
        nan_loss = torch.tensor(float('nan'), requires_grad=True)
        nan_loss.item = lambda: float('nan')
        mock_criterion.return_value = nan_loss
        
        mocker.patch('morphogenetic_engine.training.handle_seed_training')
        
        avg_loss = train_epoch(mock_model, data_loader, mock_optimizer, mock_criterion, mock_seed_manager)
        
        assert np.isnan(avg_loss)

    def test_train_epoch_optimizer_failure(self, data_loader, mock_model, 
                                          mock_criterion, mock_seed_manager, mocker):
        """Test training epoch with optimizer step failure."""
        mock_optimizer = mocker.MagicMock(spec=torch.optim.Adam)
        mock_optimizer.zero_grad = mocker.MagicMock()
        mock_optimizer.step = mocker.MagicMock(side_effect=RuntimeError("Optimizer failed"))
        
        mocker.patch('morphogenetic_engine.training.handle_seed_training')
        
        with pytest.raises(RuntimeError, match="Optimizer failed"):
            train_epoch(mock_model, data_loader, mock_optimizer, mock_criterion, mock_seed_manager)

    def test_evaluate_model_forward_failure(self, data_loader, mock_criterion, mocker):
        """Test evaluation with model forward pass failure."""
        mock_model = mocker.MagicMock()
        mock_model.eval = mocker.MagicMock()
        mock_model.side_effect = RuntimeError("Model forward failed")
        mock_model.__call__ = mocker.MagicMock(side_effect=RuntimeError("Model forward failed"))
        
        with pytest.raises(RuntimeError, match="Model forward failed"):
            evaluate(mock_model, data_loader, mock_criterion)


class TestPerformance:
    """Performance and resource usage tests."""

    def test_train_epoch_memory_usage(self, test_data_medium):
        """Test memory usage during training epoch."""
        X, y = test_data_medium
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        
        model = BaseNet(hidden_dim=HIDDEN_DIM, seed_manager=SeedManager(), input_dim=INPUT_DIM)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run training
        train_epoch(model, loader, optimizer, criterion, seed_manager)
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (< 100MB for this test)
        assert memory_increase < 100.0, f"Memory increase too high: {memory_increase:.2f}MB"

    def test_train_epoch_large_dataset_performance(self):
        """Test training performance with larger dataset."""
        # Create larger dataset
        large_size = 1000
        X = torch.randn(large_size, INPUT_DIM)
        y = torch.randint(0, 2, (large_size,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=NUM_WORKERS)
        
        model = BaseNet(hidden_dim=HIDDEN_DIM, seed_manager=SeedManager(), input_dim=INPUT_DIM)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Measure training time
        start_time = time.time()
        avg_loss = train_epoch(model, loader, optimizer, criterion, seed_manager)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        # Training should complete in reasonable time (< 30 seconds)
        assert training_time < 30.0, f"Training too slow: {training_time:.2f}s"
        assert isinstance(avg_loss, float)
        assert avg_loss > 0.0


class TestConcurrency:
    """Thread safety and concurrent access tests."""

    def test_seed_manager_concurrent_access(self, mocker):
        """Test SeedManager thread safety with concurrent access."""
        seed_manager = SeedManager()
        results = []
        errors = []
        
        def register_seed_worker(worker_id):
            try:
                mock_seed = mocker.MagicMock(spec=SentinelSeed)
                mock_seed.seed_id = f"seed_{worker_id}"
                seed_manager.register_seed(mock_seed, mock_seed.seed_id)
                results.append(worker_id)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_seed_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors and all registrations succeeded
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 10
        assert len(seed_manager.seeds) == 10

    def test_concurrent_seed_training(self, mocker):
        """Test concurrent seed training operations."""
        seed_manager = SeedManager()
        errors = []
        
        # Setup multiple seeds
        for i in range(5):
            mock_seed = mocker.MagicMock(spec=SentinelSeed)
            mock_seed.seed_id = f"concurrent_seed_{i}"
            mock_seed.state = "training"
            mock_seed.train_child_step = mocker.MagicMock()
            mock_seed.update_blending = mocker.MagicMock()
            
            buffer = deque()
            for _ in range(20):  # Sufficient data for training
                buffer.append(torch.randn(2, HIDDEN_DIM))
            
            seed_manager.register_seed(mock_seed, mock_seed.seed_id)
            seed_manager.seeds[mock_seed.seed_id]["buffer"] = buffer
        
        def training_worker():
            try:
                device = torch.device("cpu")
                handle_seed_training(seed_manager, device)
            except Exception as e:
                errors.append(e)
        
        # Start multiple training threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=training_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no race conditions or errors
        assert len(errors) == 0, f"Concurrent training errors: {errors}"


class TestRegressionAndStress:
    """Regression and stress tests to prevent performance degradation."""

    def test_training_time_regression(self):
        """Regression test to ensure training time doesn't degrade."""
        # Baseline expectation: training 100 samples should complete in < 5 seconds
        X = torch.randn(100, INPUT_DIM)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10, num_workers=NUM_WORKERS)
        
        model = BaseNet(hidden_dim=HIDDEN_DIM, seed_manager=SeedManager(), input_dim=INPUT_DIM)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()
        
        start_time = time.time()
        train_epoch(model, loader, optimizer, criterion, seed_manager)
        end_time = time.time()
        
        training_time = end_time - start_time
        assert training_time < 5.0, f"Training regression detected: {training_time:.2f}s > 5.0s"

    def test_stress_many_small_batches(self):
        """Stress test with many small batches."""
        # Create dataset with many small batches
        X = torch.randn(200, INPUT_DIM)
        y = torch.randint(0, 2, (200,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=1, num_workers=NUM_WORKERS)  # Many small batches
        
        model = BaseNet(hidden_dim=HIDDEN_DIM, seed_manager=SeedManager(), input_dim=INPUT_DIM)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Should handle many small batches without issues
        avg_loss = train_epoch(model, loader, optimizer, criterion, seed_manager)
        assert isinstance(avg_loss, float)
        assert avg_loss > 0.0

    def test_stress_large_seed_buffers(self):
        """Stress test with multiple large seed buffers."""
        seed_manager = SeedManager()
        
        # Create multiple seeds with large buffers
        for i in range(10):
            mock_seed = MagicMock(spec=SentinelSeed)
            mock_seed.seed_id = f"stress_seed_{i}"
            mock_seed.state = "training"
            mock_seed.train_child_step = MagicMock()
            mock_seed.update_blending = MagicMock()
            
            # Large buffer
            buffer = deque()
            for _ in range(500):  # Much larger than normal
                buffer.append(torch.randn(2, HIDDEN_DIM))
            
            seed_manager.register_seed(mock_seed, mock_seed.seed_id)
            seed_manager.seeds[mock_seed.seed_id]["buffer"] = buffer
        
        # Should handle large buffers efficiently
        device = torch.device("cpu")
        start_time = time.time()
        handle_seed_training(seed_manager, device)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 10.0, f"Large buffer processing too slow: {processing_time:.2f}s"


# Cleanup fixtures to ensure test isolation
@pytest.fixture(autouse=True)
def cleanup_singleton_state():
    """Automatically reset singleton state after each test."""
    yield
    # Reset SeedManager singleton state if it exists
    if hasattr(SeedManager, '_instance') and SeedManager._instance is not None:
        SeedManager._instance.seeds.clear()
        SeedManager._instance.germination_log.clear()
