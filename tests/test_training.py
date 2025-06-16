"""
Test suite for training and evaluation functions.

This module contains tests for the morphogenetic architecture training system,
including tests for:

- Training epoch functionality with different configurations
- Model evaluation functions
- Optimizer and scheduler integration
- Seed training during training epochs

Test Classes:
    TestTrainEpoch: Tests for training epoch functionality
    TestEvaluate: Tests for model evaluation
"""

from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from morphogenetic_engine.components import BaseNet
from morphogenetic_engine.core import SeedManager
from morphogenetic_engine.training import evaluate, train_epoch


class TestTrainEpoch:
    """Test suite for training epoch function."""

    def test_train_epoch_basic(self):
        """Test basic training epoch functionality."""
        # Create simple test data
        X = torch.randn(16, 2)
        y = torch.randint(0, 2, (16,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and components
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()

        # Test training epoch
        avg_loss = train_epoch(model, loader, optimizer, criterion, seed_manager)

        assert isinstance(avg_loss, float)
        assert avg_loss > 0.0

    def test_train_epoch_with_scheduler(self):
        """Test training epoch with learning rate scheduler."""
        # Create test data
        X = torch.randn(16, 2)
        y = torch.randint(0, 2, (16,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and components
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Train for one epoch
        train_epoch(model, loader, optimizer, criterion, seed_manager, scheduler)

        # Check that scheduler stepped
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_train_epoch_no_optimizer(self):
        """Test train_epoch with no optimizer."""
        # Create test data
        X = torch.randn(8, 2)
        y = torch.randint(0, 2, (8,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and components
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        criterion = torch.nn.CrossEntropyLoss()

        # Test with no optimizer
        avg_loss = train_epoch(model, loader, None, criterion, seed_manager)

        assert isinstance(avg_loss, float)
        assert avg_loss > 0.0

    def test_train_epoch_empty_loader(self):
        """Test train_epoch with empty loader."""
        # Create empty dataset
        X = torch.empty(0, 2)
        y = torch.empty(0, dtype=torch.long)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and components
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()

        # Test with empty loader
        avg_loss = train_epoch(model, loader, optimizer, criterion, seed_manager)

        assert np.isclose(avg_loss, 0.0)  # Should return 0.0 for empty loader

    def test_train_epoch_seed_training(self):
        """Test that seeds are trained during training epoch."""
        # Create test data
        X = torch.randn(16, 2)
        y = torch.randint(0, 2, (16,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and components
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()

        # Initialize a seed for training (get first seed from the new structure)
        first_seed = model.get_all_seeds()[0]
        first_seed.initialize_child()

        # Add some data to seed buffer
        buffer = seed_manager.seeds[first_seed.seed_id]["buffer"]
        for _ in range(15):
            buffer.append(torch.randn(2, 32))

        # Mock the train_child_step to verify it's called
        with patch.object(first_seed, "train_child_step") as mock_train:
            train_epoch(model, loader, optimizer, criterion, seed_manager)
            assert mock_train.call_count > 0

    def test_train_epoch_large_buffer_sampling(self):
        """Test that large seed buffers are properly sampled during training."""
        # Create test data
        X = torch.randn(16, 2)
        y = torch.randint(0, 2, (16,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and components
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        seed_manager = SeedManager()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()

        # Initialize a seed for training and fill buffer with many samples
        first_seed = model.get_all_seeds()[0]
        first_seed.initialize_child()
        buffer = seed_manager.seeds[first_seed.seed_id]["buffer"]

        # Add > 64 samples to trigger sampling logic (lines 97-98)
        for _ in range(80):
            buffer.append(torch.randn(2, 32))

        # Verify buffer has many samples
        assert len(buffer) > 64

        # Train epoch should use sampling logic for large buffers
        avg_loss = train_epoch(model, loader, optimizer, criterion, seed_manager)
        assert avg_loss >= 0.0  # Basic sanity check


class TestEvaluate:
    """Test suite for evaluation function."""

    def test_evaluate_basic(self):
        """Test basic evaluation functionality."""
        # Create test data
        X = torch.randn(16, 2)
        y = torch.randint(0, 2, (16,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and criterion
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        criterion = torch.nn.CrossEntropyLoss()

        # Evaluate
        loss, accuracy = evaluate(model, loader, criterion)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_evaluate_perfect_model(self):
        """Test evaluation with a perfect model."""
        # Create simple linearly separable data
        X = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
        y = torch.tensor([1, 0, 1, 0])
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2, num_workers=0)

        # Create a perfect linear model
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight[0, 0] = -1.0  # For class 0
            model.weight[0, 1] = 0.0
            model.weight[1, 0] = 1.0  # For class 1
            model.weight[1, 1] = 0.0
            model.bias[0] = 0.0
            model.bias[1] = 0.0

        criterion = torch.nn.CrossEntropyLoss()

        # Evaluate
        loss, accuracy = evaluate(model, loader, criterion)

        assert np.isclose(accuracy, 1.0)  # Perfect accuracy
        assert loss < 0.1  # Very low loss

    def test_evaluate_empty_loader(self):
        """Test evaluation with empty data loader."""
        # Create empty dataset
        X = torch.empty(0, 2)
        y = torch.empty(0, dtype=torch.long)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        # Create model and criterion
        model = BaseNet(hidden_dim=32, seed_manager=SeedManager(), input_dim=2)
        criterion = torch.nn.CrossEntropyLoss()

        # This should handle empty loader gracefully
        loss, accuracy = evaluate(model, loader, criterion)

        # With empty loader, loss should be 0 and accuracy undefined (NaN)
        assert np.isclose(loss, 0.0)
        assert np.isnan(accuracy) or np.isclose(accuracy, 0.0)
