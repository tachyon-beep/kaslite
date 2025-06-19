"""
Tests for CIFAR-10 dataset support in morphogenetic engine.

This module tests CIFAR-10 integration including:
- Dataset loading and preprocessing
- CLI argument handling
- End-to-end experiment execution
- Error handling for missing dependencies
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from morphogenetic_engine.datasets import create_cifar10, CIFAR10_AVAILABLE
from morphogenetic_engine.cli.arguments import parse_experiment_arguments
from morphogenetic_engine.experiment import build_model_and_agents


@pytest.mark.slow
class TestCIFAR10Dataset:
    """Test CIFAR-10 dataset loading and preprocessing."""

    @pytest.mark.skipif(not CIFAR10_AVAILABLE, reason="torchvision not available")
    def test_create_cifar10_basic_functionality(self):
        """Test basic CIFAR-10 dataset loading."""
        # Load a small subset to avoid long test times
        X, y = create_cifar10(data_dir="data/test_cifar", train=True)
        
        # Check shapes
        assert X.shape == (50000, 3072), f"Expected shape (50000, 3072), got {X.shape}"
        assert y.shape == (50000,), f"Expected shape (50000,), got {y.shape}"
        
        # Check data types
        assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"
        assert y.dtype == np.int64, f"Expected int64, got {y.dtype}"
        
        # Check value ranges
        assert 0.0 <= X.min() <= X.max() <= 1.0, f"X values should be in [0,1], got [{X.min()}, {X.max()}]"
        
        # Check labels
        unique_labels = sorted(np.unique(y))
        expected_labels = list(range(10))
        assert unique_labels == expected_labels, f"Expected labels {expected_labels}, got {unique_labels}"
        
        # Check class balance (CIFAR-10 training set has 5000 samples per class)
        label_counts = np.bincount(y)
        assert len(label_counts) == 10, f"Expected 10 classes, got {len(label_counts)}"
        assert all(count == 5000 for count in label_counts), f"Expected 5000 samples per class, got {label_counts.tolist()}"

    @pytest.mark.skipif(not CIFAR10_AVAILABLE, reason="torchvision not available")
    def test_create_cifar10_test_set(self):
        """Test loading CIFAR-10 test set."""
        X, y = create_cifar10(data_dir="data/test_cifar", train=False)
        
        # Test set should have 10000 samples
        assert X.shape == (10000, 3072), f"Expected test set shape (10000, 3072), got {X.shape}"
        assert y.shape == (10000,), f"Expected test set shape (10000,), got {y.shape}"
        
        # Should still have all 10 classes
        unique_labels = sorted(np.unique(y))
        assert unique_labels == list(range(10)), f"Test set should have all 10 classes"

    def test_create_cifar10_missing_torchvision(self):
        """Test error handling when torchvision is not available."""
        with patch('morphogenetic_engine.datasets.CIFAR10_AVAILABLE', False):
            with pytest.raises(ImportError, match="torchvision is required for CIFAR-10 support"):
                create_cifar10()

    @pytest.mark.skipif(not CIFAR10_AVAILABLE, reason="torchvision not available")
    def test_create_cifar10_data_consistency(self):
        """Test that repeated calls return consistent data."""
        X1, y1 = create_cifar10(data_dir="data/test_cifar", train=True)
        X2, y2 = create_cifar10(data_dir="data/test_cifar", train=True)
        
        # Should be identical (deterministic)
        np.testing.assert_array_equal(X1, X2, "Repeated calls should return identical data")
        np.testing.assert_array_equal(y1, y2, "Repeated calls should return identical labels")


@pytest.mark.slow
class TestCIFAR10CLIIntegration:
    """Test CIFAR-10 integration with CLI arguments."""

    def test_cifar10_in_problem_type_choices(self):
        """Test that cifar10 is included in CLI choices."""
        # Parse arguments with cifar10
        with patch('sys.argv', ['test', '--problem_type', 'cifar10']):
            args = parse_experiment_arguments()
            assert args.problem_type == 'cifar10'

    def test_cifar10_specific_arguments(self):
        """Test CLI arguments specific to CIFAR-10."""
        test_args = [
            'test', 
            '--problem_type', 'cifar10',
            '--input_dim', '3072',
            '--batch_size', '128',
            '--hidden_dim', '512'
        ]
        
        with patch('sys.argv', test_args):
            args = parse_experiment_arguments()
            assert args.problem_type == 'cifar10'
            assert args.input_dim == 3072
            assert args.batch_size == 128
            assert args.hidden_dim == 512


@pytest.mark.slow
class TestCIFAR10ModelIntegration:
    """Test CIFAR-10 integration with model building."""

    def test_build_model_and_agents_cifar10(self, mocker):
        """Test model building for CIFAR-10."""
        # Mock arguments for CIFAR-10
        mock_args = Mock()
        mock_args.problem_type = 'cifar10'
        mock_args.input_dim = 3072
        mock_args.hidden_dim = 256
        mock_args.num_layers = 8
        mock_args.seeds_per_layer = 1
        mock_args.blend_steps = 30
        mock_args.shadow_lr = 0.001
        mock_args.progress_thresh = 0.6
        mock_args.drift_warn = 0.12
        mock_args.acc_threshold = 0.95
        
        device = torch.device('cpu')
        
        model, seed_manager, loss_fn, kasmina = build_model_and_agents(mock_args, device)
        
        # Check model configuration for CIFAR-10
        assert model.input_dim == 3072, "Input dimension should be 3072 for flattened CIFAR-10"
        assert model.output_dim == 10, "Output dimension should be 10 for CIFAR-10 classes"
        assert model.hidden_dim == 256, "Hidden dimension should match args"
        
        # Check loss function
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss), "Should use CrossEntropyLoss for multi-class"
        
        # Test forward pass with correct input shape
        batch_size = 4
        x = torch.randn(batch_size, 3072)
        output = model(x)
        assert output.shape == (batch_size, 10), f"Output shape should be ({batch_size}, 10), got {output.shape}"

    def test_build_model_synthetic_datasets(self, mocker):
        """Test that synthetic datasets still use binary classification."""
        mock_args = Mock()
        mock_args.problem_type = 'spirals'  # Non-CIFAR dataset
        mock_args.input_dim = 3
        mock_args.hidden_dim = 128
        mock_args.num_layers = 8
        mock_args.seeds_per_layer = 1
        mock_args.blend_steps = 30
        mock_args.shadow_lr = 0.001
        mock_args.progress_thresh = 0.6
        mock_args.drift_warn = 0.12
        mock_args.acc_threshold = 0.95
        
        device = torch.device('cpu')
        
        model, seed_manager, loss_fn, kasmina = build_model_and_agents(mock_args, device)
        
        # Should default to binary classification
        assert model.output_dim == 2, "Synthetic datasets should use binary classification"


@pytest.mark.slow
class TestCIFAR10EndToEnd:
    """End-to-end integration tests for CIFAR-10."""

    @pytest.mark.skipif(not CIFAR10_AVAILABLE, reason="torchvision not available")
    def test_cifar10_dataloaders_creation(self):
        """Test creating DataLoaders for CIFAR-10."""
        from morphogenetic_engine.runners import get_dataloaders
        
        # Mock arguments
        mock_args = Mock()
        mock_args.problem_type = 'cifar10'
        mock_args.n_samples = 2000  # This is ignored for CIFAR-10
        mock_args.train_frac = 0.8
        mock_args.batch_size = 32
        
        train_loader, val_loader = get_dataloaders(mock_args)
        
        # Check that input_dim was set correctly
        assert mock_args.input_dim == 3072, "input_dim should be set to 3072 for CIFAR-10"
        
        # Check DataLoader properties
        assert hasattr(train_loader, '__iter__'), "train_loader should be iterable"
        assert hasattr(val_loader, '__iter__'), "val_loader should be iterable"
        
        # Check batch shapes
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        train_x, train_y = train_batch
        val_x, val_y = val_batch
        
        assert train_x.shape[1] == 3072, f"Training batch feature dimension should be 3072, got {train_x.shape[1]}"
        assert val_x.shape[1] == 3072, f"Validation batch feature dimension should be 3072, got {val_x.shape[1]}"
        
        # Check label range
        assert 0 <= train_y.min() <= train_y.max() <= 9, f"Training labels should be in [0,9], got [{train_y.min()}, {train_y.max()}]"
        assert 0 <= val_y.min() <= val_y.max() <= 9, f"Validation labels should be in [0,9], got [{val_y.min()}, {val_y.max()}]"

    def test_cifar10_experiment_arguments_parsing(self):
        """Test parsing complete CIFAR-10 experiment arguments."""
        test_args = [
            'test',
            '--problem_type', 'cifar10',
            '--hidden_dim', '512',
            '--batch_size', '128',
            '--warm_up_epochs', '10',
            '--adaptation_epochs', '20',
            '--lr', '0.001',
            '--device', 'cpu'
        ]
        
        with patch('sys.argv', test_args):
            args = parse_experiment_arguments()
            
            # Verify all CIFAR-10 specific settings
            assert args.problem_type == 'cifar10'
            assert args.hidden_dim == 512
            assert args.batch_size == 128
            assert args.warm_up_epochs == 10
            assert args.adaptation_epochs == 20
            assert args.lr == 0.001
            assert args.device == 'cpu'


@pytest.mark.slow
class TestCIFAR10ErrorHandling:
    """Test error handling for CIFAR-10 related issues."""

    def test_invalid_data_dir(self):
        """Test handling of invalid data directory."""
        if not CIFAR10_AVAILABLE:
            pytest.skip("torchvision not available")
            
        # This should still work as torchvision will create the directory
        X, y = create_cifar10(data_dir="/tmp/test_cifar_invalid", train=True)
        assert X.shape[0] > 0, "Should successfully load data even with non-existent directory"

    def test_cifar10_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        with patch('morphogenetic_engine.datasets.CIFAR10_AVAILABLE', False):
            with pytest.raises(ImportError, match="torchvision is required"):
                create_cifar10()


@pytest.mark.slow
class TestCIFAR10Performance:
    """Performance and memory tests for CIFAR-10."""

    @pytest.mark.skipif(not CIFAR10_AVAILABLE, reason="torchvision not available")
    def test_cifar10_memory_usage(self):
        """Test that CIFAR-10 loading doesn't consume excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        X, y = create_cifar10(data_dir="data/test_cifar", train=True)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # CIFAR-10 (50k images × 3072 float32) should be ~600MB, allow some overhead
        assert memory_increase < 1000, f"Memory increase {memory_increase:.1f}MB seems excessive for CIFAR-10"

    @pytest.mark.skipif(not CIFAR10_AVAILABLE, reason="torchvision not available")
    def test_cifar10_loading_time(self):
        """Test that CIFAR-10 loads in reasonable time."""
        import time
        
        start_time = time.time()
        X, y = create_cifar10(data_dir="data/test_cifar", train=True)
        load_time = time.time() - start_time
        
        # Should load in under 30 seconds even on slow systems
        assert load_time < 30, f"CIFAR-10 loading took {load_time:.1f}s, which seems too slow"


@pytest.mark.integration
@pytest.mark.slow
class TestCIFAR10WorkflowIntegration:
    """Integration tests for complete CIFAR-10 workflows."""

    @pytest.mark.skipif(not CIFAR10_AVAILABLE, reason="torchvision not available")
    def test_complete_cifar10_experiment_setup(self):
        """Test complete setup of a CIFAR-10 experiment."""
        from morphogenetic_engine.runners import get_dataloaders
        from morphogenetic_engine.experiment import build_model_and_agents
        
        # Mock complete experiment arguments
        mock_args = Mock()
        mock_args.problem_type = 'cifar10'
        mock_args.train_frac = 0.8
        mock_args.batch_size = 64
        mock_args.hidden_dim = 256
        mock_args.num_layers = 4  # Smaller for testing
        mock_args.seeds_per_layer = 1
        mock_args.blend_steps = 30
        mock_args.shadow_lr = 0.001
        mock_args.progress_thresh = 0.6
        mock_args.drift_warn = 0.12
        mock_args.acc_threshold = 0.95
        
        device = torch.device('cpu')
        
        # Test data loading
        train_loader, val_loader = get_dataloaders(mock_args)
        assert mock_args.input_dim == 3072, "input_dim should be set correctly"
        
        # Test model building
        model, seed_manager, loss_fn, kasmina = build_model_and_agents(mock_args, device)
        
        # Test forward pass
        train_batch = next(iter(train_loader))
        x, y = train_batch
        
        output = model(x)
        loss = loss_fn(output, y)
        
        assert output.shape == (x.shape[0], 10), "Output should have 10 classes"
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        # Test that all components are compatible
        assert model.input_dim == x.shape[1], "Model input_dim should match data"
        assert model.output_dim == 10, "Model should output 10 classes"
