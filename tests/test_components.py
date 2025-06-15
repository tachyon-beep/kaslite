"""Comprehensive tests for the components module."""
import os
import sys
import torch
import pytest
from unittest.mock import Mock, patch, MagicMock
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from morphogenetic_engine.components import SentinelSeed, BaseNet
from morphogenetic_engine.core import SeedManager


class TestSentinelSeed:
    """Test suite for SentinelSeed class."""
    
    def test_initialization(self):
        """Test SentinelSeed initialization."""
        seed = SentinelSeed(
            seed_id="test_seed",
            dim=64,
            blend_steps=25,
            shadow_lr=2e-3,
            progress_thresh=0.7,
            drift_warn=0.15
        )
        
        assert seed.seed_id == "test_seed"
        assert seed.dim == 64
        assert seed.blend_steps == 25
        assert seed.progress_thresh == pytest.approx(0.7)
        assert seed.alpha == pytest.approx(0.0)
        assert seed.state == "dormant"
        assert seed.training_progress == pytest.approx(0.0)
        assert seed.drift_warn == pytest.approx(0.15)
        
        # Check child network structure
        assert isinstance(seed.child, torch.nn.Sequential)
        assert len(seed.child) == 3  # Linear, ReLU, Linear
        
        # Check optimizer
        assert isinstance(seed.child_optim, torch.optim.Adam)
        assert isinstance(seed.child_loss, torch.nn.MSELoss)
        
    def test_set_state_transitions(self):
        """Test state transition handling."""
        seed = SentinelSeed("test_seed", dim=32)
        manager = seed.seed_manager
        manager.germination_log.clear()
        
        # Test transition to training
        seed._set_state("training")
        assert seed.state == "training"
        assert manager.seeds["test_seed"]["state"] == "training"
        assert manager.seeds["test_seed"]["status"] == "pending"
        
        # Test transition to active
        seed._set_state("active")
        assert seed.state == "active"
        assert manager.seeds["test_seed"]["state"] == "active"
        assert manager.seeds["test_seed"]["status"] == "active"
        
        # Check transitions were logged
        assert len(manager.germination_log) == 2
        
    def test_redundant_state_transition(self):
        """Test that redundant state transitions are ignored."""
        seed = SentinelSeed("test_seed", dim=32)
        manager = seed.seed_manager
        manager.germination_log.clear()
        
        seed._set_state("training")
        log_count_after_first = len(manager.germination_log)
        
        # Same state transition should be ignored
        seed._set_state("training")
        log_count_after_second = len(manager.germination_log)
        
        assert log_count_after_first == log_count_after_second
        
    def test_initialize_as_identity(self):
        """Test that child network is initialized as near-identity."""
        seed = SentinelSeed("test_seed", dim=32)
        
        # Check that parameters are initially zero
        for param in seed.child.parameters():
            assert torch.allclose(param, torch.zeros_like(param))
            assert not param.requires_grad  # Should be frozen initially
            
    def test_initialize_child(self):
        """Test proper child network initialization."""
        seed = SentinelSeed("test_seed", dim=32)
        
        seed.initialize_child()
        
        assert seed.state == "training"
        # Parameters should no longer be zero after Kaiming initialization
        weight_params = [p for p in seed.child.parameters() if p.dim() > 1]
        assert len(weight_params) > 0
        for param in weight_params:
            assert not torch.allclose(param, torch.zeros_like(param))
            assert param.requires_grad  # Should be trainable now
            
    def test_train_child_step_dormant_state(self):
        """Test that training step is ignored in dormant state."""
        seed = SentinelSeed("test_seed", dim=32)
        initial_progress = seed.training_progress
        
        inputs = torch.randn(4, 32)
        seed.train_child_step(inputs)
        
        # Progress should not change in dormant state
        assert seed.training_progress == initial_progress
        assert seed.state == "dormant"
        
    def test_train_child_step_empty_input(self):
        """Test training step with empty input."""
        seed = SentinelSeed("test_seed", dim=32)
        seed.initialize_child()
        
        empty_input = torch.empty(0, 32)
        initial_progress = seed.training_progress
        
        seed.train_child_step(empty_input)
        
        # Progress should not change with empty input
        assert seed.training_progress == initial_progress
        
    def test_train_child_step_training_state(self):
        """Test training step in training state."""
        seed = SentinelSeed("test_seed", dim=32, progress_thresh=0.5)
        seed.initialize_child()
        
        inputs = torch.randn(4, 32)
        initial_progress = seed.training_progress
        
        # Train for several steps
        for _ in range(60):  # Should exceed progress threshold
            seed.train_child_step(inputs)
            
        assert seed.training_progress > initial_progress
        assert seed.state == "blending"  # Should transition to blending
        assert seed.alpha == pytest.approx(0.0)  # Alpha starts at 0 in blending
        
    def test_update_blending(self):
        """Test blending update process."""
        seed = SentinelSeed("test_seed", dim=32, blend_steps=10)
        seed._set_state("blending")
        seed.alpha = 0.0
        
        # Update blending several times
        for _ in range(5):
            seed.update_blending()
            
        assert seed.alpha > 0.0
        assert seed.state == "blending"
        
        # Complete blending
        for _ in range(10):
            seed.update_blending()
            
        assert seed.alpha >= 0.99
        assert seed.state == "active"
        
    def test_update_blending_non_blending_state(self):
        """Test that blending update is ignored in non-blending states."""
        seed = SentinelSeed("test_seed", dim=32)
        initial_alpha = seed.alpha
        
        seed.update_blending()  # Should do nothing in dormant state
        
        assert seed.alpha == initial_alpha
        assert seed.state == "dormant"
        
    def test_forward_dormant_state(self):
        """Test forward pass in dormant state."""
        seed = SentinelSeed("test_seed", dim=32)
        x = torch.randn(4, 32)
        
        output = seed.forward(x)
        
        # Should return input unchanged in dormant state
        assert torch.equal(output, x)
        
        # Should append to buffer
        buffer = seed.seed_manager.seeds["test_seed"]["buffer"]
        assert len(buffer) == 1
        assert torch.equal(buffer[0], x.detach())
        
    def test_forward_training_state(self):
        """Test forward pass in training state."""
        seed = SentinelSeed("test_seed", dim=32)
        seed.initialize_child()
        x = torch.randn(4, 32)
        
        output = seed.forward(x)
        
        # Should return input unchanged in training state
        assert torch.equal(output, x)
        
    def test_forward_blending_state(self):
        """Test forward pass in blending state."""
        seed = SentinelSeed("test_seed", dim=32)
        seed.initialize_child()
        seed._set_state("blending")
        seed.alpha = 0.5
        
        x = torch.randn(4, 32)
        
        with patch.object(seed.seed_manager, 'record_drift') as mock_record:
            output = seed.forward(x)
            mock_record.assert_called_once()
            
        # Output should be blend of input and child output
        assert output.shape == x.shape
        assert not torch.equal(output, x)  # Should be different due to blending
        
    def test_forward_active_state(self):
        """Test forward pass in active state."""
        seed = SentinelSeed("test_seed", dim=32)
        seed.initialize_child()
        seed._set_state("active")
        
        x = torch.randn(4, 32)
        
        with patch.object(seed.seed_manager, 'record_drift') as mock_record:
            output = seed.forward(x)
            mock_record.assert_called_once()
            
        # Output should be input + child output (residual connection)
        assert output.shape == x.shape
        
    def test_drift_warning_only_in_blending(self):
        """Test that drift warnings only occur during blending state."""
        seed = SentinelSeed("test_seed", dim=32, drift_warn=0.01)  # Low threshold
        seed.initialize_child()
        
        x = torch.randn(4, 32)
        
        # Test active state - should not warn
        seed._set_state("active")
        with patch('morphogenetic_engine.components.logging.warning') as mock_warn:
            seed.forward(x)
            mock_warn.assert_not_called()
            
        # Test blending state - should warn if drift is high
        seed._set_state("blending")
        seed.alpha = 0.5
        
        # Create high drift by making child return very different values
        original_child = seed.child
        # Replace with a module that returns very different output
        seed.child = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU()
        )
        # Initialize to produce high values to create drift
        with torch.no_grad():
            first_layer = seed.child[0]
            first_layer.weight.data.fill_(10.0)  # type: ignore
            if first_layer.bias is not None:
                first_layer.bias.data.fill_(100.0)  # type: ignore
            
        try:
            with patch('morphogenetic_engine.components.logging.warning') as mock_warn:
                seed.forward(x)
                mock_warn.assert_called_once()
                args = mock_warn.call_args[0]
                assert "(blending)" in args[0]
        finally:
            # Restore original child
            seed.child = original_child
                
    def test_get_health_signal_insufficient_data(self):
        """Test health signal with insufficient buffer data."""
        seed = SentinelSeed("test_seed", dim=32)
        
        # With empty or small buffer, should return infinity
        health = seed.get_health_signal()
        assert health == float("inf")
        
        # Add some data but not enough
        buffer = seed.seed_manager.seeds["test_seed"]["buffer"]
        for _ in range(5):  # Less than 10 required
            buffer.append(torch.randn(2, 32))
            
        health = seed.get_health_signal()
        assert health == float("inf")
        
    def test_get_health_signal_sufficient_data(self):
        """Test health signal with sufficient buffer data."""
        seed = SentinelSeed("test_seed", dim=32)
        
        # Add sufficient data
        buffer = seed.seed_manager.seeds["test_seed"]["buffer"]
        for _ in range(15):
            buffer.append(torch.randn(2, 32))
            
        health = seed.get_health_signal()
        assert isinstance(health, float)
        assert health != float("inf")
        assert health >= 0.0

    def test_state_transitions_complete(self):
        """Test all state transitions including dormant."""
        seed = SentinelSeed("test_seed", dim=32)
        
        # Test transition to training
        seed._set_state("training")
        assert seed.state == "training"
        assert seed.seed_manager.seeds["test_seed"]["status"] == "pending"
        
        # Test transition to blending
        seed._set_state("blending")
        assert seed.state == "blending"
        assert seed.seed_manager.seeds["test_seed"]["status"] == "pending"
        
        # Test transition to active
        seed._set_state("active")
        assert seed.state == "active"
        assert seed.seed_manager.seeds["test_seed"]["status"] == "active"
        
        # Test transition back to dormant (this hits line 65)
        seed._set_state("dormant")
        assert seed.state == "dormant"
        assert seed.seed_manager.seeds["test_seed"]["status"] == "dormant"


class TestBaseNet:
    """Test suite for BaseNet class."""
    
    def test_initialization_default_params(self):
        """Test BaseNet initialization with default parameters."""
        net = BaseNet()
        
        # Check default hidden dimension
        assert net.fc1.out_features == 64
        assert net.fc2.in_features == 64
        
        # Check that all seeds are present
        for i in range(1, 9):
            assert hasattr(net, f'seed{i}')
            seed = getattr(net, f'seed{i}')
            assert isinstance(seed, SentinelSeed)
            assert seed.seed_id == f'seed{i}'
            assert seed.dim == 64
            
    def test_initialization_custom_params(self):
        """Test BaseNet initialization with custom parameters."""
        net = BaseNet(
            hidden_dim=128,
            blend_steps=20,
            shadow_lr=5e-4,
            progress_thresh=0.8,
            drift_warn=0.05
        )
        
        # Check custom hidden dimension
        assert net.fc1.out_features == 128
        
        # Check that seed parameters are passed correctly
        seed = net.seed1
        assert seed.dim == 128
        assert seed.blend_steps == 20
        assert seed.shadow_lr == pytest.approx(5e-4)
        assert seed.progress_thresh == pytest.approx(0.8)
        assert seed.drift_warn == pytest.approx(0.05)
        
    def test_freeze_backbone(self):
        """Test backbone freezing functionality."""
        net = BaseNet(hidden_dim=32)
        
        # Initialize all seeds to make their parameters trainable  
        for i in range(1, 9):  # seed1 through seed8
            getattr(net, f'seed{i}').initialize_child()
        
        # Now all parameters should be trainable
        for name, param in net.named_parameters():
            assert param.requires_grad, f"Parameter {name} should be trainable"
            
        # Freeze backbone
        net.freeze_backbone()
        
        # Check that non-seed parameters are frozen
        for name, param in net.named_parameters():
            if "seed" not in name:
                assert not param.requires_grad, f"Backbone parameter {name} should be frozen"
            else:
                assert param.requires_grad, f"Seed parameter {name} should remain trainable"
                # Seed parameters might be frozen (if not initialized) or trainable
                pass  # Don't check seed parameters as they have their own logic
                
    def test_forward_pass_shapes(self):
        """Test forward pass produces correct output shapes."""
        net = BaseNet(hidden_dim=64)
        
        # Test different batch sizes
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 2)  # Input dimension is 2
            output = net.forward(x)
            
            assert output.shape == (batch_size, 2)  # Output dimension is 2
            
    def test_forward_pass_deterministic(self):
        """Test that forward pass is deterministic given same input."""
        net = BaseNet(hidden_dim=32)
        
        # Set eval mode for deterministic behavior
        net.eval()
        
        x = torch.randn(4, 2)
        
        with torch.no_grad():
            output1 = net.forward(x)
            output2 = net.forward(x)
            
        assert torch.allclose(output1, output2)
        
    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        net = BaseNet(hidden_dim=32)
        x = torch.randn(4, 2, requires_grad=True)
        
        output = net.forward(x)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        
        # Check that non-frozen parameters have gradients
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                
    def test_seed_integration(self):
        """Test that seeds are properly integrated into forward pass."""
        net = BaseNet(hidden_dim=32)
        
        # Initialize one seed
        net.seed1.initialize_child()
        
        x = torch.randn(4, 2)
        
        # Check that seed buffers are populated during forward pass
        buffer_before = len(net.seed1.seed_manager.seeds["seed1"]["buffer"])
        
        net.forward(x)
        
        buffer_after = len(net.seed1.seed_manager.seeds["seed1"]["buffer"])
        
        # Buffer should have increased (seed is not active yet)
        assert buffer_after > buffer_before
        
    def test_all_seeds_different_instances(self):
        """Test that all seeds are separate instances."""
        net = BaseNet(hidden_dim=32)
        
        seeds = [getattr(net, f'seed{i}') for i in range(1, 9)]
        
        # All seeds should be different instances
        for i, seed1 in enumerate(seeds):
            for j, seed2 in enumerate(seeds):
                if i != j:
                    assert seed1 is not seed2
                    assert seed1.seed_id != seed2.seed_id
                    
    def test_network_architecture_consistency(self):
        """Test that network architecture is consistent."""
        net = BaseNet(hidden_dim=64)
        
        # Check layer dimensions match
        assert net.fc1.out_features == net.fc2.in_features == 64
        assert net.fc2.out_features == net.fc3.in_features == 64
        assert net.fc3.out_features == net.fc4.in_features == 64
        assert net.fc4.out_features == net.fc5.in_features == 64
        assert net.fc5.out_features == net.fc6.in_features == 64
        assert net.fc6.out_features == net.fc7.in_features == 64
        assert net.fc7.out_features == net.fc8.in_features == 64
        assert net.fc8.out_features == net.out.in_features == 64
        assert net.out.out_features == 2
        
        # Check that all seeds have correct dimension
        for i in range(1, 9):
            seed = getattr(net, f'seed{i}')
            assert seed.dim == 64
