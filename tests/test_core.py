"""Comprehensive tests for the core module."""
import os
import sys
import time
import threading
import pytest
import torch
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from morphogenetic_engine.core import SeedManager, KasminaMicro


class TestSeedManager:
    """Test suite for SeedManager singleton class."""
    
    def test_singleton_pattern(self):
        """Test that SeedManager follows singleton pattern."""
        manager1 = SeedManager()
        manager2 = SeedManager()
        assert manager1 is manager2
        
    def test_thread_safety(self):
        """Test thread safety of singleton initialization."""
        managers = []
        
        def create_manager():
            managers.append(SeedManager())
            
        threads = [threading.Thread(target=create_manager) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        # All should be the same instance
        assert all(m is managers[0] for m in managers)
        
    def test_register_seed(self):
        """Test seed registration functionality."""
        manager = SeedManager()
        manager.seeds.clear()
        
        mock_seed = Mock()
        manager.register_seed(mock_seed, "test_seed")
        
        assert "test_seed" in manager.seeds
        assert manager.seeds["test_seed"]["module"] is mock_seed
        assert manager.seeds["test_seed"]["status"] == "dormant"
        assert manager.seeds["test_seed"]["state"] == "dormant"
        assert manager.seeds["test_seed"]["alpha"] == 0.0
        assert len(manager.seeds["test_seed"]["buffer"]) == 0
        
    def test_append_to_buffer(self):
        """Test buffer append functionality."""
        manager = SeedManager()
        manager.seeds.clear()
        
        # Register a seed first
        mock_seed = Mock()
        manager.register_seed(mock_seed, "test_seed")
        
        # Test appending tensors
        tensor1 = torch.randn(4, 8)
        tensor2 = torch.randn(4, 8)
        
        manager.append_to_buffer("test_seed", tensor1)
        manager.append_to_buffer("test_seed", tensor2)
        
        buffer = manager.seeds["test_seed"]["buffer"]
        assert len(buffer) == 2
        assert torch.equal(buffer[0], tensor1.detach())
        assert torch.equal(buffer[1], tensor2.detach())
        
    def test_append_to_nonexistent_seed(self):
        """Test appending to non-existent seed doesn't crash."""
        manager = SeedManager()
        tensor = torch.randn(4, 8)
        
        # Should not raise an exception
        manager.append_to_buffer("nonexistent", tensor)
        
    def test_request_germination_success(self):
        """Test successful germination request."""
        manager = SeedManager()
        manager.seeds.clear()
        manager.germination_log.clear()
        
        mock_seed = Mock()
        mock_seed.initialize_child = Mock()
        manager.register_seed(mock_seed, "test_seed")
        
        result = manager.request_germination("test_seed")
        
        assert result is True
        mock_seed.initialize_child.assert_called_once()
        assert manager.seeds["test_seed"]["status"] == "active"
        assert len(manager.germination_log) == 1
        assert manager.germination_log[0]["success"] is True
        
    def test_request_germination_already_active(self):
        """Test germination request on already active seed."""
        manager = SeedManager()
        manager.seeds.clear()
        
        mock_seed = Mock()
        mock_seed.initialize_child = Mock()
        manager.register_seed(mock_seed, "test_seed")
        manager.seeds["test_seed"]["status"] = "active"
        
        result = manager.request_germination("test_seed")
        
        assert result is False
        mock_seed.initialize_child.assert_not_called()
        
    def test_request_germination_nonexistent_seed(self):
        """Test germination request on non-existent seed."""
        manager = SeedManager()
        
        result = manager.request_germination("nonexistent")
        
        assert result is False
        
    def test_request_germination_exception_handling(self):
        """Test germination request with initialization failure."""
        manager = SeedManager()
        manager.seeds.clear()
        manager.germination_log.clear()
        
        mock_seed = Mock()
        mock_seed.initialize_child = Mock(side_effect=RuntimeError("Init failed"))
        manager.register_seed(mock_seed, "test_seed")
        
        result = manager.request_germination("test_seed")
        
        assert result is False
        assert manager.seeds["test_seed"]["status"] == "failed"
        assert len(manager.germination_log) == 1
        assert manager.germination_log[0]["success"] is False
        
    def test_record_transition(self):
        """Test state transition recording."""
        manager = SeedManager()
        manager.germination_log.clear()
        
        before_time = time.time()
        manager.record_transition("test_seed", "dormant", "training")
        after_time = time.time()
        
        assert len(manager.germination_log) == 1
        log_entry = manager.germination_log[0]
        assert log_entry["seed_id"] == "test_seed"
        assert log_entry["from"] == "dormant"
        assert log_entry["to"] == "training"
        assert before_time <= log_entry["timestamp"] <= after_time
        
    def test_record_drift(self):
        """Test drift recording functionality."""
        manager = SeedManager()
        manager.seeds.clear()
        
        mock_seed = Mock()
        manager.register_seed(mock_seed, "test_seed")
        
        manager.record_drift("test_seed", 0.123)
        
        assert manager.seeds["test_seed"]["telemetry"]["drift"] == 0.123
        
    def test_record_drift_nonexistent_seed(self):
        """Test drift recording for non-existent seed."""
        manager = SeedManager()
        
        # Should not raise an exception
        manager.record_drift("nonexistent", 0.123)


class TestKasminaMicro:
    """Test suite for KasminaMicro germination controller."""
    
    def test_initialization(self):
        """Test KasminaMicro initialization."""
        manager = SeedManager()
        km = KasminaMicro(manager, patience=10, delta=1e-3, acc_threshold=0.9)
        
        assert km.seed_manager is manager
        assert km.patience == 10
        assert km.delta == 1e-3
        assert km.acc_threshold == 0.9
        assert km.plateau == 0
        assert km.prev_loss == float("inf")
        
    def test_step_no_plateau(self):
        """Test step with improving loss (no plateau)."""
        manager = SeedManager()
        km = KasminaMicro(manager, patience=3, delta=1e-3)
        
        # First step should not trigger germination
        result = km.step(1.0, 0.5)
        assert result is False
        assert km.plateau == 0
        assert km.prev_loss == 1.0
        
        # Improving loss should reset plateau
        result = km.step(0.8, 0.6)
        assert result is False
        assert km.plateau == 0
        assert km.prev_loss == 0.8
        
    def test_step_plateau_detection(self):
        """Test plateau detection without triggering germination."""
        manager = SeedManager()
        km = KasminaMicro(manager, patience=3, delta=1e-3, acc_threshold=0.9)
        
        # Set initial loss
        km.step(1.0, 0.95)  # High accuracy, above threshold
        
        # Create plateau but with high accuracy (shouldn't germinate)
        for i in range(5):
            result = km.step(1.0001, 0.95)  # Minimal loss change, high accuracy
            assert result is False  # No germination due to high accuracy

    def test_step_germination_trigger(self):
        """Test germination triggering under correct conditions."""
        manager = SeedManager()
        manager.seeds.clear()

        # Mock a dormant seed
        mock_seed = Mock()
        mock_seed.get_health_signal = Mock(return_value=0.1)  # Low health signal
        manager.register_seed(mock_seed, "test_seed")

        km = KasminaMicro(manager, patience=2, delta=1e-3, acc_threshold=0.9)

        # Set initial loss
        result1 = km.step(1.0, 0.5)  # Low accuracy, below threshold
        assert result1 is False  # Should not trigger yet
        assert km.plateau == 0  # No plateau yet

        # Create plateau with low accuracy - use smaller deltas than threshold
        result2 = km.step(1.0 + 1e-4, 0.5)  # Plateau starts (diff = 1e-4 < 1e-3)
        assert result2 is False  # Should not trigger yet
        assert km.plateau == 1  # Plateau count = 1

        with patch.object(manager, 'request_germination', return_value=True) as mock_germ:
            result3 = km.step(1.0 + 2e-4, 0.5)  # Plateau continues (diff = 1e-4 < 1e-3) 
            assert result3 is True  # Should trigger germination (plateau = 2 >= patience = 2)
            mock_germ.assert_called_once_with("test_seed")
            mock_germ.assert_called_once_with("test_seed")
            
    def test_select_seed_no_dormant_seeds(self):
        """Test seed selection when no dormant seeds available."""
        manager = SeedManager()
        manager.seeds.clear()
        km = KasminaMicro(manager)
        
        result = km._select_seed()
        assert result is None
        
    def test_select_seed_choose_worst_health(self):
        """Test seed selection chooses seed with worst health signal."""
        manager = SeedManager()
        manager.seeds.clear()
        
        # Create mock seeds with different health signals
        mock_seed1 = Mock()
        mock_seed1.get_health_signal = Mock(return_value=0.5)
        mock_seed2 = Mock()
        mock_seed2.get_health_signal = Mock(return_value=0.1)  # Worst (lowest)
        mock_seed3 = Mock()
        mock_seed3.get_health_signal = Mock(return_value=0.3)
        
        manager.register_seed(mock_seed1, "seed1")
        manager.register_seed(mock_seed2, "seed2")
        manager.register_seed(mock_seed3, "seed3")
        
        km = KasminaMicro(manager)
        result = km._select_seed()
        
        assert result == "seed2"  # Should select seed with lowest health signal
        
    def test_select_seed_ignores_non_dormant(self):
        """Test seed selection ignores non-dormant seeds."""
        manager = SeedManager()
        manager.seeds.clear()
        
        mock_seed1 = Mock()
        mock_seed1.get_health_signal = Mock(return_value=0.1)
        mock_seed2 = Mock()
        mock_seed2.get_health_signal = Mock(return_value=0.2)
        
        manager.register_seed(mock_seed1, "seed1")
        manager.register_seed(mock_seed2, "seed2")
        
        # Make seed1 non-dormant
        manager.seeds["seed1"]["status"] = "active"
        
        km = KasminaMicro(manager)
        result = km._select_seed()
        
        assert result == "seed2"  # Should select the dormant seed
        
    def test_accuracy_threshold_prevents_germination(self):
        """Test that high accuracy prevents germination even with plateau."""
        manager = SeedManager()
        manager.seeds.clear()
        
        mock_seed = Mock()
        mock_seed.get_health_signal = Mock(return_value=0.1)
        manager.register_seed(mock_seed, "test_seed")
        
        km = KasminaMicro(manager, patience=1, delta=1e-3, acc_threshold=0.8)
        
        # Create plateau with high accuracy
        km.step(1.0, 0.9)  # Above threshold
        
        result = km.step(1.0001, 0.9)  # Plateau but high accuracy
        assert result is False  # No germination
        
    def test_plateau_reset_on_improvement(self):
        """Test that plateau counter resets on loss improvement."""
        manager = SeedManager()
        km = KasminaMicro(manager, patience=3, delta=1e-3)
        
        # Build up plateau
        km.step(1.0, 0.5)
        km.step(1.0001, 0.5)  # plateau = 1
        km.step(1.0002, 0.5)  # plateau = 2
        
        assert km.plateau == 2
        
        # Improve loss significantly
        km.step(0.8, 0.5)
        
        assert km.plateau == 0  # Should reset
