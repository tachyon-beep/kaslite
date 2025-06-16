"""Tests for the SentinelSeed component."""

# pylint: disable=protected-access

import pytest
import torch

from morphogenetic_engine.components import SentinelSeed
from morphogenetic_engine.core import SeedManager


class TestSentinelSeed:
    """Test suite for SentinelSeed class."""

    def _create_seed_manager(self):
        """Helper to create a seed manager for tests."""
        return SeedManager()

    def test_initialization(self):
        """Test SentinelSeed initialization."""
        seed_manager = SeedManager()
        seed = SentinelSeed(
            seed_id="test_seed",
            dim=64,
            seed_manager=seed_manager,
            blend_steps=25,
            shadow_lr=2e-3,
            progress_thresh=0.7,
            drift_warn=0.15,
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

    def test_state_transitions(self):
        """Test state transition handling."""
        seed_manager = self._create_seed_manager()
        seed = SentinelSeed("test_seed", 32, seed_manager)
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

        # Test redundant transition is ignored
        log_count_before = len(manager.germination_log)
        seed._set_state("active")  # Same state
        assert len(manager.germination_log) == log_count_before

    def test_child_network_initialization(self):
        """Test child network initialization and identity setup."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())

        # Initially should be identity (zero weights)
        for param in seed.child.parameters():
            assert torch.allclose(param, torch.zeros_like(param))
            assert not param.requires_grad

        # After initialization, should have proper weights
        seed.initialize_child()
        assert seed.state == "training"

        weight_params = [p for p in seed.child.parameters() if p.dim() > 1]
        for param in weight_params:
            assert not torch.allclose(param, torch.zeros_like(param))
            assert param.requires_grad

    def test_training_behavior(self):
        """Test training step behavior across different states."""
        seed = SentinelSeed(
            "test_seed",
            dim=32,
            seed_manager=self._create_seed_manager(),
            progress_thresh=0.5,
        )

        inputs = torch.randn(4, 32)
        initial_progress = seed.training_progress

        # Should ignore training in dormant state
        seed.train_child_step(inputs)
        assert seed.training_progress == initial_progress
        assert seed.state == "dormant"

        # Should ignore empty input
        seed.initialize_child()
        empty_input = torch.empty(0, 32)
        initial_progress = seed.training_progress
        seed.train_child_step(empty_input)
        assert seed.training_progress == initial_progress

        # Should progress in training state
        for _ in range(60):  # Should exceed progress threshold
            seed.train_child_step(inputs)

        assert seed.training_progress > initial_progress
        assert seed.state == "blending"
        assert seed.alpha == pytest.approx(0.0)

    def test_blending_updates(self):
        """Test blending process."""
        seed = SentinelSeed(
            "test_seed",
            dim=32,
            seed_manager=self._create_seed_manager(),
            blend_steps=10,
        )
        seed._set_state("blending")

        initial_alpha = seed.alpha
        seed.update_blending()
        assert seed.alpha > initial_alpha

        # Should not update in non-blending state
        seed._set_state("active")
        alpha_before = seed.alpha
        seed.update_blending()
        assert seed.alpha == alpha_before

    def test_forward_pass_behavior(self):
        """Test forward pass in different states."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())
        inputs = torch.randn(4, 32)

        # Dormant: should return input unchanged
        output = seed.forward(inputs)
        assert torch.allclose(output, inputs)

        # Training: should return input unchanged (train_child_step is separate)
        seed.initialize_child()
        output = seed.forward(inputs)
        assert torch.allclose(output, inputs)

        # But train_child_step should increase progress
        initial_progress = seed.training_progress
        seed.train_child_step(inputs)
        assert seed.training_progress > initial_progress

        # Blending: should blend child output with input
        seed._set_state("blending")
        seed.alpha = 0.5
        output = seed.forward(inputs)
        assert output.shape == inputs.shape
        assert not torch.allclose(output, inputs)  # Should be modified

        # Active: should return child output
        seed._set_state("active")
        output = seed.forward(inputs)
        assert output.shape == inputs.shape

    def test_drift_monitoring(self):
        """Test drift monitoring during forward pass."""
        seed = SentinelSeed(
            "test_seed",
            dim=32,
            seed_manager=self._create_seed_manager(),
            drift_warn=0.1,
        )

        # Set up for blending state where drift warnings occur
        seed.initialize_child()
        seed._set_state("blending")
        seed.alpha = 0.5

        # Forward pass should compute and record drift
        inputs = torch.randn(4, 32)
        output = seed.forward(inputs)

        # Verify output shape and that drift was recorded
        assert output.shape == inputs.shape
        telemetry = seed.seed_manager.seeds[seed.seed_id]["telemetry"]
        assert "drift" in telemetry
        assert isinstance(telemetry["drift"], float)

    def test_health_signal_computation(self):
        """Test health signal computation."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())

        # Insufficient data should return infinity
        health = seed.get_health_signal()
        assert health == float("inf")

        # Add sufficient data
        seeds_dict = seed.seed_manager.seeds["test_seed"]
        buffer = seeds_dict["buffer"]
        for _ in range(15):
            buffer.append(torch.randn(2, 32))

        health = seed.get_health_signal()
        assert isinstance(health, float)
        assert health != float("inf")
        assert health >= 0.0

    def test_complete_state_cycle(self):
        """Test all state transitions including dormant."""
        seed = SentinelSeed("test_seed", dim=32, seed_manager=self._create_seed_manager())

        # Complete cycle: dormant -> training -> blending -> active -> dormant
        assert seed.state == "dormant"

        seed._set_state("training")
        assert seed.state == "training"
        assert seed.seed_manager.seeds["test_seed"]["status"] == "pending"

        seed._set_state("blending")
        assert seed.state == "blending"
        assert seed.seed_manager.seeds["test_seed"]["status"] == "pending"

        seed._set_state("active")
        assert seed.state == "active"
        assert seed.seed_manager.seeds["test_seed"]["status"] == "active"

        seed._set_state("dormant")
        assert seed.state == "dormant"
        assert seed.seed_manager.seeds["test_seed"]["status"] == "dormant"
