"""Tests for soft landing functionality and state transitions in the morphogenetic engine."""
# pylint: disable=protected-access
import random
import pytest
import torch

from morphogenetic_engine.components import BaseNet, SentinelSeed
from morphogenetic_engine.core import SeedManager


def test_training_to_blending():
    """Test that seed transitions from training to blending state after sufficient training steps."""
    seed_manager = SeedManager()
    seed = SentinelSeed("s1", 4, seed_manager)
    seed.initialize_child()
    dummy = torch.zeros(2, 4)
    seed.seed_manager.seeds[seed.seed_id]["buffer"].append(dummy)
    for _ in range(100):
        seed.train_child_step(dummy)
    assert seed.state == "blending"


def test_blending_to_active():
    """Test that seed transitions from blending to active state and alpha reaches 1.0."""
    seed_manager = SeedManager()
    seed = SentinelSeed("s2", 4, seed_manager)
    seed.initialize_child()
    dummy = torch.zeros(2, 4)
    seed.seed_manager.seeds[seed.seed_id]["buffer"].append(dummy)
    for _ in range(100):
        seed.train_child_step(dummy)
    for _ in range(seed.blend_steps):
        seed.update_blending()
    assert seed.state == "active"
    assert seed.alpha == pytest.approx(1.0, abs=1e-6)


def test_forward_shapes():
    """Test that seed forward pass maintains correct tensor shapes throughout state transitions."""
    seed_manager = SeedManager()
    seed = SentinelSeed("s3", 4, seed_manager)
    x = torch.randn(5, 4)
    out = seed(x)
    assert out.shape == x.shape
    seed.initialize_child()
    out = seed(x)
    assert out.shape == x.shape
    seed.seed_manager.seeds[seed.seed_id]["buffer"].append(x)
    for _ in range(100):
        seed.train_child_step(x)
    for _ in range(seed.blend_steps):
        seed.update_blending()
    out = seed(x)
    assert out.shape == x.shape


def test_grad_leak_blocked():
    """Test that gradient computation is properly isolated during seed training."""
    model = BaseNet(hidden_dim=4, seed_manager=SeedManager(), input_dim=2)
    seed = model.seed1

    # Ensure the seed is registered and initialize it
    seed.initialize_child()
    x = torch.randn(3, 4, requires_grad=True)
    seed.seed_manager.seeds[seed.seed_id]["buffer"].append(x)
    for _ in range(5):
        seed.train_child_step(x)
    for name, p in model.named_parameters():
        if "seed" not in name:
            assert p.grad is None


def test_redundant_transition_logged_once():
    """Test that redundant state transitions are logged only once to avoid spam."""
    manager = SeedManager()
    manager.seeds.clear()
    manager.germination_log.clear()
    seed = SentinelSeed("r1", 4, manager)
    before = len(manager.germination_log)
    seed._set_state("training")
    mid = len(manager.germination_log)
    seed._set_state("training")
    after = len(manager.germination_log)
    assert mid == before + 1
    assert after == mid


def test_buffer_shape_sampling():
    """Test that buffer sampling produces correct batch shapes with proper tensor concatenation."""
    buf = [torch.randn(64, 128), torch.randn(16, 128)]
    sample_tensors = random.sample(list(buf), min(64, len(buf)))
    batch = torch.cat(sample_tensors, dim=0)
    if batch.size(0) > 64:
        idx = torch.randperm(batch.size(0), device=batch.device)[:64]
        batch = batch[idx]
    assert batch.shape[0] == 64
