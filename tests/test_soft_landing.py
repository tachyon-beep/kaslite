import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from morphogenetic_engine.components import SentinelSeed
from morphogenetic_engine.core import SeedManager


def test_training_to_blending():
    seed = SentinelSeed("s1", dim=4)
    seed.initialize_child()
    dummy = torch.zeros(2, 4)
    seed.seed_manager.seeds[seed.seed_id]["buffer"].append(dummy)
    for _ in range(100):
        seed.train_child_step(dummy)
    assert seed.state == "blending"


def test_blending_to_active():
    seed = SentinelSeed("s2", dim=4)
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
    seed = SentinelSeed("s3", dim=4)
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
    manager = SeedManager()
    manager.seeds.clear()
    manager.germination_log.clear()
    seed = SentinelSeed("g1", dim=4)
    seed.initialize_child()
    x = torch.randn(3, 4, requires_grad=True)
    seed.seed_manager.seeds[seed.seed_id]["buffer"].append(x)
    for _ in range(5):
        seed.train_child_step(x)
    assert x.grad is None


def test_redundant_transition_logged_once():
    manager = SeedManager()
    manager.seeds.clear()
    manager.germination_log.clear()
    seed = SentinelSeed("r1", dim=4)
    before = len(manager.germination_log)
    seed._set_state("training")
    mid = len(manager.germination_log)
    seed._set_state("training")
    after = len(manager.germination_log)
    assert mid == before + 1
    assert after == mid
