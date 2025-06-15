import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from morphogenetic_engine.components import SentinelSeed


def test_training_to_blending():
    seed = SentinelSeed("s1", dim=4)
    seed.initialize_child()
    dummy = torch.zeros(2, 4)
    for _ in range(100):
        seed.train_child_step(dummy)
    assert seed.state == "blending"


def test_blending_to_active():
    seed = SentinelSeed("s2", dim=4)
    seed.initialize_child()
    dummy = torch.zeros(2, 4)
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
    for _ in range(100):
        seed.train_child_step(x)
    for _ in range(seed.blend_steps):
        seed.update_blending()
    out = seed(x)
    assert out.shape == x.shape
