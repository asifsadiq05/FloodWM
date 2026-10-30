import numpy as np
import pytest
torch = pytest.importorskip("torch")
from floodwm.config import load_config
from floodwm.data import FloodDataset, generate_split
from floodwm.world_model import FloodWorldModel
from floodwm.train import rollout_loss


def test_model_shapes_and_gradients():
    cfg = load_config(); cfg["data"]["n_train"] = 2
    ds = FloodDataset(generate_split(cfg, 2, 0))
    static, forcing, depth = [t.unsqueeze(0) for t in ds[0]]
    m = FloodWorldModel(8)
    out = m(depth[:, 0], static, forcing[:, :5])
    assert out.shape == (1, 5, 48, 48) and out.min() >= 0
    loss = rollout_loss(m, static, forcing, depth, 5); loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_assimilation_changes_state():
    m = FloodWorldModel(8)
    static = torch.zeros(1, 1, 16, 16); d0 = torch.zeros(1, 16, 16)
    s = m.init_state(d0, static)
    obs = torch.ones(1, 16, 16); mask = torch.zeros(1, 16, 16); mask[:, :8] = 1
    s2 = m.assimilate(s, obs, static, mask)
    assert not torch.allclose(s[0][:, :, :8], s2[0][:, :, :8])
    assert torch.allclose(s[0][:, :, 8:], s2[0][:, :, 8:])
