import numpy as np
from floodwm.config import load_config
from floodwm.simulator import make_scenario, step
from floodwm.network import build_grid_roads
from floodwm.planner import plan_route, execute_route


def _sc(seed=0):
    return make_scenario(load_config(), np.random.default_rng(seed))


def test_simulator_nonnegative_and_finite():
    d = _sc()["depth"]
    assert np.isfinite(d).all() and d.min() >= 0 and d.max() > 0.05


def test_mass_balance_without_boundary():
    z = np.zeros((8, 8), np.float32) + np.random.rand(8, 8).astype(np.float32) * 0.1
    d = np.zeros_like(z); d[3:5, 3:5] = 1.0
    d2 = step(d, z, np.zeros_like(z), np.zeros_like(z), infil=0.0)
    assert abs(d2.sum() - d.sum()) < 1e-3     # no rain, no sink reached yet


def test_planner_blocks_flooded_edges():
    cfg = load_config(); sc = _sc(1)
    g, sh = build_grid_roads(48, 48, 4, 100.0, sc["z"], 4, np.random.default_rng(0))
    origin = next(n for n in g.nodes if n not in sh)
    dry = np.zeros_like(sc["depth"])
    path, t = plan_route(g, origin, sh, dry, 0, 1800, 1.4, 0.3)
    assert path is not None and path[-1] in sh
    flooded = dry + 5.0
    p2, _ = plan_route(g, origin, sh, flooded, 0, 1800, 1.4, 0.3)
    assert p2 is None
    ok, _ = execute_route(g, path, flooded, 0, 1800, 1.4, 0.3)
    assert not ok
