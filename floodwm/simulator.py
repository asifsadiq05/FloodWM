"""Physics-lite 2D flood simulator (diffusive-wave style) on a DEM grid.

It plays the role of the "real world" that the AI world model learns from.
In production, replace it with HEC-RAS / LISFLOOD-FP / TUFLOW outputs and
observed gauge/satellite data (see docs/DATA_SOURCES.md).
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def make_dem(h, w, rng):
    """Synthetic delta-like terrain: gentle slope to the south, a meandering river."""
    y = np.arange(h)[:, None] * np.ones((1, w))
    x = np.ones((h, 1)) * np.arange(w)[None, :]
    slope = 3.0 * (1 - y / h)
    noise = gaussian_filter(rng.normal(size=(h, w)), 4) * 6.0
    center = w / 2 + 0.12 * w * np.sin(y / (h / 5.0) + rng.uniform(0, 6.28))
    dist = np.abs(x - center)
    channel = -1.2 * np.exp(-(dist / 2.0) ** 2)
    floodplain = 0.8 * np.clip(dist / (0.3 * w), 0, 1)
    z = slope + noise + channel + floodplain
    z = z - z.min()
    river_mask = dist < 2.0
    return z.astype(np.float32), river_mask


def make_forcing(steps, h, w, river_mask, rng):
    """Storm rainfall (m per step) + upstream river inflow (m per step on top-row river cells)."""
    t = np.arange(steps)
    peak = rng.uniform(3, 9)
    width = rng.uniform(2, 5)
    amp = rng.uniform(0.012, 0.035)
    hyeto = amp * np.exp(-0.5 * ((t - peak) / width) ** 2)
    spatial = gaussian_filter(rng.normal(size=(h, w)), 8)
    spatial = 1.0 + 0.5 * spatial / (np.abs(spatial).max() + 1e-6)
    rain = (hyeto[:, None, None] * spatial[None]).astype(np.float32)
    lag = int(rng.integers(2, 6))
    q = np.zeros(steps)
    q[lag:] = hyeto[:-lag] * rng.uniform(15, 35)
    inflow = np.zeros((steps, h, w), dtype=np.float32)
    inflow[:, 0, :] = q[:, None] * river_mask[0][None, :]
    return rain, inflow


def step(depth, z, rain, inflow, flow_coeff=0.2, infil=0.002):
    """One explicit time step. Mass-conserving except sink at the south boundary."""
    w = z + depth
    wp = np.pad(w, 1, mode="edge")
    wp[-1, 1:-1] = z.min() - 0.5  # southern boundary drains to the sea
    up, down = wp[:-2, 1:-1], wp[2:, 1:-1]
    left, right = wp[1:-1, :-2], wp[1:-1, 2:]
    cap = depth / 4.0
    o_up = np.minimum(flow_coeff * np.maximum(w - up, 0), cap)
    o_dn = np.minimum(flow_coeff * np.maximum(w - down, 0), cap)
    o_lf = np.minimum(flow_coeff * np.maximum(w - left, 0), cap)
    o_rt = np.minimum(flow_coeff * np.maximum(w - right, 0), cap)
    inflow_n = np.zeros_like(depth)
    inflow_n[1:] += o_dn[:-1]
    inflow_n[:-1] += o_up[1:]
    inflow_n[:, 1:] += o_rt[:, :-1]
    inflow_n[:, :-1] += o_lf[:, 1:]
    new = depth - (o_up + o_dn + o_lf + o_rt) + inflow_n + rain + inflow
    new = np.maximum(new - infil * (new > 0), 0.0)
    return new.astype(np.float32)


def simulate(z, rain, inflow, flow_coeff=0.2, infil=0.002, depth0=None):
    """Returns depth[T+1, H, W] including the initial state."""
    steps = rain.shape[0]
    d = np.zeros_like(z) if depth0 is None else depth0.copy()
    out = [d]
    for t in range(steps):
        d = step(d, z, rain[t], inflow[t], flow_coeff, infil)
        out.append(d)
    return np.stack(out)


def make_scenario(cfg, rng):
    g, s = cfg["grid"], cfg["sim"]
    z, river = make_dem(g["height"], g["width"], rng)
    rain, inflow = make_forcing(s["steps"], g["height"], g["width"], river, rng)
    depth = simulate(z, rain, inflow, s["flow_coeff"], s["infiltration_m"])
    return {"z": z, "river": river, "rain": rain, "inflow": inflow, "depth": depth}
