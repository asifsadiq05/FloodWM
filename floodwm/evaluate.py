"""Forecast skill metrics and evacuation-policy benchmark."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from .config import load_config, ROOT
from .data import FloodDataset, generate_split
from .network import build_grid_roads, sample_origins
from .planner import plan_route, execute_route
from .world_model import FloodWorldModel


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    m = FloodWorldModel(cfg["model"]["hidden_channels"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, cfg


@torch.no_grad()
def forecast(model, scenario, device="cpu"):
    """Predict depth[T+1,H,W] from the initial state and forcing forecast."""
    static = torch.from_numpy(scenario["z"] / FloodDataset.Z_SCALE)[None, None].float().to(device)
    forcing = torch.from_numpy(np.stack([scenario["rain"], scenario["inflow"]], 1)
                               / FloodDataset.F_SCALE)[None].float().to(device)
    d0 = torch.from_numpy(scenario["depth"][0])[None].float().to(device)
    pred = model(d0, static, forcing)[0].cpu().numpy()
    return np.concatenate([scenario["depth"][:1], pred], 0)


def skill(true, pred, thr=0.1):
    err = pred[1:] - true[1:]
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    tw, pw = true[1:] > thr, pred[1:] > thr
    hits, miss, fa = (tw & pw).sum(), (tw & ~pw).sum(), (~tw & pw).sum()
    csi = float(hits / max(hits + miss + fa, 1))          # critical success index
    peak_err = float(np.abs(pred.max(0) - true.max(0)).mean())
    return {"rmse_m": rmse, "mae_m": mae, "csi_0.1m": csi, "peak_depth_mae_m": peak_err}


def evacuation_benchmark(cfg, model, scenarios, seed=0, device="cpu"):
    ev, g_, s_ = cfg["evacuation"], cfg["grid"], cfg["sim"]
    dt_s = s_["dt_minutes"] * 60
    rng = np.random.default_rng(seed)
    res = {k: [0, 0] for k in ["static_shortest", "reactive", "ai_forecast", "oracle"]}
    for sc in scenarios:
        g, shelters = build_grid_roads(g_["height"], g_["width"], ev["road_spacing_cells"],
                                       g_["cell_size_m"], sc["z"], ev["n_shelters"], rng)
        origins = sample_origins(g, shelters, sc["depth"][-1], ev["n_origins"], rng)
        pred = forecast(model, sc, device)
        zeros = np.zeros_like(sc["depth"])
        policies = {
            "static_shortest": (zeros, False),
            "reactive": (sc["depth"], True),     # current conditions only, no forecast
            "ai_forecast": (pred, False),
            "oracle": (sc["depth"], False),
        }
        for name, (tl, frozen) in policies.items():
            for o in origins:
                path, _ = plan_route(g, o, shelters, tl, ev["depart_step"], dt_s,
                                     ev["walk_speed_mps"], ev["max_safe_depth_m"], risk_weight=1.0,
                                     frozen=frozen)
                ok, _ = execute_route(g, path, sc["depth"], ev["depart_step"], dt_s,
                                      ev["walk_speed_mps"], ev["max_safe_depth_m"])
                res[name][0] += int(ok); res[name][1] += 1
    return {k: v[0] / max(v[1], 1) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "outputs" / "world_model.pt"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "metrics.json"))
    a = ap.parse_args()
    model, cfg = load_model(a.ckpt)
    te = generate_split(cfg, cfg["data"]["n_test"], cfg["data"]["seed"] + 2)
    sk = [skill(s["depth"], forecast(model, s)) for s in te]
    forecast_metrics = {k: float(np.mean([m[k] for m in sk])) for k in sk[0]}
    evac = evacuation_benchmark(cfg, model, te[:10])
    out = {"forecast": forecast_metrics, "evacuation_success_rate": evac}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
