"""End-to-end demo: train -> evaluate -> figure. Run: python scripts/run_demo.py [--epochs 12]"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from floodwm.config import load_config, ROOT
from floodwm.data import generate_split
from floodwm.evaluate import forecast, load_model, skill, evacuation_benchmark
from floodwm.network import build_grid_roads, sample_origins
from floodwm.planner import plan_route
from floodwm.train import train

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--skip-train", action="store_true")
a = ap.parse_args()
cfg = load_config()
if a.epochs:
    cfg["train"]["epochs"] = a.epochs
ckpt = ROOT / "outputs" / "world_model.pt"
if not a.skip_train:
    train(cfg, ckpt)
model, cfg = load_model(ckpt)
te = generate_split(cfg, cfg["data"]["n_test"], cfg["data"]["seed"] + 2)
sk = [skill(s["depth"], forecast(model, s)) for s in te]
print({k: round(float(np.mean([m[k] for m in sk])), 4) for k in sk[0]})
print("evacuation success:", evacuation_benchmark(cfg, model, te[:8]))

sc = te[0]; pred = forecast(model, sc)
k = int(np.argmax(sc["depth"].reshape(sc["depth"].shape[0], -1).max(1)))
ev, gc, sim = cfg["evacuation"], cfg["grid"], cfg["sim"]
rng = np.random.default_rng(1)
g, shelters = build_grid_roads(gc["height"], gc["width"], ev["road_spacing_cells"], gc["cell_size_m"],
                               sc["z"], ev["n_shelters"], rng)
origin = sample_origins(g, shelters, sc["depth"][-1], 1, rng)[0]
path, t = plan_route(g, origin, shelters, pred, ev["depart_step"], sim["dt_minutes"] * 60,
                     ev["walk_speed_mps"], ev["max_safe_depth_m"], risk_weight=1.0)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
vmax = max(sc["depth"][k].max(), 0.5)
for a_, d, title in [(ax[0], sc["depth"][k], f"True flood depth (step {k})"),
                     (ax[1], pred[k], "World-model forecast")]:
    im = a_.imshow(d, cmap="Blues", vmin=0, vmax=vmax); a_.set_title(title)
ax[2].imshow(np.abs(pred[k] - sc["depth"][k]), cmap="Reds"); ax[2].set_title("Absolute error (m)")
for a_ in ax[:2]:
    for s in shelters: a_.plot(s[1], s[0], "g^", ms=10)
    if path: a_.plot([p[1] for p in path], [p[0] for p in path], "k-", lw=2)
    a_.plot(origin[1], origin[0], "ro")
plt.tight_layout()
out = ROOT / "outputs" / "demo_forecast.png"; out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=130); print("saved", out)
