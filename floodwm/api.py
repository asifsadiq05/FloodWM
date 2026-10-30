"""FastAPI service: /health, /forecast, /evacuate.  Run: uvicorn floodwm.api:app --port 8000"""
import os
from functools import lru_cache
from typing import List, Optional, Tuple
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .config import ROOT
from .evaluate import load_model, forecast
from .network import build_grid_roads
from .planner import plan_route

app = FastAPI(title="floodwm", version="0.1.0",
              description="AI world model for flood forecasting and evacuation routing.")
CKPT = os.environ.get("FLOODWM_CKPT", str(ROOT / "outputs" / "world_model.pt"))


@lru_cache(maxsize=1)
def _model():
    return load_model(CKPT)


class ForecastRequest(BaseModel):
    dem: List[List[float]] = Field(..., description="Elevation grid (m), H x W")
    depth0: List[List[float]] = Field(..., description="Current flood depth grid (m)")
    rain: List[List[List[float]]] = Field(..., description="Rain forecast (m/step), T x H x W")
    river_inflow: Optional[List[List[List[float]]]] = None


class EvacuateRequest(ForecastRequest):
    origins: List[Tuple[int, int]] = Field(..., description="(row, col) of evacuees")
    depart_step: int = 0


def _scenario(req: ForecastRequest):
    dem = np.asarray(req.dem, np.float32)
    rain = np.asarray(req.rain, np.float32)
    inflow = np.asarray(req.river_inflow, np.float32) if req.river_inflow else np.zeros_like(rain)
    d0 = np.asarray(req.depth0, np.float32)
    if dem.shape != d0.shape or rain.shape[1:] != dem.shape or inflow.shape != rain.shape:
        raise HTTPException(422, "dem, depth0, rain and river_inflow shapes are inconsistent")
    depth = np.concatenate([d0[None], np.zeros_like(rain)], 0)
    return {"z": dem, "rain": rain, "inflow": inflow, "depth": depth}


@app.get("/health")
def health():
    try:
        _model()
        return {"status": "ok", "model_loaded": True}
    except Exception as e:  # noqa
        return {"status": "degraded", "model_loaded": False, "detail": str(e)}


@app.post("/forecast")
def api_forecast(req: ForecastRequest):
    model, _ = _model()
    sc = _scenario(req)
    pred = forecast(model, sc)
    return {"max_depth_m": pred.max(0).tolist(),
            "peak_step": pred.reshape(pred.shape[0], -1).max(1).argmax().item(),
            "depth_by_step": pred.tolist()}


@app.post("/evacuate")
def api_evacuate(req: EvacuateRequest):
    model, cfg = _model()
    sc = _scenario(req)
    pred = forecast(model, sc)
    ev, gc, sim = cfg["evacuation"], cfg["grid"], cfg["sim"]
    h, w = sc["z"].shape
    g, shelters = build_grid_roads(h, w, ev["road_spacing_cells"], gc["cell_size_m"], sc["z"],
                                   ev["n_shelters"], np.random.default_rng(0))
    routes = []
    for o in req.origins:
        node = min(g.nodes, key=lambda n: abs(n[0] - o[0]) + abs(n[1] - o[1]))
        path, t = plan_route(g, node, shelters, pred, req.depart_step, sim["dt_minutes"] * 60,
                             ev["walk_speed_mps"], ev["max_safe_depth_m"], risk_weight=1.0)
        routes.append({"origin": list(o), "path": path, "eta_minutes":
                       None if path is None else (t - req.depart_step * sim["dt_minutes"] * 60) / 60})
    return {"shelters": shelters, "routes": routes,
            "disclaimer": "Decision support only. Confirm with official authorities before acting."}
