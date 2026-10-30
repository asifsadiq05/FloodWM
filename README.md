# floodwm — AI World Model for Flood Evacuation & Disaster Response 🌊

A learned **world model** that forecasts how a flood evolves over a city (given rain and river
forecasts), then uses those predictions to plan **time-aware evacuation routes** to shelters.

```
rain + river forecast ─┐
DEM / terrain ─────────┼─► Latent World Model (ConvGRU) ─► flood depth [t+1 … t+T]
live observations ─────┘        ▲ assimilation                    │
(gauges, satellite, reports) ───┘                                 ▼
                                   Road graph + shelters ─► Time-dependent router ─► routes / ETAs
                                                                  │
                                                         FastAPI service + dashboard/alerts
```

## Why a world model (not just a classifier)?
It learns the *dynamics*: given a state and a future action/forcing (rain, dam release, embankment
breach), it rolls forward. That lets you ask **"what if?"** (e.g. breach at node X, +30% rain) and
re-plan evacuation before the water arrives.

## Quick start
```bash
pip install -r requirements.txt
make test                 # unit tests
python scripts/run_demo.py --epochs 12   # train + evaluate + figure in outputs/
uvicorn floodwm.api:app --port 8000      # serve the API
```

## Layout
| Path | Purpose |
|---|---|
| `floodwm/simulator.py` | Physics-lite flood simulator (stand-in for HEC-RAS / LISFLOOD-FP) |
| `floodwm/world_model.py` | Encoder + 2-layer ConvGRU latent dynamics + decoder, observation assimilation |
| `floodwm/train.py` | Multi-step rollout training with curriculum |
| `floodwm/network.py` | Road graph + shelters (synthetic; OSM loader stub) |
| `floodwm/planner.py` | Earliest-arrival routing that respects flood depth at arrival time |
| `floodwm/evaluate.py` | RMSE / CSI / peak error + evacuation benchmark (static vs reactive vs AI vs oracle) |
| `floodwm/api.py` | `/forecast`, `/evacuate`, `/health` |
| `docs/` | Architecture, data sources, deployment, safety & ethics, roadmap |

## Evaluation protocol
1. **Forecast skill** — RMSE/MAE (m), CSI at 0.1 m, peak-depth error, error vs lead time.
2. **Decision skill** — % of evacuees reaching a shelter without hitting unsafe water, under four
   policies: *static shortest path*, *reactive (current conditions only)*, *AI forecast*, *oracle*.
   The gap AI-vs-reactive is the value of the world model; AI-vs-oracle is the remaining headroom.

## Going from simulation to real life
Read `docs/DATA_SOURCES.md` and `docs/ROADMAP.md`. Short version: pretrain on simulator output of
your real city, fine-tune on historical events and satellite flood maps, validate on held-out
floods, and **keep a human in the loop** (`docs/SAFETY_ETHICS.md`).
