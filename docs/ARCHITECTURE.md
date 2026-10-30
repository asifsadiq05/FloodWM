# Architecture

## 1. State & dynamics
- **State** `s_t`: latent grid (C×H×W) from a conv encoder of (depth₀, terrain).
- **Transition**: two stacked ConvGRU cells; input = [rain_t, river_inflow_t, terrain, previous depth].
  Local 3×3 kernels mirror the local, neighbour-to-neighbour physics of overland flow.
- **Decoder**: 1×1/3×3 conv → ReLU → depth ≥ 0 (metres).
- **Assimilation**: `assimilate(state, obs, mask)` overwrites latent cells where observations
  exist (gauge, satellite flood extent, citizen reports) — enables 4D-Var/EnKF-style updating.

## 2. Training
- Teacher data: simulator trajectories (replace/augment with HEC-RAS, LISFLOOD-FP, historical events).
- Loss: smooth-L1 on depth, 5× weight on flooded cells; **rollout** loss over up to 12 steps
  (curriculum) to limit error accumulation.
- Scale-up: UNet/FNO/graph-neural-net backbones, patch-based training for large cities, mixed precision.

## 3. Evacuation planner
- Graph from OpenStreetMap (nodes = intersections, edges = road segments with the DEM cells they cross).
- Time-dependent Dijkstra: an edge is usable at arrival time *t* only if predicted max depth
  < safe threshold (0.3 m pedestrians; ~0.5 m small vehicles; ~0.15 m for vulnerable people).
- Speed degrades with depth; optional risk weight penalises deep-water edges.
- Extensions: shelter capacity (min-cost flow), contraflow, vehicle vs foot, multi-modal, staged
  zone-by-zone evacuation orders, uncertainty-aware routing (ensemble → chance constraints).

## 4. Serving
`FastAPI` → `/forecast` (depth grids, peak time) and `/evacuate` (routes + ETA). Containerised with
`Dockerfile`; put behind an authenticated gateway; cache the model; log every request/response.

## 5. Uncertainty (recommended next)
Train a deep ensemble or use MC-dropout; produce exceedance probabilities P(depth > 0.3 m) per cell;
route on the 90th-percentile depth, not the mean.
