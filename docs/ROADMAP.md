# Roadmap: simulator → real deployment

| Phase | Weeks | Deliverable |
|---|---|---|
| 0. Baseline | 1–2 | This repo running; metrics reproduced |
| 1. Real terrain | 3–4 | Real DEM + OSM roads for your city; hydro-conditioned |
| 2. Teacher sims | 5–7 | LISFLOOD-FP/HEC-RAS runs over 100+ rain/river scenarios; retrain world model |
| 3. Real data | 8–10 | Sentinel-1 flood maps + gauge data; fine-tune, hindcast 2–3 historical floods |
| 4. Better model | 11–12 | UNet/FNO backbone, ensembles, uncertainty, observation assimilation tests |
| 5. Planner v2 | 13–14 | Shelter capacities, vehicles vs foot, vulnerable-group constraints |
| 6. Product | 15–16 | Web map dashboard, alerts, API hardening, user study with responders |
| 7. Pilot | 17+ | Shadow-mode run in a flood season; compare with official decisions |

**Paper/thesis angles**: value of forecast-aware routing vs reactive routing; world-model rollout
error vs lead time; assimilation of sparse sensors; equity-aware evacuation.
