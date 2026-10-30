# Deployment & operations

1. **Pilot area** — one upazila/ward with a known flood history. Align with the local disaster
   management committee early.
2. **Run modes** — (a) nightly batch forecasts from NWP rainfall; (b) event mode, re-forecast
   every 1–3 h as new gauge/satellite data arrive.
3. **Infra** — Docker + Kubernetes/VM, GPU optional (CPU fine for small grids). Object store for
   rasters, Postgres/PostGIS for routes/shelters, Redis cache, Prometheus/Grafana monitoring.
4. **Outputs** — GeoTIFF/GeoJSON flood maps and routes, SMS/CAP alerts, web map (Leaflet/MapLibre),
   offline PDF maps for responders.
5. **Resilience** — works offline at the edge (laptop/Raspberry-class) with last-known forecast;
   fallback to static evacuation plan if the model is unavailable or stale.
6. **MLOps** — retrain after each major event; track experiments (MLflow/W&B); canary new models
   against the previous one on held-out floods; drift monitoring on input forcings.
7. **Security** — authN/Z, TLS, audit logs, no personal location data stored without consent.
