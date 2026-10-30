# Real-world data (Bangladesh-focused examples; adapt to your region)

| Need | Source | Notes |
|---|---|---|
| Terrain (DEM) | SRTM 30 m, Copernicus GLO-30, ALOS AW3D30; LiDAR if available | Hydro-condition the DEM (burn rivers, embankments, culverts) |
| Rainfall | GPM IMERG, CHIRPS, Bangladesh Met Department (BMD) radar/gauges, ECMWF/GFS forecasts | Use *forecast* fields as model forcing |
| River levels / discharge | BWDB & FFWC gauges, GloFAS (Copernicus) | FFWC publishes daily flood bulletins |
| Observed flood extent | Sentinel-1 SAR (works through cloud), Sentinel-2, MODIS, Global Flood Database, UNOSAT | Label for training & validation |
| Roads, buildings, shelters | OpenStreetMap (`osmnx`), HOT OSM, cyclone-shelter lists from DDM/MoDMR | Verify shelter capacity & accessibility |
| Population | WorldPop, GHSL, census | Weight origins by residents |
| Land cover / soil | ESA WorldCover, SoilGrids | Infiltration & roughness |
| Tide / surge | Tide gauges, ocean forecast models | Essential for coastal / delta areas |
| Ground truth reports | Crowd reports, Facebook/Telegram groups, call-centre logs | Noisy — assimilate with low weight |

Pipeline: `raw → clean/reproject (EPSG:32646 for Dhaka) → resample to common grid → tile →
data/processed/*.npz`. Version datasets (DVC) and keep per-event metadata.
