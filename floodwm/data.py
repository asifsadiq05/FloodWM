import numpy as np
import torch
from torch.utils.data import Dataset
from .simulator import make_scenario


def generate_split(cfg, n, seed):
    rng = np.random.default_rng(seed)
    return [make_scenario(cfg, rng) for _ in range(n)]


class FloodDataset(Dataset):
    """Each item: static DEM, forcing[T,2,H,W] (rain, river inflow), depth[T+1,H,W]."""
    Z_SCALE = 10.0
    F_SCALE = 0.05

    def __init__(self, scenarios):
        self.s = scenarios

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        static = torch.from_numpy(s["z"] / self.Z_SCALE)[None]
        forcing = torch.from_numpy(np.stack([s["rain"], s["inflow"]], 1) / self.F_SCALE)
        depth = torch.from_numpy(s["depth"])
        return static.float(), forcing.float(), depth.float()
