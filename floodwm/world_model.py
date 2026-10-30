"""Latent world model: encoder -> ConvGRU latent dynamics driven by forcing -> decoder.

State s_t is a spatial latent grid. The model rolls forward conditioned on
future rain/river forecasts (exogenous inputs), and can assimilate partial
observations (gauges, satellite flood maps, crowd reports) mid-rollout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k // 2
        self.gates = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, k, padding=p)
        self.cand = nn.Conv2d(in_ch + hid_ch, hid_ch, k, padding=p)

    def forward(self, x, s):
        zr = torch.sigmoid(self.gates(torch.cat([x, s], 1)))
        z, r = zr.chunk(2, 1)
        c = torch.tanh(self.cand(torch.cat([x, r * s], 1)))
        return (1 - z) * s + z * c


class FloodWorldModel(nn.Module):
    def __init__(self, hidden=24, static_ch=1, forcing_ch=2):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + static_ch, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1))
        self.cell1 = ConvGRUCell(forcing_ch + static_ch + 1, hidden)
        self.cell2 = ConvGRUCell(hidden, hidden)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1))

    def init_state(self, depth0, static):
        s = torch.tanh(self.encoder(torch.cat([depth0.unsqueeze(1), static], 1)))
        return (s, s)

    def assimilate(self, state, depth_obs, static, mask):
        """Blend an observation (B,H,W) into the latent state where mask (B,H,W) == 1."""
        enc = torch.tanh(self.encoder(torch.cat([depth_obs.unsqueeze(1), static], 1)))
        m = mask.unsqueeze(1)
        return tuple((1 - m) * s + m * enc for s in state)

    def step(self, state, prev_depth, static, forcing_t):
        x = torch.cat([forcing_t, static, prev_depth.unsqueeze(1)], 1)
        s1 = self.cell1(x, state[0])
        s2 = self.cell2(s1, state[1])
        # residual update + smooth non-negativity (a hard ReLU can die and output zeros forever)
        depth = F.softplus(prev_depth + self.decoder(s2).squeeze(1), beta=30.0)
        return (s1, s2), depth

    def forward(self, depth0, static, forcing, steps=None):
        """forcing: (B,T,C,H,W). Returns depth forecast (B,T,H,W)."""
        T = forcing.shape[1] if steps is None else steps
        state, prev, outs = self.init_state(depth0, static), depth0, []
        for t in range(T):
            state, prev = self.step(state, prev, static, forcing[:, t])
            outs.append(prev)
        return torch.stack(outs, 1)
