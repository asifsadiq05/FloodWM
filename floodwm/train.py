"""Train the world model: multi-step rollout loss, weighted toward deeper water."""
import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import load_config, ROOT
from .data import FloodDataset, generate_split
from .world_model import FloodWorldModel


def rollout_loss(model, static, forcing, depth, rollout_steps):
    pred = model(depth[:, 0], static, forcing[:, :rollout_steps])
    tgt = depth[:, 1:rollout_steps + 1]
    w = 1.0 + 4.0 * (tgt > 0.1).float()          # emphasise flooded cells
    return (w * torch.nn.functional.smooth_l1_loss(pred, tgt, reduction="none", beta=0.05)).mean()


def train(cfg, out_path):
    tc, dev = cfg["train"], torch.device(cfg["train"]["device"])
    torch.manual_seed(cfg["data"]["seed"])
    tr = generate_split(cfg, cfg["data"]["n_train"], cfg["data"]["seed"])
    va = generate_split(cfg, cfg["data"]["n_val"], cfg["data"]["seed"] + 1)
    dl = DataLoader(FloodDataset(tr), batch_size=tc["batch_size"], shuffle=True)
    dv = DataLoader(FloodDataset(va), batch_size=tc["batch_size"])
    model = FloodWorldModel(cfg["model"]["hidden_channels"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tc["epochs"])
    best = float("inf")
    for ep in range(tc["epochs"]):
        t0, tot = time.time(), 0.0
        model.train()
        # curriculum: lengthen the rollout as training progresses
        rs = min(tc["rollout_steps"], 4 + 4 * ep)
        for static, forcing, depth in dl:
            static, forcing, depth = static.to(dev), forcing.to(dev), depth.to(dev)
            loss = rollout_loss(model, static, forcing, depth, rs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()
        sched.step()
        model.eval()
        vt = 0.0
        with torch.no_grad():
            for static, forcing, depth in dv:
                vt += rollout_loss(model, static.to(dev), forcing.to(dev), depth.to(dev),
                                   cfg["sim"]["steps"]).item()
        vt /= len(dv)
        print(f"epoch {ep+1:02d} rollout={rs:02d} train={tot/len(dl):.5f} val={vt:.5f} ({time.time()-t0:.1f}s)")
        if vt < best:
            best = vt
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg}, out_path)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "world_model.pt"))
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    train(cfg, a.out)
