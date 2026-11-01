#!/usr/bin/env python3
"""Self-supervised temporal occupancy model for HPDmobile (TS2Vec-style, causal).

Each leave-one-home-out fold
  1. pretrains a causal dilated CNN on unlabeled slot sequences with the TS2Vec hierarchical
     contrastive loss (two overlapping random crops + timestamp masking), and
  2. evaluates home-level occupancy on the held-out home with
       probe    : logistic regression on frozen representations + time of day
       finetune : pretrained encoder + linear head, trained with labels
       scratch  : the same network trained with labels only
       feat_lr / feat_hgb : hand-crafted features (inputs, trailing means, time of day)

Input per 10-s slot: envelope statistics from src/hpd_slots.py aggregated over hubs (max, mean),
standardized per home with the median and IQR of that home's own audio (no labels used).
With --transductive the encoder is pretrained once on the unlabeled audio of all homes.

Example:
    python3 src/hpd_ts2vec.py --out experiments/hpd_ts2vec_v0
"""
import argparse
import copy
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HOMES = ["H1", "H2", "H3", "H4", "H5", "H6"]
SLOTS = 8640


def load_home(path):
    z = np.load(path)
    x = z["x"].astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = np.concatenate([np.nanmax(x, 1), np.nanmean(x, 1)], 1)
        med = np.nanmedian(f, 0)
        iqr = np.nanpercentile(f, 75, 0) - np.nanpercentile(f, 25, 0)
    f = (f - med) / np.maximum(iqr, 1e-3)
    miss = np.isnan(f[:, :1]).astype(np.float32)
    cover = np.mean(~np.isnan(x[:, :, 0]), 1, keepdims=True).astype(np.float32)
    feats = np.concatenate([np.nan_to_num(f), cover, miss], 1).astype(np.float32)
    ang = 2 * np.pi * (np.arange(len(f)) % SLOTS) / SLOTS
    tod = np.stack([np.sin(ang), np.cos(ang)], 1).astype(np.float32)
    return feats, tod, z["occupied"].astype(np.int64)


class Block(nn.Module):
    def __init__(self, c, d, k=3):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(c, c, k, dilation=d)
        self.c2 = nn.Conv1d(c, c, k, dilation=d)

    def forward(self, x):
        h = self.c1(F.pad(F.gelu(x), (self.pad, 0)))
        return x + self.c2(F.pad(F.gelu(h), (self.pad, 0)))


class Encoder(nn.Module):
    """Causal TS2Vec-style encoder (B, T, cin) -> (B, T, cout); receptive field 4093 slots (11.4 h)."""

    def __init__(self, cin, hid=64, cout=64, depth=10):
        super().__init__()
        self.inp = nn.Linear(cin, hid)
        self.blocks = nn.Sequential(*[Block(hid, 2 ** i) for i in range(depth)])
        self.out = nn.Conv1d(hid, cout, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, mask=False):
        h = self.inp(x)
        if mask:  # timestamp masking
            h = h * (torch.rand(h.shape[:2], device=h.device) > 0.5).unsqueeze(-1)
        return self.drop(self.out(self.blocks(h.transpose(1, 2)))).transpose(1, 2)


def contrast(z1, z2, dim):
    """TS2Vec instance-wise (dim=0) or temporal (dim=1) contrastive loss for (B, T, C) inputs."""
    if dim == 0:
        z1, z2 = z1.transpose(0, 1), z2.transpose(0, 1)
    n = z1.size(1)
    if n == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], 1)
    sim = z @ z.transpose(1, 2)
    logits = -F.log_softmax(torch.tril(sim, -1)[:, :, :-1] + torch.triu(sim, 1)[:, :, 1:], -1)
    i = torch.arange(n, device=z.device)
    return (logits[:, i, n + i - 1].mean() + logits[:, n + i, i].mean()) / 2


def hier_loss(z1, z2, alpha=0.5):
    loss, d = 0.0, 0
    while z1.size(1) > 1:
        loss = loss + alpha * contrast(z1, z2, 0) + (1 - alpha) * contrast(z1, z2, 1)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), 2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), 2).transpose(1, 2)
    return (loss + alpha * contrast(z1, z2, 0)) / (d + 1)


def windows(data, L, B, rng, need_labels):
    xs, ts, ys = [], [], []
    while len(xs) < B:
        f, t, y = data[rng.integers(len(data))]
        s = rng.integers(0, len(f) - L)
        if f[s:s + L, -1].mean() > 0.5 or (need_labels and (y[s:s + L] >= 0).mean() < 0.5):
            continue
        xs.append(f[s:s + L]); ts.append(t[s:s + L]); ys.append(y[s:s + L])
    return [torch.from_numpy(np.stack(a)) for a in (xs, ts, ys)]


def limit_labels(y, n_days, rng):
    """Keep the labels of n_days random labelled days (0 = keep all)."""
    if n_days <= 0:
        return y
    days = np.unique(np.nonzero(y >= 0)[0] // SLOTS)
    keep = rng.choice(days, min(n_days, len(days)), replace=False)
    return np.where(np.isin(np.arange(len(y)) // SLOTS, keep), y, -1)


def pretrain(data, args, dev, rng):
    enc = Encoder(data[0][0].shape[1]).to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
    L, log = args.crop, []
    for it in range(args.ssl_iters):
        x = windows(data, L, args.batch, rng, False)[0].to(dev)
        crop = int(rng.integers(2, L + 1))
        left = int(rng.integers(0, L - crop + 1)); right = left + crop
        eleft = int(rng.integers(0, left + 1)); eright = int(rng.integers(right, L + 1))
        z1 = enc(x[:, eleft:right], mask=True)[:, -crop:]
        z2 = enc(x[:, left:eright], mask=True)[:, :crop]
        loss = hier_loss(z1, z2)
        opt.zero_grad(); loss.backward(); opt.step()
        log.append(loss.item())
    return enc, log


class Occupancy(nn.Module):
    def __init__(self, enc, rdim=64):
        super().__init__()
        self.enc, self.head = enc, nn.Linear(rdim + 2, 1)

    def forward(self, x, tod):
        return self.head(torch.cat([self.enc(x), tod], -1)).squeeze(-1)


def train_sup(model, data, args, dev, rng):
    pos = np.mean(np.concatenate([y[y >= 0] for _, _, y in data]))
    pw = torch.tensor((1 - pos) / pos, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_sup, weight_decay=1e-2)
    L, warm = args.sup_len, args.sup_len // 3  # no loss on the first third (little causal context)
    for it in range(args.sup_iters):
        x, t, y = (a.to(dev) for a in windows(data, L, args.batch, rng, True))
        m = y >= 0
        m[:, :warm] = False
        loss = F.binary_cross_entropy_with_logits(model(x, t)[m], y[m].float(), pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def run(model_fn, x, dev):
    return model_fn(torch.from_numpy(x)[None].to(dev))[0].cpu().numpy()


def scores(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"balanced_accuracy": balanced_accuracy_score(y, yhat), "f1_occupied": f1_score(y, yhat),
            "auc": roc_auc_score(y, p)}


def trailing(f, w):
    c = np.cumsum(np.vstack([np.zeros((1, f.shape[1])), f]), 0)
    idx = np.arange(1, len(f) + 1)
    return (c[idx] - c[np.maximum(idx - w, 0)]) / np.minimum(idx, w)[:, None]


def hand_features(f, t):
    base = f[:, [0, 2, 6, 8]]  # max over hubs of mean/max, mean over hubs of mean/max
    return np.concatenate([f, t] + [trailing(base, w) for w in (6, 30, 180, 720)], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", type=Path, default=Path("data/features/hpdmobile_slots"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-home", nargs="*", default=HOMES)
    ap.add_argument("--transductive", action="store_true")
    ap.add_argument("--ssl-iters", type=int, default=600)
    ap.add_argument("--crop", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sup-iters", type=int, default=800)
    ap.add_argument("--sup-len", type=int, default=6144)
    ap.add_argument("--lr-sup", type=float, default=1e-3)
    ap.add_argument("--stride", type=int, default=6, help="label subsampling for probe and feature models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-days", type=int, default=0, help="labelled days kept per training home (0 = all)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = {h: load_home(args.slots / f"{h}.npz") for h in HOMES}
    shared = pretrain(list(data.values()), args, dev, rng) if args.transductive else None
    rows, ssl_logs = [], {}
    for test in args.test_home:
        t0 = time.time()
        lrng = np.random.default_rng(1000 + args.seed)
        train = [(f, t, limit_labels(y, args.label_days, lrng)) for f, t, y in (data[h] for h in HOMES if h != test)]
        ft, tt, yt = data[test]
        lab = yt >= 0
        enc, log = shared if shared else pretrain(train, args, dev, rng)
        ssl_logs[test] = log
        preds = {}
        # probe: frozen representations + time of day
        enc.eval()
        rep = lambda f: run(enc, f, dev)
        xs = np.concatenate([np.concatenate([rep(f), t], 1)[y >= 0][::args.stride] for f, t, y in train])
        ys = np.concatenate([y[y >= 0][::args.stride] for _, _, y in train])
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        preds["probe"] = probe.fit(xs, ys).predict_proba(np.concatenate([rep(ft), tt], 1)[lab])[:, 1]
        # finetune and scratch
        for name, e in (("finetune", copy.deepcopy(enc)), ("scratch", Encoder(ft.shape[1]).to(dev))):
            model = Occupancy(e).to(dev)
            model.train()
            train_sup(model, train, args, dev, rng).eval()
            with torch.no_grad():
                logit = model(torch.from_numpy(ft)[None].to(dev), torch.from_numpy(tt)[None].to(dev))[0]
            preds[name] = torch.sigmoid(logit).cpu().numpy()[lab]
        # hand-crafted features
        xs = np.concatenate([hand_features(f, t)[y >= 0][::args.stride] for f, t, y in train])
        xt = hand_features(ft, tt)[lab]
        for name, m in (("feat_lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
                        ("feat_hgb", HistGradientBoostingClassifier(class_weight="balanced", random_state=0))):
            preds[name] = m.fit(xs, ys).predict_proba(xt)[:, 1]
        for name, p in preds.items():
            rows.append({"test_home": test, "method": name, "n_test": int(lab.sum()), **scores(yt[lab], p)})
            print(f"{test} {name:8s} " + " ".join(f"{k}={v:.3f}" for k, v in rows[-1].items() if isinstance(v, float)), flush=True)
        print(f"{test}: {time.time() - t0:.0f} s", flush=True)
    res = pd.DataFrame(rows)
    tag = ("transductive" if args.transductive else "strict") + (f"_ld{args.label_days}" if args.label_days else "")
    res.to_csv(args.out / f"results_{tag}_seed{args.seed}.csv", index=False, float_format="%.4f")
    pd.DataFrame(ssl_logs).to_csv(args.out / f"ssl_loss_{tag}_seed{args.seed}.csv", index=False, float_format="%.4f")
    print(res.groupby("method")[["balanced_accuracy", "f1_occupied", "auc"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
