#!/usr/bin/env python3
"""Check the time alignment between audio and occupancy labels in HPDmobile.

For each home the home-level loudness (max over hubs of the log mean envelope, averaged over
30 minutes) is compared with the occupancy label shifted by -12 h ... +12 h. A clear AUC peak
far from 0 h would suggest a clock or time-zone offset between the two streams.
A positive shift pairs the audio at time t with the label at time t + shift.

Example:
    python3 src/hpd_check_alignment.py --home H1 H2 H3 H4 H5 H6
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SLOT_S = 10


def smooth(v, w):
    m = ~np.isnan(v)
    s = np.convolve(np.nan_to_num(v), np.ones(w), "same")
    n = np.convolve(m.astype(float), np.ones(w), "same")
    return np.where(n > w / 4, s / np.maximum(n, 1), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", nargs="+", required=True)
    ap.add_argument("--slots", type=Path, default=Path("data/features/hpdmobile_slots"))
    ap.add_argument("--step-min", type=int, default=30)
    ap.add_argument("--max-hours", type=int, default=12)
    args = ap.parse_args()
    step = args.step_min * 60 // SLOT_S
    max_shift = args.max_hours * 3600 // SLOT_S
    for home in args.home:
        z = np.load(args.slots / f"{home}.npz")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loud = smooth(np.nanmax(z["x"][:, :, 0].astype(np.float32), 1), 180)
        y = z["occupied"]
        curve = []
        for s in range(-max_shift, max_shift + 1, step):
            a, b = (loud[: len(loud) - s], y[s:]) if s >= 0 else (loud[-s:], y[: len(y) + s])
            k = ~np.isnan(a) & (b >= 0)
            auc = roc_auc_score(b[k], a[k]) if len(np.unique(b[k])) == 2 else np.nan
            curve.append((s * SLOT_S / 3600, auc))
        h, auc = np.array(curve).T
        best = int(np.nanargmax(auc))
        print(f"{home}: AUC {auc[h == 0][0]:.3f} at 0 h, best {auc[best]:.3f} at {h[best]:+.1f} h | "
              + " ".join(f"{a:.2f}" for a in auc[::2]), flush=True)


if __name__ == "__main__":
    main()
