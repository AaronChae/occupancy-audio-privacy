#!/usr/bin/env python3
"""Per-slot audio features for occupancy baselines (HPDmobile).

For every 10-second slot this summarises each hub's clip with log-amplitude statistics,
aggregates them over the hubs of the home (max and mean), adds rolling context over the
previous 1, 5 and 30 minutes, adds time of day, and joins the home-level labels.

Input:  data/processed/hpdmobile/<home>/audio/<hub>/<day>.npy and labels/<day>.npz
Output: data/features/hpdmobile/<home>.csv.gz (one row per labelled slot with audio)

Example:
    python3 src/features_hpdmobile.py --home H1 H2 H4
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SLOTS = 8640
STATS = ("lmean", "lstd", "lmax", "lp10")
WINDOWS = {"r1": 6, "r5": 30, "r30": 180}  # slots of 10 s


def clip_stats(arr):
    """log10(x + 1) of per-clip mean, std, max and 10th percentile; NaN for missing clips."""
    with np.errstate(invalid="ignore"):
        return {
            "lmean": np.log10(arr.mean(axis=1) + 1),
            "lstd": np.log10(arr.std(axis=1) + 1),
            "lmax": np.log10(arr.max(axis=1) + 1),
            "lp10": np.log10(np.percentile(arr, 10, axis=1) + 1),
        }


def day_frame(home_dir, day):
    hub_files = sorted(home_dir.glob(f"audio/*/{day}.npy"))
    per_hub = [clip_stats(np.load(f)) for f in hub_files]
    cols = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slots
        for s in STATS:
            stack = np.vstack([h[s] for h in per_hub])
            cols[f"{s}_max"] = np.nanmax(stack, axis=0)
            cols[f"{s}_mean"] = np.nanmean(stack, axis=0)
        cols["n_hubs"] = np.sum(~np.isnan(np.vstack([h["lmean"] for h in per_hub])), axis=0)
    df = pd.DataFrame(cols)
    for name, w in WINDOWS.items():
        roll = df["lmean_max"].rolling(w, min_periods=1)
        df[f"{name}_mean"] = roll.mean()
        df[f"{name}_max"] = roll.max()
    hours = np.arange(SLOTS) * 10 / 3600
    df["tod_sin"] = np.sin(2 * np.pi * hours / 24)
    df["tod_cos"] = np.cos(2 * np.pi * hours / 24)
    labels = np.load(home_dir / "labels" / f"{day}.npz")
    df["occupied"] = labels["occupied"]
    df["number"] = labels["number"]
    df.insert(0, "slot", np.arange(SLOTS))
    df.insert(0, "day", day)
    keep = (df["occupied"] >= 0) & (df["n_hubs"] > 0)
    return df[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", nargs="+", required=True)
    ap.add_argument("--processed", type=Path, default=Path("data/processed/hpdmobile"))
    ap.add_argument("--out", type=Path, default=Path("data/features/hpdmobile"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for home in args.home:
        t0 = time.time()
        home_dir = args.processed / home
        days = sorted(p.stem for p in (home_dir / "labels").glob("*.npz")
                      if any(home_dir.glob(f"audio/*/{p.stem}.npy")))
        df = pd.concat([day_frame(home_dir, d) for d in days], ignore_index=True)
        df.insert(0, "home", home)
        df.to_csv(args.out / f"{home}.csv.gz", index=False, float_format="%.5g")
        print(f"{home}: {len(days)} days, {len(df)} slots, occupied {100 * df['occupied'].mean():.1f}%, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
