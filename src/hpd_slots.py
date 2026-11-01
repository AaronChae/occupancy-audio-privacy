#!/usr/bin/env python3
"""Per-slot envelope statistics for HPDmobile on a continuous timeline.

For each home this writes data/features/hpdmobile_slots/<home>.npz with
    x         float16 (n_slots, n_hubs, 6): log1p of mean, std, max, p10, p50, p90 of each
              10-s clip (800 envelope values); NaN where the clip is missing
    occupied  int8 (n_slots,): home-level label, -1 where missing
    start     first calendar day; slot k starts k * 10 s after that day's midnight
    hubs      hub names
Days without any data inside the recording period are kept as missing, so slot k + 8640
is always exactly one day after slot k.

Example:
    python3 src/hpd_slots.py --home H1 H2
"""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np

SLOTS = 8640
STATS = ("mean", "std", "max", "p10", "p50", "p90")


def clip_stats(a):
    """a: (8640, 800) envelope values with NaN rows for missing clips."""
    q = np.percentile(a, [10, 50, 90], axis=1)
    s = np.stack([a.mean(1), a.std(1), a.max(1), q[0], q[1], q[2]], 1)
    return np.log1p(np.maximum(s, 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", nargs="+", required=True)
    ap.add_argument("--processed", type=Path, default=Path("data/processed/hpdmobile"))
    ap.add_argument("--out", type=Path, default=Path("data/features/hpdmobile_slots"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for home in args.home:
        d = args.processed / home
        hubs = sorted(p.name for p in (d / "audio").iterdir() if p.is_dir())
        days = sorted({p.stem for p in (d / "labels").glob("*.npz")} |
                      {p.stem for p in (d / "audio").glob("*/*.npy")})
        first, last = (dt.date.fromisoformat(days[0]), dt.date.fromisoformat(days[-1]))
        n_days = (last - first).days + 1
        x = np.full((n_days * SLOTS, len(hubs), len(STATS)), np.nan, np.float16)
        occ = np.full(n_days * SLOTS, -1, np.int8)
        for i in range(n_days):
            day = (first + dt.timedelta(days=i)).isoformat()
            sl = slice(i * SLOTS, (i + 1) * SLOTS)
            for j, hub in enumerate(hubs):
                f = d / "audio" / hub / f"{day}.npy"
                if f.exists():
                    x[sl, j] = clip_stats(np.load(f)).astype(np.float16)
            f = d / "labels" / f"{day}.npz"
            if f.exists():
                occ[sl] = np.load(f)["occupied"]
        np.savez(args.out / f"{home}.npz", x=x, occupied=occ, start=first.isoformat(), hubs=np.array(hubs))
        print(f"{home}: {n_days} calendar days, {len(hubs)} hubs, audio {100 * np.mean(~np.isnan(x[:, :, 0])):.1f}%, "
              f"labels {100 * np.mean(occ >= 0):.1f}%, occupied {100 * np.mean(occ[occ >= 0] == 1):.1f}%", flush=True)


if __name__ == "__main__":
    main()
