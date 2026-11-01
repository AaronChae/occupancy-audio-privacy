"""Check whether the DCASE 2016 Task 2 test mixtures reuse the released training sounds.

For every annotated event in a mixture, the mixture segment around the event is compared with
every released isolated training sound of the same class by the maximum normalized
cross-correlation (NCC) over all lags. A sound that was mixed in unchanged (up to gain) gives an
NCC close to 1; an unrelated recording of the same class stays low. The development mixtures,
which the challenge documents as built from the training sounds, serve as the positive control.

Output: experiments/summary/dcase2016_leak.csv (one row per event) and a summary on stdout.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).parent))
import office_data as od  # noqa: E402

MARGIN = 0.25  # seconds of context kept on each side of the annotated event


def ncc_max(seg, tpl):
    """Maximum normalized cross-correlation of tpl sliding over seg (both 1-D float arrays)."""
    if len(tpl) > len(seg):
        seg, tpl = tpl, seg
    tpl = tpl - tpl.mean()
    seg = seg - seg.mean()
    num = fftconvolve(seg, tpl[::-1], mode="valid")
    c = np.concatenate([[0.0], np.cumsum(seg ** 2)])
    win = np.maximum(c[len(tpl):] - c[: len(seg) - len(tpl) + 1], 0.0)  # windowed energy of seg
    den = np.sqrt(win * (tpl ** 2).sum()) + 1e-9
    return float(np.max(np.abs(num) / den))


def mixtures(d):
    out = []
    for w in sorted((d / "sound").glob("*.wav")):
        events = []
        for line in open(d / "annotation" / (w.stem + ".txt")):
            p = line.split()
            if len(p) >= 3:
                events.append((float(p[0]), float(p[1]), p[2]))
        out.append((w.stem, od.load(w).numpy(), events))
    return out


def main():
    raw = Path("data/raw/dcase2016_task2")
    train_dir = raw / "dcase2016_task2_train_dev/dcase2016_task2_train_dev/dcase2016_task2_train"
    bank = {}
    for p in sorted(train_dir.glob("*.wav")):
        c = re.fullmatch(r"([a-zA-Z]+?)(\d+)\.wav", p.name)[1]
        bank.setdefault(c, []).append((p.name, od.load(p).numpy()))
    print("training sounds:", {c: len(v) for c, v in bank.items()}, flush=True)

    sets = {}
    for d in sorted(raw.rglob("annotation")):
        if (d.parent / "sound").is_dir():
            sets[d.parent.name] = mixtures(d.parent)
    print("mixture sets:", {k: len(v) for k, v in sets.items()}, flush=True)

    rows = []
    for name, mixes in sets.items():
        for stem, x, events in mixes:
            for on, off, c in events:
                if c not in bank:
                    continue
                a, b = max(0, int((on - MARGIN) * od.SR)), min(len(x), int((off + MARGIN) * od.SR))
                seg = x[a:b]
                best = max((ncc_max(seg, t), fn) for fn, t in bank[c])
                rows.append({"set": name, "mixture": stem, "onset": on, "offset": off, "class": c,
                             "best_ncc": round(best[0], 4), "best_train_file": best[1]})
        v = np.array([r["best_ncc"] for r in rows if r["set"] == name])
        print("%s: %d events, NCC median %.2f, share > 0.9: %.2f, > 0.7: %.2f, > 0.5: %.2f"
              % (name, len(v), np.median(v), (v > 0.9).mean(), (v > 0.7).mean(), (v > 0.5).mean()), flush=True)

    out = Path("experiments/summary/dcase2016_leak.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("leak check done", flush=True)


if __name__ == "__main__":
    main()
