#!/usr/bin/env python3
"""Convert HPDmobile audio and ground truth into per hub-day arrays.

Audio CSV clips are read directly from the downloaded zip files (no extraction).
For each home, hub and day this writes

    <out>/<home>/audio/<hub>/<YYYY-MM-DD>.npy   float32 (8640, 800), NaN rows for missing clips
    <out>/<home>/labels/<YYYY-MM-DD>.npz        occupied and number, int8 (8640,), -1 if missing
    <out>/<home>/manifest.csv                   clip counts per hub-day

Row k of a day is the 10-second slot that starts k * 10 seconds after midnight.
Each clip holds 800 values (80 Hz x 10 s) of the released audio signal.

Example:
    python3 src/preprocess_hpdmobile.py --home H1 H2 H4
    python3 src/preprocess_hpdmobile.py --home H1 --hub RS1 --day 2019-12-07
"""
import argparse
import csv
import os
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

SLOTS = 8640    # 10-second slots per day
SAMPLES = 800   # values per clip (80 Hz x 10 s)
AUDIO_RE = re.compile(
    r"^(H\d)_(\w+?)_AUDIO/(\d{4}-\d{2}-\d{2})/\d{4}/"
    r"\d{4}-\d{2}-\d{2}_(\d{2})(\d{2})(\d{2})_\w+?_H\d\.csv$"
)
LABEL_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(H\d)_groundtruth\.csv$")


def slot(hh, mm, ss):
    return (int(hh) * 3600 + int(mm) * 60 + int(ss)) // 10


def convert_labels(raw, home, out, days=None):
    """Write one labels file per day; return {day: number of labelled slots}."""
    counts = {}
    with zipfile.ZipFile(raw / f"{home}_GROUNDTRUTH.zip") as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            m = LABEL_RE.match(base)
            if "__MACOSX" in name or not m or (days and m[1] not in days):
                continue
            day = m[1]
            occupied = np.full(SLOTS, -1, np.int8)
            number = np.full(SLOTS, -1, np.int8)
            for row in csv.DictReader(z.read(name).decode("utf-8-sig").splitlines()):
                hh, mm, ss = row["timestamp"].split(" ")[1].split(":")
                k = slot(hh, mm, ss)
                occupied[k] = int(float(row["occupied"]))
                number[k] = int(float(row["number"]))
            path = out / home / "labels" / f"{day}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, occupied=occupied, number=number)
            counts[day] = int((occupied >= 0).sum())
    return counts


def is_complete(path):
    """True if path holds a full (SLOTS, SAMPLES) float32 array (not a cut-off write)."""
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == (SLOTS, SAMPLES) and arr.dtype == np.float32
    except Exception:
        return False


def save_array(path, arr):
    """Write via a temporary name so an interrupted run never leaves a cut-off final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        np.save(path, arr)  # rewrite a damaged file in place
        return
    tmp = path.with_name(path.stem + ".partial.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def convert_audio(raw, home, out, hubs=None, days=None, overwrite=False, deadline=None):
    """Write one audio array per hub-day; return a list of per hub-day stats.

    Hub-days left when the deadline (time.time() value) passes are reported as "pending".
    """
    stats = []
    with zipfile.ZipFile(raw / f"{home}_AUDIO.zip") as z:
        groups = defaultdict(list)
        for info in z.infolist():
            m = AUDIO_RE.match(info.filename)
            if not m:
                continue
            _, hub, day, hh, mm, ss = m.groups()
            if (hubs and hub not in hubs) or (days and day not in days):
                continue
            groups[(hub, day)].append((slot(hh, mm, ss), info))

        for (hub, day), items in sorted(groups.items()):
            path = out / home / "audio" / hub / f"{day}.npy"
            if not overwrite and is_complete(path):
                stats.append({"hub": hub, "day": day, "clips": len(items), "status": "skipped"})
                continue
            if deadline and time.time() > deadline:
                stats.append({"hub": hub, "day": day, "clips": len(items), "status": "pending"})
                continue
            arr = np.full((SLOTS, SAMPLES), np.nan, np.float32)
            seen = np.zeros(SLOTS, bool)
            bad = dup = 0
            for k, info in items:
                values = np.array(z.read(info).split(), dtype=np.float32)
                if values.size != SAMPLES:
                    bad += 1
                    fixed = np.full(SAMPLES, np.nan, np.float32)
                    fixed[: min(SAMPLES, values.size)] = values[:SAMPLES]
                    values = fixed
                if seen[k]:
                    dup += 1
                seen[k] = True
                arr[k] = values
            save_array(path, arr)
            stats.append({"hub": hub, "day": day, "clips": int(seen.sum()),
                          "bad_length": bad, "duplicates": dup, "status": "written"})
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", nargs="+", required=True, help="homes to convert, e.g. H1 H2")
    ap.add_argument("--hub", nargs="*", help="only these hubs, e.g. RS1")
    ap.add_argument("--day", nargs="*", help="only these days, e.g. 2019-12-07")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/hpdmobile"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/hpdmobile"))
    ap.add_argument("--overwrite", action="store_true", help="rewrite existing audio arrays")
    ap.add_argument("--max-seconds", type=float,
                    help="stop starting new hub-days after this many seconds; re-run to continue")
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + args.max_seconds if args.max_seconds else None
    for home in args.home:
        t0 = time.time()
        labels = convert_labels(args.raw, home, args.out, args.day)
        stats = convert_audio(args.raw, home, args.out, args.hub, args.day, args.overwrite, deadline)
        manifest = args.out / home / "manifest.csv"
        new = not manifest.exists()
        with open(manifest, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["home", "hub", "day", "clips", "missing", "bad_length",
                            "duplicates", "label_slots"])
            for s in stats:
                if s["status"] != "written":
                    continue
                clips = s["clips"]
                w.writerow([home, s["hub"], s["day"], clips, SLOTS - clips, s["bad_length"],
                            s["duplicates"], labels.get(s["day"], 0)])
        count = {k: sum(s["status"] == k for s in stats) for k in ("written", "skipped", "pending")}
        print(f"{home}: {len(labels)} label days, {count['written']} hub-days written, "
              f"{count['skipped']} already done, {count['pending']} pending, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
