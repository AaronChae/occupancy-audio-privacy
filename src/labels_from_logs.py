#!/usr/bin/env python3
"""Occupancy intervals from entry and exit logs (sign-in sheets, door sensors, class schedules).

Input: a CSV with a column time (ISO timestamps) and either
    event   enter or exit, one person per row, or
    count   the number of people present from that time on.
Output: a CSV with columns start,end,occupied, the label format of src/pipeline.py: between two
consecutive log times the room counts as occupied when at least one person is present.

The room is taken as vacant before the first row unless --initial gives the number of people
present at that time; --end closes the last interval. Rows with the same time are applied in file
order. Exits that would make the count negative are clipped to zero and reported.

Example:
    python src/labels_from_logs.py door_log.csv --end 2026-11-05T18:00:00 --out labels/room1.csv
"""
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd


def intervals(df, initial=0, end=None):
    df = df.copy()
    df["t"] = [dt.datetime.fromisoformat(str(t)) for t in df["time"]]
    df = df.sort_values("t", kind="stable")
    count, prev, out, clipped = initial, None, [], 0
    for _, row in df.iterrows():
        if prev is not None and row["t"] > prev:
            out.append((prev, row["t"], int(count > 0)))
        if "count" in df:
            count = int(row["count"])
        else:
            ev = str(row["event"]).strip().lower()
            if ev == "enter":
                count += 1
            elif ev == "exit":
                if count == 0:
                    clipped += 1
                count = max(count - 1, 0)
            else:
                raise SystemExit(f"unknown event {row['event']!r} at {row['time']}")
        prev = row["t"]
    if end is not None and prev is not None and end > prev:
        out.append((prev, end, int(count > 0)))
    merged = []
    for a, b, o in out:   # join consecutive intervals with the same label
        if merged and merged[-1][2] == o and merged[-1][1] == a:
            merged[-1] = (merged[-1][0], b, o)
        else:
            merged.append((a, b, o))
    return merged, clipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--initial", type=int, default=0, help="people present before the first row")
    ap.add_argument("--end", default="", help="ISO time that closes the last interval")
    args = ap.parse_args()
    df = pd.read_csv(args.log)
    if "time" not in df or not ({"event", "count"} & set(df.columns)):
        raise SystemExit("the log needs a time column and an event or count column")
    out, clipped = intervals(df, args.initial, dt.datetime.fromisoformat(args.end) if args.end else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([(a.isoformat(), b.isoformat(), o) for a, b, o in out],
                 columns=["start", "end", "occupied"]).to_csv(args.out, index=False)
    occ = sum((b - a).total_seconds() for a, b, o in out if o)
    tot = sum((b - a).total_seconds() for a, b, o in out)
    print(f"{len(out)} intervals, {tot / 3600:.1f} h labelled, {occ / max(tot, 1):.1%} occupied -> {args.out}"
          + (f"; {clipped} exit(s) with nobody counted present were clipped" if clipped else ""))


if __name__ == "__main__":
    main()
