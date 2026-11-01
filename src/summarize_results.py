#!/usr/bin/env python3
"""Collect the experiment outputs into small summary tables.

Writes experiments/summary/office_utility.csv, privacy.csv and hpd_loho.csv
(the HPDmobile table is the mean over the six held-out homes per run and method).

Example:
    python3 src/summarize_results.py
"""
import glob
import json
from pathlib import Path

import pandas as pd


def main():
    exp, out = Path("experiments"), Path("experiments/summary")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(exp.glob("office_v*/*/results.json")):
        for test_set, r in json.load(open(f))["utility"].items():
            rows.append({"version": f.parts[-3], "representation": f.parts[-2], "test_set": test_set,
                         **{k: r[k] for k in ("human_auc", "human_f1", "macro_auc", "seconds", "human_rate")}})
    pd.DataFrame(rows).to_csv(out / "office_utility.csv", index=False, float_format="%.4f")
    rows = []
    for f in sorted(exp.glob("privacy_v*/*/privacy.json")):
        r = json.load(open(f))
        rows.append({"version": f.parts[-3], "representation": f.parts[-2],
                     **{k: v for k, v in r.items() if isinstance(v, float) and k != "seconds"}})
    pd.DataFrame(rows).to_csv(out / "privacy.csv", index=False, float_format="%.4f")
    rows = []
    for f in sorted(glob.glob(str(exp / "hpd_ts2vec_v0" / "results_*.csv"))):
        run = Path(f).stem.replace("results_", "")
        m = pd.read_csv(f).groupby("method")[["balanced_accuracy", "f1_occupied", "auc"]].mean().reset_index()
        m.insert(0, "run", run)
        rows.append(m)
    pd.concat(rows).to_csv(out / "hpd_loho.csv", index=False, float_format="%.4f")
    print("summaries written to", out)


if __name__ == "__main__":
    main()
