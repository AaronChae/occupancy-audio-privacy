#!/usr/bin/env python3
"""Occupancy baselines on HPDmobile per-slot features.

Feature sets
    time   : time of day only (no audio), to show how much audio adds
    audio  : log-amplitude statistics over hubs plus 1/5/30-minute rolling context
    both   : audio + time

Models: majority class, logistic regression, histogram gradient boosting.

Protocols
    loho      : leave one home out (train on the other homes, test on the held-out home)
    timesplit : within each home, train on the first 70% of days, test on the rest

Example:
    python3 src/baseline_occupancy.py --home H1 H2 H4 --protocol loho
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AUDIO = [f"{s}_{a}" for s in ("lmean", "lstd", "lmax", "lp10") for a in ("max", "mean")] + \
        [f"{w}_{a}" for w in ("r1", "r5", "r30") for a in ("mean", "max")]
TIME = ["tod_sin", "tod_cos"]
FEATURE_SETS = {"time": TIME, "audio": AUDIO, "both": AUDIO + TIME}


def models():
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "hgb": lambda: HistGradientBoostingClassifier(random_state=0),
    }


def scores(y, p):
    occ, emp = y == 1, y == 0
    return {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "f1_occupied": f1_score(y, p, pos_label=1, zero_division=0),
        "missed_occupied": float((p[occ] == 0).mean()) if occ.any() else np.nan,   # "false off"
        "false_occupied": float((p[emp] == 1).mean()) if emp.any() else np.nan,    # "false on"
    }


def splits(df, protocol, homes):
    if protocol == "loho":
        for h in homes:
            yield h, df[df.home != h], df[df.home == h]
    else:
        for h in homes:
            d = df[df.home == h]
            days = sorted(d.day.unique())
            cut = days[int(len(days) * 0.7)]
            yield h, d[d.day < cut], d[d.day >= cut]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", nargs="+", required=True)
    ap.add_argument("--protocol", choices=("loho", "timesplit"), required=True)
    ap.add_argument("--features", type=Path, default=Path("data/features/hpdmobile"))
    ap.add_argument("--out", type=Path, default=Path("experiments/baseline_v0"))
    ap.add_argument("--test-home", nargs="*",
                    help="only run these folds and append them to the results file (to split long runs)")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every n-th slot (1 = every 10 s, 6 = every minute) to speed up training")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(args.features / f"{h}.csv.gz") for h in args.home], ignore_index=True)
    if args.stride > 1:
        df = df[df.slot % args.stride == 0]
    rows = []
    for test_home, train, test in splits(df, args.protocol, args.home):
        if args.test_home and test_home not in args.test_home:
            continue
        y_tr, y_te = train.occupied.to_numpy(), test.occupied.to_numpy()
        majority = int(np.bincount(y_tr).argmax())
        rows.append({"protocol": args.protocol, "test_home": test_home, "features": "-", "model": "majority",
                     "n_train": len(train), "n_test": len(test), **scores(y_te, np.full_like(y_te, majority))})
        for fs, cols in FEATURE_SETS.items():
            x_tr = train[cols].fillna(0).to_numpy()
            x_te = test[cols].fillna(0).to_numpy()
            for name, make in models().items():
                t0 = time.time()
                p = make().fit(x_tr, y_tr).predict(x_te)
                rows.append({"protocol": args.protocol, "test_home": test_home, "features": fs, "model": name,
                             "n_train": len(train), "n_test": len(test), **scores(y_te, p),
                             "fit_s": round(time.time() - t0, 1)})
                print(f"{args.protocol} {test_home} {fs:5s} {name:6s} "
                      f"bal_acc={rows[-1]['balanced_accuracy']:.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res["stride"] = args.stride
    path = args.out / f"results_{args.protocol}.csv"
    if args.test_home and path.exists():
        res.to_csv(path, mode="a", header=False, index=False, float_format="%.4f")
    else:
        res.to_csv(path, index=False, float_format="%.4f")
    print(res.groupby(["features", "model"])[["accuracy", "balanced_accuracy", "f1_occupied",
                                               "missed_occupied", "false_occupied"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
