#!/usr/bin/env python3
"""End-to-end pipeline: one audio recording -> stage 1 (on-device encoder) -> stage 2 (temporal model) -> occupancy.

The two stages of the paper are trained and evaluated separately because no public dataset pairs
office audio with occupancy labels. This script joins them the way a deployment would:

    stage1    audio file(s) -> per-second embeddings z, sound-class probabilities p and the
              human-activity score s (Eq. 2), computed in 10-s blocks exactly as on the device
              (the noise floor of Eq. 1 is taken over each block). Written as .npz streams.
    pretrain  stage-1 streams, no labels -> temporal encoder pretrained with the TS2Vec loss (stage2.pt)
    fit       stage-1 streams + occupancy intervals -> linear probe on the frozen encoder (probe.joblib)
    predict   audio file or stage-1 stream (+ stage2.pt + probe.joblib) -> occupancy probability per 10-s slot
    evaluate  predictions + occupancy intervals -> AUC, F1 and balanced accuracy

Stage 2 reads, per 10-s slot, the mean and the maximum of the per-second embeddings, the mean,
maximum and active fraction of the human-activity score, the coverage of the slot and a missing
flag (2d + 5 channels; 69 for d = 32), standardized per site with the median and interquartile
range of that site's own stream, without labels (as for the homes in the paper). The temporal
encoder, the TS2Vec loss and the probe are the ones of src/hpd_ts2vec.py.

Occupancy labels are interval files: a CSV with columns start,end,occupied (ISO timestamps,
occupied 0 or 1); src/labels_from_logs.py builds one from entry and exit logs. A slot takes the
label of the interval that contains its whole 10 s; other slots are unlabelled.

Examples:
    python src/pipeline.py stage1 --ckpt experiments/office_v1/learned_floor_adv3/model.pt \
        --start 2026-11-05T09:00:00 --out streams/room1 recordings/room1_day1.wav
    python src/pipeline.py pretrain --streams streams/room1 streams/room2 --out models/stage2.pt
    python src/pipeline.py fit --streams streams/room1 --labels labels/room1.csv \
        --stage2 models/stage2.pt --out models/probe.joblib
    python src/pipeline.py predict --ckpt .../model.pt --stage2 models/stage2.pt --probe models/probe.joblib \
        --start 2026-11-06T09:00:00 --out pred.csv recordings/room1_day2.wav
    python src/pipeline.py evaluate --pred pred.csv --labels labels/room1_day2.csv

No stage-2 model trained on embedding streams exists yet: it needs the office recordings planned
in the paper. Until then, stage1 and predict without --stage2 give the per-second and per-slot
human-activity scores of the trained encoder.
"""
import argparse
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpd_ts2vec as ht   # noqa: E402
import office_data as od   # noqa: E402
import office_models as om   # noqa: E402

SR = om.SR
BLOCK = 10          # seconds per on-device block and per stage-2 slot
DAY = 86400
HUMAN = od.HUMAN.numpy()


# ----------------------------------------------------------------------------------------- stage 1
def make_rep(name, d, floor):
    if name == "learned":
        return om.SecondEncoder(d, floor=floor)
    if name == "logmel_stats":
        return om.LogMelStats(floor)
    if name == "envelope":
        return om.Envelope()
    raise ValueError(name)


def load_stage1(ckpt, dev):
    """Representation and activity head from a train_office.py checkpoint."""
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    a = ck["args"]
    rep = make_rep(a["rep"], a["dim"], a.get("floor", False))
    rep.load_state_dict(ck["rep"])
    d = a["dim"] if a["rep"] == "learned" else rep.dim
    head = om.EventHead(d, len(od.CLASSES))
    head.load_state_dict(ck["head"])
    return rep.to(dev).eval(), head.to(dev).eval(), a


@torch.no_grad()
def stage1_stream(wav, rep, head, dev, block=BLOCK):
    """wav (N,) at 16 kHz -> z (T, d), p (T, 16), s (T,) for the T whole seconds, block by block."""
    zs, ps = [], []
    n = block * SR
    for start in range(0, (wav.shape[0] // SR) * SR, n):
        seg = wav[start: start + n]
        if seg.shape[0] < SR:
            break
        z = rep(seg[None].to(dev))               # (1, t, d), noise floor over this block only
        zs.append(z[0].float().cpu().numpy())
        ps.append(torch.sigmoid(head(z))[0].float().cpu().numpy())
    z, p = np.concatenate(zs), np.concatenate(ps)
    return z, p, p[:, HUMAN].max(1)


def parse_time(s):
    return dt.datetime.fromisoformat(s) if s else None


def save_stream(path, z, p, s, start, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, z=z.astype(np.float16), p=p.astype(np.float16), s=s.astype(np.float32),
                        start=np.array(start.isoformat() if start else ""), meta=np.array(json.dumps(meta)))


def load_streams(paths):
    """Stage-1 .npz files -> one stream per site folder, seconds placed by start time (NaN in gaps)."""
    files = []
    for p in paths:
        files += sorted(Path(p).glob("*.npz")) if Path(p).is_dir() else [Path(p)]
    parts = []
    for f in files:
        z = np.load(f)
        start = parse_time(str(z["start"]))
        parts.append((start, z["z"].astype(np.float32), z["s"].astype(np.float32), f))
    if any(st is None for st, *_ in parts):
        if len(parts) > 1:
            raise SystemExit("streams without a start time cannot be placed on one timeline; pass one file")
        st, z, s, _ = parts[0]
        return z, s, None
    parts.sort(key=lambda t: t[0])
    t0 = parts[0][0]
    n = max(int((st - t0).total_seconds()) + len(z) for st, z, _, _ in parts)
    Z = np.full((n, parts[0][1].shape[1]), np.nan, np.float32)
    S = np.full(n, np.nan, np.float32)
    for st, z, s, f in parts:
        o = int((st - t0).total_seconds())
        if not np.all(np.isnan(S[o: o + len(s)])):
            warnings.warn(f"{f.name} overlaps an earlier stream; later file wins")
        Z[o: o + len(z)], S[o: o + len(s)] = z, s
    return Z, S, t0


# ----------------------------------------------------------------------------------------- slots
def slot_features(z, s, start, slot=BLOCK):
    """Per-slot stage-2 input (n, 2d + 5), time of day (n, 2) and slot start times.

    Channels: mean and max of z over the seconds present, mean, max and active fraction (s >= 0.5)
    of the human-activity score, coverage (seconds present / slot) and a missing flag; the 2d + 3
    value channels are standardized by the median and IQR of this stream (no labels)."""
    n = -(-len(s) // slot)
    pad = n * slot - len(s)
    z = np.concatenate([z, np.full((pad, z.shape[1]), np.nan, np.float32)]).reshape(n, slot, -1)
    s = np.concatenate([s, np.full(pad, np.nan, np.float32)]).reshape(n, slot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = np.concatenate([np.nanmean(z, 1), np.nanmax(z, 1), np.nanmean(s, 1, keepdims=True),
                            np.nanmax(s, 1, keepdims=True), np.nanmean(s >= 0.5, 1, keepdims=True)], 1)
        present = ~np.isnan(s)
        cover = present.mean(1, keepdims=True).astype(np.float32)
        miss = (cover == 0).astype(np.float32)
        f[cover[:, 0] == 0] = np.nan
        med = np.nanmedian(f, 0)
        iqr = np.nanpercentile(f, 75, 0) - np.nanpercentile(f, 25, 0)
    f = (f - med) / np.maximum(iqr, 1e-3)
    feats = np.concatenate([np.nan_to_num(f), cover, miss], 1).astype(np.float32)
    if start is None:
        return feats, np.zeros((n, 2), np.float32), None
    times = [start + dt.timedelta(seconds=k * slot) for k in range(n)]
    ang = 2 * np.pi * np.array([(t.hour * 3600 + t.minute * 60 + t.second) / DAY for t in times])
    tod = np.stack([np.sin(ang), np.cos(ang)], 1).astype(np.float32)
    return feats, tod, times


def slot_stat(s, fn, slot=BLOCK):
    """fn (nanmean or nanmax) of the per-second score over each slot."""
    pad = (-len(s)) % slot
    a = np.concatenate([s, np.full(pad, np.nan, np.float32)]).reshape(-1, slot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, 1)


def read_intervals(path):
    df = pd.read_csv(path)
    return [(parse_time(a), parse_time(b), int(o)) for a, b, o in zip(df["start"], df["end"], df["occupied"])]


def slot_labels(times, intervals, slot=BLOCK):
    """Label of the interval that contains the whole slot, else -1."""
    y = np.full(len(times), -1, np.int64)
    for k, t in enumerate(times):
        for a, b, o in intervals:
            if a <= t and t + dt.timedelta(seconds=slot) <= b:
                y[k] = o
                break
    return y


def site_data(stream_dirs, labels=None):
    """[(feats, tod, y)] per site folder, as hpd_ts2vec expects."""
    out = []
    for i, d in enumerate(stream_dirs):
        z, s, start = load_streams([d])
        feats, tod, times = slot_features(z, s, start)
        y = np.full(len(feats), -1, np.int64)
        if labels:
            if times is None:
                raise SystemExit("labelled streams need a start time")
            y = slot_labels(times, read_intervals(labels[i]))
        out.append((feats, tod, y))
    return out


# ----------------------------------------------------------------------------------------- stage 2
def load_stage2(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    enc = ht.Encoder(ck["cin"]).to(dev)
    enc.load_state_dict(ck["enc"])
    return enc.eval(), ck.get("meta", {})


def representations(enc, feats, dev):
    with torch.no_grad():
        return enc(torch.from_numpy(feats)[None].to(dev))[0].cpu().numpy()


def cmd_stage1(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rep, head, meta = load_stage1(args.ckpt, dev)
    start = parse_time(args.start)
    for audio in args.audio:
        wav = od.load(audio)
        z, p, s = stage1_stream(wav, rep, head, dev)
        out = args.out / (Path(audio).stem + ".npz")
        save_stream(out, z, p, s, start, {"ckpt": str(args.ckpt), "rep": meta["rep"], "dim": int(z.shape[1]),
                                         "floor": bool(meta.get("floor", False)), "audio": str(audio)})
        print(f"{audio}: {len(s)} s, human activity in {(s >= 0.5).mean():.1%} of seconds -> {out}")
        if start is not None:
            start = start + dt.timedelta(seconds=len(s))   # consecutive files continue the timeline


def cmd_pretrain(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = site_data(args.streams)
    short = [len(f) for f, _, _ in data if len(f) <= args.crop]
    if short:
        raise SystemExit(f"every stream must be longer than --crop ({args.crop} slots); got {short}")
    enc, log = ht.pretrain(data, args, dev, rng)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"enc": enc.state_dict(), "cin": data[0][0].shape[1],
                "meta": {"streams": [str(s) for s in args.streams], "ssl_iters": args.ssl_iters, "crop": args.crop,
                         "batch": args.batch, "seed": args.seed, "loss": log}}, args.out)
    print(f"pretrained on {len(data)} stream(s), {sum(len(f) for f, _, _ in data)} slots; "
          f"loss {log[0]:.3f} -> {log[-1]:.3f}; saved {args.out}")


def cmd_fit(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if len(args.labels) != len(args.streams):
        raise SystemExit("one --labels file per --streams folder")
    enc, _ = load_stage2(args.stage2, dev)
    data = site_data(args.streams, args.labels)
    xs = np.concatenate([np.concatenate([representations(enc, f, dev), t], 1)[y >= 0][::args.stride] for f, t, y in data])
    ys = np.concatenate([y[y >= 0][::args.stride] for _, _, y in data])
    if len(np.unique(ys)) < 2:
        raise SystemExit(f"labels contain one class only ({ys[:1]}); need occupied and vacant slots")
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    probe.fit(xs, ys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"probe": probe, "stage2": str(args.stage2), "n_labelled": int(len(ys))}, args.out)
    print(f"probe fitted on {len(ys)} labelled slots ({ys.mean():.1%} occupied) from {len(data)} stream(s); saved {args.out}")


def cmd_predict(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = Path(args.input)
    if src.suffix == ".npz":
        z, s, start = load_streams([src])
    else:
        if args.ckpt is None:
            raise SystemExit("--ckpt is needed to run stage 1 on an audio file")
        rep, head, _ = load_stage1(args.ckpt, dev)
        z, _, s = stage1_stream(od.load(src), rep, head, dev)
        start = parse_time(args.start)
    feats, tod, times = slot_features(z, s, start)
    df = pd.DataFrame({"slot": np.arange(len(feats)),
                       "slot_start": [t.isoformat() for t in times] if times is not None else [k * BLOCK for k in range(len(feats))],
                       "human_activity_mean": slot_stat(s, np.nanmean), "human_activity_max": slot_stat(s, np.nanmax)})
    if args.stage2 and args.probe:
        if start is None:
            warnings.warn("no --start given: the time of day fed to the probe is midnight for every slot")
        enc, _ = load_stage2(args.stage2, dev)
        probe = joblib.load(args.probe)["probe"]
        r = representations(enc, feats, dev)
        df["occupancy_probability"] = probe.predict_proba(np.concatenate([r, tod], 1))[:, 1]
    else:
        print("no --stage2/--probe given: writing stage-1 activity per slot only (stage 2 not trained yet)")
    df.to_csv(args.out, index=False, float_format="%.4f")
    print(f"{len(df)} slots -> {args.out}")


def cmd_evaluate(args):
    df = pd.read_csv(args.pred)
    if "occupancy_probability" not in df:
        raise SystemExit("predictions have no occupancy_probability column (run predict with --stage2 and --probe)")
    times = [parse_time(t) for t in df["slot_start"]]
    y = slot_labels(times, read_intervals(args.labels))
    m = y >= 0
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        raise SystemExit("need labelled slots of both classes to evaluate")
    res = ht.scores(y[m], df["occupancy_probability"].values[m])
    res["n_labelled"] = int(m.sum())
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("stage1", help="audio -> per-second embeddings and activity scores")
    s1.add_argument("audio", nargs="+")
    s1.add_argument("--ckpt", type=Path, required=True, help="model.pt from src/train_office.py")
    s1.add_argument("--start", default="", help="ISO start time of the first file, e.g. 2026-11-05T09:00:00")
    s1.add_argument("--out", type=Path, required=True, help="folder for the .npz streams")
    s1.set_defaults(fn=cmd_stage1)
    pt = sub.add_parser("pretrain", help="TS2Vec pretraining of the temporal encoder on unlabelled streams")
    pt.add_argument("--streams", type=Path, nargs="+", required=True, help="one folder of .npz streams per site")
    pt.add_argument("--out", type=Path, required=True)
    pt.add_argument("--ssl-iters", type=int, default=600)
    pt.add_argument("--crop", type=int, default=3000, help="window length in slots (3000 = 8.3 h)")
    pt.add_argument("--batch", type=int, default=8)
    pt.add_argument("--seed", type=int, default=0)
    pt.set_defaults(fn=cmd_pretrain)
    ft = sub.add_parser("fit", help="linear probe on the frozen temporal encoder")
    ft.add_argument("--streams", type=Path, nargs="+", required=True)
    ft.add_argument("--labels", type=Path, nargs="+", required=True, help="start,end,occupied intervals per site")
    ft.add_argument("--stage2", type=Path, required=True)
    ft.add_argument("--out", type=Path, required=True)
    ft.add_argument("--stride", type=int, default=6, help="keep every k-th labelled slot (6 = one per minute)")
    ft.set_defaults(fn=cmd_fit)
    pr = sub.add_parser("predict", help="audio or stream -> occupancy probability per slot")
    pr.add_argument("input", help="audio file or a stage-1 .npz stream")
    pr.add_argument("--ckpt", type=Path, help="stage-1 model.pt (for audio input)")
    pr.add_argument("--start", default="", help="ISO start time (audio input)")
    pr.add_argument("--stage2", type=Path)
    pr.add_argument("--probe", type=Path)
    pr.add_argument("--out", type=Path, required=True)
    pr.set_defaults(fn=cmd_predict)
    ev = sub.add_parser("evaluate", help="score predictions against occupancy intervals")
    ev.add_argument("--pred", type=Path, required=True)
    ev.add_argument("--labels", type=Path, required=True)
    ev.set_defaults(fn=cmd_evaluate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
