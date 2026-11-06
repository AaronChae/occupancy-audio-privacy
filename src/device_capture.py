#!/usr/bin/env python3
"""On-device capture: microphone -> stage 1 -> embeddings on disk. The audio is never written.

Reads 16 kHz mono audio in 10-s blocks, runs the encoder and the activity head of a
train_office.py checkpoint on each block, keeps the per-second embeddings z, class probabilities p
and human-activity score s, and discards the samples. Every --flush seconds (and at the end) the
buffered seconds are written as one .npz stream in the format of src/pipeline.py stage1, named by
its start time, into --out. Those streams are what the server-side stage 2 reads
(src/pipeline.py pretrain / fit / predict).

--source mic needs the sounddevice package and an input device; --source file replays a wav file
through the same code path, for testing without a microphone.

Examples:
    python src/device_capture.py --source mic --ckpt models/model.pt --out streams/room1
    python src/device_capture.py --source file --audio test.wav --start 2026-11-05T09:00:00 \
        --ckpt models/model.pt --out streams/test --flush 60
"""
import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl   # noqa: E402

BLOCK_SAMPLES = pl.BLOCK * pl.SR


def mic_blocks(device=None):
    import sounddevice as sd
    with sd.InputStream(samplerate=pl.SR, channels=1, dtype="float32", device=device, blocksize=BLOCK_SAMPLES) as st:
        while True:
            audio, overflowed = st.read(BLOCK_SAMPLES)
            if overflowed:
                print("warning: input overflow, some samples were dropped", file=sys.stderr)
            yield torch.from_numpy(audio[:, 0].copy())


def file_blocks(path):
    wav = pl.load_audio(path)
    for i in range(0, wav.shape[0], BLOCK_SAMPLES):
        seg = wav[i: i + BLOCK_SAMPLES]
        if seg.shape[0] >= pl.SR:
            yield seg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("mic", "file"), required=True)
    ap.add_argument("--audio", type=Path, help="wav file for --source file")
    ap.add_argument("--device", help="sounddevice input device name or index (mic)")
    ap.add_argument("--ckpt", type=Path, required=True, help="model.pt from src/train_office.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", default="", help="ISO start time for --source file (mic uses the clock)")
    ap.add_argument("--flush", type=int, default=3600, help="seconds of embeddings per written file")
    ap.add_argument("--max-seconds", type=int, default=0, help="stop after this many seconds (0 = run until stopped)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rep, head, meta = pl.load_stage1(args.ckpt, dev)
    info = {"ckpt": str(args.ckpt), "rep": meta["rep"], "dim": int(meta["dim"]), "floor": bool(meta.get("floor", False)),
            "source": args.source}
    blocks = mic_blocks(args.device) if args.source == "mic" else file_blocks(args.audio)
    start = dt.datetime.now().replace(microsecond=0) if args.source == "mic" else pl.parse_time(args.start)
    buf, buf_start, done = [], start, 0

    def flush():
        if not buf:
            return
        z = np.concatenate([b[0] for b in buf]); p = np.concatenate([b[1] for b in buf]); s = np.concatenate([b[2] for b in buf])
        name = buf_start.strftime("%Y%m%dT%H%M%S") if buf_start else f"seconds_{done - len(s):08d}"
        pl.save_stream(args.out / f"{name}.npz", z, p, s, buf_start, info)
        print(f"wrote {len(s)} s -> {args.out / (name + '.npz')} (activity in {(s >= 0.5).mean():.1%} of seconds)", flush=True)

    try:
        for seg in blocks:
            t0 = time.time()
            z, p, s = pl.stage1_stream(seg, rep, head, dev)
            buf.append((z, p, s)); done += len(s)
            if args.source == "mic" and time.time() - t0 > pl.BLOCK:
                print("warning: stage 1 is slower than real time on this device", file=sys.stderr)
            if sum(len(b[2]) for b in buf) >= args.flush:
                flush()
                buf, buf_start = [], (start + dt.timedelta(seconds=done) if start else None)
            if args.max_seconds and done >= args.max_seconds:
                break
    except KeyboardInterrupt:
        pass
    flush()


if __name__ == "__main__":
    main()
