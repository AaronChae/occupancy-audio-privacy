#!/usr/bin/env python3
"""Content attack on a stage-1 stream: what can be rebuilt from the transmitted embeddings of one recording?

Trains the speech-reconstruction decoder of src/attack_privacy.py for a stage-1 checkpoint (embedding
sequence -> 80-band log-mel at 100 Hz, 15 epochs on LibriSpeech dev-clean, the same recipe as in the
paper) and applies it to the per-second embeddings of a recording (the .npz written by
src/pipeline.py stage1 or src/device_capture.py). The predicted log-mel is inverted with Griffin-Lim
and transcribed with Whisper, so a person can read and listen to what an attacker who holds only
the embeddings would get. The original audio is not needed; if it is given, it is transcribed as a
reference and STOI is reported.

Outputs in --out: decoder.pt (reusable), recon_<stream>.wav (first --seconds of the reconstruction),
transcript.json (Whisper text per 30-s window, reconstruction and, if given, original).

Example (on a GPU machine with data/raw/librispeech):
    python src/attack_recording.py --ckpt models/office_v1/learned_floor_adv3/model.pt \
        --stream streams/room1/lecture.npz --seconds 300 --out experiments/attack_recording
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attack_privacy as ap_   # noqa: E402
import office_data as od   # noqa: E402

WIN = 30 * od.SR   # Whisper's window


def train_decoder(rep, raw, dev, epochs, seed=0):
    """The decoder recipe of attack_privacy.content_attack, returning the decoder and its input standardizer."""
    idx = od.librispeech_index(raw, "dev-clean")
    waves = [w[: w.numel() // od.SR * od.SR] for w in od.load_many([p for _, p, _ in idx])]
    waves = [w for w in waves if w.numel() >= od.SR]
    mel = torchaudio.transforms.MelSpectrogram(**ap_.MEL).to(dev)
    logmel = lambda w: torch.log(mel(w.to(dev)) + 1e-5).T[: w.numel() // 160].cpu()
    xtr, ytr = ap_.featurize(rep, waves, dev), [logmel(w) for w in waves]
    mu, sd = torch.cat(xtr).mean(0), torch.cat(xtr).std(0).clamp(min=1e-5)
    xtr = [(x - mu) / sd for x in xtr]
    dec = ap_.Decoder(xtr[0].shape[1], 1).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-2)
    rng, crop, up = np.random.default_rng(seed), 8, 100
    for ep in range(epochs):
        tot = 0.0
        for b in np.array_split(rng.permutation(len(xtr)), len(xtr) // 16):
            xs, ys = [], []
            for i in b:
                n = min(len(xtr[i]), crop)
                s = int(rng.integers(0, len(xtr[i]) - n + 1))
                xs.append(xtr[i][s: s + n]); ys.append(ytr[i][s * up: (s + n) * up])
            x, _ = ap_.pad(xs); y, ylen = ap_.pad(ys)
            pred = dec(x.to(dev))[:, : y.shape[1]]
            m = (torch.arange(y.shape[1])[None] < ylen[:, None]).to(dev).unsqueeze(-1)
            loss = ((pred - y.to(dev)).abs() * m).sum() / (m.sum() * 80)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"decoder epoch {ep + 1}/{epochs} loss {tot / (len(xtr) // 16):.3f}", flush=True)
    return dec.eval(), mu, sd


def to_wave(logmel, dev):
    inv = torchaudio.transforms.InverseMelScale(n_stft=513, n_mels=80, sample_rate=od.SR).to(dev)
    gl = torchaudio.transforms.GriffinLim(n_fft=1024, win_length=640, hop_length=160, n_iter=60).to(dev)
    with torch.no_grad():
        return gl(inv(torch.exp(logmel.to(dev)).T.clamp(min=0))).cpu()


def transcribe(asr, wav):
    pieces = [wav[i: i + WIN] for i in range(0, wav.numel(), WIN)]
    pieces = [p for p in pieces if p.numel() >= od.SR]
    return [o["text"].strip() for o in asr([p.numpy() for p in pieces], batch_size=8)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="stage-1 model.pt (learned representation)")
    ap.add_argument("--stream", type=Path, required=True, help="stage-1 .npz of the recording")
    ap.add_argument("--audio", type=Path, help="the original recording, for a reference transcript and STOI")
    ap.add_argument("--seconds", type=int, default=300, help="length of the excerpt to rebuild and transcribe")
    ap.add_argument("--offset", type=int, default=0, help="start of the excerpt in the stream, seconds")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--decoder", type=Path, help="reuse a decoder.pt from an earlier run instead of training")
    ap.add_argument("--dec-epochs", type=int, default=15)
    ap.add_argument("--asr", default="openai/whisper-small.en")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev, t0 = ("cuda" if torch.cuda.is_available() else "cpu"), time.time()
    rep = ap_.build_rep(argparse.Namespace(rep="learned", ckpt=args.ckpt, floor=False))
    if args.decoder:
        ck = torch.load(args.decoder, map_location=dev, weights_only=False)
        dec = ap_.Decoder(ck["d"], 1).to(dev); dec.load_state_dict(ck["dec"]); dec.eval()
        mu, sd = ck["mu"], ck["sd"]
    else:
        dec, mu, sd = train_decoder(rep, args.raw, dev, args.dec_epochs)
        torch.save({"dec": dec.state_dict(), "d": mu.numel(), "mu": mu, "sd": sd, "ckpt": str(args.ckpt)}, args.out / "decoder.pt")
    print(f"decoder ready, {time.time() - t0:.0f} s", flush=True)

    z = torch.from_numpy(np.load(args.stream)["z"].astype(np.float32))[args.offset: args.offset + args.seconds]
    with torch.no_grad():
        lm = dec(((z - mu) / sd)[None].to(dev))[0]
    recon = to_wave(lm, dev)
    name = args.stream.stem
    torchaudio.save(str(args.out / f"recon_{name}.wav"), recon[None], od.SR)
    res = {"stream": str(args.stream), "offset": args.offset, "seconds": int(z.shape[0]), "ckpt": str(args.ckpt)}
    if args.audio:
        from pystoi import stoi
        wav = od.load(args.audio)[args.offset * od.SR: (args.offset + z.shape[0]) * od.SR]
        n = min(wav.numel(), recon.numel())
        res["stoi"] = float(stoi(wav[:n].numpy(), recon[:n].numpy(), od.SR))
    os.environ.setdefault("HF_HOME", os.path.abspath("data/hf_cache"))
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=args.asr, device=0 if dev == "cuda" else -1)
    res["transcript_recon"] = transcribe(asr, recon)
    if args.audio:
        res["transcript_original"] = transcribe(asr, wav)
        res["word_recall"] = ap_.lcs_recall(res["transcript_original"], res["transcript_recon"])
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(args.out / "transcript.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("transcript")}))
    print("reconstruction, first windows:", " | ".join(res["transcript_recon"][:4]))


if __name__ == "__main__":
    main()
