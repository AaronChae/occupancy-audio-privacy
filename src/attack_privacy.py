#!/usr/bin/env python3
"""Privacy attacks on audio representations, with LibriSpeech speakers never seen by the encoder.

The attacker knows the representation and has labelled audio of the target speakers (worst case).
    speaker  closed-set identification of the 40 test-clean speakers (utterances split 50/50)
    sex      speaker sex, trained on the dev-clean speakers and tested on the test-clean speakers
    content  a decoder trained on dev-clean maps the representation to an 80-band log-mel
             spectrogram; test-clean reconstructions are inverted with Griffin-Lim and transcribed
             with Whisper (word error rate), and compared with the clean audio (STOI)
Speech is clean and close to the microphone, the most favorable case for the attacker.
References in every run: Whisper on the clean audio, and on Griffin-Lim inversion of the true mel.

Representations: --rep learned --ckpt <model.pt> | posteriors --ckpt <model.pt> | logmel_stats | envelope |
logmel_frames

Example:
    python3 src/attack_privacy.py --rep learned --ckpt experiments/office_v0/learned_adv1/model.pt \
        --out experiments/privacy_v0/learned_adv1
"""
import argparse
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
import office_data as od   # noqa: E402
import office_models as om  # noqa: E402

MEL = dict(sample_rate=od.SR, n_fft=1024, win_length=640, hop_length=160, n_mels=80)


class Posteriors(nn.Module):
    """Only the per-second class probabilities of a trained model leave the device (16 values)."""

    def __init__(self, enc, head):
        super().__init__()
        self.enc, self.head = enc, head

    def forward(self, wav):
        return torch.sigmoid(self.head(self.enc(wav)))


def build_rep(args):
    if args.rep == "posteriors":
        ck = torch.load(args.ckpt, map_location="cpu")
        enc = om.SecondEncoder(ck["args"]["dim"], floor=ck["args"].get("floor", False))
        head = om.EventHead(ck["args"]["dim"], len(od.CLASSES))
        enc.load_state_dict(ck["rep"]); head.load_state_dict(ck["head"])
        return Posteriors(enc, head)
    if args.rep == "learned":
        ck = torch.load(args.ckpt, map_location="cpu")
        rep = om.SecondEncoder(ck["args"]["dim"], floor=ck["args"].get("floor", False))
        rep.load_state_dict(ck["rep"])
        return rep
    if args.rep == "logmel_stats":
        return om.LogMelStats(args.floor)
    return {"envelope": om.Envelope, "logmel_frames": om.LogMelFrames}[args.rep]()


@torch.no_grad()
def featurize(rep, waves, dev):
    rep = rep.to(dev).eval()
    return [rep(w.to(dev)[None])[0].float().cpu() for w in waves]


def pad(xs):
    lens = torch.tensor([len(x) for x in xs])
    out = torch.zeros(len(xs), int(lens.max()), xs[0].shape[1])
    for i, x in enumerate(xs):
        out[i, : len(x)] = x
    return out, lens


class SeqNet(nn.Module):
    """BiGRU over a representation sequence; inputs above 25 Hz are first downsampled by strided convolutions."""

    def __init__(self, d, n_out, rate, h=128):
        super().__init__()
        layers, c, self.k = [], d, 0
        while rate > 25:
            layers += [nn.Conv1d(c, h, 5, stride=2, padding=2), nn.GELU()]
            c, rate, self.k = h, rate / 2, self.k + 1
        self.down = nn.Sequential(*layers)
        self.inp = nn.Linear(c, h)
        self.gru = nn.GRU(h, h, 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(4 * h, n_out)

    def forward(self, x, lens):
        if self.k:
            x = self.down(x.transpose(1, 2)).transpose(1, 2)
            for _ in range(self.k):
                lens = (lens + 1) // 2
        packed = nn.utils.rnn.pack_padded_sequence(self.inp(x), lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = nn.utils.rnn.pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=x.shape[1])
        m = (torch.arange(h.shape[1], device=h.device)[None] < lens.to(h.device)[:, None]).unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1)
        return self.out(torch.cat([mean, h.masked_fill(~m, -1e4).amax(1)], 1))


def standardizer(xs):
    cat = torch.cat(xs)
    mu, sd = cat.mean(0), cat.std(0).clamp(min=1e-5)
    return lambda x: (x - mu) / sd


def classify(xtr, ytr, xte, yte, n_out, rate, dev, epochs, seed=0):
    torch.manual_seed(seed)
    norm = standardizer(xtr)
    xtr, xte = [norm(x) for x in xtr], [norm(x) for x in xte]
    model = SeqNet(xtr[0].shape[1], n_out, rate).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    ytr, rng = torch.tensor(ytr), np.random.default_rng(seed)
    for ep in range(epochs):
        for b in np.array_split(rng.permutation(len(xtr)), max(1, len(xtr) // 32)):
            x, lens = pad([xtr[i] for i in b])
            loss = F.cross_entropy(model(x.to(dev), lens), ytr[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    pred = []
    with torch.no_grad():
        for b in np.array_split(np.arange(len(xte)), max(1, len(xte) // 64)):
            x, lens = pad([xte[i] for i in b])
            pred.append(model(x.to(dev), lens).argmax(1).cpu())
    return float((torch.cat(pred) == torch.tensor(yte)).float().mean())


class Decoder(nn.Module):
    """Representation sequence -> 80-band log-mel frames at 100 Hz."""

    def __init__(self, d, rate, h=256):
        super().__init__()
        self.up = int(round(100 / rate))
        self.inp = nn.Linear(d, h)
        self.conv = nn.Conv1d(h, h, 5, padding=2)
        self.gru = nn.GRU(h, h, 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * h, 80)

    def forward(self, x):
        x = self.inp(x.repeat_interleave(self.up, 1)) if self.up > 1 else self.inp(x)
        x = F.gelu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        return self.out(self.gru(x)[0])


def normalize_text(t):
    return re.sub(r"[^A-Z' ]", " ", t.upper()).split()


def wer(refs, hyps):
    errs = total = 0
    for r, h in zip(refs, hyps):
        r, h = normalize_text(r), normalize_text(h)
        d = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, d[0] = d[0], i
            for j in range(1, len(h) + 1):
                cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
                prev, d[j] = d[j], cur
        errs, total = errs + d[len(h)], total + len(r)
    return errs / max(total, 1)


def lcs_recall(refs, hyps):
    """Share of reference words recovered in order (longest common subsequence); ignores insertions."""
    hit = total = 0
    for r, h in zip(refs, hyps):
        r, h = normalize_text(r), normalize_text(h)
        d = [0] * (len(h) + 1)
        for i in range(1, len(r) + 1):
            prev = 0
            for j in range(1, len(h) + 1):
                cur = prev + 1 if r[i - 1] == h[j - 1] else max(d[j], d[j - 1])
                prev, d[j] = d[j], cur
        hit, total = hit + d[len(h)], total + len(r)
    return hit / max(total, 1)


def content_attack(rep, dev_w, te_w, te_txt, rate, args, dev, asr):
    from pystoi import stoi
    mel = torchaudio.transforms.MelSpectrogram(**MEL).to(dev)
    logmel = lambda w: torch.log(mel(w.to(dev)) + 1e-5).T[: w.numel() // 160].cpu()
    xtr, ytr = featurize(rep, dev_w, dev), [logmel(w) for w in dev_w]
    norm = standardizer(xtr)
    xtr = [norm(x) for x in xtr]
    up = int(round(100 / rate))
    dec = Decoder(xtr[0].shape[1], rate).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-2)
    rng, crop = np.random.default_rng(0), 8 * rate if rate < 100 else 800
    for ep in range(args.dec_epochs):
        for b in np.array_split(rng.permutation(len(xtr)), len(xtr) // 16):
            xs, ys = [], []
            for i in b:
                n = min(len(xtr[i]), crop)
                s = int(rng.integers(0, len(xtr[i]) - n + 1))
                xs.append(xtr[i][s: s + n]); ys.append(ytr[i][s * up: (s + n) * up])
            x, _ = pad(xs); y, ylen = pad(ys)
            pred = dec(x.to(dev))[:, : y.shape[1]]
            m = (torch.arange(y.shape[1])[None] < ylen[:, None]).to(dev).unsqueeze(-1)
            loss = ((pred - y.to(dev)).abs() * m).sum() / (m.sum() * 80)
            opt.zero_grad(); loss.backward(); opt.step()
    dec.eval()
    inv = torchaudio.transforms.InverseMelScale(n_stft=513, n_mels=80, sample_rate=od.SR).to(dev)
    gl = torchaudio.transforms.GriffinLim(n_fft=1024, win_length=640, hop_length=160, n_iter=60).to(dev)
    to_wave = lambda lm: gl(inv(torch.exp(lm.to(dev)).T.clamp(min=0))).cpu()
    recon, true_gl = [], []
    with torch.no_grad():
        for x, w in zip(featurize(rep, te_w, dev), te_w):
            recon.append(to_wave(dec(norm(x)[None].to(dev))[0][: w.numel() // 160]))
            true_gl.append(to_wave(logmel(w)))
    hyp = lambda ws: [o["text"] for o in asr([w.numpy() for w in ws], batch_size=16)]
    st = [stoi(w.numpy()[: len(r)], r.numpy()[: len(w)], od.SR) for w, r in zip(te_w, recon)]
    res = {"stoi_recon": float(np.mean(st))}
    for name, ws in (("recon", recon), ("clean", te_w), ("true_mel_griffinlim", true_gl)):
        h = hyp(ws)
        res[f"wer_{name}"], res[f"word_recall_{name}"] = wer(te_txt, h), lcs_recall(te_txt, h)
    return res


def gpu_slot(n=2, lock_dir=Path("/tmp")):
    """Wait for one of n slots so that at most n runs do the memory-heavy content attack at once."""
    while True:
        for i in range(n):
            f = open(lock_dir / f"occupancy_attack_slot{i}.lock", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f   # the lock is held while the file stays open
            except BlockingIOError:
                f.close()
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep", required=True, choices=("learned", "posteriors", "logmel_stats", "envelope", "logmel_frames"))
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--floor", action="store_true", help="noise-floor normalization for logmel_stats")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dec-epochs", type=int, default=15)
    ap.add_argument("--n-content", type=int, default=200, help="test-clean utterances for the content attack")
    ap.add_argument("--asr", default="openai/whisper-small.en")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev, t0 = "cuda", time.time()
    rate = 100 if args.rep == "logmel_frames" else 1
    rep = build_rep(args)
    sex = od.speaker_sex(args.raw)
    data = {}
    for subset in ("dev-clean", "test-clean"):
        idx = od.librispeech_index(args.raw, subset)
        waves = [w[: w.numel() // od.SR * od.SR] for w in od.load_many([p for _, p, _ in idx])]
        keep = [i for i, w in enumerate(waves) if w.numel() >= od.SR]
        data[subset] = ([idx[i] for i in keep], [waves[i] for i in keep])
    print(f"loaded {sum(len(v[0]) for v in data.values())} utterances, {time.time() - t0:.0f} s", flush=True)
    res = {"rep": args.rep, "ckpt": str(args.ckpt) if args.ckpt else None}

    te_idx, te_w = data["test-clean"]
    te_x = featurize(rep, te_w, dev)
    spk = sorted({s for s, _, _ in te_idx})
    order = {}
    tr, te = [], []
    for i, (s, _, _) in enumerate(te_idx):
        order[s] = order.get(s, -1) + 1
        (tr if order[s] % 2 == 0 else te).append(i)
    y = [spk.index(s) for s, _, _ in te_idx]
    res["speaker_acc"] = classify([te_x[i] for i in tr], [y[i] for i in tr], [te_x[i] for i in te], [y[i] for i in te],
                                  len(spk), rate, dev, args.epochs)
    res["speaker_chance"] = 1 / len(spk)
    print("speaker", res["speaker_acc"], f"{time.time() - t0:.0f} s", flush=True)

    dv_idx, dv_w = data["dev-clean"]
    dv_x = featurize(rep, dv_w, dev)
    ys = lambda idx: [int(sex[s] == "F") for s, _, _ in idx]
    res["sex_acc"] = classify(dv_x, ys(dv_idx), te_x, ys(te_idx), 2, rate, dev, max(10, args.epochs // 2))
    res["sex_majority"] = max(np.mean(ys(te_idx)), 1 - np.mean(ys(te_idx)))
    print("sex", res["sex_acc"], f"{time.time() - t0:.0f} s", flush=True)

    slot = gpu_slot()  # noqa: F841
    torch.cuda.empty_cache()
    os.environ.setdefault("HF_HOME", "/work/nvme/beco/mchae1/hf_cache")
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=args.asr, device=0)
    short = [i for i, w in enumerate(te_w) if w.numel() <= 30 * od.SR]   # Whisper's 30-s window
    sel = np.random.default_rng(0).choice(short, args.n_content, replace=False)
    res.update(content_attack(rep, dv_w, [te_w[i] for i in sel], [te_idx[i][2] for i in sel], rate, args, dev, asr))
    res["seconds"] = time.time() - t0
    json.dump(res, open(args.out / "privacy.json", "w"), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
