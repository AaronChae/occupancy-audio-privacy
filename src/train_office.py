#!/usr/bin/env python3
"""Train and evaluate representations for privacy-preserving office activity sensing.

Representations (--rep):
    learned       SecondEncoder (on-device), trained jointly with the event head
    logmel_stats  fixed per-second log-mel statistics; only the head is trained
    envelope      fixed HPDmobile-style envelope; only the head is trained
With --adv > 0 the learned encoder is also trained against a speaker classifier through a gradient
reversal layer (251 speakers of LibriSpeech train-clean-100).

Training scenes are synthesized on the fly from the train rooms, train event instances and
train-clean-100 speech. Evaluation is per second on
    synth_test     scenes from the test rooms (office L212, lecture room D105, meeting room C236),
                   test event instances and test-clean speech
    dcase2013_ol   real office recordings (DCASE 2013 OL test, 11 recordings)
    dcase2016_test DCASE 2016 task 2 public test mixtures (54 x 2 min)
Metrics: AUC and F1 (threshold 0.5) of "any human activity" (max over non-device classes), and the
mean of per-class AUCs.

Example:
    python3 src/train_office.py --rep learned --adv 1.0 --out experiments/office_v0/learned_adv1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import office_data as od   # noqa: E402
import office_models as om  # noqa: E402

TRAIN_ROOMS = ["Q301", "L207", "L227", "Room112", "ConferenceRoom2", "E112"]
TEST_ROOMS = ["L212", "D105", "C236"]


def make_rep(name, d, floor=False):
    if name == "learned":
        return om.SecondEncoder(d, floor=floor), d
    if name == "logmel_stats":
        return om.LogMelStats(floor), om.LogMelStats.dim
    if name == "envelope":
        return om.Envelope(), om.Envelope.dim
    raise ValueError(name)


def speech_arrays(idx):
    return [(w * 32767).clamp(-32768, 32767).short().numpy() for w in od.load_many([p for _, p, _ in idx])]


@torch.no_grad()
def evaluate(rep, head, sets, dev, chunk=10):
    rep.eval(); head.eval()
    P, Y = [], []
    for _, x, y in sets:
        T, outs = y.shape[0], []
        for s in range(0, T, chunk):
            seg = x[s * od.SR: min(T, s + chunk) * od.SR].to(dev)[None]
            outs.append(head(rep(seg))[0].float().cpu())
        P.append(torch.cat(outs)[:T]); Y.append(y[: P[-1].shape[0]])
    rep.train(); head.train()
    P, Y = torch.sigmoid(torch.cat(P)), torch.cat(Y)
    yh, ph = Y[:, od.HUMAN].amax(1).numpy(), P[:, od.HUMAN].amax(1).numpy()
    per = {c: float(roc_auc_score(Y[:, i], P[:, i])) for i, c in enumerate(od.CLASSES) if 0 < Y[:, i].sum() < len(Y)}
    return {"human_auc": float(roc_auc_score(yh, ph)), "human_f1": float(f1_score(yh, ph >= 0.5)),
            "macro_auc": float(np.mean(list(per.values()))), "seconds": int(len(Y)),
            "human_rate": float(yh.mean()), "per_class_auc": per}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep", required=True, choices=("learned", "logmel_stats", "envelope"))
    ap.add_argument("--adv", type=float, default=0.0, help="weight of the speaker adversary (learned only)")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--floor", action="store_true", help="subtract the per-band noise floor of each 10-s input")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--p-speech", type=float, default=0.4)
    ap.add_argument("--utt-per-speaker", type=int, default=24)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    idx = od.librispeech_index(args.raw, "train-clean-100", args.utt_per_speaker, args.seed)
    speakers = sorted({s for s, _, _ in idx})
    synth = od.SceneSynth(od.event_bank(args.raw, "train"), speech_arrays(idx),
                          [speakers.index(s) for s, _, _ in idx], od.room_bank(args.raw, TRAIN_ROOMS), dev,
                          p_speech=args.p_speech)
    print(f"data ready: {len(idx)} utterances, {len(speakers)} speakers, {time.time() - t0:.0f} s", flush=True)

    rep, d = make_rep(args.rep, args.dim, args.floor)
    rep, head = rep.to(dev), om.EventHead(d, len(od.CLASSES)).to(dev)
    adv = om.SpeakerAdversary(d, len(speakers)).to(dev) if args.adv > 0 else None
    params = [p for m in (rep, head, adv) if m is not None for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    log = []
    for step in range(args.steps):
        wav, y, spk = synth.batch(args.batch, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = rep(wav)
            loss_ev = F.binary_cross_entropy_with_logits(head(z).float(), y)
            loss, loss_adv, acc_adv = loss_ev, torch.zeros(()), float("nan")
            if adv is not None:
                act = y[:, :, od.SPEECH] > 0
                keep = (spk.to(dev) >= 0) & act.any(1)
                if keep.any():
                    lam = om.dann_lambda(step, args.steps, args.adv)
                    logits = adv(om.grad_reverse(z[keep], lam).float(), act[keep])
                    target = spk.to(dev)[keep]
                    loss_adv = F.cross_entropy(logits, target)
                    acc_adv = (logits.argmax(1) == target).float().mean().item()
                    loss = loss + loss_adv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == args.steps - 1:
            log.append({"step": step, "loss_event": loss_ev.item(), "loss_adv": float(loss_adv), "acc_adv": acc_adv})
            print(json.dumps(log[-1]), f"{time.time() - t0:.0f} s", flush=True)
    torch.save({"args": vars(args) | {"raw": str(args.raw), "out": str(args.out)}, "rep": rep.state_dict(),
                "head": head.state_dict(), "adv": adv.state_dict() if adv else None, "speakers": speakers},
               args.out / "model.pt")

    del synth
    torch.cuda.empty_cache()
    te_idx = od.librispeech_index(args.raw, "test-clean", 20, 1)
    synth_te = od.SceneSynth(od.event_bank(args.raw, "test"), speech_arrays(te_idx), [0] * len(te_idx),
                             od.room_bank(args.raw, TEST_ROOMS, seed=1), dev, p_speech=args.p_speech)
    trng, scenes = np.random.default_rng(1234), []
    for i in range(30):
        w, yy, _ = synth_te.batch(40, trng)
        scenes += [(f"synth{i}_{j}", w[j].cpu(), yy[j].cpu()) for j in range(40)]
    res = {"synth_test": evaluate(rep, head, scenes, dev),
           "dcase2013_ol": evaluate(rep, head, od.dcase2013_ol(args.raw), dev),
           "dcase2016_test": evaluate(rep, head, od.dcase2016_test(args.raw), dev)}
    json.dump({"utility": res, "train_log": log, "seconds": time.time() - t0}, open(args.out / "results.json", "w"), indent=1)
    for k, v in res.items():
        print(k, {m: round(v[m], 3) for m in ("human_auc", "human_f1", "macro_auc", "human_rate")}, flush=True)


if __name__ == "__main__":
    main()
