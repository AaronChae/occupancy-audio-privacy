"""Re-score the trained office models with threshold-free metrics and on non-speech seconds.

For every configuration of office_ci.py the per-second human-activity score is computed on the
same test sets (DCASE 2013 OL, DCASE 2016 test, 400 new scenes per unseen room) and reported

  all        every second; positives are seconds with any human sound (as in the paper)
  nonspeech  seconds whose labels contain no speech; positives are seconds with a non-speech
             human sound (keyboard, page turn, cough, ...), negatives are seconds with no human
             sound. This is the still-occupant case that motivates the paper.

Metrics: AUC, average precision (AP) and equal error rate (EER). For the two real sets, 95%
bootstrap intervals over whole recordings (2,000 draws) are added for the non-speech AUC, with
paired differences for the pairs of office_ci.py.

Output: experiments/summary/office_rescore.csv and office_rescore_ci.json.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
import office_data as od  # noqa: E402
import office_models as om  # noqa: E402
from office_ci import RUNS, boot  # noqa: E402
from train_office import TEST_ROOMS, make_rep, speech_arrays  # noqa: E402

NONSPEECH_HUMAN = od.HUMAN.clone()
NONSPEECH_HUMAN[od.SPEECH] = False


@torch.no_grad()
def probs(rep, head, sets, dev, chunk=10):
    """[(labels (T, 16), probabilities (T, 16))] per recording."""
    rep.eval(); head.eval()
    out = []
    for _, x, y in sets:
        T, outs = y.shape[0], []
        for s in range(0, T, chunk):
            seg = x[s * od.SR: min(T, s + chunk) * od.SR].to(dev)[None]
            outs.append(head(rep(seg))[0].float().cpu())
        p = torch.sigmoid(torch.cat(outs)[:T])
        out.append((y[: p.shape[0]].numpy(), p.numpy()))
    return out


def subset(rec, which):
    """(y, score) of one recording for the subset 'all' or 'nonspeech'."""
    y, p = rec
    score = p[:, od.HUMAN.numpy()].max(1)
    if which == "all":
        return y[:, od.HUMAN.numpy()].max(1), score
    keep = y[:, od.SPEECH] == 0
    return y[keep][:, NONSPEECH_HUMAN.numpy()].max(1), score[keep]


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def metrics(y, s):
    if not 0 < y.sum() < len(y):
        return {"n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()), "auc": np.nan, "ap": np.nan, "eer": np.nan}
    return {"n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()), "auc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s)), "eer": eer(y, s)}


def main():
    raw, dev, exp = Path("data/raw"), "cuda", Path("experiments")
    sets = {"dcase2013_ol": od.dcase2013_ol(raw), "dcase2016_test": od.dcase2016_test(raw)}
    te_idx = od.librispeech_index(raw, "test-clean", 20, 1)
    ev, sp = od.event_bank(raw, "test"), speech_arrays(te_idx)
    synth = []
    for room in TEST_ROOMS:
        s = od.SceneSynth(ev, sp, [0] * len(te_idx), od.room_bank(raw, [room], seed=1), dev, p_speech=0.4)
        rng = np.random.default_rng(4321)
        for i in range(10):
            w, yy, _ = s.batch(40, rng)
            synth += [(room + str(i) + "_" + str(j), w[j].cpu(), yy[j].cpu()) for j in range(40)]
        del s
        torch.cuda.empty_cache()
    sets["synth_unseen_rooms"] = synth
    print("sets", {k: len(v) for k, v in sets.items()}, flush=True)

    rows, recs = [], {k: {"all": {}, "nonspeech": {}} for k in sets}
    for run in RUNS:
        ck = torch.load(exp / run / "model.pt", map_location="cpu", weights_only=False)
        a = ck["args"]
        rep, d = make_rep(a["rep"], a.get("dim", 32), a.get("floor", False))
        head = om.EventHead(d, len(od.CLASSES))
        rep.load_state_dict(ck["rep"]); head.load_state_dict(ck["head"])
        rep, head = rep.to(dev), head.to(dev)
        for k, s in sets.items():
            pr = probs(rep, head, s, dev)
            for which in ("all", "nonspeech"):
                per = [subset(r, which) for r in pr]
                recs[k][which][run] = per
                y = np.concatenate([a for a, _ in per]); sc = np.concatenate([b for _, b in per])
                m = metrics(y, sc)
                rows.append({"test_set": k, "run": run, "subset": which, **m})
                print("%-16s %-34s %-9s pos %6d neg %6d auc %.3f ap %.3f eer %.3f"
                      % (k, run, which, m["n_pos"], m["n_neg"], m["auc"], m["ap"], m["eer"]), flush=True)
        del rep, head
        torch.cuda.empty_cache()

    exp.joinpath("summary").mkdir(exist_ok=True)
    with open(exp / "summary" / "office_rescore.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    ci = {}
    for k in ("dcase2013_ol", "dcase2016_test"):
        res, diffs, n = boot(recs[k]["nonspeech"], 2000)
        ci[k] = {"units": n, "auc": res, "diff": diffs}
        print(k, "non-speech AUC, 95% intervals over", n, "recordings", flush=True)
        for r, v in res.items():
            print("   %-34s %.3f [%.3f, %.3f]" % (r, v["auc"], v["lo"], v["hi"]), flush=True)
        for p, v in diffs.items():
            print("   %-70s %+.3f [%+.3f, %+.3f] p<=0 %.3f" % (p, v["diff"], v["lo"], v["hi"], v["p_le_0"]), flush=True)
    json.dump(ci, open(exp / "summary" / "office_rescore_ci.json", "w"), indent=1)
    print("rescore done", flush=True)


if __name__ == "__main__":
    main()
