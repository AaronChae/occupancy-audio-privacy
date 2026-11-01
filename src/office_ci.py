"""Bootstrap confidence intervals and per-room results for the office experiments.

Whole recordings (DCASE 2013 OL, DCASE 2016 test) or whole scenes (new synthetic scenes of each
unseen room) are resampled with replacement. The script reports 95% percentile intervals of the
human-activity AUC and of paired AUC differences between configurations."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import office_data as od  # noqa: E402
import office_models as om  # noqa: E402
from train_office import TEST_ROOMS, make_rep, speech_arrays  # noqa: E402

RUNS = ["office_v0/envelope", "office_v0/logmel_stats", "office_v1/logmel_stats_floor",
        "office_v0/learned_adv0", "office_v0/learned_adv1", "office_v1/learned_floor_adv0",
        "office_v1/learned_floor_adv1", "office_v1/learned_floor_adv3",
        "office_v2/learned_floor_d8_adv0", "office_v2/learned_floor_d8_adv3"]
PAIRS = [("office_v1/learned_floor_adv0", "office_v1/logmel_stats_floor"),
         ("office_v0/learned_adv0", "office_v0/logmel_stats"),
         ("office_v1/learned_floor_adv0", "office_v0/learned_adv0"),
         ("office_v1/learned_floor_adv1", "office_v1/learned_floor_adv0"),
         ("office_v1/learned_floor_adv3", "office_v1/learned_floor_adv0"),
         ("office_v2/learned_floor_d8_adv0", "office_v1/learned_floor_adv0"),
         ("office_v2/learned_floor_d8_adv3", "office_v2/learned_floor_d8_adv0")]


@torch.no_grad()
def scores(rep, head, sets, dev, chunk=10):
    rep.eval(); head.eval()
    out = []
    for _, x, y in sets:
        T, outs = y.shape[0], []
        for s in range(0, T, chunk):
            seg = x[s * od.SR: min(T, s + chunk) * od.SR].to(dev)[None]
            outs.append(head(rep(seg))[0].float().cpu())
        p = torch.sigmoid(torch.cat(outs)[:T])
        yy = y[: p.shape[0]]
        out.append((yy[:, od.HUMAN].amax(1).numpy(), p[:, od.HUMAN].amax(1).numpy()))
    return out


def pooled_auc(recs, idx):
    y = np.concatenate([recs[i][0] for i in idx])
    p = np.concatenate([recs[i][1] for i in idx])
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def boot(recs_by_run, n_boot, seed=0):
    n = len(recs_by_run[RUNS[0]])
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, n, n) for _ in range(n_boot)]
    full = {r: pooled_auc(recs_by_run[r], range(n)) for r in RUNS}
    res, diffs = {}, {}
    for r in RUNS:
        v = np.array([pooled_auc(recs_by_run[r], d) for d in draws])
        res[r] = {"auc": float(full[r]), "lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5))}
    for a, b in PAIRS:
        v = np.array([pooled_auc(recs_by_run[a], d) - pooled_auc(recs_by_run[b], d) for d in draws])
        diffs[a + " - " + b] = {"diff": float(full[a] - full[b]), "lo": float(np.nanpercentile(v, 2.5)),
                                "hi": float(np.nanpercentile(v, 97.5)), "p_le_0": float(np.mean(v <= 0))}
    return res, diffs, n


def main():
    raw, dev, exp = Path("data/raw"), "cuda", Path("experiments")
    sets = {"dcase2013_ol": od.dcase2013_ol(raw), "dcase2016_test": od.dcase2016_test(raw)}
    te_idx = od.librispeech_index(raw, "test-clean", 20, 1)
    ev, sp = od.event_bank(raw, "test"), speech_arrays(te_idx)
    for room in TEST_ROOMS:
        synth = od.SceneSynth(ev, sp, [0] * len(te_idx), od.room_bank(raw, [room], seed=1), dev, p_speech=0.4)
        rng, scenes = np.random.default_rng(4321), []
        for i in range(10):
            w, yy, _ = synth.batch(40, rng)
            scenes += [(room + str(i) + "_" + str(j), w[j].cpu(), yy[j].cpu()) for j in range(40)]
        sets["synth_" + room] = scenes
        del synth
        torch.cuda.empty_cache()
    print("sets", {k: len(v) for k, v in sets.items()}, flush=True)
    recs = {k: {} for k in sets}
    for run in RUNS:
        ck = torch.load(exp / run / "model.pt", map_location="cpu", weights_only=False)
        a = ck["args"]
        rep, d = make_rep(a["rep"], a.get("dim", 32), a.get("floor", False))
        head = om.EventHead(d, len(od.CLASSES))
        rep.load_state_dict(ck["rep"]); head.load_state_dict(ck["head"])
        rep, head = rep.to(dev), head.to(dev)
        for k, s in sets.items():
            recs[k][run] = scores(rep, head, s, dev)
        ref = json.load(open(exp / run / "results.json"))["utility"]
        print("scored", run, "ol check", round(pooled_auc(recs["dcase2013_ol"][run], range(len(sets["dcase2013_ol"]))), 4),
              round(ref["dcase2013_ol"]["human_auc"], 4), flush=True)
        del rep, head
        torch.cuda.empty_cache()
    out = {}
    for k in sets:
        res, diffs, n = boot(recs[k], 1000 if k.startswith("synth") else 2000)
        out[k] = {"units": n, "auc": res, "diff": diffs}
        print(k, n, flush=True)
        for r, v in res.items():
            print("   %-34s %.3f [%.3f, %.3f]" % (r, v["auc"], v["lo"], v["hi"]), flush=True)
        for p, v in diffs.items():
            print("   %-70s %+.3f [%+.3f, %+.3f] p<=0 %.3f" % (p, v["diff"], v["lo"], v["hi"], v["p_le_0"]), flush=True)
    json.dump(out, open(exp / "summary" / "office_ci.json", "w"), indent=1)
    print("ci done", flush=True)


if __name__ == "__main__":
    main()
