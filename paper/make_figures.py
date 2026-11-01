"""Figures for the paper draft. Numbers are copied from the experiment outputs on Delta
(experiments/office_v*/results.json, experiments/privacy_v*/privacy.json,
experiments/hpd_ts2vec_v0/results_*.csv and the transition analysis)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, MID, LIGHT = "#1a1a1a", "#6b6b6b", "#b5b5b5"

# 1. loudness around label changes (z-score of 1-min max envelope, 10-min bins, mean over six homes)
arrive = [0.07,0.05,0.07,0.07,0.08,0.10,0.04,0.09,0.11,0.13,0.10,0.02,-0.01,0.07,0.00,0.04,-0.01,0.07,0.84,0.49,0.46,0.42,0.32,0.33,0.30,0.30,0.22,0.16,0.15,0.15,0.18,0.19,0.17,0.19,0.13,0.14]
leave = [0.11,0.09,0.14,0.13,0.18,0.16,0.14,0.19,0.23,0.31,0.34,0.35,0.38,0.42,0.44,0.45,0.54,0.79,0.08,-0.02,0.02,0.04,0.01,0.12,0.08,0.09,0.10,0.13,0.19,0.10,0.14,0.17,0.15,0.14,0.11,0.14]
t = (np.arange(36) * 10 - 175) / 60
fig, ax = plt.subplots(figsize=(3.4, 2.0))
ax.axvline(0, color=LIGHT, lw=0.8, ls="--")
ax.axhline(0, color=LIGHT, lw=0.6)
ax.plot(t, arrive, color=INK, lw=1.3, label="arrival")
ax.plot(t, leave, color=MID, lw=1.3, ls=(0, (4, 1.5)), label="departure")
ax.set_xlabel("time relative to label change (h)")
ax.set_ylabel("loudness (z-score)")
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figures/transitions.pdf"); plt.close(fig)

# 2. privacy-utility: speaker identification vs. human-activity AUC on real office recordings
pts = [  # label, speaker id %, AUC (DCASE 2013 OL), marker
    ("log-mel stats", 97.8, 0.682, "s"), ("log-mel stats + NF", 94.2, 0.760, "s"),
    ("HPD-style envelope", 22.4, 0.621, "^"),
    ("encoder", 76.8, 0.834, "o"), ("encoder + adv. 1", 62.3, 0.832, "o"),
    ("encoder + NF", 68.6, 0.869, "D"), ("encoder + NF + adv. 1", 60.3, 0.863, "D"),
    ("encoder + NF + adv. 3", 55.3, 0.874, "D"),
    ("encoder + NF, d = 8", 46.6, 0.859, "p"), ("encoder + NF + adv. 3, d = 8", 33.6, 0.740, "p")]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.axvline(2.5, color=LIGHT, lw=0.8, ls="--")
ax.text(3.5, 0.605, "chance", color=MID, fontsize=7.5)
place = {"log-mel stats": (95.8, 0.672, "right"), "log-mel stats + NF": (92.2, 0.750, "right"),
         "HPD-style envelope": (24.4, 0.627, "left"), "encoder": (78.5, 0.824, "left"),
         "encoder + adv. 1": (60.5, 0.820, "right"), "encoder + NF": (70.5, 0.871, "left"),
         "encoder + NF + adv. 1": (62.0, 0.853, "left"), "encoder + NF + adv. 3": (53.5, 0.888, "right"),
         "encoder + NF, d = 8": (44.6, 0.859, "right"), "encoder + NF + adv. 3, d = 8": (33.6, 0.722, "center")}
for name, x, y, m in pts:
    ax.scatter(x, y, marker=m, s=22, facecolor="white" if "NF" not in name else INK, edgecolor=INK, lw=0.8, zorder=3)
    tx, ty, ha = place[name]
    ax.text(tx, ty, name, fontsize=7, ha=ha, va="center", color=INK)
ax.set_xlim(0, 105); ax.set_ylim(0.6, 0.91)
ax.set_xlabel("speaker identification accuracy (%), lower is better")
ax.set_ylabel("activity AUC, real office")
fig.tight_layout(); fig.savefig("figures/tradeoff.pdf"); plt.close(fig)

# 3. label efficiency on HPDmobile (transductive pretraining, seed 0, mean over six held-out homes)
xs = ["1", "3", "7", "all"]
ba = {"SSL encoder + linear probe": [0.616, 0.610, 0.646, 0.674],
      "features + logistic regression": [0.565, 0.592, 0.619, 0.649],
      "features + gradient boosting": [0.590, 0.557, 0.581, 0.654],
      "same network, labels only": [0.515, 0.561, 0.588, 0.576]}
styles = [(INK, "-", "o"), (MID, "--", "s"), (MID, ":", "^"), (LIGHT, "-", "x")]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for (name, v), (c, ls, m) in zip(ba.items(), styles):
    ax.plot(range(4), v, color=c, ls=ls, marker=m, ms=3.5, lw=1.1, label=name)
ax.set_xticks(range(4), xs)
ax.set_xlabel("labelled days per training home")
ax.set_ylabel("balanced accuracy")
ax.set_ylim(0.5, 0.7)
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2)
fig.savefig("figures/label_efficiency.pdf", bbox_inches="tight"); plt.close(fig)
print("figures written")
