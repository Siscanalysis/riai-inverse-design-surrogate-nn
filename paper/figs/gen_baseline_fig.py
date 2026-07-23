#!/usr/bin/env python3
"""Baseline comparison figures from rerun/out/baselines_*.csv (4 methods, same repair)."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "..", "rerun", "out"))

def load(case):
    d = {}
    with open(os.path.join(OUT, f"baselines_{case}.csv"), newline="") as f:
        for row in csv.DictReader(f):
            d[row["method"]] = {k: float(v) for k, v in row.items() if k != "method"}
    return d

cnc, hea = load("cnc"), load("hea")
methods = ["reject_norepair", "reject_repair", "ours_iterative", "nsga2_repair"]
labels  = ["rechazo\n(sin rep.)", "rechazo\n+rep.", "nuestro\n(iter.)", "NSGA-II\n+rep."]
colors  = ["#bbbbbb", "#4c72b0", "#dd8452", "#55a868"]
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# ---- Fig 1: feasibility rate (repair is the enabler) ----
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
for ax, (name, d) in zip(axes, [("CNC (Ra)", cnc), ("HEA (simplex)", hea)]):
    rates = [d[m]["feas_rate"]*100 for m in methods]
    bars = ax.bar(range(4), rates, color=colors)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("factibilidad [%]"); ax.set_title(name)
    for b, r in zip(bars, rates):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{r:.1f}", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Factibilidad con la MISMA reparacion (sin reparar, el simplex es inviable: 0%)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_baselines_feasibility.pdf")); plt.close(fig)

# ---- Fig 2: coverage (fill%) vs best robustness (minS) trade-off, per method ----
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
# left: coverage
ax = axes[0]
for case, mk in [(cnc, "o"), (hea, "s")]:
    pass
xs = range(4)
ax.bar([x-0.2 for x in xs], [cnc[m]["mean_fill_pct"] for m in methods], width=0.4, label="CNC", color="#4c72b0")
ax.bar([x+0.2 for x in xs], [hea[m]["mean_fill_pct"] for m in methods], width=0.4, label="HEA", color="#dd8452")
ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("cobertura (fill %)"); ax.set_title("Cobertura de la variedad"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
# right: best robustness minS (lower = more robust); normalize per case to reject_repair
ax = axes[1]
def norm_minS(d):
    ref = d["reject_repair"]["min_S"]
    return [d[m]["min_S"]/ref if ref>0 else 0 for m in methods]
ax.bar([x-0.2 for x in xs], norm_minS(cnc), width=0.4, label="CNC", color="#4c72b0")
ax.bar([x+0.2 for x in xs], norm_minS(hea), width=0.4, label="HEA", color="#dd8452")
ax.axhline(1.0, color="k", lw=0.7, ls="--")
ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("min S (norm. a rechazo+rep.; menor=mas robusto)")
ax.set_title("Mejor robustez alcanzada"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Compromiso cobertura <-> robustez (misma reparacion, mismo presupuesto)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_baselines_tradeoff.pdf")); plt.close(fig)

print("wrote fig_baselines_feasibility.pdf, fig_baselines_tradeoff.pdf")
