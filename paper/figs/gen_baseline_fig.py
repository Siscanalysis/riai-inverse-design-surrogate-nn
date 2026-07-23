#!/usr/bin/env python3
"""Baseline comparison figures from rerun/out/baselines_*.csv."""
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
methods = ["reject_norepair", "reject_repair", "ours_iterative"]
labels  = ["rechazo\n(sin reparar)", "rechazo\n+ reparacion", "nuestro\n(iterativo)"]
colors  = ["#bbbbbb", "#4c72b0", "#dd8452"]

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# ---- Fig 1: feasibility rate (repair is the enabler) ----
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
for ax, (name, d) in zip(axes, [("CNC (Ra)", cnc), ("HEA (simplex)", hea)]):
    rates = [d[m]["feas_rate"]*100 for m in methods]
    bars = ax.bar(range(3), rates, color=colors)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("factibilidad [%]"); ax.set_title(name)
    for b, r in zip(bars, rates):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{r:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Factibilidad con la MISMA reparacion (sin reparar, la restriccion de simplex es inviable)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_baselines_feasibility.pdf")); plt.close(fig)

# ---- Fig 2: robustness selection gain (mean feasible vs selected minS) ----
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
for ax, (name, d) in zip(axes, [("CNC", cnc), ("HEA", hea)]):
    ref = d["reject_repair"]
    vals = [ref["mean_S"], ref["min_S"]]
    bars = ax.bar(["feasible\ntipico (mean S)", "seleccion\nrobusta (min S)"], vals,
                  color=["#c44e52", "#55a868"])
    ax.set_ylabel("sensibilidad S (menor = mas robusto)"); ax.set_title(name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.3g}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Ganancia de la seleccion por robustez sobre el mismo conjunto factible reparado")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_baselines_robustness.pdf")); plt.close(fig)

print("wrote fig_baselines_feasibility.pdf, fig_baselines_robustness.pdf")
