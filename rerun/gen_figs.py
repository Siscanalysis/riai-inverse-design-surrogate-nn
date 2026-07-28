#!/usr/bin/env python3
"""Generate the manifold figures from the rerun/out CSVs. Output: rerun/figs/*.pdf.
Prereqs: run case_a_cnc.exe and case_b_hea.exe first (they write out/cnc_manifold.csv
and out/hea_manifold.csv). Run:  python gen_figs.py"""
import csv, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "out")
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

def load(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols = {k: [] for k in r.fieldnames}
        for row in r:
            for k in r.fieldnames:
                try: cols[k].append(float(row[k]))
                except (ValueError, TypeError): cols[k].append(math.nan)
    return {k: np.array(v) for k, v in cols.items()}, r.fieldnames

plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3})

# ---------- Case A : CNC mixed-sense trade-off (F vs MRR, colored by robustness) ----------
cnc, _ = load(os.path.join(OUT, "cnc_manifold.csv"))
fig, ax = plt.subplots(figsize=(5.0, 3.6))
sc = ax.scatter(cnc["F"], cnc["MRR"], c=cnc["sensitivity_Ra"], cmap="viridis",
                s=10, alpha=0.7)
ax.set_xlabel("Cutting force $F$ [N]  (minimize $\\rightarrow$)")
ax.set_ylabel("MRR  (maximize $\\uparrow$)")
ax.set_title("CNC: mixed-sense trade-off on the fixed-$R_a$ manifold")
cb = fig.colorbar(sc, ax=ax); cb.set_label("robustness $S$ (lower = more robust)")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_cnc_tradeoff.pdf")); plt.close(fig)

# ---------- Case A : 3-D manifold in the three continuous levers ----------
fig = plt.figure(figsize=(5.0, 4.0)); ax = fig.add_subplot(111, projection="3d")
p = ax.scatter(cnc["cutting_speed"], cnc["feed"], cnc["depth_of_cut"],
               c=cnc["F"], cmap="plasma", s=8, alpha=0.6)
ax.set_xlabel("$v_c$ [m/min]"); ax.set_ylabel("$f$ [mm/rev]"); ax.set_zlabel("$a_p$ [mm]")
ax.set_title("Fixed-$R_a$ recipe manifold (2-D surface in 3 levers)")
cb = fig.colorbar(p, ax=ax, shrink=0.6, pad=0.1); cb.set_label("$F$ [N]")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_cnc_manifold3d.pdf")); plt.close(fig)

# ---------- Case B : HEA parallel coordinates of the 10 fractions ----------
hea, fields = load(os.path.join(OUT, "hea_manifold.csv"))
elems = [c for c in fields if c not in ("HV", "sensitivity")]
X = np.vstack([hea[e] for e in elems]).T                    # (n, 10)
sens = hea["sensitivity"]
order = np.argsort(-sens)                                    # robust (low S) drawn last/on top
fig, ax = plt.subplots(figsize=(6.4, 3.6))
norm = plt.Normalize(sens.min(), sens.max())
cmap = plt.cm.viridis_r
xax = np.arange(len(elems))
for i in order:
    ax.plot(xax, X[i], color=cmap(norm(sens[i])), alpha=0.35, lw=0.8)
ax.set_xticks(xax); ax.set_xticklabels(elems)
ax.set_ylabel("atomic fraction")
ax.set_title("HEA: alloy-composition manifold at fixed hardness (parallel coords)")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cb = fig.colorbar(sm, ax=ax); cb.set_label("robustness $S$ (dark = robust)")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_hea_parallel.pdf")); plt.close(fig)

# ---------- Case C : steel ductility-cost Pareto on the fixed-UTS manifold (T=27 C) ----------
steel_manifold = os.path.join(OUT, "steel_manifold.csv")
steel_pareto   = os.path.join(OUT, "steel_pareto.csv")
if os.path.exists(steel_manifold) and os.path.exists(steel_pareto):
    man, _ = load(steel_manifold)
    par, _ = load(steel_pareto)
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    sc = ax.scatter(man["cost"], man["elongation"], c=man["sensitivity_elong"], cmap="viridis",
                    s=9, alpha=0.45, label="fixed-UTS manifold")
    order = np.argsort(par["cost"])
    ax.plot(par["cost"][order], par["elongation"][order], "-o", color="crimson",
            ms=5, lw=1.6, label="ductility-cost Pareto")
    ax.set_xlabel("alloying cost proxy  (minimize $\\rightarrow$)")
    ax.set_ylabel("elongation [%]  (maximize $\\uparrow$)")
    ax.set_title("Steel: ductility-cost Pareto on the fixed-UTS manifold ($T=27^\\circ$C)")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("robustness $S$ (lower = more robust)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_steel_pareto.pdf")); plt.close(fig)

print("wrote:", sorted(f for f in os.listdir(FIGS) if f.endswith(".pdf")))
