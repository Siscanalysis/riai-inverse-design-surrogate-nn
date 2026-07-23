#!/usr/bin/env python3
"""Generate manuscript figures from the rerun/out CSVs. Output: PDF next to this file."""
import csv, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "..", "rerun", "out"))

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
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cnc_tradeoff.pdf")); plt.close(fig)

# ---------- Case A : 3-D manifold in the three continuous levers ----------
fig = plt.figure(figsize=(5.0, 4.0)); ax = fig.add_subplot(111, projection="3d")
p = ax.scatter(cnc["cutting_speed"], cnc["feed"], cnc["depth_of_cut"],
               c=cnc["F"], cmap="plasma", s=8, alpha=0.6)
ax.set_xlabel("$v_c$ [m/min]"); ax.set_ylabel("$f$ [mm/rev]"); ax.set_zlabel("$a_p$ [mm]")
ax.set_title("Fixed-$R_a$ recipe manifold (2-D surface in 3 levers)")
cb = fig.colorbar(p, ax=ax, shrink=0.6, pad=0.1); cb.set_label("$F$ [N]")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cnc_manifold3d.pdf")); plt.close(fig)

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
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_hea_parallel.pdf")); plt.close(fig)

# ---------- Case B : robustness landscape (two dominant elements) ----------
# pick the two highest-variance fractions for a 2-D projection
var = X.var(axis=0); i1, i2 = np.argsort(-var)[:2]
fig, ax = plt.subplots(figsize=(5.0, 3.6))
sc = ax.scatter(X[:, i1], X[:, i2], c=sens, cmap="viridis_r", s=18)
ax.set_xlabel(f"{elems[i1]} fraction"); ax.set_ylabel(f"{elems[i2]} fraction")
ax.set_title("HEA manifold projection, colored by robustness")
cb = fig.colorbar(sc, ax=ax); cb.set_label("robustness $S$ (dark = robust)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_hea_projection.pdf")); plt.close(fig)

print("wrote:", [f for f in os.listdir(HERE) if f.endswith(".pdf")])
