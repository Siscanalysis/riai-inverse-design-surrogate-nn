#!/usr/bin/env python3
"""
Extrapolation-risk analysis of the feasible manifolds returned by each baseline.

For a surrogate model, a returned design is trustworthy only where the net interpolates
its training data. We measure that two ways, per feasible point of each method:

  1. wall margin  = worst-case normalized distance to the nearest domain-box wall
                    (1 = box centre, 0 = on a wall). This is exactly the centrality the
                    selector uses. LOW margin => point at the box edge => extrapolation.
  2. dist-to-train = min distance to any TRAINING input, in per-dimension normalized units
                    (span-normalized L2 / sqrt(p)). HIGH => far from real data => extrapolation.

We then ask, honestly: does OUR manifold sit more in the well-sampled interior than the
baselines' (higher wall margin / lower dist-to-train), i.e. is it safer against extrapolation?

Run:  python analyze_extrapolation.py cnc|hea
"""
import sys, os, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = sys.argv[1] if len(sys.argv) > 1 else "cnc"
N_IN = {"cnc": 4, "hea": 10}[CASE]
TRAIN = {"cnc": "cnc_multi_fulldata.csv", "hea": "hea_hardness_10elt.csv"}[CASE]
METHODS = ["reject_norepair", "reject_repair", "ours_iterative"]
LABEL = {"reject_norepair": "Rechazo", "reject_repair": "Rechazo+rep",
         "ours_iterative": "Nuestro"}

def read_matrix(path, ncol):
    rows = list(csv.reader(open(path)))
    out = []
    for r in rows[1:]:
        if len(r) < ncol: continue
        try: out.append([float(v) for v in r[:ncol]])
        except ValueError: continue
    return np.array(out) if out else np.zeros((0, ncol))

# training inputs + domain box
Xtr = read_matrix(os.path.join(HERE, "data", TRAIN), N_IN)
dom_csv = os.path.join(HERE, "out", "cnc_domain.csv")
if CASE == "cnc" and os.path.exists(dom_csv):
    d = list(csv.reader(open(dom_csv)))[1:]
    lo = np.array([float(r[1]) for r in d]); up = np.array([float(r[2]) for r in d])
else:
    lo, up = Xtr.min(0), Xtr.max(0)
span = np.maximum(up - lo, 1e-9)
Xtr_n = (Xtr - lo) / span                       # normalized training cloud

def wall_margin(X):
    Xn = (X - lo) / span
    m = np.minimum(Xn, 1.0 - Xn) / 0.5          # per-dim margin, 1=centre 0=wall
    return np.clip(m, 0, 1).min(1)              # worst-case over dims

def dist_to_train(X):
    Xn = (X - lo) / span
    # min over training points of span-normalized per-dim-averaged L2 distance
    out = np.empty(len(Xn))
    for i, x in enumerate(Xn):
        dd = np.sqrt(((Xtr_n - x) ** 2).sum(1) / N_IN)
        out[i] = dd.min()
    return out

print(f"=== extrapolation analysis ({CASE}) : {len(Xtr)} training rows ===")
print(f"{'method':12s} {'n':>5s} {'wallMargin mean':>15s} {'frac<0.10':>9s} "
      f"{'d2train mean':>12s} {'d2train p90':>11s}")
summary = {}
for m in METHODS:
    p = os.path.join(HERE, "out", f"feasible_{CASE}_{m}.csv")
    if not os.path.exists(p): continue
    X = read_matrix(p, N_IN)
    if len(X) == 0:
        print(f"{LABEL[m]:12s} {0:5d}  (no feasible points)"); continue
    wm = wall_margin(X); dt = dist_to_train(X)
    summary[m] = (wm, dt)
    print(f"{LABEL[m]:12s} {len(X):5d} {wm.mean():15.3f} {np.mean(wm<0.10):9.3f} "
          f"{dt.mean():12.4f} {np.percentile(dt,90):11.4f}")

# figure: wall-margin CDF (higher/right = safer) per method
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    col = {"reject_norepair": "#999999", "reject_repair": "#1f77b4",
           "ours_iterative": "#d62728", "nsga2_repair": "#2ca02c"}
    for m in METHODS:
        if m not in summary: continue
        wm, dt = summary[m]
        xs = np.sort(wm); ys = np.arange(1, len(xs)+1)/len(xs)
        ax[0].plot(xs, ys, label=LABEL[m], color=col[m], lw=2)
        xs2 = np.sort(dt); ys2 = np.arange(1, len(xs2)+1)/len(xs2)
        ax[1].plot(xs2, ys2, label=LABEL[m], color=col[m], lw=2)
    ax[0].axvline(0.10, ls="--", color="k", lw=0.8)
    ax[0].set_xlabel("margen a la pared (1=centro, 0=borde)"); ax[0].set_ylabel("CDF")
    ax[0].set_title("Centralidad: mas a la derecha = mas seguro")
    ax[1].set_xlabel("distancia al dato de entrenamiento mas cercano")
    ax[1].set_title("Extrapolacion: mas a la izquierda = mas seguro")
    for a in ax: a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout()
    outpdf = os.path.join(HERE, "..", "paper", "figs", f"fig_extrapolation_{CASE}.pdf")
    fig.savefig(outpdf); print(f"wrote {os.path.normpath(outpdf)}")
except Exception as e:
    print(f"[figure skipped: {e}]")
