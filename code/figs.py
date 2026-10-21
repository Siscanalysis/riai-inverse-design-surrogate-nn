"""Figures of the manuscript, from out/metrics_*.csv and the returned points.

Usage (from code/):  python figs.py      (after analyze.py)

  figs/fig_sets.pdf       valid solutions against the specification tolerance (set recovery)
  figs/fig_fronts.pdf     two-objective fronts on the specification (concrete, steel), seed 1
  figs/fig_catalogue.pdf  the attainable set of the kernel catalogue, and what rounding a local
                          solution onto that catalogue does to the specification

Palette: the validated categorical order of the dataviz reference (blue, orange, aqua,
yellow, violet); every series also has its own marker, and every figure a legend.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import problems as P

HERE = Path(__file__).resolve().parent
OUT, FIGS = HERE / "out", HERE / "figs"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, MUTED = "#0b0b0b", "#52514e"
TITLES = {"concrete": "Hormigón", "steel": "Acero"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "pdf.fonttype": 42,
})


def save(fig, name):
    """PDF for the manuscript, PNG for visual checking."""
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=200)


def fig_sets():
    m = pd.read_csv(OUT / "metrics_sample.csv")
    taus = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    # The continuous cases compare statements, so the only external series here is the naive
    # alternative; the catalogue case has its own figure.
    styles = {"repair": ("Especificación como restricción", BLUE, "o"),
              "rejection": ("Muestreo por rechazo", AQUA, "^")}
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), sharey=False)
    for ax, case in zip(axes, ["concrete", "steel"]):
        for method, (label, color, marker) in styles.items():
            g = m[(m["case"] == case) & (m["method"] == method)]
            if g.empty:
                continue
            vals = np.array([[row[f"valid_{t:g}"] for t in taus] for _, row in g.iterrows()], float)
            med = np.median(vals, axis=0)
            lo, hi = np.percentile(vals, 25, axis=0), np.percentile(vals, 75, axis=0)
            y = np.maximum(med, 0.5)
            ax.plot(taus, y, color=color, marker=marker, markersize=5, label=label)
            ax.fill_between(taus, np.maximum(lo, 0.5), np.maximum(hi, 0.5), color=color, alpha=0.15, linewidth=0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_ylim(0.4, None)
        ax.set_title(TITLES[case], color=INK, fontsize=8)
        ax.set_xlabel(r"tolerancia $\tau$ sobre $|f-c|/|c|$")
    axes[0].set_ylabel("soluciones válidas")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_sets")
    plt.close(fig)


def fig_fronts(seed=1):
    styles = {"dc_constraint": ("DC, restricción", BLUE, "o"), "ga_constraint": ("GA, restricción", ORANGE, "s"),
              "nsga2": ("NSGA-II (pymoo)", AQUA, "^"), "dc_objective-matched": ("DC, objetivo fijo", YELLOW, "D"),
              "ga_objective-matched": ("GA, objetivo fijo", VIOLET, "v")}
    labels = {"concrete": ("cenizas volantes [kg/m$^3$]", "conglomerante total [kg/m$^3$]"),
              "steel": ("alargamiento [%]", "Ni+Cr+Mo+Cu+V [% en peso]")}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    for ax, case in zip(axes, ["concrete", "steel"]):
        p = P.load(case)
        for method, (label, color, marker) in styles.items():
            folder = "baselines" if method == "nsga2" else "solve"
            name = f"{case}_bi_{method}_s{seed}.csv" if method == "nsga2" else \
                f"{case}_{method.replace('_', '_', 1)}_bi_s{seed}.csv"
            if method != "nsga2":
                solver, formulation = method.split("_", 1)
                name = f"{case}_{solver}_{formulation}_bi_s{seed}.csv"
            path = OUT / folder / name
            if not path.exists():
                continue
            x, _ = P.read_rows(path, p)
            if len(x) == 0:
                continue
            y = p.evaluate(x)
            ok = p.feasible_except_spec(x, y) & (p.spec_residual(x, y) <= 1e-4)
            if not ok.any():
                continue
            v = np.column_stack([p.value(o["expression"], x[ok], y[ok]) for o in p.bi])
            ax.scatter(v[:, 0], v[:, 1], s=14, color=color, marker=marker, label=label,
                       edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_xlabel(labels[case][0]); ax.set_ylabel(labels[case][1])
        ax.set_title(TITLES[case], color=INK, fontsize=8)
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    save(fig, "fig_fronts")
    plt.close(fig)


def fig_catalogue(seed=1):
    """The catalogue case in two panels: why the equality has no solution, and why rounding
    at the end of a local solve throws the optimum out of the feasible region."""
    p = P.load("kernel")
    target = 10.0 ** p.target
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))

    x, y = P.read_rows(OUT / "baselines" / "kernel_sample_enumeration_s1.csv", p)
    residual = np.sort(p.spec_residual(x, y))
    axes[0].step(residual, np.arange(1, len(residual) + 1), color=BLUE, where="post")
    for width, label in ((p.band, r"banda $1\,\%$"), (residual[0], "más próxima")):
        axes[0].axvline(width, color=MUTED, linestyle=":", linewidth=1)
        axes[0].annotate(label, (width, 1.4), rotation=90, fontsize=6, color=MUTED,
                         ha="right", va="bottom")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlim(1e-4, 1.0)
    axes[0].set_ylim(0.8, len(residual))
    axes[0].set_xlabel(r"distancia relativa al objetivo")
    axes[0].set_ylabel("configuraciones del catálogo")
    axes[0].set_title("Conjunto alcanzable", color=INK, fontsize=8)

    bins = np.logspace(-4, 0.5, 40)
    for name, label, color in (("slsqp_relaxed-band", "antes de redondear", ORANGE),
                               ("slsqp_round-band", "después de redondear", VIOLET)):
        path = OUT / "baselines" / f"kernel_single_{name}_s{seed}.csv"
        if not path.exists():
            continue
        xs, ys = P.read_rows(path, p)
        axes[1].hist(np.clip(p.spec_residual(xs, ys), bins[0], bins[-1]), bins=bins,
                     histtype="step", color=color, label=label, linewidth=1.6)
    axes[1].axvline(p.band, color=MUTED, linestyle=":", linewidth=1)
    axes[1].annotate(r"banda $1\,\%$", (p.band, 1.0), rotation=90, fontsize=6, color=MUTED,
                     ha="right", va="bottom")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"residuo de la especificación $|f-c|/|c|$")
    axes[1].set_ylabel("soluciones")
    axes[1].set_title("Efecto del redondeo al catálogo", color=INK, fontsize=8)
    axes[1].margins(y=0.25)                       # headroom over the spike at the band edge
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_catalogue")
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    fig_sets()
    fig_fronts()
    fig_catalogue()
    print("figures written to", FIGS)
