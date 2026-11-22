"""Figures for the README from the CSVs that bench/work_precision.jl writes.

    julia --project=. bench/work_precision.jl
    python3 scripts/figures.py
"""

import csv
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
COLOR = {"RadauIIA5": BLUE, "Rosenbrock23": ORANGE, "DOPRI5": VIOLET, "BackwardEuler": AQUA}
NAME = {"RadauIIA5": "Radau IIA (order 5)", "Rosenbrock23": "Rosenbrock 2(3)", "DOPRI5": "Dormand-Prince 5(4)",
        "BackwardEuler": "backward Euler"}
MARK = {"RadauIIA5": "o", "Rosenbrock23": "s", "DOPRI5": "D", "BackwardEuler": "^"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9.5, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def ring():
    return dict(markeredgecolor=SURFACE, markeredgewidth=1.2)


def read(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def stability():
    grid = defaultdict(dict)
    for r in read("stability.csv"):
        grid[r["method"]][(float(r["re"]), float(r["im"]))] = float(r["absR"])
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, m in zip(axes, ["DOPRI5", "Rosenbrock23", "RadauIIA5", "BackwardEuler"]):
        pts = grid[m]
        res = sorted({k[0] for k in pts})
        ims = sorted({k[1] for k in pts})
        z = np.array([[pts[(x, y)] for x in res] for y in ims])
        z = np.where(np.isfinite(z), z, 1e9)
        ax.contourf(res, ims, (z <= 1).astype(float), levels=[0.5, 1.5], colors=[COLOR[m]], alpha=0.28)
        ax.contour(res, ims, z, levels=[1.0], colors=[COLOR[m]], linewidths=1.6)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.axvline(0, color=AXIS, linewidth=0.8)
        ax.set_aspect("equal")
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_title(NAME[m], loc="left", fontsize=10)
        ax.set_xlabel("Re(hλ)")
        ax.grid(False)
    axes[0].set_ylabel("Im(hλ)")
    fig.suptitle("Stability regions: where |R(hλ)| ≤ 1 (shaded)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "stability.png", dpi=150)
    plt.close(fig)


def orders():
    rows = read("orders.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in ["BackwardEuler", "Rosenbrock23", "RadauIIA5", "DOPRI5"]:
        h = [float(r["h"]) for r in rows if r["method"] == m]
        e = [float(r["error"]) for r in rows if r["method"] == m]
        ax.loglog(h, e, color=COLOR[m], marker=MARK[m], label=NAME[m], **ring())
    for p, y0 in ((1, 0.25), (2, 1.5e-2), (5, 2e-6)):
        hh = np.array([2 ** -7.5, 2 ** -2])
        ax.loglog(hh, y0 * (hh / hh[-1]) ** p, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(hh[0] * 0.92, y0 * (hh[0] / hh[-1]) ** p, f"slope {p}", ha="right", va="center", fontsize=8,
                color=INK2)
    ax.set_xlabel("fixed step h")
    ax.set_ylabel("error at t = 1")
    ax.set_title("Orders of convergence (rotation, exact solution)", loc="left")
    ax.set_xlim(2 ** -9.3, 0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "orders.png", dpi=150)
    plt.close(fig)


def work_precision():
    rows = read("work_precision.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    titles = {"rober": "Robertson, t to 1e11", "vdpol": "Van der Pol, ε = 1e-6"}
    for ax, prob in zip(axes, ["rober", "vdpol"]):
        for m in ["RadauIIA5", "Rosenbrock23", "DOPRI5"]:
            pts = [(float(r["digits"]), int(r["nf"])) for r in rows if r["problem"] == prob and r["method"] == m]
            if not pts:
                continue
            d, nf = zip(*pts)
            ax.semilogy(d, nf, color=COLOR[m], marker=MARK[m], label=NAME[m], **ring())
        ax.set_xlabel("correct digits at the end (vs the Test Set reference)")
        ax.set_title(titles[prob], loc="left")
    axes[0].set_ylabel("right-hand side evaluations")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "work_precision.png", dpi=150)
    plt.close(fig)


def solutions():
    rober = read("rober_trace.csv")
    steps = read("vdpol_steps.csv")
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1, 1, 1]})
    t = np.array([float(r["t"]) for r in rober])
    keep = t > 0
    for key, label, colour, scale in (("y1", "y1", BLUE, 1), ("y2", "y2 × 10⁴", ORANGE, 1e4), ("y3", "y3", AQUA, 1)):
        ax.semilogx(t[keep], scale * np.array([float(r[key]) for r in rober])[keep], color=colour, label=label)
    ax.set_xlabel("t")
    ax.set_title("Robertson (Radau IIA)", loc="left")
    ax.legend(loc="center left")
    for m in ["RadauIIA5", "Rosenbrock23", "DOPRI5"]:
        tt = [float(r["t"]) for r in steps if r["method"] == m]
        y1 = [float(r["y1"]) for r in steps if r["method"] == m]
        hh = [float(r["h"]) for r in steps if r["method"] == m]
        if m == "RadauIIA5":
            bx.plot(tt, y1, color=INK, linewidth=1.5)
        cx.semilogy(tt, hh, color=COLOR[m], label=NAME[m], linewidth=1.2 if m == "DOPRI5" else 1.6)
    bx.set_xlabel("t")
    bx.set_title("Van der Pol, ε = 1e-6: y1", loc="left")
    cx.set_xlabel("t")
    cx.set_title("Step sizes at rtol = 1e-6", loc="left")
    cx.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "solutions.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    stability()
    orders()
    work_precision()
    solutions()
    print("wrote", ", ".join(p.name for p in sorted((ROOT / "docs").glob("*.png"))))
