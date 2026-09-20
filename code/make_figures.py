"""Generate individual GraphPTC analysis subfigures into ../figs/ as PDFs.
Each figure is standalone and sized for a three-across subfigure row.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_style as S
import data as D

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)
S.setup()

SZ = (2.75, 2.45)  # per-subfigure size


def _save(fig, name):
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def fig_scaling():
    fig, ax = plt.subplots(figsize=SZ)
    x = np.arange(len(D.SCALING_BINS))
    for bk in D.SCALING_BACKBONES:
        ax.plot(x, D.SCALING_GAIN[bk], marker="o", ms=3.4, color=S.BACKBONE[bk], label=bk)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(["$\\leq$4", "5-8", "9-14", "$\\geq$15"], fontsize=7)
    ax.set_xlabel("Execution-graph size", fontsize=8)
    ax.set_ylabel("Gain over Direct Tool Calling (points)", fontsize=7)
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left", handlelength=1.0, labelspacing=0.2)
    _save(fig, "scaling.pdf")


def fig_efficiency():
    fig, ax = plt.subplots(figsize=SZ)
    for key, lab, c, mk, ls in [("Direct", "Direct Tool Calling", S.C_DIRECT, "o", "--"), ("PTC", "PTC", S.C_PTC, "s", "-."), ("GraphPTC", "GraphPTC", S.C_GRAPHPTC, "D", "-")]:
        ax.plot(D.EFF_BUDGET, D.EFF[key], marker=mk, ms=3.4, ls=ls, color=c, label=lab)
    ax.set_xlabel("Interaction-round budget", fontsize=8)
    ax.set_ylabel("Cumulative pass rate (%)", fontsize=8)
    ax.set_xticks(D.EFF_BUDGET)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.4)
    _save(fig, "efficiency.pdf")


def fig_recovery():
    fig, ax = plt.subplots(figsize=SZ)
    bks = [b for b in D.BACKBONES if b in D.RECOVERY]
    x = np.arange(len(bks)); w = 0.36
    ax.bar(x - w/2, [D.RECOVERY[b][0] for b in bks], w, color=S.C_DIRECT, label="intent realized")
    ax.bar(x + w/2, [D.RECOVERY[b][1] for b in bks], w, color=S.C_GRAPHPTC, label="not realized")
    ax.set_xticks(x); ax.set_xticklabels(bks, rotation=20, ha="right", fontsize=6.2)
    ax.set_ylabel("$P(\\mathrm{recover})$ (%)", fontsize=8)
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.0)
    _save(fig, "recovery.pdf")


def fig_capgain():
    fig, ax = plt.subplots(figsize=SZ)
    bks = [b for b in D.BACKBONES if D.GROUP[b] == "frontier"] + [b for b in D.BACKBONES if D.GROUP[b] == "second-tier"]
    x = np.arange(len(bks))
    colors = [S.C_GRAPHPTC if D.GROUP[b] == "second-tier" else S.C_DIRECT for b in bks]
    ax.bar(x, [D.GAIN_OVER_DIRECT[b] for b in bks], 0.6, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(bks, rotation=20, ha="right", fontsize=6.2)
    ax.set_ylabel("Gain over Direct Tool Calling (points)", fontsize=7)
    ax.grid(axis="y")
    ax.legend(handles=[Patch(color=S.C_DIRECT, label="frontier"),
                       Patch(color=S.C_GRAPHPTC, label="second-tier")],
              frameon=False, fontsize=6.2, loc="upper left", handlelength=1.0)
    _save(fig, "gain.pdf")


def fig_fidelity():
    fig, ax = plt.subplots(figsize=SZ)
    bks = [b for b in D.BACKBONES if b in D.FIDELITY]
    x = np.arange(len(bks))
    ax.bar(x, [D.FIDELITY[b] for b in bks], 0.6, color=S.C_GRAPHPTC)
    ax.set_ylim(80, 100)
    ax.set_xticks(x); ax.set_xticklabels(bks, rotation=20, ha="right", fontsize=6.2)
    ax.set_ylabel("Realized rate (%)", fontsize=8)
    ax.grid(axis="y")
    _save(fig, "fidelity.pdf")


def fig_isolation():
    fig, ax = plt.subplots(figsize=SZ)
    bks = [b for b in D.BACKBONES if b in D.FAILURE_ISOLATION]
    x = np.arange(len(bks))
    ax.bar(x, [D.FAILURE_ISOLATION[b] for b in bks], 0.6, color=S.C_PTC)
    ax.set_ylim(0, 100)
    ax.set_xticks(x); ax.set_xticklabels(bks, rotation=20, ha="right", fontsize=6.2)
    ax.set_ylabel("Isolated failures (%)", fontsize=8)
    ax.grid(axis="y")
    _save(fig, "isolation.pdf")


if __name__ == "__main__":
    fig_scaling()
    fig_efficiency()
    fig_recovery()
    fig_capgain()
    fig_fidelity()
    fig_isolation()
    print("figures written to", os.path.abspath(OUT))
