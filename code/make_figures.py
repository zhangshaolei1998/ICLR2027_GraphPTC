"""Generate the three GraphPTC analysis figures into ../figs/ as PDFs."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import plot_style as S
import data as D

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)
S.setup()


def _panel_captions(fig, axes, caps):
    """Place panel captions (a)/(b)/(c) centered below each subplot."""
    fig.tight_layout(w_pad=1.4, rect=[0, 0.11, 1, 1])
    for ax, cap in zip(axes, caps):
        pos = ax.get_position()
        xc = 0.5 * (pos.x0 + pos.x1)
        fig.text(xc, 0.015, cap, ha="center", va="bottom", fontsize=8)


def fig_scaling():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    x = np.arange(len(D.SCALING_BINS))
    for bk in D.BACKBONES:
        ax.plot(x, D.SCALING_GAIN[bk], marker="o", ms=3.5, color=S.BACKBONE[bk], label=bk)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["$\\leq$4", "5-8", "9-14", "$\\geq$15"])
    ax.set_xlabel("Execution-graph size (program blocks)")
    ax.set_ylabel("Gain over Direct (points)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=1, loc="upper left", handlelength=1.2, borderaxespad=0.2)
    fig.savefig(os.path.join(OUT, "scaling.pdf"))
    plt.close(fig)


def fig_efficiency():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    arms = [("Direct", S.C_DIRECT, "o", "--"),
            ("PTC", S.C_PTC, "s", "-."),
            ("GraphPTC", S.C_GRAPHPTC, "D", "-")]
    for name, c, mk, ls in arms:
        ax.plot(D.EFF_BUDGET, D.EFF[name], marker=mk, ms=3.5, ls=ls, color=c, label=name)
    ax.set_xlabel("Interaction-round budget")
    ax.set_ylabel("Cumulative pass rate (%)")
    ax.set_xticks(D.EFF_BUDGET)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    fig.savefig(os.path.join(OUT, "efficiency.pdf"))
    plt.close(fig)


def fig_recovery():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    bks = D.BACKBONES
    x = np.arange(len(bks))
    w = 0.36
    realized = [D.RECOVERY[b][0] for b in bks]
    unreal = [D.RECOVERY[b][1] for b in bks]
    ax.bar(x - w/2, realized, w, color=S.C_DIRECT, label="intent realized")
    ax.bar(x + w/2, unreal, w, color=S.C_GRAPHPTC, label="intent not realized")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("-", "-\n", 0) for b in bks], rotation=18, ha="right")
    ax.set_ylabel("$P(\\mathrm{recover})$ (%)")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    fig.savefig(os.path.join(OUT, "recovery.pdf"))
    plt.close(fig)


def fig_combined():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.15))

    # (a) complexity scaling
    ax = axes[0]
    x = np.arange(len(D.SCALING_BINS))
    for bk in D.BACKBONES:
        ax.plot(x, D.SCALING_GAIN[bk], marker="o", ms=3.2, color=S.BACKBONE[bk], label=bk)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(["$\\leq$4", "5-8", "9-14", "$\\geq$15"])
    ax.set_xlabel("Execution-graph size")
    ax.set_ylabel("Gain over Direct (points)")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=6.4, loc="upper left", handlelength=1.0, borderaxespad=0.15, labelspacing=0.2)

    # (b) efficiency Pareto
    ax = axes[1]
    for name, c, mk, ls in [("Direct", S.C_DIRECT, "o", "--"), ("PTC", S.C_PTC, "s", "-."), ("GraphPTC", S.C_GRAPHPTC, "D", "-")]:
        ax.plot(D.EFF_BUDGET, D.EFF[name], marker=mk, ms=3.2, ls=ls, color=c, label=name)
    ax.set_xlabel("Interaction-round budget")
    ax.set_ylabel("Cumulative pass rate (%)")
    ax.set_xticks(D.EFF_BUDGET)
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right", handlelength=1.4)

    # (c) recovery
    ax = axes[2]
    bks = D.BACKBONES
    x = np.arange(len(bks)); w = 0.36
    ax.bar(x - w/2, [D.RECOVERY[b][0] for b in bks], w, color=S.C_DIRECT, label="intent realized")
    ax.bar(x + w/2, [D.RECOVERY[b][1] for b in bks], w, color=S.C_GRAPHPTC, label="not realized")
    ax.set_xticks(x)
    ax.set_xticklabels(D.BACKBONES, rotation=20, ha="right", fontsize=6.2)
    ax.set_ylabel("$P(\\mathrm{recover})$ (%)")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=6.8, loc="upper center", handlelength=1.0)

    _panel_captions(fig, axes, ["(a) Advantage scales with complexity",
                                "(b) Same accuracy, fewer rounds",
                                "(c) Failure triggers recovery"])
    fig.savefig(os.path.join(OUT, "analysis.pdf"))
    plt.close(fig)


def fig_combined2():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.15))
    bks = D.BACKBONES
    x = np.arange(len(bks))
    short = ["GPT-5.6-Sol", "Claude-Opus-5", "DeepSeek-V4-Pro", "Kimi-K3"]

    # (a) gain over Direct by backbone capability
    ax = axes[0]
    colors = [S.C_GRAPHPTC if D.GROUP[b] == "weaker" else S.C_DIRECT for b in bks]
    ax.bar(x, [D.GAIN_OVER_DIRECT[b] for b in bks], 0.6, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha="right", fontsize=6.2)
    ax.set_ylabel("Gain over Direct (points)")
    ax.grid(axis="y")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=S.C_DIRECT, label="stronger backbone"),
                       Patch(color=S.C_GRAPHPTC, label="weaker backbone")],
              frameon=False, fontsize=6.4, loc="upper left", handlelength=1.0)

    # (b) intent-effect fidelity
    ax = axes[1]
    ax.bar(x, [D.FIDELITY[b] for b in bks], 0.6, color=S.C_GRAPHPTC)
    ax.set_ylim(80, 100)
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha="right", fontsize=6.2)
    ax.set_ylabel("Realized rate (%)")
    ax.grid(axis="y")

    # (c) failure isolation
    ax = axes[2]
    ax.bar(x, [D.FAILURE_ISOLATION[b] for b in bks], 0.6, color=S.C_PTC)
    ax.set_ylim(0, 100)
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha="right", fontsize=6.2)
    ax.set_ylabel("Isolated failures (%)")
    ax.grid(axis="y")

    _panel_captions(fig, axes, ["(a) Weaker backbones gain more",
                                "(b) Declared intents are realized",
                                "(c) Failures stay local"])
    fig.savefig(os.path.join(OUT, "analysis2.pdf"))
    plt.close(fig)


def fig_tokens():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    arm_style = {"Direct": (S.C_DIRECT, "o"), "PTC": (S.C_PTC, "s"), "GraphPTC": (S.C_GRAPHPTC, "D")}
    for bk in D.BACKBONES:
        pts = D.TOKENS[bk]
        xs = [pts[a][0] for a in ("Direct", "PTC", "GraphPTC")]
        ys = [pts[a][1] for a in ("Direct", "PTC", "GraphPTC")]
        ax.plot(xs, ys, "-", color="black", lw=0.5, alpha=0.35, zorder=1)
        for a in ("Direct", "PTC", "GraphPTC"):
            c, mk = arm_style[a]
            ax.scatter(pts[a][0], pts[a][1], s=34, color=c, marker=mk, zorder=3, edgecolor="white", linewidth=0.4)
        ax.annotate(bk, (pts["GraphPTC"][0], pts["GraphPTC"][1]), fontsize=6.0,
                    textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("Mean input tokens per task (thousands)")
    ax.set_ylabel("AppWorld-Challenge TGC (%)")
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=arm_style[a][1], color="w", markerfacecolor=arm_style[a][0],
                      markersize=6, label=a) for a in ("Direct", "PTC", "GraphPTC")]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tokens.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_scaling()
    fig_efficiency()
    fig_recovery()
    fig_combined()
    fig_combined2()
    fig_tokens()
    print("figures written to", os.path.abspath(OUT))
