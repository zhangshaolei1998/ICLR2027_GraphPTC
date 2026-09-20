"""Appendix figures — turn several tables into bar charts."""
import os
import numpy as np
import matplotlib.pyplot as plt
import plot_style as S

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)
S.setup()

BK = ["GPT-5.6-Sol", "Claude-Opus-5", "DeepSeek-V4-Pro", "Kimi-K3"]

# ---- SGC on AppWorld ----
SGC = {
    "normal":    {"PTC": [83.90, 46.40, 80.40, 67.90],
                  "GraphPTC": [85.70, 92.90, 82.10, 76.80]},
    "challenge": {"PTC": [82.00, 69.10, 66.90, 66.90],
                  "GraphPTC": [87.10, 92.10, 88.50, 70.50]},
}


def fig_sgc():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55), sharey=True)
    for ax, split in zip(axes, ["normal", "challenge"]):
        x = np.arange(len(BK)); w = 0.36
        ax.bar(x - w/2, SGC[split]["PTC"], w, color=S.C_PTC, label="PTC")
        ax.bar(x + w/2, SGC[split]["GraphPTC"], w, color=S.C_GRAPHPTC, label="GraphPTC")
        ax.set_xticks(x); ax.set_xticklabels(BK, rotation=20, ha="right", fontsize=6.4)
        ax.set_title(f"AppWorld-{split.capitalize()}", fontsize=8)
        ax.grid(axis="y", zorder=0)
        ax.set_ylim(30, 100)
    axes[0].set_ylabel("Scenario Goal Completion (\\%)", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.2)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT, "sgc.pdf")); plt.close(fig)


# ---- Rounds on AutomationBench (Direct Tool Calling vs GraphPTC) ----
AB_R = {
    "Direct":   [32.4, 27.0, 54.8, 30.2],
    "GraphPTC": [11.5, 11.6, 20.1, 13.9],
}


def fig_ab_rounds():
    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    x = np.arange(len(BK)); w = 0.36
    ax.bar(x - w/2, AB_R["Direct"], w, color=S.C_DIRECT, label="Direct Tool Calling")
    ax.bar(x + w/2, AB_R["GraphPTC"], w, color=S.C_GRAPHPTC, label="GraphPTC")
    ax.set_xticks(x); ax.set_xticklabels(BK, rotation=20, ha="right", fontsize=6.4)
    ax.set_ylabel("Model rounds per task", fontsize=8)
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, fontsize=6.6, loc="upper right", handlelength=1.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "ab_rounds.pdf")); plt.close(fig)


# ---- Input tokens across benchmarks (k) ----
BM = ["AppWorld-N", "AppWorld-C", "Agent-Diff", "FanOutQA", "FRAMES", "AutomationBench"]
TOK = {
    "Direct":   [151.4, 228.8, 113.2, 128.5,  93.0, 324.4],
    "PTC":      [ 60.2,  97.3,  34.4,  49.5,  42.2, 156.6],
    "GraphPTC": [ 84.1, 133.5,  49.0,  63.9,  55.2, 171.9],
}


def fig_tokens():
    fig, ax = plt.subplots(figsize=(6.6, 2.55))
    x = np.arange(len(BM)); w = 0.27
    ax.bar(x - w,   TOK["Direct"],   w, color=S.C_DIRECT,   label="Direct Tool Calling")
    ax.bar(x,       TOK["PTC"],      w, color=S.C_PTC,      label="PTC")
    ax.bar(x + w,   TOK["GraphPTC"], w, color=S.C_GRAPHPTC, label="GraphPTC")
    ax.set_xticks(x); ax.set_xticklabels(BM, rotation=15, ha="right", fontsize=6.6)
    ax.set_ylabel("Mean input tokens per task (k)", fontsize=8)
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right", handlelength=1.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "tokens_multi.pdf")); plt.close(fig)


# ---- FRAMES gain by reasoning type ----
FR = {
    "GPT-5.6-Sol":   [1.5, 1.4, 4.7, 1.7, 2.2],
    "Claude-Opus-5": [-0.6, 1.4, 0.9, 0.4, 2.2],
}
FR_TYPES = ["Multiple\nconstraints", "Numerical", "Post-\nprocessing", "Tabular", "Temporal"]


def fig_frames_type():
    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    x = np.arange(len(FR_TYPES)); w = 0.36
    ax.bar(x - w/2, FR["GPT-5.6-Sol"],   w, color=S.BACKBONE["GPT-5.6-Sol"],   label="GPT-5.6-Sol")
    ax.bar(x + w/2, FR["Claude-Opus-5"], w, color=S.BACKBONE["Claude-Opus-5"], label="Claude-Opus-5")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(FR_TYPES, fontsize=6.2)
    ax.set_ylabel("Gain over stronger baseline (pts)", fontsize=7.6)
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left", handlelength=1.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "frames_type.pdf")); plt.close(fig)


# ---- Agent-Diff split by state-overwrite ----
OPS = {
    "Direct Tool Calling": [63.3, 79.2],
    "PTC":                 [58.3, 79.0],
    "GraphPTC":            [65.5, 78.7],
}
OPS_TYPES = ["Update or delete state", "No state overwrite"]


def fig_opsplit():
    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    x = np.arange(len(OPS_TYPES)); w = 0.24
    ax.bar(x - w, OPS["Direct Tool Calling"], w, color=S.C_DIRECT,   label="Direct Tool Calling")
    ax.bar(x,     OPS["PTC"],                 w, color=S.C_PTC,      label="PTC")
    ax.bar(x + w, OPS["GraphPTC"],            w, color=S.C_GRAPHPTC, label="GraphPTC")
    ax.set_xticks(x); ax.set_xticklabels(OPS_TYPES, fontsize=6.6)
    ax.set_ylabel("Pass rate (\\%)", fontsize=8)
    ax.set_ylim(50, 97)
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, fontsize=6.4, loc="upper center", ncol=3,
              handlelength=1.0, columnspacing=1.0, borderaxespad=0.1)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "opsplit.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_sgc()
    fig_ab_rounds()
    fig_tokens()
    fig_frames_type()
    fig_opsplit()
    print("appendix figures written to", os.path.abspath(OUT))
