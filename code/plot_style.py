"""Shared plotting style for GraphPTC analysis figures.
Unified color palette and matplotlib rcParams so all figures look consistent.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- unified palette ----
# arms
C_DIRECT   = "#9AA0A6"   # neutral gray
C_PTC      = "#E0913C"   # warm orange
C_GRAPHPTC = "#2C6E9C"   # deep blue (our method, primary)
# backbones (muted qualitative)
BACKBONE = {
    "GPT-5.6-Sol":     "#2C6E9C",  # blue
    "Claude-Opus-5":   "#C0504D",  # muted red
    "DeepSeek-V4-Pro": "#4F9D69",  # green
    "Kimi-K3":         "#8E6FB0",  # purple
}
C_POS = "#2C6E9C"
C_NEG = "#C0504D"

def setup():
    try:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]
    except Exception:
        pass
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.6,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })
