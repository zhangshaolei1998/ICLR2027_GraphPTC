"""Raw data for GraphPTC analysis figures.
All numbers are computed from the frozen run artifacts under runs/**/results.jsonl
and the per-episode execution-graph snapshots. Kept here so every figure is reproducible.
"""

BACKBONES = ["GPT-5.6-Sol", "Claude-Opus-5", "DeepSeek-V4-Pro", "Kimi-K3"]

# ---- Complexity scaling: pass-rate gain of GraphPTC over per-call use, by execution-graph size (AppWorld) ----
SCALING_BINS = ["<=4", "5-8", "9-14", ">=15"]
SCALING_GAIN = {  # GraphPTC - Direct, percentage points
    "GPT-5.6-Sol":     [11.11, 13.77, 13.25, 24.07],
    "Claude-Opus-5":   [17.65, 5.61, 10.78, 16.43],
    "DeepSeek-V4-Pro": [25.00, 18.75, 8.44, 20.56],
    "Kimi-K3":         [9.09, 26.82, 38.22, 43.24],
}

# ---- Test-time efficiency: cumulative task pass rate (%) vs interaction-round budget (AppWorld), averaged over 4 backbones ----
EFF_BUDGET = [4, 8, 12, 16, 24, 32]
EFF = {  # arm -> avg cumulative pass rate at each budget
    "Direct":   [0.00, 0.98, 13.98, 39.10, 68.08, 74.05],
    "PTC":      [4.50, 32.03, 56.08, 71.25, 82.00, 83.25],
    "GraphPTC": [4.60, 35.73, 63.28, 78.70, 89.23, 90.38],
}

# ---- Non-linear recovery: probability of issuing a Patch/Replan action, conditioned on previous-block intent realization (AppWorld) ----
RECOVERY = {  # backbone -> (P(recover | realized), P(recover | not realized)) in %
    "GPT-5.6-Sol":     (9.84, 44.81),
    "Claude-Opus-5":   (8.31, 28.44),
    "DeepSeek-V4-Pro": (6.85, 35.89),
    "Kimi-K3":         (17.38, 60.71),
}

# ---- Backbone capability grouping by average Direct Tool Calling score over the seven settings ----
DIRECT_AVG = {  # seven-setting average of the Direct Tool Calling baseline
    "GPT-5.6-Sol": 63.24, "Claude-Opus-5": 69.33,
    "DeepSeek-V4-Pro": 54.53, "Kimi-K3": 50.72,
}
GAIN_OVER_DIRECT = {  # GraphPTC minus Direct, seven-setting average (percentage points)
    "GPT-5.6-Sol": 6.30, "Claude-Opus-5": 4.50,
    "DeepSeek-V4-Pro": 8.19, "Kimi-K3": 8.76,
}
GROUP = {  # stronger vs weaker by DIRECT_AVG
    "GPT-5.6-Sol": "stronger", "Claude-Opus-5": "stronger",
    "DeepSeek-V4-Pro": "weaker", "Kimi-K3": "weaker",
}

# ---- Fidelity of intent-effect binding: fraction of declared expected changes that are realized (AppWorld) ----
FIDELITY = {  # backbone -> realized rate (%)
    "GPT-5.6-Sol": 95.2, "Claude-Opus-5": 96.5,
    "DeepSeek-V4-Pro": 97.5, "Kimi-K3": 94.3,
}

# ---- Failure isolation: fraction of block failures not followed by another failure within three blocks (AppWorld) ----
FAILURE_ISOLATION = {
    "GPT-5.6-Sol": 73.5, "Claude-Opus-5": 84.8,
    "DeepSeek-V4-Pro": 71.8, "Kimi-K3": 60.5,
}

# ---- Token economy on AppWorld-challenge: (mean input tokens per task in thousands, TGC %) ----
TOKENS = {  # backbone -> arm -> (mean_input_ktokens, TGC)
    "GPT-5.6-Sol":     {"Direct": (188.2, 86.4), "PTC": (66.2, 92.4),  "GraphPTC": (84.3, 95.6)},
    "Claude-Opus-5":   {"Direct": (268.6, 80.8), "PTC": (121.9, 79.1), "GraphPTC": (168.2, 95.0)},
    "DeepSeek-V4-Pro": {"Direct": (245.5, 65.7), "PTC": (133.2, 80.1), "GraphPTC": (189.2, 84.2)},
    "Kimi-K3":         {"Direct": (213.1, 47.0), "PTC": (68.1, 83.2),  "GraphPTC": (92.2, 84.9)},
}



