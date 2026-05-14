"""
Line plots of accuracy vs. KV cache budget on Qwen3-14B for AIME26 and HMMT25.

Layout: 1 x 2 panels.
  - Left  : AIME26
  - Right : HMMT25
Each panel: 4 method lines (RKV, CUR Fixed, CUR Resample, Sample Attn + V)
plus a horizontal "Full" reference line.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

os.makedirs("plots", exist_ok=True)

# ── Typography (match plot_pilot_study.py) ──────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# ── Data ─────────────────────────────────────────────────────────────────────
BUDGETS = [2048, 4096, 6144]

DATA = {
    "AIME26": {
        "RKV":             [40.00, 60.21, 69.37],
        "CUR Fixed":       [34.79, 61.25, 68.75],
        "CUR Resample":    [57.92, 68.75, 75.42],
        "Sample Attn + V": [48.54, 72.29, 73.12],
        "Full":            76.04,
        "FullTokens":      15325,
    },
    "HMMT25": {
        "RKV":             [25.62, 44.58, 50.00],
        "CUR Fixed":       [24.58, 39.58, 47.71],
        "CUR Resample":    [35.00, 49.38, 51.04],
        "Sample Attn + V": [32.08, 50.00, 53.54],
        "Full":            54.79,
        "FullTokens":      17847,
    },
}

# ── Palette (consistent with plot_pilot_study.py) ───────────────────────────
COLORS = {
    "RKV":             "#7E8A99",  # neutral grey (baseline)
    "CUR Fixed":       "#E48A6B",  # warm orange (baseline)
    "CUR Resample":    "#5DA975",  # green — improved CUR
    "Sample Attn + V": "#1F4F66",  # deep blue — our flagship
}
MARKERS = {
    "RKV":             "o",
    "CUR Fixed":       "s",
    "CUR Resample":    "^",
    "Sample Attn + V": "D",
}
DISPLAY_NAMES = {
    "RKV":             "R-KV",
    "CUR Fixed":       "CurDKV",
    "CUR Resample":    "VaSE-DKV",
    "Sample Attn + V": "VaSE-AttnV",
    "Full":            "Full",
}
METHOD_ORDER = ["RKV", "CUR Fixed", "CUR Resample", "Sample Attn + V"]

FULL_COLOR = "#242424"
GRID_GREY  = "#E7E7E7"
TEXT_DARK  = "#222222"

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), facecolor="white")
TASKS = ["AIME26", "HMMT25"]

for ax, task in zip(axes, TASKS):
    d = DATA[task]
    for m in METHOD_ORDER:
        ax.plot(BUDGETS, d[m],
                marker=MARKERS[m], markersize=7,
                linewidth=2.0, color=COLORS[m],
                label=DISPLAY_NAMES[m], zorder=3)
    # Full as horizontal dashed reference
    ax.axhline(d["Full"], color=FULL_COLOR, linestyle="--",
               linewidth=1.6, label=f"Full ({d['FullTokens']/1000:.1f}k tokens)", zorder=2)

    ax.set_title(f"{task}", color=TEXT_DARK, pad=8, fontweight='bold')
    ax.set_xlabel("KV Cache Budget", fontsize=16)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([f"{b}\n" for b in BUDGETS], fontweight='bold')
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for b in BUDGETS:
        ax.text(b, -0.09, f"{d['FullTokens']/b:.1f}x",
                transform=trans, ha='center', va='top',
                color='grey', fontsize=12, fontweight='bold', clip_on=False)
    if task == "AIME26":
        ax.text(-0.02, -0.05, "Tokens", transform=ax.transAxes,
                ha='right', va='center', fontweight='bold', fontsize=12,
                clip_on=False)
        ax.text(-0.02, -0.115, "Compress", transform=ax.transAxes,
                ha='right', va='center', fontweight='bold', fontsize=12,
                color='grey', clip_on=False)
    # Pad y-range below the lowest method point and above the Full line
    all_vals = [v for m in METHOD_ORDER for v in d[m]] + [d["Full"]]
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.10
    ax.set_ylim(lo - pad, hi + pad)
    # For AIME26, Full=76.04 doesn't fall on a regular tick — add it explicitly.
    if task == "AIME26":
        ticks = list(ax.get_yticks())
        ticks = [t for t in ticks if abs(t - d["Full"]) > 1.5]
        ticks.append(d["Full"])
        ticks = sorted(t for t in ticks if lo - pad <= t <= hi + pad)
        ax.set_yticks(ticks)
        ax.set_yticklabels([
            f"{d['Full']:.1f}" if abs(t - d["Full"]) < 1e-6 else f"{int(round(t))}"
            for t in ticks
        ])
    ax.yaxis.grid(True, color=GRID_GREY, linewidth=0.8, zorder=0)
    ax.xaxis.grid(True, color=GRID_GREY, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=2, colors="black")
    ax.legend(loc="lower right", frameon=False, fontsize=13,
              handlelength=1.6, handleheight=1.0)

axes[0].set_ylabel("Accuracy (%)", color="black", labelpad=6, fontsize=16)

plt.tight_layout()
plt.savefig("plots/budget.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.savefig("plots/budget.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved to plots/budget.png and plots/budget.pdf")
