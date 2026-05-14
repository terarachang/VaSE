"""
Visualizes the GSM8K pilot study (Qwen3-4B, KV budget = 512, ~4x compression).

Mirrors `tab:pilot_study`:
  - Full / Evict Random / Evict Largest V are drawn as horizontal reference lines.
  - Attn family : base + four "+ V slots" variants (ours) + two "+ Sample" variants.
  - CUR family  : base + "+ Sample G" variant.

The "+V" vs "+Sample" comparison is encoded by color:
  orange = "+V slots" (ours), teal = "+Sample" variants.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch

os.makedirs("plots", exist_ok=True)

# ── Typography ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
    "axes.titlesize": 22,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 15,
})

# ── Palette ──────────────────────────────────────────────────────────────────
# Method colors
NEUTRAL     = "#D6DCE2"   # Base eviction (Attn) — quiet neutral
NEUTRAL_CUR = "#A8C8AE"   # Base eviction (CUR) — muted sage-green (in green family)
# + Value scoring: green gradient (light → dark as # V slots grows)
OURS_1      = "#BFE3B4"   # + 16 V slots
OURS_2      = "#90C99A"   # + 32 V slots
OURS_3      = "#5DA975"   # + 64 V slots
OURS_4      = "#2D7A4B"   # + 256 V slots
OURS        = OURS_3      # legend swatch (representative shade)
SEER        = "#7AB8D8"   # + Sampling — light blue
SEER_BEST   = "#1F4F66"   # + Value scoring + Sampling — deep blue-green blend

# Reference lines
REF_FULL    = "#242424"
REF_RANDOM  = "#707070"
REF_LARGEST = "#8E8E8E"

# Structure
GRID_GREY   = "#E7E7E7"
SEP_GREY    = "#D4D4D4"
TEXT_DARK   = "#222222"
TEXT_MID    = "#555555"

# ── Bars (Attn family + CUR family) ─────────────────────────────────────────
# (label, accuracy, color, group, indent)
BARS = [
    ("Attn",                          64.3, NEUTRAL,    "attn", False),
    ("+ Sample",                      70.9, SEER,       "attn", True),
    ("+ 16 V slots",                  73.3, OURS_1,     "attn", True),
    ("+ 32 V slots",                  76.0, OURS_2,     "attn", True),
    ("+ 64 V slots",                  79.0, OURS_3,     "attn", True),
    ("+ 256 V slots",                 80.5, OURS_4,     "attn", True),
    ("+ Sample + 256 V slots",        85.2, SEER_BEST,  "attn", True),

    ("CUR",                           78.6, NEUTRAL,    "cur", False),
    ("+ Sample $\\mathcal{G}$",       87.6, SEER_BEST,  "cur",  True),
]

n      = len(BARS)
labels = [r[0] for r in BARS]
accs   = [r[1] for r in BARS]
colors = [r[2] for r in BARS]
groups = [r[3] for r in BARS]
indent = [r[4] for r in BARS]

GROUP_GAP    = 0.6   # extra spacing between Attn family and CUR family
SUBGROUP_GAP = 0.35  # extra spacing inside Attn family (entering V slots / entering Sample+V)
x_positions = np.zeros(n, dtype=float)
_boost = 0.0
for _i in range(n):
    if _i > 0:
        if groups[_i] != groups[_i - 1]:
            _boost += GROUP_GAP
        elif ("V slots" in labels[_i]) and ("V slots" not in labels[_i - 1]):
            _boost += SUBGROUP_GAP
        elif ("Sample" in labels[_i]) and ("V slots" in labels[_i]) \
                and ("Sample" not in labels[_i - 1]):
            _boost += SUBGROUP_GAP
    x_positions[_i] = _i + _boost


# ── Rounded vertical bar (rounded only on the top) ──────────────────────────
def rounded_vbar(ax, x, height, width, ry, **kw):
    if height <= 0:
        return
    rx = width * 0.10
    ry_ = min(ry, height / 2)
    left, right = x - width / 2, x + width / 2
    verts = [
        (left,  0.0),
        (right, 0.0),
        (right, height - ry_),
        (right, height),               # ctrl
        (right - rx, height),
        (left  + rx, height),
        (left,  height),               # ctrl
        (left,  height - ry_),
        (left,  0.0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(Path(verts, codes), **kw))


# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4.2), facecolor="white")

BAR_W = 0.72
ROUND_RY = 1.0
for x, val, col in zip(x_positions, accs, colors):
    rounded_vbar(ax, x, val, BAR_W, ry=ROUND_RY,
                 facecolor=col, edgecolor="none")

# value labels above each bar
for x, val in zip(x_positions, accs):
    ax.text(x, val + 1.0, f"{val:.1f}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT_DARK)

xlim_lo  = -0.6
xlim_hi  = x_positions[-1] + 0.6

# ── x-axis tick labels ──────────────────────────────────────────────────────
X_LABELS = ["SnapKV", "+Sampling", "+V16", "+V32", "+V64", "+V256",
            "+Sampling +V256", "CurDKV", "+Sampling"]
ax.set_xticks(x_positions)
ax.set_xticklabels(X_LABELS)

# ── y-axis cosmetics ────────────────────────────────────────────────────────
ymin = 50
ymax = 95
ax.set_ylim(ymin, ymax)
ax.set_yticks(range(ymin, ymax + 1, 10))
ax.set_xlim(xlim_lo, xlim_hi)
ax.yaxis.grid(True, color=GRID_GREY, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#bbbbbb")
ax.spines["left"].set_linewidth(0.7)
ax.tick_params(axis="x", length=0, pad=3, labelsize=15)
ax.tick_params(axis="y", length=0, labelsize=10.5, colors="black", pad=2)
ax.set_ylabel("Accuracy (%)", fontsize=15, color="black", labelpad=8, fontweight="bold")

# section separator between Attn family and CUR family
for i in range(1, n):
    if groups[i] != groups[i - 1]:
        x_bound = (x_positions[i] + x_positions[i - 1]) / 2
        ax.plot([x_bound, x_bound], [ymin, ymax],
                color=SEP_GREY, linewidth=0.6, linestyle="-",
                alpha=0.9, zorder=0)

# ── Legend ──────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=NEUTRAL,    label="Baseline"),
    mpatches.Patch(facecolor=OURS,       label="+ Value scoring"),
    mpatches.Patch(facecolor=SEER,       label="+ Sampling"),
    mpatches.Patch(facecolor=SEER_BEST,  label="+ Value scoring + Sampling"),
]
leg = ax.legend(handles=legend_handles, loc="lower center",
          bbox_to_anchor=(0.5, 1.02),
          fontsize=15, frameon=False, ncol=4,
          handlelength=1.3, handleheight=1.0,
          columnspacing=1.6, borderpad=0.2)
for text in leg.get_texts():
    text.set_fontweight("bold")

plt.tight_layout()
out = "plots/pilot_study.png"
plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved to {out}")
