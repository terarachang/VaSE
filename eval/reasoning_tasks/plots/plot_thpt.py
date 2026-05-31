"""
Line plots of decode throughput vs. KV cache budget on Qwen3-14B.

Layout: 1 x 2 panels.
  - Left  : 16K  (output_len 16384)
  - Right : 32K  (output_len 32768)
Each panel: method lines (RKV, CUR Resample, Sample Attn + V).
"Full" rows are ignored for now.
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch

os.makedirs("plots", exist_ok=True)


def rounded_vbar(ax, x, height, width, ry, y0=0.0, **kw):
    """Vertical bar with rounded top corners, based at y0 (matches plot_lcb_quant.py)."""
    if height <= 0:
        return
    rx = width * 0.10
    ry_ = min(ry, height / 2)
    top = y0 + height
    left, right = x - width / 2, x + width / 2
    verts = [
        (left,  y0),
        (right, y0),
        (right, top - ry_),
        (right, top),
        (right - rx, top),
        (left  + rx, top),
        (left,  top),
        (left,  top - ry_),
        (left,  y0),
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

# ── Typography (match plot_budget.py) ───────────────────────────────────────
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
# Each file maps a budget (string key) -> list of {"Method", "Throughput", ...}.
TABLE_FILES = {
    "16K": "tables/throughput_Qwen3-14B_16384.json",
    "32K": "tables/throughput_Qwen3-14B_32768.json",
}

def load_task_data(path):
    with open(path) as f:
        raw = json.load(f)
    budgets = sorted(int(b) for b in raw)
    entry = {"Full": None}
    for b in budgets:
        for row in raw[str(b)]:
            method = row["Method"]
            if method == "Full":
                # Single dense-cache reference (read from the 2048 budget cell).
                entry["Full"] = row["Throughput"]
                continue
            entry.setdefault(method, []).append(row["Throughput"])
    return budgets, entry

def load_memory(path, budget):
    """Per-method peak memory (GB) at a single budget cell"""
    with open(path) as f:
        raw = json.load(f)
    return {row["Method"]: row["Memory"] for row in raw[str(budget)]}

DATA = {}
BUDGETS = None
for task, path in TABLE_FILES.items():
    budgets, entry = load_task_data(path)
    if BUDGETS is None:
        BUDGETS = budgets
    DATA[task] = entry

MEM_BUDGET = 4096
MEMORY = load_memory(TABLE_FILES["16K"], MEM_BUDGET)

# ── Palette (consistent with plot_budget.py) ────────────────────────────────
COLORS = {
    "RKV":             "#A87BA0",  # muted purple
    "CUR Resample":    "#5C7C92",  # blue-grey
    "Sample Attn + V": "#1F4F66",  # deep blue
}

DISPLAY_NAMES = {
    "RKV":             "R-KV",
    "CUR Resample":    "VaSE-DKV",
    "Sample Attn + V": "VaSE-AttnV",
    "Full":            "Full",
}
METHOD_ORDER = ["RKV", "CUR Resample", "Sample Attn + V"]

FULL_COLOR = "#242424"
FULL_BAR_COLOR = "#F0D27A"  # yellow (matches plot_lcb_quant.py SELECT_COLOR)
GRID_GREY = "#E7E7E7"
TEXT_DARK = "#222222"

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.2), facecolor="white")
TASKS = ["16K", "32K"]

# Shared y-range for the two throughput panels (16K / 32K).
shared_vals = []
for task in TASKS:
    d = DATA[task]
    shared_vals += [v for m in METHOD_ORDER for v in d[m]]
    if d["Full"] is not None:
        shared_vals.append(d["Full"])
shared_lo, shared_hi = min(shared_vals), max(shared_vals)
shared_pad = (shared_hi - shared_lo) * 0.10
SHARED_YLIM = (shared_lo - shared_pad, shared_hi + shared_pad)

for ax, task in zip(axes, TASKS):
    d = DATA[task]
    for m in METHOD_ORDER:
        ax.plot(BUDGETS, d[m],
                marker="o", markersize=7,
                linewidth=2.6, color=COLORS[m],
                label=DISPLAY_NAMES[m], zorder=3)

    # Full as a horizontal dashed reference. The 32K run OOMs (no data),
    # so it only appears in the legend as "Full (OOM)".
    if d["Full"] is not None:
        ax.axhline(d["Full"], color=FULL_COLOR, linestyle="--",
                   linewidth=2.0, label="Full", zorder=2)
    else:
        ax.plot([], [], color=FULL_COLOR, linestyle="--",
                linewidth=2.0, label="Full (OOM)")

    ax.set_title(f"{task}", color=TEXT_DARK, pad=8, fontweight='bold')
    ax.set_xlabel("KV Cache Budget", fontsize=16)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([f"{b}" for b in BUDGETS], fontweight='bold')

    # Shared y-range across both throughput panels
    ax.set_ylim(*SHARED_YLIM)

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
    ax.legend(loc="best", frameon=True, fontsize=13,
              handlelength=1.6, handleheight=1.0,
              facecolor="white", edgecolor="#bbbbbb", framealpha=0.95)

axes[0].set_ylabel("Throughput (tokens/s)", color="black", labelpad=6, fontsize=16)

# ── Third panel: memory barplot (16K, 4096 budget) ──────────────────────────
ax = axes[2]
bar_order = [m for m in METHOD_ORDER if m in MEMORY]
if "Full" in MEMORY:
    bar_order.append("Full")
bar_colors = {**COLORS, "Full": FULL_BAR_COLOR}
xpos = list(range(len(bar_order)))
BAR_W = 0.72
MODEL_WEIGHTS = 29.5  # GB, shared base under every bar (part of the total)
for x, m in zip(xpos, bar_order):
    total = MEMORY[m]
    # Hatched "Model Weights" base, flat-topped so the method bar sits flush.
    ax.bar(x, MODEL_WEIGHTS, width=BAR_W, facecolor="#D9D9D9",
           edgecolor="#B0B0B0", hatch="///", linewidth=0, zorder=3)
    # Remaining memory fills the rest of the (unchanged) total height.
    rounded_vbar(ax, x, total - MODEL_WEIGHTS, BAR_W, ry=1.5, y0=MODEL_WEIGHTS,
                 facecolor=bar_colors[m], edgecolor="none", zorder=4)
    ax.text(x, total + 0.6, f"{total:.1f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color=TEXT_DARK)

ax.set_title(f"16K / {MEM_BUDGET} Budget", color=TEXT_DARK, pad=8, fontweight='bold')
ax.set_ylabel("Memory (GB)", color="black", labelpad=6, fontsize=16)
ax.set_xticks(xpos)
ax.set_xticklabels([DISPLAY_NAMES[m] for m in bar_order], fontweight='bold')
ax.set_xlim(xpos[0] - 0.6, xpos[-1] + 0.6)
ax.set_ylim(0, max(MEMORY.values()) * 1.15)
ax.legend(handles=[mpatches.Patch(facecolor="#D9D9D9", edgecolor="#B0B0B0",
                                  hatch="///", linewidth=0, label="14B Model Weights")],
          loc="upper left", frameon=True, fontsize=13,
          facecolor="white", edgecolor="#bbbbbb", framealpha=0.95)
ax.yaxis.grid(True, color=GRID_GREY, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#bbbbbb")
ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_color("#bbbbbb")
ax.spines["bottom"].set_linewidth(0.7)
ax.tick_params(axis="x", length=0, pad=4)
ax.tick_params(axis="y", length=0, pad=2, colors="black")

plt.tight_layout()
plt.savefig("plots/thpt.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.savefig("plots/thpt.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved to plots/thpt.png and plots/thpt.pdf")
