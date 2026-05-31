"""
Peak-memory barplots of Qwen3-14B across KV cache budgets.

Layout: 1 x 3 panels (one column per budget: 2048 / 4096 / 6144).
  - 16K  (output_len 16384, tables/throughput_Qwen3-14B_16384.json)
Each panel is a memory barplot styled like the third panel of plot_thpt.py:
a hatched "model weights" base + a rounded method bar on top.

The row shares its y-axis scale.
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch

os.makedirs("plots", exist_ok=True)


def rounded_vbar(ax, x, height, width, ry, y0=0.0, **kw):
    """Vertical bar with rounded top corners, based at y0 (matches plot_thpt.py)."""
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

# ── Typography (match plot_thpt.py) ──────────────────────────────────────────
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
ROWS = [
    {"task": "16K", "path": "tables/throughput_Qwen3-14B_16384.json", "oom": False},
]

DEVICE_MEM = 80.0  # GB; the dense cache OOMs at 32K, so its bar is capped here.


def load_memory(path):
    """Return {budget: {method: peak_memory_GB}} for every budget cell."""
    with open(path) as f:
        raw = json.load(f)
    budgets = sorted(int(b) for b in raw)
    out = {}
    for b in budgets:
        out[b] = {row["Method"]: row["Memory"] for row in raw[str(b)]}
    return budgets, out


ROW_DATA = []
BUDGETS = None
for r in ROWS:
    budgets, mem = load_memory(r["path"])
    if BUDGETS is None:
        BUDGETS = budgets
    ROW_DATA.append(mem)

# ── Palette (consistent with plot_thpt.py) ───────────────────────────────────
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
METHOD_ORDER = ["RKV", "Sample Attn + V", "CUR Resample"]

FULL_BAR_COLOR = "#F0D27A"  # yellow (matches plot_thpt.py SELECT_COLOR)
GRID_GREY = "#E7E7E7"
TEXT_DARK = "#222222"
MODEL_WEIGHTS = 29.5  # GB, shared base under every bar (part of the total)

bar_colors = {**COLORS, "Full": FULL_BAR_COLOR}
BAR_W = 0.72

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.4), facecolor="white")

for ri, (r, mem) in enumerate(zip(ROWS, ROW_DATA)):
    oom = r["oom"]

    # Shared y-range for the whole row.
    row_vals = []
    for b in BUDGETS:
        row_vals += [mem[b][m] for m in METHOD_ORDER if m in mem[b]]
        if "Full" in mem[b]:
            row_vals.append(mem[b]["Full"])
    if oom:
        row_vals.append(DEVICE_MEM)
    row_hi = max(row_vals)
    ROW_YLIM = (20, row_hi * 1.15)

    for ci, b in enumerate(BUDGETS):
        ax = axes[ci]
        cell = mem[b]

        bar_order = [m for m in METHOD_ORDER if m in cell] + ["Full"]
        xpos = list(range(len(bar_order)))

        for x, m in zip(xpos, bar_order):
            if m == "Full":
                # 16K: real dense-cache memory.  32K: OOM, capped at device limit.
                total = cell["Full"] if (not oom and "Full" in cell) else DEVICE_MEM
            else:
                total = cell[m]

            # Hatched "Model Weights" base, flat-topped so the method bar sits flush.
            ax.bar(x, MODEL_WEIGHTS, width=BAR_W, facecolor="#D9D9D9",
                   edgecolor="#B0B0B0", hatch="///", linewidth=0, zorder=3)
            # Remaining memory fills the rest of the total height.
            rounded_vbar(ax, x, total - MODEL_WEIGHTS, BAR_W, ry=1.5,
                         y0=MODEL_WEIGHTS, facecolor=bar_colors[m],
                         edgecolor="none", zorder=4)

            if m == "Full" and oom:
                ax.text(x, total + 0.6, "OOM", ha="center", va="bottom",
                        fontsize=12, fontweight="bold", color=TEXT_DARK)
            else:
                ax.text(x, total + 0.6, f"{total:.1f}", ha="center", va="bottom",
                        fontsize=12, fontweight="bold", color=TEXT_DARK)

        ax.set_title(f"{r['task']} / {b} Budget", color=TEXT_DARK, pad=8,
                     fontweight='bold')
        ax.set_xticks(xpos)
        ax.set_xticklabels([DISPLAY_NAMES[m] for m in bar_order], fontweight='bold')
        ax.set_xlim(xpos[0] - 0.6, xpos[-1] + 0.6)
        ax.set_ylim(*ROW_YLIM)

        if ci == 0:
            ax.set_ylabel("Memory (GB)", color="black", labelpad=6, fontsize=16)

        # Legend: model-weights base on every panel; flag the OOM Full bar too.
        handles = [mpatches.Patch(facecolor="#D9D9D9", edgecolor="#B0B0B0",
                                  hatch="///", linewidth=0,
                                  label="14B Model Weights")]
        if oom:
            handles.append(mpatches.Patch(facecolor=FULL_BAR_COLOR,
                                          edgecolor="none", label="Full (OOM)"))
        ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=13,
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
plt.savefig("plots/memory.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.savefig("plots/memory.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved to plots/memory.png and plots/memory.pdf")
