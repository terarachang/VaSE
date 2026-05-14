import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.path import Path
from matplotlib.patches import PathPatch

with open("tables/all_Qwen3-4B.json") as f:
    data = json.load(f)

EVICT_COLOR     = "#A8C8E8"   # soft blue
EVICT_OUR_COLOR = "#1F4F66"   # stronger blue for our eviction recipes
SELECT_COLOR    = "#F0D27A"   # soft yellow
FULL_COLOR      = "#7F7F7F"   # grey for Full KV dashed line

DISPLAY_NAMES = {
    "RKV": "R-KV",
    "CUR Fixed": "CurDKV",
    "CUR Resample": "VASE-DKV",
    "Sample Attn + V": "VASE-AttnV",
    "SeerR": "SeerAttention-R",
}

METHOD_ORDER = ["RKV", "CUR Fixed", "CUR Resample", "Sample Attn + V", "SeerR"]
METHOD_COLORS = {
    "RKV": EVICT_COLOR,
    "CUR Fixed": EVICT_COLOR,
    "CUR Resample": EVICT_OUR_COLOR,
    "Sample Attn + V": EVICT_OUR_COLOR,
    "SeerR": SELECT_COLOR,
}

entries = {e["Method"]: e["Acc"] for e in data["LiveCodeBench"]}
methods = [m for m in METHOD_ORDER if m in entries]
labels  = [DISPLAY_NAMES[m] for m in methods]
accs    = [entries[m] for m in methods]
colors  = [METHOD_COLORS[m] for m in methods]
full_acc = entries["Full"]


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
        (right, height),
        (right - rx, height),
        (left  + rx, height),
        (left,  height),
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


fig, ax = plt.subplots(figsize=(10, 7))

x_positions = np.arange(len(labels), dtype=float)
BAR_W = 0.72
ROUND_RY = 1.5
for x, val, col in zip(x_positions, accs, colors):
    rounded_vbar(ax, x, val, BAR_W, ry=ROUND_RY,
                 facecolor=col, edgecolor="none")

for x, val in zip(x_positions, accs):
    ax.text(x, val + 0.8, f"{val:.1f}",
            ha="center", va="bottom",
            fontsize=18, fontweight="bold")

ax.axhline(full_acc, color=FULL_COLOR, linestyle="--", linewidth=3)

ax.set_title("LiveCodeBench", fontsize=24, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=22, fontweight="bold")
ax.set_xlabel("")
ax.set_xticks(x_positions)
ax.set_xticklabels(labels)
ax.tick_params(axis="x", labelsize=17, rotation=10)
ax.tick_params(axis="y", labelsize=15)
for lbl in ax.get_xticklabels():
    lbl.set_fontweight("bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)

ymin = min(min(accs), full_acc) - 20
ymax = max(max(accs), full_acc) + 5
ax.set_ylim(max(ymin, 0), ymax)
ax.set_xlim(x_positions[0] - 0.6, x_positions[-1] + 0.6)

legend_handles = [
    mpatches.Patch(color=EVICT_COLOR,     label="Eviction"),
    mpatches.Patch(color=EVICT_OUR_COLOR, label="Eviction (Ours)"),
    mpatches.Patch(color=SELECT_COLOR,    label="Selection"),
    mlines.Line2D([], [], color=FULL_COLOR, linestyle="--", linewidth=3, label="Full KV"),
]
ax.legend(handles=legend_handles, loc="upper center",
          bbox_to_anchor=(0.5, 0.90), fontsize=17,
          ncol=4, frameon=True, handlelength=1.6, handleheight=1.5,
          columnspacing=1.2, handletextpad=0.5)

plt.tight_layout()
plt.savefig("plots/livecodebench.png", dpi=300)
