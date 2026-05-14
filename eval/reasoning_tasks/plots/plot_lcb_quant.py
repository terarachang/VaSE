import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.path import Path
from matplotlib.patches import PathPatch


CORR_DIR = "corr_se_range"
VCACHE_ROOT = "vcaches"
CONFIGS = ["b2g32", "b2g64", "b4g32", "b4g64"]
LCB_JSON = "tables/all_Qwen3-4B.json"
OUT_STEM = "plots/livecodebench_quant"

EVICT_COLOR     = "#A8C8E8"
EVICT_OUR_COLOR = "#1F4F66"
SELECT_COLOR    = "#F0D27A"
FULL_COLOR      = "#7F7F7F"

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


def plot_corr(ax):
    layers = sorted(
        [n for n in os.listdir(VCACHE_ROOT) if n.startswith("L")],
        key=lambda s: int(s[1:]),
    )

    rows = []
    for cfg in CONFIGS:
        corrs = torch.load(os.path.join(CORR_DIR, f"{cfg}.pt"), weights_only=False)
        assert len(corrs) == len(layers), \
            f"{cfg}: got {len(corrs)} corrs but {len(layers)} layers"
        for layer, c in zip(layers, corrs.tolist()):
            rows.append({"layer": layer, "corr": c, "config": cfg})

    df = pd.DataFrame(rows)

    avg_corr = df.groupby("config")["corr"].mean().to_dict()
    label_map = {cfg: f"{cfg} (Avg. {avg_corr[cfg]:.2f})" for cfg in CONFIGS}
    df["config"] = df["config"].map(label_map)
    hue_order = [label_map[cfg] for cfg in CONFIGS]

    sns.lineplot(ax=ax, data=df, x="layer", y="corr", hue="config",
                 hue_order=hue_order, marker="o", sort=False, linewidth=3)
    ax.set_title("Corr(Range, Quantization Error)",
                 fontsize=24, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=22, fontweight="bold")
    ax.set_ylabel("Correlation", fontsize=22, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelsize=15, rotation=45)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(title=None, borderpad=1.0,
              labelspacing=0.8, handlelength=2.6, handleheight=1.4,
              prop={"weight": "bold", "size": 20})


def plot_livecodebench(ax):
    with open(LCB_JSON) as f:
        data = json.load(f)

    entries = {e["Method"]: e["Acc"] for e in data["LiveCodeBench"]}
    methods = [m for m in METHOD_ORDER if m in entries]
    labels  = [DISPLAY_NAMES[m] for m in methods]
    accs    = [entries[m] for m in methods]
    colors  = [METHOD_COLORS[m] for m in methods]
    full_acc = entries["Full"]

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
    ax.set_ylabel("Pass@1", fontsize=22, fontweight="bold")
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
    y_lo, y_hi = max(ymin, 0), ymax
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(x_positions[0] - 0.6, x_positions[-1] + 0.6)

    legend_handles = [
        mpatches.Patch(color=EVICT_COLOR,     label="Eviction"),
        mpatches.Patch(color=EVICT_OUR_COLOR, label="Eviction (Ours)"),
        mpatches.Patch(color=SELECT_COLOR,    label="Selection"),
        mlines.Line2D([], [], color=FULL_COLOR, linestyle="--", linewidth=3,
                      label="Full KV"),
    ]
    dash_frac = (full_acc - y_lo) / (y_hi - y_lo)
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, dash_frac - 0.01), fontsize=15,
              ncol=4, frameon=True, handlelength=1.8, handleheight=1.6,
              columnspacing=1.2, handletextpad=0.5, borderpad=0.8,
              prop={"weight": "bold", "size": 15})


def main():
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
    })

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1, 1]}
    )

    plot_livecodebench(ax_left)
    plot_corr(ax_right)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = f"{OUT_STEM}.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
