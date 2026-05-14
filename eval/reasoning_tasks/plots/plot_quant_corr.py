import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


CORR_DIR = "corr_se_range"
VCACHE_ROOT = "vcaches"
CONFIGS = ["b2g32", "b2g64", "b4g32", "b4g64"]
OUT_PATH = "plots/quant_corr.png"


def main():
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

    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })
    plt.figure(figsize=(6.5, 5))
    sns.lineplot(data=df, x="layer", y="corr", hue="config",
                 marker="o", sort=False)
    plt.xlabel("Layer", fontweight="bold")
    plt.ylabel("Cor(Range, Quantization Err)", fontweight="bold")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.gca().set_axisbelow(True)
    plt.legend(title=None, fontsize=14, borderpad=1.0,
               labelspacing=0.8, handlelength=2.5)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
