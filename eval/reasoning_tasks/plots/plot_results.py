import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

with open("tables/all.json") as f:
    data = json.load(f)

def make_plot():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    DEFAULT_COLOR = "#A8C8E8"   # soft blue
    CUR_COLOR     = "#5A9EC9"   # stronger blue for CUR methods
    SEER_R_COLOR  = "#F4A896"   # soft coral
    FULL_COLOR    = "#A8D8A8"   # soft green

    tasks = ["AIME25", "GPQA-Diamond", "MATH", "AIME26", "HMMT25"]

    for i, task in enumerate(tasks):
        ax = axes[i // 3][i % 3]
        df = pd.DataFrame(data[task])
        df["Method"] = df["Method"].replace("Sample Attn + V", "Attn + V")
        palette = [
            SEER_R_COLOR if m == "SeerR" else FULL_COLOR if m == "Full" else CUR_COLOR if m in ("CUR Resample", "Attn + V") else DEFAULT_COLOR
            for m in df["Method"]
        ]
        sns.barplot(data=df, x="Method", y="Acc", ax=ax, palette=palette)
        ax.set_title(task, fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy (%)" if i % 3 == 0 else "", fontsize=16, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        margin = (df["Acc"].max() - df["Acc"].min()) * 0.5
        ax.set_ylim(df["Acc"].min() - margin, df["Acc"].max() + margin * 0.1)

    ax_legend = axes[1][2]
    ax_legend.axis('off')
    legend_handles = [
        mpatches.Patch(color=DEFAULT_COLOR, label='Eviction'),
        mpatches.Patch(color=CUR_COLOR,     label='Eviction Recipe'),
        mpatches.Patch(color=SEER_R_COLOR,  label='Selection'),
        mpatches.Patch(color=FULL_COLOR,    label='Full KV'),
    ]
    ax_legend.legend(handles=legend_handles, loc='center', fontsize=18,
                     frameon=True, title='Method', title_fontsize=20,
                     handlelength=3, handleheight=2.5)

    plt.tight_layout()
    plt.savefig("plots/all.png", dpi=300)
    #plt.show()

def make_plot_with_avg():
    from collections import defaultdict

    DEFAULT_COLOR = "#A8C8E8"
    CUR_COLOR     = "#5A9EC9"
    SEER_R_COLOR  = "#F4A896"
    FULL_COLOR    = "#A8D8A8"

    def method_color(m):
        if m == "SeerR":   return SEER_R_COLOR
        if m == "Full":    return FULL_COLOR
        if m in ("CUR Resample", "Attn + V"): return CUR_COLOR
        return DEFAULT_COLOR

    tasks = ["AIME25", "GPQA-Diamond", "MATH", "AIME26", "HMMT25"]
    METHOD_ORDER = ["RKV", "CUR", "CUR Resample", "Attn + V", "SeerR", "Full"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, task in enumerate(tasks):
        ax = axes[i // 3][i % 3]
        df = pd.DataFrame(data[task])
        df["Method"] = df["Method"].replace("Sample Attn + V", "Attn + V")
        palette = [method_color(m) for m in df["Method"]]
        sns.barplot(data=df, x="Method", y="Acc", ax=ax, palette=palette)
        ax.set_title(task, fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy (%)" if i % 3 == 0 else "", fontsize=16, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        margin = (df["Acc"].max() - df["Acc"].min()) * 0.5
        ax.set_ylim(df["Acc"].min() - margin, df["Acc"].max() + margin * 0.1)

    # Average accuracy in bottom-right
    acc_sums = defaultdict(list)
    for task in tasks:
        for entry in data[task]:
            m = entry["Method"].replace("Sample Attn + V", "Attn + V")
            if m != "Attn":
                acc_sums[m].append(entry["Acc"])

    methods = [m for m in METHOD_ORDER if m in acc_sums]
    avg_accs = [sum(acc_sums[m]) / len(acc_sums[m]) for m in methods]
    colors   = [method_color(m) for m in methods]

    ax_avg = axes[1][2]
    bars = ax_avg.bar(methods, avg_accs, color=colors)
    for bar, val in zip(bars, avg_accs):
        ax_avg.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(avg_accs) - min(avg_accs)) * 0.02,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax_avg.set_title("Average", fontsize=16, fontweight="bold")
    ax_avg.set_xlabel("")
    ax_avg.set_ylabel("")
    ax_avg.tick_params(axis="x", rotation=30, labelsize=13)
    ax_avg.tick_params(axis="y", labelsize=13)
    ax_avg.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax_avg.set_axisbelow(True)
    margin = (max(avg_accs) - min(avg_accs)) * 0.5
    ax_avg.set_ylim(min(avg_accs) - margin, max(avg_accs) + margin * 0.4)

    plt.tight_layout()
    plt.savefig("plots/all_with_avg.png", dpi=300)


def make_table():
    from collections import defaultdict
    method_order = ["RKV", "CUR", "CUR Resample", "Attn + V", "SeerR", "Full"]
    acc_sums = defaultdict(list)
    for task_data in data.values():
        for entry in task_data:
            method = entry.get("Method")
            method = method.replace("Sample Attn + V", "Attn + V")
            if method != "Attn":
                acc_sums[method].append(entry["Acc"])
    print("\n")

    print(f"{'Method':<20} {'Avg Acc':>8}")
    for method, accs in acc_sums.items():
        print(f"{method:<20} {sum(accs)/len(accs):>8.2f}")

make_plot()
make_plot_with_avg()
make_table()