import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

with open("tables/ablation.json") as f:
    data = json.load(f)

df = pd.DataFrame(list(data.items()), columns=["Method", "Accuracy"])

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=df, x="Method", y="Accuracy", ax=ax, palette="muted")

ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.set_title("GSM8k", fontsize=15, fontweight="bold")
ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.tick_params(axis="both", labelsize=13)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

for bar, val in zip(ax.patches, df["Accuracy"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("plots/ablation.png", dpi=150)
plt.show()
