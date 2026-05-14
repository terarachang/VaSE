import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from matplotlib.patches import PathPatch


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

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "Arial", "DejaVu Sans"]

fig = plt.figure(figsize=(18, 5))
gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 0.4, 1.3], wspace=0.05)
axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])]

# Panels 1 & 2: violin plots for L13 and L22
for i, (ax, l) in enumerate(zip(axes[:2], [13, 22])):
    mag_tensor = torch.load(f'vcaches/L{l}/magnitudes_range.pt', weights_only=True)
    sns.violinplot(y=mag_tensor.tolist(), ax=ax, inner=None)
    sns.boxplot(y=mag_tensor.tolist(), ax=ax, width=0.1, fliersize=3, whis=2.5, boxprops=dict(alpha=0.5))
    ax.set_title(f'Layer {l}', fontsize=15, fontweight='bold', pad=10)
    ax.xaxis.grid(False)
    if i == 0:
        ax.set_ylabel('Value-State Range', fontsize=15, fontweight='bold')
    else:
        ax.set_ylabel('')
        ax.yaxis.tick_right()

# Panel 3: bar plot
categories = ['Full', 'Evict Random', 'Evict Large V']
accuracies = [88.0, 53.2, 14.3]
colors = ['#78bd76', '#dd8e61', '#4C72B0']
ax = axes[2]
x_positions = np.arange(len(categories), dtype=float)
BAR_W = 0.72
ROUND_RY = 2.0
for x, val, col in zip(x_positions, accuracies, colors):
    rounded_vbar(ax, x, val, BAR_W, ry=ROUND_RY,
                 facecolor=col, edgecolor="none")
for x, val in zip(x_positions, accuracies):
    ax.annotate(f'{val:.1f}',
                (x, val),
                ha='center', va='center',
                fontsize=14, color='black',
                fontweight='bold',
                xytext=(0, 9),
                textcoords='offset points')
ax.set_title('GSM8k', fontsize=15, fontweight='bold', pad=10)
ax.set_xlabel('')
ax.set_ylabel('Accuracy (%)', fontsize=15, fontweight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels(categories)
ax.set_xlim(x_positions[0] - 0.6, x_positions[-1] + 0.6)
ax.xaxis.grid(False)
ax.tick_params(axis='both', labelsize=13)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
ax.set_ylim(0, 100)

plt.tight_layout(rect=[0, 0, 1, 1], pad=1.2)

# Vertical separator line in the middle of the spacer column (gs[2])
spacer = fig.add_subplot(gs[2])
spacer.axis('off')

pdf_path = 'plots/violin_evict.pdf'
png_path = 'plots/violin_evict.png'
plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, bbox_inches='tight')
print(pdf_path)
print(png_path)
