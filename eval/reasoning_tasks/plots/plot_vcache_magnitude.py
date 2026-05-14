import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
mode = sys.argv[1]
assert mode in ['l2', 'range']

n_examples = 200
layers = [int(x) for x in np.linspace(0, 31, 8)]  # 8 layers from 0 to 31

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, l in enumerate(tqdm(layers)):
    magnitudes = []
    lengths = []
    for i in range(n_examples):
        vcache = torch.load(f'vcache/L{l}/ex-{i}.pt', weights_only=False)
        if mode == 'l2':
            magnitude = vcache.norm(dim=-1)
        else:
            magnitude = vcache.amax(-1) - vcache.amin(-1)
        lengths.append(magnitude.size(-1))
        magnitudes.append(magnitude.flatten())
        del vcache
    print(f'Layer {l} Avg lengths:', np.array(lengths).mean())

    mag_tensor = torch.cat(magnitudes)
    torch.save(mag_tensor, f'vcache/L{l}/magnitudes_{mode}.pt')

    sns.violinplot(y=mag_tensor.tolist(), ax=axes[idx], inner=None)
    sns.boxplot(y=mag_tensor.tolist(), ax=axes[idx], width=0.1, fliersize=3, whis=2.5, boxprops=dict(alpha=0.5))
    axes[idx].set_title(f'Layer {l}', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('')
    del magnitudes, lengths

if mode == 'l2':
    fig.supylabel(r'$\|Value\|_2$', fontsize=16, fontweight='bold')
else:
    fig.supylabel('Value Range', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'plots/magnitudes_violin-{mode}.png', dpi=150)
