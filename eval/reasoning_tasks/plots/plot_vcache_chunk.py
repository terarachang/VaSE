import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import sys

mode = sys.argv[1]
assert mode in ['l2', 'range']

n_examples = 200
max_length = 4096
n_chunks = 16
n_sinks = 4
chunk_length = (max_length - n_sinks) // n_chunks
start_ends = [(0, n_sinks)]
starts = [int(x) for x in np.linspace(n_sinks, max_length, n_chunks)]
for i in range(len(starts)-1):
    start_ends.append((starts[i], starts[i+1]))

layers = [int(x) for x in np.linspace(0, 31, 4)]  # 4 layers from 0 to 31

fig, axes = plt.subplots(2, 2, figsize=(20, 10))
axes = axes.flatten()

for idx, l in enumerate(tqdm(layers)):
    chunkidx_to_magnitude = defaultdict(list)
    for i in range(n_examples):
        vcache = torch.load(f'vcache/L{l}/ex-{i}.pt', weights_only=False)
        if mode == 'l2':
            magnitude = vcache.norm(dim=-1)
        else:
            magnitude = vcache.amax(-1) - vcache.amin(-1)
        seq_len = magnitude.size(-1)
        for chunk_i, (start, end) in enumerate(start_ends):
            chunkidx_to_magnitude[chunk_i].append(magnitude[..., start:end].flatten())
        del vcache

    data = []
    for chunk_i in range(n_chunks):
        chunks = chunkidx_to_magnitude[chunk_i]
        if len(chunks) > 0:
            data.append(torch.cat(chunks).tolist())
            #print(start_ends[chunk_i], len(data[-1]))
    n_valid_chunks = len(data)

    sns.boxplot(data=data, ax=axes[idx], fliersize=2, whis=1.5, width=0.6)
    axes[idx].set_title(f'Layer {l}', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('')
    axes[idx].set_xticks(range(n_valid_chunks))
    axes[idx].set_xticklabels(start_ends[:n_valid_chunks], rotation=45)

if mode == 'l2':
    fig.supylabel(r'$\|Value\|_2$', fontsize=16, fontweight='bold')
else:
    fig.supylabel('Value Range', fontsize=16, fontweight='bold')
fig.supxlabel('Token Positions', fontsize=16, fontweight='bold')

plt.tight_layout()
fig.subplots_adjust(left=0.05)
plt.savefig(f'plots/magnitudes_chunk-{mode}.png', dpi=150)
print(f'Saved to plots/magnitudes_chunk-{mode}.png')
