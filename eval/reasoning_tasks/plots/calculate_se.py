import glob
import numpy as np


task_name_to_budget = {
    'aime25': 4096,
    'gpqa': 2048,
    'math': 1024,
    'aime26': 4096,
    'hmmt': 4096,
}
task_display_name = {
    'aime25': 'AIME25',
    'gpqa': 'GPQA-D',
    'math': 'MATH',
    'aime26': 'AIME26',
    'hmmt': 'HMMT25',
}
methods = [
    'attn_rkv',
    'cur_fixed_gauss',
    'cur_resample_gauss',
    'small_range_sink_sample_attn',
]
method_display_name = {
    'attn_rkv': 'R-KV',
    'cur_fixed_gauss': 'CurDKV',
    'cur_resample_gauss': 'VASE-DKV',
    'small_range_sink_sample_attn': 'VASE-AttnV',
}


def calculate_se(scores: np.ndarray) -> tuple[float, float]:
    """
    scores: shape (n_examples, n_rollouts), values are 0/1 accuracy
    Returns (mean, standard_error)
    """
    per_example_acc = scores.mean(axis=1)
    mean_acc = per_example_acc.mean()
    se = per_example_acc.std(ddof=1) / np.sqrt(len(per_example_acc))
    return mean_acc, se


def find_npy(task, method, budget):
    pattern = f"existing_results/{task}/Qwen3-14B/{method}/*budget={budget}*/total_is_correct_arr.npy"
    matches = glob.glob(pattern)
    return matches[0] if matches else None


# Collect mean±se for each (task, method)
results = {}
for task, budget in task_name_to_budget.items():
    for method in methods:
        path = find_npy(task, method, budget)
        if path:
            arr = np.load(path)  # shape: (n_runs, n_examples)
            mean_acc, se = calculate_se(arr.T)  # transpose to (n_examples, n_runs)
            results[(task, method)] = f"{mean_acc*100:.1f} ± {se*100:.1f}"
        else:
            results[(task, method)] = ""

# Print table (rows=methods, columns=tasks)
tasks = list(task_name_to_budget.keys())
display_names = [task_display_name[t] for t in tasks]
col_w = max(max(len(results[(t, m)]) for t in tasks for m in methods), max(len(n) for n in display_names)) + 2
method_w = max(len(method_display_name[m]) for m in methods) + 2

header = f"{'method':<{method_w}}" + "".join(f"{task_display_name[t]:<{col_w}}" for t in tasks)
print(header)
print("-" * len(header))
for method in methods:
    row = f"{method_display_name[method]:<{method_w}}" + "".join(f"{results[(t, method)]:<{col_w}}" for t in tasks)
    print(row)
