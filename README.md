# VaSE: Value-aware Stochastic KV Cache Eviction

## Installation

```bash
conda create -yn vase python=3.11
conda activate vase
pip install torch==2.4.0
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # we've tested on 2.7.3 and 2.8.3; don't use 2.7.4

cd eval/reasoning_tasks
```

## Eval Different Methods

```bash
bash myscripts/{task_name}/eval_{method_name}.sh
```

Note: in the `.sh` files, you can change `num_gpus` for data parallelism.

## Download LiveCodeBench

The following script downloads per-problem test cases. We tested on medium examples of LiveCodeBench-v6 (383 in total).

```bash
pip install "datasets<4"
python data/livecodebench/download_tests.py  # test data lands in data/livecodebench/livecodebench_tests
```

## Show Results

```bash
python aggregate_results.py {task_name}/{model_name}
```

See [tables/](eval/reasoning_tasks/tables/) for our results reported in the paper.

## Benchmark Throughput & Memory

Measure decode-phase throughput (tokens/s) and peak GPU memory for each KV-cache method on a single gpu:

```bash
bash benchmark/run_thpt.sh
```

This sweeps token budgets (2048/4096/6144) and output lengths (16384/32768) across different methods.

## Understand the Repo

- [eval/reasoning_tasks/repo_walkthrough.md](eval/reasoning_tasks/repo_walkthrough.md)
- This repo is modified from: https://github.com/microsoft/SeerAttention/tree/main/eval/reasoning_tasks

