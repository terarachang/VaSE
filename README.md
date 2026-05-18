# VaSE: Value-aware Stochastic KV Cache Eviction

## Installation

```bash
conda create -yn seer python=3.11
conda activate seer
pip install torch==2.4.0
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # we use 2.7.3
```

## Run Experiments

```bash
cd eval/reasoning_tasks
bash myscripts/<TASK>/eval_cur_fixed_gauss.sh
bash myscripts/<TASK>/eval_cur_resample_gauss.sh
```

Note: in the `.sh` files, you can change `num_gpus` for data parallelism.

## Run LiveCodeBench

Download LiveCodeBench-v6; we only tested on medium examples (383 in total).

```bash
pip install "datasets<4"
python data/livecodebench/download_tests.py  # test data lands in data/livecodebench/livecodebench_tests
```

## Show Results

```bash
python aggregate_results.py {task_name}/{model_name}
```

## Understand the Repo

- eval/reasoning_tasks/repo_walkthrough.md
- This repo is modified from: https://github.com/microsoft/SeerAttention/tree/main/eval/reasoning_tasks

