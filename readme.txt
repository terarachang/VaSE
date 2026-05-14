# Installation
conda create -yn seer python=3.11
conda activate seer
pip install torch==2.4.0
pip install -r requirements.txt
pip install -e .

# Run Experiments
git checkout encode
cd eval/reasoning_tasks
bash myscripts/TASK/eval_cur_fixed_gauss.sh and eval_cur_resample_gauss.sh
Note: in the .sh files, set model_dir="q4" and num_gpus=1 for quick starts

# Run Livecodebench
## Download livecodebench-v6; I only tested on medium examples (383 in total)
pip install "datasets<4"
python data/livecodebench/download_tests.py # test data in data/livecodebench/livecodebench_tests

## Unzip the model outputs
unzip livecodebench-dense.zip
unzip livecodebench.zip

## Run eval
- Find a save machine first. Ideally with good cpus. No gpus needed.
- python summary_results.py --data_name livecodebench --total_run 8 --output_dir livecodebench/Qwen3-4B/dense/livecodebench_bs8_threshold_T0_start0_blocksize64_eager # change the --output_dir accroding to different methods
- If the job finishes successfully, you'll see a overall_summary.txt under --output_dir
- If you kill the job before it's finished, you need to delete the cache*.jsonl files under the run_i folders otherwise it may cause error when re-running the script
- I can finish running the jobs but they get stuck in the middle frequently. Not sure why.
- This is the offical repo: https://github.com/livecodebench/livecodebench
- After running the eval, you can run python aggregate_results.py livecodebench to print the table


# Understand Repo
https://github.com/terarachang/SeerAttention/blob/encode/eval/reasoning_tasks/cur.md
https://github.com/terarachang/SeerAttention/blob/encode/eval/reasoning_tasks/repo_walkthrough.md

# Show Results
- python print_table.py tables/all_Qwen3-4B.json         # all tasks
- python aggregate_results.py {task_name}/{model_name}   # single-out
