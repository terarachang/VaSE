#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse
import time
from collections import deque # Use deque for efficient pop/append
from my_utils import *


def choose_task_config(model_size):
    if model_size != "32B":
        task_config = {
            "aime25":        {"bs": 8,  "total_run": 16, "n_examples": 30},
            "aime26":        {"bs": 8,  "total_run": 16, "n_examples": 30},
            "hmmt":          {"bs": 8,  "total_run": 16, "n_examples": 60},
            "math":          {"bs": 8,  "total_run": 8,  "n_examples": 500},
            "gpqa":          {"bs": 8,  "total_run": 8,  "n_examples": 198},
            "livecodebench": {"bs": 8,  "total_run": 8,  "n_examples": 383},
        }
    else:
        raise ValueError(f"Not support model_size: {model_size}")

    return task_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tasks using subprocess.")
    parser.add_argument("--model_dir", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                        help="Model directory path")
    parser.add_argument("--model_size", type=str, default="14B", help="model_size")
    parser.add_argument("--tasks", type=str, default="aime",
                        help="Comma-separated list of tasks (e.g., aime,math,gpqa)")
    parser.add_argument("--output_dir", type=str, default="./results/aime",
                        help="Directory to store output results")
    parser.add_argument("--attention_implementation", type=str, default="eager",
                        help="attention implementations")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit for the number of samples to process")
    parser.add_argument("--num_gpus", default="8", type=int)
    parser.add_argument("--sparsity_method", default='dense', type=str)
    parser.add_argument("--token_budget", default="2048", type=str)
    parser.add_argument("--max_tokens", default="32768", type=str)
    parser.add_argument("--run_id", default=0, type=int)
    parser.add_argument("--run_end_id", type=int)
    parser.add_argument("--verbose", action="store_true", help="whether to print verbose information or not")
    args, _ = parser.parse_known_args()
    parser = expand_parser_for_methods(parser, args.sparsity_method)
    args = parser.parse_args()

    limit = args.limit
    num_gpus = args.num_gpus
    max_tokens = args.max_tokens

    model_dir = args.model_dir
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    sparsity_method = args.sparsity_method
    token_budgets = [t.strip() for t in args.token_budget.split(",") if t.strip()]

    output_dir = args.output_dir
    attention_implementation = args.attention_implementation

    task_config = choose_task_config(args.model_size)

    for task in tasks:
        if task not in task_config:
            print(f"Error: Unknown task '{task}'")
            sys.exit(1)

        bs = task_config[task]["bs"]
        total_run = task_config[task]["total_run"]
        if args.run_end_id is None:
            batched_run = total_run // bs
        else:
            batched_run = args.run_end_id // bs

        print(f"\n{'='*40}")
        print(f"Starting task: {task}")
        print(f"Batch size: {bs} | total_run: {total_run}")

        if sparsity_method == "eviction":
            param_combinations = [(tb,) for tb in token_budgets]
        elif "quant" in sparsity_method or sparsity_method == "dense":
            param_combinations = [()]
        else:
            raise ValueError(f"Unknown sparsity_method: {sparsity_method}")

        for params in param_combinations:
            if "quant" in sparsity_method:
                param_desc = f"K{args.kbits}V{args.vbits}_g{args.q_group_size}_r{args.residual_length}"
                cli_params = [
                    "--kbits", str(args.kbits),
                    "--vbits", str(args.vbits),
                    "--residual_length", str(args.residual_length),
                    "--q_group_size", str(args.q_group_size),
                ]
            elif sparsity_method == "eviction":
                (token_budget,) = params
                param_desc = f"budget={token_budget}, {args.eviction_mode}"
                cli_params = [
                    "--token_budget", str(token_budget),
                    "--residual_length", str(args.residual_length),
                    "--eviction_mode", args.eviction_mode,
                    "--n_large", str(args.n_large),
                ]
                if 'range' in args.eviction_mode and args.eviction_mode != 'evict_range_cur':
                    param_desc += f", nl={args.n_large}"
                if args.rkv_lambda:
                    cli_params += ["--rkv_lambda", str(args.rkv_lambda)]
                    param_desc += f", lambda={args.rkv_lambda}"
                if args.smooth:
                    cli_params.append("--smooth")
                    param_desc += ", smooth"
                if args.temperature != 1.0:
                    cli_params += ["--temperature", str(args.temperature)]
                    param_desc += f", temp={args.temperature}"
                if args.verbose:
                    cli_params.append("--verbose")
            elif sparsity_method == "dense":
                param_desc = "dense"
                cli_params = []

            print(f"\n{'─'*30}")
            print(f"Processing Task:{task} | {sparsity_method}: {param_desc}")

            active_procs = {}
            available_gpus = deque(range(num_gpus))

            output_config_subdir = os.path.join(output_dir, f"{task}_bs{bs}_{param_desc.replace(', ', '_')}")

            # Build job queue over (rollout_id, data_shard) pairs
            num_data_parallel = num_gpus
            n_examples = task_config[task]["n_examples"]
            shard_size = n_examples // num_data_parallel

            job_queue = deque()
            run_counter_start = args.run_id // bs
            for run_i in range(run_counter_start, batched_run):
                current_run_id = run_i * bs
                for shard_id in range(num_data_parallel):
                    ex_start_i = shard_id * shard_size
                    ex_end_i = (shard_id + 1) * shard_size if shard_id < num_data_parallel - 1 else n_examples
                    job_queue.append({
                        "run_id": current_run_id,
                        "ex_start_i": ex_start_i,
                        "ex_end_i": ex_end_i,
                    })

            while job_queue or active_procs:
                for proc, info in list(active_procs.items()):
                    if proc.poll() is not None:
                        print(f"Run {info['run_id']} shard ex{info['ex_start_i']} on GPU {info['gpu_id']} finished.")
                        available_gpus.append(info['gpu_id'])
                        del active_procs[proc]

                while job_queue and available_gpus:
                    gpu_id = available_gpus.popleft()
                    job = job_queue.popleft()
                    current_run_id = job["run_id"]
                    ex_start_i = job["ex_start_i"]
                    ex_end_i = job["ex_end_i"]

                    print(f"Launching run {current_run_id} shard ex{ex_start_i}:{ex_end_i} on GPU {gpu_id}...")

                    env = os.environ.copy()
                    cmd = [
                        "python", "eval_hf.py",
                        "--model_name_or_path", model_dir,
                        "--data_name", task,
                        "--batch_size", str(bs),
                        "--output_dir", output_config_subdir,
                        "--attention_implementation", attention_implementation,
                        "--use_batch_exist",
                        "--surround_with_messages",
                        "--rank", str(gpu_id),
                        "--sparsity_method", sparsity_method,
                        "--run_id", str(current_run_id),
                        "--max_tokens", str(max_tokens),
                    ] + cli_params
                    if num_gpus > 1:
                        cmd += ["--ex_start_i", str(ex_start_i), "--ex_end_i", str(ex_end_i)]

                    proc = subprocess.Popen(cmd, env=env)
                    active_procs[proc] = {"gpu_id": gpu_id, "run_id": current_run_id, "ex_start_i": ex_start_i}

                if (job_queue and not available_gpus) or (not job_queue and active_procs):
                    time.sleep(5)

            if task != "livecodebench": # remove this line at last
                get_results_cmd = [
                    "python", "summary_results.py",
                    "--model_name_or_path", model_dir,
                    "--data_name", task,
                    "--limit", str(limit),
                    "--output_dir", output_config_subdir,
                    "--total_run", str(total_run),
                ]

                try:
                    subprocess.run(get_results_cmd, check=True)
                    print(f"Successfully generated results for {param_desc}")
                except subprocess.CalledProcessError as e:
                    print(f"Error generating results: {e}")

        print(f"\nCompleted: {task}")

    print("\n All tasks and configurations completed!")
