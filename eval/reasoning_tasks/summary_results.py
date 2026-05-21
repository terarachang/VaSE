import json
import torch
import numpy as np
import re
import importlib.util
import os
import argparse
# import vllm.envs as envs
from tqdm import tqdm
from Utils.parser import *
from Utils.data_loader import load_data
from Utils.math_normalization import *
from Utils.grader import *
from Utils.livecodebench import compute_scores as livecodebench_compute_scores
from parallel_run_hf import choose_task_config
import pickle
import subprocess

def merge_shard_files(output_runnum_subdir):
    """Merge completions_shard*.jsonl and other_info_shard*.json into their canonical names."""
    import glob
    shard_files = sorted(
        glob.glob(os.path.join(output_runnum_subdir, "completions_shard*.jsonl")),
        key=lambda f: int(re.search(r'shard(\d+)', f).group(1))
    )
    if not shard_files:
        return
    completion_filepath = os.path.join(output_runnum_subdir, "completions.jsonl")
    with open(completion_filepath, 'w') as out_f:
        for shard_file in shard_files:
            with open(shard_file, 'r') as in_f:
                out_f.write(in_f.read())

    info_files = sorted(
        glob.glob(os.path.join(output_runnum_subdir, "other_info_shard*.json")),
        key=lambda f: int(re.search(r'shard(\d+)', f).group(1))
    )

    merged_lens, merged_batch_times = [], []
    for info_file in info_files:
        with open(info_file, 'r') as f:
            info = json.load(f)
        merged_lens.extend(info['generate_lens'])
        merged_batch_times.extend(info.get('batch_times', []))
    merged_info = {}
    merged_info['generate_lens'] = merged_lens
    merged_info['batch_times'] = merged_batch_times
    with open(os.path.join(output_runnum_subdir, "other_info.json"), 'w') as f:
        json.dump(merged_info, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--limit', type=int, default=-1, help="limit")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--total_run", default=1, type=int)
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose information or not.")
    args = parser.parse_args()
    
    return args


def infer(args):
    print(args)
    examples = load_data(args.data_name, args.split, args.data_dir)

    limit = args.limit
    if limit > 0:
        examples = examples[:limit]

    if args.data_name == "livecodebench":
        if not os.path.exists("./data/livecodebench/livecodebench_tests"):
            subprocess.run(["python", "./data/livecodebench/download_tests.py"])

        with open("./data/livecodebench/test.jsonl", "r") as f:
            jobs = [json.loads(line) for line in f]
            if limit > 0:
                jobs = jobs[:limit]


    for run_i in range(args.total_run):
        merge_shard_files(os.path.join(args.output_dir, f"run_{run_i}"))
    # batch_times is identical across run_0 ... run_{bs-1}
    bs = choose_task_config(model_size=None)[args.data_name]["bs"]
    batch_times = []
    for run_i in range(0, args.total_run, bs):
        with open(os.path.join(args.output_dir, f"run_{run_i}", "other_info.json"), 'r') as f:
            batch_times.extend(json.load(f)['batch_times'])
    total_time = sum(batch_times)

    generate_lens_list = []
    no_ans_list = []
    total_is_correct_arr = np.zeros((args.total_run, len(examples)), dtype=bool)

    for run_i in range(args.total_run):
        output_runnum_subdir = os.path.join(args.output_dir, f"run_{run_i}")

        completion_filepath = os.path.join(output_runnum_subdir, "completions.jsonl")

        completions = []
        is_correct_list = []
        with open(completion_filepath, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                completions.append(item["completion"])
        if limit > 0:
            completions = completions[:limit]

        other_info_filepath = os.path.join(output_runnum_subdir, "other_info.json")

        with open(other_info_filepath, 'r') as f:
            other_info = json.load(f)

        generate_lens = other_info['generate_lens']

        print(f"Successfully loaded run{run_i}!")

        no_ans_cnt = 0
        if args.data_name == "livecodebench":
            cache_path = os.path.join(output_runnum_subdir, "cache.jsonl")
            if os.path.exists(cache_path):
                os.remove(cache_path)
            Acc, is_correct_list = livecodebench_compute_scores(jobs, completions, cache_path)
            total_is_correct_arr[run_i] = is_correct_list
            print("# Examples:", len(completions))
            print(f"# Acc: {Acc:.1%}")
            if os.path.exists(cache_path):
                os.remove(cache_path)

        else:
            # check all the correct
            assert len(examples) == len(completions), f"data: {len(examples)}, gen: {len(completions)}"
            for ex_i in range(len(completions)):
                d = examples[ex_i]
                gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
                generated_responses = [completions[ex_i]]
                generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
                is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
                is_correct = any(is_correct_list)
                total_is_correct_arr[run_i, ex_i] = is_correct
                if not is_correct:
                    if args.verbose:
                        print('truth:', gt_ans)
                        print('model:', generated_answers)
                        breakpoint()
                    if len(generated_answers) == 1 and generated_answers[0] == '': no_ans_cnt += 1

            Acc = total_is_correct_arr[run_i].mean()
            print("# Examples:", len(completions))
            print(f"# Acc: {Acc:.1%}")
            print("-"*100)

        average_generate_len = sum(generate_lens) / len(generate_lens)
        max_generate_len = max(generate_lens)

        generate_lens_list.extend(generate_lens)
        no_ans_list.append(no_ans_cnt)

        summary_filepath = os.path.join(output_runnum_subdir, "summary.txt")

        with open(summary_filepath, "w") as f:
            #f.write(f"Model Path: {args.model_name_or_path}\n")
            f.write(f"Acc: {Acc:.4f}\n")
            f.write(f"Average generate length: {average_generate_len}\n")
            f.write(f"Max generate length: {max_generate_len}\n")
            f.write(f"Total time (min): {total_time/60:.1f}\n")
            f.write("\n")


    Acc_list = total_is_correct_arr.mean(-1)
    avg_no_ans_cnt = round(sum(no_ans_list) / len(no_ans_list))
    print(f"Acc range: [{Acc_list.min():.2%}, {Acc_list.max():.2%}]")
    print(f"Average Acc: {Acc_list.mean():.2%}")
    print(f"No ans count: {avg_no_ans_cnt}")

    # generate_len
    average_generate_len = sum(generate_lens_list) / len(generate_lens_list)
    max_generate_len = max(generate_lens_list)
    print(f"Max generate length: {max_generate_len}")
    print(f"Average generate length: {round(average_generate_len)}")

    average_token_per_sec = sum(generate_lens_list) / total_time
    print(f"Total time (min): {total_time/60:.1f}")
    print(f"Average token per sec: {round(average_token_per_sec)}")

    overall_summary_filepath = os.path.join(args.output_dir, "overall_summary.txt")
    with open(overall_summary_filepath, "w") as f:
        f.write(f"# Examples: {len(examples)}\n")
        f.write(f"Total_run: {args.total_run}\n")
        f.write(f"Acc range: [{Acc_list.min():.2%}, {Acc_list.max():.2%}]\n")
        f.write(f"Average Acc: {Acc_list.mean():.2%}\n")
        f.write(f"No ans count: {avg_no_ans_cnt}\n")
        f.write(f"Average generate length: {round(average_generate_len)}\n")
        f.write(f"Max generate length: {max_generate_len}\n")
        f.write(f"Total time (min): {total_time/60:.1f}\n")
        f.write(f"Average token per sec: {round(average_token_per_sec)}\n")
        f.write("\n")


    np.save(os.path.join(args.output_dir, "total_is_correct_arr.npy"), total_is_correct_arr)
    print("Results saved to ", overall_summary_filepath)




if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    infer(args)
