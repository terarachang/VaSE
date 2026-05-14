import json
import torch
import numpy as np
from typing import List, Optional, Tuple
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
import pickle
import subprocess

def calculate_quantile_sparsity(
    all_batch_sparsitys_info: List[List[Optional[Tuple[Tuple[int, int], ...]]]],
    group_size: int = 1000
) -> List[float]:
    """
    Calculates sparsity for each quantile group of sequence steps.

    Each group aggregates results over `group_size` sequence steps across all batches.
    The sparsity is computed as 1 - (total activated blocks / total original blocks) for that group.

    Args:
        all_batch_sparsitys_info: Nested list structure from each batch.
        group_size: Number of sequence steps per quantile group.

    Returns:
        A list of sparsity values for each quantile group.
    """
    if not all_batch_sparsitys_info:
        return []

    # Compute maximum number of steps across all batches
    lengths = [len(batch_sequence_info) for batch_sequence_info in all_batch_sparsitys_info]
    max_steps = max(lengths) if lengths else 0

    if max_steps == 0:
        return []

    # Initialize per-step totals with (0, 0)
    per_step_totals = [(0, 0) for _ in range(max_steps)]

    # Aggregate across all batches for each step
    for batch_sequence_info in all_batch_sparsitys_info:
        for step_idx, step_info in enumerate(batch_sequence_info):
            if step_info is None:
                continue
            act = 0
            orig = 0
            for layer_info in step_info:
                act += layer_info[0]
                orig += layer_info[1]
            per_step_totals[step_idx] = (
                per_step_totals[step_idx][0] + act,
                per_step_totals[step_idx][1] + orig
            )

    # Group steps and compute sparsity for each group
    quantile_results = []
    num_groups = (max_steps + group_size - 1) // group_size

    for i in range(num_groups):
        start = i * group_size
        end = min((i + 1) * group_size, max_steps)
        group_act = 0
        group_orig = 0

        for step_idx in range(start, end):
            group_act += per_step_totals[step_idx][0]
            group_orig += per_step_totals[step_idx][1]

        if group_orig == 0:
            sparsity = 0.0
        else:
            sparsity = 1 - group_act / group_orig
        sparsity = round(sparsity, 2)
        quantile_results.append(sparsity)

    return quantile_results


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
    if not info_files:
        return
    merged_lens, merged_time, merged_counts = [], 0.0, 0
    for info_file in info_files:
        with open(info_file, 'r') as f:
            info = json.load(f)
        merged_lens.extend(info.get('generate_lens', []))
        merged_time += info.get('total_time', 0.0)
        merged_counts += info.get('counts', 0)
    n = len(info_files)
    merged_info = {k: v for k, v in json.load(open(info_files[0])).items()}
    merged_info['generate_lens'] = merged_lens
    merged_info['total_time'] = merged_time / n
    merged_info['counts'] = merged_counts // n
    with open(os.path.join(output_runnum_subdir, "other_info.json"), 'w') as f:
        json.dump(merged_info, f)


def parse_list(arg):
    return arg.split(',')


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
    parser.add_argument("--profile_sparsity", action="store_true", help="Flag to profile sparsity")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose information or not.")
    args = parser.parse_args()
    
    return args


def infer(args):
    print(args)
    generate_lens = []
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

    generate_lens_list = []
    total_time_list = []
    overall_sparsity_list = []
    sparsity_16k_list = []
    sparsity_32k_list = []
    no_ans_list = []
    total_is_correct_arr = np.zeros((args.total_run, len(examples)), dtype=bool)

    for run_i in range(args.total_run):
        output_runnum_subdir = os.path.join(args.output_dir, f"run_{run_i}")
        merge_shard_files(output_runnum_subdir)

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
        total_time = other_info['total_time']
        counts = other_info.get('counts', 0)
        if isinstance(counts, int):
            rate = counts / len(examples)
        else:
            rate = torch.FloatTensor(counts) / sum(generate_lens)
        if args.profile_sparsity:
            overall_sparsity = other_info['overall_sparsity']

            sparsity_info_filepath = os.path.join(output_runnum_subdir, "sparsity_info.json")

            with open(sparsity_info_filepath, 'r') as f:
                all_batch_sparsitys_info = json.load(f)

            quantile_sparsities = calculate_quantile_sparsity(all_batch_sparsitys_info, group_size=1000)

            if len(quantile_sparsities) >= 16:
                sparsity_16k = quantile_sparsities[15]
                sparsity_16k_list.append(sparsity_16k)

            if len(quantile_sparsities) >= 32:
                sparsity_32k = quantile_sparsities[31]
                sparsity_32k_list.append(sparsity_32k)
        elif "quest" in args.output_dir.lower():
            overall_sparsity = other_info['overall_sparsity']



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
            total_len = sum(generate_lens)
            print("# Examples:", len(completions))
            print(f"# Acc: {Acc:.1%}")
            print(f"# Rate: {rate:.1%}")
            print("# No Answer:", no_ans_cnt)
            print("# Time:", total_time)
            print("-"*100)

        average_generate_len = sum(generate_lens) / len(generate_lens)
        max_generate_len = max(generate_lens)

        average_time_per_token = total_time / sum(generate_lens)
        generate_lens_list.extend(generate_lens)
        total_time_list.append(total_time)
        no_ans_list.append(no_ans_cnt)
        if args.profile_sparsity or "quest" in args.output_dir.lower():
            overall_sparsity_list.append(overall_sparsity)

        summary_filepath = os.path.join(output_runnum_subdir, "summary.txt")

        with open(summary_filepath, "w") as f:
            #f.write(f"Model Path: {args.model_name_or_path}\n")
            f.write(f"Acc: {Acc:.4f}\n")
            f.write(f"Average generate length: {average_generate_len}\n")
            f.write(f"Max generate length: {max_generate_len}\n")
            f.write(f"Total time: {total_time:.2f}\n")
            f.write(f"Average time per token: {average_time_per_token}\n")
            if args.profile_sparsity:
                f.write(f"Overall sparsity: {overall_sparsity}\n")
                if len(quantile_sparsities) >= 16:
                    f.write(f"Sparsity at 16k: {sparsity_16k}\n")
                if len(quantile_sparsities) >= 32:
                    f.write(f"Sparsity at 32k: {sparsity_32k}\n")
                f.write(f"Quantile sparsities: {quantile_sparsities}\n")
            elif "quest" in args.output_dir.lower():
                f.write(f"Overall sparsity: {overall_sparsity}\n")
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
    print(f"Average generate length: {int(average_generate_len)}")

    total_time = sum(total_time_list) / len(total_time_list)
    #average_time_per_token = sum(total_time_list) / sum(generate_lens_list)
    average_token_per_sec = sum(generate_lens_list) / (sum(total_time_list)*60)
    print(f"Average time: {int(total_time)} min")
    print(f"Average token per sec: {int(average_token_per_sec)}")

    # sparsity
    if args.profile_sparsity:
        overall_sparsity = sum(overall_sparsity_list) / len(overall_sparsity_list)
        print("Overall_sparsity: ", overall_sparsity)

        if len(sparsity_16k_list) > 0:
            average_sparsity_16k = sum(sparsity_16k_list) / len(sparsity_16k_list)
            print(f"Average sparsity at 16k: {average_sparsity_16k}")
        if len(sparsity_32k_list) > 0:
            average_sparsity_32k = sum(sparsity_32k_list) / len(sparsity_32k_list)
            print(f"Average sparsity at 32k: {average_sparsity_32k}")
    elif "quest" in args.output_dir.lower():
        overall_sparsity = sum(overall_sparsity_list) / len(overall_sparsity_list)
        print("Overall sparsity: ", overall_sparsity)

    overall_summary_filepath = os.path.join(args.output_dir, "overall_summary.txt")
    with open(overall_summary_filepath, "w") as f:
        #f.write(f"Model Path: {args.model_name_or_path}\n")
        f.write(f"# Examples: {len(examples)}\n")
        f.write(f"Total_run: {args.total_run}\n")
        f.write(f"Acc range: [{Acc_list.min():.2%}, {Acc_list.max():.2%}]\n")
        f.write(f"Average Acc: {Acc_list.mean():.2%}\n")
        f.write(f"No ans count: {avg_no_ans_cnt}\n")
        f.write(f"Average generate length: {int(average_generate_len)}\n")
        f.write(f"Max generate length: {max_generate_len}\n")
        f.write(f"Average time: {int(total_time)} min\n")
        f.write(f"Average token per sec: {int(average_token_per_sec)}\n")
        if args.profile_sparsity:
            f.write(f"Overall sparsity: {overall_sparsity}\n")
            if len(sparsity_16k_list) > 0:
                f.write(f"Average sparsity at 16k: {average_sparsity_16k}\n")
            if len(sparsity_32k_list) > 0:
                f.write(f"Average sparsity at 32k: {average_sparsity_32k}\n")
        elif "quest" in args.output_dir.lower():
            f.write(f"Overall sparsity: {overall_sparsity}\n")
        f.write("\n")


    np.save(os.path.join(args.output_dir, "total_is_correct_arr.npy"), total_is_correct_arr)
    print("Results saved to ", overall_summary_filepath)




if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    infer(args)
