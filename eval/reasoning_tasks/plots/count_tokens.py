import argparse
import glob
import os
import re
import sys
sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoTokenizer
from tqdm import tqdm
from Utils.data_loader import load_data
from Utils.parser import parse_question

DATA_DIR = "./data"
SPLIT = "test"
DATASET_NAMES = ["aime25", "aime26", "hmmt", "math", "livecodebench"]

MATH_DATASETS = {"aime24", "aime25", "aime26", "hmmt", "math", "olympiadbench"}


def build_messages(question, data_name):
    if data_name in MATH_DATASETS:
        content = question + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    else:
        content = question
    return [{"role": "user", "content": content}]


def count_tokens(tokenizer, messages):
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return tokenized["input_ids"].shape[-1]


def read_avg_gen_length(data_name, tokenizer_name):
    pattern = os.path.join(data_name, os.path.basename(tokenizer_name), "dense", "*", "overall_summary.txt")
    matches = glob.glob(pattern)
    if not matches:
        return None
    with open(matches[0]) as f:
        for line in f:
            m = re.match(r"Average generate length:\s*(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B", help="Tokenizer name or path")
    parser.add_argument("--count_gen", action="store_true",
                        help="Add Gen Tokens column from dense overall_summary.txt")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True,
    )

    results = {}
    for data_name in DATASET_NAMES:
        examples = load_data(data_name, SPLIT, DATA_DIR)
        print(f"\n{data_name}: {len(examples)} examples")

        token_counts = []
        for example in tqdm(examples, desc=data_name):
            question = parse_question(example, data_name)
            messages = build_messages(question, data_name)
            n_tokens = count_tokens(tokenizer, messages)
            token_counts.append(n_tokens)

        avg = sum(token_counts) / len(token_counts)
        results[data_name] = {"avg_tokens": avg, "n_examples": len(examples)}

        if args.count_gen:
            results[data_name]["gen_tokens"] = read_avg_gen_length(data_name, args.tokenizer)

    # Print table
    col_w = 16
    n_cols = 4 if args.count_gen else 3
    total_w = col_w * n_cols + 4
    header = f"{'Dataset':<{col_w}} {'# Examples':>{col_w}} {'Prompt Tokens':>{col_w}}"
    if args.count_gen:
        header += f" {'Gen Tokens':>{col_w}}"

    print("\n" + "=" * total_w)
    print(header)
    print("-" * total_w)
    for name, info in results.items():
        row = f"{name:<{col_w}} {info['n_examples']:>{col_w}} {round(info['avg_tokens']):>{col_w}}"
        if args.count_gen:
            gen = info["gen_tokens"]
            row += f" {str(gen) if gen is not None else 'N/A':>{col_w}}"
        print(row)
    print("=" * total_w)


if __name__ == "__main__":
    main()
