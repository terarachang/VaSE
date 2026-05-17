import json
from transformers import AutoConfig
import torch
import re
import importlib.util
import os
import argparse
import random
import time
from datetime import datetime
from tqdm import tqdm
from Utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from Utils.parser import *
from Utils.data_loader import load_data
from Utils.math_normalization import *
from Utils.grader import *
import pickle
from math import comb
from my_utils import *
from generation_utils import batch_exist_generate
from typing import Optional, Tuple


def parse_list(arg):
    return arg.split(',')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--max_tokens", default=32768, type=int)
    parser.add_argument("--prompt_type", default="qwen-instruct", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--surround_with_messages", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--sparsity_method", default='dense', type=str)
    parser.add_argument("--token_budget", default=2048, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--attention_implementation", default="eager", type=str)
    parser.add_argument("--use_batch_exist", action="store_true")
    parser.add_argument("--run_id", default=0, type=int)
    parser.add_argument("--verbose", action="store_true", help="whether to print verbose information or not")
    parser.add_argument("--ex_start_i", default=0, type=int, help="Start example index for data parallelism")
    parser.add_argument("--ex_end_i", default=None, type=int, help="End example index for data parallelism")
    args, _ = parser.parse_known_args()
    parser = expand_parser_for_methods(parser, args.sparsity_method)
    args = parser.parse_args()
    
    return args

def get_conversation_prompt_by_messages(tokenizer, messages):
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    return tokenized

def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join(".", "prompts", prompt_type, f"prompt.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    
    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    
    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format


def infer(args):
        
    print(args)
    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")
    device = f"cuda:{args.rank}"

    generate_lens = []
    
    examples = load_data(args.data_name, args.split, args.data_dir)
    print(f"{args.data_name}, # Examples: {len(examples)}")
    print('-'*100)

    examples = examples[args.ex_start_i:args.ex_end_i]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    prompt_batch = []
    for example in tqdm(examples, total=len(examples)):
        # parse question and answer
        question = parse_question(example, args.data_name)
        _, few_shot_prompt, question_format = get_three_prompt(args.prompt_type, args.data_name)
        
        if args.use_few_shot:
            cur_prompt = few_shot_prompt + question_format.format(question=question)
        else:
            cur_prompt = question_format.format(question=question)
        if args.surround_with_messages:
            if args.data_name in ["aime24", "aime25", "aime26", "hmmt", "math", "olympiadbench"]:
                messages = [
                    {
                        "role": "user",
                        "content": cur_prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                    }
                ]
            else:
                # for gpqa, livecodebench
                messages = [
                    {"role": "user", "content": cur_prompt}
                ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch.append(cur_prompt)

    
    is_data_parallel = args.ex_end_i is not None
    completion_filename = f"completions_shard{args.ex_start_i}.jsonl" if is_data_parallel else "completions.jsonl"
    other_info_filename = f"other_info_shard{args.ex_start_i}.json" if is_data_parallel else "other_info.json"

    output_runnum_subdir = os.path.join(args.output_dir, f"run_{args.run_id}")
    start_i = get_ckpt_start_i(output_runnum_subdir, completion_filename)
    if start_i >= len(examples):
        print(f"Already completed. Exit.")
        exit()

    generate_lens = []
    total_time = 0

    forward_kwargs = {}
    if "quant" in args.sparsity_method:
        forward_kwargs = init_quant_configs(args)
    elif args.sparsity_method == "eviction":
        forward_kwargs = init_evict_configs(args)
        from modified.transformers.modify_forward import wrap_evict_attn_forward
        wrap_evict_attn_forward(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                        torch_dtype=torch.bfloat16,
                                        device_map=device,
                                        attn_implementation=args.attention_implementation,
                                        )
    
    model.eval()
    eos_id_from_config = getattr(model.generation_config, "eos_token_id", None)
    eos_token_id = eos_id_from_config[0] if isinstance(eos_id_from_config, list) else eos_id_from_config
    batch_size = args.batch_size

    for i in range(start_i, len(prompt_batch)):
        inputs = prompt_batch[i].to(device)
        batch_input_ids = inputs['input_ids'].expand(batch_size, -1)
        attention_mask = inputs['attention_mask'].expand(batch_size, -1)
        if i == start_i: visualize_token(tokenizer, batch_input_ids[0])

        print("start batch: ", i, flush=True)
        begin = time.time()
        if args.use_batch_exist:
            outputs = batch_exist_generate(
                model,
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                max_length = args.max_tokens,
                do_sample=True,
                verbose=args.verbose,
                **forward_kwargs,
            )
        else:
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                max_length = args.max_tokens,
                do_sample=True,
                num_return_sequences=1,
                **forward_kwargs,
            )

        end = time.time()
        batch_time = (end - begin) / 60
        total_time = total_time + batch_time
        print("get output in batch: ", i, "time:", batch_time, "output:", outputs.shape, outputs.device, flush=True)
        

        for j in range(len(outputs)):
            output_seq = outputs[j]
            output_tokens = (output_seq != eos_token_id).sum().item()
            prompt_tokens = (batch_input_ids[j] != eos_token_id).sum().item()
            generate_lens.append(output_tokens - prompt_tokens)

        completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("finish batch: ", i, flush=True)
        
        # Write after each batch
        for j in range(batch_size):
            output_runnum_subdir = os.path.join(args.output_dir, f"run_{args.run_id+j}")
            os.makedirs(output_runnum_subdir, exist_ok=True)
            completion_filepath = os.path.join(output_runnum_subdir, completion_filename)
            other_info_filepath = os.path.join(output_runnum_subdir, other_info_filename)
            ids = torch.arange(0, len(generate_lens), batch_size) + j

            other_info = {
                "generate_lens": [generate_lens[i] for i in ids],
                "total_time": total_time / batch_size,
            }
            with open(other_info_filepath, 'w') as f:
                json.dump(other_info, f)
            with open(completion_filepath, 'a') as f:
                f.write(json.dumps({"completion": completions[j]}) + '\n')

        
    print("llm generate done")

    with open(other_info_filepath, 'w') as f:
        json.dump(other_info, f)

    print(f"Successfully saved run{args.run_id}!")

    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    infer(args)
