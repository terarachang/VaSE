import argparse
import json
import os
import time

import torch

from generation_utils import batch_exist_generate
from my_utils import *
from Utils.grader import *
from Utils.parser import *


def infer(args, all_answers, num_runs):
    Acc_list = []
    generate_lens_list = []
    total_time_list = []
    no_ans_list = []

    for i in range(args.run_id, num_runs):
        output_runnum_subdir = os.path.join(args.output_dir, f"run_{i}")
        completion_filepath = os.path.join(output_runnum_subdir, "completions.jsonl")
        other_info_filepath = os.path.join(output_runnum_subdir, "other_info.json")

        completions = []
        with open(completion_filepath, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                completions.append(item["completion"])
        
        with open(other_info_filepath, 'r') as f:
            other_info = json.load(f)

        completions = completions[:args.limit]
        generate_lens = other_info['generate_lens']
        total_time = other_info['total_time']
        evict_rate = other_info['counts_evict'] / len(completions)

        # check all the correct
        assert len(all_answers) == len(completions), f"data: {len(all_answers)}, gen: {len(completions)}"
        correct_cnt = 0
        no_ans_cnt = 0
        for i in range(len(completions)):
            gt_ans = all_answers[i]
            generated_responses = [completions[i]]
            generated_answers = [extract_answer(generated_response) for generated_response in generated_responses]
            is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
            is_correct = any(is_correct_list)
            if is_correct:
                correct_cnt += 1
            else:
                if args.verbose:
                    print(completions[i])
                    print('truth:', gt_ans, 'model:',  generated_answers)
                    breakpoint()
                if len(generated_answers) == 1 and generated_answers[0] == '': no_ans_cnt += 1

        Acc = correct_cnt / len(all_answers)
        total_len = sum(generate_lens)
        print(f"# Acc: {Acc:.1%}")
        print(f"# Evict rate: {evict_rate:.1%}")
        print("# No Answer:", no_ans_cnt)
        print("# Time:", total_time)
        print("-"*100)

        average_generate_len = sum(generate_lens) / len(generate_lens)
        max_generate_len = max(generate_lens)
        average_time_per_token = total_time / sum(generate_lens)
        
        Acc_list.append(Acc)
        generate_lens_list.extend(generate_lens)
        total_time_list.append(total_time)
        no_ans_list.append(no_ans_cnt)

        summary_filepath = os.path.join(output_runnum_subdir, "summary.txt")

        with open(summary_filepath, "w") as f:
            f.write(f"Acc: {Acc:.4f}\n")
            f.write(f"Average generate length: {average_generate_len}\n")
            f.write(f"Max generate length: {max_generate_len}\n")
            f.write(f"Total time: {total_time:.2f}\n")
            f.write(f"Average time per token: {average_time_per_token}\n")
            f.write(f"Evict rate: {evict_rate}\n")

    Acc = sum(Acc_list) / len(Acc_list)
    avg_no_ans_cnt = round(sum(no_ans_list) / len(no_ans_list))
    print(f"Average Acc: {Acc:.2%}")
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

    overall_summary_filepath = os.path.join(args.output_dir, "overall_summary.txt")
    with open(overall_summary_filepath, "w") as f:
        f.write(f"Total_run: {num_runs}\n")
        f.write(f"# Examples: {args.limit}\n")
        f.write(f"Average Acc: {Acc:.2%}\n")
        f.write(f"No ans count: {avg_no_ans_cnt}\n")
        f.write(f"Average generate length: {int(average_generate_len)}\n")
        f.write(f"Max generate length: {max_generate_len}\n")
        f.write(f"Average time: {int(total_time)} min\n")
        f.write(f"Average token per sec: {int(average_token_per_sec)}\n")
    print("Results saved to ", overall_summary_filepath)


def run_eval(
    model, tokenizer, forward_kwargs,
    max_tokens,
    tokenized_data, all_answers,
    output_dir, total_run):

    model.eval()
    eos_token_id = tokenizer.eos_token_id

    generate_lens = []
    counts_evict = 0
    total_time = 0
    torch.manual_seed(0)

    start_i = get_ckpt_start_i(os.path.join(args.output_dir, "run_0"))
    for batch_i, inputs in enumerate(tokenized_data):
        if batch_i < start_i: continue
        begin = time.time()
        print("start ex:", batch_i, flush=True)
        inputs = inputs.to(model.device)
        outputs, counts = batch_exist_generate(
            model,
            **inputs,
            max_length=max_tokens,
            do_sample=True,
            verbose=args.verbose,
            **forward_kwargs)

        end = time.time()
        batch_time = (end - begin) / 60
        total_time = total_time + batch_time
        print("finish ex:", batch_i, "time:", batch_time, "output:", outputs.shape, flush=True)

        for j in range(len(outputs)):
            output_seq = outputs[j]
            output_tokens = (output_seq != eos_token_id).sum().item()
            prompt_tokens = (inputs['input_ids'][j] != eos_token_id).sum().item()
            generate_lens.append(output_tokens - prompt_tokens)

        completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        counts_evict += counts

        # Write after each batch
        for j in range(total_run):
            output_runnum_subdir = os.path.join(args.output_dir, f"run_{j}")
            os.makedirs(output_runnum_subdir, exist_ok=True)
            completion_filepath = os.path.join(output_runnum_subdir, "completions.jsonl")
            other_info_filepath = os.path.join(output_runnum_subdir, "other_info.json")
            ids = torch.arange(0, len(generate_lens), total_run) + j

            other_info = {
                "generate_lens": [generate_lens[i] for i in ids],
                "total_time": total_time / total_run,
                "counts_evict": counts_evict / total_run,
            }
            with open(other_info_filepath, 'w') as f:
                json.dump(other_info, f)
            with open(completion_filepath, 'a') as f:
                f.write(json.dumps({"completion": completions[j]}) + '\n')

    infer(args, all_answers, num_runs=total_run)


def set_output_dir(args):
    dir_name = os.path.join(args.output_dir, os.path.basename(args.model_name), args.sparsity_method)
    if args.sparsity_method == 'Quant':
        dir_name = os.path.join(dir_name, f'K{args.kbits}V{args.vbits}g{args.q_group_size}')
    elif args.sparsity_method == 'Evict':
        dir_name = os.path.join(dir_name, args.eviction_mode, str(args.token_budget))
        if args.rkv_lambda: dir_name += f'_lambda={args.rkv_lambda}'
        if args.smooth: dir_name += '_smooth'
        if args.n_large != -1: dir_name += f'_n_large={args.n_large}'
    args.output_dir = dir_name



if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument("--t_b", dest="token_budget", type=int, help="only used in sparsity_method = eviction")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--attention_implementation', type=str, default="eager")
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--n_offset', type=int, default=0)
    parser.add_argument('--total_run', type=int, default=4)
    parser.add_argument('--run_id', type=int, default=0)
    #parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--print_only', action='store_true')
    parser.add_argument('--output_dir', type=str, default='gsm8k')
    parser.add_argument("--sp", dest="sparsity_method", choices=['None', 'Quant', 'Evict'], type=str, default='Evict')
    parser.add_argument("--verbose", action="store_true")
    args, _ = parser.parse_known_args()
    parser = expand_parser_for_methods(parser, args.sparsity_method)
    args = parser.parse_args()
    if not args.print_only:
        set_output_dir(args)
    print(args)
    print('-'*100)

    # load and tokenize data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    n_return_sequences = 1 if args.print_only else args.total_run
    tokenized_data, all_answers = load_gsm8k(args.model_name, tokenizer, args.limit, args.split, n_return_sequences)
    visualize_token(tokenizer, tokenized_data[0]['input_ids'][0])

    if args.print_only:
        infer(args, all_answers, args.total_run)
    else:
        # prepare configs and override the forward pass if needed
        if args.sparsity_method == 'None':
            forward_kwargs = {}
        elif args.sparsity_method == 'Quant':
            forward_kwargs = init_quant_configs(args)
        elif args.sparsity_method == 'Evict':
            forward_kwargs = init_evict_configs(args)
            from modified.transformers.modify_forward import wrap_evict_attn_forward
            wrap_evict_attn_forward(args.model_name)
            assert args.attention_implementation == 'eager', 'currently only support eager'
        else:
            raise NotImplementedError(f'{args.sparsity_method} is not implemented')

        model = AutoModelForCausalLM.from_pretrained(args.model_name,
            torch_dtype=torch.bfloat16, device_map="auto", 
            attn_implementation=args.attention_implementation)

        run_eval(
            model=model, tokenizer=tokenizer,
            forward_kwargs=forward_kwargs,
            max_tokens=args.max_tokens,
            tokenized_data=tokenized_data, all_answers=all_answers,
            output_dir=args.output_dir, total_run=args.total_run,
            )
