import torch
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_utils import load_gsm8k, visualize_token
os.environ["TOKENIZERS_PARALLELISM"] = "false"





def hf_generation(model, tokenized_data, tokenizer, max_length):
    n_layers = model.config.num_hidden_layers
    for l in range(n_layers):
        os.makedirs(os.path.join(args.output_dir, f'L{l}'), exist_ok=True)

    for ex_i, inputs in tqdm(enumerate(tokenized_data), desc="Caching"):
        inputs = inputs.to(model.device)
        outputs = model.generate(
            **inputs, do_sample=True, max_length=max_length,
            use_cache=True, return_dict_in_generate=True,
        )

        for l in range(n_layers):
            vcache = outputs.past_key_values.layers[l].values.squeeze().cpu() # (num_kv_heads, seq_len, head_dim)
            torch.save(vcache, f'{args.output_dir}/L{l}/ex-{ex_i}.pt')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='Qwen/Qwen3-4B')
    parser.add_argument("--output_dir", type=str, default='vcache')
    parser.add_argument("--attn_impl", type=str, default='eager')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()
    print(args)
    print('-'*100)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_data, all_answers = load_gsm8k(args.model_name, tokenizer, args.n_samples, args.split, args.batch_size)
    visualize_token(tokenizer, tokenized_data[0]['input_ids'][0])
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl, device_map="cuda:0")
    hf_generation(model, tokenized_data, tokenizer, args.max_length)
