model_dir="Qwen/Qwen3-4B"
attention_implementation="eager"
max_tokens=32768
num_gpus=1
limit=-1

tasks="aime25"

#token_budget="2048,4096,6144,8192"
token_budget="4096"

mode='attn'
output_dir="./${tasks}/$(basename "$model_dir")/${mode}"

python parallel_run_hf.py \
      --model_dir "$model_dir" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention_implementation "$attention_implementation" \
      --sparsity_method "eviction" \
      --num_gpus "$num_gpus" \
      --limit "$limit" \
      --max_tokens "$max_tokens" \
      --residual_length 64 \
      --eviction_mode "$mode" \
      --token_budget "$token_budget" \
      --smooth \
