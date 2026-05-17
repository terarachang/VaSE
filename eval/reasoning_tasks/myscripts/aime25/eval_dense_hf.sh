model_arr=('Qwen/Qwen3-4B')
model_dir=${model_arr[0]}
attention_implementation="flash_attention_2"
max_tokens=32768
num_gpus=1
limit=-1

tasks="aime25"
output_dir="./${tasks}/$(basename "$model_dir")/dense"

python parallel_run_hf.py \
      --model_dir "$model_dir" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention_implementation "$attention_implementation" \
      --sparsity_method "dense" \
      --num_gpus "$num_gpus" \
      --limit "$limit" \
      --max_tokens "$max_tokens" \
