model_dir="Qwen/Qwen3-4B"
output_dir="./result_quant_prerope_key"
attention_implementation="fa2"
max_tokens=32768
num_gpus=1
limit=-1

# tasks="aime24,aime25,math,gpqa"
tasks="aime25"

python parallel_run_hf.py \
      --model_dir "$model_dir" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention_implementation "$attention_implementation" \
      --sparsity_method "quant_prerope" \
      --num_gpus "$num_gpus" \
      --limit "$limit" \
      --max_tokens "$max_tokens" \
      --remainder_length 64 \
      --nbits 2 \
      --quant_k \
