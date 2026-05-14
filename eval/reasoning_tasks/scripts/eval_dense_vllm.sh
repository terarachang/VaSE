model_dir="Qwen/Qwen3-14B"
output_dir="./result_vllm_dense"
max_tokens=32768
num_gpus=1
limit=-1

# tasks="aime26,hmmt"
tasks="aime26"


python parallel_run_vllm.py \
      --model_dir "$model_dir" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --num_gpus "$num_gpus" \
      --limit "$limit" \
      --max_tokens "$max_tokens" \

