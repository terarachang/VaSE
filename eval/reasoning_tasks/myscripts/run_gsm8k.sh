model_arr=('Qwen/Qwen3-4B' 'Qwen/Qwen3-14B')
model=${model_arr[0]}
echo $model

# Full
python run_gsm8k.py --token_budget 512 --sparsity_method dense --model_name $model
# CUR variants
python run_gsm8k.py --token_budget 512 --mode cur_fixed_gauss --model_name $model
python run_gsm8k.py --token_budget 512 --mode cur_resample_gauss --model_name $model
# SnapKV
python run_gsm8k.py --token_budget 512 --mode attn --smooth --model_name $model
# VASE-Attn variants
python run_gsm8k.py --token_budget 512 --mode small_range_sink_sample_attn --smooth --model_name $model --n_large 256
python run_gsm8k.py --token_budget 512 --mode absmax_sink_sample_attn --smooth --model_name $model --n_large 256
python run_gsm8k.py --token_budget 512 --mode var_sink_sample_attn --smooth --model_name $model --n_large 256
python run_gsm8k.py --token_budget 512 --mode l2_sink_sample_attn --smooth --model_name $model --n_large 256
# RKV variants
python run_gsm8k.py --token_budget 512 --mode attn_rkv --smooth --rkv_lambda 0.5 --model_name $model
python run_gsm8k.py --token_budget 512 --mode attn_rkv --smooth --rkv_lambda 0.9 --model_name $model
python run_gsm8k.py --token_budget 512 --mode attn_rkv --smooth --rkv_lambda 0.1 --model_name $model
