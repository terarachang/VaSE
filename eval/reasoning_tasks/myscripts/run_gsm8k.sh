model_arr=('Qwen/Qwen3-4B' 'Qwen/Qwen3-14B')
model=${model_arr[0]}
echo $model

python run_gsm8k.py --t_b 512 --mode small_range_sink_sample_attn --smooth --model_name $model --n_large 256
#python run_gsm8k.py --t_b 512 --mode absmax_sink_sample_attn --smooth --model_name $model --n_large 256
#python run_gsm8k.py --t_b 512 --mode var_sink_sample_attn --smooth --model_name $model --n_large 256
#python run_gsm8k.py --t_b 512 --mode l2_sink_sample_attn --smooth --model_name $model --n_large 256

#python run_gsm8k.py --t_b 512 --mode range_cur --model_name $model
#python run_gsm8k.py --sp Quant --model_name $model --kbits 8 --vbits 2
#python run_gsm8k.py --t_b 512 --mode small_range --model_name $model
#python run_gsm8k.py --t_b 512 --mode large_range --model_name $model
#python run_gsm8k.py --t_b 512 --mode fully_random --model_name $model
#python run_gsm8k.py --t_b 512 --mode fully_random --model_name $model
#python run_gsm8k.py --t_b 512 --sp None --model_name $model
#python run_gsm8k.py --t_b 512 --mode small_range_sink --model_name $model
#python run_gsm8k.py --t_b 512 --mode small_range_attn --smooth --model_name $model --n_large 256
#python run_gsm8k.py --t_b 512 --mode small_range_sample_attn --smooth --model_name $model --n_large 256
#python run_gsm8k.py --t_b 512 --mode attn_sample --smooth --model_name $model
#python run_gsm8k.py --t_b 512 --mode large_random --model_name $model
#python run_gsm8k.py --t_b 512 --mode attn_range --model_name $model --smooth
#python run_gsm8k.py --t_b 512 --mode cur --model_name $model
#python run_gsm8k.py --t_b 512 --mode cur_no_guassian --model_name $model
#python run_gsm8k.py --t_b 512 --mode sink --model_name $model
#python run_gsm8k.py --t_b 512 --mode attn_rkv --smooth --rkv_lambda 0.5 --model_name $model --limit 200
#python run_gsm8k.py --t_b 512 --mode attn_rkv --smooth --rkv_lambda 0.9 --model_name $model --limit 200
#python run_gsm8k.py --t_b 512 --mode attn_rkv --smooth --rkv_lambda 0.1 --model_name $model --limit 200
#python run_gsm8k.py --t_b 512 --mode attn --smooth --model_name $model

#python run_gsm8k.py --t_b 512 --mode small_range_cur --smooth --model_name $model --n_large 16 --output_dir tokens --limit 100
#python run_gsm8k.py --t_b 512 --mode small_range_sink_sample_attn --model_name $model --n_large 64 --output_dir tokens --limit 100
#python run_gsm8k.py --t_b 512 --mode small_range_sink_sample_attn --smooth --model_name $model --n_large 64 --output_dir tokens --limit 100 --temperature 0.6
#python run_gsm8k.py --t_b 512 --mode small_range_attn --smooth --model_name $model --n_large 16 --output_dir tokens --limit 200
#python run_gsm8k.py --t_b 512 --mode small_range_attn --smooth --model_name $model --n_large 32 --output_dir tokens --limit 200
#python run_gsm8k.py --t_b 512 --mode small_range_attn --smooth --model_name $model --n_large 64 --output_dir tokens --limit 200
#python run_gsm8k.py --t_b 512 --mode small_range_attn --smooth --model_name $model --n_large 128 --output_dir tokens
