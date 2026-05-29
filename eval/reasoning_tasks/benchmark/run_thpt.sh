python benchmark/throughput.py --token_budget 2048 --output_len 16384 --methods "dense,attn_rkv,range_sink_sample_attn,cur_resample_gauss"
python benchmark/throughput.py --token_budget 4096 --output_len 16384 --methods "attn_rkv,range_sink_sample_attn,cur_resample_gauss"
python benchmark/throughput.py --token_budget 6144 --output_len 16384 --methods "attn_rkv,range_sink_sample_attn,cur_resample_gauss"

python benchmark/throughput.py --token_budget 2048 --output_len 32768 --methods "attn_rkv,range_sink_sample_attn,cur_resample_gauss" # dense OOM
python benchmark/throughput.py --token_budget 4096 --output_len 32768 --methods "attn_rkv,range_sink_sample_attn,cur_resample_gauss"
python benchmark/throughput.py --token_budget 6144 --output_len 32768 --methods "attn_rkv,range_sink_sample_attn,cur_resample_gauss"
