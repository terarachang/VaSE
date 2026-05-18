# Repository Walkthrough

Brief orientation for the KV-cache compression eval pipeline.

### Tasks & data
AIME24/25/26, MATH-500, GPQA-Diamond, HMMT, LiveCodeBench. Data lives in `data/<task>/`.

### Files to know
- `parallel_run_hf.py` — multi-GPU orchestrator; launches sharded `eval_hf.py` subprocesses, then calls `summary_results.py` (skipped for livecodebench).
- `eval_hf.py` — loads model + tokenizer, runs generation on a shard, writes `completions[_shard<N>].jsonl`.
- `generation_utils.py` — `batch_exist_generate`: custom autoregressive loop with dynamic batch shrinking so EvictCache stays consistent when sequences finish.
- `modified/transformers/modify_forward.py` — monkey-patches `Qwen3Attention.forward` to cache `query_states`, enabling attention-based eviction.
- `modified/transformers/cache_utils.py` — `EvictCache` + per-layer `EvictLayer`; implements all eviction modes.
- `summary_results.py` — aggregates shards, grades math/livecodebench, writes `overall_summary.txt`.
- `aggregate_results.py <task>` — reads `overall_summary.txt` and prints a comparison table.
- `myscripts/<task>/eval_*.sh` — entry points; one script per method × task.

---

### KV-cache compression (`cache_utils.py`)

`EvictCache` drops tokens by importance. Modes selected via `--eviction_mode`:

| Mode | Importance signal |
|---|---|
| `absmax` / `var` / `l2` | Value magnitude / variance / L2 norm |
| `attn` | Top-k by attention score |
| `attn_sample` | Sample proportional to attention scores |
| `attn_rkv` | Attention blended with cosine-similarity redundancy (`--rkv_lambda`) |
| `range_sink_sample_attn` | Value range (max−min) + sink tokens + sampled attention (use with `--smooth`) |
| `cur_fixed_gauss` / `cur_resample_gauss` | CUR decomposition with fixed/resampled Gaussian projection |

Key params: `token_budget` (K, total cache size), `residual_length` (B, recent-token buffer), `n_large` (preallocated high-importance slots), `--smooth`, `--temperature`.

Constraint: `token_budget > residual_length` and `token_budget % residual_length == 0` (enforced in `init_evict_configs`).

---

### Why the attention patch exists

Stock HF calls `past_key_values.update(k, v, layer_idx, cache_kwargs)` without `query_states`. EvictCache needs it for attention-based importance, so `wrap_evict_attn_forward()` swaps in a forward that adds it:

```python
cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "query_states": query_states}
key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
```

---

### Why the custom generation loop exists

`batch_exist_generate` exists because HF `.generate()` can't shrink the batch mid-generation. A `cur_to_orig` index tensor maps active → original positions; on each EOS hit, `current_cache.batch_select_indices(active_indices_local)` drops finished rows from the cache. Returns `(generated, counter)` where `counter` is total evictions (used for sparsity).

The plain HF path (`model.generate()`) is still used when `--use_batch_exist` is not set; it works only with `DynamicCache`.

---

### Output layout

```
<output_dir>/<task_bs..._configdesc>/run_<i>/
    completions[_shard<N>].jsonl
    other_info[_shard<N>].json
    overall_summary.txt        # written by summary_results.py
```

Sharded runs (`--num_gpus > 1`) produce `_shard<N>` files; `summary_results.py` reassembles them. `get_ckpt_start_i()` counts existing completions and resumes interrupted runs.
