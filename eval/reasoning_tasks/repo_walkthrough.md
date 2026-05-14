# Repository Walkthrough


### Files to check
- `parallel_run_hf.py` — orchestrates multi-GPU parallel evaluation by launching sharded `eval_hf.py` subprocesses and managing GPU assignment via a job queue
- `eval_hf.py` — main evaluation entry point: loads the model, runs generation on a data shard, and computes overall KV-cache sparsity
- `modified/transformers/modify_forward.py` — monkey-patches the Qwen3 attention `forward` to pass `query_states` into `cache.update()`, enabling attention-based eviction
- `generation_utils.py` — custom autoregressive generation loop with dynamic batch shrinking as sequences finish, keeping custom KV caches consistent
- `summary_results.py` — aggregates sharded output files, computes quantile sparsity, and grades correctness across all eval runs


### Things to ignore
- `QuantizedCache` and anything related to quantization

### Evaluation Tasks
- **AIME25/26**, **MATH-500**, **GPQA-Diamond**, **HMMT**, **LiveCodeBench**
- Data in `data/`; multi-GPU parallel evaluation via `parallel_run_hf.py`

---

### Core: KV Cache Compression (`modified/transformers/cache_utils.py`)

**EvictCache** — drops tokens based on importance scores, with these strategies:

| Mode | Description |
|---|---|
| `absmax` | Keep tokens with largest absolute value magnitude |
| `var` | Keep tokens with highest value variance |
| `l2` | Keep tokens with highest L2 norm |
| `attn` | Keep tokens with highest attention scores |
| `attn_sample` | Sample tokens proportional to attention scores |
| `attn_rkv` | Blend attention scores + cosine similarity redundancy (`rkv_lambda` controls balance) |
| `small_range_sink_sample_attn` | Hybrid: value magnitude + sink tokens + sampled attention (with `--smooth`) |
| `cur_fixed_gauss` | CUR decomposition with fixed Gaussian projection |
| `cur_resample_gauss` | CUR decomposition with resampled Gaussian projection |

Common parameters: `token_budget` (KV cache size K), `residual_length` (buffer size B; recent tokens always in the buffer), `n_large` (the number of large-magnitude values preallocated in the cache).


Modified attention layers in `modify_forward.py` wrap Qwen3 to use either cache type.

---

### Generation & Evaluation
- `generation_utils.py`: Custom generation loop with **dynamic batch shrinking** as sequences finish; manages cache updates per step
- `eval_hf.py`: Main eval entry point; computes overall sparsity (activated blocks / total)
- `run_gsm8k.py`: Gsm8k-only; used for debug

---

### Results Pipeline
1. Outputs go to `<task>/<model>/<eviction_mode>/<config>/run_0/{completions.jsonl, summary.txt}`
2. `summary_results.py` aggregates sharded outputs, computes quantile sparsity
3. `plot_results.py` generates bar charts comparing methods vs. full KV baseline


### Modified Attention Layers (`modify_forward.py`)

`modify_forward.py` is a **monkey-patch module** — it replaces the `forward` method of the model's attention class at runtime

### What it does

**`wrap_evict_attn_forward(model_name_or_path)`** is called once at model load time. It defines a custom `forward` for the target model and swaps it in:

- `"Qwen3"` → patches `Qwen3Attention.forward`

### Why it's needed

The standard HuggingFace attention `forward` calls `past_key_values.update(k, v, layer_idx)` with only basic kwargs. The custom caches (`EvictCache`, `QuantizedCache`) need an **extra kwarg — `query_states`** — passed into `.update()` so they can compute attention-based importance scores for eviction decisions at cache update time. The stock HuggingFace code doesn't pass this, so the forward must be replaced.

```python
# The critical difference — query_states added to cache_kwargs:
cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "query_states": query_states}
key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
```

---

### Custom Generation Loop (`generation_utils.py`)

`batch_exist_generate` is a **custom autoregressive generation loop** that replaces HuggingFace's built-in `.generate()`. The key reason it exists: HuggingFace's generation doesn't natively support dynamically shrinking the batch when sequences finish, which is required to keep the custom KV caches (EvictCache/QuantizedCache) consistent.

### Initialization

Based on `cache_implementation` kwarg, it creates one of three caches:
- `EvictCache` — with eviction config
- `DynamicCache` — plain HuggingFace cache (baseline / dense)


A **`cur_to_orig`** index tensor tracks the mapping from the current (active) batch positions back to the original batch positions — this is the core bookkeeping for batch shrinking.

### Per-step loop

Each step:

1. **Forward pass** — runs the model on only the active sequences (`cur_input` is a single token per active sequence after the prefill), passing `current_cache`
2. **Token sampling** — top-p sampling with temperature
3. **Cache update** — `current_cache = outputs.past_key_values` (the cache was already updated inside the model's forward via `.update()`)
4. **Write tokens back to full batch** — `new_tokens_full` is initialized to `eos_token_id` for all positions, then active positions are filled via `cur_to_orig` indexing
5. **Detect newly finished sequences** — marks EOS hits in the global `finished` tensor
6. **Batch shrink** — if any sequence just finished:
   - Computes `active_indices_local`: indices within the *current cache* of still-active sequences
   - Calls `current_cache.batch_select_indices(active_indices_local)` to drop finished sequences from the cache
   - Shrinks `attention_mask` and updates `cur_to_orig` accordingly

### Return

Returns `(generated, counter)` where `counter` is the eviction counter from the first cache layer (how many tokens were evicted total), used downstream to compute sparsity metrics.
