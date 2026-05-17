from typing import Any, Optional
import math
import torch
import torch.nn.functional as F

from transformers.configuration_utils import PreTrainedConfig
from transformers.cache_utils import Cache, DynamicLayer


class EvictCache(Cache):
    def __init__(
        self,
        config: PreTrainedConfig,
        token_budget: int,
        residual_length: int,
        eviction_mode: str,
        **kwargs,
    ):

        config = config.get_text_config(decoder=True)
        layers = [
            EvictLayer(l, token_budget, residual_length, eviction_mode, **kwargs)
            for l in range(config.num_hidden_layers)
        ]
        super().__init__(layers=layers)


class EvictLayer(DynamicLayer):
    """
    KV cache layer backed by pre-allocated [B, max_len, Hkv, D] storage for `flash_attn_with_kvcache`
    """

    def __init__(
        self,
        layer_idx: int,
        token_budget: int,
        residual_length: int,
        eviction_mode: str,
        rkv_lambda: float,
        smooth: bool,
        verbose: bool,
        n_large: int = 200,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.token_budget = token_budget
        self.residual_length = residual_length
        self.rkv_lambda = rkv_lambda
        self.smooth = smooth
        self.verbose = verbose
        self.MODE = eviction_mode
        self.n_large = n_large
        self.temperature = temperature
        self.n_sink = 4
        self.cumulative_length = 0
        self._q_write_pos = 0  # circular buffer head for self.queries (attn modes only)

        if verbose and layer_idx == 0:
            print(
                f'Eviction Strategy: {self.MODE}, Budget: {self.token_budget}, Buffer: {self.residual_length}, '
                f'RKV_lambda: {self.rkv_lambda}, Smooth: {self.smooth}'
            )

    def lazy_initialization(self, key_states: torch.Tensor, value_states: Optional[torch.Tensor] = None) -> None:
        batch_size, prompt_len, num_kv_heads, head_dim = key_states.shape
        self.dtype = key_states.dtype
        self.device = key_states.device
        max_cache_len = max(prompt_len, self.token_budget) + self.residual_length
        self.k_cache = torch.zeros(
            batch_size, max_cache_len, num_kv_heads, head_dim,
            dtype=self.dtype, device=self.device,
        )
        self.v_cache = torch.zeros(
            batch_size, max_cache_len, num_kv_heads, head_dim,
            dtype=self.dtype, device=self.device,
        )
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self.is_initialized = True
        if self.verbose and self.layer_idx == 0:
            print(f'self.k_cache shape: {self.k_cache.shape}')

    def prepare(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (k_cache, v_cache, cache_seqlens) for flash_attn_with_kvcache."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        return self.k_cache, self.v_cache, self.cache_seqlens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "EvictLayer is driven by prepare() + post_attention_update() in the "
            "patched attention forward; the standard Cache.update() path is bypassed."
        )

    def update_queries(self, query_states: torch.Tensor) -> None:
        """
        Cache query_states into the circular [B, Hq, residual_length, D] buffer.
        Called *after* RoPE and *before* the flash-attn transpose, so query_states is in the post-RoPE
        """
        if 'attn' not in self.MODE:
            return

        q_len = query_states.shape[2]
        if q_len == 1: # decode: round-robin single-slot copy into the existing buffer.
            self.queries[:, :, self._q_write_pos:self._q_write_pos + 1, :].copy_(query_states)
            self._q_write_pos = (self._q_write_pos + 1) % self.residual_length
        else:          # prefill (first call): allocate the buffer and fill from last `n` queries.
            B, Hq, _, D = query_states.shape
            self.queries = torch.zeros(B, Hq, self.residual_length, D,
                                       dtype=query_states.dtype, device=query_states.device)
            n = min(q_len, self.residual_length)
            self.queries[:, :, :n, :].copy_(query_states[:, :, -n:, :])
            self._q_write_pos = n % self.residual_length

    def post_attention_update(self, n_new: int) -> None:
        """
        Called after `flash_attn_with_kvcache` has written n_new K/V into the cache.
        Advances cache_seqlens and periodically runs eviction.
        """
        self.cache_seqlens += n_new
        self.cumulative_length += n_new

        is_decode = (n_new == 1)
        cur_len = int(self.cache_seqlens[0].item())  # same length across the batch
        if is_decode and cur_len > self.token_budget and cur_len % self.residual_length == 0:
            self._run_eviction(cur_len)

    def _run_eviction(self, cur_len: int) -> None:
        def select_remaining_based_on_scores(ids_to_keep, scores, n_remain):
            scores.scatter_(dim=-1, index=ids_to_keep, value=float('-inf'))
            ids_remain = scores.topk(n_remain, dim=-1, largest=True).indices
            return torch.cat([ids_to_keep, ids_remain], dim=-1)

        def compute_attention_scores(query_cache, keys):
            # https://github.com/Zefan-Cai/R-KV/blob/main/HuggingFace/rkv/utils.py#L8
            batch_size, q_heads, q_len, head_dim = query_cache.shape
            kv_heads = keys.shape[1]
            num_kv_groups = q_heads // kv_heads
            assert num_kv_groups > 1, "not GQA"

            query_cache = query_cache.view(batch_size, kv_heads, num_kv_groups, q_len, head_dim)
            keys = keys.unsqueeze(2)

            # shape: [batch_size, kv_heads, num_kv_groups, q_len, kv_len]
            attn_weights = torch.matmul(
                query_cache, keys.transpose(3, 4)
            ) / (math.sqrt(head_dim) * self.temperature)

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_cache.dtype)
            scores = attn_weights.mean(-2)  # avg over q_len (window_size)
            if self.smooth:
                orig_shape = scores.shape
                scores = scores.view(batch_size, q_heads, -1)
                kernel_size = 5
                scores = F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
                scores = scores.view(orig_shape)
            scores = scores.mean(2)  # avg heads in the same q_group
            return scores

        def cal_redundancy(keys):
            # https://github.com/Zefan-Cai/R-KV/blob/main/HuggingFace/rkv/utils.py#L42
            k_norm = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
            cos_similarity = torch.matmul(k_norm, k_norm.transpose(-1, -2))
            # zero diagonal (self-similarity)
            diag = torch.eye(keys.size(2), device=self.device, dtype=torch.bool)
            cos_similarity.masked_fill_(diag.unsqueeze(0).unsqueeze(0), 0.0)
            return cos_similarity.mean(dim=2).softmax(dim=-1)

        def multinomial_sample(scores, K):
            orig_shape = scores.shape
            flat_scores = scores.view(-1, orig_shape[-1])
            ids_to_keep = torch.multinomial(flat_scores, K, replacement=False)
            return ids_to_keep.view(*orig_shape[:-1], K)

        residual = self.residual_length
        K = self.token_budget - residual

        k_candidates = self.k_cache[:, :cur_len - residual].transpose(1, 2)  # [B, Hkv, cur_len - residual, D]
        v_candidates = self.v_cache[:, :cur_len - residual].transpose(1, 2)
        batch_size, n_heads, _, head_dim = v_candidates.shape

        # Several methods use v_magnitude as the importance scores for eviction
        v_magnitude = None
        if 'range' in self.MODE:
            v_magnitude = (v_candidates.amax(-1) - v_candidates.amin(-1))
        elif 'var' in self.MODE:
            v_magnitude = v_candidates.var(dim=-1)
        elif 'l2' in self.MODE:
            v_magnitude = v_candidates.norm(dim=-1)
        if v_magnitude is not None and 'sink' in self.MODE:
            v_magnitude[..., :self.n_sink] = float('inf')
        
        # Several methods use attention as the importance scores for eviction
        if 'attn' in self.MODE:
            attn_scores = compute_attention_scores(self.queries, k_candidates)
            if 'rkv' in self.MODE:
                similarities = cal_redundancy(k_candidates)
                attn_scores = attn_scores * self.rkv_lambda - similarities * (1 - self.rkv_lambda)

        if self.MODE in ['small_range', 'small_range_sink', 'large_range']:
            ids_to_keep = v_magnitude.topk(K, dim=-1, largest=('small' in self.MODE)).indices

        elif self.MODE in ['attn', 'attn_rkv', 'attn_range']:
            if self.MODE == 'attn_range':
                attn_scores = attn_scores * v_magnitude
            ids_to_keep = attn_scores.topk(K, dim=-1, largest=True).indices

        elif self.MODE == 'attn_sample':
            ids_to_keep = multinomial_sample(attn_scores, K)

        elif self.MODE in ['small_range_attn', 'small_range_attn_rkv',
            'small_range_sample_attn', 'small_range_sink_sample_attn',
            'absmax_sink_sample_attn', 'var_sink_sample_attn', 'l2_sink_sample_attn']:
            ids_large = v_magnitude.topk(self.n_large, dim=-1, largest=True).indices
            if 'sample' in self.MODE:
                attn_scores.scatter_(dim=-1, index=ids_large, value=0.0)
                ids_attn = multinomial_sample(attn_scores, K - self.n_large)
                ids_to_keep = torch.cat([ids_large, ids_attn], dim=-1)
            else:
                ids_to_keep = select_remaining_based_on_scores(ids_large, attn_scores, K - self.n_large)

        elif self.MODE == 'attn_random':
            random_scores = torch.rand(batch_size, n_heads, v_candidates.size(2), device=self.device)
            ids_large = random_scores.topk(self.n_large, dim=-1, largest=True).indices
            ids_to_keep = select_remaining_based_on_scores(ids_large, attn_scores, K - self.n_large)

        elif self.MODE in ['evict_large_range_random', 'keep_sink_large_range_random']:
            ids_large = v_magnitude.topk(self.n_large, dim=-1, largest=True).indices
            random_scores = torch.rand(batch_size, n_heads, v_candidates.size(2), device=self.device)
            if self.MODE == 'evict_large_range_random':
                random_scores.scatter_(dim=-1, index=ids_large, value=float('-inf'))
                ids_to_keep = random_scores.topk(K, dim=-1, largest=True).indices
            else:
                ids_to_keep = select_remaining_based_on_scores(ids_large, random_scores, K - self.n_large)

        elif 'cur' in self.MODE:
            # https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/cur_press.py
            if self.MODE == 'cur_fixed_gauss':
                if not hasattr(self, 'G'):
                    r = 20
                    # per-row G to avoid biased sampling across the batch (1-2% acc gain empirically)
                    self.G = (torch.randn(batch_size, 1, head_dim, r, device=self.device) / math.sqrt(r)).to(self.dtype)
                G = self.G
            else:  # cur_resample_gauss
                r = 20
                G = (torch.randn(batch_size, 1, head_dim, r, device=self.device) / math.sqrt(r)).to(k_candidates.dtype)
            keys = k_candidates @ G
            values = v_candidates @ G
            k2 = (keys ** 2).sum(dim=-1)
            v2 = (values ** 2).sum(dim=-1)
            scores = k2 * v2
            scores /= scores.sum(dim=-1, keepdim=True)
            scores[:, :, :self.n_sink] = 1.0
            ids_to_keep = scores.topk(K, dim=-1, largest=True).indices

        else:
            raise NotImplementedError(f"{self.MODE} not matched in cache_utils.py!")

        assert ids_to_keep.size(-1) == K
        ids_to_keep = ids_to_keep.sort(dim=-1).values  # preserve positional order so sink tokens stay at the front
        ids_expand = ids_to_keep[..., None].expand(-1, -1, -1, head_dim)

        k_compress = torch.gather(k_candidates, dim=2, index=ids_expand)
        v_compress = torch.gather(v_candidates, dim=2, index=ids_expand)
        self.k_cache[:, :K].copy_(k_compress.transpose(1, 2))
        self.v_cache[:, :K].copy_(v_compress.transpose(1, 2))

        # Move residual block
        self.k_cache[:, K:self.token_budget].copy_(self.k_cache[:, cur_len - residual:cur_len])
        self.v_cache[:, K:self.token_budget].copy_(self.v_cache[:, cur_len - residual:cur_len])

        self.cache_seqlens.fill_(self.token_budget)

        if self.verbose and self.layer_idx == 0:
            print(cur_len, '->', self.token_budget)

    def get_seq_length(self) -> int:
        # use cumulative_length (not cur_len) for the position_embeddings/rope
        return self.cumulative_length

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if not self.is_initialized:
            return
        self.k_cache = self.k_cache[indices].contiguous()
        self.v_cache = self.v_cache[indices].contiguous()
        self.cache_seqlens = self.cache_seqlens[indices].contiguous()
        if hasattr(self, 'queries'):
            self.queries = self.queries[indices].contiguous()
        if hasattr(self, 'G'):
            self.G = self.G[indices].contiguous()

