from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Optional
import math
import torch
import torch.nn.functional as F

from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import (
    is_hqq_available,
    is_quanto_greater,
    is_torch_greater_or_equal,
    is_torchdynamo_compiling,
    logging,
)
from transformers.cache_utils import (
    CacheLayerMixin,
    DynamicSlidingWindowLayer,
    Cache,
)



_is_torch_greater_or_equal_than_2_7 = is_torch_greater_or_equal("2.7", accept_dev=True)


logger = logging.get_logger(__name__)




class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.
    """

    is_sliding = False

    def lazy_initialization(self, key_states: torch.Tensor, cache_query: bool = False):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        if cache_query:
            self.queries = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return -1

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be negative
        to remove `max_length` tokens.
        """
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        if self.get_seq_length() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]
            if hasattr(self, 'queries'): self.queries = self.queries[indices, ...]
            if hasattr(self, 'G'): self.G = self.G[indices, ...]



class QuantizedCache(Cache):
    """
    A quantizer cache similar to what is described in the
    [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for keys and values
    by applying quantization.
    The cache has two types of storage, one for original precision and one for the
    quantized cache. A `residual length` is set as a maximum capacity for the original precision cache. When the
    length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.
    The quantization is done per-channel with a set `q_group_size` for both keys and values, in contrast to what was
    described in the paper.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        backend (`str`):
            The quantization backend to use. One of `("quanto", "hqq").
        config (`PreTrainedConfig`):
            The config of the model for which this Cache will be used.
        nbits (`int`, *optional*, defaults to 4):
            The number of bits for quantization.
        axis_key (`int`, *optional*, defaults to 0):
            The axis on which to quantize the keys.
        axis_value (`int`, *optional*, defaults to 0):
            The axis on which to quantize the values.
        q_group_size (`int`, *optional*, defaults to 64):
            Quantization is done per-channel according to a set `q_group_size` for both keys and values.
        residual_length (`int`, *optional*, defaults to 128):
            Maximum capacity for the original precision cache
    """

    def __init__(
        self,
        backend: str,
        config: PreTrainedConfig,
        kbits: int = 8,
        vbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
        **kwargs,
    ):

        layer_class = FakeQuantizedLayer
        config = config.get_text_config(decoder=True)
        layers = [
            layer_class(l, kbits, vbits, axis_key, axis_value, q_group_size, residual_length, **kwargs)
            for l in range(config.num_hidden_layers)
        ]
        super().__init__(layers=layers)


class FakeQuantizedLayer(DynamicLayer):
    def __init__(
        self,
        layer_idx: int,
        kbits: int = 8,
        vbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
        verbose: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.kbits = kbits
        self.vbits = vbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.verbose = verbose
        self.cumulative_length = 0
        self.start = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.
        """
        is_decode = (key_states.shape[-2] == 1)
        self.cumulative_length += key_states.shape[-2]

        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

        batch_size, n_heads, kv_len, head_dim = self.values.shape
        if kv_len > self.residual_length:
            end = kv_len - self.residual_length
            if self.vbits < 16:
                vg = self.values[:, :, self.start: end].contiguous()
                dequant_values = self._pseudo_quantize_tensor(vg, self.vbits, axis=self.axis_value, assymetric=True)
                self.values[:, :, self.start: end] = dequant_values

            if self.kbits < 16:
                kg = self.keys[:, :, self.start: end].contiguous()
                dequant_keys = self._pseudo_quantize_tensor(kg, self.kbits, axis=self.axis_key, assymetric=True)
                self.keys[:, :, self.start: end] = dequant_keys

            if self.verbose and self.layer_idx == 0:
                print(kv_len, f'[{self.start}, {end}]', self.values[:, :, self.start: end].shape)
            self.start = end

        if not is_decode:
            return key_states, value_states
        else:
            return self.keys, self.values

    def _pseudo_quantize_tensor(self, tensor, nbits, axis, assymetric):
        org_shape = tensor.shape
        assert org_shape[-1] % self.q_group_size == 0
        tensor = tensor.reshape(-1, self.q_group_size) # TODO: axis
        if assymetric:
            max_val = tensor.amax(dim=1, keepdim=True)
            min_val = tensor.amin(dim=1, keepdim=True)
            max_int = 2**nbits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            max_val = tensor.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (nbits - 1) - 1
            min_int = -(2 ** (nbits - 1))
            scales = max_val / max_int
            zeros = 0

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(tensor).sum() == 0

        x_quant = torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int)
        x_dequant = (
            x_quant - zeros
        ) * scales
        assert torch.isnan(x_dequant).sum() == 0

        x_dequant = x_dequant.reshape(org_shape)
        return x_dequant

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length



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

        if verbose and layer_idx == 0:
            print(
                f'Eviction Strategy: {self.MODE}, Budget: {self.token_budget}, Buffer: {self.residual_length}, '
                f'RKV_lambda: {self.rkv_lambda}, Smooth: {self.smooth}'
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        def select_remaining_based_on_scores(ids_to_keep, scores, n_remain):
            # mask ids_to_keep to avoid repetive selection
            scores.scatter_(dim=-1, index=ids_to_keep, value=float('-inf'))
            ids_remain = scores.topk(n_remain, dim=-1, largest=True).indices
            ids_to_keep = torch.cat([ids_to_keep, ids_remain], dim=-1)
            return ids_to_keep

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
            scores = attn_weights.mean(-2) # avg over q_len (window_size)
            # Appy smoothing for every head independently
            # For each position in the sequence, replace its value with the average of neighboring positions
            if self.smooth:
                orig_shape = scores.shape # [batch_size, kv_heads, num_kv_groups, kv_len-window_size]
                scores = scores.view(batch_size, q_heads, -1)
                kernel_size = 5
                scores = F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
                scores = scores.view(orig_shape)
            scores = scores.mean(2) # avg heads in the same q_group
            return scores

        def cal_redundancy(keys):
            # https://github.com/Zefan-Cai/R-KV/blob/main/HuggingFace/rkv/utils.py#L42
            # cosine similarity: [bs, num_heads, len, len]
            k_norm = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
            similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))

            # zero diagonal (self-similarity)
            diag = torch.eye(keys.size(2), device=self.device, dtype=torch.bool)
            similarity_cos.masked_fill_(diag.unsqueeze(0).unsqueeze(0), 0.0)
            # Note: we don't need to nullify the β most recent similar tokens bc they're in the window
            # This speedup the RKV method
            return similarity_cos.mean(dim=2).softmax(dim=-1) # [bs, n_heads, len]
        
        def multinomial_sample(scores, K):
            orig_shape = scores.shape
            flat_scores = scores.view(-1, orig_shape[-1]) # [bs*n_heads, len]
            ids_to_keep = torch.multinomial(flat_scores, K, replacement=False)
            ids_to_keep = ids_to_keep.view(*orig_shape[:-1], K)
            return ids_to_keep

        self.cumulative_length += key_states.shape[-2]
        is_decode = (key_states.shape[-2] == 1)
        cache_q = ('attn' in self.MODE)

        # Initialization with empty tensor
        if not self.is_initialized:
            self.lazy_initialization(key_states, cache_q)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        if cache_q:
            self.queries = torch.cat(
                [self.queries, cache_kwargs['query_states']], dim=-2
            )[:, :, -self.residual_length:, :]

        batch_size, n_heads, kv_len, head_dim = self.values.shape

        if self.MODE == 'cur_fixed_gauss' and not hasattr(self, 'G'):
            r = 20
            # batch_size dim to ensure that different samples use a different G to avoid biased sampling
            # empirically, this yields ~2% improvements in acc
            G = torch.randn(batch_size, 1, head_dim, r, device=self.device) / math.sqrt(r)
            self.G = G.to(self.dtype)
            
        if is_decode and kv_len > self.token_budget and kv_len % self.residual_length == 0:
            v_candidates = self.values[:, :, :-self.residual_length]
            k_candidates = self.keys[:, :, :-self.residual_length]
            K = self.token_budget - self.residual_length
            # Several methods use v_magnitude as the importance scores for eviction
            v_magnitude = None
            if 'range' in self.MODE:
                v_magnitude = (v_candidates.amax(-1) - v_candidates.amin(-1))
            elif 'absmax' in self.MODE:
                v_magnitude = v_candidates.abs().amax(dim=-1)
            elif 'var' in self.MODE:
                v_magnitude = v_candidates.var(dim=-1)
            elif 'l2' in self.MODE:
                v_magnitude = v_candidates.norm(dim=-1)
            # Take care of attention sink
            if v_magnitude is not None:
                if 'sink' in self.MODE:
                    v_magnitude[..., :self.n_sink] = float('inf')
                assert v_magnitude.shape == (batch_size, n_heads, kv_len - self.residual_length)
            
            # Several methods use attention as the importance scores for eviction
            if 'attn' in self.MODE:
                attn_scores = compute_attention_scores(self.queries, k_candidates)
                if 'rkv' in self.MODE: # balance attention score and redundancy score
                    similarities = cal_redundancy(k_candidates)
                    attn_scores = attn_scores * self.rkv_lambda - similarities * (1 - self.rkv_lambda)

            # Select `ids_to_keep`
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
                ids_large = v_magnitude.topk(self.n_large, dim=-1, largest=True).indices # keep large
                if 'sample' in self.MODE:
                    attn_scores.scatter_(dim=-1, index=ids_large, value=0.0)
                    ids_attn = multinomial_sample(attn_scores, K-self.n_large)
                    ids_to_keep = torch.cat([ids_large, ids_attn], dim=-1)
                else:
                    ids_to_keep = select_remaining_based_on_scores(ids_large, attn_scores, K-self.n_large)

            elif self.MODE  == 'attn_random':
                random_scores = torch.rand(batch_size, n_heads, v_candidates.size(2), device=self.device)
                ids_large = random_scores.topk(self.n_large, dim=-1, largest=True).indices
                ids_to_keep = select_remaining_based_on_scores(ids_large, attn_scores, K-self.n_large)

            elif self.MODE in ['evict_large_range_random', 'keep_sink_large_range_random']:
                ids_large = v_magnitude.topk(self.n_large, dim=-1, largest=True).indices
                random_scores = torch.rand(batch_size, n_heads, v_candidates.size(2), device=self.device)
                if self.MODE == 'evict_large_range_random':
                    random_scores.scatter_(dim=-1, index=ids_large, value=float('-inf'))
                    ids_to_keep = random_scores.topk(K, dim=-1, largest=True).indices
                else:
                    ids_to_keep = select_remaining_based_on_scores(ids_large, random_scores, K-self.n_large)

            elif 'cur' in self.MODE:
                # https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/cur_press.py # Algorithm 1
                if hasattr(self, 'G'):
                    G = self.G
                else: # re-sample
                    r = 20
                    G = torch.randn(batch_size, 1, head_dim, r, device=self.device) / math.sqrt(r)
                    G = G.to(k_candidates.dtype)
                keys = k_candidates @ G   # [bs, n_heads, len, r]
                values = v_candidates @ G
                k2 = (keys**2).sum(dim=-1)
                v2 = (values**2).sum(dim=-1)

                scores = k2 * v2
                scores /= scores.sum(dim=-1, keepdim=True) # [bs, n_heads, len]
                
                scores[:, :, : self.n_sink] = 1.0
                ids_to_keep = scores.topk(K, dim=-1, largest=True).indices
            else:
                raise NotImplementedError(f"{self.MODE} not matched in cache_utils.py!")

            ## Eviction; keep `ids_to_keep` and lastest cache
            assert ids_to_keep.size(-1) == self.token_budget - self.residual_length
            ids_to_keep = ids_to_keep[..., None].expand(-1, -1, -1, head_dim)

            v_compress = torch.gather(v_candidates, dim=2, index=ids_to_keep)
            k_compress = torch.gather(k_candidates, dim=2, index=ids_to_keep)

            ## Update cache
            self.values = torch.cat([v_compress, self.values[:, :, -self.residual_length:]], dim=2)
            self.keys = torch.cat([k_compress, self.keys[:, :, -self.residual_length:]], dim=2)
            if self.verbose and self.layer_idx == 0:
                print(kv_len, '->', self.values.size(2))

        return self.keys, self.values


    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        # this yields a longer attention_mask but will be handled in eager attention via truncation
        # using cumulative_length ensures that the position_embeddings takes the cumulative position_ids
        return self.cumulative_length

