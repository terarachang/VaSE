import torch
from typing import Optional
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from flash_attn import flash_attn_with_kvcache


def wrap_evict_attn_forward(model_name_or_path):
    def qwen3_evict_attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]            # (B, L)
        hidden_shape = (*input_shape, -1, self.head_dim)  # for view to [B, L, H, D]

        # Project + per-head norm. Transpose to [B, H, L, D] for RoPE.
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        layer = past_key_values.layers[self.layer_idx]
        layer.update_queries(query_states)

        # flash-attn expects [B, L, H, D]
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        k_cache, v_cache, cache_seqlens = layer.prepare(key_states, value_states)

        # Fused append-and-attend: writes new key_states/value_states at offset cache_seqlens
        # in place, then attends over [0, cache_seqlens + L_new) per row.
        attn_output = flash_attn_with_kvcache(
            query_states, k_cache, v_cache,
            k=key_states, v=value_states,
            cache_seqlens=cache_seqlens,
            causal=True,
            softmax_scale=self.scaling,
            window_size=(-1, -1), # assert self.sliding_window is None
        )

        # Advance cache_seqlens; may conduct eviction when the cache is full
        layer.post_attention_update(key_states.shape[1])

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    if "Qwen3" in model_name_or_path:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb
        Qwen3Attention.forward = qwen3_evict_attn_forward
    else:
        raise ValueError(f"Unknown evict attn implementation: {model_name_or_path}")
