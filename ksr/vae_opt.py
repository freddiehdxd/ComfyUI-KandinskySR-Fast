"""VAE speed-up for the FLUX AutoencoderKL used by Kandinsky ImageSR."""

from __future__ import annotations

import torch
from torch import nn


class MatmulAttnProcessor:
    """Single-head spatial self-attention (the VAE mid block) as two GEMMs and a softmax.

    For 16384 tokens with one 512-wide head, PyTorch's SDPA has no fast kernel on Windows (no FlashAttention,
    and cuDNN rejects the head width), so it uses the memory-efficient kernel: 8.8 ms on an RTX 5090. Explicit
    matmuls take 4.0 ms and give the same result. The score matrix is tokens x tokens: 512 MB at 16 bits for
    one 1024px tile.
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        assert attn.heads == 1 and encoder_hidden_states is None and attention_mask is None
        residual = hidden_states
        batch, channels, height, width = hidden_states.shape
        tokens = hidden_states.view(batch, channels, height * width)
        if attn.group_norm is not None:
            tokens = attn.group_norm(tokens)
        tokens = tokens.transpose(1, 2)
        query, key, value = attn.to_q(tokens), attn.to_k(tokens), attn.to_v(tokens)
        scores = torch.matmul(query * (query.shape[-1] ** -0.5), key.transpose(1, 2))
        tokens = torch.matmul(scores.softmax(dim=-1), value)
        tokens = attn.to_out[1](attn.to_out[0](tokens))
        hidden_states = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def use_matmul_attention(vae: nn.Module) -> int:
    from diffusers.models.attention_processor import Attention

    count = 0
    for module in vae.modules():
        if isinstance(module, Attention) and module.heads == 1:
            module.set_processor(MatmulAttnProcessor())
            count += 1
    return count
