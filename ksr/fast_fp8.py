"""fp8 (e4m3) linear layers for the Kandinsky ImageSR DiT, executed with `torch._scaled_mm`.

The weights are quantized in memory from the original bf16 checkpoint (one scale per layer, the only mode
RTX 50-series cards accept), so no separate fp8 file and no quantization library is needed.

  * `FastFP8Linear`: activation scale from one `aminmax` pass, quantization = one bf16 multiply + one cast,
    then an fp8 GEMM, which is about 2x faster than the bf16 GEMM on cards with fp8 tensor cores.
  * `FastFP8AttnProcessor`: to_query / to_key / to_value read the same tensor, so it is quantized once for all three.
"""

from __future__ import annotations

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_kandinsky import Kandinsky5Attention, Kandinsky5AttnProcessor
from torch import nn

FP8 = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8).max)  # 448.0
HEADROOM = 0.99  # keeps bf16 rounding of the scaled activation below FP8_MAX, so no clamp pass is needed

# Layers that stay in their original precision: the tiny patch embedding and output projection, and the
# timestep-modulation layers the model itself keeps in fp32.
SKIP = ("visual_embeddings", "out_layer.out_layer", "modulation")


def fp8_supported(device: torch.device | str) -> bool:
    """True if this GPU runs an fp8 GEMM with per-tensor scales."""
    try:
        a = torch.ones(16, 16, device=device).to(FP8)
        one = torch.ones(1, 1, device=device)
        torch._scaled_mm(a, a.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        return True
    except Exception:  # noqa: BLE001 - any failure means "use bf16"
        return False


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp32 [out, in] -> (fp8 [out, in], fp32 [1, 1] scale) with weight ~= fp8 * scale."""
    hp = weight.detach().float()
    scale = hp.abs().amax().clamp_min(1e-12) / FP8_MAX
    return (hp / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale.reshape(1, 1)


def quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., K] bf16 -> ([rows, K] fp8, [1, 1] fp32 scale) with x ~= fp8 * scale."""
    flat = x.reshape(-1, x.shape[-1])
    if flat.dtype != torch.bfloat16:
        flat = flat.to(torch.bfloat16)
    low, high = torch.aminmax(flat)
    amax = torch.maximum(high, -low).float().clamp_min(1e-12)
    inverse = (FP8_MAX * HEADROOM / amax).to(torch.bfloat16)
    scale = (1.0 / inverse.float()).reshape(1, 1)  # exact inverse of the bf16 factor actually applied
    return (flat * inverse).to(FP8), scale


class FastFP8Linear(nn.Module):
    def __init__(self, qweight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        assert qweight.dtype == FP8 and weight_scale.numel() == 1, "expects per-tensor scaled fp8 weights"
        self.out_features, self.in_features = qweight.shape
        self.register_buffer("qweight", qweight.contiguous())  # [out, in] row-major -> .t() is column-major
        self.register_buffer("weight_scale", weight_scale.float().reshape(1, 1))
        self.register_buffer("bias", None if bias is None else bias.to(torch.bfloat16))

    def forward_quantized(self, quantized: torch.Tensor, scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = torch._scaled_mm(
            quantized,
            self.qweight.t(),
            scale_a=scale,
            scale_b=self.weight_scale,
            bias=self.bias,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        return out.reshape(*shape[:-1], self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized, scale = quantize_activation(x)
        return self.forward_quantized(quantized, scale, x.shape)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, weight=fp8_e4m3fn"


class FastFP8AttnProcessor(Kandinsky5AttnProcessor):
    """Kandinsky5AttnProcessor for self-attention with one shared activation quantization for q, k and v."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, rotary_emb=None, sparse_params=None):
        if encoder_hidden_states is not None or sparse_params is not None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, rotary_emb, sparse_params)

        quantized, scale = quantize_activation(hidden_states)
        shape = hidden_states.shape
        query = attn.to_query.forward_quantized(quantized, scale, shape)
        key = attn.to_key.forward_quantized(quantized, scale, shape)
        value = attn.to_value.forward_quantized(quantized, scale, shape)

        shape = query.shape[:-1]
        query = query.reshape(*shape, attn.num_heads, -1)
        key = key.reshape(*shape, attn.num_heads, -1)
        value = value.reshape(*shape, attn.num_heads, -1)

        query = attn.query_norm(query.float()).type_as(query)
        key = attn.key_norm(key.float()).type_as(key)

        def apply_rotary(x, rope):
            x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
            x_out = (rope * x_).sum(dim=-1)
            return x_out.reshape(*x.shape).to(torch.bfloat16)

        if rotary_emb is not None:
            query = apply_rotary(query, rotary_emb).type_as(query)
            key = apply_rotary(key, rotary_emb).type_as(key)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        return attn.out_layer(hidden_states.flatten(-2, -1))


def convert_to_fp8(model: nn.Module, device: torch.device | str) -> int:
    """Replace the DiT's bf16 nn.Linear layers with FastFP8Linear on `device`, in place. Returns the number converted.

    Call this while the model is still on the CPU: each layer is moved to the GPU, quantized there and its bf16
    original dropped, so the full bf16 model never has to fit in VRAM.
    """
    targets = []
    for parent_name, parent in model.named_modules():
        for name, child in parent.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if not isinstance(child, nn.Linear) or child.weight.dtype != torch.bfloat16:
                continue
            if not any(skip in full_name for skip in SKIP):
                targets.append((parent, name))
    for parent, name in targets:
        child = getattr(parent, name)
        qweight, scale = quantize_weight(child.weight.to(device))
        bias = child.bias.detach().to(device) if child.bias is not None else None
        setattr(parent, name, FastFP8Linear(qweight, scale, bias))
        del child
    use_fp8_attention(model)
    return len(targets)


def use_fp8_attention(model: nn.Module) -> None:
    """Give every attention block whose q/k/v layers are fp8 the processor that quantizes their input once."""
    for module in model.modules():
        if isinstance(module, Kandinsky5Attention) and all(
            isinstance(getattr(module, name), FastFP8Linear) for name in ("to_query", "to_key", "to_value")
        ):
            module.set_processor(FastFP8AttnProcessor())
