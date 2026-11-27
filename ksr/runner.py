"""Loading, optimizing and tiled one-step inference for Kandinsky ImageSR 2.0.

This module has no ComfyUI imports, so it can also be used from a plain Python script:

    from ksr.runner import load_model
    model = load_model(precision="fp8", compile=True)
    output = model.upscale(image_1x3xHxW_in_0_1, factor=4)
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from collections.abc import Callable

import torch
import torch.nn.functional as F

from .fast_fp8 import convert_to_fp8, fp8_supported
from .vae_opt import use_matmul_attention
from .vendor.baked_transformer import KandinskyImageSRBakedTransformer3DModel
from .vendor.tiling import boundary_aware_window, pad_to_min_tile, tile_positions

REPO = "kandinskylab/Kandinsky-ImageSR-2.0"
TILE = 1024  # output tile side; fixed by the baked model (128x128 latent)
FACTORS = (2, 4)
# Flow timestep per creativity level 0..8 (level 4 is the model's calibrated default), from the model's pipeline.
CREATIVITY_TIMESTEPS = {
    2: (0.07, 0.086, 0.118, 0.134, 0.15, 0.18, 0.21, 0.27, 0.30),
    4: (0.17, 0.1885, 0.207, 0.2255, 0.244, 0.278, 0.312, 0.346, 0.38),
}
GPU_ACCUMULATOR_LIMIT = 2 * 1024**3  # blend on the GPU unless the output canvas needs more than this


def _nearest_timestep_indices(self, timestep: torch.Tensor) -> torch.Tensor:
    """The model's timestep-table lookup without its data-dependent validity check (done in Python instead),
    which would otherwise split the compiled graph."""
    available = torch.tensor(self.baked_timesteps, device=timestep.device, dtype=torch.float32)
    return (timestep.float()[:, None] / 1000.0 - available[None]).abs().argmin(dim=1)


def cudnn_attention_supported(device: torch.device) -> bool:
    try:
        probe = torch.randn(1, 2, 64, 128, device=device, dtype=torch.bfloat16)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            F.scaled_dot_product_attention(probe, probe, probe)
        return True
    except Exception:  # noqa: BLE001
        return False


def tile_grid(height: int, width: int, factor: int, overlap: float) -> tuple[list[int], list[int]]:
    """Top and left positions of the low-resolution tiles for an input of this size."""
    lq_tile = TILE // factor
    stride = max(1, int(lq_tile * (1.0 - overlap)))
    return tile_positions(max(height, lq_tile), lq_tile, stride), tile_positions(max(width, lq_tile), lq_tile, stride)


def count_tiles(height: int, width: int, factor: int, overlap: float) -> int:
    tops, lefts = tile_grid(height, width, factor, overlap)
    return len(tops) * len(lefts)


class KandinskySR:
    def __init__(self, transformer, vae, device: torch.device, precision: str, log: Callable[[str], None]):
        self.transformer = transformer
        self.vae = vae
        self.device = device
        self.precision = precision
        self.log = log
        self.compiled = False
        self._eager = None
        self._compile_pending = False
        self.scaling_factor = float(vae.config.scaling_factor)
        self.baked_timesteps = tuple(float(value) for value in transformer.config.baked_timesteps)
        patch_t, patch_h, patch_w = (int(value) for value in transformer.config.patch_size)
        latent = int(transformer.config.baked_latent_size)
        self._rope_shape = (1 // patch_t or 1, latent // patch_h, latent // patch_w)

    # ---- optimizations -------------------------------------------------------------------------------------

    def enable_compile(self) -> bool:
        if importlib.util.find_spec("triton") is None:
            package = "triton-windows" if sys.platform == "win32" else "triton"
            self.log(f"torch.compile skipped: the '{package}' package is not installed (see the README).")
            return False
        self._eager = (self.transformer, self.vae.encoder, self.vae.decoder)
        self.transformer = torch.compile(self.transformer)
        self.vae.encoder = torch.compile(self.vae.encoder)
        self.vae.decoder = torch.compile(self.vae.decoder)
        self.compiled = True
        self._compile_pending = True
        return True

    def _disable_compile(self) -> None:
        self.transformer, self.vae.encoder, self.vae.decoder = self._eager
        self._eager = None
        self.compiled = False

    def to(self, device: torch.device | str) -> KandinskySR:
        device = torch.device(device)
        if device != self.device:
            self.transformer.to(device)
            self.vae.to(device)
            self.device = device
        return self

    # ---- inference -----------------------------------------------------------------------------------------

    def _infer_tile(self, pixels: torch.Tensor, timestep: float) -> torch.Tensor:
        """[1, 3, 1024, 1024] bicubic-upsampled tile in 0..1 -> restored tile in 0..1 (same math as the reference pipeline)."""
        device = self.device
        vae_dtype = next(self.vae.parameters()).dtype
        lq = self.vae.encode((pixels * 2.0 - 1.0).to(device=device, dtype=vae_dtype)).latent_dist.mode()
        lq = lq.float() * self.scaling_factor
        hidden_states = lq.permute(0, 2, 3, 1).unsqueeze(1).to(torch.bfloat16)
        rope_pos = tuple(torch.arange(size, device=device) for size in self._rope_shape)
        model_t = torch.full((lq.shape[0],), timestep * 1000.0, device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                timestep=model_t,
                pooled_projections=None,
                visual_rope_pos=rope_pos,
                text_rope_pos=None,
                scale_factor=(1.0, 1.0, 1.0),
                return_dict=False,
            )
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction = prediction.squeeze(1).permute(0, 3, 1, 2).float()
        corrected = lq - timestep * prediction
        with torch.autocast(device_type="cuda", dtype=vae_dtype):
            decoded = self.vae.decode((corrected / self.scaling_factor).to(vae_dtype)).sample
        return (decoded.float().clamp(-1.0, 1.0) + 1.0) * 0.5

    def _run_tile(self, pixels: torch.Tensor, timestep: float) -> torch.Tensor:
        if not self._compile_pending:
            return self._infer_tile(pixels, timestep)
        self.log("compiling the model with torch.compile (one-time per session, usually 20-60 s)...")
        start = time.perf_counter()
        try:
            output = self._infer_tile(pixels, timestep)
            self.log(f"compiled in {time.perf_counter() - start:.0f} s")
        except Exception as error:  # noqa: BLE001 - fall back to the uncompiled model on any compile failure
            self._disable_compile()
            self.log(f"torch.compile failed, continuing without it: {type(error).__name__}: {str(error)[:300]}")
            output = self._infer_tile(pixels, timestep)
        self._compile_pending = False
        return output

    @torch.no_grad()
    def upscale(
        self,
        image: torch.Tensor,
        factor: int = 4,
        creativity: int = 4,
        overlap: float = 0.25,
        on_tile: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """[1, 3, H, W] RGB in 0..1 -> [1, 3, H*factor, W*factor] RGB in 0..1 on the CPU."""
        if factor not in FACTORS:
            raise ValueError(f"factor must be one of {FACTORS}, got {factor}")
        if not 0 <= int(creativity) <= 8:
            raise ValueError(f"creativity must be 0..8, got {creativity}")
        if not 0.0 < overlap < 1.0:
            raise ValueError("overlap must be between 0 and 1")
        timestep = CREATIVITY_TIMESTEPS[factor][int(creativity)]
        if not any(abs(timestep - value) <= 1e-6 for value in self.baked_timesteps):
            raise ValueError(f"timestep {timestep} is not baked into this model")
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError(f"expected one RGB image shaped (1, 3, H, W), got {tuple(image.shape)}")

        device = self.device
        lq_tile = TILE // factor
        padded, original_h, original_w = pad_to_min_tile(image.detach().float().clamp(0.0, 1.0).cpu(), lq_tile)
        tops, lefts = tile_grid(padded.shape[-2], padded.shape[-1], factor, overlap)
        output_h, output_w = padded.shape[-2] * factor, padded.shape[-1] * factor
        blend_device = device if output_h * output_w * 16 <= GPU_ACCUMULATOR_LIMIT else torch.device("cpu")
        accumulator = torch.zeros(1, 3, output_h, output_w, device=blend_device)
        weights = torch.zeros(1, 1, output_h, output_w, device=blend_device)

        for row, top in enumerate(tops):
            for col, left in enumerate(lefts):
                tile = padded[:, :, top : top + lq_tile, left : left + lq_tile]
                tile = F.interpolate(tile, size=(TILE, TILE), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
                restored = self._run_tile(tile, timestep).to(blend_device)
                window = boundary_aware_window(
                    TILE,
                    TILE,
                    top=row == 0,
                    bottom=row == len(tops) - 1,
                    left=col == 0,
                    right=col == len(lefts) - 1,
                    device=blend_device,
                )[None, None]
                top_hr, left_hr = top * factor, left * factor
                accumulator[:, :, top_hr : top_hr + TILE, left_hr : left_hr + TILE] += restored * window
                weights[:, :, top_hr : top_hr + TILE, left_hr : left_hr + TILE] += window
                if on_tile is not None:
                    on_tile()

        result = accumulator / weights.clamp_min(1e-6)
        return result[:, :, : original_h * factor, : original_w * factor].clamp(0.0, 1.0).cpu()


def load_model(
    precision: str = "fp8",
    compile: bool = True,
    device: torch.device | str = "cuda",
    repo: str = REPO,
    log: Callable[[str], None] = print,
) -> KandinskySR:
    """Download (first time) and load the model, then apply the speed-ups.

    precision: "fp8" quantizes the transformer's linear layers in memory and runs them as fp8 GEMMs (about 6 GB of
        VRAM instead of 10); "bf16" keeps the original weights. fp8 falls back to bf16 on GPUs without fp8 support.
    compile: torch.compile on the transformer and the VAE (needs triton); skipped with a message if unavailable.
    """
    from diffusers import AutoencoderKL

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("Kandinsky ImageSR Fast needs an NVIDIA GPU (CUDA device).")
    if precision not in ("fp8", "bf16"):
        raise ValueError(f"precision must be 'fp8' or 'bf16', got {precision!r}")

    start = time.perf_counter()
    transformer = KandinskyImageSRBakedTransformer3DModel.from_pretrained(
        repo, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    if bool(getattr(transformer.config, "visual_cond", False)):
        raise RuntimeError("this checkpoint uses visual conditioning, which this node does not implement")
    if precision == "fp8":
        if fp8_supported(device):
            converted = convert_to_fp8(transformer, device)  # layer by layer, before the rest moves to the GPU
            log(f"quantized {converted} linear layers to fp8")
        else:
            precision = "bf16"
            log("this GPU has no fp8 matmul support; using bf16 instead")
    transformer = transformer.to(device).eval()
    vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=torch.bfloat16).to(device).eval()

    transformer._timestep_indices = types.MethodType(_nearest_timestep_indices, transformer)
    attention = "default"
    if sys.platform == "win32" and cudnn_attention_supported(device):
        # Windows PyTorch builds ship without FlashAttention; the cuDNN kernel is ~2.4x faster than the fallback.
        transformer.set_attention_backend("_native_cudnn")
        attention = "cuDNN"
    use_matmul_attention(vae)

    model = KandinskySR(transformer, vae, device, precision, log)
    if compile:
        model.enable_compile()
    log(
        f"loaded in {time.perf_counter() - start:.1f} s: {precision} transformer, {attention} attention, "
        f"torch.compile {'on' if model.compiled else 'off'}, {torch.cuda.memory_allocated(device) / 1024**3:.1f} GB VRAM"
    )
    return model
