"""ComfyUI nodes for fast Kandinsky ImageSR 2.0 upscaling."""

from __future__ import annotations

import gc
import os
import re

import comfy.model_management as mm
import comfy.utils
import folder_paths
import torch

from .ksr.runner import REPO, KandinskySR, count_tiles, load_model

GB = 1024**3
FP8_DIR = os.path.join(folder_paths.models_dir, "kandinsky_sr")
_LOADED: dict = {"key": None, "model": None}


def _log(message: str) -> None:
    print(f"[KandinskySR-Fast] {message}")


def fp8_file_path(model: str) -> str:
    """Where the pre-converted fp8 file for this model lives: ComfyUI/models/kandinsky_sr/<model>-fp8.safetensors."""
    name = os.path.basename(os.path.normpath(model)) if os.path.isdir(model) else model.replace("/", "--")
    return os.path.join(FP8_DIR, re.sub(r"[^A-Za-z0-9._-]+", "_", name) + "-fp8.safetensors")


def _drop_loaded() -> None:
    model = _LOADED["model"]
    _LOADED["key"], _LOADED["model"] = None, None
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    mm.soft_empty_cache()


class KandinskySRFastLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (
                    ["fp8", "bf16"],
                    {
                        "default": "fp8",
                        "tooltip": "fp8: fastest, 8-10 GB of VRAM while upscaling (RTX 40/50-series). "
                        "bf16: original weights, 11-13 GB. fp8 falls back to bf16 on GPUs without fp8 support.",
                    },
                ),
                "compile": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "torch.compile the model: the first image of a session takes 20-60 s longer, "
                        "every image after that is about 1.5x faster. Needs triton (triton-windows on Windows); "
                        "skipped automatically if it is missing.",
                    },
                ),
            },
            "optional": {
                "model": (
                    "STRING",
                    {
                        "default": REPO,
                        "tooltip": "Hugging Face repo id (downloaded to the Hugging Face cache on first use, about "
                        "10 GB) or a local folder containing the model's transformer/ and vae/ subfolders.",
                    },
                ),
                "fp8_cache": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Save the converted fp8 model (about 6 GB) to ComfyUI/models/kandinsky_sr. "
                        "Whenever that file exists it is loaded instead of the original weights, whatever this "
                        "switch says, so a machine that has the file does not need the 10 GB download.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("KANDINSKY_SR",)
    RETURN_NAMES = ("sr_model",)
    FUNCTION = "load"
    CATEGORY = "image/upscaling"
    DESCRIPTION = "Loads Kandinsky ImageSR 2.0 with fp8 weights, cuDNN attention and torch.compile."

    def load(self, precision: str, compile: bool, model: str = REPO, fp8_cache: bool = False):
        repo = model.strip() or REPO
        key = (precision, bool(compile), repo)
        fp8_file = fp8_file_path(repo)
        if _LOADED["key"] != key:
            _drop_loaded()
            device = mm.get_torch_device()
            mm.free_memory((8 if precision == "fp8" else 12) * GB, device)
            mm.soft_empty_cache()
            _LOADED["model"] = load_model(
                precision=precision, compile=compile, device=device, repo=repo, log=_log, fp8_file=fp8_file, save_fp8=fp8_cache
            )
            _LOADED["key"] = key
        elif fp8_cache and _LOADED["model"].precision == "fp8" and not os.path.isfile(fp8_file):
            _LOADED["model"].save_fp8(fp8_file, repo)  # switch turned on after the model was already loaded
        return (_LOADED["model"],)


class KandinskySRFastUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sr_model": ("KANDINSKY_SR",),
                "image": ("IMAGE",),
                "scale": (["x4", "x2"], {"default": "x4"}),
                "creativity": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 8,
                        "tooltip": "0 = stay closest to the input, 8 = invent the most detail. 4 is the model's default.",
                    },
                ),
                "overlap": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.03,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Tile overlap. 0.25 is the model's default. Lower values (for example 0.06) often "
                        "need fewer tiles and run faster, but give a slightly different result.",
                    },
                ),
                "keep_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Keep the model in VRAM after upscaling. Turn off to move it to system RAM so "
                        "other models can use the GPU.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    DESCRIPTION = "One-step tiled x2/x4 super-resolution with Kandinsky ImageSR 2.0."

    def upscale(self, sr_model: KandinskySR, image: torch.Tensor, scale: str, creativity: int, overlap: float, keep_loaded: bool):
        factor = int(scale.lstrip("x"))
        device = mm.get_torch_device()
        if sr_model.device != device:
            mm.free_memory(9 * GB, device)
            sr_model.to(device)
        else:
            mm.free_memory(3 * GB, device)  # working memory for one tile

        batch, height, width, _ = image.shape
        total = batch * count_tiles(height, width, factor, overlap)
        progress = comfy.utils.ProgressBar(total)

        def on_tile() -> None:
            progress.update(1)
            mm.throw_exception_if_processing_interrupted()

        outputs = []
        try:
            for index in range(batch):
                pixels = image[index, :, :, :3].permute(2, 0, 1).unsqueeze(0)
                result = sr_model.upscale(pixels, factor=factor, creativity=creativity, overlap=overlap, on_tile=on_tile)
                outputs.append(result.permute(0, 2, 3, 1))
        finally:
            if not keep_loaded:
                sr_model.to("cpu")
                mm.soft_empty_cache()
        return (torch.cat(outputs, dim=0),)


NODE_CLASS_MAPPINGS = {
    "KandinskySRFastLoader": KandinskySRFastLoader,
    "KandinskySRFastUpscale": KandinskySRFastUpscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KandinskySRFastLoader": "Kandinsky ImageSR 2.0 Loader (Fast)",
    "KandinskySRFastUpscale": "Kandinsky ImageSR 2.0 Upscale (Fast)",
}
