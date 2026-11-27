# ComfyUI-KandinskySR-Fast

Fast ComfyUI nodes for [Kandinsky ImageSR 2.0](https://huggingface.co/kandinskylab/Kandinsky-ImageSR-2.0), a one-step x2/x4 image super-resolution model.

The nodes run the original weights through a faster path:

- **fp8 transformer** — the linear layers are quantized to fp8 in memory at load time and run as fp8 matmuls. No separate fp8 download and no quantization library.
- **cuDNN attention** on Windows, where PyTorch ships without FlashAttention.
- **torch.compile** on the transformer and the VAE (optional, needs triton).
- **Faster VAE attention** for the 1024 px tiles.

## Speed

RTX 5090, Windows 11, x4 upscale, time per image after the first run:

| Input → output | Tiles | Reference pipeline (bf16) | This node (fp8 + compile) |
|---|---:|---:|---:|
| 512×512 → 2048×2048 | 9 | 4.93 s | 2.49 s |
| 1024×1024 → 4096×4096 | 25 | 13.74 s | 7.00 s |

Peak VRAM while upscaling drops from about 13 GB to about 8 GB. Without `torch.compile` the node takes about 1.6x as long per image as with it.

Quality was compared on a small set of photos: fp8 output is within noise of the bf16 output on a downscale-then-restore test (23.69 dB vs 23.64 dB PSNR against the original photos). Only an RTX 5090 on Windows has been tested so far.

## Install

In `ComfyUI/custom_nodes`:

```
git clone https://github.com/freddiehdxd/ComfyUI-KandinskySR-Fast
```

Then install the requirements with ComfyUI's Python (a current ComfyUI usually has all of them already):

```
python -m pip install -r ComfyUI-KandinskySR-Fast/requirements.txt
```

Restart ComfyUI. The model (about 10 GB) is downloaded to the Hugging Face cache the first time the loader runs.

### Optional: torch.compile

The `compile` option needs triton. If it is missing, the node prints a message and runs without compiling.

- **Windows:** `python -m pip install triton-windows` plus the MSVC build tools. Pick the `triton-windows` version that matches your PyTorch version (see the [triton-windows README](https://github.com/woct0rdho/triton-windows)). Tested: PyTorch 2.11 with triton-windows 3.7.1, and PyTorch 2.14 with triton-windows 3.9.0.
- **Linux:** triton is installed with PyTorch.

The first image of a session compiles the model, which takes from 20 s to about 2 minutes. Later images run at full speed.

## Nodes

Both are under **image → upscaling**. An example workflow is in `example_workflows/`.

### Kandinsky ImageSR 2.0 Loader (Fast)

| Input | Default | Meaning |
|---|---|---|
| `precision` | `fp8` | `fp8` is fastest and uses the least VRAM. `bf16` runs the original weights. `fp8` falls back to `bf16` on GPUs without fp8 matmul support. |
| `compile` | on | Use `torch.compile` (see above). |
| `model` | `kandinskylab/Kandinsky-ImageSR-2.0` | Hugging Face repo id, or a local folder that contains the model's `transformer/` and `vae/` subfolders. |

### Kandinsky ImageSR 2.0 Upscale (Fast)

| Input | Default | Meaning |
|---|---|---|
| `scale` | `x4` | `x4` or `x2`. |
| `creativity` | 4 | 0 stays closest to the input, 8 invents the most detail. 4 is the model's default. |
| `overlap` | 0.25 | Tile overlap. 0.25 is the model's default. Lower values such as 0.06 need fewer tiles on many image sizes and run faster, but the result differs slightly from the default. |
| `keep_loaded` | on | Keep the model in VRAM afterwards. Turn off to move it to system RAM so other models can use the GPU. |

Image batches are upscaled one image at a time. The node shows per-tile progress and can be interrupted between tiles.

## Requirements

- NVIDIA GPU. `fp8` needs a GPU with fp8 matmul support (RTX 40/50-series); other GPUs use `bf16` automatically.
- Free VRAM: 8 to 10 GB for `fp8`, 11 to 13 GB for `bf16` (the lower figure is with `compile` on).
- PyTorch 2.6 or newer and diffusers 0.40 or newer (tested with PyTorch 2.11 and 2.14, diffusers 0.40 and 0.43).

## Credits and license

The model and the reference pipeline are by [Kandinsky Lab](https://github.com/kandinskylab/kandinsky-image-sr). Two files of that pipeline are included unchanged in `ksr/vendor/` under their MIT license. See the [model card](https://huggingface.co/kandinskylab/Kandinsky-ImageSR-2.0) for the license of the weights.

This repository is released under the MIT license.
