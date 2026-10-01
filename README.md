# ComfyUI-AnimaT5Free

Qwen text conditioning for Anima, with support for the original T5-Free C0/C1
model and merged F128 fine-tuned checkpoints.

**[Русская инструкция](README.ru.md)** · [Architecture](ARCHITECTURE.md) ·
[Validation](TESTING.md) · [Changes](CHANGELOG.md)

This is an inference integration. It ships code, not model weights. Existing
stock-node workflows remain usable. One optional node selects the text mode
explicitly.

## Install or upgrade

1. Close ComfyUI. On upgrade, replace the entire old
   `custom_nodes/ComfyUI-AnimaT5Free` directory with this directory. Keep your
   model files in `models/`.
2. Install `requirements.txt` using the Python environment that runs ComfyUI:

   ```bash
   python -m pip install -r custom_nodes/ComfyUI-AnimaT5Free/requirements.txt
   ```

   Windows portable, from the portable distribution's root:

   ```bat
   python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-AnimaT5Free\requirements.txt
   ```

3. Restart ComfyUI. Use a ComfyUI version that includes Anima support and the
   `load_diffusion_model_state_dict(..., disable_dynamic=...)` API.

Torch is supplied by ComfyUI. The requirements do not replace its CUDA build.

## Model files

| Component | Location | Loader |
| --- | --- | --- |
| Merged F128 checkpoint, e.g. `anima-ft7000-f128.safetensors` | `models/diffusion_models/` or `models/unet/` | Load Diffusion Model |
| Original T5-Free fused checkpoint | `models/diffusion_models/` or `models/unet/` | Load Diffusion Model |
| Existing **bundled** `qwen_35_2b_base.safetensors` | `models/text_encoders/` | Load CLIP |
| `qwen_image_vae.safetensors` | `models/vae/` | Load VAE |

The bundled encoder contains `conditioner.c0.W`, `t5free_config_bytes`, the
Qwen weights, and embedded tokenizer/config JSON. A generic Qwen file with the
same filename is not this format.

A merged F128 checkpoint must contain `f128_marker`, `f128_config_bytes`, and
all `f128_receiver.*` tensors alongside its DiT weights. Receiver ranks are
read from tensor shapes and the complete state dictionary is loaded strictly.
No conversion or retraining is needed for the current merged F128 format.

Old standalone `anima-f128-mode` files and receiver-in-TE checkpoints are no
longer supported. Select the bundled Qwen file and a merged diffusion
checkpoint instead. The plugin reports an error if the receiver is missing.

## Workflow

Use Load Diffusion Model, Load CLIP, two CLIP Text Encode nodes, KSampler,
Load VAE, VAE Decode, and Save Image. For a new image, use a compatible
16-channel latent such as Empty Cosmos Latent Video with length 1.
Resolution and aspect ratio are not fixed to 768×768 by this plugin.

For a new workflow, insert **Anima Text Encoder Mode** between Load CLIP and
both CLIP Text Encode nodes:

| Mode | Use |
| --- | --- |
| **F128** | Merged F128 FT checkpoint. Produces the full carrier and skips C0/C1. |
| **T5-Free C0/C1** | Original fused T5-Free model. No T5 tokenizer is loaded. |
| **Both (compatibility)** | Existing workflows, or one encoder feeding both model families. Produces C0/C1 context plus precomputed F128 metadata. |

Without the optional node, the mode is **Both (compatibility)**. It never
depends on the order in which diffusion models and text encoders are loaded.
Both paths share one Qwen forward per uncached prompt. Select F128 explicitly
when editing conditioning with standard averaging/combining nodes; those
nodes operate on the main conditioning tensor, not the compatibility metadata.

Both positive and negative prompts must come from the selected encoder.
An empty F128 prompt follows the trained single-space Qwen path. An empty
C0/C1 prompt produces the original zero context. Changing the text mode on a
CLIP clone does not change other workflows sharing the encoder.

The tokenizer forwards literal prompt text. Comfy's parenthesis weighting,
textual inversion and CLIP layer selection are not implemented by this custom
Qwen encoder. They do not acquire those semantics just because the loader is
named CLIP.

## Memory and caching

The HF Qwen model is fully loaded onto one device before each text encode.
This contract is retained by CLIP clones. ComfyUI can offload the encoder
afterwards. No Qwen forward runs during sampling.

Each F128 diffusion model owns a registered receiver module. The receiver's
plain torch projections require it to fit on the sampling device. The DiT
uses the regular Comfy ModelPatcher with its manual-cast low-VRAM path;
dynamic/Aimdo loading is disabled for F128 and for the HF encoder. Full model
offload also moves the receiver. K/V is temporary forward data, not a retained
global or model-level cache.

Data defaults to `models/anima_t5free_cache/`:

- `qwen/`: HF model extraction, separated by a full hash of Qwen weights and
  tokenizer/config bytes. Each distinct encoder needs its own disk copy.
- `features/`: CPU hidden taps and F128 carriers, separated by model,
  tokenizer contents, pipeline revision and library versions. Tensor cache
  writes are atomic; corrupt entries are regenerated.

Set `ANIMA_T5FREE_CACHE_DIR` before starting ComfyUI to choose another location.
Deleting this directory is safe while ComfyUI is closed; its contents will be
rebuilt. Old `f128_cache/`, `taps_cache/` and `_qwen35_extract/` directories
inside the plugin are ignored. Patched/LoRA text encoders bypass the persistent
base-weight feature cache.

F128 uses the **T5 tokenizer for row spans only**, defaulting to
`google/t5-v1_1-xxl`; no T5 model weights are loaded. The tokenizer is downloaded
through Hugging Face on first use. For offline operation, cache its files in
advance and follow Hugging Face's offline settings. A programmatic loader can
set `model_options["anima_row_tokenizer"]` to the checkpoint's tokenizer name
or local directory and `model_options["anima_text_mode"]` to `f128`, `t5free`
or `dual`. The row-tokenizer identifier must agree with `t5tok_name` in an F128
checkpoint when that field is present.

## Development

```bash
python -m pip install -r requirements.txt pytest ruff
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The CPU suite covers arithmetic parity, model isolation, clones, conditioning,
cache identities and loader contracts. It does not replace generation tests
with actual model weights and a CUDA ComfyUI instance. See [TESTING.md](TESTING.md).

## License

Code: [MIT](LICENSE.md), copyright Disya. Model weights and tokenizers retain
their upstream licenses; the code license does not relicense those assets.
This integration is independently maintained and is not an official
CircleStone Labs or Qwen release.
