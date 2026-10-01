# Validation of the 0.2.0 rewrite

Validation performed on 2026-10-01, using Python 3.12.14, PyTorch 2.14.1 CPU,
Transformers 5.18.0 and safetensors 0.8.0.

## Automated checks

- **28 CPU tests passed.**
- Ruff lint and formatting checks passed.
- All Python source and test modules compiled successfully.

The checked-in tests cover:

| Area | Checks |
| --- | --- |
| d1c carrier | Exact equality against the previous calculation, conversion before pooling, overlap/EOS fallback, use of the first 25 states, row-limit errors, single-space uncond. |
| Receiver | Exact equality against the previous block calculation with and without bf16 autocast, two-sample CFG batch, nonzero residuals/gains, zero padding, strict state-dict keys and dtype contract. |
| Cross-attention | Explicit fp32 softmax/AV even under an outer autocast context. |
| Runtime | Two loaded models keep distinct receivers, stock methods stay unchanged, receiver registration, copied-owner method binding, exceptions do not retain activation state, timestep pass-through at low sigma. |
| Conditioning | Precomputed carrier bridge, direct carrier precedence, native adapter input removal, metadata/version validation. |
| Text encode | One Qwen capture for both paths, load-order independence, empty-prompt semantics, clone-local text modes, cache bypass for patched weights. |
| Loaders | Positional and keyword arguments, preservation of stock calls, model-owned receiver attachment, raw coordinates, idempotent hooks. |
| Cache | Weight-content identities, model separation, atomic writes, invalid/corrupt entries, independent Qwen extraction. |

Tests use synthetic tensors and small stand-ins for expensive Comfy/model
objects. Receiver math tests retain 51200 feature width, 16×128 heads and
512 K/V slots, with two blocks and small ranks to keep CPU memory reasonable.
They do not load the user's trained checkpoint.

## Additional local comparison against the supplied archive

The actual original modules from the uploaded archive were executed in a
separate validation harness, outside the release tree:

- Original `z_from` versus the rewritten receiver: **bitwise equal on CPU**
  for unequal prompt lengths in a CFG batch, including zero padding and
  nonzero residual matrices.
- Original C0/C1 `conditioning` versus the rewritten implementation:
  **bitwise equal on CPU** for ASCII and non-ASCII text using the same synthetic
  weights, taps and offsets.

This is arithmetic parity on those inputs, not a claim of bitwise-identical
CUDA images on every accelerator/library version.

## Current ComfyUI API check

Reviewed upstream source at
[`651ca296a73cd21c12a57eb8741d52e40dc6528f`](https://github.com/Comfy-Org/ComfyUI/tree/651ca296a73cd21c12a57eb8741d52e40dc6528f).
The actual method bodies from `sd.py`, `model_base.py`, `conds.py`,
`ldm/anima/model.py` and `ldm/cosmos/predict2.py` were executed in a CPU
API smoke harness with lightweight dependencies:

- `Anima.forward` through `MiniTrainDIT.forward/_forward`: options reached all
  28 cross-attention calls and the final-layer dtype was respected.
- `BaseModel.extra_conds`, `Anima.extra_conds` and `CONDRegular`: precomputed
  carrier routing worked; unequal lengths were not concatenated/repeated;
  batch-size repetition kept the expected dimensions.
- The actual `CLIP.clone` method retained the rewritten subclass and its
  full-load request.

The harness did not start a ComfyUI server, exercise CUDA memory pressure or
perform a full generation. GitHub Actions is configured for Python 3.10 and
3.12 CPU checks; those remote jobs have not been run as part of this archive.

## CUDA generation checks still needed

The supplied archive contains code, not diffusion/encoder/VAE weights, and
this validation environment has no CUDA device. Before tagging a public
release, run these integration checks in the target ComfyUI installation:

1. Encode and generate twice with the current merged F128 checkpoint,
   including an empty negative prompt and a fresh positive prompt.
2. Load a second merged F128 checkpoint, then run the first again. Confirm
   each keeps its own receiver and that unloading releases its VRAM.
3. Run the original T5-Free model in the same process, using C0/C1 mode.
4. Check a CLIP clone, normal offload/reload and the supported low-VRAM path.
5. Check CFG with different prompt lengths, img2img with a low starting sigma,
   and at least two resolutions/aspect ratios.

Current merged checkpoints are supported without changing weights. Older
receiver-in-TE exports are intentionally rejected. Generic Comfy save/export
nodes, other accelerator backends and concurrent multi-device execution of a
shared model have not been validated by this release.
