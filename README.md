# ComfyUI-AnimaT5Free

T5-free text conditioning for Anima, **zero custom nodes**. The package
only patches ComfyUI's model detection so the two T5-free model files
behave like stock Anima models:

| File (put in `models/…`) | Stock node that loads it |
|---|---|
| `unet/anima-t5free-fused-v0.1.safetensors` (3.97 GB) | **Load Diffusion Model** |
| `text_encoders/anima-t5free-qwen35-2b.safetensors` (4.62 GB) | **Load CLIP** (any type) |

- **UNet**: frozen Anima DiT + the T5-free conditioner (`conditioner.*`)
  in one file; the dead `llm_adapter` (135M params, T5-dependent) is
  removed. Detection patched so it identifies as `anima`.
- **Text encoder**: Qwen3.5-2B-Base + the conditioner + tokenizer + config
  in one split_files-style safetensors. Patched loading makes **the stock
  CLIPTextEncode** run the proven T5-free chain:
  Qwen taps (layers 16/22) → d1 grid → C0 ridge → C1 refiner → carrier.
  Empty negative prompt → frozen zero-context uncond.
- VAE: the usual `vae/qwen_image_vae.safetensors` via **Load VAE**.

## Workflow (all stock nodes)

```
Load Diffusion Model ┐
Load CLIP ──> CLIPTextEncode (prompt) ────────┐
        └────> CLIPTextEncode ("" negative) ───┤
EmptyCosmosLatentVideo (768x768, length 1) ────┼─> KSampler (euler/simple, 30, cfg 5)
Load VAE (qwen_image_vae) ─────────────────────┴─> VAE Decode ──> Save Image
```

Taps are cached per prompt in `taps_cache/`; the first run with a new
prompt takes a few extra seconds (Qwen forward).

## Notes

- **No T5 model and no T5 tokenizer** anywhere in the chain.
- **Examples**: the published HF examples use a prompt-salted noise
  convention and a research-runtime sampler; stock KSampler uses its own
  noise/sampler machinery, so same-seed images will differ from the
  published examples (different composition, same model quality).
- This build is **parity-level semantics** (Anima-T5-Free-Base v0.1), not
  a semantic upgrade.
- Conditioner provenance: bit-identical to `Disya/Anima-T5-Free-Base`
  `model.safetensors` (sha256 in each file's `__metadata__`).

## License

The **code** in this repository is MIT-licensed — see [LICENSE.md](LICENSE.md).

This package contains code only; it does not redistribute any model
weights. The models it drives are governed by their own licenses:

- **Anima weights** — including the fused T5-free build and the
  `Disya/Anima-T5-Free-Base` HuggingFace repo — are under the
  **CircleStone Labs Non-Commercial License v1.2**:

  > “The CircleStone Model is licensed by CircleStone Labs LLC under the
  > CircleStone Non-Commercial License. Copyright CircleStone Labs LLC.
  > IN NO EVENT SHALL CIRCLESTONE LABS LLC BE LIABLE FOR ANY CLAIM, DAMAGES
  > OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
  > OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH USE OF THIS
  > MODEL.”

  **The CircleStone Model (Anima) has been modified** (the T5-dependent
  `llm_adapter` frontend is replaced by a T5-free conditioner); this is
  not an official CircleStone Labs product and has not been endorsed or
  validated by CircleStone Labs.
- **Qwen3.5-2B-Base** — its own HuggingFace license.
