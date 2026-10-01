# 0.2.0

- Replace global receiver/holder/activation state with model-owned receivers
  and per-forward K/V in transformer options.
- Remove the class-wide Anima forward patch and model-load-order mode switch.
- Load receiver/config data from the diffusion loader's existing state
  dictionary instead of reopening safetensors files.
- Infer receiver ranks from checkpoint tensors and enforce strict loading.
- Register receivers for model size, cloning and full load/offload; remove
  the unused native `llm_adapter` from marked F128 instances.
- Retain full-load HF text encoding through CLIP cloning.
- Share one Qwen capture between both text paths. Add optional explicit mode
  selection while keeping existing stock-node workflows compatible.
- Separate extraction and prompt caches by model content. Use atomic writes,
  validate cached tensor shapes/dtypes, and bypass base caches for patched TEs.
- Remove standalone receiver/mode TE loading, `sys.path` edits, mutable
  argument defaults, ad-hoc print logging and research-only sampling helpers.
- Preserve d1c, receiver/attention math, raw F128 latent coordinates and
  timestep pass-through. Correct the documentation about the T5 row tokenizer.
- Add CPU regression tests, lint/format configuration and GitHub Actions CI.

## Upgrade impact

Replace the plugin directory and restart. Current merged F128 checkpoint and
bundled Qwen files are reused unchanged. Prompt features are recomputed in the
new model-specific cache. Old standalone F128 text files fail with a migration
message. This release changes inference integration, not training weights.
