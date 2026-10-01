# Runtime boundaries

The package installs three idempotent compatibility hooks:

| Hook | Responsibility |
| --- | --- |
| `detect_unet_config` | Recognize marked T5-Free/F128 DiT weights as Anima. |
| `load_text_encoder_state_dicts` | Recognize the bundled encoder and create a full-load CLIP subclass. Unrelated encoders use the original loader. |
| `load_diffusion_model_state_dict` | Read merged F128 data once from the supplied state dictionary, attach its receiver and select raw latent coordinates. Unmarked models use the original loader. |

There is no class-level Anima forward patch and no replacement of the
file-based diffusion loader. Receiver loading does not reopen the checkpoint.
All imports are package-relative; the plugin does not modify `sys.path`.

## Ownership and lifetime

`BaseModel.diffusion_model._f128_receiver` is a registered torch module owned
by one diffusion model. Ordinary Comfy patcher clones share that model, as
upstream clones normally do. A separately loaded checkpoint has a separate
receiver. Deep-copied modules have rebound instance methods rather than
closures pointing back to the original model.

Only marked instances get F128 `forward`, `extra_conds`, and cross-attention
methods. Native Anima instances keep their original methods.

During a forward, the receiver computes 28 `(K,V)` pairs. A copied
`transformer_options` dictionary carries those pairs through the upstream
Cosmos block loop. Each instance-bound cross-attention reads its own block
index. No active flag, holder, receiver registry or retained K/V tensor is
shared across calls. A raised exception leaves no global activation state to
reset.

F128 receivers use plain torch projections, so F128 loaders select regular
ModelPatcher. The receiver is registered for size accounting and full
load/offload. Its forward co-locates all receiver weights/buffers on the
context device, while DiT layers retain Comfy's manual-cast support. Separate
concurrent executions of the same shared model on different devices require
separate model instances; this plugin does not add a concurrent device scheduler.

## Text and conditioning contracts

The bundled TE has no diffusion-model reference. Default dual encoding always
provides C0/C1 conditioning plus F128 metadata; explicit mode travels in the
token dictionary. The optional node sets a tokenizer option on a CLIP clone,
so mode selection does not mutate the shared torch encoder.

Qwen executes once during CLIP encode. Hidden states are captured in fp16 for
C0/C1 taps. F128 converts the first 25 hidden-state entries to bf16 before
span pooling, preserving the original order of conversions. Character overlap
and nearest-centre fallback—including EOS—match the supplied implementation.
Per-layer RMS prenorm uses active rows before any padding. F128 carriers are
CPU bf16 `[rows,51200]` tensors, with a maximum of 512 rows including EOS.
Oversized prompts raise an error instead of silently changing the training
contract.

The F128 model bridge uses a direct 51200-wide conditioning tensor when
available; otherwise it uses precomputed `f128_x` metadata. It strips native
T5 adapter inputs and delegates to the original Anima `extra_conds`.
Upstream Anima uses `CONDRegular`, so unequal row counts are evaluated in
separate batches instead of repeating rows to an LCM. The receiver pads K/V
to 512 with zeros. It does not mask those padded slots out of softmax, which
would change the trained attention function.

Base projections are bf16 buffers. Residual matrices and gains are fp32, as
in the original R1Copy inference loader. Receiver matmuls use bf16 autocast;
K/V grids remain fp32. Cross-attention normalizes pre-norm K after conversion
to bf16, calculates softmax and AV explicitly in fp32, then casts to the
query input dtype for the output projection. The upstream SA/MLP/residual path
and timestep values are retained. This is not an all-bf16 rewrite.

## File boundaries

| Module | Responsibility |
| --- | --- |
| `f128_carrier.py` | State capture, row alignment, d1c math. |
| `f128_receiver.py` | Strict checkpoint skeleton, receiver and attention math. |
| `f128_runtime.py` | Per-model methods and per-call K/V routing. |
| `comfy_integration.py` | Loader/detection adapters, CLIP subclass and raw latent format. |
| `t5free_te.py` | Bundled TE API, mode selection and shared feature extraction. |
| `modeling_anima_t5free.py` | Original C0/C1 conditioning math, without unrelated research sampling/network helpers. |
| `cache.py` | Weight identities, extraction and atomic tensor cache. |

The integration is intended for loading and inference. Generic Comfy save
nodes are not a supported exporter for the bundled TE or merged F128 file
format; use the project's checkpoint export tooling to author those files.
