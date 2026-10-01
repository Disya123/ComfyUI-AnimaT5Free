"""Instance-scoped F128 adaptation; K/V lives only in one forward's options."""

from types import MethodType

import torch

from .f128_carrier import CARRIER_VERSION, DEFAULT_ROW_TOKENIZER, RW
from .f128_receiver import HD, NB, NH, explicit_attention

KV_OPTION = "anima_f128_kv"


def _cross_attention(self, x, context=None, rope_emb=None, transformer_options=None, **kwargs):
    kv = (transformer_options or {}).get(KV_OPTION)
    if kv is None:
        raise RuntimeError("F128 attention was called outside its owning diffusion forward")
    return explicit_attention(self, x, kv[self._f128_block_index])


def _forward(self, x, timesteps, context, **kwargs):
    if not torch.is_tensor(context) or context.ndim != 3 or context.shape[-1] != RW:
        raise ValueError(f"F128 diffusion model requires a [batch,rows,{RW}] carrier")
    if kwargs.get("t5xxl_ids") is not None:
        raise ValueError("F128 cannot use the native llm_adapter text path")
    options = dict(kwargs.get("transformer_options") or {})
    options[KV_OPTION] = self._f128_receiver(context)
    kwargs["transformer_options"] = options
    # Cosmos uses context.dtype for the final-layer input. Context content is
    # unused here because each cross-attention consumes the per-block K/V.
    # Respect manual compute dtype (also used for fp8 storage with cast ops).
    compute_dtype = (
        getattr(self, "_f128_compute_dtype", None) or self.final_layer.linear.weight.dtype
    )
    dummy = context.new_zeros(context.shape[0], 1, 1024, dtype=compute_dtype)
    # The training contract is t=sigma. Comfy's Anima multiplier is already 1.
    return self._f128_original_forward(x, timesteps, dummy, **kwargs)


def _extra_conds(self, **kwargs):
    from .f128_carrier import (
        CARRIER_VERSION_DUAL,
        CARRIER_VERSION_NATIVE,
    )

    native_model = str(
        self._f128_config.get("carrier", "")
    ).startswith("native-rows")
    version = kwargs.get("f128_carrier_version")
    direct = kwargs.get("cross_attn")

    # IMPORTANT: dual metadata wins over the direct CONDITIONING tensor.
    # CLIPTextEncode can execute before the diffusion checkpoint is known; in
    # that case the TE deliberately emits both d1c and native carriers from one
    # Qwen forward.  The direct tensor is only a transport/compatibility value.
    if version == CARRIER_VERSION_DUAL:
        carrier = kwargs.get(
            "f128_x_native" if native_model else "f128_x_d1c"
        )
        if not torch.is_tensor(carrier):
            raise ValueError(
                "F128 dual conditioning is missing the carrier this "
                f"checkpoint needs ({'native' if native_model else 'd1c'}). "
                "Re-encode both prompts with the bundled text encoder."
            )
    elif torch.is_tensor(direct) and direct.ndim == 3 and direct.shape[-1] == RW:
        carrier = direct
    else:
        carrier = kwargs.get("f128_x")
        if not torch.is_tensor(carrier):
            raise ValueError(
                "F128 needs conditioning from the bundled Anima Qwen text encoder. "
                "Re-encode both prompts after installing this package."
            )

    if carrier.ndim == 2:
        carrier = carrier.unsqueeze(0)
    if carrier.ndim != 3 or carrier.shape[-1] != RW or not 1 <= carrier.shape[1] <= 512:
        raise ValueError(f"Invalid F128 carrier shape {tuple(carrier.shape)}")

    if version is not None and version != CARRIER_VERSION_DUAL:
        expected = CARRIER_VERSION_NATIVE if native_model else CARRIER_VERSION
        if version != expected:
            raise ValueError(
                f"F128 carrier was encoded as {version!r} but this "
                f"checkpoint needs {expected!r} (carrier="
                f"{self._f128_config.get('carrier', 'd1c')!r}). "
                "Re-encode both prompts with the bundled text encoder."
            )

    # Native rows do not depend on a T5 row tokenizer.  Keep the legacy check
    # only for d1c checkpoints, where row-tokenizer identity is part of the
    # carrier contract.
    if not native_model:
        tokenizer = kwargs.get("f128_row_tokenizer")
        expected = self._f128_config.get("t5tok_name", DEFAULT_ROW_TOKENIZER)
        if tokenizer is not None and tokenizer != expected:
            raise ValueError(f"F128 expects row tokenizer {expected!r}, got {tokenizer!r}")

    clean = {k: v for k, v in kwargs.items() if k not in ("t5xxl_ids", "t5xxl_weights")}
    clean["cross_attn"] = carrier
    # Comfy Anima uses CONDRegular: unequal prompt lengths run separately.
    return self._f128_original_extra_conds(**clean)


def attach_runtime(base, receiver, config):
    model = base.diffusion_model
    if hasattr(model, "_f128_receiver"):
        raise ValueError("This diffusion model already has an F128 receiver")
    if len(model.blocks) != NB:
        raise ValueError(f"F128 requires {NB} DiT blocks")
    for block in model.blocks:
        ca = block.cross_attn
        if (ca.n_heads, ca.head_dim) != (NH, HD):
            raise ValueError("F128 requires 16 attention heads of width 128")
        for name in ("q_proj", "q_norm", "k_norm", "output_proj"):
            if not hasattr(ca, name):
                raise ValueError(f"Unsupported Comfy attention API: missing {name}")
    # Registered nn.Module: state/size, model moves, unload and ordinary clones
    # now include the receiver. No receiver or active flag exists at module scope.
    model._f128_receiver = receiver
    model._f128_compute_dtype = base.get_dtype_inference()
    model._f128_original_forward = model.forward
    model.forward = MethodType(_forward, model)
    for index, block in enumerate(model.blocks):
        block.cross_attn._f128_block_index = index
        block.cross_attn.forward = MethodType(_cross_attention, block.cross_attn)
    # The checkpoint replaced this frontend. Remove its unused random weights.
    if hasattr(model, "llm_adapter"):
        model.llm_adapter = None
    base._f128_config = dict(config)
    base._f128_original_extra_conds = base.extra_conds
    base.extra_conds = MethodType(_extra_conds, base)
