"""The three process-wide compatibility hooks; all runtime changes are local."""

import inspect
import logging
from functools import wraps

import comfy.latent_formats
import comfy.model_detection
import comfy.model_management
import comfy.sd
import comfy.utils
import torch

from .cache import decode_json
from .f128_receiver import F128Receiver
from .f128_runtime import attach_runtime
from .t5free_te import AnimaT5FreeClipTarget

log = logging.getLogger(__name__)
PATCH_TAG = "_anima_t5free_v2"


class FullLoadCLIP(comfy.sd.CLIP):
    """Keep Transformers on one device before every encode, including clones."""

    def load_model(self, tokens=None):
        self.cond_stage_model.cache_enabled = not (
            self.patcher.patches or self.patcher.forced_hooks
        )
        memory = 0
        estimator = getattr(self.cond_stage_model, "memory_estimation_function", None)
        if estimator is not None:
            memory = estimator(tokens or {}, device=self.patcher.load_device)
        comfy.model_management.load_models_gpu(
            [self.patcher], memory_required=memory, force_full_load=True
        )
        return self.patcher

    def clone(self, *args, **kwargs):
        clone = super().clone(*args, **kwargs)
        # Comfy's clone constructs the base CLIP type. Reapply this stateless
        # subclass so LoRA/device clones retain the full-load contract.
        clone.__class__ = type(self)
        return clone


class RawWan21(comfy.latent_formats.Wan21):
    """Wan21 layout/preview metadata, raw Qwen-VAE coordinates for F128."""

    def __init__(self):
        super().__init__()
        self.scale_factor = 1.0
        self.latents_mean = torch.zeros_like(self.latents_mean)
        self.latents_std = torch.ones_like(self.latents_std)

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent


def _replace(owner, name, wrapper):
    original = getattr(owner, name)
    if getattr(original, PATCH_TAG, False):
        return
    replacement = wraps(original)(wrapper(original))
    setattr(replacement, PATCH_TAG, True)
    setattr(owner, name, replacement)


def _detection_hook(original):
    def detect(state_dict, key_prefix, *args, **kwargs):
        config = original(state_dict, key_prefix, *args, **kwargs)
        ours = any(
            key in state_dict
            for key in (
                f"{key_prefix}conditioner.c0.W",
                f"{key_prefix}f128_marker",
                "f128_marker",
            )
        )
        if config is not None and config.get("image_model") == "cosmos_predict2" and ours:
            config = dict(config)
            config["image_model"] = "anima"
        return config

    return detect


def _text_loader_hook(original):
    signature = inspect.signature(original)

    def load(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        state_dicts = arguments.arguments.get("state_dicts", [])
        if len(state_dicts) != 1:
            return original(*args, **kwargs)
        state = state_dicts[0]
        if "f128_marker" in state and "conditioner.c0.W" not in state:
            raise ValueError(
                "Standalone F128 mode/receiver text files are retired. "
                "Use the bundled qwen_35_2b_base text encoder and a diffusion "
                "checkpoint containing f128_receiver.*."
            )
        if "conditioner.c0.W" not in state:
            return original(*args, **kwargs)
        return FullLoadCLIP(
            AnimaT5FreeClipTarget,
            embedding_directory=arguments.arguments.get("embedding_directory"),
            parameters=comfy.utils.calculate_parameters(state),
            tokenizer_data={},
            state_dict=[state],
            model_options=arguments.arguments.get("model_options") or {},
            disable_dynamic=True,
        )

    return load


def _diffusion_loader_hook(original):
    signature = inspect.signature(original)

    def load(sd, *args, **kwargs):
        from . import f128_dit
        if "f128_marker" not in sd:
            # Stock checkpoint: drop the F128 session flag so the encode
            # hook falls back to the full stock path (no stale state).
            f128_dit.F128_CTX["f128_loaded"] = False
            return original(sd, *args, **kwargs)
        if "f128_config_bytes" not in sd:
            raise ValueError("F128 checkpoint is missing f128_config_bytes")
        config = decode_json(sd["f128_config_bytes"], "f128_config_bytes")
        receiver_sd = {
            k.removeprefix("f128_receiver."): v
            for k, v in sd.items()
            if k.startswith("f128_receiver.")
        }
        if not receiver_sd:
            raise ValueError("F128 checkpoint contains no receiver. Use a merged F128 checkpoint.")
        receiver = F128Receiver(
            receiver_sd,
            no_grid=config.get("carrier") == "native-rows-v2",
        )
        # Session flag ONLY: lets the encode hook skip the dead stock
        # C0/C1 pass while an F128 model is the active diffusion model.
        # Carrier selection is NOT read here: the hook always emits dual
        # carriers and _extra_conds picks per-model from its own config.
        f128_dit.F128_CTX["f128_loaded"] = True
        # Normalize net.* before Comfy filters the prefix, preserving the marker
        # for detection. Extract metadata from the already loaded state dict.
        has_net = any(k.startswith("net.") for k in sd)
        weights = {
            k.removeprefix("net.") if has_net else k: v
            for k, v in sd.items()
            if not k.startswith(("f128_receiver.", "f128_config_bytes"))
        }
        arguments = signature.bind(weights, *args, **kwargs)
        # Receiver uses plain torch ops; use regular ModelPatcher. DiT's manual
        # casts still support low-VRAM loading; full offload moves the receiver.
        arguments.arguments["disable_dynamic"] = True
        patcher = original(*arguments.args, **arguments.kwargs)
        if patcher is None:
            raise RuntimeError("ComfyUI could not detect the F128 diffusion checkpoint")
        base = patcher.model
        attach_runtime(base, receiver, config)
        # Latent contract is an EXPLICIT per-checkpoint field, orthogonal
        # to the carrier: "raw" = trained in raw Qwen-VAE coordinates
        # (d1c FT line, native continuation), "wan21" = stock backbone
        # coordinates. Legacy checkpoints without the field fall back to
        # the family inference WITH a warning (native-rows stock pair ->
        # Wan21; d1c FT -> raw).
        latent_field = str(config.get("latent", "")).strip().lower()
        if latent_field == "raw":
            base.latent_format = RawWan21()
            base.model_config.latent_format = base.latent_format
            latent_contract = "raw Qwen-VAE coordinates"
        elif latent_field == "wan21":
            latent_contract = "stock Wan21 coordinates"
        elif str(config.get("carrier", "")).startswith("native-rows"):
            log.warning(
                "F128: checkpoint has no explicit 'latent' field; "
                "native-rows stock pair -> assuming stock Wan21. "
                "Retrain continuation ckpts MUST emit latent=raw."
            )
            latent_contract = "stock Wan21 (legacy inferred)"
        else:
            base.latent_format = RawWan21()
            base.model_config.latent_format = base.latent_format
            latent_contract = "raw (legacy inferred for d1c FT)"
        # Honest size for the memory manager: Comfy estimated the DiT
        # weights already; add the attached receiver on top instead of
        # lying with zero.
        patcher.size += sum(
            v.numel() * v.element_size() for v in receiver_sd.values()
        )
        log.info(
            "Loaded F128 receiver into its owning model (step %s); %s",
            config.get("step", "unknown"),
            latent_contract,
        )
        return patcher

    return load


def install():
    _replace(comfy.model_detection, "detect_unet_config", _detection_hook)
    _replace(comfy.sd, "load_text_encoder_state_dicts", _text_loader_hook)
    _replace(comfy.sd, "load_diffusion_model_state_dict", _diffusion_loader_hook)
