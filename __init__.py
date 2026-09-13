# -*- coding: utf-8 -*-
"""ComfyUI-AnimaT5Free: T5-free text conditioning for Anima.

ZERO custom nodes - this package only patches ComfyUI so the T5-free
models behave like stock Anima models:

  1. UNet: anima-t5free-fused-v0.1.safetensors (Anima DiT + our
     conditioner, dead llm_adapter removed) auto-detects as "anima" -
     load it with the STOCK "Load Diffusion Model" node.
  2. Text encoder: anima-t5free-qwen35-2b.safetensors (Qwen3.5-2B-Base +
     conditioner + tokenizer + config, split_files style) loads with the
     STOCK "Load CLIP" node (any type); STOCK CLIPTextEncode then
     produces the T5-free carrier exactly where the old T5-based
     conditioning went.

Stock workflow: Load Diffusion Model + Load CLIP + CLIPTextEncode (x2)
+ EmptyCosmosLatentVideo (768x768, length 1) + KSampler + Load VAE
(qwen_image_vae.safetensors) + VAE Decode + Save Image.

No T5 model and no T5 tokenizer anywhere in the chain.
"""
import sys
from pathlib import Path

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import comfy.model_management                  # noqa: E402,F401 (comfy first)
import comfy.model_detection as _md            # noqa: E402
import comfy.sd                                # noqa: E402
import comfy.utils                             # noqa: E402

from . import t5free_te                         # noqa: E402

# ---- patch 1: fused UNet detects as anima ----------------------------------
if not getattr(_md.detect_unet_config, "_anima_t5free_patched", False):
    _orig_detect_unet_config = _md.detect_unet_config

    def _detect_unet_config_t5free(state_dict, key_prefix, metadata=None):
        cfg = _orig_detect_unet_config(state_dict, key_prefix, metadata)
        if cfg is not None and cfg.get("image_model") == "cosmos_predict2" \
                and f"{key_prefix}conditioner.c0.W" in state_dict:
            cfg["image_model"] = "anima"
        return cfg

    _detect_unet_config_t5free._anima_t5free_patched = True
    _detect_unet_config_t5free.__wrapped__ = _orig_detect_unet_config
    _md.detect_unet_config = _detect_unet_config_t5free

# ---- patch 2: our TE file loads through the stock CLIP machinery -----------
if not getattr(comfy.sd.load_text_encoder_state_dicts,
               "_anima_t5free_patched", False):
    _orig_ltes = comfy.sd.load_text_encoder_state_dicts

    def _ltes_t5free(*args, **kwargs):
        state_dicts = args[0] if args else kwargs.get("state_dicts", [])
        if len(state_dicts) == 1 and "conditioner.c0.W" in state_dicts[0]:
            parameters = comfy.utils.calculate_parameters(state_dicts[0])
            return comfy.sd.CLIP(
                t5free_te._AnimaT5FreeClipTarget,
                embedding_directory=kwargs.get("embedding_directory"),
                parameters=parameters,
                tokenizer_data={},
                state_dict=list(state_dicts),
                model_options=kwargs.get("model_options", {}),
                disable_dynamic=kwargs.get("disable_dynamic", False))
        return _orig_ltes(*args, **kwargs)

    _ltes_t5free._anima_t5free_patched = True
    _ltes_t5free.__wrapped__ = _orig_ltes
    comfy.sd.load_text_encoder_state_dicts = _ltes_t5free

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
