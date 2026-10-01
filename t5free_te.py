"""Comfy text-encoder API for the existing bundled Qwen/C0/C1 file.

The default emits the original C0/C1 conditioning plus F128 metadata. It is
independent of which diffusion models have been loaded. Both paths share one
Qwen forward. An optional tokenizer option selects only one path.
"""

import hashlib
import json
import logging

import torch
from torch import nn

from .cache import TensorCache, cache_root, decode_json, extract_qwen
from .configuration_anima_t5free import AnimaT5FreeConfig
from .f128_carrier import (
    CARRIER_VERSION,
    CARRIER_VERSION_DUAL,
    CARRIER_VERSION_NATIVE,
    DEFAULT_ROW_TOKENIZER,
    RW,
    capture_states,
    carrier_from_states,
    carrier_from_states_native,
    load_row_tokenizer,
    normalize_text,
)
from .modeling_anima_t5free import AnimaT5FreeModel

log = logging.getLogger(__name__)


class AnimaT5FreeTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data=None, **kwargs):
        pass

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        options = kwargs.get("tokenizer_options", {})
        return {
            "anima_t5free_text": str(text),
            "anima_text_mode": options.get("anima_text_mode"),
        }

    def untokenize(self, token_weight_pair):
        return token_weight_pair

    def state_dict(self):
        return {}

    def decode(self, token_ids, **kwargs):
        return ""


class AnimaT5FreeTEModel(nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options=None, **kwargs):
        super().__init__()
        self.dtypes = {torch.float16, torch.float32}
        self._te_dtype = dtype
        self.options = dict(model_options or {})
        self.default_mode = self.options.get("anima_text_mode", "dual")
        self.cache_enabled = True
        self.qwen = None
        self.cond = None
        self.qtok = None
        self.row_tokenizer = None
        self.row_tokenizer_name = self.options.get("anima_row_tokenizer", DEFAULT_ROW_TOKENIZER)
        self.cfg = None
        self.lex = None
        self.source_id = None

    def set_clip_options(self, options):
        pass

    def reset_clip_options(self):
        pass

    def load_sd(self, sd):
        if self.qwen is not None:
            raise ValueError("Anima text encoder expects exactly one bundled state dictionary")
        config = decode_json(sd["t5free_config_bytes"], "t5free_config_bytes")
        self.cfg = AnimaT5FreeConfig(**config)
        self.cond = AnimaT5FreeModel(self.cfg).eval().requires_grad_(False)
        self.cond.load_state_dict(
            {k: v for k, v in sd.items() if k.startswith("conditioner.")}, strict=True
        )
        raw_lexicon = decode_json(sd["t5free_lexicon_bytes"], "t5free_lexicon_bytes")
        self.lex = {k: (int(v[0][0]), tuple(map(int, v[0][1]))) for k, v in raw_lexicon.items()}
        directory, self.source_id = extract_qwen(sd, cache_root())
        from transformers import AutoModel, AutoTokenizer

        self.qwen = (
            AutoModel.from_pretrained(
                str(directory),
                dtype=torch.float16,
                local_files_only=True,
                attn_implementation="sdpa",
            )
            .eval()
            .requires_grad_(False)
        )
        self.qtok = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        if not self.qtok.is_fast:
            raise ValueError("Anima conditioning requires a fast Qwen tokenizer")
        # No T5 model, receiver or diffusion-model reference is held by the TE.
        return [], []

    def _cache(self, mode):
        import tokenizers
        import transformers

        if mode != "t5free" and self.row_tokenizer is None:
            self.row_tokenizer = load_row_tokenizer(self.row_tokenizer_name)
        row_json = self.row_tokenizer.backend_tokenizer.to_str() if mode != "t5free" else "none"
        namespace = json.dumps(
            {
                # F128/dual cache must never reuse the old d1c-only cache:
                # one Qwen forward now stores BOTH legacy d1c and native rows.
                "version": CARRIER_VERSION_DUAL if mode != "t5free" else CARRIER_VERSION,
                "qwen": self.source_id,
                "row_tokenizer": hashlib.sha256(row_json.encode()).hexdigest(),
                "layers": self.cfg.qwen_layers,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__,
                "mode": mode,
                "execution_device": str(self.qwen.get_input_embeddings().weight.device)
                if self.qwen is not None
                else "unknown",
                "qwen_dtype": str(self.qwen.get_input_embeddings().weight.dtype)
                if self.qwen is not None
                else "unknown",
            },
            sort_keys=True,
        )
        return TensorCache(cache_root(), namespace)

    def _valid_features(self, data, need_f128):
        if not isinstance(data, dict) or not isinstance(data.get("taps"), dict):
            return False
        for layer in self.cfg.qwen_layers:
            tap = data["taps"].get(layer)
            if not torch.is_tensor(tap) or tap.ndim != 2 or tap.shape[-1] != 2048:
                return False
            if tap.dtype != torch.float16 or not torch.isfinite(tap).all():
                return False
        if need_f128:
            # Order-independent F128 encoding carries both domains.  Comfy may
            # execute CLIPTextEncode before the diffusion checkpoint is loaded,
            # so the text encoder cannot safely guess d1c vs native rows.
            for key in ("X_d1c", "X_native"):
                x = data.get(key)
                if not torch.is_tensor(x) or x.ndim != 2 or x.shape[1] != RW:
                    return False
                if not 1 <= x.shape[0] <= 512 or x.dtype != torch.bfloat16:
                    return False
                if not torch.isfinite(x).all():
                    return False
        return True

    def _features(self, text, mode):
        cache = self._cache(mode)
        data = cache.read(text) if self.cache_enabled else None
        need_f128 = mode != "t5free"
        if data is not None and self._valid_features(data, need_f128):
            return data
        if data is not None:
            log.warning("Ignoring invalid cached Anima features for model %s", self.source_id)
        states = capture_states(self.qwen, self.qtok, text)
        if max(self.cfg.qwen_layers) >= len(states):
            raise ValueError("Configured Qwen tap is outside the captured hidden states")
        data = {"taps": {layer: states[layer] for layer in self.cfg.qwen_layers}}
        if need_f128:
            # Build both carriers from the SAME Qwen forward.  This removes the
            # graph-order race: the diffusion checkpoint may be resolved only
            # after CLIPTextEncode, and _extra_conds will select the matching
            # carrier at sampling time.
            rows = self.row_tokenizer(text, return_offsets_mapping=True).offset_mapping
            offsets = self.qtok(text, return_offsets_mapping=True).offset_mapping
            data["X_d1c"] = carrier_from_states(states, rows, offsets)
            data["X_native"] = carrier_from_states_native(states)
        if self.cache_enabled:
            try:
                cache.write(text, data)
            except OSError as exc:
                log.warning("Could not persist Anima prompt features: %s", exc)
        return data

    @torch.no_grad()
    def encode_token_weights(self, tokens):
        mode = (
            (tokens.get("anima_text_mode") or self.default_mode)
            if isinstance(tokens, dict)
            else self.default_mode
        )
        if mode not in ("dual", "f128", "t5free"):
            raise ValueError(f"Unknown Anima text mode {mode!r}")
        raw_text = tokens["anima_t5free_text"] if isinstance(tokens, dict) else str(tokens)
        # Original C0/C1 uncond is zeros. F128 uncond is the live single-space path.
        if mode == "t5free" and not raw_text.strip():
            return self._zero_conditioning(), None, {}
        text = normalize_text(raw_text)
        features = self._features(text, mode)
        metadata = {}
        if mode != "t5free":
            metadata = {
                "f128_x_d1c": features["X_d1c"],
                "f128_x_native": features["X_native"],
                "f128_row_tokenizer": self.row_tokenizer_name,
                "f128_carrier_version": CARRIER_VERSION_DUAL,
            }
        if mode == "f128":
            # A CONDITIONING object still needs a concrete cross_attn tensor.
            # Use d1c only as the transport value; F128 runtime MUST select
            # f128_x_d1c/f128_x_native from metadata when version == DUAL.
            # This keeps explicit F128 mode order-independent too.
            return features["X_d1c"].unsqueeze(0), None, metadata
        if not raw_text.strip():
            conditioning = self._zero_conditioning()
        else:
            conditioning = self.cond.conditioning(text, self.qtok, self.lex, features["taps"])
            if self._te_dtype is not None:
                conditioning = conditioning.to(self._te_dtype)
            conditioning = conditioning.cpu()
        return conditioning, None, metadata

    def _zero_conditioning(self):
        return torch.zeros(
            (1, self.cfg.max_rows, self.cfg.context_dim),
            dtype=self._te_dtype or torch.bfloat16,
        )


class AnimaT5FreeClipTarget:
    clip = AnimaT5FreeTEModel
    tokenizer = AnimaT5FreeTokenizer
    params = {}
