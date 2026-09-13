# -*- coding: utf-8 -*-
"""Stock-flow text encoder for ComfyUI-AnimaT5Free.

AnimaT5FreeTokenizer + AnimaT5FreeTEModel implement the comfy text
encoder API at the CLIP-wrapper level:

  CLIPLoader(anima-t5free-qwen35-2b.safetensors)
    -> CLIPTextEncode(text)
       -> tokenize_with_weights -> {"anima_t5free_text": text}
       -> encode_token_weights -> the T5-free carrier (Qwen3.5-2B taps
          at layers 16/22 -> d1 grid -> C0 -> C1 -> bf16/fp16 carrier)
    -> standard CONDITIONING; the Anima DiT consumes it raw
       (extra_conds with no t5xxl_ids never touches llm_adapter)

The qwen weights, tokenizer, our config and lexicon all live inside the
single TE safetensors (split_files style), extracted on load.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

_HERE = Path(__file__).resolve().parent
import sys
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from configuration_anima_t5free import AnimaT5FreeConfig   # noqa: E402
from modeling_anima_t5free import AnimaT5FreeModel         # noqa: E402


def _bytes_to_str(t):
    return bytes(t.cpu().numpy().tolist()).decode("utf-8")


def _norm_lex(raw):
    return {k: (int(v[0][0]), tuple(int(r) for r in v[0][1]))
            for k, v in raw.items()}


class AnimaT5FreeTokenizer:
    """Comfy tokenizer API shim: carries the RAW text to the TE."""

    def __init__(self, embedding_directory=None, tokenizer_data={},
                 **kwargs):
        pass

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        return {"anima_t5free_text": text,
                "anima_t5free_pairs": [[(0, 1.0)]]}

    def untokenize(self, token_weight_pair):
        return token_weight_pair

    def state_dict(self):
        return {}

    def decode(self, token_ids, **kwargs):
        return ""


class AnimaT5FreeTEModel(nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={},
                 **kwargs):
        super().__init__()
        self.dtypes = {dtype} if dtype is not None else set()
        self._te_dtype = dtype
        self.cond = None           # AnimaT5FreeModel (conditioner)
        self.qwen = None           # Qwen3.5-2B-Base (transformers)
        self.qtok = None
        self.lex = None
        self.cfg = None

    # ---- comfy CLIP API no-ops ------------------------------------------
    def set_clip_options(self, options):
        pass

    def reset_clip_options(self):
        pass

    # ---- weights: everything comes from the single TE safetensors -------
    def load_sd(self, sd):
        cond_sd = {k: v for k, v in sd.items()
                   if k.startswith("conditioner.")}
        assert cond_sd, "not an Anima-T5-Free TE file (no conditioner.*)"
        self.cfg = AnimaT5FreeConfig(
            **json.loads(_bytes_to_str(sd["t5free_config_bytes"])))
        cond = AnimaT5FreeModel(self.cfg)
        cond.load_state_dict(cond_sd, strict=True)
        self.cond = cond.eval()
        self.lex = _norm_lex(json.loads(
            _bytes_to_str(sd["t5free_lexicon_bytes"])))

        # qwen part -> extract to a local HF-style dir, load via transformers
        exdir = _HERE / "_qwen35_extract"
        exdir.mkdir(parents=True, exist_ok=True)
        (exdir / "config.json").write_text(
            _bytes_to_str(sd["t5free_qwen_config_bytes"]), encoding="utf-8")
        (exdir / "tokenizer.json").write_text(
            _bytes_to_str(sd["t5free_tokenizer_json_bytes"]),
            encoding="utf-8")
        (exdir / "tokenizer_config.json").write_text(
            _bytes_to_str(sd["t5free_tokenizer_config_bytes"]),
            encoding="utf-8")
        qwen_sd = {k: v for k, v in sd.items()
                   if not k.startswith(("conditioner.", "t5free_"))}
        import safetensors.torch
        safetensors.torch.save_file(qwen_sd, str(exdir / "model.safetensors"))
        del qwen_sd
        from transformers import AutoModel, AutoTokenizer
        self.qwen = AutoModel.from_pretrained(
            str(exdir), dtype=torch.float16, local_files_only=True).eval()
        for p in self.qwen.parameters():
            p.requires_grad_(False)
        self.qtok = AutoTokenizer.from_pretrained(
            str(exdir), local_files_only=True)
        return ([], [])

    # ---- encoding: the proven T5-free chain --------------------------------
    def _taps(self, text):
        cache = _HERE / "taps_cache"
        cache.mkdir(parents=True, exist_ok=True)
        tp = cache / (hashlib.sha256(text.encode()).hexdigest()[:16]
                      + ".taps.pt")
        if not tp.is_file():
            enc = self.qtok(text, return_tensors="pt", truncation=False)
            with torch.no_grad():
                hs = self.qwen(
                    **{k: v.to(self.qwen.device) for k, v in enc.items()},
                    output_hidden_states=True).hidden_states
            torch.save({L: hs[L][0].half().cpu()
                        for L in self.cfg.qwen_layers}, tp)
        return tp

    def encode_token_weights(self, token_weight_pairs):
        text = token_weight_pairs["anima_t5free_text"] \
            if isinstance(token_weight_pairs, dict) else token_weight_pairs
        if not str(text).strip():
            z = torch.zeros((1, self.cfg.max_rows, self.cfg.context_dim),
                            dtype=self._te_dtype or torch.bfloat16,
                            device=self.cond.device)
            return (z, None, {})
        tp = self._taps(text)
        carrier = self.cond.conditioning(text, self.qtok, self.lex, str(tp))
        if self._te_dtype is not None:
            carrier = carrier.to(self._te_dtype)
        return (carrier, None, {})


class _AnimaT5FreeClipTarget:
    clip = AnimaT5FreeTEModel
    tokenizer = AnimaT5FreeTokenizer
    params = {}
