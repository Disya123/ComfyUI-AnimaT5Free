# -*- coding: utf-8 -*-
"""f128_te: stock-flow text encoder for the F128 FT line.

AnimaF128Tokenizer + AnimaF128TEModel follow t5free_te.py's comfy CLIP
API shim. The TE FILE (models/text_encoders/anima-f128-receiver-*.safetensors)
carries, split_files-style:

  f128_marker            uint8[1]        -> routes our CLIP branch
  f128_config_bytes      uint8[n]        -> json: t5tok_name, version,
                                           source ckpt sha16, step
  f128_receiver.*        bf16 tensors   -> the 120-tensor R1Copy state
                                            (our trained receiver/adapter)

Qwen3.5-2B-Base + its tokenizer load from the node's _qwen35_extract
(the exact dir t5free_te.py unpacks from the published TE file).

Encoding = the trainer's _capture + build_x_aligned(d1c) chain VERBATIM:
  Qwen fp16 forward (sdpa, no use_cache, output_hidden_states)
  -> H = stack(h[0].to(bf16))                     [26, L, 2048]
  -> T5 offsets as the ROW MAP ONLY (content = pure Qwen)
  -> align_rows (c_v2c_common VERBATIM)
  -> per-layer masked mean over overlapping qwen tokens
  -> FIXED per-layer RMS prenorm (d1c)
  -> X [n, 51200] fp32  == the carrier the DiT patch expects.

Empty prompt -> " " (the training uncond contract:
receiver(S_Q(" ")), the same path - NOT zeros).

X is cached per prompt text in f128_cache/ (sha16 key).
"""
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

_HERE = Path(__file__).resolve().parent

# One package identity: import siblings through the package, so every
# module (loader, hook, runtime) shares the exact same module objects.
from . import f128_carrier                # noqa: E402
from . import f128_dit                    # noqa: E402

NL, RW, EPS = f128_dit.NL, f128_dit.RW, f128_dit.EPS


def _bytes_to_str(t):
    return bytes(t.cpu().numpy().tolist()).decode("utf-8")


def _find_tokenizer_dir(name):
    """anima_l2p.conditioning._find_tokenizer_dir VERBATIM."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(
            name, allow_patterns=["tokenizer*", "vocab*", "merges*",
                                  "spiece.model", "special_tokens_map.json",
                                  "config.json"], local_files_only=True)
    except Exception:
        return snapshot_download(
            name, allow_patterns=["tokenizer*", "vocab*", "merges*",
                                  "spiece.model", "special_tokens_map.json",
                                  "config.json"])


def _align_rows(cap, n, t5, qtok):
    """c_v2c_common.align_rows VERBATIM (numpy char-span overlap)."""
    import numpy as np
    t5o = t5(cap, return_offsets_mapping=True).offset_mapping
    qo = qtok(cap, return_offsets_mapping=True).offset_mapping
    qcen = np.array([(a + b) / 2 for a, b in qo])
    row2tok = []
    for i in range(n):
        a, b = t5o[i]
        toks = [j for j, (qa, qb) in enumerate(qo)
                if qa < b and qb > a] if a < b else []
        if not toks:
            c = (a + b) / 2 if a < b else a
            toks = [int(np.argmin(np.abs(qcen - c)))]
        row2tok.append(toks)
    return row2tok


class AnimaF128Tokenizer:
    """Comfy tokenizer API shim: carries the RAW text to the TE."""

    def __init__(self, embedding_directory=None, tokenizer_data={},
                 **kwargs):
        pass

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        return {"anima_f128_text": text,
                "anima_f128_pairs": [[(0, 1.0)]]}

    def untokenize(self, token_weight_pair):
        return token_weight_pair

    def state_dict(self):
        return {}

    def decode(self, token_ids, **kwargs):
        return ""


class AnimaF128TEModel(nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={},
                 **kwargs):
        super().__init__()
        self.dtypes = {dtype} if dtype is not None else set()
        self._te_dtype = dtype
        self.qwen = None
        self.qtok = None
        self.t5 = None
        self.res = None
        self.cfg = None

    def set_clip_options(self, options):
        pass

    def reset_clip_options(self):
        pass

    # ---- weights: [СЛИТНО] ресивер живёт В UNET-файле; TE-файл = мини-
    # флаг (marker+config). Старые TE-файлы с f128_receiver.* внутри
    # (ft200-формат) по-прежнему работают: их ресивер регистрируется
    # здесь. Qwen — из _qwen35_extract ноды (t5free_te unpack dir).
    def load_sd(self, sd):
        assert "f128_marker" in sd, \
            "not an anima-f128 TE file (no f128_marker)"
        self.cfg = json.loads(_bytes_to_str(sd["f128_config_bytes"]))
        res_sd = {k[len("f128_receiver."):]: v
                  for k, v in sd.items()
                  if k.startswith("f128_receiver.")}
        if res_sd:
            base = (res_sd["Pk"], res_sd["Qk"],
                    res_sd["Pv"], res_sd["Qv"])
            res = f128_dit.R1Copy(base, res_sd["KNW"],
                                  rank_map=f128_dit.RANK_MAP, seed=0)
            res.load_state_dict(res_sd, strict=True)
            self.res = res.eval()
            f128_dit.F128_CTX["res"] = self.res
            print(f"[F128] TE-файл со встроенным ресивером: "
                  f"{len(res_sd)} тензоров (legacy-формат)")
        else:
            self.res = f128_dit.F128_CTX.get("res")
            print("[F128] TE-флаг режима: ресивер "
                  + ("уже от unet-файла" if self.res is not None
                     else "ждём от unet-файла (Load Diffusion "
                          "Model)"))

        exdir = _HERE / "_qwen35_extract"
        from transformers import AutoModel, AutoTokenizer
        self.qwen = AutoModel.from_pretrained(
            str(exdir), dtype=torch.float16, local_files_only=True).eval()
        for p in self.qwen.parameters():
            p.requires_grad_(False)
        self.qtok = AutoTokenizer.from_pretrained(
            str(exdir), local_files_only=True)
        self.t5 = AutoTokenizer.from_pretrained(
            _find_tokenizer_dir(self.cfg.get(
                "t5tok_name", "google/t5-v1_1-xxl")))
        return ([], [])

    # ---- the d1c carrier chain ------------------------------------------
    def _carrier(self, text):
        return build_x(text, self.qwen, self.qtok, self.t5)

    def encode_token_weights(self, token_weight_pairs):
        text = token_weight_pairs["anima_f128_text"] \
            if isinstance(token_weight_pairs, dict) else token_weight_pairs
        text = str(text) if str(text).strip() else " "
        X = self._carrier(text)
        if self._te_dtype is not None:
            X = X.to(self._te_dtype)
        return (X[None], None, {})


# ---- [ХУК ШТАТНОГО QWEN-TE 2026-10-01] -----------------------------------
# Юзер выбирает в CLIP-слот их штатный anima-t5free Qwen-файл (напр.
# qwen_35_2b_base.safetensors) БЕЗ изменений. Если загружен f128-unet
# (ресивер слитно в unet-файле, F128_CTX["res"] установлен), encode
# строит НАШ d1c X-носитель из ТОГО ЖЕ их Qwen-объекта (te.qwen /
# te.qtok); иначе — их штатный C0/C1 carrier (их сток-анима не трога-
# ется). Порядок входов KSampler (model = вход #1) гарантирует загруз-
# ку unet до CLIPTextEncode; обратный случай ловит guard в
# f128_dit.Anima.forward.
_T5TOK = {}


def _shared_t5tok():
    if "t5" not in _T5TOK:
        from transformers import AutoTokenizer
        _T5TOK["t5"] = AutoTokenizer.from_pretrained(
            _find_tokenizer_dir("google/t5-v1_1-xxl"))
    return _T5TOK["t5"]


def build_x(text, qwen, qtok, t5, cache_dir=None):
    """d1c X-carrier [n,51200] bf16 from a full native Qwen forward.

    Trainer contract VERBATIM: T5 row map -> masked mean per group ->
    per-layer RMS; cache f128_cache/*.x.pt. The native-rows carrier is
    built by build_x_native_only (the hook always emits BOTH carriers;
    _extra_conds picks per checkpoint, so the encode side never needs
    to know which model will consume it).
    """
    text = str(text) if str(text).strip() else " "
    cache = Path(cache_dir) if cache_dir else _HERE / "f128_cache"
    cache.mkdir(parents=True, exist_ok=True)
    ck = cache / (hashlib.sha256(text.encode()).hexdigest()[:16] + ".x.pt")
    if ck.is_file():
        return torch.load(ck, map_location="cpu",
                          weights_only=True)["X"]
    with torch.inference_mode():
        # Comfy may move/offload HF modules between prompt encodes.
        # qwen.device is not a strong enough contract after model management;
        # input_ids must live with the actual token embedding weight.
        qdev = qwen.get_input_embeddings().weight.device

        # A direct Qwen forward is valid only while the CLIP wrapper has the
        # complete model loaded.  Detect a split/offloaded model explicitly.
        qdevices = {p.device for p in qwen.parameters()}
        if len(qdevices) != 1 or next(iter(qdevices)) != qdev:
            print(
                "[F128] Qwen split detected during CLIP encode: "
                f"{sorted(str(d) for d in qdevices)}; "
                f"forcing the whole HF Qwen onto {qdev}"
            )
            qwen.to(qdev)
            qdevices = {p.device for p in qwen.parameters()}
            if len(qdevices) != 1 or next(iter(qdevices)) != qdev:
                raise RuntimeError(
                    "[F128] could not reunite Qwen on one device: "
                    f"{sorted(str(d) for d in qdevices)}; expected {qdev}"
                )

        enc = {k: v.to(qdev) for k, v in
               qtok(text, return_tensors="pt").items()}
        out = qwen(**enc, use_cache=False,
                  output_hidden_states=True,
                  return_dict=True)
        H = torch.stack([h[0].to(torch.bfloat16)
                         for h in out.hidden_states]).cpu()
    t5o = t5(text, return_offsets_mapping=True).offset_mapping
    n = len(t5o)
    if n == 0:
        raise ValueError("empty T5 row map")
    r2t = _align_rows(text, n, t5, qtok)
    L = H.shape[1]
    for toks in r2t:
        if not toks or max(toks) >= L:
            raise ValueError(f"f128 alignment gate FAIL: "
                             f"empty group or tok >= L={L}")
    maxg = max(len(t) for t in r2t)
    idx = torch.zeros(n, maxg, dtype=torch.long)
    msk = torch.zeros(n, maxg, dtype=torch.bool)
    for i, toks in enumerate(r2t):
        idx[i, :len(toks)] = torch.tensor(toks)
        msk[i, :len(toks)] = True
    cnt = msk.sum(1, keepdim=True).clamp(min=1).float()
    rows = []
    for l in range(NL):
        g = H[l][idx].float() * msk.unsqueeze(-1)
        rows.append(g.sum(1) / cnt)
    x = torch.cat(rows, dim=1)                      # [n,51200]
    xb = x.view(n, NL, 2048)
    rms = xb.pow(2).mean(dim=(0, 2)).add(EPS).rsqrt()
    x = (xb * rms.view(1, NL, 1)).view(n, NL * 2048)
    X = x.detach().cpu().to(torch.bfloat16)   # эталон: d1c -> bf16
    torch.save({"X": X}, ck)
    return X


def build_x_native_only(text, qwen, qtok, cache_dir=None):
    """Native-rows носитель [Lq,51200] с отдельным кэшем (.xnat.pt).
    Для dual-encode, когда домен модели ещё неизвестен (unet не
    загружен): hook строит d1c + native, extra_conds выбирает."""
    text = str(text) if str(text).strip() else " "
    cache = Path(cache_dir) if cache_dir else _HERE / "f128_cache"
    cache.mkdir(parents=True, exist_ok=True)
    ck = cache / (hashlib.sha256(
        (text + "\x00native").encode()).hexdigest()[:16] + ".xnat.pt")
    if ck.is_file():
        return torch.load(ck, map_location="cpu",
                          weights_only=True)["X"]
    with torch.inference_mode():
        qdev = qwen.get_input_embeddings().weight.device
        qdevices = {p.device for p in qwen.parameters()}
        if len(qdevices) != 1 or next(iter(qdevices)) != qdev:
            qwen.to(qdev)
        enc = {k: v.to(qdev) for k, v in
               qtok(text, return_tensors="pt").items()}
        out = qwen(**enc, use_cache=False,
                  output_hidden_states=True, return_dict=True)
        H = torch.stack([h[0].to(torch.bfloat16)
                         for h in out.hidden_states]).cpu()
    X = f128_carrier.carrier_from_states_native(H)
    torch.save({"X": X}, ck)
    return X


def hook_t5free(te):
    """Precompute F128 carrier during CLIPTextEncode, robust to Comfy offload.

    F128 path:
      Qwen -> d1c X [L,51200] happens HERE while CLIP is loaded.
      If an F128 UNet is already active, the legacy C0/C1 conditioner is
      completely bypassed because its 1024-wide carrier is dead work.

    Order-race / stock compatibility:
      If CLIPTextEncode happens before diffusion-model loading, we cannot know
      yet whether the target model is F128 or stock. In that case we preserve
      the stock carrier, but first co-locate the entire legacy conditioner on
      the actual Qwen embedding device. This prevents Comfy's post-generation
      partial offload state from causing CPU/CUDA index_select failures.
    """
    if getattr(te.encode_token_weights, "_f128_precompute_hook_v5", False):
        return

    orig = te.encode_token_weights

    def encode_token_weights(token_weight_pairs):
        text = token_weight_pairs["anima_t5free_text"] \
            if isinstance(token_weight_pairs, dict) \
            else token_weight_pairs
        text = str(text) if str(text).strip() else " "

        qdev = te.qwen.get_input_embeddings().weight.device

        # Build F128 carriers FIRST, while Comfy has CLIP/Qwen loaded.
        # ALWAYS dual (d1c + native): _extra_conds picks the carrier the
        # loaded checkpoint needs, so the encode side never depends on
        # load order, cached encodes, or global module state.
        Xd = build_x(text, te.qwen, te.qtok, _shared_t5tok())
        Xn = build_x_native_only(text, te.qwen, te.qtok)
        meta = {
            "f128_text": text,
            "f128_x_d1c": Xd,
            "f128_x_native": Xn,
            "f128_carrier_version": f128_carrier.CARRIER_VERSION_DUAL,
        }
        x_shape = f"d1c{tuple(Xd.shape)}+nat{tuple(Xn.shape)}"

        # If F128 UNet is already active, stock C0/C1 is unused.
        if f128_dit.F128_CTX.get("f128_loaded", False):
            dtype = getattr(te, "_te_dtype", None) or torch.bfloat16
            cond = torch.zeros((1, 512, 1024), dtype=dtype, device="cpu")
            print(
                f"[F128] CLIP X {x_shape}; "
                "F128 active -> stock C0/C1 BYPASSED"
            )
            return (cond, None, meta)

        # CLIP may execute before Load Diffusion Model. Preserve stock-Anima
        # compatibility, but repair a split/offloaded conditioner first.
        if getattr(te, "cond", None) is not None:
            before = {p.device for p in te.cond.parameters()}
            if len(before) != 1 or next(iter(before)) != qdev:
                print(
                    "[F128] reuniting stock C0/C1 conditioner on "
                    f"{qdev}; was {sorted(str(d) for d in before)}"
                )
                te.cond.to(qdev)

            after = {p.device for p in te.cond.parameters()}
            if len(after) != 1 or next(iter(after)) != qdev:
                raise RuntimeError(
                    "[F128] failed to co-locate stock conditioner: "
                    f"{sorted(str(d) for d in after)}, expected {qdev}"
                )

        out = orig(token_weight_pairs)
        cond, pooled = out[:2]
        orig_meta = dict(out[2]) if len(out) > 2 and isinstance(out[2], dict) else {}
        orig_meta.update(meta)

        print(
            f"[F128] CLIP X {x_shape}; "
            "model not known yet -> dual carriers + stock carrier produced"
        )
        return (cond, pooled, orig_meta)

    encode_token_weights._f128_precompute_hook_v5 = True
    encode_token_weights.__wrapped__ = orig
    te.encode_token_weights = encode_token_weights
    print("[F128] stock Qwen-TE: v5 CLIP-time precompute hook ACTIVE")


class _AnimaF128ClipTarget:
    clip = AnimaF128TEModel
    tokenizer = AnimaF128Tokenizer
    params = {}
