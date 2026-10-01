"""Training-compatible d1c carrier: Qwen states → T5 row map → [rows, 51200].

T5 supplies character spans only. The first 25 Qwen hidden-state entries,
including the embedding entry, supply every feature. Output is CPU bf16.
"""

import torch

NL, WIDTH, RW, MAX_ROWS, EPS = 25, 2048, 51200, 512, 1e-6
DEFAULT_ROW_TOKENIZER = "google/t5-v1_1-xxl"
CARRIER_VERSION = "d1c-25x2048-bf16-v2"


def normalize_text(text):
    text = str(text)
    return text if text.strip() else " "


def align_rows(row_offsets, qwen_offsets):
    """Character overlap, with the original nearest-centre fallback (EOS too)."""
    if not qwen_offsets:
        raise ValueError("Qwen tokenizer returned no offsets")
    centres = [(a + b) / 2 for a, b in qwen_offsets]
    groups = []
    for a, b in row_offsets:
        group = [i for i, (qa, qb) in enumerate(qwen_offsets) if qa < b and qb > a] if a < b else []
        if not group:
            centre = (a + b) / 2 if a < b else a
            group = [min(range(len(centres)), key=lambda i: abs(centres[i] - centre))]
        groups.append(group)
    return groups


def carrier_from_states(hidden_states, row_offsets, qwen_offsets):
    """Preserve bf16 capture, fp32 span means and per-layer RMS prenorm."""
    n = len(row_offsets)
    if not 1 <= n <= MAX_ROWS:
        raise ValueError(f"F128 requires 1..{MAX_ROWS} rows (including EOS); got {n}")
    if len(hidden_states) < NL:
        raise ValueError(f"F128 requires at least {NL} Qwen hidden-state entries")
    H = torch.stack([h.to(device="cpu", dtype=torch.bfloat16) for h in hidden_states[:NL]])
    if H.ndim != 3 or H.shape[2] != WIDTH:
        raise ValueError(f"F128 expected Qwen states [layers,tokens,{WIDTH}], got {H.shape}")
    if len(qwen_offsets) != H.shape[1]:
        raise ValueError("Qwen offsets do not match the captured token sequence")
    groups = align_rows(row_offsets, qwen_offsets)
    max_group = max(map(len, groups))
    indices = torch.zeros(n, max_group, dtype=torch.long)
    mask = torch.zeros(n, max_group, dtype=torch.bool)
    for i, group in enumerate(groups):
        indices[i, : len(group)] = torch.tensor(group)
        mask[i, : len(group)] = True
    count = mask.sum(1, keepdim=True).clamp(min=1).float()
    rows = [(H[layer][indices].float() * mask.unsqueeze(-1)).sum(1) / count for layer in range(NL)]
    x = torch.cat(rows, dim=1).view(n, NL, WIDTH)
    rms = x.pow(2).mean(dim=(0, 2)).add(EPS).rsqrt()
    result = (x * rms.view(1, NL, 1)).view(n, RW).to(torch.bfloat16)
    if not torch.isfinite(result).all():
        raise ValueError("Qwen produced non-finite F128 features")
    return result


def load_row_tokenizer(name=DEFAULT_ROW_TOKENIZER):
    from transformers import AutoTokenizer

    # Hugging Face handles the local cache, offline mode and downloads.
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("F128 row mapping requires a fast tokenizer with offsets")
    return tokenizer


def capture_states(qwen, tokenizer, text):
    """One complete forward, during CLIP encode, shared by both text paths."""
    device = qwen.get_input_embeddings().weight.device
    devices = {p.device for p in qwen.parameters()} | {b.device for b in qwen.buffers()}
    if devices != {device}:
        raise RuntimeError(
            f"Qwen is split across {sorted(map(str, devices))}. "
            "Use the package's full-load CLIP wrapper and restart ComfyUI."
        )
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    with torch.inference_mode():
        output = qwen(
            **{k: v.to(device) for k, v in enc.items()},
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        states = tuple(h[0].to(device="cpu", dtype=torch.float16) for h in output.hidden_states)
    return states
