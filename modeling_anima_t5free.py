"""AnimaT5FreeModel - T5-free text conditioning frontend for Anima.

Goal of this release is compatibility/parity, not semantic enhancement.

Pipeline:
  prompt text
    -> Qwen3.5-2B-Base hidden taps at the configured layers (external
       dependency, config.text_encoder)
    -> d1 char segmenter + lexicon -> row spans (T5-free grid)
    -> per-row features x_s = concat(mean taps per layer), standardized
    -> C0 = mu(pos,len) + Xs @ W                       [frozen ridge]
    -> C1 residual refiner over frozen C0              [frozen Phase-R]
    -> carrier context (rows, context_dim), bfloat16
    -> frozen Anima DiT (external dependency, config.base_model):
       Euler over `steps` sigmas, CFG as configured; the uncond arm
       consumes no conditioner
    -> latent -> VAE decode (external; see README for the ComfyUI path)

No T5 model and no T5 tokenizer are loaded anywhere in this module.

"""

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel

from .configuration_anima_t5free import AnimaT5FreeConfig


# ---- C1 residual refiner (frozen Phase-R architecture) --------------------
class Block(nn.Module):
    """Pre-LN bidir block; attention over ACTIVE rows only (pad keys get a
    finite -1e4 so pad-row queries stay finite; their output is zeroed at
    the refiner head)."""

    def __init__(self, width, heads, mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.ln2 = nn.LayerNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.mlp = nn.Sequential(nn.Linear(width, mlp), nn.GELU(), nn.Linear(mlp, width))
        self.scale = width**-0.5
        self.nheads = heads
        self.width = width

    def forward(self, h, active):
        B, T, _ = h.shape
        x = self.ln1(h)
        w, nh = self.width, self.nheads
        q = self.q(x).reshape(B, T, nh, -1).transpose(1, 2)
        k = self.k(x).reshape(B, T, nh, -1).transpose(1, 2)
        v = self.v(x).reshape(B, T, nh, -1).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) * self.scale
        a = a.masked_fill(~active[:, None, None, :], -1e4)
        a = a.softmax(dim=-1)
        att = (a @ v).transpose(1, 2).reshape(B, T, w)
        h = h + self.o(att)
        return h + self.mlp(self.ln2(h))


class C1Refiner(nn.Module):
    """x_s (B,R,4096) fp32, c0 (B,R,1024) fp32, active (B,R) bool
    -> pred = c0 + Delta, Delta zero on pad rows."""

    def __init__(self, width, heads, mlp, blocks, in_dim, out_dim):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([Block(width, heads, mlp) for _ in range(blocks)])
        self.out = nn.Linear(width, out_dim)

    def forward(self, x_s, c0, active):
        h = self.inp(torch.cat([c0, x_s], dim=-1))
        for blk in self.blocks:
            h = blk(h, active)
        return c0 + self.out(h) * active.unsqueeze(-1)


# ---- d1 char segmenter -----------------------------------------------------
class CharSegmenter(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.emb = nn.Embedding(257, 32)
        self.conv = nn.Sequential(
            nn.Conv1d(32, hidden, 5, dilation=1, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, dilation=2, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, dilation=4, padding=8),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, dilation=8, padding=16),
            nn.GELU(),
        )
        self.out = nn.Conv1d(hidden, 1, 1)

    def forward(self, x):  # (B, L) byte ids
        h = self.conv(self.emb(x).transpose(1, 2))
        return self.out(h).squeeze(1)  # (B, L) start logits


# ---- C0 ridge ---------------------------------------------------------------
class C0(nn.Module):
    """Frozen ridge: x_s -> mu(pos,len) + Xs @ W. All tensors are buffers."""

    def __init__(self, feature_dim, context_dim, mu_keys):
        super().__init__()
        self.register_buffer("W", torch.zeros(feature_dim, context_dim))
        self.register_buffer("m", torch.zeros(feature_dim))
        self.register_buffer("s", torch.zeros(feature_dim))
        self.register_buffer("mu_table", torch.zeros(len(mu_keys), context_dim))
        self.register_buffer("lam", torch.zeros(()))
        self.register_buffer("n_train", torch.zeros((), dtype=torch.int64))
        self.mu_keys = list(mu_keys)


# ---- model -------------------------------------------------------------------
class AnimaT5FreeModel(PreTrainedModel):
    config_class = AnimaT5FreeConfig
    base_model_prefix = "conditioner"

    def __init__(self, config):
        super().__init__(config)
        mu_keys = list(getattr(config, "mu_keys", None) or [])
        self.conditioner = nn.Module()
        self.conditioner.refiner = C1Refiner(
            config.refiner_width,
            config.refiner_heads,
            config.refiner_mlp,
            config.refiner_blocks,
            config.context_dim + config.feature_dim,
            config.context_dim,
        )
        self.conditioner.segmenter = CharSegmenter(config.segmenter_hidden)
        self.conditioner.c0 = C0(config.feature_dim, config.context_dim, mu_keys)
        self.mu_keys = list(mu_keys)
        self.post_init()

    # ---- bucket helpers ---------------------------------------------------
    def _pb(self, i):
        for a, b in self.config.pos_buckets:
            if a <= i < b:
                return f"[{a},{b})"
        return f"[{self.config.pos_buckets[-1][0]},{self.config.pos_buckets[-1][1]})"

    def _lb(self, n):
        for a, b in self.config.len_buckets:
            if a <= n < b:
                return f"L[{a},{b})"
        return f"L[{self.config.len_buckets[-1][0]},{self.config.len_buckets[-1][1]})"

    def _mu(self):
        tab = self.conditioner.c0.mu_table
        out = {}
        for i, k in enumerate(self.mu_keys):
            j = k.index(")") + 1
            out[(k[:j], k[j:])] = tab[i]
        return out

    def mu_rows(self, n):
        mu = self._mu()
        return torch.stack([mu[(self._pb(i), self._lb(n))] for i in range(n)])

    # ---- T5-free grid ----------------------------------------------------
    @staticmethod
    def units_of(cap):
        out = []
        i = 0
        while i < len(cap):
            if cap[i] == " ":
                i += 1
                continue
            j = i
            while j < len(cap) and cap[j] != " ":
                j += 1
            out.append((i, j))
            i = j
        return out

    def seg_logits(self, cap):
        """Byte-level cut logits; None for non-ASCII captions (OOV path:
        lexicon-only spans)."""
        if len(cap.encode("utf-8", errors="replace")) != len(cap):
            return None
        if len(cap) > self.config.max_prompt_bytes:
            raise ValueError(f"C0/C1 prompt exceeds {self.config.max_prompt_bytes} bytes")
        ids = torch.tensor(
            [
                min(255, c)
                for c in cap.encode("utf-8", errors="replace")[: self.config.max_prompt_bytes]
            ]
        )
        with torch.no_grad():
            return self.conditioner.segmenter(ids.unsqueeze(0).to(self.device))[0].sigmoid().cpu()

    def predicted_spans(self, cap, lex, logits):
        """T5-free grid: list of row char-spans, plus the EOS row (0, 0)."""
        spans = []
        for b, e in self.units_of(cap):
            key = cap[b:e]
            if key in lex:
                n_ph, rel = lex[key]
            else:
                n_ph, rel = 0, ()
                if logits is not None:
                    rel = tuple(
                        int(a) - b
                        for a in logits.index_select(0, torch.arange(b + 1, e))
                        .gt(0.5)
                        .nonzero()
                        .flatten()
                        .add(b + 1)
                        .tolist()
                    )
            cuts = [b] + [b + r for r in rel]
            spans.extend([(b, b + 1)] * n_ph)
            spans.append((cuts[0], cuts[1] if len(cuts) > 1 else e))
            for i in range(1, len(cuts)):
                spans.append((cuts[i], cuts[i + 1] if i + 1 < len(cuts) else e))
        spans.append((0, 0))
        return spans

    # ---- features + C0 ---------------------------------------------------
    def build_x_c0(self, caption, spans, taps, qtok):
        n = len(spans)
        if not 1 <= n <= self.config.max_rows:
            raise ValueError(f"C0/C1 needs 1..{self.config.max_rows} rows; got {n}")
        c0m = self.conditioner.c0
        qo = qtok(caption, return_offsets_mapping=True).offset_mapping
        qcen = np.array([(a + b) / 2 for a, b in qo])
        row2tok = []
        for a, b in spans:
            toks = [j for j, (qa, qb) in enumerate(qo) if qa < b and qb > a] if a < b else []
            if not toks:
                cc = (a + b) / 2 if a < b else a
                toks = [int(np.argmin(np.abs(qcen - cc)))]
            row2tok.append(toks)
        layers = self.config.qwen_layers
        X = torch.stack(
            [torch.cat([taps[L][toks].mean(0) for L in layers]).float() for toks in row2tok]
        ).to(c0m.W.device)
        Xs = (X - c0m.m) / c0m.s
        # bit-parity contract: the original v2c ridge stores W
        # column-major (strides (1, 4096)); this transposed view
        # reproduces that exact cuBLAS path, matching the proven
        # chain bit-for-bit.
        W = c0m.W.t().contiguous().t()
        c0 = (self.mu_rows(n).to(c0m.W.device) + Xs @ W).cpu()
        x_s = Xs.half().cpu()
        if not torch.isfinite(x_s).all() or not torch.isfinite(c0).all():
            raise ValueError("C0/C1 produced non-finite features")
        return x_s, c0

    # ---- carrier ----------------------------------------------------------
    @torch.no_grad()
    def conditioning(self, prompt, qtok, lex, taps):
        """-> carrier context (1, max_rows, context_dim), bfloat16."""
        spans = self.predicted_spans(prompt, lex, self.seg_logits(prompt))
        x_s, c0 = self.build_x_c0(prompt, spans, taps, qtok)
        R = self.config.max_rows
        n = x_s.shape[0]
        x = torch.zeros(1, R, self.config.feature_dim, dtype=torch.float16)
        c0b = torch.zeros(1, R, self.config.context_dim)
        active = torch.zeros(1, R, dtype=torch.bool)
        x[0, :n] = x_s
        c0b[0, :n] = c0
        active[0, :n] = True
        pred = self.conditioner.refiner(
            x.to(self.device).float(), c0b.to(self.device), active.to(self.device)
        )
        return pred.to(torch.bfloat16)
