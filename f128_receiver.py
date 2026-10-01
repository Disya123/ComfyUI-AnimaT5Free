"""F128 receiver and explicit cross-attention math, independent of ComfyUI."""

from contextlib import nullcontext

import torch
from torch import nn

from .f128_carrier import EPS, MAX_ROWS, RW

NB, NH, HD = 28, 16, 128
SCALE = HD**-0.5


class F128Receiver(nn.Module):
    """Inference skeleton inferred from the checkpoint, loaded strictly.

    Base projections remain bf16; residual matrices and gains remain fp32,
    matching the original R1Copy load_state_dict conversion. State-dict names
    remain unchanged. KNW is retained for checkpoint compatibility.
    """

    def __init__(self, state_dict):
        super().__init__()
        for name in ("Pk", "Qk", "Pv", "Qv", "KNW", "gk", "gv", "s"):
            if name not in state_dict:
                raise ValueError(f"F128 receiver is missing {name}")
        rank = state_dict["Pk"].shape[-1]
        shapes = {
            "Pk": (NB, RW, rank),
            "Pv": (NB, RW, rank),
            "Qk": (NB, rank, NH * HD),
            "Qv": (NB, rank, NH * HD),
            "gk": (NB,),
            "gv": (NB,),
            "s": (NB, NH),
        }
        for name, shape in shapes.items():
            if tuple(state_dict[name].shape) != shape:
                raise ValueError(f"F128 {name}: expected {shape}, got {state_dict[name].shape}")
        for name in ("Pk", "Qk", "Pv", "Qv"):
            self.register_buffer(name, torch.empty_like(state_dict[name], dtype=torch.bfloat16))
        self.register_buffer("KNW", torch.empty_like(state_dict["KNW"]))
        for name in ("Bk", "Ak", "Bv", "Av"):
            matrices = nn.ParameterList()
            for b in range(NB):
                key = f"{name}.{b}"
                if key not in state_dict:
                    raise ValueError(f"F128 receiver is missing {key}")
                r = state_dict[f"Bk.{b}"].shape[0]
                expected = (r, RW) if name in ("Bk", "Bv") else (NH * HD, r)
                if tuple(state_dict[key].shape) != expected:
                    raise ValueError(
                        f"F128 {key}: expected {expected}, got {state_dict[key].shape}"
                    )
                matrices.append(
                    nn.Parameter(torch.empty(expected, dtype=torch.float32), requires_grad=False)
                )
            setattr(self, name, matrices)
        for name in ("gk", "gv", "s"):
            setattr(
                self,
                name,
                nn.Parameter(torch.empty(shapes[name], dtype=torch.float32), requires_grad=False),
            )
        self.base_rank = rank
        self.load_state_dict(state_dict, strict=True)
        self.eval()

    def block_kv(self, block, features):
        batch, rows, _ = features.shape
        flat = features.reshape(batch * rows, RW)
        hidden = flat @ torch.cat(
            (self.Pk[block].to(flat.dtype), self.Pv[block].to(flat.dtype)), dim=1
        )
        k = hidden[:, : self.base_rank] @ self.Qk[block].to(hidden.dtype)
        v = hidden[:, self.base_rank :] @ self.Qv[block].to(hidden.dtype)
        k = k.float().view(batch, rows, NH, HD)
        v = v.float().view(batch, rows, NH, HD)
        h = flat.float()
        h = h * h.pow(2).mean(-1, keepdim=True).add(EPS).rsqrt()
        dk = torch.nn.functional.silu(h @ self.Bk[block].t()) @ self.Ak[block].t()
        dv = torch.nn.functional.silu(h @ self.Bv[block].t()) @ self.Av[block].t()
        k = k + self.gk[block] * dk.view(batch, rows, NH, HD)
        v = v * self.s[block].view(1, 1, NH, 1) + self.gv[block] * dv.view(batch, rows, NH, HD)
        k_grid = k.new_zeros(batch, MAX_ROWS, NH, HD)
        v_grid = v.new_zeros(batch, MAX_ROWS, NH, HD)
        k_grid[:, :rows] = k
        v_grid[:, :rows] = v
        return (
            k_grid.permute(0, 2, 1, 3).contiguous(),
            v_grid.permute(0, 2, 1, 3).contiguous(),
        )

    @torch.no_grad()
    def forward(self, context, autocast=True):
        if context.ndim != 3 or context.shape[-1] != RW:
            raise ValueError(f"F128 context must have shape [batch,rows,{RW}]")
        if not 1 <= context.shape[1] <= MAX_ROWS or context.shape[0] < 1:
            raise ValueError(f"F128 context has invalid shape {tuple(context.shape)}")
        # Plain torch projections cannot use Comfy's per-layer dynamic casts.
        # This registered module follows its owning model on full offload.
        self.to(device=context.device)
        cm = (
            torch.autocast(context.device.type, dtype=torch.bfloat16) if autocast else nullcontext()
        )
        with cm:
            features = context.float()
            return tuple(self.block_kv(b, features) for b in range(NB))


def explicit_attention(attention, x, kv):
    """Training path: pre-k_norm K → bf16 norm → fp32 softmax and AV."""
    k, v = kv
    if k.shape[0] != x.shape[0] or v.shape != k.shape:
        raise ValueError("F128 attention batch or K/V shapes do not match")
    q = attention.q_proj(x).view(x.shape[0], x.shape[1], NH, HD).transpose(1, 2)
    q = attention.q_norm(q)
    k = attention.k_norm(k.to(torch.bfloat16))
    # Autocast must be disabled here: these two matmuls are explicitly fp32.
    with torch.autocast(x.device.type, enabled=False):
        weights = torch.softmax((q.float() @ k.float().transpose(-1, -2)) * SCALE, dim=-1)
        result = weights @ v.float()
    result = result.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1).contiguous()
    return attention.output_proj(result.to(x.dtype))
