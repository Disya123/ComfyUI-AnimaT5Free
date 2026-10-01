# -*- coding: utf-8 -*-
"""f128_dit: F128 receiver inference + tar-injection for ComfyUI-AnimaT5Free.

Loads our FT-trained DiT (unet file: net.* bf16 + f128_marker) through
the STOCK "Load Diffusion Model" node and injects the F128 carrier the
same way the trainer/court did:

  context (B, L, 51200)  [the d1c X-carrier produced by f128_te]
    -> z_from(res, Xs)            [bf16 autocast, port of ft_train z_from]
    -> holder.z = 28 x (zK, zV)   [(B,16,512,128) fp32, PRE-k_norm K]
    -> patched cross_attn.forward: the N0-certified explicit CA math
       VERBATIM (tar28_model.install_tar): q=q_norm(q_proj(x)),
       k=k_norm(zK bf16), v=zV, fp32 softmax * SCALE, o_proj.

Anima.forward class-patch: a 51200-wide context is OUR carrier; the
cosmos code path gets a dummy bf16 tensor (1,1,1024) only for the
final-layer cast, and F128_ACTIVE routes every cross_attn to the
holder instead of the (removed-from-training) llm_adapter path.
"""
import torch
from torch import nn

NB, NH, HD, NGRID = 28, 16, 128, 512
NL = 25
RW = NL * 2048                      # 51200
EPS = 1e-6
SCALE = 1.0 / (HD ** 0.5)
RANK_LO, RANK_HI, SPLIT = 4, 16, 20
RANK_MAP = {b: 128 for b in range(3, 17)}   # F128 probe rank map

F128_CTX = {"res": None, "holder": None, "installed": set(), "f128_loaded": False}
F128_ACTIVE = False


def rmsnorm_rows(x):
    """hm1_r1c_canary.rmsnorm_rows VERBATIM."""
    return x * x.pow(2).mean(-1, keepdim=True).add(EPS).rsqrt()


def base_rows(b, Xb, Pk, Qk, Pv, Qv, rank):
    """hm1_r1c_canary.base_rows VERBATIM (rank passed in)."""
    K, Lm = Xb.shape[0], Xb.shape[1]
    flat = Xb.reshape(K * Lm, RW)
    hid = flat @ torch.cat((Pk[b].to(flat.dtype),
                            Pv[b].to(flat.dtype)), dim=1)
    k_pre = (hid[:, :rank] @ Qk[b].to(hid.dtype)) \
        .float().view(K, Lm, NH, HD)
    v_pre = (hid[:, rank:] @ Qv[b].to(hid.dtype)) \
        .float().view(K, Lm, NH, HD)
    return k_pre, v_pre


class R1Copy(nn.Module):
    """Skeleton of hm1_r1c_canary.R1Copy with the F128 rank map; all
    weights (incl. KNW/base) arrive through load_state_dict from the
    f128_receiver.* tensors shipped in the TE file (strict)."""

    def __init__(self, base, knw, rank_map=None, seed=0):
        super().__init__()
        Pk, Qk, Pv, Qv = base
        for name, t in (("Pk", Pk), ("Qk", Qk),
                        ("Pv", Pv), ("Qv", Qv)):
            self.register_buffer(name, t.detach().clone().bfloat16())
        self.register_buffer("KNW", knw.detach().clone())
        self.blocks = list(range(NB))
        self.bidx = {gb: i for i, gb in enumerate(self.blocks)}
        self.base_idx = {gb: gb for gb in range(NB)}
        self.base_rank = self.Pk.shape[2]
        g = torch.Generator().manual_seed(seed)
        self.ranks = []
        self.Bk = nn.ParameterList()
        self.Ak = nn.ParameterList()
        self.Bv = nn.ParameterList()
        self.Av = nn.ParameterList()
        for b in self.blocks:
            r = rank_map.get(b, RANK_HI if b >= SPLIT else RANK_LO) \
                if rank_map else None
            if r is None:
                r = RANK_HI if b >= SPLIT else RANK_LO
            self.ranks.append(r)
            self.Bk.append(nn.Parameter(
                torch.randn(r, RW, generator=g) * 0.02))
            self.Ak.append(nn.Parameter(torch.zeros(2048, r)))
            self.Bv.append(nn.Parameter(
                torch.randn(r, RW, generator=g) * 0.02))
            self.Av.append(nn.Parameter(torch.zeros(2048, r)))
        ng = len(self.blocks)
        self.gk = nn.Parameter(torch.ones(ng))
        self.gv = nn.Parameter(torch.ones(ng))
        self.s = nn.Parameter(torch.ones(ng, NH))


def fa1_kv(res, b, Xb):
    """fa1_receiver.fa1_kv VERBATIM (device follows Xb)."""
    K, Lm = Xb.shape[0], Xb.shape[1]
    k_lin, v_lin = base_rows(res.base_idx[b], Xb, res.Pk, res.Qk,
                             res.Pv, res.Qv, rank=res.base_rank)
    i = res.bidx[b]
    h = rmsnorm_rows(Xb.reshape(K * Lm, RW).float())
    dk = torch.nn.functional.silu(h @ res.Bk[i].t()) @ res.Ak[i].t()
    dv = torch.nn.functional.silu(h @ res.Bv[i].t()) @ res.Av[i].t()
    kc = k_lin + res.gk[i] * dk.view(K, Lm, NH, HD)
    vc = v_lin * res.s[i].view(1, 1, NH, 1) \
        + res.gv[i] * dv.view(K, Lm, NH, HD)
    k_grid = kc.new_zeros(K, NGRID, NH, HD)
    v_grid = vc.new_zeros(K, NGRID, NH, HD)
    k_grid[:, :Lm] = kc
    v_grid[:, :Lm] = vc
    return k_grid, v_grid


def to_install(k_grid, v_grid):
    """fa1_receiver.to_install VERBATIM."""
    kp, vp = k_grid.permute(0, 2, 1, 3), v_grid.permute(0, 2, 1, 3)
    return [(kp[s:s + 1].contiguous(), vp[s:s + 1].contiguous())
            for s in range(kp.shape[0])]


def z_from(res, Xs, autocast=True):
    """Port of ft_train.z_from, kv_batch arm (the only correct arm for
    B>1 CFG batches; the trainer gated it at 1e-5). Zero pad rows give
    strictly zero K/V rows (zero-pad contract). Returns the 28x
    (zK, zV) list for the holder."""
    cm = (torch.autocast(Xb_device(Xs).type, dtype=torch.bfloat16)
          if autocast else torch.autocast("cpu", enabled=False))
    z = [[] for _ in range(NB)]
    with torch.no_grad(), cm:
        Lm = max(int(X.shape[0]) for X in Xs)
        Xb = torch.zeros((len(Xs), Lm, Xs[0].shape[1]),
                         device=Xs[0].device, dtype=torch.float32)
        for i, X in enumerate(Xs):
            Xb[i, :X.shape[0]] = X.float()
        for b in range(NB):
            k_pre, v_g = fa1_kv(res, b, Xb)
            for kb, vb in to_install(k_pre, v_g):
                z[b].append((kb, vb))
    return [(torch.cat([k for k, _ in z[b]], 0),
             torch.cat([v for _, v in z[b]], 0)) for b in range(NB)]


def Xb_device(Xs):
    return Xs[0].device


def _rearrange_q(x, ca):
    b, l, _ = x.shape
    q = ca.q_proj(x).view(b, l, ca.n_heads, ca.head_dim)
    return q.transpose(1, 2)              # b h l d


def install_tar(model, holder):
    """Port of tar28_model.install_tar onto comfy's cosmos Attention
    (same member names: q_proj/q_norm/k_norm/output_proj). The
    forward swap checks F128_ACTIVE so stock models stay untouched."""
    for bi, blk in enumerate(model.blocks):
        ca = blk.cross_attn
        orig = ca.forward

        def make(bi, ca, orig):
            def fwd(x, context=None, rope_emb=None,
                    transformer_options=None, **kw):
                if not F128_ACTIVE:
                    return orig(x, context, rope_emb=rope_emb,
                                transformer_options=transformer_options,
                                **kw)
                zK, zV = holder.z[bi]
                q = ca.q_norm(_rearrange_q(x, ca))
                k = ca.k_norm(zK.to(torch.bfloat16))
                v = zV
                att = torch.softmax(
                    (q.float() @ k.float().transpose(-1, -2)) * SCALE,
                    dim=-1)
                out = att @ v.float()
                out = out.transpose(1, 2).reshape(
                    x.shape[0], x.shape[1], -1).contiguous()
                return ca.output_proj(out.to(x.dtype))
            return fwd

        ca.forward = make(bi, ca, orig)


def ensure_installed(model):
    key = id(model)
    if key in F128_CTX["installed"]:
        return
    if F128_CTX["holder"] is None:
        F128_CTX["holder"] = type("H", (), {"z": None})()
    install_tar(model, F128_CTX["holder"])
    F128_CTX["installed"].add(key)



def install_f128_cond_bridge(base):
    """For F128 BaseModel, replace stock 1024 conditioning by the F128 carrier
    precomputed during CLIPTextEncode.

    NO Qwen forward is allowed here.  KSampler runs after CLIP model management
    may offload Qwen, so sampler-time Qwen execution is inherently unsafe.
    """
    if getattr(base, "_f128_cond_bridge_v4", False):
        return

    import types
    orig_extra_conds = base.extra_conds

    def extra_conds_f128(this, **kwargs):
        cross_attn = kwargs.get("cross_attn", None)
        X = kwargs.get("f128_x", None)

        # Dedicated F128 TE can already provide [B,L,51200].
        if torch.is_tensor(cross_attn) and cross_attn.dim() == 3 \
                and cross_attn.shape[-1] == RW:
            return orig_extra_conds(**kwargs)

        if X is None:
            raise RuntimeError(
                "[F128] F128 model received no precomputed f128_x metadata. "
                "The stock Qwen-TE v4 CLIP-time hook did not run."
            )
        if not torch.is_tensor(X):
            raise RuntimeError(
                f"[F128] f128_x metadata is not a tensor: {type(X)}"
            )

        # Conditioning metadata stores one prompt carrier [L,51200].
        # Normalize any accidental leading singleton batch.
        while X.dim() > 2 and X.shape[0] == 1:
            X = X[0]
        if X.dim() != 2 or X.shape[-1] != RW:
            raise RuntimeError(
                f"[F128] bad precomputed f128_x shape {tuple(X.shape)}; "
                f"expected [L,{RW}]"
            )

        kwargs = dict(kwargs)
        kwargs["cross_attn"] = X.unsqueeze(0)

        if not getattr(this, "_f128_bridge_logged_v4", False):
            print(
                "[F128] v4 cond bridge ACTIVE: "
                f"precomputed X {tuple(kwargs['cross_attn'].shape)}; "
                "NO sampler-time Qwen forward"
            )
            this._f128_bridge_logged_v4 = True

        return orig_extra_conds(**kwargs)

    base.extra_conds = types.MethodType(extra_conds_f128, base)
    base._f128_cond_bridge_v4 = True

def patch_anima_forward():
    """Class-level patch of comfy.ldm.anima.model.Anima.forward:
    our 51200-wide context -> carrier path; anything else -> stock."""
    import comfy.ldm.anima.model as _am
    if getattr(_am.Anima, "_f128_patched", False):
        return
    _orig = _am.Anima.forward

    def forward(self, x, timesteps, context, **kwargs):
        res = F128_CTX["res"]
        if torch.is_tensor(context) and context.dim() == 3:
            if context.shape[-1] == RW:
                if res is None:
                    raise RuntimeError(
                        "[F128] TE = f128 (X-carrier 51200), но ресивер "
                        "НЕ загружен: UNet-файл не f128. Выбери "
                        "anima-ft*-f128.safetensors в Load Diffusion "
                        "Model — тихо-полумодель запрещена.")
                # Ресивер собран на CPU (из unet-файла); comfy тащит
                # conditioning на device модели -> один перенос (no-op
                # при повторе). Без этого fa1_kv падает на
                # CUDA-mm с CPU mat2 (2026-10-01, лог юзера).
                res = res.to(context.device)
                F128_CTX["res"] = res
                ensure_installed(self)
                Xs = [context[i] for i in range(context.shape[0])]
                F128_CTX["holder"].z = z_from(res, Xs)
                # comfy/cosmos держит residual в fp32 и кастует вход
                # final_layer к dtype crossattn_emb (predict2:930);
                # весы модели (напр. fp16) должны встретить вход
                # СВОЕГО dtype -> dummy строим в dtype весов final_layer,
                # как их штатный Half-carrier (2026-10-01, лог юзера).
                wdtype = self.final_layer.linear.weight.dtype
                dummy = context.new_zeros(
                    (context.shape[0], 1, 1024)).to(wdtype)
                # Comfy ModelSamplingFlux.timestep(sigma) == sigma.
                # FT was trained with t == sigma in [0,1].
                # NEVER divide this value by 1000.
                global F128_ACTIVE
                F128_ACTIVE = True
                try:
                    if not getattr(self, "_f128_t_logged", False):
                        tmin = float(timesteps.float().min())
                        tmax = float(timesteps.float().max())
                        print(f"[F128] DiT timestep PASS-THROUGH: [{tmin:.6g}, {tmax:.6g}]")
                        if tmax <= 0.002:
                            raise RuntimeError(
                                "[F128] timestep unexpectedly ~1e-3; "
                                "expected Comfy/FT sigma scale ~[0,1]"
                            )
                        self._f128_t_logged = True
                    return _orig(self, x, timesteps, dummy,
                                 **kwargs)
                finally:
                    F128_ACTIVE = False
            if res is not None and context.shape[-1] in (1024, 2048) \
                    and context.shape[1] > 1:
                # f128-модель + ЧУЖОЙ конд (штатный t5free C0/C1 carrier
                # / прочий TE) = генерация БЕЗ ресивера. Такой тест
                # 2026-10-01 тихо прошёл у юзера и стоил неверных
                # выводов — теперь это громкая ошибка.
                raise RuntimeError(
                    "[F128] UNet = f128-FT, но контекст "
                    f"{tuple(context.shape)} — это НЕ f128-носитель. "
                    "Либо в CLIP-слоте НЕ anima-t5free Qwen-файл "
                    "(поставь штатный qwen_35_2b_base), либо CLIP-"
                    "encode исполнлся ДО загрузки unet (редкий "
                    "порядок нод) — перезапусти Comfy и прогони "
                    "снова.")
        return _orig(self, x, timesteps, context, **kwargs)

    forward._f128_patched = True
    forward.__wrapped__ = _orig
    _am.Anima.forward = forward


def patch_load_diffusion():
    """Wrap comfy.sd.load_diffusion_model for F128 RAW latent space.

    ComfyUI's stock Anima config uses Wan21 latent normalization.  The F128
    trainer/court used RAW qwen-VAE coordinates, so our model must expose an
    identity process_in/process_out while retaining Wan21 metadata such as
    channel count and temporal layout.

    Important: comfy.sd.load_diffusion_model returns a ModelPatcher.  The
    BaseModel used by sampling is ModelPatcher.model and it copied
    model_config.latent_format into BaseModel.latent_format during __init__.
    Therefore changing only model_config.latent_format after loading is too
    late; BaseModel.latent_format itself must be replaced.
    """
    import comfy.sd
    import comfy.latent_formats as lf

    if getattr(comfy.sd.load_diffusion_model, "_f128_patched", False):
        return

    class _RawWan21(lf.Wan21):
        """Wan21 metadata with identity latent coordinate transforms."""

        def __init__(self):
            super().__init__()
            self.scale_factor = 1.0
            self.latents_mean = torch.zeros_like(self.latents_mean)
            self.latents_std = torch.ones_like(self.latents_std)

    def _assert_raw_identity(fmt):
        """Fail loudly if a future ComfyUI change breaks the RAW contract."""
        if float(fmt.scale_factor) != 1.0:
            raise RuntimeError("F128 RAW latent patch FAIL: scale_factor != 1")
        if torch.count_nonzero(fmt.latents_mean).item() != 0:
            raise RuntimeError("F128 RAW latent patch FAIL: mean is not zero")
        if not torch.equal(fmt.latents_std, torch.ones_like(fmt.latents_std)):
            raise RuntimeError("F128 RAW latent patch FAIL: std is not one")

        probe = torch.randn((1, 16, 1, 4, 4), dtype=torch.float32)
        if not torch.equal(fmt.process_in(probe), probe):
            raise RuntimeError("F128 RAW latent patch FAIL: process_in not identity")
        if not torch.equal(fmt.process_out(probe), probe):
            raise RuntimeError("F128 RAW latent patch FAIL: process_out not identity")

    _orig = comfy.sd.load_diffusion_model

    def _ldm_f128(unet_path, model_options={}, disable_dynamic=False):
        model_patcher = _orig(
            unet_path,
            model_options=model_options,
            disable_dynamic=disable_dynamic,
        )

        ours = False
        try:
            from safetensors import safe_open
            with safe_open(str(unet_path), framework="pt") as f:
                ours = "f128_marker" in f.keys()
        except Exception as exc:
            # Do not break stock models just because marker inspection failed.
            print(f"[F128] marker check skipped for {unet_path}: {exc}")

        if not ours or model_patcher is None:
            return model_patcher

        F128_CTX["f128_loaded"] = True

        # [СЛИТНО 2026-10-01] ресивер + конфиг едут ВНУТРИ unet-файла
        # (f128_receiver.* + f128_config_bytes рядом с net.*): модель
        # = ОДИН файл; TE-слот нужен только как флаг-режима (мини-
        # файл anima-f128-mode). Прежний расклад «ресивер в TE-файле
        # 759MB» позволял юзеру выбрать штатный Qwen-TE и ТИХО
        # тестировать модель без ресивера — запрещено guard'ом выше.
        import json as _json
        try:
            from safetensors import safe_open
            with safe_open(str(unet_path), framework="pt") as f:
                keys = set(f.keys())
                if "f128_config_bytes" in keys:
                    cfg = _json.loads(bytes(
                        f.get_tensor("f128_config_bytes")
                        .tolist()).decode("utf-8"))
                    res_sd = {
                        k[len("f128_receiver."):]: f.get_tensor(k)
                        for k in keys if k.startswith("f128_receiver.")}
                    if res_sd:
                        base = (res_sd["Pk"], res_sd["Qk"],
                                res_sd["Pv"], res_sd["Qv"])
                        res = R1Copy(base, res_sd["KNW"],
                                     rank_map=RANK_MAP, seed=0)
                        res.load_state_dict(res_sd, strict=True)
                        F128_CTX["res"] = res.eval()
                        F128_CTX["cfg"] = cfg
                        print(f"[F128] ресивер СЛИТНО из unet-файла: "
                              f"{len(res_sd)} тензоров "
                              f"(step {cfg.get('step')}, carrier "
                              f"{cfg.get('carrier', 'd1c')})")
        except Exception as exc:
            raise RuntimeError(
                f"[F128] не смог прочитать слитный ресивер из "
                f"{unet_path}: {exc}")

        # load_diffusion_model() returns ModelPatcher/CoreModelPatcher.
        # The actual comfy.model_base.BaseModel is .model.
        base = getattr(model_patcher, "model", None)
        if base is None:
            raise RuntimeError(
                "F128 RAW latent patch FAIL: diffusion loader returned an "
                "object without .model"
            )
        if not hasattr(base, "latent_format"):
            raise RuntimeError(
                "F128 RAW latent patch FAIL: BaseModel has no latent_format"
            )

        # Make stock Qwen-TE -> F128 carrier conversion independent of
        # CLIP/model node execution order.
        install_f128_cond_bridge(base)

        raw_format = _RawWan21()
        _assert_raw_identity(raw_format)

        # This is the field BaseModel.process_latent_in/out actually use.
        base.latent_format = raw_format

        # Keep the config metadata in sync for clones/reloads/introspection.
        model_config = getattr(base, "model_config", None)
        if model_config is not None:
            model_config.latent_format = raw_format

        # Verify the live BaseModel, not merely the temporary object above.
        if base.latent_format is not raw_format:
            raise RuntimeError(
                "F128 RAW latent patch FAIL: BaseModel rejected latent_format"
            )
        _assert_raw_identity(base.latent_format)

        print(
            "[F128] RAW latent format ACTIVE "
            f"({type(base.latent_format).__name__}): "
            f"scale={float(base.latent_format.scale_factor):.1f} "
            f"mean_abs_max={float(base.latent_format.latents_mean.abs().max()):.1f} "
            f"std=[{float(base.latent_format.latents_std.min()):.1f},"
            f"{float(base.latent_format.latents_std.max()):.1f}]"
        )

        return model_patcher

    _ldm_f128._f128_patched = True
    _ldm_f128.__wrapped__ = _orig
    comfy.sd.load_diffusion_model = _ldm_f128
