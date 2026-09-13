# -*- coding: utf-8 -*-
"""AnimaT5Free configuration.

T5-free text conditioning frontend for Anima: Qwen3.5-2B taps ->
d1 row planner -> ridge C0 -> C1 residual refiner -> frozen Anima DiT.
This repository ships ONLY the conditioner; the Qwen text encoder and the
Anima DiT/VAE are dependency references resolved at load time.

config.json keeps the parameters in nested dicts (conditioner.*, sampler.*)
for readability; this class promotes them to flat attributes so the model
class can read config.qwen_layers, config.cfg, ... directly.
"""
from transformers import PretrainedConfig


class AnimaT5FreeConfig(PretrainedConfig):
    model_type = "anima_t5free"

    def __init__(
        self,
        context_dim=1024,
        max_rows=512,
        feature_dim=4096,
        qwen_layers=(16, 22),
        refiner_width=512,
        refiner_heads=8,
        refiner_mlp=2048,
        refiner_blocks=2,
        segmenter_hidden=64,
        max_prompt_bytes=5120,
        pos_buckets=((0, 20), (20, 40), (40, 80), (80, 160), (160, 512)),
        len_buckets=((0, 80), (80, 150), (150, 512)),
        mu_keys=None,
        cfg=5.0,
        steps=30,
        latent_shape=(16, 1, 96, 96),
        latent_mean=None,
        latent_std=None,
        base_model="circlestone-labs/Anima",
        text_encoder="Qwen/Qwen3.5-2B-Base",
        **kwargs,
    ):
        self.context_dim = context_dim
        self.max_rows = max_rows
        self.feature_dim = feature_dim
        self.qwen_layers = list(qwen_layers)
        self.refiner_width = refiner_width
        self.refiner_heads = refiner_heads
        self.refiner_mlp = refiner_mlp
        self.refiner_blocks = refiner_blocks
        self.segmenter_hidden = segmenter_hidden
        self.max_prompt_bytes = max_prompt_bytes
        self.pos_buckets = [list(p) for p in pos_buckets]
        self.len_buckets = [list(l) for l in len_buckets]
        self.mu_keys = list(mu_keys or [])
        self.cfg = cfg
        self.steps = steps
        self.latent_shape = list(latent_shape)
        self.latent_mean = list(latent_mean or [])
        self.latent_std = list(latent_std or [])
        self.base_model = base_model
        self.text_encoder = text_encoder

        # promote nested dicts from a written config.json (round-trip)
        cond = kwargs.pop("conditioner", None)
        if isinstance(cond, dict):
            for k in ("context_dim", "max_rows", "feature_dim",
                      "qwen_layers"):
                if k in cond:
                    setattr(self, k, cond[k])
            r = cond.get("refiner") or {}
            if r:
                self.refiner_width = r.get("width", self.refiner_width)
                self.refiner_heads = r.get("heads", self.refiner_heads)
                self.refiner_mlp = r.get("mlp", self.refiner_mlp)
                self.refiner_blocks = r.get("blocks",
                                             self.refiner_blocks)
            sg = cond.get("segmenter") or {}
            if sg:
                self.segmenter_hidden = sg.get("hidden",
                                                self.segmenter_hidden)
                self.max_prompt_bytes = sg.get("max_prompt_bytes",
                                               self.max_prompt_bytes)
            if isinstance(cond.get("c0"), dict) and cond["c0"].get("mu_keys"):
                self.mu_keys = list(cond["c0"]["mu_keys"])
            if cond.get("pos_buckets"):
                self.pos_buckets = [list(p) for p in cond["pos_buckets"]]
            if cond.get("len_buckets"):
                self.len_buckets = [list(l) for l in cond["len_buckets"]]
        samp = kwargs.pop("sampler", None)
        if isinstance(samp, dict):
            self.cfg = samp.get("cfg", self.cfg)
            self.steps = samp.get("steps", self.steps)
            if samp.get("latent_shape"):
                self.latent_shape = list(samp["latent_shape"])
            if samp.get("latent_mean"):
                self.latent_mean = list(samp["latent_mean"])
            if samp.get("latent_std"):
                self.latent_std = list(samp["latent_std"])
        super().__init__(**kwargs)
