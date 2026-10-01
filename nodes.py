"""Optional explicit text mode; old stock-node workflows use dual mode."""


class AnimaTextEncoderMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "mode": (["F128", "T5-Free C0/C1", "Both (compatibility)"],),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "select"
    CATEGORY = "conditioning/anima"
    DESCRIPTION = "Select F128 for merged FT checkpoints, C0/C1 for the original T5-Free model."

    def select(self, clip, mode):
        from .t5free_te import AnimaT5FreeTEModel

        if not isinstance(clip.cond_stage_model, AnimaT5FreeTEModel):
            raise ValueError("Anima Text Encoder Mode requires the bundled Anima Qwen encoder")
        modes = {
            "F128": "f128",
            "T5-Free C0/C1": "t5free",
            "Both (compatibility)": "dual",
        }
        clone = clip.clone()
        clone.set_tokenizer_option("anima_text_mode", modes[mode])
        return (clone,)
