"""Anima T5-Free and F128 conditioning integration for ComfyUI."""

from .comfy_integration import install
from .nodes import AnimaTextEncoderMode

install()

NODE_CLASS_MAPPINGS = {"AnimaTextEncoderMode": AnimaTextEncoderMode}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaTextEncoderMode": "Anima Text Encoder Mode"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
