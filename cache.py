"""Content-addressed Qwen extraction and atomic, model-specific tensor caches."""

import hashlib
import json
import logging
import os
import pickle
import tempfile
from pathlib import Path

import torch
from filelock import FileLock

log = logging.getLogger(__name__)


def cache_root():
    override = os.environ.get("ANIMA_T5FREE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    import folder_paths

    return Path(folder_paths.models_dir) / "anima_t5free_cache"


def decode_json(tensor, name):
    if tensor.dtype != torch.uint8 or tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional uint8 JSON tensor")
    try:
        value = json.loads(bytes(tensor.cpu().tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def tensor_digest(tensors):
    """Hash all weight bytes, names and layouts; filenames are not identities."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)])
        digest.update(header.encode("utf-8"))
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def extract_qwen(state_dict, root):
    """Publish a complete HF directory once, under its full content digest."""
    files = {
        "config.json": "t5free_qwen_config_bytes",
        "tokenizer.json": "t5free_tokenizer_json_bytes",
        "tokenizer_config.json": "t5free_tokenizer_config_bytes",
    }
    for key in files.values():
        if key not in state_dict:
            raise ValueError(f"Bundled Qwen text encoder is missing {key}")
        decode_json(state_dict[key], key)
    weights = {k: v for k, v in state_dict.items() if not k.startswith(("conditioner.", "t5free_"))}
    if not weights:
        raise ValueError("Bundled text encoder contains no Qwen weights")
    identity = tensor_digest({**weights, **{k: state_dict[k] for k in files.values()}})
    root = Path(root) / "qwen"
    root.mkdir(parents=True, exist_ok=True)
    target = root / identity
    with FileLock(str(root / f"{identity}.lock")):
        if not (target / "complete").is_file():
            from safetensors.torch import save_file

            target.mkdir(exist_ok=True)
            # A failed write has no completion marker and is retried on load.
            for filename, key in files.items():
                (target / filename).write_bytes(bytes(state_dict[key].cpu().tolist()))
            save_file(
                {k: v.contiguous() for k, v in weights.items()},
                str(target / "model.safetensors"),
            )
            (target / "complete").write_text(identity, encoding="ascii")
    return target, identity


class TensorCache:
    def __init__(self, root, namespace):
        key = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        self.directory = Path(root) / "features" / key

    def path(self, text):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self.directory / f"{key}.pt"

    def read(self, text):
        path = self.path(text)
        if not path.is_file():
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except (
            OSError,
            RuntimeError,
            EOFError,
            ValueError,
            IndexError,
            pickle.UnpicklingError,
        ) as exc:
            log.warning("Ignoring unreadable Anima feature cache %s: %s", path, exc)
            return None

    def write(self, text, tensors):
        self.directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        os.close(handle)
        try:
            torch.save(tensors, temporary)
            os.replace(temporary, self.path(text))
        finally:
            Path(temporary).unlink(missing_ok=True)
