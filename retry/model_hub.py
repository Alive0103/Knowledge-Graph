from __future__ import annotations

import inspect
import os
from pathlib import Path

from vendor_utils import ensure_vendor_path


def looks_like_model_dir(path: Path) -> bool:
    return (
        (path / "config.json").exists()
        and (
            (path / "model.safetensors").exists()
            or (path / "pytorch_model.bin").exists()
            or (path / "model.safetensors.index.json").exists()
            or (path / "pytorch_model.bin.index.json").exists()
        )
    )


def download_hf_snapshot(
    model_name: str,
    output_dir: Path,
    overwrite: bool = False,
    hf_endpoint: str | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> Path:
    """
    Download a Hugging Face model snapshot into a local directory.

    The helper keeps the logic shared between entity-linking model recovery and
    alignment baselines so both paths can run in a fresh environment.
    """

    ensure_vendor_path()

    from huggingface_hub import snapshot_download

    output_dir = Path(output_dir)
    if looks_like_model_dir(output_dir) and not overwrite:
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    kwargs = {
        "repo_id": model_name,
        "local_dir": str(output_dir),
        "allow_patterns": allow_patterns
        or [
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors",
            "*.bin",
        ],
        "ignore_patterns": ignore_patterns
        or [
            "*.h5",
            "*.msgpack",
            "*.onnx",
        ],
    }
    signature = inspect.signature(snapshot_download)
    if "local_dir_use_symlinks" in signature.parameters:
        kwargs["local_dir_use_symlinks"] = False
    if hf_endpoint and "endpoint" in signature.parameters:
        kwargs["endpoint"] = hf_endpoint

    snapshot_download(**kwargs)
    return output_dir
