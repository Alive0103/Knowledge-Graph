from __future__ import annotations

from pathlib import Path

from model_hub import download_hf_snapshot


DEFAULT_BASE_MODEL_NAME = "distilbert-base-multilingual-cased"
DEFAULT_BASE_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "entity_linking_base" / DEFAULT_BASE_MODEL_NAME
def download_base_model(
    model_name: str = DEFAULT_BASE_MODEL_NAME,
    output_dir: Path = DEFAULT_BASE_MODEL_DIR,
    overwrite: bool = False,
    hf_endpoint: str | None = None,
) -> Path:
    """
    Download a multilingual base model locally so training and vectorization can
    run without depending on the old missing server weights.
    """

    return download_hf_snapshot(
        model_name=model_name,
        output_dir=Path(output_dir),
        overwrite=overwrite,
        hf_endpoint=hf_endpoint,
    )
