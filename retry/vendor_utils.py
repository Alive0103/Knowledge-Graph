from __future__ import annotations

import os
import sys
from pathlib import Path


def retry_root() -> Path:
    return Path(__file__).resolve().parent


def vendor_dir() -> Path:
    return retry_root() / "_vendor"


def ensure_vendor_path() -> Path | None:
    """
    Append retry/_vendor to sys.path so project-local transformer dependencies
    are available without shadowing globally installed torch.
    """

    path = vendor_dir()
    if not path.exists():
        return None

    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return path
