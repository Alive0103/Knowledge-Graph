#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from alignment.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
