from __future__ import annotations

import os
import sys
from pathlib import Path


if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RETRY_DIR = Path(__file__).resolve().parent
if str(RETRY_DIR) not in sys.path:
    sys.path.insert(0, str(RETRY_DIR))

from entity_linking.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
