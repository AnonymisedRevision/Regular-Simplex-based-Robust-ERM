from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .io import ensure_dir


def setup_logger(out_dir: str | Path, name: str = "srerm_skin") -> logging.Logger:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger
