"""Root folders for data and checkpoints.

This module does not import torch. Light scripts can import it.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "configs" / "checkpoints.yaml"


def data_root() -> Path:
    """Return the data root: ``$OAT_DATA_ROOT``, else ``<repo>/data``."""
    return Path(os.environ.get("OAT_DATA_ROOT", REPO_ROOT / "data"))


def checkpoint_root() -> Path:
    """Return the checkpoint root: ``$OAT_CKPT_ROOT``, else ``<repo>/checkpoints``."""
    return Path(os.environ.get("OAT_CKPT_ROOT", REPO_ROOT / "checkpoints"))
