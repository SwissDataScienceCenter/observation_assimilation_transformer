"""Download the released checkpoints, result files and Europe dataset."""

import hashlib
from pathlib import Path

SUMS_NAME = "SHA256SUMS"
_CHUNK = 1 << 20


def file_digest(path: Path, algorithm: str) -> str:
    """Return the hex digest of the file at *path* (``md5`` or ``sha256``)."""
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()
