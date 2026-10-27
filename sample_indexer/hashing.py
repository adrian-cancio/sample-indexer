"""Content hashing. File contents are streamed through SHA-256 and never stored or logged."""

from __future__ import annotations

import hashlib
import os


def hash_file(path: str | os.PathLike) -> str:
    """Return the hex SHA-256 of a file, opened strictly read-only."""
    flags = os.O_RDONLY | getattr(os, "O_NOATIME", 0)
    try:
        fd = os.open(path, flags)
    except PermissionError:
        # O_NOATIME needs file ownership; fall back to a plain read-only open.
        fd = os.open(path, os.O_RDONLY)
    with os.fdopen(fd, "rb", buffering=0) as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
