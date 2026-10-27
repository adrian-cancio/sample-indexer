"""Defaults and shared constants."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ENV_DB = "SAMPLE_INDEXER_DB"
ENV_ROOT = "SAMPLE_INDEXER_ROOT"

DEFAULT_DB = REPO_ROOT / "data" / "index.sqlite"
DEFAULT_ROOT = Path.home() / "Música" / "OMNi" / "User Library"

DEFAULT_WORKERS = 8
DEFAULT_BATCH_SIZE = 500

# Supported extensions (lower-case; matching is case-insensitive) and their class.
EXT_CLASS = {
    ".wav": "audio",
    ".aif": "audio",
    ".mid": "midi",
}


def normalize_root(root: str | os.PathLike) -> str:
    """Canonical string form of a library root (absolute, no trailing slash).

    Symlinks are deliberately not resolved so the stored value matches what the
    user typed/configured.
    """
    return str(Path(root).expanduser().absolute())


def default_db_path() -> Path:
    env = os.environ.get(ENV_DB)
    return Path(env).expanduser() if env else DEFAULT_DB


def default_root() -> Path:
    env = os.environ.get(ENV_ROOT)
    return Path(env).expanduser() if env else DEFAULT_ROOT
