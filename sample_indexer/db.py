"""SQLite connection, explicit transactions and schema migrations."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .migrations import LATEST_VERSION, MIGRATIONS


class DatabaseNotFound(Exception):
    pass


class SchemaTooNew(Exception):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _casefold(value):
    return value.casefold() if value is not None else None


def connect(path: str | Path, *, create: bool = True) -> sqlite3.Connection:
    path = Path(path)
    if not path.exists():
        if not create:
            raise DatabaseNotFound(f"database not found: {path} (run `init` first)")
        path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None -> autocommit; transactions are explicit (see `transaction`).
    conn = sqlite3.connect(path, isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.create_function("casefold", 1, _casefold, deterministic=True)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def schema_version(conn: sqlite3.Connection) -> int:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if not exists:
        return 0
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return row["v"] or 0


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply pending migrations in order, each atomically. Returns applied versions."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    current = schema_version(conn)
    if current > LATEST_VERSION:
        raise SchemaTooNew(
            f"database schema v{current} is newer than this code (v{LATEST_VERSION})"
        )
    applied = []
    for version, sql in MIGRATIONS:
        if version <= current:
            continue
        script = (
            "BEGIN IMMEDIATE;\n"
            f"{sql}\n"
            f"INSERT INTO schema_version(version, applied_at) VALUES ({version}, '{utcnow()}');\n"
            "COMMIT;"
        )
        try:
            conn.executescript(script)
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        applied.append(version)
    return applied


def open_db(path: str | Path, *, create: bool = False) -> sqlite3.Connection:
    """Connect; with create=True also create and migrate. Otherwise require an existing,
    up-to-date database."""
    conn = connect(path, create=create)
    if create:
        migrate(conn)
    else:
        version = schema_version(conn)
        if version != LATEST_VERSION:
            conn.close()
            raise DatabaseNotFound(
                f"database {path} is at schema v{version}, expected v{LATEST_VERSION} (run `init`)"
            )
    return conn
