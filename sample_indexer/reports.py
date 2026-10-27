"""Read-only reporting: stats, query and manifest verification."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .config import EXT_CLASS


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def stats(conn: sqlite3.Connection) -> str:
    out: list[str] = []
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731

    present = one("SELECT COUNT(*) FROM path WHERE missing=0")
    missing = one("SELECT COUNT(*) FROM path WHERE missing=1")
    out.append(f"paths present      {present}")
    out.append(f"paths missing      {missing}")
    out.append(f"unique resources   {one('SELECT COUNT(*) FROM resource')}")

    audio_exts = [e for e, c in EXT_CLASS.items() if c == "audio"]
    midi_exts = [e for e, c in EXT_CLASS.items() if c == "midi"]
    counts = {
        r["ext"]: r["n"]
        for r in conn.execute("SELECT ext, COUNT(*) AS n FROM path WHERE missing=0 GROUP BY ext")
    }
    audio = sum(counts.get(e, 0) for e in audio_exts)
    midi = sum(counts.get(e, 0) for e in midi_exts)
    out.append(f"present by kind    audio {audio} + midi {midi} = {audio + midi}")

    out.append("present by ext:")
    for ext, n in sorted(counts.items()):
        out.append(f"  {ext:<8} {n}")

    out.append("present by top-level folder:")
    for r in conn.execute(
        "SELECT top_folder, COUNT(*) AS n FROM path WHERE missing=0 GROUP BY top_folder"
        " ORDER BY top_folder"
    ):
        out.append(f"  {r['top_folder'] or '(root)':<20} {r['n']}")

    dup = conn.execute(
        "SELECT COUNT(*) AS groups, COALESCE(SUM(n),0) AS paths FROM ("
        " SELECT COUNT(*) AS n FROM path WHERE missing=0 GROUP BY resource_id HAVING n > 1)"
    ).fetchone()
    out.append(f"duplicate sha256   {dup['groups']} resources on {dup['paths']} present paths")
    for r in conn.execute(
        "SELECT r.sha256, COUNT(*) AS n FROM path p JOIN resource r ON r.id=p.resource_id"
        " WHERE p.missing=0 GROUP BY p.resource_id HAVING n > 1 ORDER BY n DESC, r.sha256 LIMIT 10"
    ):
        out.append(f"  {r['sha256']}  x{r['n']}")
    return "\n".join(out)


def query(
    conn: sqlite3.Connection,
    *,
    path_contains: str | None = None,
    ext: str | None = None,
    folder: str | None = None,
    include_missing: bool = False,
    limit: int = 50,
) -> list[sqlite3.Row]:
    where, params = [], []
    if not include_missing:
        where.append("p.missing = 0")
    if path_contains:
        # casefold() is a Unicode-aware SQL function (SQLite's LIKE only folds ASCII).
        where.append("casefold(p.rel_path) LIKE ? ESCAPE '\\'")
        params.append(f"%{_like_escape(path_contains.casefold())}%")
    if ext:
        where.append("p.ext = ?")
        params.append("." + ext.lower().lstrip("."))
    if folder:
        prefix = folder.strip("/")
        where.append("(casefold(p.rel_path) LIKE ? ESCAPE '\\' OR casefold(p.rel_path) = ?)")
        params.extend([f"{_like_escape(prefix.casefold())}/%", prefix.casefold()])
    sql = (
        "SELECT p.rel_path, p.ext, p.size, p.missing, r.sha256 FROM path p"
        " JOIN resource r ON r.id = p.resource_id"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY p.rel_path"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


@dataclass
class ManifestDiff:
    manifest_rows: int = 0
    db_rows: int = 0
    sha_mismatch: list[tuple[str, str, str]] = field(default_factory=list)  # rel, manifest, db
    size_mismatch: list[tuple[str, int, int]] = field(default_factory=list)
    only_in_manifest: list[str] = field(default_factory=list)
    only_in_db: list[str] = field(default_factory=list)

    @property
    def differences(self) -> int:
        return (
            len(self.sha_mismatch)
            + len(self.size_mismatch)
            + len(self.only_in_manifest)
            + len(self.only_in_db)
        )

    def format(self, max_lines: int = 20) -> str:
        lines = [
            f"manifest rows      {self.manifest_rows}",
            f"db present paths   {self.db_rows}",
            f"sha256 mismatches  {len(self.sha_mismatch)}",
            f"size mismatches    {len(self.size_mismatch)}",
            f"only in manifest   {len(self.only_in_manifest)}",
            f"only in db         {len(self.only_in_db)}",
            f"differences        {self.differences}",
        ]
        for label, items in (
            ("sha256", self.sha_mismatch),
            ("size", self.size_mismatch),
            ("only in manifest", self.only_in_manifest),
            ("only in db", self.only_in_db),
        ):
            for item in items[:max_lines]:
                lines.append(f"  {label}: {item}")
        return "\n".join(lines)


def verify_manifest(
    conn: sqlite3.Connection, csv_path: str | Path, root: str | None = None
) -> ManifestDiff:
    """Compare present DB paths with a manifest CSV (sha256,rel_path,size,mtime).

    mtime is deliberately not compared (the manifest has whole seconds; the DB keeps ns).
    """
    diff = ManifestDiff()
    manifest: dict[str, tuple[str, int]] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_cols = {"sha256", "rel_path", "size", "mtime"} - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"manifest lacks columns: {sorted(missing_cols)}")
        for row in reader:
            diff.manifest_rows += 1
            manifest[row["rel_path"]] = (row["sha256"].strip().lower(), int(row["size"]))

    sql = (
        "SELECT p.rel_path, p.size, r.sha256 FROM path p JOIN resource r ON r.id=p.resource_id"
        " WHERE p.missing=0"
    )
    params: list = []
    if root:
        sql += " AND p.root=?"
        params.append(root)
    db: dict[str, tuple[str, int]] = {}
    for r in conn.execute(sql, params):
        diff.db_rows += 1
        db[r["rel_path"]] = (r["sha256"], r["size"])

    for rel in sorted(manifest.keys() | db.keys()):
        m, d = manifest.get(rel), db.get(rel)
        if m is None:
            diff.only_in_db.append(rel)
        elif d is None:
            diff.only_in_manifest.append(rel)
        else:
            if m[0] != d[0]:
                diff.sha_mismatch.append((rel, m[0], d[0]))
            if m[1] != d[1]:
                diff.size_mismatch.append((rel, m[1], d[1]))
    return diff
