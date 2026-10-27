"""Incremental reconciliation of a library tree against the index.

Algorithm (see also AGENTS.md / README):

1. Walk the root (deterministic order: entries sorted by name, depth first; symlinks are
   not followed or indexed). Files with a supported extension (.wav .aif .mid,
   case-insensitive) are "discovered"; everything else is only counted per extension.
   `--limit N` stops the walk after N discovered files (a *partial* scan).
2. Classify every discovered file against the stored `path` rows of the same root:
     * same (rel_path, size, mtime_ns), not missing -> unchanged: skipped, no I/O beyond stat
       (only `last_seen` is refreshed).
     * same (rel_path, size, mtime_ns), flagged missing -> restored: flag cleared, no hash.
     * rel_path known but size/mtime differ -> changed: re-hash.
     * rel_path unknown -> new.
3. Rename detection (full scans only): stored paths that were not discovered are "vanished".
   (size, mtime_ns) is only used to pick move CANDIDATES (equal size+mtime does NOT imply
   equal content). EVERY new/relocated path is hashed; when the SHA-256 resolves to the
   resource of an unambiguous vanished candidate, the move is confirmed and the existing
   path row is updated in place (rel_path/top_folder/ext/last_seen, missing cleared). Any
   other content gets/reuses its own resource by SHA-256, so metadata is never inherited
   from a different file. Ambiguous candidate groups (several resources share size+mtime)
   are never treated as moves.
   Documented limitation: an UNCHANGED path (same rel_path, size, mtime_ns) is not
   re-hashed, so a same-size/same-mtime in-place content edit goes undetected.
4. New and changed files are hashed with a bounded thread pool (SHA-256, streamed). The
   main thread alone writes to SQLite, committing every `batch_size` results. A
   changed file points its path row at the (possibly new) resource; the old resource row
   is kept with all its metadata.
5. Vanished paths that were not consumed by a move are flagged `missing=1` (rows are never
   deleted). Skipped for partial (`--limit`) scans and for interrupted scans.

Resume / Ctrl-C: every batch is committed, and the next run re-derives its work from the
database, so an interrupted scan resumes simply by running `scan` again.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .config import DEFAULT_BATCH_SIZE, DEFAULT_WORKERS, EXT_CLASS, normalize_root
from .db import transaction, utcnow
from .hashing import hash_file

WINDOW_PER_WORKER = 4


class ChangedWhileHashing(Exception):
    pass


@dataclass
class FileInfo:
    rel_path: str
    size: int
    mtime_ns: int
    ext: str

    @property
    def top_folder(self) -> str:
        head, sep, _ = self.rel_path.partition("/")
        return head if sep else ""


@dataclass
class ScanReport:
    root: str = ""
    status: str = "running"  # complete | interrupted | aborted
    partial: bool = False  # --limit used
    discovered: int = 0
    ignored: Counter = field(default_factory=Counter)
    unchanged: int = 0
    restored: int = 0
    new_paths: int = 0
    changed: int = 0
    moved: int = 0
    hashed: int = 0
    marked_missing: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def ignored_total(self) -> int:
        return sum(self.ignored.values())

    def format(self) -> str:
        lines = [
            f"scan {self.status}{' (partial: --limit)' if self.partial else ''}: {self.root}",
            f"  discovered     {self.discovered}",
            f"  unchanged      {self.unchanged}",
            f"  restored       {self.restored}",
            f"  new            {self.new_paths}",
            f"  changed        {self.changed}",
            f"  moved          {self.moved}",
            f"  hashed         {self.hashed}",
            f"  newly missing  {self.marked_missing}",
            f"  errors         {len(self.errors)}",
            f"  ignored        {self.ignored_total}"
            + (
                " (" + ", ".join(f"{e}: {n}" for e, n in sorted(self.ignored.items())) + ")"
                if self.ignored
                else ""
            ),
            f"  elapsed        {self.elapsed_s:.2f}s",
        ]
        for err in self.errors[:20]:
            lines.append(f"  ! {err}")
        if len(self.errors) > 20:
            lines.append(f"  ! ... {len(self.errors) - 20} more")
        return "\n".join(lines)


def walk_library(
    root: str, limit: int | None, report: ScanReport
) -> Iterator[FileInfo]:
    """Yield supported files in deterministic order; count everything else as ignored."""
    emitted = 0

    def visit(abs_dir: str, rel_dir: str) -> Iterator[FileInfo]:
        nonlocal emitted
        try:
            with os.scandir(abs_dir) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as exc:
            report.errors.append(f"cannot list {rel_dir or '.'}: {exc.strerror or exc}")
            return
        for entry in entries:
            if limit is not None and emitted >= limit:
                return
            rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            try:
                if entry.is_symlink():
                    report.ignored["(symlink)"] += 1
                    continue
                if entry.is_dir(follow_symlinks=False):
                    yield from visit(entry.path, rel)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    report.ignored["(special)"] += 1
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext not in EXT_CLASS:
                    report.ignored[ext or "(none)"] += 1
                    continue
                try:
                    rel.encode("utf-8")
                except UnicodeEncodeError:
                    report.errors.append(f"undecodable file name skipped: {rel!r}")
                    continue
                st = entry.stat(follow_symlinks=False)
            except OSError as exc:
                report.errors.append(f"cannot stat {rel}: {exc.strerror or exc}")
                continue
            emitted += 1
            yield FileInfo(rel, st.st_size, st.st_mtime_ns, ext)

    yield from visit(root, "")


class Scanner:
    def __init__(
        self,
        conn: sqlite3.Connection,
        root: str | os.PathLike,
        *,
        workers: int = DEFAULT_WORKERS,
        limit: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        hash_func: Callable[[str], str] = hash_file,
        rename_heuristic: bool = True,
        on_result: Callable[[ScanReport], None] | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")
        self.conn = conn
        self.root = normalize_root(root)
        self.workers = workers
        self.limit = limit
        self.batch_size = max(1, batch_size)
        self.hash_func = hash_func
        self.rename_heuristic = rename_heuristic
        self.on_result = on_result  # called in the main thread after each hash result
        self.progress = progress or (lambda msg: None)
        self.report = ScanReport(root=self.root, partial=limit is not None)
        self._now = utcnow()
        self._run_id: int | None = None
        self._writes = 0  # writes in the currently open batch transaction
        self._tx_open = False

    # ---- batching helpers -------------------------------------------------
    def _begin(self) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        self._tx_open = True
        self._writes = 0

    def _commit(self) -> None:
        if self._tx_open:
            self.conn.execute("COMMIT")
            self._tx_open = False
            self._writes = 0

    def _rollback(self) -> None:
        if self._tx_open or self.conn.in_transaction:
            self.conn.execute("ROLLBACK")
        self._tx_open = False
        self._writes = 0

    def _wrote(self) -> None:
        self._writes += 1
        if self._writes >= self.batch_size:
            self._commit()
            self._begin()

    # ---- main entry ---------------------------------------------------------
    def run(self) -> ScanReport:
        rep = self.report
        started = time.monotonic()
        cur = self.conn.execute(
            "INSERT INTO scan_run(root, started_at, status, limit_n, workers)"
            " VALUES (?, ?, 'running', ?, ?)",
            (self.root, self._now, self.limit, self.workers),
        )
        self._run_id = cur.lastrowid
        try:
            self._reconcile()
            rep.status = "complete"
        except KeyboardInterrupt:
            self._commit()  # keep everything finished so far; resume = run again
            rep.status = "interrupted"
        except BaseException:
            self._rollback()
            rep.status = "aborted"
            rep.elapsed_s = time.monotonic() - started
            self._finish_run()
            raise
        rep.elapsed_s = time.monotonic() - started
        self._finish_run()
        return rep

    def _finish_run(self) -> None:
        rep = self.report
        self.conn.execute(
            "UPDATE scan_run SET finished_at=?, status=?, discovered=?, unchanged=?,"
            " new_paths=?, changed=?, moved=?, hashed=?, marked_missing=?, errors=? WHERE id=?",
            (
                utcnow(), rep.status, rep.discovered, rep.unchanged, rep.new_paths,
                rep.changed, rep.moved, rep.hashed, rep.marked_missing, len(rep.errors),
                self._run_id,
            ),
        )

    # ---- phases ---------------------------------------------------------------
    def _reconcile(self) -> None:
        rep = self.report
        if not os.path.isdir(self.root):
            raise NotADirectoryError(f"library root is not a directory: {self.root}")

        discovered: dict[str, FileInfo] = {}
        for info in walk_library(self.root, self.limit, rep):
            discovered[info.rel_path] = info
        rep.discovered = len(discovered)

        db_rows: dict[str, sqlite3.Row] = {
            r["rel_path"]: r
            for r in self.conn.execute(
                "SELECT id, resource_id, rel_path, size, mtime_ns, missing FROM path WHERE root=?",
                (self.root,),
            )
        }

        refresh_ids: list[int] = []  # unchanged -> refresh last_seen
        restored: list[sqlite3.Row] = []
        new_files: list[FileInfo] = []
        changed: list[tuple[FileInfo, sqlite3.Row]] = []
        for rel, info in discovered.items():
            row = db_rows.get(rel)
            if row is None:
                new_files.append(info)
            elif row["size"] == info.size and row["mtime_ns"] == info.mtime_ns:
                if row["missing"]:
                    restored.append(row)
                else:
                    refresh_ids.append(row["id"])
            else:
                changed.append((info, row))

        # Rename heuristic: only on a complete walk, otherwise un-walked rows would look vanished.
        vanished = (
            [r for rel, r in db_rows.items() if rel not in discovered] if not rep.partial else []
        )
        # (size, mtime_ns) is only a CANDIDATE filter: equal size+mtime does not imply equal
        # content, so every new path is hashed and a move is accepted only when the SHA-256
        # resolves to the candidate's own resource (see _store).
        self._move_candidates = defaultdict(list)
        self._consumed: set[int] = set()
        if self.rename_heuristic and vanished and new_files:
            groups: dict[tuple[int, int], list[sqlite3.Row]] = defaultdict(list)
            for r in vanished:
                groups[(r["size"], r["mtime_ns"])].append(r)
            for key, group in groups.items():
                if len({r["resource_id"] for r in group}) == 1:  # ambiguous groups: no move
                    self._move_candidates[key] = group

        rep.unchanged = len(refresh_ids)
        rep.restored = len(restored)
        rep.new_paths = len(new_files)
        rep.changed = len(changed)

        self._begin()
        try:
            for path_id in refresh_ids:
                self.conn.execute("UPDATE path SET last_seen=? WHERE id=?", (self._now, path_id))
                self._wrote()
            for row in restored:
                self.conn.execute(
                    "UPDATE path SET missing=0, missing_since=NULL, last_seen=? WHERE id=?",
                    (self._now, row["id"]),
                )
                self._wrote()
            self._commit()

            jobs: list[tuple[FileInfo, sqlite3.Row | None]] = [(i, None) for i in new_files]
            jobs += [(i, r) for i, r in changed]
            jobs.sort(key=lambda j: j[0].rel_path)
            self._hash_and_store(jobs)

            if not rep.partial:
                self._mark_missing([r for r in vanished if r["id"] not in self._consumed])
        finally:
            self._commit()

    def _hash_task(self, info: FileInfo) -> str:
        abs_path = os.path.join(self.root, *info.rel_path.split("/"))
        sha = self.hash_func(abs_path)
        st = os.stat(abs_path)
        if st.st_size != info.size or st.st_mtime_ns != info.mtime_ns:
            raise ChangedWhileHashing("file changed while hashing; will retry on next scan")
        return sha

    def _hash_and_store(self, jobs: list[tuple[FileInfo, sqlite3.Row | None]]) -> None:
        if not jobs:
            return
        rep = self.report
        window = self.workers * WINDOW_PER_WORKER
        pending: deque[tuple[FileInfo, sqlite3.Row | None, Future]] = deque()
        executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="hash")
        it = iter(jobs)
        self._begin()
        try:
            while True:
                while len(pending) < window:
                    job = next(it, None)
                    if job is None:
                        break
                    pending.append((job[0], job[1], executor.submit(self._hash_task, job[0])))
                if not pending:
                    break
                info, old_row, fut = pending.popleft()
                try:
                    sha = fut.result()
                except Exception as exc:  # unreadable / changed while hashing
                    reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
                    rep.errors.append(f"{info.rel_path}: {type(exc).__name__}: {reason}")
                    continue
                rep.hashed += 1
                self._store(info, old_row, sha)
                if self.on_result:
                    self.on_result(rep)
                if rep.hashed % 2000 == 0:
                    self.progress(f"  hashed {rep.hashed}/{len(jobs)}")
        finally:
            for _, _, fut in pending:
                fut.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            self._commit()

    def _store(self, info: FileInfo, old_row: sqlite3.Row | None, sha: str) -> None:
        self.conn.execute(
            "INSERT INTO resource(sha256, size, ext, first_seen) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(sha256) DO NOTHING",
            (sha, info.size, info.ext, self._now),
        )
        resource_id = self.conn.execute(
            "SELECT id FROM resource WHERE sha256=?", (sha,)
        ).fetchone()["id"]
        if old_row is None:
            candidates = self._move_candidates.get((info.size, info.mtime_ns), [])
            src = next(
                (c for c in candidates
                 if c["resource_id"] == resource_id and c["id"] not in self._consumed),
                None,
            )
            if src is not None:
                # Hash-confirmed move: same content, same size/mtime, path vanished -> update
                # the existing path row in place (no 'missing' ghost; resource untouched).
                self._consumed.add(src["id"])
                self.report.moved += 1
                self.report.new_paths -= 1
                self.conn.execute(
                    "UPDATE path SET rel_path=?, top_folder=?, ext=?, size=?, mtime_ns=?,"
                    " last_seen=?, missing=0, missing_since=NULL WHERE id=?",
                    (info.rel_path, info.top_folder, info.ext, info.size, info.mtime_ns,
                     self._now, src["id"]),
                )
                self._wrote()
                return
            self.conn.execute(
                "INSERT INTO path(resource_id, root, rel_path, top_folder, ext, size, mtime_ns,"
                " first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (resource_id, self.root, info.rel_path, info.top_folder, info.ext, info.size,
                 info.mtime_ns, self._now, self._now),
            )
        else:
            self.conn.execute(
                "UPDATE path SET resource_id=?, ext=?, size=?, mtime_ns=?, last_seen=?,"
                " missing=0, missing_since=NULL WHERE id=?",
                (resource_id, info.ext, info.size, info.mtime_ns, self._now, old_row["id"]),
            )
        self._wrote()

    def _mark_missing(self, rows: list[sqlite3.Row]) -> None:
        self._begin()
        for row in rows:
            if row["missing"]:
                continue
            self.conn.execute(
                "UPDATE path SET missing=1, missing_since=? WHERE id=?", (self._now, row["id"])
            )
            self.report.marked_missing += 1
            self._wrote()
        self._commit()
