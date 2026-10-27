"""Stage `loops` (L3-5): full ffmpeg decode + tempo / beats / bars / key DSP for loop candidates.
Versioned and resumable through `resource_stage` (stage = 'loops'). DSP lives in `loops_dsp`.

Selection (documented in reports/L3-5.md; unknown labels are NOT excluded). A resource is selected
when it is audio, has a present path and at least one rule fires:
  folder    : a PRESENT path lies in `Samples/Loops/*`, `Samples/Synths/{Synth Loops, Basslines,
              Sequences, Acid Riffs, Hooks, Abstract Loops}` or `Samples/FX & Atmo/FX Loops`;
  kind      : resource.kind = 'loop' (L3-3 hint, any confidence);
  bpm_name  : resource.bpm_name is not NULL (an explicit BPM token in a name).
`.mid` is never selected. `resource.loops_select` stores the rules that fired.

Decode: reuses `lowend.ffmpeg_decode` (`ffmpeg -xerror -threads 1`, whole first audio stream, native
rate/channels, float32 pipe). Memory is bounded by `max_duration_s` (default 300 s, header AND decoded
frame count) and `max_decode_bytes` (float32 PCM bytes, default 256 MiB); files beyond a cap are
`unresolved_cap`, never truncated.

Writes: estimates (`bpm_est`, `beats`, `key_est` and the evidence columns) are written ONLY for an
`ok` analysis, so a later inconclusive run (timeout, stale, ...) never erases earlier conclusive values.
`valid` is written only with conclusive evidence (1 = whole stream decoded cleanly, 0 = conclusive
source failure); it is never reset to NULL. Hints (`bpm_name`, `key_name`) are compared after the
estimate, never used to produce it.

Concurrency: bounded ThreadPoolExecutor (default 2 workers), one ffmpeg + numpy analysis per worker; the
main thread is the only SQLite writer and commits every `batch_size` results (checkpoint).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from . import loops_dsp
from .db import utcnow
from .lowend import DECODE_FAILED, DEFAULT_DECODE_TIMEOUT_S, VALID_SCOPE, DecodeResult, ffmpeg_decode
from .metadata import ProcessRegistry

LOOPS_STAGE = "loops"
LOOPS_VERSION = 1

DEFAULT_LOOPS_WORKERS = 2
DEFAULT_LOOPS_BATCH = 50
DEFAULT_MAX_DURATION_S = 300.0
DEFAULT_MAX_DECODE_BYTES = 256 * 1024 * 1024  # float32 PCM bytes held per file
WINDOW_PER_WORKER = 2

LOOP_FOLDERS: set[tuple[str, str, str | None]] = {
    ("samples", "loops", None),
    ("samples", "synths", "synth loops"),
    ("samples", "synths", "basslines"),
    ("samples", "synths", "sequences"),
    ("samples", "synths", "acid riffs"),
    ("samples", "synths", "hooks"),
    ("samples", "synths", "abstract loops"),
    ("samples", "fx & atmo", "fx loops"),
}
BPM_BUCKETS = ((100, "<100"), (130, "100-130"), (150, "130-150"), (170, "150-170"))


def effective_caps(max_duration_s: float, max_decode_bytes: int) -> tuple[float, int]:
    """Caps can only be tightened: the DSP refuses anything over its own limits."""
    return min(max_duration_s, loops_dsp.MAX_SECONDS), min(max_decode_bytes, loops_dsp.MAX_BYTES)


def effective_overrides(overrides: dict | None) -> dict:
    out = {}
    for k, v in (overrides or {}).items():
        if k not in loops_dsp.DEFAULTS:
            raise ValueError(f"unknown DSP setting: {k}")
        out[k] = v
    return out


def settings(max_duration_s: float, max_decode_bytes: int, overrides: dict | None = None) -> dict:
    dur, nbytes = effective_caps(max_duration_s, max_decode_bytes)
    return {
        "stage_version": LOOPS_VERSION,
        "dsp_version": loops_dsp.VERSION,
        "dsp": dict(loops_dsp.DEFAULTS, **effective_overrides(overrides)),
        "max_duration_s": dur,
        "max_decode_bytes_float32": nbytes,
        "decode": "ffmpeg -xerror -threads 1, native rate/channels, f32le pipe, whole stream",
        "valid_scope": VALID_SCOPE,
    }


def settings_sig(max_duration_s: float, max_decode_bytes: int, overrides: dict | None = None) -> str:
    payload = json.dumps(settings(max_duration_s, max_decode_bytes, overrides), sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


# ----------------------------------------------------------------------------- selection


def selection_rules(rel_paths: list[str], kind: str | None, bpm_name: float | None) -> dict | None:
    """Which selection rules fire for one resource, or None when it is not a candidate."""
    rules: list[str] = []
    folders: list[str] = []
    for rel in sorted(rel_paths):
        dirs = [p.casefold() for p in rel.split("/")[:-1]]
        top = dirs[0] if dirs else ""
        group = dirs[1] if len(dirs) > 1 else None
        sub = dirs[2] if len(dirs) > 2 else None
        for key in ((top, group, sub), (top, group, None)):
            if key in LOOP_FOLDERS:
                folders.append("/".join(p for p in key if p))
                break
    if folders:
        rules.append("folder")
    if kind == "loop":
        rules.append("kind")
    if bpm_name is not None:
        rules.append("bpm_name")
    if not rules:
        return None
    return {"rules": rules, "kind": kind, "bpm_name": bpm_name, "folders": sorted(set(folders))}


def bpm_bucket(bpm: float | None) -> str:
    if bpm is None:
        return "none"
    for edge, name in BPM_BUCKETS:
        if bpm < edge:
            return name
    return ">=170"


# ----------------------------------------------------------------------------- one resource


@dataclass
class Outcome:
    status: str
    error: str | None = None
    frames: int | None = None
    analysis: dict | None = None
    spawned: bool = False
    seconds: float = 0.0


@dataclass
class Item:
    rid: int
    sr: int | None
    channels: int | None
    duration_s: float | None
    probe_status: str | None
    bpm_name: float | None
    key_name: str | None
    select: dict
    paths: list[tuple[str, str, int, int]]  # (root, rel_path, size, mtime_ns)


def analyse_item(
    item: Item,
    *,
    decode: Callable[..., DecodeResult],
    max_duration_s: float,
    max_bytes: int,
    overrides: dict | None = None,
) -> Outcome:
    t0 = time.monotonic()
    out = _analyse_item(item, decode, max_duration_s, max_bytes, overrides)
    out.seconds = time.monotonic() - t0
    return out


def _analyse_item(item, decode, max_duration_s, max_bytes, overrides) -> Outcome:
    if item.probe_status != "ok" or not item.sr or not item.channels:
        return Outcome("unresolved_metadata", f"header metadata missing (probe_status={item.probe_status},"
                       f" sr={item.sr}, channels={item.channels}); not decoded")
    if item.channels > 2:
        return Outcome("unresolved_metadata", f"{item.channels} channels unsupported by L3-5; not decoded")
    if item.duration_s is not None and item.duration_s > max_duration_s:
        return Outcome("unresolved_cap", f"duration {item.duration_s:.2f}s exceeds cap {max_duration_s:g}s; not decoded")
    if item.duration_s is not None and item.duration_s * item.sr * item.channels * 4 > max_bytes:
        return Outcome("unresolved_cap", f"float32 PCM footprint exceeds {max_bytes} bytes; not decoded")
    last = Outcome("unreadable", "no present path to decode")
    for root, rel, size, mtime_ns in item.paths:
        path = os.path.join(root, *rel.split("/"))
        try:
            st = os.stat(path)
        except OSError as exc:
            last = Outcome("unreadable", f"{type(exc).__name__}: {exc.strerror or exc}")
            continue
        if (st.st_size, st.st_mtime_ns) != (size, mtime_ns):
            last = Outcome("stale", "file changed since scan (size/mtime differ); rescan")
            continue
        res = decode(path, item.channels)
        try:
            st2 = os.stat(path)
        except OSError:
            st2 = None
        if st2 is None or (st2.st_size, st2.st_mtime_ns) != (size, mtime_ns):
            last = Outcome("stale", "file changed while decoding; rescan", spawned=res.spawned)
            continue
        if res.status != "ok":
            return Outcome(res.status, res.error, frames=res.frames, spawned=res.spawned)
        # header duration can be wrong/absent: re-check the real decoded size before analysing
        if res.frames / item.sr > max_duration_s or res.frames * item.channels * 4 > max_bytes:
            return Outcome("unresolved_cap", f"decoded {res.frames} frames ({res.frames / item.sr:.2f}s) exceed caps;"
                           " not analysed, not truncated", frames=res.frames, spawned=True)
        try:
            analysis = loops_dsp.analyze(res.samples, item.sr, bpm_hint=item.bpm_name, key_hint=item.key_name,
                                         overrides=overrides)
        except Exception as exc:  # decode succeeded (valid=1) but the analysis faulted
            return Outcome("analysis_error", f"{type(exc).__name__}: {exc}", frames=res.frames, spawned=True)
        return Outcome("ok", frames=res.frames, analysis=analysis, spawned=True)
    return last


# ----------------------------------------------------------------------------- stage runner


@dataclass
class LoopsReport:
    status: str = "running"
    stop_reason: str | None = None
    mode: str = "default"
    audio_with_path: int = 0
    candidates: int = 0
    by_rule: dict = field(default_factory=lambda: defaultdict(int))
    selected: int = 0
    already_done: int = 0
    decoded: int = 0
    ok: int = 0
    statuses: dict = field(default_factory=lambda: defaultdict(int))
    tempo_reasons: dict = field(default_factory=lambda: defaultdict(int))
    key_reasons: dict = field(default_factory=lambda: defaultdict(int))
    valid1: int = 0
    valid0: int = 0
    pending_after: int = 0
    elapsed_s: float = 0.0
    item_seconds: float = 0.0

    @property
    def processed(self) -> int:
        return sum(self.statuses.values())

    @property
    def unresolved(self) -> int:
        return sum(n for s, n in self.statuses.items() if s != "ok" and s != DECODE_FAILED)

    def format(self) -> str:
        lines = [
            f"loops {self.status}" + (f" ({self.stop_reason})" if self.stop_reason else "") + f" [{self.mode}]",
            f"  audio resources with a present path {self.audio_with_path}",
            f"  loop candidates (rules)             {self.candidates}",
        ]
        for rule, n in sorted(self.by_rule.items()):
            lines.append(f"    rule {rule:<24} {n}")
        lines += [
            f"  already complete (this version)     {self.already_done}",
            f"  selected this run                   {self.selected}",
            f"  ffmpeg subprocesses                 {self.decoded}",
            f"  processed                           {self.processed} (ok {self.ok},"
            f" decode failed {self.statuses.get(DECODE_FAILED, 0)}, unresolved {self.unresolved})",
            f"  valid=1 / valid=0 written           {self.valid1} / {self.valid0}",
            f"  still pending                       {self.pending_after}",
            f"  elapsed                             {self.elapsed_s:.2f}s (worker time {self.item_seconds:.1f}s)",
        ]
        for status, n in sorted(self.statuses.items()):
            if status != "ok":
                lines.append(f"  status {status:<22} {n}")
        for name, d in (("tempo", self.tempo_reasons), ("key", self.key_reasons)):
            for reason, n in sorted(d.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {name} reason {reason:<30} {n}")
        return "\n".join(lines)


class LoopsStage:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        workers: int = DEFAULT_LOOPS_WORKERS,
        limit: int | None = None,
        force: bool = False,
        retry_errors: bool = False,
        stratify: bool = False,
        seed: int = 0,
        timeout: float = DEFAULT_DECODE_TIMEOUT_S,
        batch_size: int = DEFAULT_LOOPS_BATCH,
        max_seconds: float | None = None,
        max_duration_s: float = DEFAULT_MAX_DURATION_S,
        max_decode_bytes: int = DEFAULT_MAX_DECODE_BYTES,
        overrides: dict | None = None,
        ffmpeg_bin: str = "ffmpeg",
        decode_func: Callable[..., DecodeResult] | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")
        if timeout <= 0 or max_duration_s <= 0 or max_decode_bytes <= 0:
            raise ValueError("timeout, max-duration and max-decode-bytes must be > 0")
        self.conn = conn
        self.workers = workers
        self.limit = limit
        self.force = force
        self.retry_errors = retry_errors
        self.stratify = stratify
        self.seed = seed
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self.max_seconds = max_seconds
        self.max_duration_s, self.max_decode_bytes = effective_caps(max_duration_s, max_decode_bytes)
        self.overrides = effective_overrides(overrides)
        self.ffmpeg_bin = ffmpeg_bin
        self.progress = progress or (lambda msg: None)
        self.registry = ProcessRegistry()
        self._custom_decode = decode_func
        self.sig = settings_sig(self.max_duration_s, self.max_decode_bytes, self.overrides)
        self.settings = settings(self.max_duration_s, self.max_decode_bytes, self.overrides)
        self.report = LoopsReport(
            mode=",".join(m for m, on in (("force", force), ("retry-errors", retry_errors),
                                          ("stratified", stratify)) if on) or "default"
        )
        self._run_id: int | None = None
        self._started = 0.0
        self._writes = 0
        self._tx_open = False

    # ---- worker -------------------------------------------------------------------------
    def _decode(self, path: str, channels: int) -> DecodeResult:
        if self._custom_decode is not None:
            return self._custom_decode(path, channels)
        return ffmpeg_decode(path, channels, timeout=self.timeout, max_bytes=self.max_decode_bytes,
                             ffmpeg_bin=self.ffmpeg_bin, registry=self.registry)

    def _worker(self, item: Item) -> Outcome:
        return analyse_item(item, decode=self._decode, max_duration_s=self.max_duration_s,
                            max_bytes=self.max_decode_bytes, overrides=self.overrides)

    # ---- selection ----------------------------------------------------------------------
    def select(self, *, count_report: bool = True) -> list[Item]:
        conn, rep = self.conn, self.report
        stages = {
            r["resource_id"]: (r["version"], r["error"], r["input_sig"])
            for r in conn.execute(
                "SELECT resource_id, version, error, input_sig FROM resource_stage WHERE stage=?", (LOOPS_STAGE,))
        }
        paths: dict[int, list[tuple[str, str, int, int]]] = defaultdict(list)
        for r in conn.execute(
            "SELECT resource_id, root, rel_path, size, mtime_ns FROM path WHERE missing=0 ORDER BY rel_path"
        ):
            paths[r["resource_id"]].append((r["root"], r["rel_path"], r["size"], r["mtime_ns"]))
        items: list[Item] = []
        audio = cands = done = 0
        by_rule: dict = defaultdict(int)
        for r in conn.execute(
            "SELECT id, kind, bpm_name, key_name, sr, channels, duration_s, probe_status FROM resource"
            " WHERE ext != '.mid' ORDER BY id"
        ):
            plist = paths.get(r["id"])
            if not plist:
                continue
            audio += 1
            sel = selection_rules([p[1] for p in plist], r["kind"], r["bpm_name"])
            if sel is None:
                continue
            cands += 1
            by_rule["+".join(sel["rules"])] += 1
            row = stages.get(r["id"])
            if self.retry_errors and not self.force:
                need = row is not None and row[0] == LOOPS_VERSION and row[2] == self.sig and row[1] is not None
            else:
                need = self.force or row is None or row[0] != LOOPS_VERSION or row[2] != self.sig
            if need:
                items.append(Item(r["id"], r["sr"], r["channels"], r["duration_s"], r["probe_status"],
                                  r["bpm_name"], r["key_name"], sel, plist))
            else:
                done += 1
        if count_report:
            rep.audio_with_path, rep.candidates, rep.already_done = audio, cands, done
            rep.by_rule.update(by_rule)
        if self.stratify:
            items = stratified_order(items, self.seed)
        if self.limit is not None:
            items = items[: self.limit]
        if count_report:
            rep.selected = len(items)
        return items

    # ---- batching -----------------------------------------------------------------------
    def _begin(self) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        self._tx_open, self._writes = True, 0

    def _checkpoint(self, final: bool = False) -> None:
        if self._tx_open:
            self._flush_run_row(None)
            self.conn.execute("COMMIT")
            self._tx_open, self._writes = False, 0
        if not final:
            self._begin()

    def _flush_run_row(self, status: str | None) -> None:
        rep = self.report
        self.conn.execute(
            "UPDATE loops_run SET status=COALESCE(?, status), selected=?, processed=?, decoded=?, ok=?,"
            " failed=?, unresolved=?, elapsed_s=? WHERE id=?",
            (status, rep.selected, rep.processed, rep.decoded, rep.ok,
             rep.statuses.get(DECODE_FAILED, 0), rep.unresolved, time.monotonic() - self._started, self._run_id),
        )

    # ---- main ---------------------------------------------------------------------------
    def run(self) -> LoopsReport:
        rep = self.report
        if self._custom_decode is None and shutil.which(self.ffmpeg_bin) is None:
            raise FileNotFoundError(f"{self.ffmpeg_bin} not found on PATH (install ffmpeg)")
        self._started = time.monotonic()
        cur = self.conn.execute(
            "INSERT INTO loops_run(started_at, status, limit_n, workers, mode) VALUES (?, 'running', ?, ?, ?)",
            (utcnow(), self.limit, self.workers, rep.mode),
        )
        self._run_id = cur.lastrowid
        try:
            items = self.select()
            self._begin()
            self._process(items)
            rep.status = "interrupted" if rep.stop_reason else "complete"
        except KeyboardInterrupt:
            self.registry.kill_all()
            rep.status, rep.stop_reason = "interrupted", "keyboard interrupt"
        except BaseException:
            self.registry.kill_all()
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            self._tx_open = False
            rep.status = "aborted"
            self._finish()
            raise
        try:
            self._checkpoint(final=True)
        finally:
            self._finish()
        return rep

    def _finish(self) -> None:
        rep = self.report
        rep.elapsed_s = time.monotonic() - self._started
        if self.conn.in_transaction:
            self.conn.execute("ROLLBACK")
        self._tx_open = False
        saved = (self.limit, self.stratify)
        self.limit, self.stratify = None, False
        try:
            rep.pending_after = len(self.select(count_report=False))
        finally:
            self.limit, self.stratify = saved
        self.conn.execute("BEGIN IMMEDIATE")
        self._flush_run_row(rep.status)
        self.conn.execute("UPDATE loops_run SET finished_at=? WHERE id=?", (utcnow(), self._run_id))
        self.conn.execute("COMMIT")

    def _process(self, items: list[Item]) -> None:
        rep = self.report
        window = self.workers * WINDOW_PER_WORKER
        executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="loops")
        pending: deque[tuple[Item, Future]] = deque()
        it = iter(items)
        done = 0
        try:
            while True:
                while len(pending) < window:
                    if self.max_seconds is not None and time.monotonic() - self._started >= self.max_seconds:
                        rep.stop_reason = f"--max-seconds {self.max_seconds:g} reached"
                        it = iter(())
                    item = next(it, None)
                    if item is None:
                        break
                    pending.append((item, executor.submit(self._worker, item)))
                if not pending:
                    break
                item, fut = pending.popleft()
                self._store(item, fut.result())
                done += 1
                if self._writes >= self.batch_size:
                    self._checkpoint()
                    self.progress(
                        f"  checkpoint {done}/{len(items)} decoded={rep.decoded} ok={rep.ok}"
                        f" failed={rep.statuses.get(DECODE_FAILED, 0)} unresolved={rep.unresolved}"
                        f" {time.monotonic() - self._started:.0f}s")
        except BaseException:
            self.registry.kill_all()
            raise
        finally:
            for _, f in pending:
                f.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    # ---- storing (main thread only) -------------------------------------------------------
    def _store(self, item: Item, out: Outcome) -> None:
        rep, conn = self.report, self.conn
        now = utcnow()
        rep.statuses[out.status] += 1
        rep.item_seconds += out.seconds
        if out.spawned:
            rep.decoded += 1
        select_json = json.dumps(item.select, sort_keys=True, separators=(",", ":"))
        if out.status == "ok":
            rep.ok += 1
            a = out.analysis
            rep.tempo_reasons[a.get("tempo_reason") or "estimated"] += 1
            rep.key_reasons[a.get("key_reason") or "estimated"] += 1
            descriptor = json.dumps(
                {"stage_version": LOOPS_VERSION, "settings": self.settings, "decoded_frames": out.frames,
                 "analysis": a},
                ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)
            conn.execute(
                "UPDATE resource SET valid=1, valid_error=NULL, valid_scope=?, decoded_frames=?,"
                " loops_status='ok', loops_error=NULL, loops_at=?, loops_select=?, loops_json=?,"
                " bpm_est=?, bpm_confidence=?, bpm_reason=?, beats=?, bars_est=?, bpm_hint_check=?,"
                " length_check=?, key_est=?, key_score=?, key_margin=?, key_strength=?, key_reason=?,"
                " tuning_cents=?, key_hint_check=? WHERE id=?",
                (VALID_SCOPE, out.frames, now, select_json, descriptor,
                 a["bpm_est"], a["tempo_confidence"], a["tempo_reason"], a["beats"], a["bars_est"],
                 a["tempo_hint_comparison"]["relation"] if item.bpm_name is not None else None,
                 a["length_vs_est"]["status"],
                 a["key_est"], a["key_score"], a["key_margin"], a["tonal_strength"], a["key_reason"],
                 a["tuning_cents"], a["key_hint_comparison"]["relation"] if item.key_name else None, item.rid),
            )
            rep.valid1 += 1
        else:
            # Never touch estimates here; `valid` only with conclusive evidence.
            if out.status == "analysis_error":
                conn.execute(
                    "UPDATE resource SET valid=1, valid_error=NULL, valid_scope=?, decoded_frames=? WHERE id=?",
                    (VALID_SCOPE, out.frames, item.rid))
                rep.valid1 += 1
            elif out.status == DECODE_FAILED:
                conn.execute("UPDATE resource SET valid=0, valid_error=?, valid_scope=? WHERE id=?",
                             (out.error, VALID_SCOPE, item.rid))
                rep.valid0 += 1
            conn.execute("UPDATE resource SET loops_status=?, loops_error=?, loops_at=?, loops_select=? WHERE id=?",
                         (out.status, out.error, now, select_json, item.rid))
        conn.execute(
            "INSERT OR REPLACE INTO resource_stage(resource_id, stage, version, done_at, error, input_sig)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (item.rid, LOOPS_STAGE, LOOPS_VERSION, now, None if out.status == "ok" else f"{out.status}: {out.error}",
             self.sig))
        self._writes += 1


def stratified_order(items: list[Item], seed: int) -> list[Item]:
    """Round-robin over (folder, bpm_name bucket) strata; deterministic hash order inside each."""
    strata: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for it in items:
        folder = it.paths[0][1].rsplit("/", 1)[0] if it.paths else ""
        strata[(folder, bpm_bucket(it.bpm_name))].append(it)

    def rank(it: Item) -> str:
        return hashlib.sha256(f"{seed}:{it.rid}".encode()).hexdigest()

    queues = [deque(sorted(v, key=rank)) for _, v in sorted(strata.items())]
    out: list[Item] = []
    while queues:
        nxt = []
        for q in queues:
            out.append(q.popleft())
            if q:
                nxt.append(q)
        queues = nxt
    return out
