"""Stage `lowend` (L3-4): full ffmpeg decode + deterministic low-end DSP for kick/rumble/bass
candidates. Versioned and resumable through `resource_stage` (stage = 'lowend').

Selection (documented in reports/L3-4.md; nothing is dropped for low label confidence):
  R1 `category`  : resource.category in {kick, rumble, bass} (any confidence, any source);
  R2 `folder`    : at least one PRESENT path lies in a low-end source folder (see LOWEND_FOLDERS),
                   even when the label stage ended `unknown` (folder/name conflict, negated name).
A resource is selected when it is audio, has a present path, and satisfies R1 or R2. `.mid` is never
selected. `resource.lowend_select` stores the exact rules that fired and the matched paths.

Decode: `ffmpeg -xerror -threads 1` over the WHOLE first audio stream at the native sample rate and
channel count, as float32 PCM on a pipe, read in 1 MiB chunks. Per-file memory is bounded by
`max_decode_bytes` (default 128 MiB of float32) and `max_duration_s` (default 120 s, from the probe
header); files beyond a cap are NOT truncated: they are recorded `unresolved_cap` and `valid` stays NULL.

Validity (the only stage allowed to write `resource.valid`):
  valid = 1  ffmpeg exit 0 with empty stderr for the entire stream;
  valid = 0  conclusive decode failure of the source (exit != 0 with a data/corruption message);
  NULL       anything inconclusive: timeout, missing/crashing tool, unreadable/stale file, caps,
             decode warnings, zero decoded frames.

Concurrency: bounded ThreadPoolExecutor (default 2 workers), each worker = one ffmpeg process plus the
numpy analysis of that file; the main thread is the only SQLite writer and commits every
`batch_size` results (checkpoint). --max-seconds / Ctrl-C kill in-flight ffmpeg processes, commit
finished work and return `interrupted`; a new invocation resumes from the database state.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import labels, lowend_dsp
from .db import utcnow
from .metadata import ProcessRegistry, _strata_order

LOWEND_STAGE = "lowend"
LOWEND_VERSION = 1
VALID_SCOPE = "full_decode_ffmpeg_xerror"

DEFAULT_LOWEND_WORKERS = 2
DEFAULT_DECODE_TIMEOUT_S = 60.0
DEFAULT_LOWEND_BATCH = 50
DEFAULT_MAX_DURATION_S = 120.0
DEFAULT_MAX_DECODE_BYTES = 128 * 1024 * 1024  # float64 analysis footprint cap (pipe cap = half)
CHUNK_BYTES = 1 << 20
WINDOW_PER_WORKER = 2

LOWEND_CATEGORIES = ("kick", "rumble", "bass")
# casefolded (top, group, sub) source folders that warrant inspection (sub None = any sub-folder).
LOWEND_FOLDERS: dict[tuple[str, str, str | None], str] = {
    ("samples", "drums - one shots", "kicks"): "kick",
    ("samples", "drums - one shots", "rumble"): "rumble",
    ("samples", "loops", "kick loops"): "kick",
    ("samples", "loops", "bass loops"): "bass",
    ("samples", "synths", "basslines"): "bass",
}

# Messages that mean "the SOURCE bytes did not decode" (conclusive), as opposed to tool faults.
_SOURCE_FAILURE_MARKERS = (
    "invalid data found", "corrupt", "error while decoding", "invalid data", "truncated",
    "header missing", "could not find codec parameters", "not enough data", "invalid pcm",
    "moov atom not found", "end of file",
)

DECODE_FAILED = "decode_failed"
CONCLUSIVE_VALID_STATUSES = {"ok", "analysis_error", DECODE_FAILED}


def effective_caps(max_duration_s: float, max_decode_bytes: int) -> tuple[float, int]:
    """Caps can only be tightened: the DSP refuses anything over its own limits."""
    return min(max_duration_s, lowend_dsp.MAX_SECONDS), min(max_decode_bytes, lowend_dsp.MAX_BYTES)


def settings(max_duration_s: float, max_decode_bytes: int) -> dict:
    dur, nbytes = effective_caps(max_duration_s, max_decode_bytes)
    return {
        "stage_version": LOWEND_VERSION,
        "dsp_version": lowend_dsp.VERSION,
        "max_duration_s": dur,
        "max_analysis_bytes_float64": nbytes,
        "decode": "ffmpeg -xerror -threads 1, native rate/channels, f32le pipe, whole stream",
        "valid_scope": VALID_SCOPE,
    }


# resource column <- lowend_dsp.analyze() key. Units: peak/rms/dc_offset are fractions of full
# scale (1.0); crest is dB (20*log10(peak/rms)); clipping is a fraction of samples; band_* are
# mean-square power (full-scale sine = 0.5) per band; sub_peak_hz / decay_ms in Hz / ms.
COLUMN_MAP = {
    "peak": "sample_peak", "rms": "rms", "crest": "crest_db", "dc_offset": "dc_offset",
    "clipping": "clipping_fraction", "clipped_samples": "clipped_samples",
    "band_sub": "band_sub", "band_low": "band_low", "band_mid": "band_mid", "band_high": "band_high",
    "sub_peak_hz": "sub_peak_hz", "f0_hz": "f0_hz", "decay_ms": "decay_ms",
    "stereo_corr_low": "stereo_corr_low",
}


def to_columns(analysis: dict) -> dict:
    return {col: analysis.get(key) for col, key in COLUMN_MAP.items()}


def settings_sig(max_duration_s: float, max_decode_bytes: int) -> str:
    payload = json.dumps(settings(max_duration_s, max_decode_bytes), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


# ----------------------------------------------------------------------------- selection


def selection_rules(rel_paths: list[str], category: str | None) -> dict | None:
    """Which selection rules fire for one resource, or None when it is not a candidate."""
    rules: list[str] = []
    folders: list[str] = []
    if category in LOWEND_CATEGORIES:
        rules.append("category")
    for rel in sorted(rel_paths):
        dirs = [p.casefold() for p in rel.split("/")[:-1]]
        top = dirs[0] if dirs else ""
        group = dirs[1] if len(dirs) > 1 else None
        sub = dirs[2] if len(dirs) > 2 else None
        for key in ((top, group, sub), (top, group, None)):
            if key in LOWEND_FOLDERS:
                folders.append(f"{'/'.join(p for p in key if p)}={LOWEND_FOLDERS[key]}")
                break
    if folders:
        rules.append("folder")
    if not rules:
        return None
    return {"rules": rules, "category": category, "folders": sorted(set(folders))}


# ----------------------------------------------------------------------------- decoding


@dataclass
class DecodeResult:
    # ok | decode_failed | decode_warnings | empty_audio | unresolved_cap | unreadable | stale |
    # timeout | tool_error
    status: str
    error: str | None = None
    samples: np.ndarray | None = None  # (frames, channels) float32
    frames: int | None = None
    spawned: bool = True


def classify_failure(returncode: int, stderr: str) -> str:
    low = stderr.lower()
    if "permission denied" in low or "no such file" in low:
        return "unreadable"
    if returncode < 0:  # killed by a signal we did not send
        return "tool_error"
    if any(m in low for m in _SOURCE_FAILURE_MARKERS):
        return DECODE_FAILED
    return "tool_error"


def ffmpeg_decode(
    path: str,
    channels: int,
    *,
    timeout: float = DEFAULT_DECODE_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_DECODE_BYTES,
    ffmpeg_bin: str = "ffmpeg",
    registry: ProcessRegistry | None = None,
) -> DecodeResult:
    """Decode the whole first audio stream to float32 (frames, channels). Streams the pipe in
    CHUNK_BYTES reads; a total beyond `max_bytes` aborts as unresolved_cap (never truncates)."""
    argv = [
        ffmpeg_bin, "-nostdin", "-v", "error", "-xerror", "-threads", "1",
        "-protocol_whitelist", "file", "-i", "file:" + path,
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-threads", "1",
        "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return DecodeResult("tool_error", f"cannot run {ffmpeg_bin}: {exc.strerror or exc}", spawned=False)
    if registry is not None:
        registry.add(proc)
    err_chunks: list[bytes] = []

    def drain_stderr() -> None:
        total = 0
        for chunk in iter(lambda: proc.stderr.read(4096), b""):
            if total < 8192:
                err_chunks.append(chunk)
            total += len(chunk)

    drainer = threading.Thread(target=drain_stderr, daemon=True)
    drainer.start()
    flags = {"timeout": False, "cap": False}

    def on_timeout() -> None:
        flags["timeout"] = True
        try:
            proc.kill()
        except OSError:
            pass

    timer = threading.Timer(timeout, on_timeout)
    timer.start()
    buf = bytearray()
    try:
        while True:
            chunk = proc.stdout.read(CHUNK_BYTES)
            if not chunk:
                break
            buf += chunk
            if len(buf) > max_bytes:
                flags["cap"] = True
                proc.kill()
                break
        proc.wait()
    finally:
        timer.cancel()
        drainer.join(timeout=5)
        proc.stdout.close()
        proc.stderr.close()
        if registry is not None:
            registry.discard(proc)
    stderr = b"".join(err_chunks).decode("utf-8", "replace")
    stderr = " ".join(stderr.split())[:300]
    if flags["timeout"]:
        return DecodeResult("timeout", f"ffmpeg exceeded {timeout:g}s timeout")
    if registry is not None and registry.stopping:
        return DecodeResult("tool_error", "interrupted")
    if flags["cap"]:
        return DecodeResult("unresolved_cap", f"decoded PCM exceeds {max_bytes} bytes (float32); not truncated")
    if proc.returncode != 0:
        return DecodeResult(classify_failure(proc.returncode, stderr), stderr or f"exit status {proc.returncode}")
    if stderr:
        return DecodeResult("decode_warnings", stderr)
    frame_bytes = 4 * channels
    if len(buf) % frame_bytes:
        return DecodeResult("tool_error", f"decoded {len(buf)} bytes not a multiple of {channels}x4")
    frames = len(buf) // frame_bytes
    if frames == 0:
        return DecodeResult("empty_audio", "decode succeeded but produced zero frames", frames=0)
    samples = np.frombuffer(buf, dtype="<f4").reshape(frames, channels)  # zero-copy view
    return DecodeResult("ok", samples=samples, frames=frames)


# ----------------------------------------------------------------------------- one resource


@dataclass
class Outcome:
    status: str
    error: str | None = None
    frames: int | None = None
    analysis: dict | None = None
    spawned: bool = False


@dataclass
class Item:
    rid: int
    sr: int | None
    channels: int | None
    duration_s: float | None
    probe_status: str | None
    select: dict
    paths: list[tuple[str, str, int, int]]  # (root, rel_path, size, mtime_ns)


def analyse_item(
    item: Item,
    *,
    decode: Callable[..., DecodeResult],
    max_duration_s: float,
    max_bytes: int,
) -> Outcome:
    if item.probe_status != "ok" or not item.sr or not item.channels:
        return Outcome("unresolved_metadata", f"header metadata missing (probe_status={item.probe_status},"
                       f" sr={item.sr}, channels={item.channels}); not decoded")
    if item.channels > 2:
        return Outcome("unresolved_metadata", f"{item.channels} channels unsupported by L3-4; not decoded")
    if item.duration_s is not None and item.duration_s > max_duration_s:
        return Outcome("unresolved_cap", f"duration {item.duration_s:.2f}s exceeds cap {max_duration_s:g}s; not decoded")
    if item.duration_s is not None and item.duration_s * item.sr * item.channels * 8 > max_bytes:
        return Outcome("unresolved_cap", f"float64 analysis footprint exceeds {max_bytes} bytes; not decoded")
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
        if res.frames / item.sr > max_duration_s or res.frames * item.channels * 8 > max_bytes:
            return Outcome("unresolved_cap", f"decoded {res.frames} frames ({res.frames / item.sr:.2f}s) exceed caps;"
                           " not analysed, not truncated", frames=res.frames, spawned=True)
        try:
            analysis = lowend_dsp.analyze(res.samples.astype(np.float64), item.sr)
        except Exception as exc:  # decode succeeded (valid=1) but the analysis faulted
            return Outcome("analysis_error", f"{type(exc).__name__}: {exc}", frames=res.frames, spawned=True)
        return Outcome("ok", frames=res.frames, analysis=analysis, spawned=True)
    return last


# ----------------------------------------------------------------------------- stage runner


@dataclass
class LowendReport:
    status: str = "running"
    stop_reason: str | None = None
    mode: str = "default"
    audio_with_path: int = 0
    candidates: int = 0  # selected by the rules, regardless of stage state
    by_rule: dict = field(default_factory=lambda: defaultdict(int))
    selected: int = 0  # needing work this run (after --limit)
    already_done: int = 0
    decoded: int = 0  # ffmpeg subprocesses spawned
    ok: int = 0
    statuses: dict = field(default_factory=lambda: defaultdict(int))
    valid1: int = 0
    valid0: int = 0
    pending_after: int = 0
    elapsed_s: float = 0.0

    @property
    def processed(self) -> int:
        return sum(self.statuses.values())

    @property
    def unresolved(self) -> int:
        return sum(n for s, n in self.statuses.items() if s != "ok" and s != DECODE_FAILED)

    def format(self) -> str:
        lines = [
            f"lowend {self.status}" + (f" ({self.stop_reason})" if self.stop_reason else "") + f" [{self.mode}]",
            f"  audio resources with a present path {self.audio_with_path}",
            f"  low-end candidates (rules)          {self.candidates}",
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
            f"  elapsed                             {self.elapsed_s:.2f}s",
        ]
        for status, n in sorted(self.statuses.items()):
            if status != "ok":
                lines.append(f"  status {status:<22} {n}")
        return "\n".join(lines)


class LowendStage:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        workers: int = DEFAULT_LOWEND_WORKERS,
        limit: int | None = None,
        force: bool = False,
        retry_errors: bool = False,
        stratify: bool = False,
        seed: int = 0,
        timeout: float = DEFAULT_DECODE_TIMEOUT_S,
        batch_size: int = DEFAULT_LOWEND_BATCH,
        max_seconds: float | None = None,
        max_duration_s: float = DEFAULT_MAX_DURATION_S,
        max_decode_bytes: int = DEFAULT_MAX_DECODE_BYTES,
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
        # caps may only be tightened below the DSP limits; max_decode_bytes is the float64
        # analysis footprint (the float32 pipe is capped at half of it)
        self.max_duration_s, self.max_decode_bytes = effective_caps(max_duration_s, max_decode_bytes)
        self.ffmpeg_bin = ffmpeg_bin
        self.progress = progress or (lambda msg: None)
        self.registry = ProcessRegistry()
        self._custom_decode = decode_func
        self.sig = settings_sig(max_duration_s, max_decode_bytes)
        self.report = LowendReport(
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
        return ffmpeg_decode(path, channels, timeout=self.timeout, max_bytes=self.max_decode_bytes // 2,
                             ffmpeg_bin=self.ffmpeg_bin, registry=self.registry)

    def _worker(self, item: Item) -> Outcome:
        return analyse_item(item, decode=self._decode, max_duration_s=self.max_duration_s,
                            max_bytes=self.max_decode_bytes)

    # ---- selection ----------------------------------------------------------------------
    def select(self) -> list[Item]:
        conn, rep = self.conn, self.report
        stages = {
            r["resource_id"]: (r["version"], r["error"], r["input_sig"])
            for r in conn.execute(
                "SELECT resource_id, version, error, input_sig FROM resource_stage WHERE stage=?",
                (LOWEND_STAGE,),
            )
        }
        paths: dict[int, list[tuple[str, str, int, int]]] = defaultdict(list)
        for r in conn.execute(
            "SELECT resource_id, root, rel_path, size, mtime_ns FROM path WHERE missing=0 ORDER BY rel_path"
        ):
            paths[r["resource_id"]].append((r["root"], r["rel_path"], r["size"], r["mtime_ns"]))
        items: list[Item] = []
        for r in conn.execute(
            "SELECT id, category, sr, channels, duration_s, probe_status, lowend_select FROM resource"
            " WHERE ext != '.mid' ORDER BY id"
        ):
            plist = paths.get(r["id"])
            if not plist:
                continue
            rep.audio_with_path += 1
            sel = selection_rules([p[1] for p in plist], r["category"])
            if sel is None:
                continue
            rep.candidates += 1
            rep.by_rule["+".join(sel["rules"])] += 1
            row = stages.get(r["id"])
            if self.retry_errors and not self.force:
                # retry mode touches only current-version rows that recorded an error
                need = (row is not None and row[0] == LOWEND_VERSION and row[2] == self.sig
                        and row[1] is not None)
            else:
                need = self.force or row is None or row[0] != LOWEND_VERSION or row[2] != self.sig
            if need:
                items.append(Item(r["id"], r["sr"], r["channels"], r["duration_s"], r["probe_status"], sel, plist))
            else:
                rep.already_done += 1
        if self.stratify:
            tmp = [_StratumView(i) for i in items]
            items = [v.item for v in _strata_order(tmp, self.seed)]  # type: ignore[arg-type]
        if self.limit is not None:
            items = items[: self.limit]
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
            "UPDATE lowend_run SET status=COALESCE(?, status), selected=?, processed=?, decoded=?, ok=?,"
            " failed=?, unresolved=?, elapsed_s=? WHERE id=?",
            (status, rep.selected, rep.processed, rep.decoded, rep.ok,
             rep.statuses.get(DECODE_FAILED, 0), rep.unresolved, time.monotonic() - self._started,
             self._run_id),
        )

    # ---- main ---------------------------------------------------------------------------
    def run(self) -> LowendReport:
        rep = self.report
        if self._custom_decode is None and shutil.which(self.ffmpeg_bin) is None:
            raise FileNotFoundError(f"{self.ffmpeg_bin} not found on PATH (install ffmpeg)")
        self._started = time.monotonic()
        cur = self.conn.execute(
            "INSERT INTO lowend_run(started_at, status, limit_n, workers, mode) VALUES (?, 'running', ?, ?, ?)",
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
        rep.pending_after = self._pending_count()
        self.conn.execute("BEGIN IMMEDIATE")
        self._flush_run_row(rep.status)
        self.conn.execute("UPDATE lowend_run SET finished_at=? WHERE id=?", (utcnow(), self._run_id))
        self.conn.execute("COMMIT")

    def _pending_count(self) -> int:
        probe = LowendStage(self.conn, max_duration_s=self.max_duration_s,
                            max_decode_bytes=self.max_decode_bytes, decode_func=lambda *a: None)  # type: ignore[arg-type]
        return len(probe.select())

    def _process(self, items: list[Item]) -> None:
        rep = self.report
        window = self.workers * WINDOW_PER_WORKER
        executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="lowend")
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
                        f" {time.monotonic() - self._started:.0f}s"
                    )
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
        if out.spawned:
            rep.decoded += 1
        if out.status == "ok":
            rep.ok += 1
        a = out.analysis or {}
        # full-decode validity: 1 = whole stream decoded; 0 = conclusive source failure; else NULL
        valid = valid_error = None
        if out.status in ("ok", "analysis_error"):
            valid = 1
            rep.valid1 += 1
        elif out.status == DECODE_FAILED:
            valid, valid_error = 0, out.error
            rep.valid0 += 1
        cols = to_columns(a) if a else {}
        descriptor = None
        if a:
            descriptor = json.dumps(
                {"stage_version": LOWEND_VERSION, "settings": settings(self.max_duration_s, self.max_decode_bytes),
                 "decoded_frames": out.frames, "analysis": a},
                ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False,
            )
        conn.execute(
            "UPDATE resource SET valid=?, valid_error=?, valid_scope=?, decoded_frames=?, lowend_status=?,"
            " lowend_error=?, lowend_at=?, lowend_select=?, lowend_json=?, peak=?, rms=?, crest=?,"
            " dc_offset=?, clipping=?, clipped_samples=?, band_sub=?, band_low=?, band_mid=?, band_high=?,"
            " sub_peak_hz=?, f0_hz=?, decay_ms=?, stereo_corr_low=? WHERE id=?",
            (valid, valid_error, VALID_SCOPE if valid is not None else None, out.frames, out.status,
             None if out.status == "ok" else out.error, now,
             json.dumps(item.select, sort_keys=True, separators=(",", ":")), descriptor,
             cols.get("peak"), cols.get("rms"), cols.get("crest"), cols.get("dc_offset"),
             cols.get("clipping"), cols.get("clipped_samples"), cols.get("band_sub"), cols.get("band_low"),
             cols.get("band_mid"), cols.get("band_high"), cols.get("sub_peak_hz"), cols.get("f0_hz"),
             cols.get("decay_ms"), cols.get("stereo_corr_low"), item.rid),
        )
        conn.execute(
            "INSERT OR REPLACE INTO resource_stage(resource_id, stage, version, done_at, error, input_sig)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (item.rid, LOWEND_STAGE, LOWEND_VERSION, now,
             None if out.status == "ok" else f"{out.status}: {out.error}", self.sig),
        )
        self._writes += 1


class _StratumView:
    """Adapter so metadata._strata_order (folder x ext round-robin) can order lowend Items."""

    def __init__(self, item: Item):
        self.item = item
        self.rid = item.rid
        self.paths = item.paths
        self.ext = os.path.splitext(item.paths[0][1])[1].lower() if item.paths else ""
