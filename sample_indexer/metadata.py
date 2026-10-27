"""Stage `metadata` (L3-3): cheap ffprobe header metadata + provisional path/name labels.

Two resumable, versioned stages share one command and are tracked in `resource_stage`:

* `probe`  (PROBE_VERSION)  - one ffprobe subprocess per audio resource, using ONE present
  path. Scope is HEADER/CONTAINER ONLY: a successful probe says the container header parsed,
  never that the audio decodes. `resource.valid` / `valid_error` are NOT written here (they
  stay NULL until a later full-decode stage). MIDI is never given to ffprobe.
* `labels` (LABELS_VERSION) - pure path/name interpretation (see labels.py) for audio and
  MIDI. `resource_stage.input_sig` fingerprints the inputs (present paths + duration), so a
  renamed file refreshes its path-derived labels although its hash is unchanged.

Selection (per run): resources with at least one present path, ordered by id (or round-robin
over folder x extension strata with --stratify) and capped by --limit. A resource is selected
when a stage row is missing, has an older version, or (labels) a different input_sig. A stage
row that recorded an ERROR at the current version counts as processed: it is NOT retried
automatically (no retry storm); `--retry-errors` re-selects exactly those, `--force` redoes
everything selected.

Concurrency: a bounded ThreadPoolExecutor, each worker running ONE ffprobe subprocess (argv
list, no shell, stdin closed, hard timeout). The main thread is the only SQLite writer; it
commits every `batch_size` results (a checkpoint) and refreshes the `metadata_run` row.
Ctrl-C (or --max-seconds) kills in-flight subprocesses, commits finished work and returns
status `interrupted`; running again resumes from the database state.
"""

from __future__ import annotations

import hashlib
import json
import math
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

from . import labels
from .db import utcnow

PROBE_STAGE = "probe"
LABELS_STAGE = "labels"
PROBE_VERSION = 1
LABELS_VERSION = labels.LABELS_VERSION
PROBE_SCOPE = "header_container"

# Outcomes that say something about the FILE (header parsed / is not an audio container).
# The others (timeout, tool_error, stale, unreadable) are about this attempt, not the content.
CONCLUSIVE_STATUSES = {"ok", "empty", "malformed", "no_audio_stream"}

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_META_WORKERS = 4
DEFAULT_BATCH = 200
WINDOW_PER_WORKER = 4

_FFPROBE_FIELDS = (
    "format=format_name,duration:"
    "stream=codec_type,codec_name,sample_fmt,sample_rate,channels,"
    "bits_per_sample,bits_per_raw_sample,duration"
)


# ----------------------------------------------------------------------------- probing


@dataclass
class ProbeResult:
    status: str  # ok|empty|malformed|no_audio_stream|unreadable|stale|timeout|tool_error
    error: str | None = None
    duration_s: float | None = None
    sr: int | None = None
    channels: int | None = None
    bits: int | None = None
    bits_source: str | None = None
    container: str | None = None
    codec_name: str | None = None
    sample_fmt: str | None = None
    sample_type: str | None = None  # 'int' | 'float' (from the codec name), None if unknown
    probed: bool = True  # False when no subprocess was spawned (empty file, unreadable)


def _num(value, cast):
    """Parse an ffprobe field; 'N/A', missing, non-finite or non-positive -> None."""
    if value in (None, "", "N/A"):
        return None
    try:
        out = cast(value)
    except (TypeError, ValueError):
        return None
    if isinstance(out, float) and not math.isfinite(out):
        return None
    return out if out > 0 else None


def sample_type_of(codec_name: str | None) -> str | None:
    if not codec_name:
        return None
    if codec_name.startswith("pcm_f"):
        return "float"
    if codec_name.startswith(("pcm_s", "pcm_u")):
        return "int"
    return None  # compressed / unknown codec: do not guess


def parse_ffprobe_json(text: str) -> ProbeResult:
    """Map ffprobe -of json output to a ProbeResult. Unknown fields stay None."""
    try:
        doc = json.loads(text)
    except ValueError as exc:
        return ProbeResult("tool_error", f"unparseable ffprobe output: {exc}")
    streams = [s for s in doc.get("streams") or [] if s.get("codec_type") == "audio"]
    if not streams:
        return ProbeResult("no_audio_stream", "no audio stream found in container")
    stream, fmt = streams[0], doc.get("format") or {}
    codec = stream.get("codec_name")
    # bits: prefer bits_per_raw_sample (true coded depth, e.g. 24 for s24 carried in s32);
    # else bits_per_sample (16 for s16, 32 for pcm_f32le). Both absent/0 -> unknown (None).
    bits = bits_source = None
    for key in ("bits_per_raw_sample", "bits_per_sample"):
        value = _num(stream.get(key), int)
        if value is not None:
            bits, bits_source = value, key
            break
    duration = _num(stream.get("duration"), float)
    if duration is None:
        duration = _num(fmt.get("duration"), float)
    return ProbeResult(
        "ok",
        duration_s=duration,
        sr=_num(stream.get("sample_rate"), int),
        channels=_num(stream.get("channels"), int),
        bits=bits,
        bits_source=bits_source,
        container=fmt.get("format_name"),
        codec_name=codec,
        sample_fmt=stream.get("sample_fmt") or None,
        sample_type=sample_type_of(codec),
    )


class ProcessRegistry:
    """Tracks live subprocesses so an interrupt can kill them all."""

    def __init__(self):
        self._lock = threading.Lock()
        self._procs: set[subprocess.Popen] = set()
        self.stopping = False

    def add(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.add(proc)
            if self.stopping:
                proc.kill()

    def discard(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.discard(proc)

    def kill_all(self) -> None:
        with self._lock:
            self.stopping = True
            for proc in list(self._procs):
                try:
                    proc.kill()
                except OSError:
                    pass


def ffprobe_file(
    path: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    ffprobe_bin: str = "ffprobe",
    registry: ProcessRegistry | None = None,
) -> ProbeResult:
    """Run ffprobe on one file (argv list, no shell). HEADER-level only; never decodes."""
    argv = [
        ffprobe_bin, "-v", "error", "-protocol_whitelist", "file",
        "-select_streams", "a:0", "-of", "json", "-show_entries", _FFPROBE_FIELDS,
        "file:" + path,
    ]
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace",
        )
    except OSError as exc:
        return ProbeResult("tool_error", f"cannot run {ffprobe_bin}: {exc.strerror or exc}")
    if registry is not None:
        registry.add(proc)
    try:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return ProbeResult("timeout", f"ffprobe exceeded {timeout:g}s timeout")
    finally:
        if registry is not None:
            registry.discard(proc)
    if registry is not None and registry.stopping:
        return ProbeResult("tool_error", "interrupted")
    if proc.returncode != 0:
        msg = " ".join(err.split())[:300] or f"exit status {proc.returncode}"
        # ffprobe exits 1 with a message for unparseable input; other codes/signals are tool faults.
        if proc.returncode == 1:
            if "Permission denied" in msg or "No such file" in msg:
                return ProbeResult("unreadable", msg)
            return ProbeResult("malformed", msg)
        return ProbeResult("tool_error", f"ffprobe exit {proc.returncode}: {msg}")
    return parse_ffprobe_json(out)


def probe_candidates(
    candidates: list[tuple[str, int, int]],
    *,
    probe: Callable[[str], ProbeResult],
) -> ProbeResult:
    """Probe the first usable present path of a resource: (abs_path, size, mtime_ns) tuples.

    A path whose file is gone or no longer matches the scanned size/mtime is skipped in favour
    of the next present path; if none is usable the last problem is reported (never a guess).
    """
    last = ProbeResult("unreadable", "no present path to probe", probed=False)
    for abs_path, size, mtime_ns in candidates:
        try:
            st = os.stat(abs_path)
        except OSError as exc:
            last = ProbeResult("unreadable", f"{type(exc).__name__}: {exc.strerror or exc}", probed=False)
            continue
        if st.st_size != size or st.st_mtime_ns != mtime_ns:
            last = ProbeResult("stale", "file changed since scan (size/mtime differ); rescan", probed=False)
            continue
        if st.st_size == 0:
            return ProbeResult("empty", "zero-byte file", probed=False)
        result = probe(abs_path)
        try:
            st2 = os.stat(abs_path)
        except OSError:
            st2 = None
        if st2 is None or (st2.st_size, st2.st_mtime_ns) != (size, mtime_ns):
            last = ProbeResult("stale", "file changed while probing; rescan", probed=result.probed)
            continue
        return result
    return last


# ----------------------------------------------------------------------------- stage runner


@dataclass
class Item:
    rid: int
    ext: str
    duration_s: float | None
    label_basis: str | None
    paths: list[tuple[str, str, int, int]]  # (root, rel_path, size, mtime_ns) sorted by rel_path
    need_probe: bool
    need_labels: bool

    @property
    def is_audio(self) -> bool:
        return self.ext != ".mid"


@dataclass
class MetadataReport:
    status: str = "running"  # complete | interrupted | aborted
    stop_reason: str | None = None
    mode: str = "default"
    candidates: int = 0  # resources with a present path
    audio: int = 0
    midi: int = 0
    selected: int = 0
    probed: int = 0  # ffprobe subprocesses actually spawned
    probe_ok: int = 0
    probe_errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    labelled: int = 0
    midi_skipped: int = 0  # MIDI resources not sent to ffprobe
    already_done: int = 0
    pending_after: int = 0
    elapsed_s: float = 0.0

    @property
    def processed(self) -> int:
        return self.probe_ok + sum(self.probe_errors.values())

    def format(self) -> str:
        lines = [
            f"metadata {self.status}" + (f" ({self.stop_reason})" if self.stop_reason else "")
            + f" [{self.mode}]",
            f"  resources with a present path  {self.candidates} (audio {self.audio}, midi {self.midi})",
            f"  already complete (this version) {self.already_done}",
            f"  selected this run              {self.selected}",
            f"  ffprobe subprocesses           {self.probed}",
            f"  audio processed                {self.processed} (ok {self.probe_ok}, errors {sum(self.probe_errors.values())})",
            f"  midi skipped (not probed)      {self.midi_skipped}",
            f"  labels written                 {self.labelled}",
            f"  still pending                  {self.pending_after}",
            f"  elapsed                        {self.elapsed_s:.2f}s",
        ]
        for status, n in sorted(self.probe_errors.items()):
            lines.append(f"  probe error {status:<16} {n}")
        return "\n".join(lines)


def input_sig(paths: list[tuple[str, str, int, int]], duration_s: float | None) -> str:
    """Fingerprint of everything the labels stage reads (present paths + duration)."""
    payload = json.dumps(
        [[(r, p) for r, p, _, _ in paths], duration_s], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _strata_order(items: list[Item], seed: int) -> list[Item]:
    """Round-robin over (folder, ext) strata; deterministic pseudo-random order inside each."""
    strata: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for it in items:
        folder = it.paths[0][1].rsplit("/", 1)[0] if it.paths else ""
        strata[(folder, it.ext)].append(it)

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


class MetadataStage:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        workers: int = DEFAULT_META_WORKERS,
        limit: int | None = None,
        force: bool = False,
        retry_errors: bool = False,
        stratify: bool = False,
        seed: int = 0,
        timeout: float = DEFAULT_TIMEOUT_S,
        batch_size: int = DEFAULT_BATCH,
        max_seconds: float | None = None,
        ffprobe_bin: str = "ffprobe",
        probe_func: Callable[[str], ProbeResult] | None = None,
        progress: Callable[[str], None] | None = None,
        on_result: Callable[[MetadataReport], None] | None = None,
    ):
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
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
        self.ffprobe_bin = ffprobe_bin
        self.progress = progress or (lambda msg: None)
        self.on_result = on_result  # main thread, after each stored result
        self.registry = ProcessRegistry()
        self._custom_probe = probe_func
        self.report = MetadataReport(
            mode=",".join(
                m for m, on in (("force", force), ("retry-errors", retry_errors),
                                ("stratified", stratify)) if on
            ) or "default"
        )
        self._run_id: int | None = None
        self._started = 0.0
        self._writes = 0
        self._tx_open = False

    # ---- probe function (runs in worker threads) -----------------------------------------
    def _probe_one(self, abs_path: str) -> ProbeResult:
        if self._custom_probe is not None:
            return self._custom_probe(abs_path)
        return ffprobe_file(
            abs_path, timeout=self.timeout, ffprobe_bin=self.ffprobe_bin, registry=self.registry
        )

    def _worker(self, item: Item) -> ProbeResult:
        candidates = [
            (os.path.join(root, *rel.split("/")), size, mtime)
            for root, rel, size, mtime in item.paths
        ]
        return probe_candidates(candidates, probe=self._probe_one)

    # ---- selection -----------------------------------------------------------------------
    def select(self) -> list[Item]:
        conn, rep = self.conn, self.report
        stages: dict[tuple[int, str], tuple[int, str | None, str | None]] = {
            (r["resource_id"], r["stage"]): (r["version"], r["error"], r["input_sig"])
            for r in conn.execute(
                "SELECT resource_id, stage, version, error, input_sig FROM resource_stage"
                " WHERE stage IN (?, ?)", (PROBE_STAGE, LABELS_STAGE)
            )
        }
        paths: dict[int, list[tuple[str, str, int, int]]] = defaultdict(list)
        for r in conn.execute(
            "SELECT resource_id, root, rel_path, size, mtime_ns FROM path WHERE missing=0"
            " ORDER BY rel_path"
        ):
            paths[r["resource_id"]].append((r["root"], r["rel_path"], r["size"], r["mtime_ns"]))
        items: list[Item] = []
        for r in conn.execute(
            "SELECT id, ext, duration_s, label_basis FROM resource ORDER BY id"
        ):
            plist = paths.get(r["id"])
            if not plist:
                continue  # no present path: nothing to analyse (never probe vanished files)
            rep.candidates += 1
            audio = r["ext"] != ".mid"
            rep.audio += audio
            rep.midi += not audio
            need_probe = False
            if audio:
                row = stages.get((r["id"], PROBE_STAGE))
                need_probe = (
                    self.force
                    or row is None
                    or row[0] != PROBE_VERSION
                    or (self.retry_errors and row[1] is not None)
                )
            lrow = stages.get((r["id"], LABELS_STAGE))
            need_labels = (
                self.force
                or need_probe
                or lrow is None
                or lrow[0] != LABELS_VERSION
                or lrow[2] != input_sig(plist, r["duration_s"])
            )
            if not audio:
                rep.midi_skipped += 1
            if need_probe or need_labels:
                items.append(Item(r["id"], r["ext"], r["duration_s"], r["label_basis"], plist,
                                  need_probe, need_labels))
            else:
                rep.already_done += 1
        if self.retry_errors and not self.force:
            # retry mode touches only resources whose recorded probe error is being retried
            items = [i for i in items if i.need_probe]
        if self.stratify:
            items = _strata_order(items, self.seed)
        if self.limit is not None:
            items = items[: self.limit]
        rep.selected = len(items)
        return items

    # ---- batching ------------------------------------------------------------------------
    def _begin(self) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        self._tx_open, self._writes = True, 0

    def _checkpoint(self, final: bool = False) -> None:
        if self._tx_open:
            self._flush_run_row(commit_status=None)
            self.conn.execute("COMMIT")
            self._tx_open, self._writes = False, 0
        if not final:
            self._begin()

    def _flush_run_row(self, commit_status: str | None) -> None:
        rep = self.report
        self.conn.execute(
            "UPDATE metadata_run SET status=COALESCE(?, status), selected=?, processed=?, probed=?,"
            " probe_ok=?, probe_errors=?, labelled=?, elapsed_s=? WHERE id=?",
            (commit_status, rep.selected, rep.processed, rep.probed, rep.probe_ok,
             sum(rep.probe_errors.values()), rep.labelled, time.monotonic() - self._started,
             self._run_id),
        )

    # ---- main ----------------------------------------------------------------------------
    def run(self) -> MetadataReport:
        rep = self.report
        if self._custom_probe is None and shutil.which(self.ffprobe_bin) is None:
            raise FileNotFoundError(f"{self.ffprobe_bin} not found on PATH (install ffmpeg)")
        self._started = time.monotonic()
        cur = self.conn.execute(
            "INSERT INTO metadata_run(started_at, status, limit_n, workers, mode)"
            " VALUES (?, 'running', ?, ?, ?)",
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
        self._flush_run_row(commit_status=rep.status)
        self.conn.execute("UPDATE metadata_run SET finished_at=? WHERE id=?", (utcnow(), self._run_id))
        self.conn.execute("COMMIT")

    def _pending_count(self) -> int:
        """Resources still lacking current-version work (ignores --limit, force, retry)."""
        probe = MetadataStage(self.conn)
        return len(probe.select())

    def _process(self, items: list[Item]) -> None:
        rep = self.report
        window = self.workers * WINDOW_PER_WORKER
        executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="probe")
        pending: deque[tuple[Item, Future | None]] = deque()
        it = iter(items)
        done_count = 0
        try:
            while True:
                while len(pending) < window:
                    if self.max_seconds is not None and time.monotonic() - self._started >= self.max_seconds:
                        rep.stop_reason = f"--max-seconds {self.max_seconds:g} reached"
                        it = iter(())
                    item = next(it, None)
                    if item is None:
                        break
                    pending.append((item, executor.submit(self._worker, item) if item.need_probe else None))
                if not pending:
                    break
                item, fut = pending.popleft()
                probe = fut.result() if fut is not None else None
                self._store(item, probe)
                done_count += 1
                if self.on_result:
                    self.on_result(rep)
                if self._writes >= self.batch_size:
                    self._checkpoint()
                    self.progress(
                        f"  checkpoint {done_count}/{len(items)} probed={rep.probed}"
                        f" ok={rep.probe_ok} errors={sum(rep.probe_errors.values())}"
                        f" {time.monotonic() - self._started:.0f}s"
                    )
        except BaseException:
            # Ctrl-C or a fault: kill in-flight ffprobe processes so shutdown does not wait
            # for them; results of unfinished items are discarded and redone on resume.
            self.registry.kill_all()
            raise
        finally:
            for _, f in pending:
                if f is not None:
                    f.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    # ---- storing (main thread only) --------------------------------------------------------
    def _store(self, item: Item, probe: ProbeResult | None) -> None:
        rep, conn = self.report, self.conn
        now = utcnow()
        duration = item.duration_s
        if probe is not None:
            if probe.probed:
                rep.probed += 1
            if probe.status == "ok":
                rep.probe_ok += 1
            else:
                rep.probe_errors[probe.status] += 1
            err = None if probe.status == "ok" else f"{probe.status}: {probe.error}"
            # Header-level facts only; valid/valid_error are never touched here.
            if probe.status in CONCLUSIVE_STATUSES:
                conn.execute(
                    "UPDATE resource SET duration_s=?, sr=?, bits=?, channels=?, bits_source=?,"
                    " container=?, codec_name=?, sample_fmt=?, sample_type=?, probe_status=?,"
                    " probe_error=?, probe_scope=?, probe_at=? WHERE id=?",
                    (probe.duration_s, probe.sr, probe.bits, probe.channels, probe.bits_source,
                     probe.container, probe.codec_name, probe.sample_fmt, probe.sample_type,
                     probe.status, probe.error, PROBE_SCOPE, now, item.rid),
                )
            else:
                # transient/inconclusive (timeout, tool fault, stale, unreadable): record the
                # status only, keep any earlier conclusive header facts.
                conn.execute(
                    "UPDATE resource SET probe_status=?, probe_error=?, probe_scope=?, probe_at=?"
                    " WHERE id=?", (probe.status, probe.error, PROBE_SCOPE, now, item.rid),
                )
                duration = item.duration_s
            conn.execute(
                "INSERT OR REPLACE INTO resource_stage(resource_id, stage, version, done_at, error,"
                " input_sig) VALUES (?, ?, ?, ?, ?, NULL)",
                (item.rid, PROBE_STAGE, PROBE_VERSION, now, err),
            )
            if probe.status in CONCLUSIVE_STATUSES:
                duration = probe.duration_s
            self._writes += 1
        if item.need_labels:
            self._store_labels(item, duration, now)

    def _store_labels(self, item: Item, duration: float | None, now: str) -> None:
        rep, conn = self.report, self.conn
        sig = input_sig(item.paths, duration)
        try:
            res = labels.interpret_resource([p[1] for p in item.paths], item.ext, duration)
        except Exception as exc:  # never let one odd name abort the run
            conn.execute(
                "INSERT OR REPLACE INTO resource_stage(resource_id, stage, version, done_at, error,"
                " input_sig) VALUES (?, ?, ?, ?, ?, ?)",
                (item.rid, LABELS_STAGE, LABELS_VERSION, now, f"{type(exc).__name__}: {exc}", sig),
            )
            self._writes += 1
            return
        # name-parsed fields are always (re)derived; kind/category only while this stage owns them
        # (label_basis NULL or ours). A later DSP stage that sets another basis is never clobbered.
        conn.execute(
            "UPDATE resource SET bpm_name=?, bpm_name_raw=?, key_name=?, key_name_raw=?,"
            " beats_hint=?, bpm_check=?, label_evidence=? WHERE id=?",
            (res.bpm_name, res.bpm_name_raw, res.key_name, res.key_name_raw, res.beats_hint,
             res.bpm_check, res.evidence_json(), item.rid),
        )
        if item.label_basis in (None, labels.LABEL_BASIS):
            conn.execute(
                "UPDATE resource SET kind=?, kind_source=?, kind_confidence=?, category=?,"
                " category_source=?, category_confidence=?, label_basis=? WHERE id=?",
                (res.kind, res.kind_source, res.kind_confidence, res.category, res.category_source,
                 res.category_confidence, labels.LABEL_BASIS, item.rid),
            )
        conn.execute(
            "INSERT OR REPLACE INTO resource_stage(resource_id, stage, version, done_at, error,"
            " input_sig) VALUES (?, ?, ?, ?, NULL, ?)",
            (item.rid, LABELS_STAGE, LABELS_VERSION, now, sig),
        )
        rep.labelled += 1
        self._writes += 1

