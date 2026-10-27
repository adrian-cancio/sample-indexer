"""L3-6 streaming full-stream descriptors, isolated from earlier DSP measurements.

One ffmpeg process splits the first audio stream: unchanged native PCM feeds the
streaming analyzer; loudnorm's INPUT measurements feed loudness descriptors. Neither
normalized PCM nor output loudness is used. Only the main thread writes SQLite.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field

import numpy as np

from .db import utcnow
from .lowend import Item, VALID_SCOPE, classify_failure, _StratumView
from .metadata import ProcessRegistry, _strata_order

AUDIO_STAGE = "audio"
AUDIO_VERSION = 1
DEFAULT_AUDIO_WORKERS = 2
DEFAULT_AUDIO_BATCH = 50
DEFAULT_DECODE_TIMEOUT_S = 120.0
CHUNK_BYTES = 65536
STDERR_BYTES = 65536


def settings() -> dict:
    return {"stage_version": AUDIO_VERSION, "pcm": "native f32le first audio stream",
            "meter": "loudnorm input_i/input_tp I=-23:TP=-2:LRA=7",
            "dsp_version": getattr(_dsp(), "VERSION", 1)}


def _dsp():
    # Lazy import lets schema/CLI tooling load before the independent DSP module exists.
    from . import audio_dsp
    return audio_dsp


def settings_sig() -> str:
    return hashlib.sha256(json.dumps(settings(), sort_keys=True).encode()).hexdigest()[:32]


@dataclass
class Outcome:
    status: str
    error: str | None = None
    frames: int = 0
    analysis: dict | None = None
    spawned: bool = False
    full_decode: bool = False


def parse_loudness(stderr: str) -> tuple[dict, str]:
    """Remove only loudnorm's JSON, leaving decoder warnings for validity decisions."""
    matches = list(re.finditer(r'\{[^{}]*"input_i"[^{}]*\}', stderr, re.S))
    if not matches:
        return {}, stderr
    match = matches[-1]
    try:
        data = json.loads(match.group())
        values = {}
        for source, dest in (("input_i", "integrated_lufs"), ("input_tp", "true_peak_dbtp")):
            value = float(data[source])
            values[dest] = value if math.isfinite(value) else None
        values["loudness_status"] = "ok" if values["integrated_lufs"] is not None else "below_gate_or_short"
    except (ValueError, KeyError, TypeError):
        return {}, stderr
    # Level-prefixed info logging exposes the meter without confusing input paths
    # or informational codec descriptions with decoder diagnostics.
    return values, stderr[:match.start()] + stderr[match.end():]


def _diagnostics(text: str) -> str:
    """ffmpeg info-level is required for loudnorm JSON; identify suspicious diagnostics."""
    lines = []
    for line in text.splitlines():
        # WAV commonly omits a layout mask. Guessing mono/stereo from the native
        # channel count does not imply a damaged or incomplete source decode.
        if re.search(r"\[warning\] Guessed Channel Layout: (?:mono|stereo)$", line):
            continue
        if re.search(r"\[(?:warning|error|fatal|panic)\]", line, re.I):
            lines.append(line.strip())
    return " ".join(lines)[:1000]


def ffmpeg_decode(path: str, sr: int, channels: int, *, timeout: float = DEFAULT_DECODE_TIMEOUT_S,
                  ffmpeg_bin: str = "ffmpeg", registry: ProcessRegistry | None = None,
                  analyzer_factory=None) -> Outcome:
    """Read bounded frame-aligned blocks; drain bounded stderr concurrently; kill on timeout."""
    if sr <= 0 or channels <= 0 or channels > 64:
        return Outcome("unresolved_metadata", "unsupported sample rate/channel count")
    try:
        module = _dsp() if analyzer_factory is None else None
        factory = analyzer_factory or getattr(module, "StreamingAnalyzer", None) or module.AudioAccumulator
        analyzer = factory(sr, channels)
    except Exception as exc:
        return Outcome("analysis_error", f"{type(exc).__name__}: {exc}")
    argv = [ffmpeg_bin, "-nostdin", "-hide_banner", "-nostats", "-v", "level+info", "-xerror",
            "-threads", "1", "-protocol_whitelist", "file", "-i", "file:" + path,
            "-filter_complex_threads", "1", "-filter_complex",
            "[0:a:0]asplit=2[pcm][meter];[meter]loudnorm=I=-23:TP=-2:LRA=7:print_format=json[loud]",
            "-map", "[pcm]", "-ar", str(sr), "-ac", str(channels), "-threads", "1",
            "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1",
            "-map", "[loud]", "-threads", "1", "-f", "null", "-"]
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        return Outcome("tool_error", str(exc))
    if registry:
        registry.add(proc)
    tail = bytearray()
    diagnostic_lines = []

    def drain():
        pending_line = b""
        for chunk in iter(lambda: proc.stderr.read(4096), b""):
            tail.extend(chunk)
            if len(tail) > STDERR_BYTES:
                del tail[:-STDERR_BYTES]
            pending_line += chunk
            lines = pending_line.split(b"\n")
            pending_line = lines.pop()
            for line in lines:
                diagnostic = _diagnostics(line.decode("utf-8", "replace"))
                if diagnostic and len(diagnostic_lines) < 8:
                    diagnostic_lines.append(diagnostic)
            # Bound even a malformed diagnostic line without a newline.
            if len(pending_line) > STDERR_BYTES:
                diagnostic = _diagnostics(pending_line.decode("utf-8", "replace"))
                if diagnostic and len(diagnostic_lines) < 8:
                    diagnostic_lines.append(diagnostic)
                pending_line = pending_line[-4096:]
        diagnostic = _diagnostics(pending_line.decode("utf-8", "replace"))
        if diagnostic and len(diagnostic_lines) < 8:
            diagnostic_lines.append(diagnostic)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    expired = threading.Event()

    def expire():
        expired.set()
        try:
            proc.kill()
        except OSError:
            pass

    timer = threading.Timer(timeout, expire)
    timer.start()
    frames = 0
    remainder = b""
    analysis_error = None
    try:
        while True:
            chunk = proc.stdout.read(CHUNK_BYTES)
            if not chunk:
                break
            block = remainder + chunk
            aligned = len(block) // (4 * channels) * (4 * channels)
            if aligned:
                samples = np.frombuffer(block[:aligned], dtype="<f4").reshape(-1, channels)
                frames += len(samples)
                if analysis_error is None:
                    try:
                        analyzer.update(samples)
                    except Exception as exc:
                        analysis_error = f"{type(exc).__name__}: {exc}"
            remainder = block[aligned:]
        proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        thread.join()
        proc.stdout.close()
        proc.stderr.close()
        if registry:
            registry.discard(proc)
    if expired.is_set():
        return Outcome("timeout", f"ffmpeg exceeded {timeout:g}s", frames, spawned=True)
    if registry and registry.stopping:
        return Outcome("interrupted", "stage interrupted", frames, spawned=True)
    stderr = tail.decode("utf-8", "replace")
    if proc.returncode:
        return Outcome(classify_failure(proc.returncode, stderr), _diagnostics(stderr) or stderr[-1000:],
                       frames, spawned=True)
    loudness, remaining = parse_loudness(stderr)
    warnings = " ".join(diagnostic_lines) or _diagnostics(remaining)
    if warnings:
        return Outcome("decode_warnings", warnings, frames, spawned=True)
    if remainder or not frames:
        return Outcome("empty_audio" if not frames else "tool_error", "empty or incomplete PCM frames",
                       frames, spawned=True)
    if analysis_error:
        return Outcome("analysis_error", analysis_error, frames, spawned=True, full_decode=True)
    try:
        analysis = analyzer.finish()
        analysis.update(loudness)
        reasons = analysis.setdefault("null_reasons", {})
        for key in ("integrated_lufs", "true_peak_dbtp"):
            if analysis.get(key) is None:
                analysis[key] = None
                reasons[key] = ("meter_unavailable" if not loudness else
                                "shorter_than_400ms_gate" if key == "integrated_lufs" and frames / sr < 0.4 else
                                "silent_or_below_absolute_gate" if key == "integrated_lufs" else "silent")
        if not loudness:
            analysis["loudness_status"] = "unavailable"
        # Reject non-JSON descriptors before crossing back to the SQLite thread.
        json.dumps(analysis, allow_nan=False)
    except Exception as exc:
        return Outcome("analysis_error", f"{type(exc).__name__}: {exc}", frames,
                       spawned=True, full_decode=True)
    return Outcome("ok", frames=frames, analysis=analysis, spawned=True, full_decode=True)


@dataclass
class AudioReport:
    status: str = "running"
    stop_reason: str | None = None
    selected: int = 0
    already_done: int = 0
    audio_with_path: int = 0
    decoded: int = 0
    ok: int = 0
    valid1: int = 0
    valid0: int = 0
    pending_after: int = 0
    elapsed_s: float = 0.0
    statuses: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def processed(self):
        return sum(self.statuses.values())

    @property
    def unresolved(self):
        return sum(n for status, n in self.statuses.items() if status not in ("ok", "decode_failed"))

    def format(self):
        return (f"audio {self.status}" + (f" ({self.stop_reason})" if self.stop_reason else "") +
                f"\n  selected {self.selected}; already complete {self.already_done}"
                f"\n  processed {self.processed}; ok {self.ok}; unresolved {self.unresolved}"
                f"\n  ffmpeg subprocesses {self.decoded}; valid=1 / valid=0 {self.valid1} / {self.valid0}"
                f"\n  statuses {dict(self.statuses)}"
                f"\n  still pending {self.pending_after}; elapsed {self.elapsed_s:.2f}s")


class AudioStage:
    def __init__(self, conn, *, workers=DEFAULT_AUDIO_WORKERS, limit=None, force=False,
                 retry_errors=False, stratify=False, seed=0, timeout=DEFAULT_DECODE_TIMEOUT_S,
                 batch_size=DEFAULT_AUDIO_BATCH, max_seconds=None, ffmpeg_bin="ffmpeg",
                 decode_func=None, progress=None):
        if workers < 1 or batch_size < 1 or timeout <= 0 or (limit is not None and limit < 0):
            raise ValueError("invalid worker, batch, timeout or limit setting")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        self.conn, self.workers, self.limit = conn, workers, limit
        self.force, self.retry_errors = force, retry_errors
        self.stratify, self.seed = stratify, seed
        self.timeout, self.batch_size, self.max_seconds = timeout, batch_size, max_seconds
        self.ffmpeg_bin, self.decode_func = ffmpeg_bin, decode_func
        self.progress = progress or (lambda msg: None)
        self.registry = ProcessRegistry()
        self.report = AudioReport()
        self.sig = settings_sig()

    def select(self):
        paths = defaultdict(list)
        for r in self.conn.execute("SELECT resource_id, root, rel_path, size, mtime_ns FROM path"
                                   " WHERE missing=0 ORDER BY rel_path"):
            paths[r["resource_id"]].append((r["root"], r["rel_path"], r["size"], r["mtime_ns"]))
        stages = {r["resource_id"]: r for r in self.conn.execute(
            "SELECT resource_id, version, input_sig, error FROM resource_stage WHERE stage='audio'")}
        items = []
        for r in self.conn.execute("SELECT id,sr,channels,duration_s,probe_status FROM resource"
                                   " WHERE lower(ext) NOT IN ('.mid','.midi') ORDER BY id"):
            if r["id"] not in paths:
                continue
            self.report.audio_with_path += 1
            old = stages.get(r["id"])
            current = old is not None and old["version"] == AUDIO_VERSION and old["input_sig"] == self.sig
            need = self.force or (current and old["error"] is not None if self.retry_errors else not current)
            if need:
                items.append(Item(r["id"], r["sr"], r["channels"], r["duration_s"], r["probe_status"],
                                  {}, paths[r["id"]]))
            else:
                self.report.already_done += 1
        if self.stratify:
            items = [v.item for v in _strata_order([_StratumView(i) for i in items], self.seed)]
        if self.limit is not None:
            items = items[:self.limit]
        self.report.selected = len(items)
        return items

    def _worker(self, item):
        if not item.sr or not item.channels or item.probe_status != "ok":
            return Outcome("unresolved_metadata", "native sample rate/channels unavailable; run metadata")
        last = Outcome("unreadable", "no readable present path")
        for root, rel, size, mtime in item.paths:
            path = os.path.join(root, *rel.split("/"))
            try:
                st = os.stat(path)
            except OSError as exc:
                last = Outcome("unreadable", str(exc))
                continue
            if (st.st_size, st.st_mtime_ns) != (size, mtime):
                last = Outcome("stale", "file changed since scan; rescan")
                continue
            if self.decode_func:
                out = self.decode_func(path, item.sr, item.channels)
            else:
                out = ffmpeg_decode(path, item.sr, item.channels, timeout=self.timeout,
                                    ffmpeg_bin=self.ffmpeg_bin, registry=self.registry)
            try:
                after = os.stat(path)
                unchanged = (after.st_size, after.st_mtime_ns) == (size, mtime)
            except OSError:
                unchanged = False
            if not unchanged:
                return Outcome("stale", "file changed during decode; rescan", spawned=out.spawned)
            return out
        return last

    def _store(self, item, out):
        now = utcnow()
        descriptor = json.dumps({"stage_version": AUDIO_VERSION, "settings": settings(),
                                 "decoded_frames": out.frames, "analysis": out.analysis},
                                sort_keys=True, allow_nan=False)
        error = None if out.status == "ok" else f"{out.status}: {out.error}"
        self.conn.execute("INSERT OR REPLACE INTO audio_measurement"
                          "(resource_id,status,error,measured_at,descriptor) VALUES (?,?,?,?,?)",
                          (item.rid, out.status, error, now, descriptor))
        self.conn.execute("INSERT OR REPLACE INTO resource_stage"
                          "(resource_id,stage,version,done_at,error,input_sig) VALUES (?,?,?,?,?,?)",
                          (item.rid, AUDIO_STAGE, AUDIO_VERSION, now, error, self.sig))
        # Inconclusive attempts MUST preserve previous validity, scope and frame count.
        if out.full_decode or out.status == "decode_failed":
            valid = int(out.full_decode)
            self.conn.execute("UPDATE resource SET valid=?,valid_error=?,valid_scope=?,decoded_frames=? WHERE id=?",
                              (valid, None if valid else out.error, VALID_SCOPE, out.frames, item.rid))
            an = out.analysis or {}
            try:
                self.conn.execute("UPDATE resource SET centroid=?,rolloff=?,flatness=?,lufs=?,true_peak=? WHERE id=?",
                                  (an.get("spectral_centroid_hz"), an.get("spectral_rolloff85_hz"),
                                   an.get("spectral_flatness"), an.get("integrated_lufs"),
                                   an.get("true_peak_dbtp"), item.rid))
            except sqlite3.OperationalError:
                pass
            if valid:
                self.report.valid1 += 1
            else:
                self.report.valid0 += 1
        self.report.statuses[out.status] += 1
        self.report.decoded += int(out.spawned)
        self.report.ok += int(out.status == "ok")

    def run(self):
        if self.decode_func is None and shutil.which(self.ffmpeg_bin) is None:
            raise FileNotFoundError(f"{self.ffmpeg_bin} not found on PATH")
        started = time.monotonic()
        items = iter(self.select())
        deadline = started + self.max_seconds if self.max_seconds is not None else None
        pending = {}
        executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="audio")
        writes = 0
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            exhausted = False
            while pending or not exhausted:
                if deadline is not None and time.monotonic() >= deadline:
                    self.report.stop_reason = "max-seconds reached"
                    self.registry.kill_all()
                    break
                while not exhausted and len(pending) < self.workers * 2:
                    item = next(items, None)
                    if item is None:
                        exhausted = True
                        break
                    pending[executor.submit(self._worker, item)] = item
                if not pending:
                    break
                done, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    self._store(pending.pop(future), future.result())
                    writes += 1
                    if writes >= self.batch_size:
                        self.conn.execute("COMMIT")
                        self.progress(f"audio checkpoint: {self.report.processed}/{self.report.selected} processed; "
                                      f"decoded={self.report.decoded} ok={self.report.ok} "
                                      f"statuses={dict(self.report.statuses)} elapsed={time.monotonic()-started:.1f}s")
                        self.conn.execute("BEGIN IMMEDIATE")
                        writes = 0
        except KeyboardInterrupt:
            self.report.stop_reason = "keyboard interrupt"
            self.registry.kill_all()
        except BaseException:
            self.registry.kill_all()
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        if self.conn.in_transaction:
            self.conn.execute("COMMIT")
        self.report.status = "interrupted" if self.report.stop_reason else "complete"
        self.report.elapsed_s = time.monotonic() - started
        probe = AudioStage(self.conn, decode_func=self.decode_func)
        self.report.pending_after = len(probe.select())
        return self.report
