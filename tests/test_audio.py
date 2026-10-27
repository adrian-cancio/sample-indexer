"""L3-6 runner integration tests; all source files and databases are temporary."""
import json
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sample_indexer import audio, db
from sample_indexer.migrations import MIGRATIONS
from tests.test_lowend import write_pcm16, sine


class Accumulator:
    """Test-only PCM witness: detects normalized, resampled or truncated PCM."""
    def __init__(self, sr, channels):
        self.frames = 0
        self.peak = 0.0

    def update(self, samples):
        self.frames += len(samples)
        self.peak = max(self.peak, float(np.max(np.abs(samples))))

    def finish(self):
        return {"frames": self.frames, "peak": self.peak}


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "test.wav"

    def decode(self, **kwargs):
        return audio.ffmpeg_decode(str(self.path), 44100, 1,
                                   analyzer_factory=Accumulator, **kwargs)

    def test_original_pcm_and_input_loudness(self):
        write_pcm16(self.path, sine(1000, seconds=1.2, amp=0.25))
        out = self.decode()
        self.assertEqual(out.status, "ok", out.error)
        self.assertTrue(out.full_decode)
        self.assertEqual(out.frames, 52920)
        self.assertAlmostEqual(out.analysis["peak"], 0.25, places=3)
        self.assertAlmostEqual(out.analysis["true_peak_dbtp"], -12.04, delta=0.15)
        self.assertAlmostEqual(out.analysis["integrated_lufs"], -15.05, delta=0.3)

    def test_short_and_silent_are_not_invalid(self):
        for data in (np.zeros(44100), sine(1000, seconds=0.01)):
            with self.subTest(frames=len(data)):
                write_pcm16(self.path, data)
                out = self.decode()
                self.assertEqual(out.status, "ok", out.error)
                self.assertTrue(out.full_decode)
                self.assertIsNone(out.analysis["integrated_lufs"])
                expected = ("shorter_than_400ms_gate" if len(data) < 17640
                            else "silent_or_below_absolute_gate")
                self.assertEqual(out.analysis["null_reasons"]["integrated_lufs"], expected)
                json.dumps(out.analysis, allow_nan=False)

    def test_actual_dsp_api(self):
        write_pcm16(self.path, sine(1000))
        out = audio.ffmpeg_decode(str(self.path), 44100, 1)
        self.assertEqual(out.status, "ok", out.error)
        self.assertTrue(out.full_decode)
        json.dumps(out.analysis, allow_nan=False)

    def test_long_stream_has_no_duration_cap_and_bounded_updates(self):
        # Exceeds the earlier stages' 300-second cap without allocating a full PCM array.
        sr, seconds = 8000, 301
        block = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 8192).astype("<i2")
        with wave.open(str(self.path), "wb") as stream:
            stream.setparams((1, 2, sr, 0, "NONE", "not compressed"))
            for _ in range(seconds):
                stream.writeframesraw(block.tobytes())

        class Bounded(Accumulator):
            def update(self, samples):
                if samples.nbytes > audio.CHUNK_BYTES:
                    raise AssertionError("unbounded PCM update")
                super().update(samples)

        out = audio.ffmpeg_decode(str(self.path), sr, 1, analyzer_factory=Bounded)
        self.assertEqual(out.status, "ok", out.error)
        self.assertTrue(out.full_decode)
        self.assertEqual(out.frames, sr * seconds)
        self.assertEqual(out.analysis["frames"], sr * seconds)
        self.assertAlmostEqual(out.analysis["peak"], 0.25, places=3)

    def test_corrupt_is_conclusive(self):
        self.path.write_bytes(b"not a wave file")
        out = self.decode()
        self.assertEqual(out.status, "decode_failed", out.error)
        self.assertFalse(out.full_decode)

    def test_early_warning_survives_stderr_tail_eviction(self):
        fake = Path(self.tmp.name) / "fake_ffmpeg"
        fake.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "sys.stderr.buffer.write(b'[warning] early decoder warning\\n')\n"
            "sys.stderr.buffer.write(b'[info] harmless detail\\n' * 5000)\n"
            "sys.stderr.buffer.flush()\n"
            "sys.stdout.buffer.write(b'\\x00\\x00\\x80\\x3e' * 1024)\n"
            "sys.stdout.buffer.flush()\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        out = self.decode(ffmpeg_bin=str(fake))
        self.assertEqual(out.status, "decode_warnings", out.error)
        self.assertFalse(out.full_decode)
        self.assertTrue(out.spawned)
        self.assertEqual(out.frames, 1024)
        self.assertIn("early decoder warning", out.error)

    def test_real_process_timeout(self):
        write_pcm16(self.path, sine(1000))
        out = self.decode(timeout=0.00001)
        self.assertEqual(out.status, "timeout")
        self.assertFalse(out.full_decode)

    def test_analyzer_failure_still_finishes_full_decode(self):
        class Broken(Accumulator):
            def update(self, samples):
                raise RuntimeError("DSP fault")
        write_pcm16(self.path, sine(1000))
        out = audio.ffmpeg_decode(str(self.path), 44100, 1, analyzer_factory=Broken)
        self.assertEqual(out.status, "analysis_error", out.error)
        self.assertTrue(out.full_decode)
        self.assertEqual(out.frames, 35280)


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "test.wav"
        self.path.write_bytes(b"fixture")
        self.conn = sqlite3.connect(":memory:", isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.addCleanup(self.conn.close)
        self.conn.executescript("""
            CREATE TABLE resource(id INTEGER PRIMARY KEY,ext TEXT,sr INTEGER,channels INTEGER,
              duration_s REAL,probe_status TEXT,valid INTEGER,valid_error TEXT,valid_scope TEXT,
              decoded_frames INTEGER);
            CREATE TABLE path(resource_id INTEGER,root TEXT,rel_path TEXT,size INTEGER,
              mtime_ns INTEGER,missing INTEGER);
            CREATE TABLE resource_stage(resource_id INTEGER,stage TEXT,version INTEGER,
              done_at TEXT,error TEXT,input_sig TEXT,PRIMARY KEY(resource_id,stage));
            CREATE TABLE audio_measurement(resource_id INTEGER PRIMARY KEY,status TEXT,
              error TEXT,measured_at TEXT,descriptor TEXT);
        """)
        self.conn.execute("INSERT INTO resource VALUES(1,'.wav',44100,1,1,'ok',1,NULL,'old',12)")
        st = self.path.stat()
        self.conn.execute("INSERT INTO path VALUES(?,?,?,?,?,0)",
                          (1, self.tmp.name, self.path.name, st.st_size, st.st_mtime_ns))
        self.patch = patch.object(audio, "_dsp", return_value=SimpleNamespace(VERSION=1))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def run_stage(self, out=None, **kwargs):
        return audio.AudioStage(self.conn, decode_func=lambda *args: out or audio.Outcome(
            "ok", frames=100, analysis={"rms": 0.1}, full_decode=True), **kwargs).run()

    def test_resume_force_retry_and_validity_preservation(self):
        report = self.run_stage(audio.Outcome("timeout", "deadline", spawned=True))
        self.assertEqual(report.processed, 1)
        row = self.conn.execute("SELECT valid,valid_scope,decoded_frames FROM resource").fetchone()
        self.assertEqual(tuple(row), (1, "old", 12))
        self.assertEqual(self.run_stage().processed, 0)
        self.assertEqual(self.run_stage(retry_errors=True).processed, 1)
        self.assertEqual(self.run_stage(retry_errors=True).processed, 0)
        self.assertEqual(self.run_stage(force=True).processed, 1)
        descriptor = json.loads(self.conn.execute("SELECT descriptor FROM audio_measurement").fetchone()[0])
        self.assertEqual(descriptor["analysis"], {"rms": 0.1})

    def test_settings_signature_invalidates_only_semantic_changes(self):
        initial = audio.settings_sig()
        self.assertEqual(self.run_stage().processed, 1)
        self.assertEqual(self.run_stage(workers=1, timeout=999, batch_size=1).processed, 0)
        with patch.object(audio, "_dsp", return_value=SimpleNamespace(VERSION=2)):
            self.assertNotEqual(audio.settings_sig(), initial)
            self.assertEqual(self.run_stage().processed, 1)
            stored = self.conn.execute("SELECT input_sig FROM resource_stage").fetchone()[0]
            self.assertEqual(stored, audio.settings_sig())
            descriptor = json.loads(self.conn.execute(
                "SELECT descriptor FROM audio_measurement").fetchone()[0])
            self.assertEqual(descriptor["settings"]["dsp_version"], 2)
            self.assertEqual(self.run_stage().processed, 0)

    def test_timeout_checkpoint_is_committed_and_retryable(self):
        checkpoints = []

        def checkpoint(message):
            self.assertFalse(self.conn.in_transaction)
            row = self.conn.execute("SELECT error FROM resource_stage").fetchone()
            self.assertIn("timeout", row[0])
            checkpoints.append(message)

        report = self.run_stage(audio.Outcome("timeout", "deadline", spawned=True),
                                batch_size=1, progress=checkpoint)
        self.assertEqual(report.statuses["timeout"], 1)
        self.assertEqual(report.decoded, 1)
        self.assertEqual(len(checkpoints), 1)
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.run_stage().processed, 0)
        self.assertEqual(self.run_stage(retry_errors=True).processed, 1)
        self.assertIsNone(self.conn.execute("SELECT error FROM resource_stage").fetchone()[0])
        self.assertEqual(self.run_stage(retry_errors=True).processed, 0)

    def test_stale_source_never_decodes(self):
        self.path.write_bytes(b"changed fixture")
        report = self.run_stage()
        self.assertEqual(report.statuses["stale"], 1)
        self.assertEqual(report.valid1, 0)

    def test_midi_and_missing_excluded(self):
        self.conn.execute("UPDATE resource SET ext='.mid'")
        self.assertEqual(self.run_stage().selected, 0)
        self.conn.execute("UPDATE resource SET ext='.wav'")
        self.conn.execute("UPDATE path SET missing=1")
        self.assertEqual(self.run_stage().selected, 0)

    def test_max_seconds_leaves_work_resumable(self):
        def slow(*args):
            time.sleep(0.1)
            return audio.Outcome("ok", full_decode=True)
        report = audio.AudioStage(self.conn, max_seconds=0.01, decode_func=slow).run()
        self.assertEqual(report.status, "interrupted")
        self.assertEqual(report.pending_after, 1)
        self.assertEqual(self.run_stage().processed, 1)


class MigrationSafetyTests(unittest.TestCase):
    def test_v4_to_v5_keeps_pre_migration_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.sqlite"
            conn = db.connect(path)
            try:
                with patch.object(db, "MIGRATIONS", [(v, sql) for v, sql in MIGRATIONS if v <= 4]):
                    db.migrate(conn)
                conn.execute("INSERT INTO resource(sha256,size,ext,first_seen) VALUES(?,?,?,?)",
                             ("a" * 64, 7, ".wav", "fixture"))
                self.assertEqual(db.schema_version(conn), 4)
            finally:
                conn.close()
            migrated = db.open_db(path, create=True)
            try:
                self.assertEqual(db.schema_version(migrated), db.LATEST_VERSION)
                self.assertEqual(migrated.execute("SELECT size FROM resource").fetchone()[0], 7)
                self.assertEqual(migrated.execute("SELECT COUNT(*) FROM audio_measurement").fetchone()[0], 0)
            finally:
                migrated.close()
            backups = list(Path(tmp).glob("index.sqlite.bak-v4-*"))
            self.assertEqual(len(backups), 1)
            backup = db.connect(backups[0], create=False)
            try:
                self.assertEqual(db.schema_version(backup), 4)
                self.assertEqual(backup.execute("SELECT size FROM resource").fetchone()[0], 7)
                self.assertIsNone(backup.execute(
                    "SELECT name FROM sqlite_master WHERE name='audio_measurement'").fetchone())
            finally:
                backup.close()


if __name__ == "__main__":
    unittest.main()
