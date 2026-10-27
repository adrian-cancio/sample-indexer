import hashlib
import json
import os
import shutil
import stat
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from sample_indexer import cli, db, lowend
from sample_indexer.lowend import LowendStage, selection_rules
from sample_indexer.metadata import MetadataStage
from sample_indexer.scanner import Scanner

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
SR = 44100
KICKS = "Samples/Drums - One Shots/Kicks"


def sine(freq, seconds=0.8, sr=SR, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def write_pcm16(path, data, sr=SR):
    data = np.asarray(data)
    ch = 1 if data.ndim == 1 else data.shape[1]
    pcm = np.round(np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def write_pcm24(path, data, sr=SR):
    data = np.asarray(data)
    ch = 1 if data.ndim == 1 else data.shape[1]
    v = np.round(np.clip(data, -1, 1) * 8388607).astype("<i4").reshape(-1)
    raw = v.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(raw)


def write_f32(path, data, sr=SR):
    data = np.asarray(data, dtype="<f4")
    ch = 1 if data.ndim == 1 else data.shape[1]
    raw = data.reshape(-1).tobytes()
    fmt = struct.pack("<HHIIHH", 3, ch, sr, sr * ch * 4, ch * 4, 32)
    body = b"WAVEfmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(raw)) + raw
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sample-indexer-lowend-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "lib"
        self.root.mkdir()
        self.conn = db.open_db(self.tmp / "index.sqlite", create=True)
        self.addCleanup(self.conn.close)
        self._t = 1_700_000_000

    def put(self, rel, writer, data, **kw):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        writer(p, data, **kw)
        self._t += 10
        os.utime(p, ns=(self._t * 10**9,) * 2)
        return p

    def put_bytes(self, rel, data):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        self._t += 10
        os.utime(p, ns=(self._t * 10**9,) * 2)
        return p

    def prepare(self):
        Scanner(self.conn, self.root, workers=2).run()
        MetadataStage(self.conn, workers=2).run()

    def stage(self, **kw):
        kw.setdefault("workers", 2)
        return LowendStage(self.conn, **kw)

    def res(self, rel):
        return self.conn.execute(
            "SELECT r.* FROM resource r JOIN path p ON p.resource_id=r.id WHERE p.rel_path=?", (rel,)
        ).fetchone()


# ----------------------------------------------------------------------------- selection (pure)


class SelectionRuleTests(unittest.TestCase):
    def test_category_rule_fires_for_any_confidence(self):
        sel = selection_rules(["Samples/Unsorted/x.wav"], "bass")
        self.assertEqual(sel["rules"], ["category"])

    def test_folder_rule_fires_when_category_is_unknown(self):
        # "no kick" in the name makes the label stage say unknown; the folder still warrants inspection
        sel = selection_rules([f"{KICKS}/Sub no kick.wav"], "unknown")
        self.assertEqual(sel["rules"], ["folder"])
        self.assertEqual(sel["folders"], ["samples/drums - one shots/kicks=kick"])

    def test_both_rules(self):
        sel = selection_rules([f"{KICKS}/a.wav"], "kick")
        self.assertEqual(sel["rules"], ["category", "folder"])

    def test_non_candidates(self):
        for rel, cat in (("Samples/Drums - One Shots/Snares/a.wav", "snare"),
                         ("Samples/FX & Atmo/Impacts/a.wav", "impact"),
                         ("Samples/Loops/Beat Loops/a.wav", "drums"),
                         ("Samples/Unsorted/a.wav", "unknown")):
            self.assertIsNone(selection_rules([rel], cat), rel)

    def test_folders_for_loops_basslines_rumble(self):
        for rel in ("Samples/Loops/Kick Loops/a.wav", "Samples/Loops/Bass Loops/a.wav",
                    "Samples/Synths/Basslines/a.wav", "Samples/Drums - One Shots/Rumble/a.wav"):
            self.assertIsNotNone(selection_rules([rel], "unknown"), rel)

    def test_any_present_path_counts(self):
        sel = selection_rules(["Samples/Unsorted/a.wav", f"{KICKS}/a.wav"], "unknown")
        self.assertEqual(sel["rules"], ["folder"])


# ----------------------------------------------------------------------------- real ffmpeg


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not installed")
class RealDecodeTests(Base):
    def test_sine_through_real_ffmpeg_int16_int24_float(self):
        self.put(f"{KICKS}/s16.wav", write_pcm16, sine(60))
        self.put(f"{KICKS}/s24.wav", write_pcm24, sine(60))
        self.put(f"{KICKS}/f32.wav", write_f32, sine(60))
        self.prepare()
        rep = self.stage().run()
        self.assertEqual((rep.status, rep.ok, rep.decoded, rep.valid1, rep.valid0), ("complete", 3, 3, 3, 0))
        for name in ("s16", "s24", "f32"):
            r = self.res(f"{KICKS}/{name}.wav")
            self.assertEqual((r["valid"], r["valid_scope"], r["lowend_status"]), (1, lowend.VALID_SCOPE, "ok"), name)
            self.assertAlmostEqual(r["f0_hz"], 60.0, delta=2.0, msg=name)
            self.assertAlmostEqual(r["peak"], 0.5, delta=1e-3, msg=name)
            self.assertAlmostEqual(r["rms"], 0.5 / np.sqrt(2), delta=1e-3, msg=name)
            self.assertEqual(r["decoded_frames"], int(SR * 0.8), name)
            self.assertEqual(r["clipped_samples"], 0)

    def test_stereo_antiphase_and_mono_through_real_decode(self):
        s = sine(60)
        self.put(f"{KICKS}/same.wav", write_pcm16, np.stack([s, s], axis=1))
        self.put(f"{KICKS}/anti.wav", write_pcm16, np.stack([s, -s], axis=1))
        self.put(f"{KICKS}/mono.wav", write_pcm16, s)
        self.prepare()
        self.stage().run()
        self.assertAlmostEqual(self.res(f"{KICKS}/same.wav")["stereo_corr_low"], 1.0, delta=1e-3)
        self.assertAlmostEqual(self.res(f"{KICKS}/anti.wav")["stereo_corr_low"], -1.0, delta=1e-3)
        self.assertIsNone(self.res(f"{KICKS}/mono.wav")["stereo_corr_low"])
        desc = json.loads(self.res(f"{KICKS}/mono.wav")["lowend_json"])
        self.assertEqual(desc["analysis"]["stereo_mode"], "mono")

    def test_noise_has_no_f0_and_silence_has_no_f0_or_correlation(self):
        rng = np.random.default_rng(1)
        self.put(f"{KICKS}/noise.wav", write_pcm16, rng.uniform(-0.5, 0.5, SR))
        self.put(f"{KICKS}/silence.wav", write_pcm16, np.zeros((SR, 2)))
        self.prepare()
        self.stage().run()
        noise = self.res(f"{KICKS}/noise.wav")
        self.assertIsNone(noise["f0_hz"])
        self.assertIsNotNone(json.loads(noise["lowend_json"])["analysis"]["f0_reason"])
        sil = self.res(f"{KICKS}/silence.wav")
        self.assertEqual((sil["valid"], sil["lowend_status"]), (1, "ok"))
        self.assertIsNone(sil["f0_hz"])
        self.assertIsNone(sil["stereo_corr_low"])  # undefined, not 1

    def test_descriptor_carries_provenance_and_selection(self):
        self.put(f"{KICKS}/k.wav", write_pcm16, sine(50))
        self.prepare()
        self.stage().run()
        r = self.res(f"{KICKS}/k.wav")
        desc = json.loads(r["lowend_json"])
        self.assertEqual(desc["stage_version"], lowend.LOWEND_VERSION)
        self.assertEqual(desc["settings"]["dsp_version"], lowend.lowend_dsp.VERSION)
        self.assertIn("settings", desc["analysis"])
        self.assertEqual(desc["analysis"]["provenance"]["module"], "sample_indexer.lowend_dsp")
        sel = json.loads(r["lowend_select"])
        self.assertEqual(sel["rules"], ["category", "folder"])
        # raw L3-3 hints are preserved next to the DSP result
        self.assertEqual(r["category"], "kick")
        self.assertEqual(r["label_basis"], "path_name_provisional")
        self.assertIsNotNone(r["label_evidence"])

    def test_truncated_file_is_conclusively_invalid(self):
        good = self.put(f"{KICKS}/good.wav", write_pcm16, sine(60))
        raw = good.read_bytes()
        self.put_bytes(f"{KICKS}/trunc.wav", raw[: 44 + (len(raw) - 44) // 2])
        self.prepare()
        rep = self.stage().run()
        r = self.res(f"{KICKS}/trunc.wav")
        self.assertEqual((r["valid"], r["lowend_status"]), (0, "decode_failed"))
        self.assertTrue(r["valid_error"])
        self.assertIsNone(r["peak"])  # no measurements from a failed decode
        self.assertEqual(rep.valid0, 1)
        self.assertEqual(self.res(f"{KICKS}/good.wav")["valid"], 1)

    def test_header_only_file_is_not_marked_invalid_or_valid(self):
        good = self.put(f"{KICKS}/good.wav", write_pcm16, sine(60))
        self.put_bytes(f"{KICKS}/hdr.wav", good.read_bytes()[:44])
        self.prepare()
        self.stage().run()
        r = self.res(f"{KICKS}/hdr.wav")
        self.assertIsNone(r["valid"])  # zero frames decoded: inconclusive, not valid
        self.assertEqual(r["lowend_status"], "empty_audio")

    def test_sources_are_never_modified(self):
        p = self.put(f"{KICKS}/k.wav", write_pcm16, sine(60))
        before = (sha(p), p.stat().st_mtime_ns, p.stat().st_size, stat.S_IMODE(p.stat().st_mode))
        self.prepare()
        self.stage().run()
        after = (sha(p), p.stat().st_mtime_ns, p.stat().st_size, stat.S_IMODE(p.stat().st_mode))
        self.assertEqual(before, after)

    def test_l33_columns_untouched(self):
        self.put(f"{KICKS}/k.wav", write_pcm16, sine(60))
        self.prepare()
        keep = ("kind", "category", "category_source", "probe_status", "probe_scope", "duration_s", "sr",
                "channels", "bits", "label_evidence", "bpm_name", "key_name", "beats_hint")
        before = {k: self.res(f"{KICKS}/k.wav")[k] for k in keep}
        self.stage().run()
        after = {k: self.res(f"{KICKS}/k.wav")[k] for k in keep}
        self.assertEqual(before, after)
        r = self.res(f"{KICKS}/k.wav")
        for col in ("lufs", "true_peak", "bpm_est", "key_est", "beats"):
            self.assertIsNone(r[col], col)  # later stages: never guessed here

    def test_only_candidates_are_processed(self):
        self.put(f"{KICKS}/k.wav", write_pcm16, sine(60))
        self.put("Samples/Drums - One Shots/Snares/s.wav", write_pcm16, sine(200))
        self.put("Samples/Unsorted/u.wav", write_pcm16, sine(61))  # distinct content (identity = sha256)
        self.prepare()
        rep = self.stage().run()
        self.assertEqual((rep.candidates, rep.selected, rep.decoded), (1, 1, 1))
        self.assertIsNone(self.res("Samples/Drums - One Shots/Snares/s.wav")["lowend_status"])
        self.assertIsNone(self.res("Samples/Unsorted/u.wav")["valid"])

    def test_ambiguous_label_in_kick_folder_is_still_analysed(self):
        self.put(f"{KICKS}/Sub no kick.wav", write_pcm16, sine(60))
        self.prepare()
        self.assertEqual(self.res(f"{KICKS}/Sub no kick.wav")["category"], "unknown")
        rep = self.stage().run()
        self.assertEqual(rep.ok, 1)
        self.assertEqual(json.loads(self.res(f"{KICKS}/Sub no kick.wav")["lowend_select"])["rules"], ["folder"])

    def test_rerun_decodes_nothing_and_force_redoes(self):
        self.put(f"{KICKS}/a.wav", write_pcm16, sine(60))
        self.put(f"{KICKS}/b.wav", write_pcm16, sine(50))
        self.prepare()
        self.stage().run()
        rep = self.stage().run()
        self.assertEqual((rep.selected, rep.decoded, rep.already_done, rep.pending_after), (0, 0, 2, 0))
        rep = self.stage(force=True).run()
        self.assertEqual((rep.selected, rep.decoded), (2, 2))

    def test_version_bump_and_setting_change_reselect(self):
        self.put(f"{KICKS}/a.wav", write_pcm16, sine(60))
        self.prepare()
        self.stage().run()
        self.conn.execute("UPDATE resource_stage SET version=0 WHERE stage='lowend'")
        self.assertEqual(self.stage().run().decoded, 1)
        # a different cap changes the settings signature: results at other settings are stale
        self.assertEqual(self.stage(max_duration_s=60).run().decoded, 1)

    def test_limit_then_resume_and_checkpoints(self):
        for i in range(5):
            self.put(f"{KICKS}/k{i}.wav", write_pcm16, sine(50 + i))
        self.prepare()
        rep = self.stage(limit=2, batch_size=1).run()
        self.assertEqual((rep.decoded, rep.pending_after), (2, 3))
        rep = self.stage(batch_size=2).run()
        self.assertEqual((rep.decoded, rep.pending_after, rep.status), (3, 0, "complete"))
        runs = self.conn.execute("SELECT status, decoded FROM lowend_run ORDER BY id").fetchall()
        self.assertEqual([tuple(r) for r in runs], [("complete", 2), ("complete", 3)])

    def test_max_seconds_stops_cleanly_and_resumes(self):
        for i in range(4):
            self.put(f"{KICKS}/k{i}.wav", write_pcm16, sine(50 + i))
        self.prepare()
        rep = self.stage(max_seconds=0.0).run()
        self.assertEqual((rep.status, rep.decoded), ("interrupted", 0))
        self.assertEqual(self.stage().run().decoded, 4)

    def test_stage_error_is_recorded_and_retry_errors_selects_it(self):
        good = self.put(f"{KICKS}/good.wav", write_pcm16, sine(60))
        raw = good.read_bytes()
        self.put_bytes(f"{KICKS}/trunc.wav", raw[: 44 + (len(raw) - 44) // 2])
        self.prepare()
        self.stage().run()
        self.assertEqual(self.stage().run().decoded, 0)  # recorded error counts as processed
        rep = self.stage(retry_errors=True).run()
        self.assertEqual((rep.selected, rep.decoded), (1, 1))

    def test_duration_cap_is_unresolved_not_truncated(self):
        self.put(f"{KICKS}/long.wav", write_pcm16, sine(60, seconds=2.0))
        self.prepare()
        rep = self.stage(max_duration_s=1.0).run()
        r = self.res(f"{KICKS}/long.wav")
        self.assertEqual((r["lowend_status"], r["valid"], rep.decoded), ("unresolved_cap", None, 0))
        self.assertIsNone(r["peak"])
        self.assertIn("not decoded", r["lowend_error"])

    def test_decoded_size_cap_aborts_without_truncating(self):
        # header claims nothing odd, but the pipe cap (half of the float64 footprint) is tiny
        self.put(f"{KICKS}/k.wav", write_pcm16, sine(60, seconds=1.0))
        self.prepare()
        # bypass the pre-check by lying about the cap at decode time only
        res = lowend.ffmpeg_decode(str(self.root / KICKS / "k.wav"), 1, max_bytes=4096)
        self.assertEqual(res.status, "unresolved_cap")
        self.assertIsNone(res.samples)

    def test_missing_and_stale_files(self):
        p = self.put(f"{KICKS}/gone.wav", write_pcm16, sine(60))
        q = self.put(f"{KICKS}/stale.wav", write_pcm16, sine(60, seconds=0.9))
        self.prepare()
        p.unlink()
        os.utime(q, ns=(1, 1))
        rep = self.stage().run()
        self.assertEqual(self.res(f"{KICKS}/gone.wav")["lowend_status"], "unreadable")
        self.assertEqual(self.res(f"{KICKS}/stale.wav")["lowend_status"], "stale")
        self.assertEqual(rep.decoded, 0)
        for rel in ("gone.wav", "stale.wav"):
            self.assertIsNone(self.res(f"{KICKS}/{rel}")["valid"])  # inconclusive never writes valid

    def test_missing_ffmpeg_binary_raises(self):
        self.put(f"{KICKS}/k.wav", write_pcm16, sine(60))
        self.prepare()
        with self.assertRaises(FileNotFoundError):
            self.stage(ffmpeg_bin="ffmpeg-does-not-exist").run()

    def test_cli_lowend_and_exit_codes(self):
        self.put(f"{KICKS}/k.wav", write_pcm16, sine(60))
        self.prepare()
        self.conn.close()
        out = Path(self.tmp / "out.txt")
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["--db", str(self.tmp / "index.sqlite"), "lowend", "--workers", "1"])
        self.assertEqual(code, 0)
        self.assertIn("lowend complete", buf.getvalue())
        self.conn = db.open_db(self.tmp / "index.sqlite", create=False)
        self.addCleanup(self.conn.close)
        self.assertEqual(self.res(f"{KICKS}/k.wav")["valid"], 1)


# ----------------------------------------------------------------------------- fake ffmpeg


def make_fake(path, body):
    Path(path).write_text("#!/bin/sh\n" + body)
    os.chmod(path, 0o755)


class FakeFfmpegTests(Base):
    def test_timeout_is_inconclusive_not_invalid(self):
        self.put(f"{KICKS}/k.wav", write_pcm16, sine(60))
        self.prepare()
        fake = self.tmp / "ffmpeg-sleep"
        make_fake(fake, "exec sleep 30\n")  # exec: ffmpeg is a single process; no orphan holds the pipe
        rep = self.stage(ffmpeg_bin=str(fake), timeout=0.3).run()
        r = self.res(f"{KICKS}/k.wav")
        self.assertEqual((r["lowend_status"], r["valid"], r["valid_error"]), ("timeout", None, None))
        self.assertEqual(rep.valid0, 0)

    def test_tool_crash_is_inconclusive(self):
        self.put(f"{KICKS}/k.wav", write_pcm16, sine(60))
        self.prepare()
        fake = self.tmp / "ffmpeg-crash"
        make_fake(fake, "echo 'segfault-ish unknown problem' >&2\nexit 7\n")
        self.stage(ffmpeg_bin=str(fake)).run()
        r = self.res(f"{KICKS}/k.wav")
        self.assertEqual((r["lowend_status"], r["valid"]), ("tool_error", None))

    def test_argv_is_single_threaded_xerror_file_protocol(self):
        self.put(f"{KICKS}/-i weird $(x).wav", write_pcm16, sine(60))
        self.prepare()
        log = self.tmp / "argv.txt"
        make_fake(self.tmp / "ffmpeg-log", f'printf "%s\\n" "$@" > "{log}"\nexit 7\n')
        self.stage(ffmpeg_bin=str(self.tmp / "ffmpeg-log")).run()
        argv = log.read_text().splitlines()
        self.assertIn("-xerror", argv)
        self.assertEqual(argv.count("-threads"), 2)
        self.assertTrue(all(argv[i + 1] == "1" for i, a in enumerate(argv) if a == "-threads"))
        self.assertEqual(argv[argv.index("-protocol_whitelist") + 1], "file")
        self.assertTrue(argv[argv.index("-i") + 1].startswith("file:"))

    def test_default_workers_two_and_bound(self):
        self.assertEqual(lowend.DEFAULT_LOWEND_WORKERS, 2)
        with self.assertRaises(ValueError):
            LowendStage(self.conn, workers=0)


class ClassifyTests(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(lowend.classify_failure(183, "corrupt input packet in stream 0"), "decode_failed")
        self.assertEqual(lowend.classify_failure(1, "foo: No such file or directory"), "unreadable")
        self.assertEqual(lowend.classify_failure(-9, "whatever"), "tool_error")
        self.assertEqual(lowend.classify_failure(1, "unknown thing"), "tool_error")


class MigrationV3Tests(unittest.TestCase):
    def test_upgrade_from_v2_backs_up_and_adds_columns(self):
        from sample_indexer.migrations import MIGRATIONS
        import sqlite3
        tmp = Path(tempfile.mkdtemp(prefix="sample-indexer-mig3-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        path = tmp / "old.sqlite"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        for v, sql in MIGRATIONS[:2]:
            conn.executescript(sql)
            conn.execute("INSERT INTO schema_version VALUES (?, 'x')", (v,))
        conn.execute("INSERT INTO resource(sha256,size,ext,first_seen) VALUES (?,?,?,?)", ("a" * 64, 3, ".wav", "t"))
        conn.commit()
        conn.close()
        new = db.open_db(path, create=True)
        self.addCleanup(new.close)
        self.assertEqual(db.schema_version(new), db.LATEST_VERSION)
        backups = list(tmp.glob("old.sqlite.bak-v2-*"))
        self.assertEqual(len(backups), 1)
        b = sqlite3.connect(backups[0])
        self.assertEqual(b.execute("SELECT MAX(version) FROM schema_version").fetchone()[0], 2)
        self.assertNotIn("lowend_status", {r[1] for r in b.execute("PRAGMA table_info(resource)")})
        b.close()
        row = new.execute("SELECT valid, lowend_status, peak FROM resource").fetchone()
        self.assertEqual(tuple(row), (None, None, None))


if __name__ == "__main__":
    unittest.main()
