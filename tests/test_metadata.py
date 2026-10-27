import contextlib
import io
import json
import os
import shutil
import sqlite3
import stat
import struct
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path

from sample_indexer import cli, db, labels, metadata, reports
from sample_indexer.metadata import MetadataStage, ProbeResult, parse_ffprobe_json
from sample_indexer.migrations import MIGRATION_1
from sample_indexer.scanner import Scanner

HAVE_FFPROBE = shutil.which("ffprobe") is not None


# ----------------------------------------------------------------------------- fixtures


def make_wav(path, *, sr=44100, channels=1, sampwidth=2, seconds=0.25):
    """Integer PCM WAV written with the stdlib `wave` module."""
    frames = int(sr * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sr)
        w.writeframes(b"\x00" * (frames * channels * sampwidth))


def make_float_wav(path, *, sr=48000, channels=1, seconds=0.1):
    """IEEE-float32 WAV (format tag 3); the stdlib `wave` writer cannot emit it."""
    frames = int(sr * seconds)
    data = struct.pack("<%df" % (frames * channels), *([0.0] * (frames * channels)))
    fmt = struct.pack("<HHIIHH", 3, channels, sr, sr * channels * 4, channels * 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(data)) + data
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def _ieee_extended(rate):
    exp = 16383 + rate.bit_length() - 1
    mant = rate << (64 - rate.bit_length())
    return struct.pack(">HQ", exp, mant)


def make_aiff(path, *, sr=44100, channels=2, bits=16, seconds=0.1):
    frames = int(sr * seconds)
    sound = b"\x00" * (frames * channels * (bits // 8))
    comm = struct.pack(">hIh", channels, frames, bits) + _ieee_extended(sr)
    ssnd = struct.pack(">II", 0, 0) + sound
    chunks = b"COMM" + struct.pack(">I", len(comm)) + comm + b"SSND" + struct.pack(">I", len(ssnd)) + ssnd
    Path(path).write_bytes(b"FORM" + struct.pack(">I", len(chunks) + 4) + b"AIFF" + chunks)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sample-indexer-meta-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "lib"
        self.root.mkdir()
        self.conn = db.open_db(self.tmp / "index.sqlite", create=True)
        self.addCleanup(self.conn.close)
        self._t = 1_700_000_000

    def put(self, rel, writer=None, data=None, **kw):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if writer:
            writer(p, **kw)
        else:
            p.write_bytes(data if data is not None else b"")
        self._t += 10
        os.utime(p, ns=(self._t * 10**9,) * 2)
        return p

    def scan(self):
        return Scanner(self.conn, self.root, workers=2).run()

    def stage(self, **kw):
        kw.setdefault("workers", 2)
        return MetadataStage(self.conn, **kw)

    def res(self, rel):
        return self.conn.execute(
            "SELECT r.* FROM resource r JOIN path p ON p.resource_id=r.id WHERE p.rel_path=?", (rel,)
        ).fetchone()

    def stage_row(self, rel, stage):
        return self.conn.execute(
            "SELECT s.* FROM resource_stage s JOIN path p ON p.resource_id=s.resource_id"
            " WHERE p.rel_path=? AND s.stage=?", (rel, stage),
        ).fetchone()


# ----------------------------------------------------------------------------- real ffprobe


@unittest.skipUnless(HAVE_FFPROBE, "ffprobe not installed")
class RealFfprobeTests(Base):
    def test_integer_and_float_fixtures(self):
        self.put("Samples/Drums - One Shots/Kicks/k16.wav", make_wav, sr=44100, channels=1, sampwidth=2, seconds=0.5)
        self.put("Samples/Drums - One Shots/Kicks/k24.wav", make_wav, sr=48000, channels=2, sampwidth=3, seconds=0.25)
        self.put("Samples/Drums - One Shots/Kicks/k32i.wav", make_wav, sr=96000, channels=2, sampwidth=4, seconds=0.1)
        self.put("Samples/Drums - One Shots/Kicks/f32.wav", make_float_wav, sr=48000, channels=1, seconds=0.1)
        self.put("Samples/Drums - One Shots/Kicks/a16.aif", make_aiff, sr=44100, channels=2, bits=16)
        self.scan()
        rep = self.stage().run()
        self.assertEqual((rep.status, rep.probe_ok, rep.probed), ("complete", 5, 5))
        r = self.res("Samples/Drums - One Shots/Kicks/k16.wav")
        self.assertEqual((r["sr"], r["channels"], r["bits"], r["codec_name"], r["sample_type"]),
                         (44100, 1, 16, "pcm_s16le", "int"))
        self.assertAlmostEqual(r["duration_s"], 0.5, places=3)
        self.assertEqual((r["probe_status"], r["probe_scope"]), ("ok", "header_container"))
        r = self.res("Samples/Drums - One Shots/Kicks/k24.wav")
        self.assertEqual((r["sr"], r["channels"], r["bits"], r["bits_source"], r["sample_type"]),
                         (48000, 2, 24, "bits_per_raw_sample", "int"))
        r = self.res("Samples/Drums - One Shots/Kicks/k32i.wav")
        self.assertEqual((r["sr"], r["bits"], r["codec_name"], r["sample_type"]), (96000, 32, "pcm_s32le", "int"))
        # float32 is explicit, not an integer depth
        r = self.res("Samples/Drums - One Shots/Kicks/f32.wav")
        self.assertEqual((r["codec_name"], r["sample_fmt"], r["sample_type"], r["bits"], r["bits_source"]),
                         ("pcm_f32le", "flt", "float", 32, "bits_per_sample"))
        r = self.res("Samples/Drums - One Shots/Kicks/a16.aif")
        self.assertEqual((r["container"], r["sr"], r["bits"], r["channels"]), ("aiff", 44100, 16, 2))

    def test_valid_is_never_set_by_the_probe(self):
        self.put("a.wav", make_wav)
        self.put("bad.wav", data=b"RIFFjunkjunkjunk")
        self.scan()
        self.stage().run()
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM resource WHERE valid IS NOT NULL OR valid_error IS NOT NULL").fetchone()[0], 0
        )

    def test_empty_corrupt_and_truncated_are_reported_not_fatal(self):
        self.put("empty.wav", data=b"")
        self.put("junk.wav", data=b"this is not audio at all" * 8)
        good = self.put("good.wav", make_wav, seconds=1.0)
        trunc = self.root / "trunc.wav"
        trunc.write_bytes(good.read_bytes()[:1000])
        os.utime(trunc, ns=((self._t + 5) * 10**9,) * 2)
        self.scan()
        rep = self.stage().run()
        self.assertEqual(rep.status, "complete")
        self.assertEqual(rep.probe_ok, 2)  # good + truncated (header parses; no decode claim)
        self.assertEqual(dict(rep.probe_errors), {"empty": 1, "malformed": 1})
        empty, junk = self.res("empty.wav"), self.res("junk.wav")
        self.assertEqual(empty["probe_status"], "empty")
        self.assertEqual(junk["probe_status"], "malformed")
        self.assertIsNone(junk["sr"])  # unknown stays NULL, not 0 / invented
        self.assertIsNone(junk["duration_s"])
        self.assertIsNone(junk["valid"])  # unknown != invalid
        self.assertEqual(rep.probed, 3)  # the empty file spawned no subprocess
        # the truncated header still "probes ok" and only the header-claimed duration shrinks
        self.assertEqual(self.res("trunc.wav")["probe_status"], "ok")
        self.assertIsNone(self.res("trunc.wav")["valid"])
        self.assertIn("malformed", self.stage_row("junk.wav", "probe")["error"])

    def test_leading_dash_filename_is_not_an_option(self):
        self.put("-i.wav", make_wav, seconds=0.2)
        self.put("sp ace & 'q' $(x).wav", make_wav, seconds=0.3)  # different bytes -> own resource
        self.scan()
        rep = self.stage().run()
        self.assertEqual((rep.probe_ok, sum(rep.probe_errors.values())), (2, 0))

    def test_aif_extension_but_wav_content_still_probes_what_it_is(self):
        self.put("x.aif", make_wav)
        self.scan()
        self.stage().run()
        self.assertEqual(self.res("x.aif")["container"], "wav")

    def test_cli_metadata_coverage_and_second_run_probes_nothing(self):
        self.put("Samples/Loops/Beat Loops/Beat 128BPM.wav", make_wav, seconds=7.5, sr=8000)
        self.put("MIDI/Basslines/a.mid", data=b"MThd")
        self.scan()
        dbp = str(self.tmp / "index.sqlite")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--db", dbp, "metadata", "--workers", "2"]), 0)
        self.assertIn("ffprobe subprocesses           1", out.getvalue())
        self.assertIn("midi skipped (not probed)      1", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--db", dbp, "metadata", "--workers", "2"]), 0)
        self.assertIn("ffprobe subprocesses           0", out.getvalue())
        self.assertIn("selected this run              0", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["--db", dbp, "coverage"]), 0)
        text = out.getvalue()
        self.assertIn("probe stage v1: processed 1/1 audio (errors 0); pending 0", text)
        self.assertIn("valid set (full decode): 0", text)
        self.assertIn("unvalidated", text)
        r = self.res("Samples/Loops/Beat Loops/Beat 128BPM.wav")
        self.assertEqual((r["bpm_name"], r["bpm_check"], r["beats_hint"], r["kind"]), (128.0, "whole_bars", 16.0, "loop"))
        self.assertIsNone(r["bpm_est"])
        self.assertIsNone(r["beats"])  # `beats` stays reserved for measured values


# ----------------------------------------------------------------------------- stage behaviour (fake probe)


def ok_probe(path):
    return ProbeResult("ok", duration_s=1.0, sr=44100, channels=2, bits=16, bits_source="bits_per_sample",
                       container="wav", codec_name="pcm_s16le", sample_fmt="s16", sample_type="int")


class StageTests(Base):
    def populate(self, n=10, midi=0):
        for i in range(n):
            self.put(f"Samples/Drums - One Shots/Kicks/k{i:02d}.wav", data=f"audio-{i}".encode())
        for i in range(midi):
            self.put(f"MIDI/Basslines/m{i}.mid", data=f"midi-{i}".encode())
        self.scan()

    def test_midi_is_never_probed_nor_marked_valid(self):
        self.populate(2, midi=3)
        seen = []
        rep = self.stage(probe_func=lambda p: (seen.append(p), ok_probe(p))[1]).run()
        self.assertEqual(len(seen), 2)
        self.assertFalse(any(p.endswith(".mid") for p in seen))
        self.assertEqual((rep.midi_skipped, rep.probed, rep.labelled), (3, 2, 5))
        for r in self.conn.execute("SELECT * FROM resource WHERE ext='.mid'"):
            self.assertEqual((r["kind"], r["kind_source"]), ("midi", "ext"))
            self.assertIsNone(r["valid"])
            self.assertIsNone(r["probe_status"])
            self.assertIsNone(r["duration_s"])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM resource_stage WHERE stage='probe'").fetchone()[0], 2
        )

    def test_idempotent_second_run_does_nothing(self):
        self.populate(6, midi=2)
        calls = []
        probe = lambda p: (calls.append(p), ok_probe(p))[1]
        self.stage(probe_func=probe).run()
        before = [tuple(r) for r in self.conn.execute("SELECT resource_id, stage, version, done_at FROM resource_stage ORDER BY 1,2")]
        n = len(calls)
        rep = self.stage(probe_func=probe).run()
        self.assertEqual((len(calls), rep.selected, rep.probed, rep.labelled, rep.pending_after), (n, 0, 0, 0, 0))
        after = [tuple(r) for r in self.conn.execute("SELECT resource_id, stage, version, done_at FROM resource_stage ORDER BY 1,2")]
        self.assertEqual(before, after)

    def test_force_redoes_everything(self):
        self.populate(4)
        calls = []
        probe = lambda p: (calls.append(p), ok_probe(p))[1]
        self.stage(probe_func=probe).run()
        self.stage(probe_func=probe, force=True).run()
        self.assertEqual(len(calls), 8)

    def test_version_bump_reselects(self):
        self.populate(3)
        self.stage(probe_func=ok_probe).run()
        self.conn.execute("UPDATE resource_stage SET version=0 WHERE stage='probe'")
        calls = []
        self.stage(probe_func=lambda p: (calls.append(p), ok_probe(p))[1]).run()
        self.assertEqual(len(calls), 3)

    def test_partial_limit_then_resume(self):
        self.populate(12)
        calls = []
        probe = lambda p: (calls.append(p), ok_probe(p))[1]
        rep = self.stage(probe_func=probe, limit=5).run()
        self.assertEqual((rep.selected, rep.probed, rep.pending_after), (5, 5, 7))
        rep = self.stage(probe_func=probe).run()
        self.assertEqual((rep.selected, rep.probed, rep.pending_after), (7, 7, 0))
        self.assertEqual(len(calls), 12)
        self.assertEqual(len(set(calls)), 12)  # nothing analysed twice

    def test_interrupt_checkpoints_and_resumes(self):
        self.populate(30)
        done = []

        def interrupt(rep):
            done.append(1)
            if len(done) == 12:
                raise KeyboardInterrupt

        rep = self.stage(probe_func=ok_probe, batch_size=5, workers=1, on_result=interrupt).run()
        self.assertEqual(rep.status, "interrupted")
        n = self.conn.execute("SELECT COUNT(*) FROM resource_stage WHERE stage='probe'").fetchone()[0]
        self.assertEqual(n, 12)  # everything finished before the Ctrl-C was committed
        calls = []
        rep = self.stage(probe_func=lambda p: (calls.append(p), ok_probe(p))[1]).run()
        self.assertEqual((rep.status, len(calls), rep.pending_after), ("complete", 18, 0))
        statuses = [r[0] for r in self.conn.execute("SELECT status FROM metadata_run ORDER BY id")]
        self.assertEqual(statuses, ["interrupted", "complete"])

    def test_max_seconds_stops_cleanly(self):
        self.populate(20)

        def slow(path):
            time.sleep(0.05)
            return ok_probe(path)

        rep = self.stage(probe_func=slow, workers=1, max_seconds=0.2).run()
        self.assertEqual(rep.status, "interrupted")
        self.assertIn("max-seconds", rep.stop_reason)
        self.assertTrue(0 < rep.probed < 20)
        self.assertGreater(rep.pending_after, 0)

    def test_concurrency_is_bounded_and_single_writer(self):
        self.populate(24)
        lock = threading.Lock()
        live = peak = 0
        threads = set()

        def probe(path):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
                threads.add(threading.get_ident())
            time.sleep(0.02)
            with lock:
                live -= 1
            return ok_probe(path)

        self.stage(probe_func=probe, workers=3).run()
        self.assertLessEqual(peak, 3)
        self.assertGreater(peak, 1)
        self.assertNotIn(threading.get_ident(), threads)  # probes ran off the main (DB) thread

    def test_errors_are_recorded_and_rerunnable_independently(self):
        self.populate(4)
        bad = {"k01.wav", "k03.wav"}

        def flaky(path):
            if os.path.basename(path) in bad:
                return ProbeResult("malformed", "boom")
            return ok_probe(path)

        rep = self.stage(probe_func=flaky).run()
        self.assertEqual((rep.probe_ok, dict(rep.probe_errors)), (2, {"malformed": 2}))
        row = self.stage_row("Samples/Drums - One Shots/Kicks/k01.wav", "probe")
        self.assertEqual(row["version"], metadata.PROBE_VERSION)
        self.assertEqual(row["error"], "malformed: boom")
        # no retry storm: a plain rerun leaves recorded errors alone
        calls = []
        counting = lambda p: (calls.append(p), ok_probe(p))[1]
        self.assertEqual(self.stage(probe_func=counting).run().selected, 0)
        self.assertEqual(calls, [])
        # explicit retry touches only the errored resources
        rep = self.stage(probe_func=counting, retry_errors=True).run()
        self.assertEqual((len(calls), rep.probe_ok), (2, 2))
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM resource_stage WHERE stage='probe' AND error IS NOT NULL").fetchone()[0], 0
        )

    def test_transient_failure_keeps_earlier_conclusive_facts(self):
        self.populate(1)
        self.stage(probe_func=ok_probe).run()
        rid = self.conn.execute("SELECT id FROM resource").fetchone()[0]
        self.stage(probe_func=lambda p: ProbeResult("timeout", "slow"), force=True).run()
        r = self.conn.execute("SELECT * FROM resource WHERE id=?", (rid,)).fetchone()
        self.assertEqual((r["probe_status"], r["sr"], r["duration_s"]), ("timeout", 44100, 1.0))

    def test_resource_with_two_paths_is_probed_once(self):
        self.put("a/one.wav", data=b"same bytes")
        self.put("b/two.wav", data=b"same bytes")
        self.scan()
        calls = []
        self.stage(probe_func=lambda p: (calls.append(p), ok_probe(p))[1]).run()
        self.assertEqual(len(calls), 1)

    def test_missing_paths_are_never_probed(self):
        gone = self.put("gone.wav", data=b"gone")
        self.put("stay.wav", data=b"stay")
        self.scan()
        gone.unlink()
        self.scan()
        calls = []
        rep = self.stage(probe_func=lambda p: (calls.append(os.path.basename(p)), ok_probe(p))[1]).run()
        self.assertEqual(calls, ["stay.wav"])
        self.assertEqual(rep.candidates, 1)

    def test_file_changed_after_scan_is_stale_not_probed(self):
        p = self.put("a.wav", data=b"one")
        self.scan()
        os.utime(p, ns=((self._t + 99) * 10**9,) * 2)
        calls = []
        rep = self.stage(probe_func=lambda q: (calls.append(q), ok_probe(q))[1]).run()
        self.assertEqual(calls, [])
        self.assertEqual(dict(rep.probe_errors), {"stale": 1})

    def test_stratified_selection_covers_folders(self):
        for folder, n in (("Kicks", 30), ("Snares", 30), ("Claps", 3)):
            for i in range(n):
                self.put(f"Samples/Drums - One Shots/{folder}/{i}.wav", data=f"{folder}{i}".encode())
        self.scan()
        rep = self.stage(probe_func=ok_probe, limit=9, stratify=True).run()
        self.assertEqual(rep.selected, 9)
        folders = [r[0] for r in self.conn.execute(
            "SELECT DISTINCT p.rel_path FROM path p JOIN resource_stage s ON s.resource_id=p.resource_id AND s.stage='probe'"
        )]
        by_folder = {f.split("/")[2] for f in folders}
        self.assertEqual(by_folder, {"Kicks", "Snares", "Claps"})

    def test_zero_limit_selects_nothing(self):
        self.populate(2)
        self.assertEqual(self.stage(probe_func=ok_probe, limit=0).run().selected, 0)

    def test_run_bookkeeping_row(self):
        self.populate(3)
        self.stage(probe_func=ok_probe).run()
        row = self.conn.execute("SELECT * FROM metadata_run").fetchone()
        self.assertEqual((row["status"], row["selected"], row["probe_ok"]), ("complete", 3, 3))
        self.assertIsNotNone(row["finished_at"])


# ----------------------------------------------------------------------------- labels in the stage


class LabelStageTests(Base):
    def test_rename_refreshes_path_labels_without_reprobing(self):
        p = self.put("Samples/Drums - One Shots/Kicks/x.wav", data=b"content")
        self.scan()
        calls = []
        probe = lambda q: (calls.append(q), ok_probe(q))[1]
        self.stage(probe_func=probe).run()
        self.assertEqual(self.res("Samples/Drums - One Shots/Kicks/x.wav")["category"], "kick")
        (self.root / "Samples/Drums - One Shots/Snares").mkdir(parents=True)
        os.rename(p, self.root / "Samples/Drums - One Shots/Snares/x 130BPM.wav")
        rep = self.scan()
        self.assertEqual((rep.moved, rep.hashed), (1, 1))
        rep = self.stage(probe_func=probe).run()
        self.assertEqual((rep.selected, rep.probed, rep.labelled), (1, 0, 1))  # labels only
        self.assertEqual(len(calls), 1)
        r = self.res("Samples/Drums - One Shots/Snares/x 130BPM.wav")
        self.assertEqual((r["category"], r["bpm_name"], r["bpm_name_raw"]), ("snare", 130.0, "130BPM"))
        self.assertIn("Snares", r["label_evidence"])
        self.assertNotIn("Kicks", r["label_evidence"])  # no stale evidence
        self.assertEqual(self.stage(probe_func=probe).run().selected, 0)

    def test_second_path_added_preserves_ambiguity(self):
        self.put("Samples/Drums - One Shots/Kicks/a.wav", data=b"dup")
        self.scan()
        self.stage(probe_func=ok_probe).run()
        self.put("Samples/Drums - One Shots/Snares/a.wav", data=b"dup")
        self.scan()
        rep = self.stage(probe_func=ok_probe).run()
        self.assertEqual(rep.labelled, 1)
        r = self.res("Samples/Drums - One Shots/Kicks/a.wav")
        self.assertEqual((r["category"], r["category_source"]), ("unknown", "path_conflict"))
        ev = json.loads(r["label_evidence"])
        self.assertEqual(len(ev["paths"]), 2)

    def test_does_not_overwrite_kind_owned_by_a_later_stage(self):
        self.put("Samples/Drums - One Shots/Kicks/a.wav", data=b"x")
        self.scan()
        self.stage(probe_func=ok_probe).run()
        self.conn.execute("UPDATE resource SET kind='loop', category='bass', label_basis='dsp_v1'")
        self.conn.execute("UPDATE resource_stage SET version=0 WHERE stage='labels'")
        self.stage(probe_func=ok_probe).run()
        r = self.res("Samples/Drums - One Shots/Kicks/a.wav")
        self.assertEqual((r["kind"], r["category"], r["label_basis"]), ("loop", "bass", "dsp_v1"))
        self.assertIsNotNone(r["label_evidence"])  # evidence still refreshed

    def test_future_dsp_columns_are_never_written(self):
        self.put("Samples/Loops/Beat Loops/Beat 128BPM Cmin.wav", data=b"x")
        self.scan()
        self.stage(probe_func=ok_probe).run()
        r = self.res("Samples/Loops/Beat Loops/Beat 128BPM Cmin.wav")
        for col in ("bpm_est", "key_est", "beats", "f0_hz", "lufs", "centroid", "valid", "valid_error", "tempo", "bars"):
            self.assertIsNone(r[col], col)

    def test_bpm_without_duration_has_no_beats_hint(self):
        self.put("Samples/Loops/Beat Loops/Beat 128BPM.wav", data=b"x")
        self.scan()
        self.stage(probe_func=lambda p: ProbeResult("malformed", "bad")).run()
        r = self.res("Samples/Loops/Beat Loops/Beat 128BPM.wav")
        self.assertEqual(r["bpm_name"], 128.0)
        self.assertIsNone(r["beats_hint"])
        self.assertIsNone(r["bpm_check"])


# ----------------------------------------------------------------------------- probe internals


class ParseTests(unittest.TestCase):
    def test_unknown_fields_stay_null(self):
        doc = {"streams": [{"codec_type": "audio", "codec_name": "adpcm_ms", "sample_rate": "N/A",
                            "channels": 0, "bits_per_sample": 0, "duration": "N/A"}],
               "format": {"format_name": "wav", "duration": "N/A"}}
        r = parse_ffprobe_json(json.dumps(doc))
        self.assertEqual(r.status, "ok")
        self.assertEqual((r.sr, r.channels, r.bits, r.duration_s, r.sample_type), (None,) * 5)
        self.assertEqual(r.codec_name, "adpcm_ms")  # compressed: type not guessed

    def test_precision_is_not_invented(self):
        doc = {"streams": [{"codec_type": "audio", "codec_name": "pcm_s24le", "sample_fmt": "s32",
                            "sample_rate": "44100", "channels": 2, "bits_per_sample": 24,
                            "bits_per_raw_sample": "24", "duration": "1.500000"}], "format": {"format_name": "wav"}}
        r = parse_ffprobe_json(json.dumps(doc))
        self.assertEqual((r.bits, r.bits_source, r.sample_fmt), (24, "bits_per_raw_sample", "s32"))  # not 32
        doc["streams"][0].pop("bits_per_raw_sample")
        doc["streams"][0]["bits_per_sample"] = 0
        self.assertIsNone(parse_ffprobe_json(json.dumps(doc)).bits)

    def test_duration_falls_back_to_container_field(self):
        doc = {"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}], "format": {"duration": "2.5"}}
        self.assertEqual(parse_ffprobe_json(json.dumps(doc)).duration_s, 2.5)

    def test_no_audio_stream_and_garbage(self):
        self.assertEqual(parse_ffprobe_json('{"streams": [], "format": {}}').status, "no_audio_stream")
        self.assertEqual(parse_ffprobe_json("not json").status, "tool_error")

    def test_sample_type_mapping(self):
        self.assertEqual(metadata.sample_type_of("pcm_f64le"), "float")
        self.assertEqual(metadata.sample_type_of("pcm_u8"), "int")
        self.assertIsNone(metadata.sample_type_of("flac"))
        self.assertIsNone(metadata.sample_type_of(None))


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sample-indexer-proc-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def script(self, name, body):
        p = self.tmp / name
        p.write_text("#!/bin/sh\n" + body + "\n")
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
        return str(p)

    def test_probe_timeout_kills_the_process(self):
        slow = self.script("slow-ffprobe", "exec sleep 30")
        target = self.tmp / "x.wav"
        target.write_bytes(b"x")
        t0 = time.monotonic()
        r = metadata.ffprobe_file(str(target), timeout=0.4, ffprobe_bin=slow)
        self.assertLess(time.monotonic() - t0, 5)
        self.assertEqual(r.status, "timeout")
        self.assertIn("0.4", r.error)

    def test_timeout_through_stage_is_recorded_not_retried(self):
        root = self.tmp / "lib"
        root.mkdir()
        (root / "a.wav").write_bytes(b"abc")
        conn = db.open_db(self.tmp / "i.sqlite", create=True)
        self.addCleanup(conn.close)
        Scanner(conn, root, workers=1).run()
        slow = self.script("slow2", "exec sleep 30")
        rep = MetadataStage(conn, workers=1, timeout=0.3, ffprobe_bin=slow).run()
        self.assertEqual(dict(rep.probe_errors), {"timeout": 1})
        row = conn.execute("SELECT * FROM resource_stage WHERE stage='probe'").fetchone()
        self.assertTrue(row["error"].startswith("timeout"))
        self.assertEqual(MetadataStage(conn, workers=1, timeout=0.3, ffprobe_bin=slow).run().selected, 0)

    def test_tool_failure_modes(self):
        target = self.tmp / "x.wav"
        target.write_bytes(b"x")
        r = metadata.ffprobe_file(str(target), ffprobe_bin=self.script("crash", "echo oops >&2; exit 139"))
        self.assertEqual(r.status, "tool_error")
        r = metadata.ffprobe_file(str(target), ffprobe_bin=self.script("garbage", "echo not-json"))
        self.assertEqual(r.status, "tool_error")
        r = metadata.ffprobe_file(str(target), ffprobe_bin=str(self.tmp / "does-not-exist"))
        self.assertEqual(r.status, "tool_error")

    def test_argv_is_a_list_and_path_is_prefixed(self):
        rec = self.tmp / "argv.txt"
        probe = self.script("rec", f'printf "%s\\n" "$@" > {rec}; echo \'{{"streams":[],"format":{{}}}}\'')
        target = self.tmp / "-weird $(touch pwned) name.wav"
        target.write_bytes(b"x")
        metadata.ffprobe_file(str(target), ffprobe_bin=probe)
        lines = rec.read_text().splitlines()
        self.assertEqual(lines[-1], "file:" + str(target))
        self.assertFalse((self.tmp / "pwned").exists())
        self.assertNotIn("-i", lines[:-1])

    def test_missing_ffprobe_binary_is_a_clear_error(self):
        conn = db.open_db(self.tmp / "i.sqlite", create=True)
        self.addCleanup(conn.close)
        with self.assertRaises(FileNotFoundError):
            MetadataStage(conn, ffprobe_bin="definitely-not-ffprobe-xyz").run()


# ----------------------------------------------------------------------------- migration / backup


class MigrationTests(unittest.TestCase):
    def test_upgrade_from_v1_backs_up_first_and_preserves_data(self):
        tmp = Path(tempfile.mkdtemp(prefix="sample-indexer-mig-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        path = tmp / "old.sqlite"
        conn = sqlite3.connect(path)
        conn.executescript(MIGRATION_1)
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        conn.execute("INSERT INTO schema_version VALUES (1, 'x')")
        conn.execute("INSERT INTO resource(sha256,size,ext,first_seen) VALUES (?,?,?,?)", ("a" * 64, 3, ".wav", "t"))
        conn.commit()
        conn.close()
        new = db.open_db(path, create=True)
        self.addCleanup(new.close)
        self.assertEqual(db.schema_version(new), db.LATEST_VERSION)
        backups = list(tmp.glob("old.sqlite.bak-v1-*"))
        self.assertEqual(len(backups), 1)
        b = sqlite3.connect(backups[0])
        self.assertEqual(b.execute("SELECT MAX(version) FROM schema_version").fetchone()[0], 1)
        self.assertEqual(b.execute("SELECT COUNT(*) FROM resource").fetchone()[0], 1)
        cols = {r[1] for r in b.execute("PRAGMA table_info(resource)")}
        self.assertNotIn("probe_status", cols)
        b.close()
        self.assertEqual(new.execute("SELECT COUNT(*), MAX(probe_status) FROM resource").fetchone()[0], 1)

    def test_fresh_database_makes_no_backup(self):
        tmp = Path(tempfile.mkdtemp(prefix="sample-indexer-mig-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        db.open_db(tmp / "n.sqlite", create=True).close()
        self.assertEqual(list(tmp.glob("*.bak-*")), [])

    def test_backup_command_and_refuses_overwrite(self):
        tmp = Path(tempfile.mkdtemp(prefix="sample-indexer-mig-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        conn = db.open_db(tmp / "n.sqlite", create=True)
        self.addCleanup(conn.close)
        dest = tmp / "copy.sqlite"
        db.backup_db(conn, dest)
        self.assertTrue(dest.exists())
        with self.assertRaises(FileExistsError):
            db.backup_db(conn, dest)

    def test_coverage_report_on_empty_db(self):
        tmp = Path(tempfile.mkdtemp(prefix="sample-indexer-mig-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        conn = db.open_db(tmp / "n.sqlite", create=True)
        self.addCleanup(conn.close)
        self.assertIn("audio 0, midi 0", reports.metadata_coverage(conn))


if __name__ == "__main__":
    unittest.main()
