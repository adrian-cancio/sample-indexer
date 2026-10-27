import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sample_indexer import cli, db, loops
from sample_indexer.loops import LoopsStage, selection_rules
from sample_indexer.lowend import DecodeResult
from sample_indexer.metadata import MetadataStage
from sample_indexer.scanner import Scanner
from tests.test_loops_dsp import SR, click_train, tones, write_wav16

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
LOOPS = "Samples/Loops/Beat Loops"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sample-indexer-loops-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "lib"
        self.root.mkdir()
        self.conn = db.open_db(self.tmp / "index.sqlite", create=True)
        self.addCleanup(self.conn.close)
        self._t = 1_700_000_000

    def put(self, rel, data, sr=SR):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        write_wav16(p, data, sr)
        self._t += 10
        os.utime(p, ns=(self._t * 10**9,) * 2)
        return p

    def prepare(self):
        Scanner(self.conn, self.root, workers=2).run()
        MetadataStage(self.conn, workers=2).run()

    def stage(self, **kw):
        kw.setdefault("workers", 2)
        return LoopsStage(self.conn, **kw)

    def res(self, rel):
        return self.conn.execute(
            "SELECT r.* FROM resource r JOIN path p ON p.resource_id=r.id WHERE p.rel_path=?", (rel,)).fetchone()


class SelectionTests(unittest.TestCase):
    def test_each_rule(self):
        self.assertEqual(selection_rules(["Samples/Loops/Beat Loops/a.wav"], "unknown", None)["rules"], ["folder"])
        self.assertEqual(selection_rules(["Samples/Unsorted/a.wav"], "loop", None)["rules"], ["kind"])
        self.assertEqual(selection_rules(["Samples/Unsorted/a.wav"], None, 128.0)["rules"], ["bpm_name"])
        self.assertEqual(selection_rules(["Samples/Unsorted/a.wav"], "loop", 128.0)["rules"], ["kind", "bpm_name"])

    def test_listed_folders(self):
        for rel in ("Samples/Loops/Kick Loops/a.wav", "Samples/Synths/Synth Loops/a.wav",
                    "Samples/Synths/Basslines/a.wav", "Samples/Synths/Sequences/a.wav",
                    "Samples/Synths/Acid Riffs/a.wav", "Samples/Synths/Hooks/a.wav",
                    "Samples/Synths/Abstract Loops/a.wav", "Samples/FX & Atmo/FX Loops/a.wav"):
            self.assertIsNotNone(selection_rules([rel], None, None), rel)

    def test_non_candidates(self):
        for rel in ("Samples/Drums - One Shots/Kicks/a.wav", "Samples/Synths/Synth Shots/a.wav",
                    "Samples/FX & Atmo/Impacts/a.wav", "Samples/Unsorted/a.wav"):
            self.assertIsNone(selection_rules([rel], "one_shot", None), rel)

    def test_stratified_order_round_robin_over_folder_and_bpm(self):
        mk = lambda rid, folder, bpm: loops.Item(rid, SR, 1, 1.0, "ok", bpm, None, {}, [("r", f"{folder}/{rid}.wav", 1, 1)])  # noqa: E731
        items = [mk(1, "A", 100.0), mk(2, "A", 100.0), mk(3, "A", 100.0), mk(4, "B", None)]
        order = [i.rid for i in loops.stratified_order(items, 0)]
        self.assertEqual(order[:2].count(4), 1)  # the lone stratum is served in the first round


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not installed")
class StageTests(Base):
    def test_unknown_label_is_selected_and_measured_with_hint_comparisons(self):
        self.put(f"{LOOPS}/clicks 120BPM Cmaj.wav", click_train(120, 8.0))
        self.put("Samples/Unsorted/mystery.wav", tones([60, 64, 67]))  # no loop label: not selected
        self.put("Samples/Synths/Hooks/pad Amin.wav", tones([57, 60, 64]))  # folder only, key hint
        self.prepare()
        rep = self.stage().run()
        self.assertEqual(rep.status, "complete")
        self.assertEqual((rep.candidates, rep.ok, rep.decoded), (2, 2, 2))
        r = self.res(f"{LOOPS}/clicks 120BPM Cmaj.wav")
        self.assertAlmostEqual(r["bpm_est"], 120.0, delta=0.6)
        self.assertEqual(r["bpm_hint_check"], "exact")
        self.assertEqual(r["length_check"], "whole_bars")
        self.assertAlmostEqual(r["beats"], 16.0, delta=0.1)
        self.assertAlmostEqual(r["bars_est"], 4.0, delta=0.03)
        self.assertEqual(r["valid"], 1)
        self.assertEqual(r["loops_status"], "ok")
        h = self.res("Samples/Synths/Hooks/pad Amin.wav")
        self.assertEqual(h["key_est"], "Amin")
        self.assertEqual(h["key_hint_check"], "same")
        self.assertIsNone(h["bpm_hint_check"])
        self.assertIsNone(self.res("Samples/Unsorted/mystery.wav")["loops_status"])
        sel = json.loads(r["loops_select"])
        self.assertEqual(sel["rules"], ["folder", "kind", "bpm_name"])
        js = json.loads(r["loops_json"])
        self.assertEqual(js["settings"]["dsp"]["bpm_min"], 60.0)
        self.assertEqual(len(js["analysis"]["chroma12"]), 12)
        self.assertEqual(js["analysis"]["tempo_hint_comparison"]["relation"], "exact")

    def test_label_columns_and_lowend_columns_untouched(self):
        self.put(f"{LOOPS}/a 128BPM.wav", click_train(128, 8.0))
        self.prepare()
        before = dict(self.res(f"{LOOPS}/a 128BPM.wav"))
        self.stage().run()
        after = dict(self.res(f"{LOOPS}/a 128BPM.wav"))
        changed = {k for k in before if before[k] != after[k]}
        self.assertEqual(changed & {"kind", "category", "bpm_name", "key_name", "label_evidence", "f0_hz", "peak",
                                    "sub_peak_hz", "lowend_status", "lowend_json", "duration_s", "sr"}, set())

    def test_rerun_is_free_force_and_setting_change_reselect(self):
        self.put(f"{LOOPS}/a 128BPM.wav", click_train(128, 8.0))
        self.prepare()
        self.assertEqual(self.stage().run().decoded, 1)
        rep = self.stage().run()
        self.assertEqual((rep.selected, rep.decoded, rep.already_done), (0, 0, 1))
        self.assertEqual(self.stage(force=True).run().decoded, 1)
        rep = self.stage(overrides={"bpm_max": 190.0}).run()  # settings signature changed
        self.assertEqual(rep.decoded, 1)
        self.assertEqual(json.loads(self.res(f"{LOOPS}/a 128BPM.wav")["loops_json"])["settings"]["dsp"]["bpm_max"], 190.0)
        with self.assertRaises(ValueError):
            self.stage(overrides={"nope": 1})

    def test_limit_then_resume_with_checkpoints_and_max_seconds(self):
        for i in range(5):
            self.put(f"{LOOPS}/l{i} 120BPM.wav", click_train(120, 4.0 + 0.1 * i))
        self.prepare()
        rep = self.stage(limit=2, batch_size=1).run()
        self.assertEqual((rep.ok, rep.pending_after), (2, 3))
        rep = self.stage(max_seconds=0.0).run()
        self.assertEqual((rep.status, rep.decoded), ("interrupted", 0))
        rep = self.stage().run()
        self.assertEqual((rep.ok, rep.pending_after), (3, 0))
        runs = [tuple(r) for r in self.conn.execute("SELECT status, decoded FROM loops_run ORDER BY id")]
        self.assertEqual(runs, [("complete", 2), ("interrupted", 0), ("complete", 3)])

    def test_cap_is_unresolved_never_truncated(self):
        self.put(f"{LOOPS}/long 120BPM.wav", click_train(120, 8.0))
        self.prepare()
        rep = self.stage(max_duration_s=5.0).run()
        r = self.res(f"{LOOPS}/long 120BPM.wav")
        self.assertEqual((rep.statuses["unresolved_cap"], rep.decoded), (1, 0))
        self.assertEqual(r["loops_status"], "unresolved_cap")
        self.assertIsNone(r["bpm_est"])
        self.assertIsNone(r["valid"])
        # byte cap on the real decode (header duration unknown to the cap check path)
        rep = self.stage(max_decode_bytes=100_000, force=True).run()
        self.assertEqual(rep.statuses["unresolved_cap"], 1)
        self.assertIn("not decoded", self.res(f"{LOOPS}/long 120BPM.wav")["loops_error"])

    def test_decoded_size_cap_when_header_lies(self):
        self.put(f"{LOOPS}/a 120BPM.wav", click_train(120, 8.0))
        self.prepare()
        self.conn.execute("UPDATE resource SET duration_s=1.0")  # header claims 1 s
        rep = self.stage(max_duration_s=5.0).run()
        self.assertEqual(rep.statuses["unresolved_cap"], 1)
        self.assertIn("not analysed, not truncated", self.res(f"{LOOPS}/a 120BPM.wav")["loops_error"])
        self.assertIsNone(self.res(f"{LOOPS}/a 120BPM.wav")["bpm_est"])

    def test_inconclusive_rerun_does_not_erase_earlier_estimates(self):
        p = self.put(f"{LOOPS}/a 120BPM.wav", click_train(120, 8.0))
        self.prepare()
        self.stage().run()
        est = self.res(f"{LOOPS}/a 120BPM.wav")["bpm_est"]
        self.assertIsNotNone(est)
        p.unlink()  # file vanishes -> unreadable on the forced rerun
        rep = self.stage(force=True).run()
        r = self.res(f"{LOOPS}/a 120BPM.wav")
        self.assertEqual(rep.statuses["unreadable"], 1)
        self.assertEqual(r["loops_status"], "unreadable")
        self.assertEqual(r["bpm_est"], est)
        self.assertEqual(r["valid"], 1)

    def test_truncated_file_is_decode_failed_valid0_and_no_estimates(self):
        good = self.put(f"{LOOPS}/good 120BPM.wav", click_train(120, 8.0))
        raw = good.read_bytes()
        p = self.root / LOOPS / "trunc 120BPM.wav"
        p.write_bytes(raw[: 44 + (len(raw) - 44) // 2])  # data chunk header claims more than present
        self.prepare()
        rep = self.stage().run()
        r = self.res(f"{LOOPS}/trunc 120BPM.wav")
        self.assertEqual((r["valid"], r["loops_status"]), (0, "decode_failed"))
        self.assertTrue(r["valid_error"])
        self.assertIsNone(r["bpm_est"])
        self.assertIsNone(r["loops_json"])
        self.assertEqual((rep.valid0, rep.valid1), (1, 1))

    def test_decoded_in_worker_threads_and_only_main_writes(self):
        import threading
        names = []

        def fake(path, channels):
            names.append(threading.current_thread().name)
            return DecodeResult("ok", samples=np.zeros((SR * 3, 1), dtype=np.float32), frames=SR * 3)

        for i in range(4):
            self.put(f"{LOOPS}/l{i} 120BPM.wav", click_train(120, 4.0 + 0.1 * i))
        self.prepare()
        rep = self.stage(decode_func=fake).run()
        self.assertEqual(rep.ok, 4)
        self.assertTrue(all(n.startswith("loops") for n in names), names)
        self.assertEqual(rep.workers if hasattr(rep, "workers") else 2, 2)
        self.assertEqual(loops.DEFAULT_LOOPS_WORKERS, 2)

    def test_retry_errors_only_redoes_recorded_errors(self):
        self.put(f"{LOOPS}/long 120BPM.wav", click_train(120, 8.0))   # over the 5 s cap
        self.put(f"{LOOPS}/short 120BPM.wav", click_train(120, 4.0))
        self.prepare()
        rep = self.stage(max_duration_s=5.0).run()
        self.assertEqual((rep.ok, rep.statuses["unresolved_cap"]), (1, 1))
        # an error at the current version/settings counts as processed: a plain rerun does nothing
        self.assertEqual(self.stage(max_duration_s=5.0).run().selected, 0)
        # --retry-errors reselects exactly the errored resource
        rep = self.stage(max_duration_s=5.0, retry_errors=True).run()
        self.assertEqual((rep.selected, rep.statuses["unresolved_cap"]), (1, 1))
        # with a raised cap the errored one is reselected (settings signature changed)
        rep = self.stage(max_duration_s=20.0).run()
        self.assertEqual(rep.selected, 2)

    def test_silence_and_noise_files_stage_ok_with_null_estimates(self):
        self.put(f"{LOOPS}/silence 120BPM.wav", np.zeros((SR * 4, 2)))
        self.put(f"{LOOPS}/noise 120BPM.wav", 0.2 * np.random.default_rng(1).standard_normal(SR * 6))
        self.prepare()
        rep = self.stage().run()
        self.assertEqual(rep.ok, 2)
        s = self.res(f"{LOOPS}/silence 120BPM.wav")
        n = self.res(f"{LOOPS}/noise 120BPM.wav")
        self.assertEqual((s["bpm_est"], s["bpm_reason"], s["key_reason"]), (None, "silence", "silence"))
        self.assertEqual((n["bpm_est"], n["bpm_reason"], n["key_est"]), (None, "weak_periodicity", None))
        self.assertEqual(s["bpm_hint_check"], "unavailable")

    def test_cli_exit_code_and_flags(self):
        self.put(f"{LOOPS}/a 128BPM.wav", click_train(128, 6.0))
        self.prepare()
        self.conn.close()
        code = cli.main(["--db", str(self.tmp / "index.sqlite"), "loops", "--workers", "2", "--bpm-max", "190"])
        self.assertEqual(code, 0)
        conn = db.open_db(self.tmp / "index.sqlite")
        self.addCleanup(conn.close)
        self.assertEqual(conn.execute("SELECT loops_status FROM resource WHERE loops_status IS NOT NULL").fetchone()[0], "ok")
        self.conn = conn


class MigrationV4Tests(unittest.TestCase):
    def test_upgrade_from_v3_backs_up_and_adds_columns(self):
        from sample_indexer.migrations import MIGRATIONS
        tmp = Path(tempfile.mkdtemp(prefix="sample-indexer-mig4-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        path = tmp / "old.sqlite"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        for v, sql in MIGRATIONS[:3]:
            conn.executescript(sql)
            conn.execute("INSERT INTO schema_version VALUES (?, 'x')", (v,))
        conn.execute("INSERT INTO resource(sha256,size,ext,first_seen,bpm_name) VALUES (?,?,?,?,?)",
                     ("a" * 64, 3, ".wav", "t", 128.0))
        conn.commit()
        conn.close()
        new = db.open_db(path, create=True)
        self.addCleanup(new.close)
        self.assertEqual(db.schema_version(new), db.LATEST_VERSION)
        backups = list(tmp.glob("old.sqlite.bak-v3-*"))
        self.assertEqual(len(backups), 1)
        b = sqlite3.connect(backups[0])
        self.assertEqual(b.execute("SELECT MAX(version) FROM schema_version").fetchone()[0], 3)
        self.assertNotIn("loops_status", {r[1] for r in b.execute("PRAGMA table_info(resource)")})
        b.close()
        row = new.execute("SELECT bpm_name, bpm_est, key_est, beats, loops_status FROM resource").fetchone()
        self.assertEqual(tuple(row), (128.0, None, None, None, None))


if __name__ == "__main__":
    unittest.main()
