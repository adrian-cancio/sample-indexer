import hashlib
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from sample_indexer import db, reports
from sample_indexer.hashing import hash_file
from sample_indexer.scanner import Scanner


class CountingHasher:
    """hash_func wrapper that counts calls; can raise KeyboardInterrupt after N calls."""

    def __init__(self, interrupt_after=None):
        self.calls = []
        self.interrupt_after = interrupt_after
        self._lock = threading.Lock()

    def __call__(self, path):
        with self._lock:
            self.calls.append(path)
            n = len(self.calls)
        if self.interrupt_after is not None and n > self.interrupt_after:
            raise KeyboardInterrupt
        return hash_file(path)


class IndexerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sample-indexer-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "lib"
        self.root.mkdir()
        self.conn = db.open_db(self.tmp / "index.sqlite", create=True)
        self.addCleanup(self.conn.close)
        self._t = 1_700_000_000

    def write(self, rel, data, mtime=None):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        self._t += 10
        os.utime(p, ns=((mtime or self._t) * 10**9,) * 2)
        return p

    def scan(self, hasher=None, **kw):
        hasher = hasher or CountingHasher()
        kw.setdefault("workers", 3)
        scanner = Scanner(self.conn, self.root, hash_func=hasher, **kw)
        return scanner.run(), hasher

    def paths(self, include_missing=True):
        sql = "SELECT rel_path, missing, resource_id FROM path"
        if not include_missing:
            sql += " WHERE missing=0"
        return {r["rel_path"]: r for r in self.conn.execute(sql + " ORDER BY rel_path")}

    def count(self, table):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class ScanTests(IndexerTestCase):
    def test_first_scan_indexes_and_counts_ignored(self):
        self.write("Samples/Kicks/a.wav", b"kick")
        self.write("Samples/Loops/b.WAV", b"loop")
        self.write("Samples/Loops/c.aif", b"aif")
        self.write("MIDI/x.mid", b"midi")
        self.write("MIDI/notes.txt", b"ignore me")
        self.write("README", b"ignore me too")
        rep, hasher = self.scan()
        self.assertEqual(rep.status, "complete")
        self.assertEqual((rep.discovered, rep.new_paths, rep.hashed), (4, 4, 4))
        self.assertEqual(len(hasher.calls), 4)
        self.assertEqual(rep.ignored_total, 2)
        self.assertEqual(rep.ignored[".txt"], 1)
        self.assertEqual(self.count("resource"), 4)
        rows = {r["rel_path"]: r for r in self.conn.execute("SELECT * FROM path")}
        self.assertEqual(rows["Samples/Loops/b.WAV"]["ext"], ".wav")
        self.assertEqual(rows["Samples/Loops/b.WAV"]["top_folder"], "Samples")
        res = self.conn.execute("SELECT * FROM resource WHERE sha256=?", (hashlib.sha256(b"kick").hexdigest(),)).fetchone()
        self.assertEqual((res["size"], res["ext"]), (4, ".wav"))
        self.assertIsNone(res["kind"])
        self.assertIsNone(res["bpm_est"])

    def test_rescan_without_changes_hashes_nothing(self):
        self.write("a.wav", b"1")
        self.write("d/b.mid", b"2")
        self.scan()
        rep, hasher = self.scan()
        self.assertEqual(hasher.calls, [])
        self.assertEqual((rep.hashed, rep.unchanged, rep.new_paths, rep.marked_missing), (0, 2, 0, 0))

    def test_content_edit_rehashes_and_keeps_old_resource(self):
        p = self.write("a.wav", b"version one")
        self.scan()
        old = self.paths()["a.wav"]["resource_id"]
        p.write_bytes(b"version two!")
        os.utime(p, ns=((self._t + 99) * 10**9,) * 2)
        rep, hasher = self.scan()
        self.assertEqual((rep.changed, rep.hashed, len(hasher.calls)), (1, 1, 1))
        new = self.paths()["a.wav"]["resource_id"]
        self.assertNotEqual(old, new)
        self.assertEqual(self.count("resource"), 2)  # old resource row and metadata retained
        self.assertEqual(self.count("path"), 1)

    def test_same_size_and_mtime_edit_is_not_detected(self):
        # Documented limitation of the (size, mtime) skip rule.
        p = self.write("a.wav", b"AAAA", mtime=1_700_000_500)
        self.scan()
        p.write_bytes(b"BBBB")
        os.utime(p, ns=(1_700_000_500 * 10**9,) * 2)
        rep, hasher = self.scan()
        self.assertEqual(hasher.calls, [])

    def test_rename_keeps_resource_and_does_not_rehash(self):
        p = self.write("Samples/old/kick.wav", b"kick data")
        self.scan()
        self.conn.execute("UPDATE resource SET kind='one_shot', f0_hz=55.0 WHERE 1")
        before = self.paths()["Samples/old/kick.wav"]
        (self.root / "Samples/new").mkdir(parents=True)
        os.rename(p, self.root / "Samples/new/kick renamed.wav")  # mtime preserved by rename
        rep, hasher = self.scan()
        self.assertEqual(hasher.calls, [])
        self.assertEqual((rep.moved, rep.new_paths, rep.marked_missing, rep.hashed), (1, 0, 0, 0))
        paths = self.paths()
        # Definition: a detected move UPDATES the existing path row in place (same row id),
        # so no 'missing' ghost is left behind.
        self.assertEqual(list(paths), ["Samples/new/kick renamed.wav"])
        row = self.conn.execute("SELECT id, resource_id, missing FROM path").fetchone()
        self.assertEqual(row["resource_id"], before["resource_id"])
        self.assertEqual(row["missing"], 0)
        res = self.conn.execute("SELECT kind, f0_hz FROM resource").fetchone()
        self.assertEqual((res["kind"], res["f0_hz"]), ("one_shot", 55.0))
        self.assertEqual(self.count("resource"), 1)

    def test_move_with_touched_mtime_falls_back_to_hash(self):
        p = self.write("a/x.wav", b"same bytes")
        self.scan()
        rid = self.paths()["a/x.wav"]["resource_id"]
        os.makedirs(self.root / "b")
        os.rename(p, self.root / "b/x.wav")
        os.utime(self.root / "b/x.wav", ns=((self._t + 500) * 10**9,) * 2)
        rep, hasher = self.scan()
        self.assertEqual(len(hasher.calls), 1)
        self.assertEqual(self.count("resource"), 1)  # same sha256 -> same resource
        paths = self.paths()
        self.assertEqual(paths["b/x.wav"]["resource_id"], rid)
        self.assertEqual(paths["a/x.wav"]["missing"], 1)  # old path flagged, row kept
        self.assertEqual(paths["b/x.wav"]["missing"], 0)

    def test_ambiguous_rename_candidates_are_hashed(self):
        a = self.write("a.wav", b"AAAA", mtime=1_700_000_000)
        b = self.write("b.wav", b"BBBB", mtime=1_700_000_000)  # same size+mtime, different content
        self.scan()
        os.rename(a, self.root / "a2.wav")
        os.rename(b, self.root / "b2.wav")
        rep, hasher = self.scan()
        self.assertEqual(rep.moved, 0)
        self.assertEqual(len(hasher.calls), 2)
        self.assertEqual(self.count("resource"), 2)

    def test_deletion_marks_missing_and_keeps_rows(self):
        a = self.write("a.wav", b"aa")
        self.write("b.wav", b"bb")
        self.scan()
        a.unlink()
        rep, hasher = self.scan()
        self.assertEqual((rep.marked_missing, hasher.calls), (1, []))
        paths = self.paths()
        self.assertEqual(paths["a.wav"]["missing"], 1)
        self.assertEqual(paths["b.wav"]["missing"], 0)
        self.assertEqual(self.count("resource"), 2)
        # idempotent: a further scan does not re-count it
        rep, _ = self.scan()
        self.assertEqual(rep.marked_missing, 0)
        # reappearing with identical size/mtime clears the flag without hashing
        self.write("a.wav", b"aa", mtime=self.conn.execute("SELECT mtime_ns FROM path WHERE rel_path='a.wav'").fetchone()[0] // 10**9)
        rep, hasher = self.scan()
        self.assertEqual((rep.restored, hasher.calls), (1, []))
        self.assertEqual(self.paths()["a.wav"]["missing"], 0)

    def test_duplicate_content_two_paths_one_resource(self):
        self.write("a/one.wav", b"dup")
        self.write("b/two.wav", b"dup")
        self.write("c/three.wav", b"other")
        self.scan()
        self.assertEqual(self.count("resource"), 2)
        self.assertEqual(self.count("path"), 3)
        p = self.paths()
        self.assertEqual(p["a/one.wav"]["resource_id"], p["b/two.wav"]["resource_id"])
        self.assertIn("duplicate sha256   1 resources on 2 present paths", reports.stats(self.conn))

    def test_zero_byte_and_non_audio_files(self):
        self.write("empty.wav", b"")
        self.write("empty2.mid", b"")
        self.write("junk.wav", b"not really audio")
        self.write("notes.txt", b"x")
        self.write("cover.JPG", b"x")
        self.write("noext", b"x")
        rep, _ = self.scan()
        self.assertEqual(rep.status, "complete")
        self.assertEqual(rep.errors, [])
        self.assertEqual(rep.hashed, 3)
        self.assertEqual(rep.ignored, {".txt": 1, ".jpg": 1, "(none)": 1})
        # the two empty files share the sha256 of b"" -> one resource
        self.assertEqual(self.count("resource"), 2)
        empty = hashlib.sha256(b"").hexdigest()
        self.assertEqual(
            self.conn.execute("SELECT size FROM resource WHERE sha256=?", (empty,)).fetchone()[0], 0
        )

    def test_unreadable_file_is_reported_not_fatal(self):
        self.write("ok.wav", b"ok")
        bad = self.write("bad.wav", b"nope")
        os.chmod(bad, 0)
        self.addCleanup(os.chmod, bad, 0o644)
        if os.access(bad, os.R_OK):
            self.skipTest("running with privileges that bypass file permissions")
        rep, _ = self.scan()
        self.assertEqual(rep.status, "complete")
        self.assertEqual(len(rep.errors), 1)
        self.assertEqual(list(self.paths()), ["ok.wav"])
        os.chmod(bad, 0o644)
        rep, _ = self.scan()
        self.assertEqual((rep.errors, rep.hashed), ([], 1))  # retried next scan

    def test_interrupted_scan_then_resume(self):
        for i in range(30):
            self.write(f"s/f{i:02d}.wav", f"data-{i}".encode())
        rep, hasher = self.scan(CountingHasher(interrupt_after=12), workers=1, batch_size=5)
        self.assertEqual(rep.status, "interrupted")
        done = self.count("path")
        self.assertTrue(0 < done < 30, done)
        rep, hasher = self.scan(workers=1)
        self.assertEqual(rep.status, "complete")
        self.assertEqual(len(hasher.calls), 30 - done)  # only the remainder is hashed
        self.assertEqual(self.count("path"), 30)
        self.assertEqual(self.count("resource"), 30)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM path WHERE missing=1").fetchone()[0], 0)
        runs = [r[0] for r in self.conn.execute("SELECT status FROM scan_run ORDER BY id")]
        self.assertEqual(runs, ["interrupted", "complete"])

    def test_interrupt_does_not_mark_missing(self):
        for i in range(6):
            self.write(f"f{i}.wav", f"d{i}".encode())
        self.scan()
        (self.root / "f0.wav").unlink()
        self.write("new.wav", b"brand new")
        rep, _ = self.scan(CountingHasher(interrupt_after=0), workers=1)
        self.assertEqual(rep.status, "interrupted")
        self.assertEqual(self.paths()["f0.wav"]["missing"], 0)

    def test_limit_scan_is_partial_and_never_marks_missing(self):
        for i in range(10):
            self.write(f"f{i:02d}.wav", f"d{i}".encode())
        rep, hasher = self.scan(limit=4)
        self.assertEqual((rep.discovered, rep.hashed, rep.partial), (4, 4, True))
        rep, hasher = self.scan(limit=4)
        self.assertEqual((rep.hashed, rep.unchanged), (0, 4))
        rep, hasher = self.scan()  # full
        self.assertEqual((rep.hashed, rep.unchanged, rep.marked_missing), (6, 4, 0))
        self.assertEqual(self.count("path"), 10)

    def test_changed_while_hashing_is_not_stored(self):
        p = self.write("a.wav", b"abc")

        def racing_hash(path):
            sha = hash_file(path)
            os.utime(path, ns=((self._t + 777) * 10**9,) * 2)
            return sha

        rep, _ = self.scan(racing_hash)
        self.assertEqual(len(rep.errors), 1)
        self.assertEqual(self.count("path"), 0)
        rep, _ = self.scan()
        self.assertEqual((rep.errors, rep.hashed), ([], 1))

    def test_symlinks_are_not_followed(self):
        self.write("real.wav", b"real")
        os.symlink(self.root / "real.wav", self.root / "link.wav")
        os.symlink(self.tmp, self.root / "linkdir")
        rep, _ = self.scan()
        self.assertEqual((rep.hashed, self.count("path")), (1, 1))
        self.assertEqual(rep.ignored["(symlink)"], 2)

    def test_parallel_workers_produce_same_result(self):
        for i in range(40):
            self.write(f"d{i % 4}/f{i}.wav", f"payload {i}".encode() * (i + 1))
        rep, hasher = self.scan(workers=8, batch_size=7)
        self.assertEqual((rep.hashed, self.count("resource")), (40, 40))
        for r in self.conn.execute("SELECT p.rel_path, r.sha256 FROM path p JOIN resource r ON r.id=p.resource_id"):
            self.assertEqual(r["sha256"], hash_file(self.root / r["rel_path"]))

    def test_missing_root_is_an_error(self):
        scanner = Scanner(self.conn, self.tmp / "nope")
        with self.assertRaises(NotADirectoryError):
            scanner.run()
        self.assertEqual(self.conn.execute("SELECT status FROM scan_run").fetchone()[0], "aborted")


class ReportTests(IndexerTestCase):
    def test_stats_query_and_manifest(self):
        self.write("Samples/Kicks/a.wav", b"k1")
        self.write("Samples/Kicks/b.WAV", b"k2")
        self.write("MIDI/c.mid", b"m")
        gone = self.write("Samples/Hats/d.aif", b"h")
        self.scan()
        gone.unlink()
        self.scan()
        text = reports.stats(self.conn)
        self.assertIn("paths present      3", text)
        self.assertIn("paths missing      1", text)
        self.assertIn("audio 2 + midi 1 = 3", text)
        self.assertIn("Samples", text)
        rows = reports.query(self.conn, ext="WAV")
        self.assertEqual([r["rel_path"] for r in rows], ["Samples/Kicks/a.wav", "Samples/Kicks/b.WAV"])
        self.assertEqual(len(reports.query(self.conn, folder="samples/kicks")), 2)
        self.assertEqual(len(reports.query(self.conn, path_contains="HATS")), 0)
        self.assertEqual(len(reports.query(self.conn, path_contains="HATS", include_missing=True)), 1)

        csv_path = self.tmp / "m.csv"
        lines = ["sha256,rel_path,size,mtime"]
        for r in self.conn.execute("SELECT p.rel_path, p.size, r.sha256 FROM path p JOIN resource r ON r.id=p.resource_id WHERE missing=0"):
            lines.append(f"{r['sha256']},{r['rel_path']},{r['size']},0")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertEqual(reports.verify_manifest(self.conn, csv_path).differences, 0)

        lines[1] = "0" * 64 + lines[1][64:]
        lines.append(f"{'1' * 64},Samples/ghost.wav,1,0")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        diff = reports.verify_manifest(self.conn, csv_path)
        self.assertEqual((len(diff.sha_mismatch), len(diff.only_in_manifest), len(diff.only_in_db)), (1, 1, 0))


class SchemaTests(IndexerTestCase):
    def test_schema_has_future_analysis_columns_and_stage_table(self):
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(resource)")}
        expected = {
            "kind", "category", "confidence", "duration_s", "sr", "bits", "channels", "valid",
            "valid_error", "bpm_name", "bpm_est", "beats", "key_name", "key_est", "f0_hz",
            "sub_peak_hz", "centroid", "rolloff", "flatness", "band_sub", "band_low", "band_mid",
            "band_high", "lufs", "true_peak", "crest", "dc_offset", "clipping", "stereo_corr_low",
            "decay_ms", "ppq", "tempo", "bars", "notes", "pitch_min", "pitch_max", "density",
            "scale_est", "tracks",
        }
        self.assertFalse(expected - cols, expected - cols)
        stage_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(resource_stage)")}
        self.assertEqual(stage_cols, {"resource_id", "stage", "version", "done_at", "error"})
        with self.assertRaises(Exception):
            self.conn.execute(
                "INSERT INTO resource(sha256,size,ext,first_seen,kind) VALUES (?,?,?,?,?)",
                ("a" * 64, 1, ".wav", "x", "bogus"),
            )

    def test_migrate_is_idempotent(self):
        self.assertEqual(db.migrate(self.conn), [])
        self.assertEqual(db.schema_version(self.conn), 1)

    def test_open_without_init_fails(self):
        with self.assertRaises(db.DatabaseNotFound):
            db.open_db(self.tmp / "absent.sqlite", create=False)


if __name__ == "__main__":
    unittest.main()
