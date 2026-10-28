"""Tests for L3-9 extended technical queries: numeric bounds, nulls, provenance, errors, hostile SQL."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import sample_indexer
from sample_indexer import cli, db, reports


class QueryTechnicalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.sqlite"
        self.conn = db.open_db(self.db_path, create=True)
        # Seed test resources
        self._seed_data()

    def tearDown(self):
        self.conn.close()
        self.temp_dir.cleanup()

    def _seed_data(self):
        now = db.utcnow()
        # Row 1: Valid kick, f0=55, decay=200, mono, sr=44100
        self.conn.execute("""
            INSERT INTO resource (id, sha256, size, ext, first_seen, duration_s, sr, channels,
                                  valid, probe_status, f0_hz, sub_peak_hz, decay_ms, stereo_corr_low)
            VALUES (1, '1111111111111111111111111111111111111111111111111111111111111111', 1000, '.wav',
                    ?, 0.5, 44100, 1, 1, 'ok', 55.0, 50.0, 200.0, NULL)
        """, (now,))
        self.conn.execute("""
            INSERT INTO path (id, resource_id, root, rel_path, top_folder, ext, size, mtime_ns, first_seen, last_seen, missing)
            VALUES (1, 1, '/root', 'Samples/Kicks/Kick1.wav', 'Samples', '.wav', 1000, 0, ?, ?, 0)
        """, (now, now))

        # Row 2: Valid loop, bpm_name=130, bpm_est=130.2, key_est='F min', stereo, sr=48000
        self.conn.execute("""
            INSERT INTO resource (id, sha256, size, ext, first_seen, duration_s, sr, channels,
                                  valid, probe_status, bpm_name, bpm_est, bpm_confidence, key_name, key_est, stereo_corr_low)
            VALUES (2, '2222222222222222222222222222222222222222222222222222222222222222', 2000, '.wav',
                    ?, 4.0, 48000, 2, 1, 'ok', 130.0, 130.2, 0.85, 'F minor', 'F minor', 0.95)
        """, (now,))
        self.conn.execute("""
            INSERT INTO path (id, resource_id, root, rel_path, top_folder, ext, size, mtime_ns, first_seen, last_seen, missing)
            VALUES (2, 2, '/root', 'Samples/Loops/BassLoop.wav', 'Samples', '.wav', 2000, 0, ?, ?, 0)
        """, (now, now))

        # Row 3: MIDI file, notes=120, density=4.5, bars=4.0
        self.conn.execute("""
            INSERT INTO resource (id, sha256, size, ext, first_seen, duration_s, ext, midi_status,
                                  notes, pitch_min, pitch_max, density, bars, tempo)
            VALUES (3, '3333333333333333333333333333333333333333333333333333333333333333', 500, '.mid',
                    ?, 8.0, '.mid', 'ok', 120, 36, 72, 4.5, 4.0, 120.0)
        """, (now,))
        self.conn.execute("""
            INSERT INTO path (id, resource_id, root, rel_path, top_folder, ext, size, mtime_ns, first_seen, last_seen, missing)
            VALUES (3, 3, '/root', 'MIDI/Theme.mid', 'MIDI', '.mid', 500, 0, ?, ?, 0)
        """, (now, now))

        # Row 4: Error/Corrupt file (valid=0)
        self.conn.execute("""
            INSERT INTO resource (id, sha256, size, ext, first_seen, duration_s, valid, valid_error, probe_status)
            VALUES (4, '4444444444444444444444444444444444444444444444444444444444444444', 300, '.wav',
                    ?, 1.0, 0, 'corrupt', 'ok')
        """, (now,))
        self.conn.execute("""
            INSERT INTO path (id, resource_id, root, rel_path, top_folder, ext, size, mtime_ns, first_seen, last_seen, missing)
            VALUES (4, 4, '/root', 'Samples/Corrupt.wav', 'Samples', '.wav', 300, 0, ?, ?, 0)
        """, (now, now))

    def test_default_excludes_errors(self):
        rows = reports.query(self.conn, ext="wav")
        paths = [r["rel_path"] for r in rows]
        self.assertIn("Samples/Kicks/Kick1.wav", paths)
        self.assertIn("Samples/Loops/BassLoop.wav", paths)
        self.assertNotIn("Samples/Corrupt.wav", paths)

    def test_opt_in_includes_errors(self):
        rows = reports.query(self.conn, ext="wav", include_errors=True)
        paths = [r["rel_path"] for r in rows]
        self.assertIn("Samples/Corrupt.wav", paths)

    def test_numeric_bounds_exclude_nulls(self):
        # Kick has f0=55, loop has NULL f0. Filtering f0_min=50 should only return Kick.
        rows = reports.query(self.conn, f0_min=50.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rel_path"], "Samples/Kicks/Kick1.wav")

        # Range filter
        rows = reports.query(self.conn, f0_min=60.0)
        self.assertEqual(len(rows), 0)

    def test_bpm_provenance_and_confidence(self):
        # BPM est only
        rows = reports.query(self.conn, bpm_min=125, bpm_max=135, bpm_provenance="est")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rel_path"], "Samples/Loops/BassLoop.wav")

        # Confidence threshold
        rows = reports.query(self.conn, bpm_confidence_min=0.9)
        self.assertEqual(len(rows), 0)
        rows = reports.query(self.conn, bpm_confidence_min=0.8)
        self.assertEqual(len(rows), 1)

    def test_midi_notes_and_density(self):
        rows = reports.query(self.conn, ext="mid", midi_notes_min=100, midi_density_min=4.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rel_path"], "MIDI/Theme.mid")

        rows = reports.query(self.conn, ext="mid", midi_notes_min=200)
        self.assertEqual(len(rows), 0)

    def test_sql_injection_defense(self):
        # Hostile input in path_contains
        hostile = "'; DROP TABLE resource; --"
        rows = reports.query(self.conn, path_contains=hostile)
        self.assertEqual(len(rows), 0)
        # Ensure resource table still exists and has rows
        count = self.conn.execute("SELECT COUNT(*) FROM resource").fetchone()[0]
        self.assertEqual(count, 4)

    def test_query_json_cli(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            ret = cli.main(["--db", str(self.db_path), "query", "--json", "--limit", "10"])
        self.assertEqual(ret, 0)
        parsed = json.loads(buf.getvalue())
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 3)  # row 4 is corrupt/error and excluded by default
        paths = [item["rel_path"] for item in parsed]
        self.assertIn("Samples/Kicks/Kick1.wav", paths)

    def test_package_exports(self):
        self.assertEqual(sample_indexer.__version__, "0.1.0")
        self.assertIs(sample_indexer.query, reports.query)
        # Verify query can be called via package export
        results = sample_indexer.query(self.conn, ext="wav")
        self.assertEqual(len(results), 2)



if __name__ == "__main__":
    unittest.main()
