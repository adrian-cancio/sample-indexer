"""MIDI runner tests use temporary libraries and databases only."""
import tempfile
import unittest
from pathlib import Path
from sample_indexer import db
from sample_indexer.scanner import Scanner
from sample_indexer.midi import MidiStage


def smf():
    track = bytes.fromhex('00903c648360803c0000ff2f00')
    return b'MThd'+(6).to_bytes(4,'big')+bytes.fromhex('0000000101e0')+b'MTrk'+len(track).to_bytes(4,'big')+track


class MidiStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'library'
        self.root.mkdir()
        (self.root/'a.mid').write_bytes(smf())
        (self.root/'bad.mid').write_bytes(b'bad')
        self.conn = db.open_db(Path(self.tmp.name)/'index.sqlite', create=True)
        self.addCleanup(self.conn.close)
        Scanner(self.conn, self.root).run()

    def test_resume_error_force_and_preservation(self):
        self.conn.execute('UPDATE resource SET bpm_name=150, valid=1')
        first = MidiStage(self.conn, limit=1).run()
        self.assertEqual(first.processed, 1)
        second = MidiStage(self.conn).run()
        self.assertEqual(second.processed, 1)
        self.assertEqual(MidiStage(self.conn).run().parsed, 0)
        retry = MidiStage(self.conn, retry_errors=True).run()
        self.assertEqual(retry.processed, 1)
        self.assertEqual(retry.errors, 1)
        self.assertIn('bad.mid', retry.failed_paths[0]['path'])
        self.assertEqual(MidiStage(self.conn, force=True).run().processed, 2)
        row = self.conn.execute('SELECT * FROM resource WHERE notes IS NOT NULL').fetchone()
        self.assertEqual(row['notes'], 1)
        self.assertEqual(row['bpm_name'], 150)
        self.assertEqual(row['valid'], 1)

    def test_coverage_excludes_missing_resources(self):
        import json
        from sample_indexer.midi import coverage
        MidiStage(self.conn).run()
        self.conn.execute('UPDATE path SET missing=1')
        self.assertEqual(json.loads(coverage(self.conn))['pending'], 0)
        self.assertEqual(json.loads(coverage(self.conn))['current'], 0)

    def test_unexpected_failure_marks_aborted(self):
        from unittest.mock import patch
        with patch('sample_indexer.midi.midi_parser.parse_bytes', side_effect=RuntimeError('bug')):
            with self.assertRaises(RuntimeError):
                MidiStage(self.conn).run()
        self.assertEqual(self.conn.execute('SELECT status FROM midi_run').fetchone()[0], 'aborted')

    def test_v5_upgrade_has_backup(self):
        import sqlite3
        from sample_indexer.migrations import MIGRATIONS
        old_path = Path(self.tmp.name)/'old.sqlite'
        old = sqlite3.connect(old_path)
        old.execute('CREATE TABLE schema_version(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)')
        for version, sql in MIGRATIONS[:5]:
            old.executescript(sql)
            old.execute('INSERT INTO schema_version VALUES (?,?)', (version, 'test'))
            old.commit()
        old.close()
        upgraded = db.open_db(old_path, create=True)
        self.addCleanup(upgraded.close)
        self.assertEqual(db.schema_version(upgraded), db.LATEST_VERSION)
        backups = list(old_path.parent.glob('old.sqlite.bak-v5-*'))
        self.assertEqual(len(backups), 1)
        backup = db.connect(backups[0], create=False)
        self.addCleanup(backup.close)
        self.assertEqual(db.schema_version(backup), 5)

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            MidiStage(self.conn, limit=-1)
