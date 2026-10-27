"""Versioned sequential MIDI parsing; only MIDI-owned columns are written.

Each resource is checkpointed atomically. Errors count as processed. All attempted
paths and failures are retained; source bytes are never stored or modified.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random

from . import db
from . import midi_parser

MIDI_VERSION = 1


@dataclass
class MidiReport:
    selected: int = 0
    processed: int = 0
    parsed: int = 0
    errors: int = 0
    status: str = 'complete'
    failed_paths: list = field(default_factory=list)

    def format(self):
        return json.dumps(self.__dict__, indent=2)


class MidiStage:
    def __init__(self, conn, *, limit=None, force=False, retry_errors=False,
                 stratify=False, seed=0):
        if limit is not None and limit < 0:
            raise ValueError('limit must be nonnegative')
        self.conn, self.limit = conn, limit
        self.force, self.retry_errors = force, retry_errors
        self.stratify, self.seed = stratify, seed

    def run(self):
        rows = self.conn.execute('''SELECT r.*, s.version AS stage_version,
            s.error AS stage_error FROM resource r
            LEFT JOIN resource_stage s ON s.resource_id=r.id AND s.stage='midi'
            WHERE r.id IN (SELECT p.resource_id FROM path p
                WHERE p.missing=0 AND p.ext IN ('.mid','.midi','.rmi')) ORDER BY r.id''').fetchall()
        rows = [r for r in rows if self.force or
                (self.retry_errors and r['stage_version'] == MIDI_VERSION and r['stage_error']) or
                (not self.retry_errors and (r['stage_version'] or 0) < MIDI_VERSION)]
        if self.stratify:
            groups = {}
            for row in rows:
                folder = self.conn.execute('SELECT top_folder, rel_path FROM path WHERE resource_id=? AND missing=0 ORDER BY rel_path LIMIT 1', (row['id'],)).fetchone()
                key = str(Path(folder['rel_path']).parent)
                groups.setdefault(key, []).append(row)
            rng = random.Random(self.seed)
            for group in groups.values():
                rng.shuffle(group)
            rows = []
            while any(groups.values()):
                for key in sorted(groups):
                    if groups[key]:
                        rows.append(groups[key].pop())
        if self.limit is not None:
            rows = rows[:self.limit]
        report = MidiReport(selected=len(rows))
        run_id = self.conn.execute('INSERT INTO midi_run(started_at,status,selected) VALUES (?, ?, ?)', (db.utcnow(), 'running', len(rows))).lastrowid
        try:
            for row in rows:
                failures, attempted, descriptor = [], [], None
                paths = self.conn.execute('SELECT * FROM path WHERE resource_id=? AND missing=0 ORDER BY root,rel_path', (row['id'],)).fetchall()
                for path in paths:
                    full = Path(path['root'])/path['rel_path']
                    attempted.append(str(full))
                    try:
                        before = full.stat()
                        if (before.st_size, before.st_mtime_ns) != (path['size'], path['mtime_ns']):
                            raise ValueError('stale source: size/mtime changed since scan')
                        # Bounded read and content identity check before parsing.
                        with full.open('rb') as stream:
                            content = stream.read(midi_parser.MAX_FILE_BYTES + 1)
                        if len(content) > midi_parser.MAX_FILE_BYTES:
                            raise ValueError('MIDI file size cap exceeded')
                        if hashlib.sha256(content).hexdigest() != row['sha256']:
                            raise ValueError('stale source: SHA-256 differs from resource')
                        report.parsed += 1
                        descriptor = midi_parser.parse_bytes(content)
                        after = full.stat()
                        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                            raise ValueError('source changed during parse')
                        break
                    except (OSError, ValueError) as exc:
                        descriptor = None
                        failures.append({'path': str(full), 'error': str(exc)})
                error = None if descriptor is not None else '; '.join(f"{x['path']}: {x['error']}" for x in failures) or 'no present path'
                status = 'error' if error else ('empty' if descriptor.get('note_count') == 0 else 'ok')
                now = db.utcnow()
                if descriptor is not None:
                    descriptor['provenance'] = {'stage_version': MIDI_VERSION, 'sha256': row['sha256'], 'attempted_paths': attempted, 'failed_paths': failures}
                with db.transaction(self.conn):
                    self.conn.execute('INSERT OR REPLACE INTO midi_measurement VALUES (?,?,?,?,?,?)', (row['id'], status, error, now, json.dumps(descriptor, allow_nan=False) if descriptor is not None else None, json.dumps(failures)))
                    if descriptor is not None:
                        columns = ('ppq','tempo','bars','notes','pitch_min','pitch_max','density','scale_est','tracks')
                        tempo_map = descriptor.get('tempo_map')
                        values = [descriptor.get('ppq'),
                                  tempo_map[0]['bpm'] if tempo_map and len(tempo_map) == 1 else None,
                                  descriptor.get('bars'), descriptor.get('note_count'),
                                  descriptor.get('pitch_min'), descriptor.get('pitch_max'),
                                  descriptor.get('notes_per_beat'),
                                  descriptor.get('key_estimate', {}).get('key'),
                                  descriptor.get('track_count')]
                        self.conn.execute('UPDATE resource SET '+','.join(k+'=?' for k in columns)+',midi_ticks=?,midi_beats=?,midi_duration_s=?,midi_status=? WHERE id=?', (*values, descriptor.get('ticks'), descriptor.get('beats'), descriptor.get('duration_seconds'), status, row['id']))
                    else:
                        self.conn.execute('UPDATE resource SET midi_status=? WHERE id=?', (status, row['id']))
                    self.conn.execute('INSERT OR REPLACE INTO resource_stage(resource_id,stage,version,done_at,error,input_sig) VALUES (?,\'midi\',?,?,?,?)', (row['id'], MIDI_VERSION, now, error, row['sha256']))
                    report.processed += 1
                    report.errors += bool(error)
                    report.failed_paths.extend(failures)
                    self.conn.execute('UPDATE midi_run SET processed=?,parsed=?,errors=? WHERE id=?', (report.processed, report.parsed, report.errors, run_id))
        except KeyboardInterrupt:
            report.status = 'interrupted'
        except BaseException:
            report.status = 'aborted'
            raise
        finally:
            self.conn.execute('UPDATE midi_run SET finished_at=?,status=? WHERE id=?', (db.utcnow(), report.status, run_id))
        return report


def coverage(conn):
    rows = conn.execute('SELECT status, COUNT(*) n FROM midi_measurement GROUP BY status').fetchall()
    eligible = conn.execute("SELECT COUNT(*) FROM resource r WHERE r.id IN (SELECT p.resource_id FROM path p WHERE missing=0 AND p.ext IN ('.mid','.midi','.rmi'))").fetchone()[0]
    current = conn.execute("SELECT COUNT(*) FROM resource_stage s WHERE stage='midi' AND version=? AND s.resource_id IN (SELECT p.resource_id FROM path p WHERE missing=0 AND p.ext IN ('.mid','.midi','.rmi'))", (MIDI_VERSION,)).fetchone()[0]
    return json.dumps({'eligible': eligible, 'current': current, 'pending': eligible-current, 'statuses': {r['status']: r['n'] for r in rows}}, indent=2)
