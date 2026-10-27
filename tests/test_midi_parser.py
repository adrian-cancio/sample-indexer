"""Synthetic, independent SMF fixtures; never access the user library."""
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sample_indexer import midi_parser as midi


def vlq(n):
    result = [n & 127]
    while n >> 7:
        n >>= 7
        result.insert(0, (n & 127) | 128)
    return bytes(result)


def event(delta, payload):
    return vlq(delta) + payload


def smf(*tracks, fmt=0, division=480):
    return b'MThd' + struct.pack('>IHHH', 6, fmt, len(tracks), division) + b''.join(
        b'MTrk' + struct.pack('>I', len(t)) + t for t in tracks)


EOT = b'\x00\xff\x2f\x00'
NOTE = b'\x00\x90\x3c\x64' + event(480, b'\x3c\x00') + EOT


class MidiParserTests(unittest.TestCase):
    def test_running_status_zero_velocity_and_file(self):
        data = smf(NOTE)
        result = midi.parse_bytes(data)
        self.assertEqual(result['note_count'], 1)
        self.assertEqual(result['ticks'], 480)
        self.assertEqual(result['beats'], 1)
        self.assertEqual(result['duration_seconds'], .5)
        self.assertEqual(result['pitch_min'], 60)
        self.assertEqual(result['velocity_mean'], 100)
        self.assertEqual(result['polyphony_max'], 1)
        self.assertEqual(result['notes_per_beat'], 1)
        json.dumps(result, allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'x.mid'
            path.write_bytes(data)
            self.assertEqual(midi.parse_file(path), result)

    def test_tempo_meter_sysex(self):
        track = (b'\x00\xff\x51\x03\x07\xa1\x20'
                 b'\x00\xff\x58\x04\x03\x02\x18\x08'
                 b'\x00\xf0\x02\x01\xf7' + b'\x00\x90\x3c\x40'
                 + event(480, b'\xff\x51\x03\x0f\x42\x40')
                 + event(480, b'\x80\x3c\x00') + EOT)
        result = midi.parse_bytes(smf(track))
        self.assertEqual(result['duration_seconds'], 1.5)
        self.assertEqual(result['meter_map'][0]['numerator'], 3)
        self.assertEqual(result['tempo_map'][1]['bpm'], 60)

    def test_conductor_maps_not_multiplied_by_tracks(self):
        conductor = b''.join(event(1, b'\xff\x51\x03\x07\xa1\x20') for _ in range(100)) + EOT
        result = midi.parse_bytes(smf(conductor, *([EOT] * 100), fmt=1))
        self.assertEqual(len(result['tempo_map']), 101)
        self.assertTrue(all('tempo_map' not in t for t in result['tracks']))

    def test_meter_change_and_explicit_default_provenance(self):
        track = (b'\x00\x90\x3c\x64' + event(480, b'\xff\x58\x04\x03\x02\x18\x08')
                 + event(480, b'\x80\x3c\x00') + EOT)
        result = midi.parse_bytes(smf(track))
        self.assertEqual([m['numerator'] for m in result['meter_map']], [4, 3])
        self.assertAlmostEqual(result['bars'], 0.583333, places=5)
        self.assertEqual(result['bars_basis'], 'integrated_meter_map')
        self.assertEqual(result['tempo_provenance'], 'smf_default_120_bpm_until_first_event')
        self.assertEqual(result['duration_seconds'], 1)

    def test_formats(self):
        conductor = b'\x00\xff\x51\x03\x0f\x42\x40' + EOT
        result = midi.parse_bytes(smf(conductor, NOTE, fmt=1))
        self.assertEqual(result['duration_seconds'], 1)
        independent = midi.parse_bytes(smf(conductor, NOTE, fmt=2))
        self.assertFalse(independent['coherent_timeline'])
        self.assertIsNone(independent['duration_seconds'])
        self.assertIsNone(independent['beats'])
        self.assertEqual(independent['tracks'][1]['duration_seconds'], .5)

    def test_percussion_excluded_from_pitch_class(self):
        track = b'\x00\x99\x24\x7f' + event(480, b'\x89\x24\x00') + EOT
        result = midi.parse_bytes(smf(track))
        self.assertEqual(result['note_count'], 1)
        self.assertEqual(result['percussion_note_count'], 1)
        self.assertEqual(sum(result['pitch_class_histogram']), 0)
        self.assertIsNone(result['key_estimate']['key'])

    def test_polyphony_and_ambiguous_key(self):
        track = b'\x00\x90\x3c\x40\x00\x90\x40\x60' + event(480, b'\x80\x3c\x00') + b'\x00\x80\x40\x00' + EOT
        result = midi.parse_bytes(smf(track))
        self.assertEqual(result['polyphony_max'], 2)
        self.assertEqual(result['velocity_mean'], 80)
        self.assertIn('margin', result['key_estimate'])
        self.assertIn('ambiguous', result['key_estimate'])

    def test_rmid_padding(self):
        data = smf(NOTE)
        chunk = b'JUNK' + struct.pack('<I', 1) + b'x\x00'
        chunk += b'data' + struct.pack('<I', len(data)) + data + (b'\x00' if len(data) % 2 else b'')
        wrapped = b'RIFF' + struct.pack('<I', 4 + len(chunk)) + b'RMID' + chunk
        self.assertEqual(midi.parse_bytes(wrapped), midi.parse_bytes(data))

    def test_malformed(self):
        bad_tracks = [b'', b'\x00\x90\x3c', b'\x00\x3c\x40' + EOT,
                      b'\x81\x80\x80\x80\x00' + EOT,
                      b'\x00\xff\x2f\x01\x00', EOT + b'\x00',
                      b'\x00\xff\x51\x02\x01\x02' + EOT,
                      b'\x00\xff\x51\x03\x00\x00\x00' + EOT,
                      b'\x00\x90\x80\x40' + EOT,
                      b'\x00\xf0\x7f' + EOT]
        for track in bad_tracks:
            with self.subTest(track=track), self.assertRaises(ValueError):
                midi.parse_bytes(smf(track))
        for data in [b'', smf(NOTE)[:-1], smf(NOTE) + b'x', smf(NOTE, division=0),
                     smf(NOTE, division=0xe728), smf(NOTE, fmt=3), smf(NOTE, NOTE)]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                midi.parse_bytes(data)

    def test_overlapping_same_pitch_and_unclosed_notes(self):
        track = (b'\x00\x90\x3c\x20\x00\x3c\x60'
                 + event(240, b'\x80\x3c\x00') + event(240, b'\xff\x2f\x00'))
        result = midi.parse_bytes(smf(track))
        self.assertEqual(result['note_count'], 2)
        self.assertEqual(result['polyphony_max'], 2)
        self.assertEqual(result['unclosed_note_count'], 1)
        self.assertEqual(result['duration_seconds'], .5)

    def test_running_status_preserved_across_meta_cancelled_by_sysex(self):
        # Spec: SysEx cancels running status; Meta events do not cancel running status.
        meta_middle = b'\x00\xff\x01\x00'
        track_meta = b'\x00\x90\x3c\x40' + meta_middle + b'\x00\x3c\x00' + EOT
        res = midi.parse_bytes(smf(track_meta))
        self.assertEqual(res['note_count'], 1)

        sysex_middle = b'\x00\xf7\x00'
        track_sysex = b'\x00\x90\x3c\x40' + sysex_middle + b'\x00\x3c\x00' + EOT
        with self.assertRaises(ValueError):
            midi.parse_bytes(smf(track_sysex))

    def test_every_truncation_rejected(self):
        data = smf(NOTE)
        for end in range(len(data)):
            with self.subTest(end=end), self.assertRaises(ValueError):
                midi.parse_bytes(data[:end])

    def test_uniform_pitch_classes_ambiguous(self):
        track = b''.join(b'\x00\x90' + bytes([pitch, 64])
                         + event(1, b'\x80' + bytes([pitch, 0])) for pitch in range(60, 72)) + EOT
        result = midi.parse_bytes(smf(track))
        self.assertEqual(result['pitch_class_histogram'], [1] * 12)
        self.assertTrue(result['key_estimate']['ambiguous'])
        self.assertIsNone(result['key_estimate']['key'])

    def test_unknown_alien_chunks_ignored_per_spec(self):
        alien_data = b'TestCustomChunkData'
        alien_chunk = b'ALIN' + struct.pack('>I', len(alien_data)) + alien_data
        # Place unknown chunk before, between, and after valid MTrk chunks
        data = (b'MThd' + struct.pack('>IHHH', 6, 1, 2, 480)
                + alien_chunk
                + b'MTrk' + struct.pack('>I', len(NOTE)) + NOTE
                + alien_chunk
                + b'MTrk' + struct.pack('>I', len(NOTE)) + NOTE
                + alien_chunk)
        result = midi.parse_bytes(data)
        self.assertEqual(result['track_count'], 2)
        self.assertEqual(result['note_count'], 2)
        # But truncated or oversized alien chunk must raise ValueError
        bad_alien = b'ALIN' + struct.pack('>I', len(alien_data) + 10) + alien_data
        with self.assertRaises(ValueError):
            midi.parse_bytes(b'MThd' + struct.pack('>IHHH', 6, 1, 1, 480) + bad_alien)

    def test_bounds(self):
        with patch.object(midi, 'MAX_FILE_BYTES', 8), self.assertRaises(ValueError):
            midi.parse_bytes(smf(NOTE))
        with patch.object(midi, 'MAX_EVENTS', 1), self.assertRaises(ValueError):
            midi.parse_bytes(smf(NOTE))


if __name__ == '__main__':
    unittest.main()
