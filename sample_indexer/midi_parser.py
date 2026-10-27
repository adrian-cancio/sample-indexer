"""Bounded stdlib SMF/RMID parsing and compact, JSON-safe musical summaries.

Durations follow tempo events, ignoring sustain/controllers (key-down polyphony).
Unclosed notes are clipped at EOT and reported. Pitch classes use melodic note
onsets, not duration; the profile estimate is heuristic, never a ground truth key.
Format 2 exposes independent track summaries and no aggregate timeline or key.
"""
from collections import Counter, defaultdict, deque
import math
import struct

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_EVENTS = 1_000_000
MAX_TRACKS = 4096


def parse_file(path):
    """Read at most the size limit plus one byte, read-only; raise ValueError."""
    with open(path, 'rb') as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    return parse_bytes(data)


def _vlq(data, pos):
    value = 0
    for _ in range(4):
        if pos >= len(data):
            raise ValueError('Truncated VLQ')
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 127)
        if not byte & 128:
            return value, pos
    raise ValueError('VLQ exceeds four bytes')


def _track(data, budget):
    pos = tick = running = count = percussion = unmatched = 0
    active = defaultdict(deque)
    notes, tempos, meters = [], [], []
    ended = False
    while pos < len(data):
        budget[0] += 1
        if budget[0] > MAX_EVENTS:
            raise ValueError('MIDI event limit exceeded')
        delta, pos = _vlq(data, pos)
        tick += delta
        if pos >= len(data):
            raise ValueError('Missing event status')
        status = data[pos]
        if status & 128:
            pos += 1
        elif running:
            status = running
        else:
            raise ValueError('Running status without channel status')
        if 0x80 <= status <= 0xef:
            running = status
            size = 1 if status >> 4 in (12, 13) else 2
            payload = data[pos:pos + size]
            if len(payload) != size or any(x > 127 for x in payload):
                raise ValueError('Invalid channel event data')
            pos += size
            kind, channel = status >> 4, status & 15
            if kind == 9 and payload[1]:
                count += 1
                percussion += channel == 9
                active[channel, payload[0]].append((tick, payload[1]))
            elif kind == 8 or (kind == 9 and not payload[1]):
                queue = active[channel, payload[0]]
                if queue:
                    start, velocity = queue.popleft()
                    notes.append((start, tick, payload[0], velocity, channel))
                else:
                    unmatched += 1
        elif status in (0xf0, 0xf7, 0xff):
            if status in (0xf0, 0xf7):
                running = 0
            if status == 0xff:
                if pos >= len(data) or data[pos] > 127:
                    raise ValueError('Invalid meta type')
                meta = data[pos]
                pos += 1
            length, pos = _vlq(data, pos)
            end = pos + length
            if end > len(data):
                raise ValueError('Truncated meta/SysEx payload')
            payload = data[pos:end]
            pos = end
            if status == 0xff:
                expected = {0x2f: 0, 0x51: 3, 0x58: 4, 0x59: 2, 0x54: 5, 0x20: 1, 0x21: 1, 0x00: 2}
                if meta in expected and length != expected[meta]:
                    raise ValueError('Invalid fixed-length meta event')
                if meta == 0x2f:
                    if pos != len(data):
                        raise ValueError('Bytes after end-of-track')
                    ended = True
                    break
                if meta == 0x51:
                    tempo = int.from_bytes(payload, 'big')
                    if not tempo:
                        raise ValueError('Zero tempo')
                    tempos.append((tick, tempo))
                if meta == 0x58:
                    if not payload[0] or payload[1] > 30:
                        raise ValueError('Invalid time signature')
                    meters.append((tick, payload[0], 1 << payload[1]))
        else:
            raise ValueError('Unsupported system status in SMF')
    if not ended:
        raise ValueError('Missing end-of-track')
    unclosed = 0
    for (channel, pitch), queue in active.items():
        for start, velocity in queue:
            unclosed += 1
            notes.append((start, tick, pitch, velocity, channel))
    return dict(ticks=tick, notes=notes, tempos=tempos, meters=meters,
                note_count=count, percussion_note_count=percussion,
                unclosed_note_count=unclosed, unmatched_note_off_count=unmatched)


def _map(events, default):
    by_tick = {0: default}
    for tick, *values in sorted(events, key=lambda x: x[0]):
        by_tick[tick] = tuple(values)
    return sorted(by_tick.items())


def _key(hist):
    """Pearson correlation with Krumhansl profiles, explicit sparse ambiguity."""
    profiles = [('major', (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)),
                ('minor', (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17))]
    mean = sum(hist) / 12
    centered = [x - mean for x in hist]
    norm = math.sqrt(sum(x*x for x in centered))
    if not norm:
        return dict(key=None, score=None, margin=None, ambiguous=True,
                    method='krumhansl_onset_pearson')
    scores = []
    names = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
    for mode, profile in profiles:
        pm = sum(profile) / 12
        pn = math.sqrt(sum((v-pm)**2 for v in profile))
        for root in range(12):
            score = sum(centered[i] * (profile[(i-root) % 12]-pm) for i in range(12)) / (norm*pn)
            scores.append((score, names[root] + ' ' + mode))
    scores.sort(reverse=True)
    score, candidate = scores[0]
    margin = score - scores[1][0]
    ambiguous = sum(x > 0 for x in hist) < 3 or margin < .08 or score < .5
    return dict(key=None if ambiguous else candidate, candidate=candidate,
                score=score, margin=margin, ambiguous=ambiguous,
                method='krumhansl_onset_pearson')


def _summary(tracks, ppq, tempos, meters):
    ticks = max(t['ticks'] for t in tracks)
    tempo_map = _map(tempos, (500000,))
    meter_map = _map(meters, (4, 4))
    seconds = 0.
    for i, (tick, (tempo,)) in enumerate(tempo_map):
        end = tempo_map[i+1][0] if i+1 < len(tempo_map) else ticks
        seconds += max(0, min(end, ticks)-tick) * tempo / ppq / 1_000_000
    bars = 0.
    for i, (tick, (num, den)) in enumerate(meter_map):
        end = meter_map[i+1][0] if i+1 < len(meter_map) else ticks
        ticks_per_bar = 4 * ppq * num / den
        if ticks_per_bar > 0:
            bars += max(0, min(end, ticks) - tick) / ticks_per_bar
    notes = [n for t in tracks for n in t['notes']]
    histogram = [0] * 12
    changes = Counter()
    for start, end, pitch, velocity, channel in notes:
        if channel != 9:
            histogram[pitch % 12] += 1
        if end > start:
            changes[start] += 1
            changes[end] -= 1
    poly = maximum = 0
    for tick in sorted(changes):
        poly += changes[tick]
        maximum = max(maximum, poly)
    pitches = [n[2] for n in notes]
    velocities = [n[3] for n in notes]
    beats = ticks / ppq
    result = {name: sum(t[name] for t in tracks) for name in
              ('note_count', 'percussion_note_count', 'unclosed_note_count', 'unmatched_note_off_count')}
    result.update(ticks=ticks, beats=beats, duration_seconds=seconds, bars=bars,
                  bars_basis='integrated_meter_map',
                  tempo_map=[dict(tick=t, microseconds_per_quarter=v[0], bpm=60_000_000/v[0]) for t, v in tempo_map],
                  meter_map=[dict(tick=t, numerator=v[0], denominator=v[1]) for t, v in meter_map],
                  pitch_min=min(pitches) if pitches else None,
                  pitch_max=max(pitches) if pitches else None,
                  velocity_min=min(velocities) if velocities else None,
                  velocity_max=max(velocities) if velocities else None,
                  velocity_mean=sum(velocities)/len(velocities) if velocities else None,
                  polyphony_max=maximum, notes_per_beat=len(notes)/beats if beats else None,
                  pitch_class_histogram=histogram, key_estimate=_key(histogram))
    return result


def parse_bytes(data):
    """Parse complete SMF 0/1/2 or RIFF RMID bytes; malformed input raises ValueError.

    Maps include explicit MIDI defaults at tick zero; simultaneous changes use
    stable track/event order (last wins). SMPTE division is explicitly unsupported.
    No per-note arrays escape this API. Limits apply across all tracks.
    """
    if len(data) > MAX_FILE_BYTES:
        raise ValueError('MIDI file size limit exceeded')
    if data[:4] == b'RIFF':
        if len(data) < 12 or data[8:12] != b'RMID' or int.from_bytes(data[4:8], 'little') != len(data)-8:
            raise ValueError('Invalid RIFF RMID length/type')
        pos, embedded = 12, None
        while pos < len(data):
            if pos + 8 > len(data):
                raise ValueError('Truncated RIFF chunk')
            kind = data[pos:pos+4]
            length = int.from_bytes(data[pos+4:pos+8], 'little')
            start = pos + 8
            pos = start + length + (length & 1)
            if pos > len(data):
                raise ValueError('Truncated RIFF chunk/padding')
            if kind == b'data':
                if embedded is not None:
                    raise ValueError('Multiple RMID data chunks')
                embedded = data[start:start+length]
        if embedded is None:
            raise ValueError('Missing RMID data chunk')
        data = embedded
    if len(data) < 14 or data[:4] != b'MThd':
        raise ValueError('Missing SMF header')
    length = int.from_bytes(data[4:8], 'big')
    if length < 6 or 8 + length > len(data):
        raise ValueError('Invalid SMF header length')
    fmt, count, division = struct.unpack('>HHH', data[8:14])
    if fmt not in (0, 1, 2) or not 1 <= count <= MAX_TRACKS or (fmt == 0 and count != 1):
        raise ValueError('Invalid SMF format/track count')
    if division & 0x8000:
        raise ValueError('SMPTE time division is unsupported')
    if not division:
        raise ValueError('Zero PPQ division')
    pos, tracks, budget = 8 + length, [], [0]
    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError('Truncated SMF chunk header')
        kind = data[pos:pos+4]
        length = int.from_bytes(data[pos+4:pos+8], 'big')
        pos += 8
        if pos + length > len(data):
            raise ValueError('Invalid SMF chunk length')
        # SMF 1.1 requires readers to ignore alien chunks by their length.
        if kind == b'MTrk':
            if len(tracks) >= count:
                raise ValueError('Invalid track length/count')
            tracks.append(_track(data[pos:pos+length], budget))
        pos += length
    if len(tracks) != count:
        raise ValueError('Track count mismatch')
    tempos = [e for t in tracks for e in t['tempos']]
    meters = [e for t in tracks for e in t['meters']]
    # Synchronous formats expose one global map, not a copy for every track.
    summaries = ([_summary([t], division, t['tempos'], t['meters']) for t in tracks]
                 if fmt == 2 else
                 [dict(ticks=t['ticks'], note_count=t['note_count'],
                       percussion_note_count=t['percussion_note_count'],
                       unclosed_note_count=t['unclosed_note_count'],
                       unmatched_note_off_count=t['unmatched_note_off_count'])
                  for t in tracks])
    result = _summary(tracks, division, tempos, meters)
    if fmt == 2:
        for name in ('ticks', 'beats', 'duration_seconds', 'polyphony_max', 'notes_per_beat'):
            result[name] = None
        result['tempo_map'] = result['meter_map'] = None
        result['key_estimate'] = dict(key=None, score=None, margin=None, ambiguous=True,
                                      reason='format_2_independent_sequences')
    result['timing_policy'] = 'tempo conflicts at same tick: last track/event wins; sustain ignored'
    result['tempo_conflicts'] = len(tempos) - len({tick for tick, *_ in tempos})
    result['tempo_provenance'] = ('smf_default_120_bpm_until_first_event' if not any(tick == 0 for tick, *_ in tempos) else 'explicit_events')
    result['meter_provenance'] = ('smf_default_4_4_until_first_event' if not any(tick == 0 for tick, *_ in meters) else 'explicit_events')
    result['useful_note_content'] = bool(result['note_count'])
    result.update(format=fmt, ppq=division, track_count=count,
                  coherent_timeline=fmt != 2, tracks=summaries, event_count=budget[0])
    return result
