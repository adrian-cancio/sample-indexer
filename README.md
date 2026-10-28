# sample-indexer

A SQLite index and technical analysis engine for sample and MIDI libraries. Built on Python standard library, system `ffprobe`/`ffmpeg` binaries, and `numpy` for digital signal processing (DSP).

The sample library is strictly **read-only**: files are opened `O_RDONLY` and never modified, moved, or deleted.

---

## Requirements

- **Python**: `>= 3.12`
- **System tools**: `ffmpeg` and `ffprobe` must be available in `PATH`.
  - Fedora: `sudo dnf install ffmpeg-free` (or RPM Fusion `ffmpeg`)
  - Ubuntu / Debian: `sudo apt install ffmpeg`
  - Arch Linux: `sudo pacman -S ffmpeg`
- **Python dependencies**: `numpy` (installed automatically via `pip`).

---

## Installation

Install directly from the source repository:

```bash
pip install .
```

Or for development (editable mode):

```bash
pip install -e .
```

Once installed, the CLI tool is available as `sample-indexer` (or via `python3 -m sample_indexer`).

---

## Quick start

```bash
# Initialize or migrate database
sample-indexer init

# Scan library: compute SHA-256 identity and track paths
sample-indexer scan --root /path/to/samples

# Probe audio containers with ffprobe and extract name/folder labels
sample-indexer metadata --workers 4

# Run full decode & low-end DSP (sub-bass, pitch f0, decay) on kick/bass candidates
sample-indexer lowend --workers 2

# Run full decode & tempo/meter/key DSP on loop candidates
sample-indexer loops --workers 2

# General streaming timbre, true peak, and integrated loudness (LUFS)
sample-indexer audio --workers 2

# Parse Standard MIDI Files (SMF/RMID) into musical note metrics
sample-indexer midi

# Check coverage, health, and library statistics
sample-indexer stats
sample-indexer coverage
sample-indexer midi-coverage

# Query samples by technical criteria
sample-indexer query --folder "Samples/Kicks" --ext wav --limit 20

# Create a safe, online SQLite backup
sample-indexer backup index.backup.sqlite
```

Global option `--db PATH` (placed before the subcommand) or the environment variable `SAMPLE_INDEXER_DB` selects the database location (default: `data/index.sqlite`).
`SAMPLE_INDEXER_ROOT` or `--root PATH` configures the library root.

Exit codes:
- `0`: success
- `1`: differences detected / file or probe errors
- `2`: usage error or database error
- `130`: interrupted (SIGINT / Ctrl-C)

---

## Analysis pipeline

| Stage | Subcommand | Reads | Writes |
|---|---|---|---|
| **Identity** | `scan` | File content (SHA-256), `stat` | `resource` (sha256, size, ext), `path` (rel_path, mtime) |
| **Probe** | `metadata` | Header via single `ffprobe` per audio file | duration, sample rate, channels, bit depth, codec |
| **Labels** | `metadata` | File path and name only | kind, category, bpm_name, key_name, label evidence |
| **Low-End** | `lowend` | Full decode via `ffmpeg -xerror` | `valid`/`valid_error`, peak, RMS, crest factor, bands, sub peak Hz, f0 pitch Hz, decay ms, stereo correlation |
| **Loops** | `loops` | Full decode via `ffmpeg -xerror` | `bpm_est`, `beats`, `bars_est`, `key_est`, confidence metrics, chroma analysis |
| **Audio** | `audio` | Complete streaming PCM | Integrated LUFS, true peak dBTP, spectral centroid, rolloff, flatness, structured descriptors |
| **MIDI** | `midi` | SMF / RMID events (stdlib parser) | Note counts, pitch bounds, note density, tempo, bars |

### Identity and reconciliation (`scan`)
- Identity is determined by the content SHA-256 (`resource`), while file paths are recorded as attributes (`path`).
- Paths with unchanged `(rel_path, size, mtime_ns)` are not re-hashed.
- Relocated or renamed files are recognized by matching their SHA-256 with vanished records.
- Deleted or missing files are marked with `missing=1`; database records are never hard-deleted.

### Container probing and heuristics (`metadata`)
- Fast, non-destructive header inspection using `ffprobe` captures duration, sample rate, bit depth, and codec.
- Heuristic classification extracts provisional `kind` (`one_shot`, `loop`, `midi`, `unknown`), `category`, and filename BPM/key hints.
- Audio validity is left `NULL` until a stage actually performs a conclusive full decode.

### Low-End & Kick analysis (`lowend`)
- Focuses on kick drums, 808s, and bass candidates.
- Measures fundamental pitch ($f_0$) via time-domain autocorrelation, sub-bass peak frequency via high-resolution zero-padded FFT, decay time ($T_{60}$ equivalent), and low-frequency stereo correlation ($< 150$ Hz).

### Loop & Tempo analysis (`loops`)
- Targets loop candidates (folder conventions or loop labels).
- Measures tempo via onset spectral flux autocorrelation across candidate BPM ranges.
- Estimates musical key using 12-bin chroma profiling correlated against Krumhansl-Schmuckler key profiles.

### Comprehensive timbre & loudness (`audio`)
- Streams full audio signals through chunked spectral analysis.
- Computes integrated loudness (LUFS) and true peak (dBTP) using EBU R128 metering standards.
- Evaluates spectral centroid, 85% spectral rolloff, and spectral flatness.

### MIDI parsing (`midi`)
- Pure Python Standard MIDI File (SMF format 0/1/2) and RMID parser without external dependencies.
- Calculates total notes, pitch range (`pitch_min`, `pitch_max`), note density (notes/beat), bar counts, and tempo events.

---

## Querying the index

### CLI examples

```bash
# 1. Kick one-shots with sub frequency peak in 45-65 Hz range
sample-indexer query --folder "Samples/Drums - One Shots/Kicks" --sub-peak-min 45 --sub-peak-max 65 --limit 10

# 2. Fast loops (>= 140 BPM) with measured tempo confidence >= 0.20
sample-indexer query --bpm-min 140 --bpm-confidence-min 0.20 --limit 10

# 3. Stereo loops with high mono compatibility in the low end (stereo_corr_low >= 0.85)
sample-indexer query --stereo-corr-low-min 0.85 --limit 10

# 4. Short percussion/hi-hat one-shots (duration <= 0.15s, sr=44100)
sample-indexer query --duration-max 0.15 --sr 44100 --limit 10

# 5. Long atmospheric textures (duration >= 10.0s)
sample-indexer query --duration-min 10.0 --limit 10

# 6. Dense MIDI files (density >= 6.0 notes/beat)
sample-indexer query --ext mid --midi-density-min 6.0 --limit 10

# 7. Long MIDI arrangements with 100+ note events
sample-indexer query --ext mid --midi-notes-min 100 --limit 10

# 8. Samples matching musical key 'F min' (from filename hint or measured chroma)
sample-indexer query --key "F min" --limit 10

# 9. Basslines with fundamental pitch f0 between 40 and 90 Hz
sample-indexer query --f0-min 40 --f0-max 90 --limit 10

# 10. Locate known corrupted or unparseable files
sample-indexer query --only-errors

# 11. Output structured JSON including technical audio descriptors
sample-indexer query --folder "Samples/Loops/Bass Loops" --bpm-min 130 --bpm-max 140 --json --limit 5
```

### Python API

`sample-indexer` can be imported directly into Python scripts:

```python
import sqlite3
from sample_indexer import query

conn = sqlite3.connect("data/index.sqlite")
conn.row_factory = sqlite3.Row

# Query kicks tuned between 48 Hz and 55 Hz
rows = query(
    conn,
    folder="Samples/Drums - One Shots/Kicks",
    sub_peak_min=48.0,
    sub_peak_max=55.0,
    limit=25,
)

for row in rows:
    print(f"{row['rel_path']} - {row['sub_peak_hz']:.1f} Hz")
```

---

## Safety and execution semantics

- **Resumable execution**: Each stage tracks progress in `resource_stage`. Interrupted runs (Ctrl-C / `--max-seconds`) commit finished checkpoints and exit cleanly.
- **Error handling**: Corrupted files or failed decodes record the failure status in the index and count as processed to prevent retry storms. Use `--retry-errors` to retry them or `--force` to reprocess.
- **Database backups**: Schema migrations automatically create timestamped SQLite backups before applying changes. Manual backups can be created at any time using `sample-indexer backup <dest>`.
- **Subprocess security**: All external calls to `ffmpeg` and `ffprobe` use explicit argument lists (`file:<path>`) without shell interpolation.

---

## Running tests

The test suite runs against temporary file trees and scratch SQLite databases:

```bash
python3 -m unittest discover
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
