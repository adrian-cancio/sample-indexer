# sample-indexer

SQLite index for a techno/schranz sample + MIDI library, with incremental reconciliation.
Stdlib (Python 3.14, sqlite3, hashlib, concurrent.futures, argparse, subprocess) plus the
system `ffprobe`/`ffmpeg` binaries (L3-3 probe, L3-4 through L3-6 decode) and the already-installed `numpy`
(L3-4 through L3-6 DSP only). Do not add other third-party dependencies; the user installs system packages
with dnf. All committed code, comments and docs are in
English. User-facing docs: `README.md`; reports per phase: `reports/`.

## Hard rules

- The library (`~/Música/OMNi/User Library/`) is READ-ONLY: never modify, move or delete
  anything in it. Files are opened `O_RDONLY` only (ffprobe only reads). Never log or store
  file contents (paths are fine).
- Tests must only use temp trees (`tempfile`), never the real library.
- Pilot/experimental runs use a scratch DB (`--db $TMPDIR/...`), not the default DB.
- Back up the default DB (`backup DEST`, SQLite backup API) before a migration or a full
  processing run. `init`/`scan` also auto-copy to `<db>.bak-v<old>-<stamp>` before migrating.
- No shell interpolation of file names: subprocesses use argv lists (`file:<path>`).

## Commands

```
python3 -m sample_indexer init
python3 -m sample_indexer scan [--root PATH] [--limit N] [--workers 8] [--batch-size 500]
python3 -m sample_indexer metadata [--limit N] [--workers 4] [--timeout 30] [--batch-size 200] [--max-seconds S] [--force] [--retry-errors] [--stratify [--seed S]]
python3 -m sample_indexer lowend [--limit N] [--workers 2] [--timeout 60] [--batch-size 50] [--max-seconds S] [--max-duration 120] [--max-decode-mib 128] [--force] [--retry-errors] [--stratify [--seed S]]
python3 -m sample_indexer loops [--limit N] [--workers 2] [--timeout 60] [--batch-size 50] [--max-seconds S] [--max-duration 300] [--max-decode-mib 256] [--bpm-min 60 --bpm-max 200 --min-peak 0.15 --min-prominence 0.10] [--force] [--retry-errors] [--stratify [--seed S]]
python3 -m sample_indexer audio [--limit N] [--workers 2] [--timeout 60] [--batch-size 50] [--max-seconds S] [--force] [--retry-errors] [--stratify [--seed S]]
python3 -m sample_indexer midi [--limit N] [--force] [--retry-errors] [--stratify [--seed S]]
python3 -m sample_indexer midi-coverage
python3 -m sample_indexer coverage
python3 -m sample_indexer stats
python3 -m sample_indexer backup DEST
python3 -m sample_indexer verify-manifest MANIFEST.csv
python3 -m sample_indexer query [--path SUBSTR] [--ext wav] [--folder Samples/Kicks] [--include-missing] [--limit 50] [--sha]
python3 -m unittest discover          # full test suite, exit code 0 expected
```

Global flag `--db PATH` (before the subcommand) or env `SAMPLE_INDEXER_DB` overrides the
DB (default `data/index.sqlite`, git-ignored). `SAMPLE_INDEXER_ROOT` overrides the default
library root. Exit codes: 0 ok, 1 differences/file or probe errors, 2 usage/DB error, 130 interrupted.

## Layout

- `sample_indexer/migrations.py` – ordered `(version, sql)` list; append only, never edit.
- `sample_indexer/db.py` – connection (WAL), `migrate()`, `open_db()`, `backup_db()`.
- `sample_indexer/scanner.py` – reconcile algorithm (documented in the module docstring).
- `sample_indexer/metadata.py` – L3-3 stage runner: ffprobe probe + label stage (docstring).
- `sample_indexer/labels.py` – pure path/name interpretation (kind, category, bpm/key tokens).
- `sample_indexer/lowend.py` – L3-4 stage runner: candidate selection, ffmpeg full decode (`-xerror`),
  writes conclusive full-decode `valid`/`valid_error` (also owned by loops/audio).
- `sample_indexer/lowend_dsp.py` – L3-4 pure numpy measurements (`analyze(samples, sr)`).
- `sample_indexer/loops.py` – L3-5 stage runner: loop candidate selection, reuses `lowend.ffmpeg_decode`,
  fills `bpm_est`/`beats`/`key_est` + evidence columns, never erases earlier conclusive values (docstring).
- `sample_indexer/loops_dsp.py` – L3-5 pure numpy tempo / length / key measurements (`analyze(samples, sr)`).
- `sample_indexer/audio.py` – L3-6 general streaming full-decode/loudness runner; method-isolated `audio_measurement` descriptors.
- `sample_indexer/audio_dsp.py` – bounded streaming time-domain and chunked spectral measurements.
- `sample_indexer/midi.py` – L3-7 sequential resumable event parsing; MIDI-only columns and isolated provenance.
- `sample_indexer/midi_parser.py` – bounded stdlib SMF/RMID parser; format-2 timelines independent, SMPTE explicitly unsupported.
- `sample_indexer/reports.py` – `stats`, `metadata_coverage`, `query`, `verify_manifest`.
- `sample_indexer/cli.py` – argparse entrypoint.
- `tests/` – unittest (`test_scanner`, `test_labels`, `test_metadata`, `test_lowend`, `test_lowend_dsp`, `test_loops`, `test_loops_dsp`).

## Design invariants

- Identity = SHA-256 of content (`resource`). Paths are attributes (`path`), relative to a stored `root`.
- Every new/relocated path is hashed; (size, mtime) only nominates move candidates and never
  decides identity. Unchanged (rel_path, size, mtime) paths are not re-hashed (documented limitation).
- Rows are never deleted: vanished paths get `missing=1`.
- Analysis columns in `resource` are nullable and filled by later stages (L3-3..L3-7), each
  tracked in `resource_stage(resource_id, stage, version, done_at, error, input_sig)`. A schema
  change = new migration + bump of expected version in tests. Errors recorded at the current
  version count as processed (rerun only with `--retry-errors`/`--force`).
- Only the main thread writes to SQLite; worker threads hash or run bounded ffprobe/ffmpeg subprocesses and DSP.
- The probe is HEADER/CONTAINER scope: it never sets `valid`/`valid_error` (NULL = unknown, not
  invalid) – only a later full-decode stage may. MIDI is never given to ffprobe.
- `kind`/`category`/`bpm_name`/`key_name` from L3-3 are provisional path/name hints with stored
  evidence (`label_evidence`), `label_basis='path_name_provisional'`; later DSP stages own
  `*_est`, `beats`, spectral and loudness columns and must not be pre-filled by guesses.
  `beats_hint` is derived from `bpm_name`, never a measurement.
