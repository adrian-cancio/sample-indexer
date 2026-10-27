# sample-indexer

SQLite index for a techno/schranz sample + MIDI library, with incremental reconciliation.
Stdlib only (Python 3.14, sqlite3, hashlib, concurrent.futures, argparse). Do not add
third-party dependencies; the user installs system packages with dnf. All committed code,
comments and docs are in English.

## Hard rules

- The library (`~/Música/OMNi/User Library/`) is READ-ONLY: never modify, move or delete
  anything in it. Files are opened `O_RDONLY` only. Never log or store file contents
  (paths are fine).
- Tests must only use temp trees (`tempfile`), never the real library.
- Pilot/experimental runs use a scratch DB (`--db $TMPDIR/...`), not the default DB.

## Commands

```
python3 -m sample_indexer init
python3 -m sample_indexer scan [--root PATH] [--limit N] [--workers 8] [--batch-size 500]
python3 -m sample_indexer stats
python3 -m sample_indexer verify-manifest MANIFEST.csv
python3 -m sample_indexer query [--path SUBSTR] [--ext wav] [--folder Samples/Kicks] [--include-missing] [--limit 50] [--sha]
python3 -m unittest discover          # full test suite, exit code 0 expected
```

Global flag `--db PATH` (before the subcommand) or env `SAMPLE_INDEXER_DB` overrides the
DB (default `data/index.sqlite`, git-ignored). `SAMPLE_INDEXER_ROOT` overrides the default
library root. Exit codes: 0 ok, 1 differences/file errors, 2 usage/DB error, 130 interrupted.

## Layout

- `sample_indexer/migrations.py` – ordered `(version, sql)` list; append only, never edit.
- `sample_indexer/db.py` – connection (WAL), `migrate()`, `open_db()`.
- `sample_indexer/scanner.py` – reconcile algorithm (documented in the module docstring).
- `sample_indexer/reports.py` – `stats`, `query`, `verify_manifest`.
- `sample_indexer/cli.py` – argparse entrypoint.
- `tests/` – unittest.

## Design invariants

- Identity = SHA-256 of content (`resource`). Paths are attributes (`path`), relative to a stored `root`.
- Rows are never deleted: vanished paths get `missing=1`.
- Analysis columns in `resource` are nullable and filled by later stages (L3-3..L3-7), each
  tracked in `resource_stage(resource_id, stage, version, done_at, error)`. A schema change
  = new migration + bump of expected version in tests.
- Only the main thread writes to SQLite; worker threads only hash.
