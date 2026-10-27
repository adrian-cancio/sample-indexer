"""Command line interface (stdlib argparse)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import db, reports
from .config import DEFAULT_WORKERS, default_db_path, default_root, normalize_root
from .scanner import Scanner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sample_indexer", description="Index a sample/MIDI library into SQLite."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="database path (default: $SAMPLE_INDEXER_DB or <repo>/data/index.sqlite)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create/migrate the database")

    p_scan = sub.add_parser("scan", help="reconcile the library with the index")
    p_scan.add_argument("--root", type=Path, default=None, help="library root (default: $SAMPLE_INDEXER_ROOT or the OMNi User Library)")
    p_scan.add_argument("--limit", type=int, default=None, help="stop after N discovered files (partial scan: never marks missing)")
    p_scan.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"hash threads (default {DEFAULT_WORKERS})")
    p_scan.add_argument("--batch-size", type=int, default=500, help="DB writes per commit (default 500)")

    sub.add_parser("stats", help="counts by ext/folder, missing, duplicates")

    p_ver = sub.add_parser("verify-manifest", help="compare the DB with a manifest CSV")
    p_ver.add_argument("csv", type=Path)
    p_ver.add_argument("--root", type=Path, default=None, help="restrict to one root")

    p_q = sub.add_parser("query", help="filter indexed paths")
    p_q.add_argument("--path", dest="path_contains", help="case-insensitive substring of the relative path")
    p_q.add_argument("--ext", help="extension, e.g. wav")
    p_q.add_argument("--folder", help="relative folder prefix, e.g. Samples/Kicks")
    p_q.add_argument("--include-missing", action="store_true")
    p_q.add_argument("--limit", type=int, default=50, help="max rows (0 = unlimited)")
    p_q.add_argument("--sha", action="store_true", help="print sha256 as first column")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = args.db or default_db_path()
    try:
        if args.command == "init":
            conn = db.open_db(db_path, create=True)
            print(f"database {db_path} at schema v{db.schema_version(conn)}")
            return 0

        if args.command == "scan":
            # scan creates/migrates implicitly: a missing DB should never block a first run.
            conn = db.open_db(db_path, create=True)
            root = args.root or default_root()
            print(f"db   {db_path}")
            scanner = Scanner(
                conn, root, workers=args.workers, limit=args.limit,
                batch_size=args.batch_size, progress=lambda m: print(m, file=sys.stderr),
            )
            report = scanner.run()
            print(report.format())
            if report.status == "interrupted":
                return 130
            return 1 if report.errors else 0

        conn = db.open_db(db_path, create=False)
        if args.command == "stats":
            print(reports.stats(conn))
            return 0
        if args.command == "verify-manifest":
            root = normalize_root(args.root) if args.root else None
            diff = reports.verify_manifest(conn, args.csv, root)
            print(diff.format())
            return 0 if diff.differences == 0 else 1
        if args.command == "query":
            rows = reports.query(
                conn, path_contains=args.path_contains, ext=args.ext, folder=args.folder,
                include_missing=args.include_missing, limit=args.limit,
            )
            for r in rows:
                flag = " [missing]" if r["missing"] else ""
                prefix = f"{r['sha256']}\t" if args.sha else ""
                print(f"{prefix}{r['rel_path']}\t{r['size']}{flag}")
            print(f"({len(rows)} rows)", file=sys.stderr)
            return 0
    except (db.DatabaseNotFound, db.SchemaTooNew, NotADirectoryError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
