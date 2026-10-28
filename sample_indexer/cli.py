"""Command line interface (stdlib argparse)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db, reports
from .config import DEFAULT_WORKERS, default_db_path, default_root, normalize_root
from .lowend import (
    DEFAULT_DECODE_TIMEOUT_S, DEFAULT_LOWEND_BATCH, DEFAULT_LOWEND_WORKERS,
    DEFAULT_MAX_DECODE_BYTES, DEFAULT_MAX_DURATION_S, LowendStage,
)
from . import loops as loops_stage
from .metadata import DEFAULT_BATCH, DEFAULT_META_WORKERS, DEFAULT_TIMEOUT_S, MetadataStage
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

    p_meta = sub.add_parser(
        "metadata", help="ffprobe header metadata + provisional path/name labels (resumable)"
    )
    p_meta.add_argument("--limit", type=int, default=None, help="process at most N resources this run")
    p_meta.add_argument("--workers", type=int, default=DEFAULT_META_WORKERS, help=f"concurrent ffprobe processes (default {DEFAULT_META_WORKERS})")
    p_meta.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help=f"per-ffprobe timeout in seconds (default {DEFAULT_TIMEOUT_S:g})")
    p_meta.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help=f"results per checkpoint commit (default {DEFAULT_BATCH})")
    p_meta.add_argument("--max-seconds", type=float, default=None, help="stop cleanly (checkpointed) after this walltime")
    p_meta.add_argument("--force", action="store_true", help="redo selected resources even if complete at the current version")
    p_meta.add_argument("--retry-errors", action="store_true", help="only re-probe resources whose recorded probe error is at the current version")
    p_meta.add_argument("--stratify", action="store_true", help="select round-robin across folder x extension strata instead of by id (pilots)")
    p_meta.add_argument("--seed", type=int, default=0, help="stratified-selection seed (default 0)")

    p_low = sub.add_parser(
        "lowend", help="full ffmpeg decode + low-end DSP for kick/rumble/bass candidates (resumable)"
    )
    p_low.add_argument("--limit", type=int, default=None, help="process at most N resources this run")
    p_low.add_argument("--workers", type=int, default=DEFAULT_LOWEND_WORKERS, help=f"concurrent ffmpeg processes (default {DEFAULT_LOWEND_WORKERS})")
    p_low.add_argument("--timeout", type=float, default=DEFAULT_DECODE_TIMEOUT_S, help=f"per-file ffmpeg timeout in seconds (default {DEFAULT_DECODE_TIMEOUT_S:g})")
    p_low.add_argument("--batch-size", type=int, default=DEFAULT_LOWEND_BATCH, help=f"results per checkpoint commit (default {DEFAULT_LOWEND_BATCH})")
    p_low.add_argument("--max-seconds", type=float, default=None, help="stop cleanly (checkpointed) after this walltime")
    p_low.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION_S, help=f"files longer than this many seconds are recorded unresolved, never truncated (default {DEFAULT_MAX_DURATION_S:g})")
    p_low.add_argument("--max-decode-mib", type=int, default=DEFAULT_MAX_DECODE_BYTES >> 20, help=f"per-file float64 analysis footprint cap in MiB (default {DEFAULT_MAX_DECODE_BYTES >> 20})")
    p_low.add_argument("--force", action="store_true", help="redo selected resources even if complete at the current version")
    p_low.add_argument("--retry-errors", action="store_true", help="only redo resources whose recorded lowend error is at the current version")
    p_low.add_argument("--stratify", action="store_true", help="select round-robin across folder x extension strata instead of by id (pilots)")
    p_low.add_argument("--seed", type=int, default=0, help="stratified-selection seed (default 0)")

    p_loop = sub.add_parser(
        "loops", help="full ffmpeg decode + tempo/beats/bars/key DSP for loop candidates (resumable)"
    )
    p_loop.add_argument("--limit", type=int, default=None, help="process at most N resources this run")
    p_loop.add_argument("--workers", type=int, default=loops_stage.DEFAULT_LOOPS_WORKERS, help=f"concurrent ffmpeg+DSP workers (default {loops_stage.DEFAULT_LOOPS_WORKERS})")
    p_loop.add_argument("--timeout", type=float, default=DEFAULT_DECODE_TIMEOUT_S, help=f"per-file ffmpeg timeout in seconds (default {DEFAULT_DECODE_TIMEOUT_S:g})")
    p_loop.add_argument("--batch-size", type=int, default=loops_stage.DEFAULT_LOOPS_BATCH, help=f"results per checkpoint commit (default {loops_stage.DEFAULT_LOOPS_BATCH})")
    p_loop.add_argument("--max-seconds", type=float, default=None, help="stop cleanly (checkpointed) after this walltime")
    p_loop.add_argument("--max-duration", type=float, default=loops_stage.DEFAULT_MAX_DURATION_S, help=f"files longer than this many seconds are recorded unresolved_cap, never truncated (default {loops_stage.DEFAULT_MAX_DURATION_S:g})")
    p_loop.add_argument("--max-decode-mib", type=int, default=loops_stage.DEFAULT_MAX_DECODE_BYTES >> 20, help=f"per-file float32 PCM cap in MiB (default {loops_stage.DEFAULT_MAX_DECODE_BYTES >> 20})")
    p_loop.add_argument("--bpm-min", type=float, default=None, help="tempo candidate range low bound (default 60)")
    p_loop.add_argument("--bpm-max", type=float, default=None, help="tempo candidate range high bound (default 200)")
    p_loop.add_argument("--min-peak", type=float, default=None, help="min autocorrelation height of the best tempo peak (default 0.15)")
    p_loop.add_argument("--min-prominence", type=float, default=None, help="min prominence of the best tempo peak (default 0.10)")
    p_loop.add_argument("--force", action="store_true", help="redo selected resources even if complete at the current version/settings")
    p_loop.add_argument("--retry-errors", action="store_true", help="only redo resources whose recorded loops error is at the current version")
    p_loop.add_argument("--stratify", action="store_true", help="select round-robin across folder x bpm_name strata instead of by id (pilots)")
    p_loop.add_argument("--seed", type=int, default=0, help="stratified-selection seed (default 0)")

    p_audio = sub.add_parser("audio", help="streaming full-decode timbre and loudness (resumable)")
    p_audio.add_argument("--limit", type=int, default=None)
    p_audio.add_argument("--workers", type=int, default=2)
    p_audio.add_argument("--timeout", type=float, default=60.0)
    p_audio.add_argument("--batch-size", type=int, default=50)
    p_audio.add_argument("--max-seconds", type=float, default=None)
    p_audio.add_argument("--force", action="store_true")
    p_audio.add_argument("--retry-errors", action="store_true")
    p_audio.add_argument("--stratify", action="store_true")
    p_audio.add_argument("--seed", type=int, default=0)

    p_midi = sub.add_parser("midi", help="bounded stdlib MIDI event parsing (resumable)")
    p_midi.add_argument("--limit", type=int, default=None)
    p_midi.add_argument("--force", action="store_true")
    p_midi.add_argument("--retry-errors", action="store_true")
    p_midi.add_argument("--stratify", action="store_true")
    p_midi.add_argument("--seed", type=int, default=0)
    sub.add_parser("midi-coverage", help="MIDI coverage and statuses")
    sub.add_parser("coverage", help="stage coverage and metadata distributions")

    p_bak = sub.add_parser("backup", help="consistent SQLite backup of the database")
    p_bak.add_argument("dest", type=Path, help="backup file to create (must not exist)")

    p_ver = sub.add_parser("verify-manifest", help="compare the DB with a manifest CSV")
    p_ver.add_argument("csv", type=Path)
    p_ver.add_argument("--root", type=Path, default=None, help="restrict to one root")

    p_q = sub.add_parser("query", help="filter indexed paths")
    p_q.add_argument("--path", dest="path_contains", help="case-insensitive substring of the relative path")
    p_q.add_argument("--ext", help="extension, e.g. wav")
    p_q.add_argument("--folder", help="relative folder prefix, e.g. Samples/Kicks")
    p_q.add_argument("--only-errors", action="store_true", help="filter only known invalid/corrupted files")
    p_q.add_argument("--include-missing", action="store_true")
    p_q.add_argument("--include-errors", action="store_true", help="include known invalid/corrupted files")
    p_q.add_argument("--duration-min", type=float, help="minimum duration in seconds")
    p_q.add_argument("--duration-max", type=float, help="maximum duration in seconds")
    p_q.add_argument("--sr", type=int, help="exact sample rate in Hz")
    p_q.add_argument("--channels", type=int, help="exact channel count")
    p_q.add_argument("--f0-min", type=float, help="minimum fundamental frequency in Hz")
    p_q.add_argument("--f0-max", type=float, help="maximum fundamental frequency in Hz")
    p_q.add_argument("--sub-peak-min", type=float, help="minimum sub peak frequency in Hz")
    p_q.add_argument("--sub-peak-max", type=float, help="maximum sub peak frequency in Hz")
    p_q.add_argument("--decay-min", type=float, help="minimum decay in ms")
    p_q.add_argument("--decay-max", type=float, help="maximum decay in ms")
    p_q.add_argument("--stereo-corr-low-min", type=float, help="minimum low-end stereo correlation")
    p_q.add_argument("--stereo-corr-low-max", type=float, help="maximum low-end stereo correlation")
    p_q.add_argument("--lufs-min", type=float, help="minimum integrated LUFS")
    p_q.add_argument("--lufs-max", type=float, help="maximum integrated LUFS")
    p_q.add_argument("--true-peak-min", type=float, help="minimum true peak in dBTP")
    p_q.add_argument("--true-peak-max", type=float, help="maximum true peak in dBTP")
    p_q.add_argument("--centroid-min", type=float, help="minimum spectral centroid in Hz")
    p_q.add_argument("--centroid-max", type=float, help="maximum spectral centroid in Hz")
    p_q.add_argument("--rolloff-min", type=float, help="minimum spectral rolloff 85 percent in Hz")
    p_q.add_argument("--rolloff-max", type=float, help="maximum spectral rolloff 85 percent in Hz")
    p_q.add_argument("--flatness-min", type=float, help="minimum spectral flatness")
    p_q.add_argument("--flatness-max", type=float, help="maximum spectral flatness")
    p_q.add_argument("--bpm-min", type=float, help="minimum tempo BPM")
    p_q.add_argument("--bpm-max", type=float, help="maximum tempo BPM")
    p_q.add_argument("--bpm-provenance", choices=["est", "name", "either"], default="either", help="tempo provenance")
    p_q.add_argument("--bpm-confidence-min", type=float, help="minimum tempo confidence")
    p_q.add_argument("--key", help="musical key (e.g. 'F min' or 'C')")
    p_q.add_argument("--key-provenance", choices=["est", "name", "either"], default="either", help="key provenance")
    p_q.add_argument("--midi-notes-min", type=int, help="minimum MIDI note count")
    p_q.add_argument("--midi-notes-max", type=int, help="maximum MIDI note count")
    p_q.add_argument("--pitch-min", type=int, help="minimum MIDI pitch number")
    p_q.add_argument("--pitch-max", type=int, help="maximum MIDI pitch number")
    p_q.add_argument("--midi-density-min", type=float, help="minimum MIDI note density (notes/beat)")
    p_q.add_argument("--midi-density-max", type=float, help="maximum MIDI note density (notes/beat)")
    p_q.add_argument("--limit", type=int, default=50, help="max rows (0 = unlimited)")
    p_q.add_argument("--sha", action="store_true", help="print sha256 as first column")
    p_q.add_argument("--json", action="store_true", help="output query results in JSON format")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    opened: list = []
    try:
        return _dispatch(args, opened)
    finally:
        for conn in opened:
            conn.close()


def _dispatch(args: argparse.Namespace, opened: list) -> int:
    db_path = args.db or default_db_path()

    def open_db(*, create: bool):
        conn = db.open_db(db_path, create=create)
        opened.append(conn)
        return conn

    try:
        if args.command == "init":
            conn = open_db(create=True)
            print(f"database {db_path} at schema v{db.schema_version(conn)}")
            return 0

        if args.command == "scan":
            # scan creates/migrates implicitly: a missing DB should never block a first run.
            conn = open_db(create=True)
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

        if args.command == "backup":
            # works on any schema version (needed BEFORE a migration), never migrates
            conn = db.connect(db_path, create=False)
            opened.append(conn)
            print(f"backup {db.backup_db(conn, args.dest)}")
            return 0

        conn = open_db(create=False)
        if args.command == "midi":
            from .midi import MidiStage
            report = MidiStage(conn, limit=args.limit, force=args.force,
                               retry_errors=args.retry_errors, stratify=args.stratify,
                               seed=args.seed).run()
            print(report.format())
            return 130 if report.status == "interrupted" else (1 if report.errors else 0)
        if args.command == "midi-coverage":
            from .midi import coverage
            print(coverage(conn))
            return 0
        if args.command == "stats":
            print(reports.stats(conn))
            return 0
        if args.command == "coverage":
            print(reports.metadata_coverage(conn))
            return 0
        if args.command == "metadata":
            stage = MetadataStage(
                conn, workers=args.workers, limit=args.limit, force=args.force,
                retry_errors=args.retry_errors, stratify=args.stratify, seed=args.seed,
                timeout=args.timeout, batch_size=args.batch_size, max_seconds=args.max_seconds,
                progress=lambda m: print(m, file=sys.stderr),
            )
            print(f"db   {db_path}")
            report = stage.run()
            print(report.format())
            if report.status == "interrupted":
                return 130 if report.stop_reason == "keyboard interrupt" else 0
            return 1 if report.probe_errors else 0
        if args.command == "lowend":
            stage = LowendStage(
                conn, workers=args.workers, limit=args.limit, force=args.force,
                retry_errors=args.retry_errors, stratify=args.stratify, seed=args.seed,
                timeout=args.timeout, batch_size=args.batch_size, max_seconds=args.max_seconds,
                max_duration_s=args.max_duration, max_decode_bytes=args.max_decode_mib << 20,
                progress=lambda m: print(m, file=sys.stderr),
            )
            print(f"db   {db_path}")
            report = stage.run()
            print(report.format())
            if report.status == "interrupted":
                return 130 if report.stop_reason == "keyboard interrupt" else 0
            return 1 if (report.statuses.get("decode_failed") or report.unresolved) else 0
        if args.command == "loops":
            overrides = {k: v for k, v in (
                ("bpm_min", args.bpm_min), ("bpm_max", args.bpm_max),
                ("tempo_min_peak", args.min_peak), ("tempo_min_prominence", args.min_prominence)) if v is not None}
            stage = loops_stage.LoopsStage(
                conn, workers=args.workers, limit=args.limit, force=args.force,
                retry_errors=args.retry_errors, stratify=args.stratify, seed=args.seed,
                timeout=args.timeout, batch_size=args.batch_size, max_seconds=args.max_seconds,
                max_duration_s=args.max_duration, max_decode_bytes=args.max_decode_mib << 20,
                overrides=overrides, progress=lambda m: print(m, file=sys.stderr),
            )
            print(f"db   {db_path}")
            report = stage.run()
            print(report.format())
            if report.status == "interrupted":
                return 130 if report.stop_reason == "keyboard interrupt" else 0
            return 1 if (report.statuses.get("decode_failed") or report.unresolved) else 0
        if args.command == "audio":
            from .audio import AudioStage
            stage = AudioStage(
                conn, workers=args.workers, limit=args.limit, force=args.force,
                retry_errors=args.retry_errors, stratify=args.stratify, seed=args.seed,
                timeout=args.timeout, batch_size=args.batch_size, max_seconds=args.max_seconds,
                progress=lambda m: print(m, file=sys.stderr),
            )
            print(f"db   {db_path}")
            report = stage.run()
            print(report.format())
            if report.status == "interrupted":
                return 130 if report.stop_reason == "keyboard interrupt" else 0
            return 1 if (report.statuses.get("decode_failed") or report.unresolved) else 0
        if args.command == "verify-manifest":
            root = normalize_root(args.root) if args.root else None
            diff = reports.verify_manifest(conn, args.csv, root)
            print(diff.format())
            return 0 if diff.differences == 0 else 1
        if args.command == "query":
            rows = reports.query(
                conn, path_contains=args.path_contains, ext=args.ext, folder=args.folder,
                include_missing=args.include_missing, include_errors=args.include_errors,
                only_errors=args.only_errors,
                duration_min=args.duration_min, duration_max=args.duration_max,
                sr=args.sr, channels=args.channels,
                f0_min=args.f0_min, f0_max=args.f0_max,
                sub_peak_min=args.sub_peak_min, sub_peak_max=args.sub_peak_max,
                decay_min=args.decay_min, decay_max=args.decay_max,
                stereo_corr_low_min=args.stereo_corr_low_min,
                stereo_corr_low_max=args.stereo_corr_low_max,
                lufs_min=args.lufs_min, lufs_max=args.lufs_max,
                true_peak_min=args.true_peak_min, true_peak_max=args.true_peak_max,
                centroid_min=args.centroid_min, centroid_max=args.centroid_max,
                rolloff_min=args.rolloff_min, rolloff_max=args.rolloff_max,
                flatness_min=args.flatness_min, flatness_max=args.flatness_max,
                bpm_min=args.bpm_min, bpm_max=args.bpm_max,
                bpm_provenance=args.bpm_provenance,
                bpm_confidence_min=args.bpm_confidence_min,
                key=args.key, key_provenance=args.key_provenance,
                midi_notes_min=args.midi_notes_min, midi_notes_max=args.midi_notes_max,
                pitch_min=args.pitch_min, pitch_max=args.pitch_max,
                midi_density_min=args.midi_density_min,
                midi_density_max=args.midi_density_max,
                limit=args.limit,
            )
            if args.json:
                res = []
                for r in rows:
                    item = dict(r)
                    if "audio_descriptor" in item and item["audio_descriptor"]:
                        try:
                            item["audio_descriptor"] = json.loads(item["audio_descriptor"])
                        except Exception:
                            pass
                    res.append(item)
                print(json.dumps(res, indent=2))
            else:
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
