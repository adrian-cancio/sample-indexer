"""Read-only reporting: stats, query and manifest verification."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .config import EXT_CLASS


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def stats(conn: sqlite3.Connection) -> str:
    out: list[str] = []
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731

    present = one("SELECT COUNT(*) FROM path WHERE missing=0")
    missing = one("SELECT COUNT(*) FROM path WHERE missing=1")
    out.append(f"paths present      {present}")
    out.append(f"paths missing      {missing}")
    out.append(f"unique resources   {one('SELECT COUNT(*) FROM resource')}")

    audio_exts = [e for e, c in EXT_CLASS.items() if c == "audio"]
    midi_exts = [e for e, c in EXT_CLASS.items() if c == "midi"]
    counts = {
        r["ext"]: r["n"]
        for r in conn.execute("SELECT ext, COUNT(*) AS n FROM path WHERE missing=0 GROUP BY ext")
    }
    audio = sum(counts.get(e, 0) for e in audio_exts)
    midi = sum(counts.get(e, 0) for e in midi_exts)
    out.append(f"present by kind    audio {audio} + midi {midi} = {audio + midi}")

    out.append("present by ext:")
    for ext, n in sorted(counts.items()):
        out.append(f"  {ext:<8} {n}")

    out.append("present by top-level folder:")
    for r in conn.execute(
        "SELECT top_folder, COUNT(*) AS n FROM path WHERE missing=0 GROUP BY top_folder"
        " ORDER BY top_folder"
    ):
        out.append(f"  {r['top_folder'] or '(root)':<20} {r['n']}")

    dup = conn.execute(
        "SELECT COUNT(*) AS groups, COALESCE(SUM(n),0) AS paths FROM ("
        " SELECT COUNT(*) AS n FROM path WHERE missing=0 GROUP BY resource_id HAVING n > 1)"
    ).fetchone()
    out.append(f"duplicate sha256   {dup['groups']} resources on {dup['paths']} present paths")
    for r in conn.execute(
        "SELECT r.sha256, COUNT(*) AS n FROM path p JOIN resource r ON r.id=p.resource_id"
        " WHERE p.missing=0 GROUP BY p.resource_id HAVING n > 1 ORDER BY n DESC, r.sha256 LIMIT 10"
    ):
        out.append(f"  {r['sha256']}  x{r['n']}")
    return "\n".join(out)


def metadata_coverage(conn: sqlite3.Connection) -> str:
    """Coverage of the L3-3 stages over resources that have a present path."""
    from .metadata import LABELS_STAGE, LABELS_VERSION, PROBE_STAGE, PROBE_VERSION

    out: list[str] = []
    # IN (not a correlated EXISTS): with ~all paths present, the planner picks the unselective
    # idx_path_missing for EXISTS and degrades to O(n^2) on a full-size index.
    base = "FROM resource r WHERE r.id IN (SELECT resource_id FROM path WHERE missing=0)"
    one = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731
    audio = one(f"SELECT COUNT(*) {base} AND r.ext != '.mid'")
    midi = one(f"SELECT COUNT(*) {base} AND r.ext = '.mid'")
    out.append(f"resources with a present path: audio {audio}, midi {midi}")
    stage_q = (
        f"SELECT COUNT(*) {base} AND r.ext {{op}} '.mid' AND EXISTS (SELECT 1 FROM resource_stage s"
        " WHERE s.resource_id=r.id AND s.stage=? AND s.version=?{extra})"
    )
    probed = one(stage_q.format(op="!=", extra=""), PROBE_STAGE, PROBE_VERSION)
    probe_err = one(stage_q.format(op="!=", extra=" AND s.error IS NOT NULL"), PROBE_STAGE, PROBE_VERSION)
    labelled = one(
        f"SELECT COUNT(*) {base} AND EXISTS (SELECT 1 FROM resource_stage s WHERE s.resource_id=r.id"
        " AND s.stage=? AND s.version=?)", LABELS_STAGE, LABELS_VERSION,
    )
    out.append(f"probe stage v{PROBE_VERSION}: processed {probed}/{audio} audio (errors {probe_err}); pending {audio - probed}")
    out.append(f"labels stage v{LABELS_VERSION}: processed {labelled}/{audio + midi}; pending {audio + midi - labelled}")
    out.append("probe scope: header/container only; valid/valid_error unset until a full-decode stage")
    out.append(f"valid set (full decode): {one(f'SELECT COUNT(*) {base} AND r.valid IS NOT NULL')}")

    def table(title: str, sql: str, *params) -> None:
        out.append(title)
        for row in conn.execute(sql, params):
            vals = [("null" if v is None else str(v)) for v in tuple(row)[:-1]]
            out.append(f"  {' / '.join(vals):<40} {row[-1]}")

    ok = f"{base} AND r.probe_status = 'ok'"
    table("probe status (audio):",
          f"SELECT COALESCE(r.probe_status, '(not probed)'), COUNT(*) {base} AND r.ext != '.mid'"
          " GROUP BY 1 ORDER BY 2 DESC")
    table("sample rate (probe ok):", f"SELECT sr, COUNT(*) {ok} GROUP BY sr ORDER BY sr")
    table("channels (probe ok):", f"SELECT channels, COUNT(*) {ok} GROUP BY channels ORDER BY channels")
    table("bits / source field (probe ok):",
          f"SELECT bits, bits_source, COUNT(*) {ok} GROUP BY bits, bits_source ORDER BY bits")
    table("codec / sample_fmt / type (probe ok):",
          f"SELECT codec_name, sample_fmt, sample_type, COUNT(*) {ok}"
          " GROUP BY codec_name, sample_fmt, sample_type ORDER BY 4 DESC")
    table("container (probe ok):", f"SELECT container, COUNT(*) {ok} GROUP BY container ORDER BY 2 DESC")
    out.append(f"duration null among probe ok: {one(f'SELECT COUNT(*) {ok} AND r.duration_s IS NULL')}")
    table("kind (provisional, path/name):",
          f"SELECT COALESCE(kind,'(unlabelled)'), kind_source, COUNT(*) {base}"
          " GROUP BY kind, kind_source ORDER BY 3 DESC")
    table("category (provisional, path/name):",
          f"SELECT COALESCE(category,'(unlabelled)'), COUNT(*) {base} GROUP BY category ORDER BY 2 DESC")
    table("confidence (kind / category):",
          f"SELECT kind_confidence, category_confidence, COUNT(*) {base}"
          " GROUP BY 1, 2 ORDER BY 3 DESC")
    out.append(f"bpm_name present: {one(f'SELECT COUNT(*) {base} AND r.bpm_name IS NOT NULL')}"
               f" (bpm_check whole_bars/whole_beats/off_grid: "
               + "/".join(str(one(f"SELECT COUNT(*) {base} AND r.bpm_check=?", c))
                          for c in ("whole_bars", "whole_beats", "off_grid")) + ")")
    out.append(f"key_name present: {one(f'SELECT COUNT(*) {base} AND r.key_name IS NOT NULL')}")
    out.append("labels are provisional path/name hints; classification accuracy is unvalidated (L3-8)")
    return "\n".join(out)


def query(
    conn: sqlite3.Connection,
    *,
    path_contains: str | None = None,
    ext: str | None = None,
    folder: str | None = None,
    include_missing: bool = False,
    include_errors: bool = False,
    only_errors: bool = False,
    duration_min: float | None = None,
    duration_max: float | None = None,
    sr: int | None = None,
    channels: int | None = None,
    f0_min: float | None = None,
    f0_max: float | None = None,
    sub_peak_min: float | None = None,
    sub_peak_max: float | None = None,
    decay_min: float | None = None,
    decay_max: float | None = None,
    stereo_corr_low_min: float | None = None,
    stereo_corr_low_max: float | None = None,
    lufs_min: float | None = None,
    lufs_max: float | None = None,
    true_peak_min: float | None = None,
    true_peak_max: float | None = None,
    centroid_min: float | None = None,
    centroid_max: float | None = None,
    rolloff_min: float | None = None,
    rolloff_max: float | None = None,
    flatness_min: float | None = None,
    flatness_max: float | None = None,
    bpm_min: float | None = None,
    bpm_max: float | None = None,
    bpm_provenance: str | None = None,  # 'est', 'name', 'either'
    bpm_confidence_min: float | None = None,
    key: str | None = None,
    key_provenance: str | None = None,  # 'est', 'name', 'either'
    midi_notes_min: int | None = None,
    midi_notes_max: int | None = None,
    pitch_min: int | None = None,
    pitch_max: int | None = None,
    midi_density_min: float | None = None,
    midi_density_max: float | None = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    where, params = [], []
    if not include_missing:
        where.append("p.missing = 0")
    if only_errors:
        where.append("(r.valid = 0 OR (r.probe_status IS NOT NULL AND r.probe_status != 'ok') OR r.midi_status = 'error' OR (am.status IS NOT NULL AND am.status != 'ok'))")
    elif not include_errors:
        # Exclude known invalid or error states unless opt-in:
        # valid=0 is conclusive decode failure; probe_status!='ok' is probe failure; midi_status='error' is parsing failure;
        # and resource 27954 has decode_warnings (header-only empty output).
        where.append("(r.valid IS NULL OR r.valid = 1)")
        where.append("(r.probe_status IS NULL OR r.probe_status = 'ok')")
        where.append("(r.midi_status IS NULL OR r.midi_status IN ('ok', 'empty'))")
        where.append("(am.status IS NULL OR am.status = 'ok')")

    if path_contains:
        where.append("casefold(p.rel_path) LIKE ? ESCAPE '\\'")
        params.append(f"%{_like_escape(path_contains.casefold())}%")
    if ext:
        where.append("p.ext = ?")
        params.append("." + ext.lower().lstrip("."))
    if folder:
        prefix = folder.strip("/")
        where.append("(casefold(p.rel_path) LIKE ? ESCAPE '\\' OR casefold(p.rel_path) = ?)")
        params.extend([f"{_like_escape(prefix.casefold())}/%", prefix.casefold()])

    # Numeric bounds exclude NULLs explicitly per contract
    if duration_min is not None:
        where.append("r.duration_s IS NOT NULL AND r.duration_s >= ?")
        params.append(duration_min)
    if duration_max is not None:
        where.append("r.duration_s IS NOT NULL AND r.duration_s <= ?")
        params.append(duration_max)
    if sr is not None:
        where.append("r.sr IS NOT NULL AND r.sr = ?")
        params.append(sr)
    if channels is not None:
        where.append("r.channels IS NOT NULL AND r.channels = ?")
        params.append(channels)

    if f0_min is not None:
        where.append("r.f0_hz IS NOT NULL AND r.f0_hz >= ?")
        params.append(f0_min)
    if f0_max is not None:
        where.append("r.f0_hz IS NOT NULL AND r.f0_hz <= ?")
        params.append(f0_max)
    if sub_peak_min is not None:
        where.append("r.sub_peak_hz IS NOT NULL AND r.sub_peak_hz >= ?")
        params.append(sub_peak_min)
    if sub_peak_max is not None:
        where.append("r.sub_peak_hz IS NOT NULL AND r.sub_peak_hz <= ?")
        params.append(sub_peak_max)
    if decay_min is not None:
        where.append("r.decay_ms IS NOT NULL AND r.decay_ms >= ?")
        params.append(decay_min)
    if decay_max is not None:
        where.append("r.decay_ms IS NOT NULL AND r.decay_ms <= ?")
        params.append(decay_max)
    if stereo_corr_low_min is not None:
        where.append("r.stereo_corr_low IS NOT NULL AND r.stereo_corr_low >= ?")
        params.append(stereo_corr_low_min)
    if stereo_corr_low_max is not None:
        where.append("r.stereo_corr_low IS NOT NULL AND r.stereo_corr_low <= ?")
        params.append(stereo_corr_low_max)

    if lufs_min is not None:
        where.append("r.lufs IS NOT NULL AND r.lufs >= ?")
        params.append(lufs_min)
    if lufs_max is not None:
        where.append("r.lufs IS NOT NULL AND r.lufs <= ?")
        params.append(lufs_max)
    if true_peak_min is not None:
        where.append("r.true_peak IS NOT NULL AND r.true_peak >= ?")
        params.append(true_peak_min)
    if true_peak_max is not None:
        where.append("r.true_peak IS NOT NULL AND r.true_peak <= ?")
        params.append(true_peak_max)
    if centroid_min is not None:
        where.append("r.centroid IS NOT NULL AND r.centroid >= ?")
        params.append(centroid_min)
    if centroid_max is not None:
        where.append("r.centroid IS NOT NULL AND r.centroid <= ?")
        params.append(centroid_max)
    if rolloff_min is not None:
        where.append("r.rolloff IS NOT NULL AND r.rolloff >= ?")
        params.append(rolloff_min)
    if rolloff_max is not None:
        where.append("r.rolloff IS NOT NULL AND r.rolloff <= ?")
        params.append(rolloff_max)
    if flatness_min is not None:
        where.append("r.flatness IS NOT NULL AND r.flatness >= ?")
        params.append(flatness_min)
    if flatness_max is not None:
        where.append("r.flatness IS NOT NULL AND r.flatness <= ?")
        params.append(flatness_max)

    # BPM with provenance and confidence
    bpm_prov = (bpm_provenance or "either").lower()
    if bpm_confidence_min is not None:
        where.append("r.bpm_confidence IS NOT NULL AND r.bpm_confidence >= ?")
        params.append(bpm_confidence_min)
    if bpm_min is not None or bpm_max is not None:
        col_expr = "r.bpm_est" if bpm_prov == "est" else ("r.bpm_name" if bpm_prov == "name" else "COALESCE(r.bpm_est, r.bpm_name)")
        if bpm_min is not None:
            where.append(f"{col_expr} IS NOT NULL AND {col_expr} >= ?")
            params.append(bpm_min)
        if bpm_max is not None:
            where.append(f"{col_expr} IS NOT NULL AND {col_expr} <= ?")
            params.append(bpm_max)

    # Key with provenance
    if key is not None:
        key_prov = (key_provenance or "either").lower()
        if key_prov == "est":
            where.append("casefold(r.key_est) = ?")
            params.append(key.casefold())
        elif key_prov == "name":
            where.append("casefold(r.key_name) = ?")
            params.append(key.casefold())
        else:
            where.append("(casefold(r.key_est) = ? OR casefold(r.key_name) = ?)")
            params.extend([key.casefold(), key.casefold()])

    # MIDI filters
    if midi_notes_min is not None:
        where.append("r.notes IS NOT NULL AND r.notes >= ?")
        params.append(midi_notes_min)
    if midi_notes_max is not None:
        where.append("r.notes IS NOT NULL AND r.notes <= ?")
        params.append(midi_notes_max)
    if pitch_min is not None:
        where.append("r.pitch_min IS NOT NULL AND r.pitch_min >= ?")
        params.append(pitch_min)
    if pitch_max is not None:
        where.append("r.pitch_max IS NOT NULL AND r.pitch_max <= ?")
        params.append(pitch_max)
    if midi_density_min is not None:
        where.append("r.density IS NOT NULL AND r.density >= ?")
        params.append(midi_density_min)
    if midi_density_max is not None:
        where.append("r.density IS NOT NULL AND r.density <= ?")
        params.append(midi_density_max)

    sql = (
        "SELECT p.rel_path, p.ext, p.size, p.missing, r.sha256, r.duration_s, r.sr, r.channels,"
        " r.f0_hz, r.sub_peak_hz, r.decay_ms, r.stereo_corr_low,"
        " r.bpm_name, r.bpm_est, r.bpm_confidence, r.key_name, r.key_est,"
        " r.ppq, r.tempo, r.bars, r.notes, r.pitch_min, r.pitch_max, r.density,"
        " r.kind, r.category, am.descriptor as audio_descriptor"
        " FROM path p"
        " JOIN resource r ON r.id = p.resource_id"
        " LEFT JOIN audio_measurement am ON am.resource_id = r.id"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY p.rel_path"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


@dataclass
class ManifestDiff:
    manifest_rows: int = 0
    db_rows: int = 0
    sha_mismatch: list[tuple[str, str, str]] = field(default_factory=list)  # rel, manifest, db
    size_mismatch: list[tuple[str, int, int]] = field(default_factory=list)
    only_in_manifest: list[str] = field(default_factory=list)
    only_in_db: list[str] = field(default_factory=list)

    @property
    def differences(self) -> int:
        return (
            len(self.sha_mismatch)
            + len(self.size_mismatch)
            + len(self.only_in_manifest)
            + len(self.only_in_db)
        )

    def format(self, max_lines: int = 20) -> str:
        lines = [
            f"manifest rows      {self.manifest_rows}",
            f"db present paths   {self.db_rows}",
            f"sha256 mismatches  {len(self.sha_mismatch)}",
            f"size mismatches    {len(self.size_mismatch)}",
            f"only in manifest   {len(self.only_in_manifest)}",
            f"only in db         {len(self.only_in_db)}",
            f"differences        {self.differences}",
        ]
        for label, items in (
            ("sha256", self.sha_mismatch),
            ("size", self.size_mismatch),
            ("only in manifest", self.only_in_manifest),
            ("only in db", self.only_in_db),
        ):
            for item in items[:max_lines]:
                lines.append(f"  {label}: {item}")
        return "\n".join(lines)


def verify_manifest(
    conn: sqlite3.Connection, csv_path: str | Path, root: str | None = None
) -> ManifestDiff:
    """Compare present DB paths with a manifest CSV (sha256,rel_path,size,mtime).

    mtime is deliberately not compared (the manifest has whole seconds; the DB keeps ns).
    """
    diff = ManifestDiff()
    manifest: dict[str, tuple[str, int]] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_cols = {"sha256", "rel_path", "size", "mtime"} - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"manifest lacks columns: {sorted(missing_cols)}")
        for row in reader:
            diff.manifest_rows += 1
            manifest[row["rel_path"]] = (row["sha256"].strip().lower(), int(row["size"]))

    sql = (
        "SELECT p.rel_path, p.size, r.sha256 FROM path p JOIN resource r ON r.id=p.resource_id"
        " WHERE p.missing=0"
    )
    params: list = []
    if root:
        sql += " AND p.root=?"
        params.append(root)
    db: dict[str, tuple[str, int]] = {}
    for r in conn.execute(sql, params):
        diff.db_rows += 1
        db[r["rel_path"]] = (r["sha256"], r["size"])

    for rel in sorted(manifest.keys() | db.keys()):
        m, d = manifest.get(rel), db.get(rel)
        if m is None:
            diff.only_in_db.append(rel)
        elif d is None:
            diff.only_in_manifest.append(rel)
        else:
            if m[0] != d[0]:
                diff.sha_mismatch.append((rel, m[0], d[0]))
            if m[1] != d[1]:
                diff.size_mismatch.append((rel, m[1], d[1]))
    return diff
