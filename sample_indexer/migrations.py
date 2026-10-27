"""Ordered schema migrations. Append new (version, sql) tuples; never edit old ones."""

MIGRATION_1 = """
-- One row per unique file content (sha256). Analysis columns are nullable and
-- are filled by later stages (L3-3..L3-7); the reconciler only writes the
-- identity columns (sha256, size, ext, first_seen) plus kind='midi' for .mid.
CREATE TABLE resource (
    id              INTEGER PRIMARY KEY,
    sha256          TEXT NOT NULL UNIQUE CHECK (length(sha256) = 64),
    size            INTEGER NOT NULL,
    ext             TEXT NOT NULL,              -- lower-case, with dot
    first_seen      TEXT NOT NULL,              -- UTC ISO-8601

    kind            TEXT CHECK (kind IS NULL OR kind IN ('one_shot','loop','midi','unknown')),
    category        TEXT,
    confidence      REAL,

    -- audio container / validity
    duration_s      REAL,
    sr              INTEGER,
    bits            INTEGER,
    channels        INTEGER,
    valid           INTEGER CHECK (valid IS NULL OR valid IN (0,1)),
    valid_error     TEXT,

    -- tempo / key (…_name: parsed from the file name, …_est: estimated from audio)
    bpm_name        REAL,
    bpm_est         REAL,
    beats           REAL,
    key_name        TEXT,
    key_est         TEXT,

    -- spectral
    f0_hz           REAL,
    sub_peak_hz     REAL,
    centroid        REAL,
    rolloff         REAL,
    flatness        REAL,
    band_sub        REAL,                       -- band_energy: sub/low/mid/high
    band_low        REAL,
    band_mid        REAL,
    band_high       REAL,

    -- loudness / dynamics / stereo
    lufs            REAL,
    true_peak       REAL,
    crest           REAL,
    dc_offset       REAL,
    clipping        REAL,
    stereo_corr_low REAL,
    decay_ms        REAL,

    -- MIDI
    ppq             INTEGER,
    tempo           REAL,
    bars            REAL,
    notes           INTEGER,
    pitch_min       INTEGER,                    -- pitch_range = [pitch_min, pitch_max]
    pitch_max       INTEGER,
    density         REAL,
    scale_est       TEXT,
    tracks          INTEGER
);
CREATE INDEX idx_resource_kind ON resource(kind);
CREATE INDEX idx_resource_ext  ON resource(ext);

-- Current (and historical, flagged missing) locations of resources.
-- Paths are stored relative to `root`; (root, rel_path) is unique.
CREATE TABLE path (
    id            INTEGER PRIMARY KEY,
    resource_id   INTEGER NOT NULL REFERENCES resource(id),
    root          TEXT NOT NULL,
    rel_path      TEXT NOT NULL,
    top_folder    TEXT NOT NULL,                -- first path component ('' for root-level files)
    ext           TEXT NOT NULL,                -- lower-case, with dot
    size          INTEGER NOT NULL,
    mtime_ns      INTEGER NOT NULL,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    missing       INTEGER NOT NULL DEFAULT 0 CHECK (missing IN (0,1)),
    missing_since TEXT,
    UNIQUE (root, rel_path)
);
CREATE INDEX idx_path_resource ON path(resource_id);
CREATE INDEX idx_path_missing  ON path(missing);
CREATE INDEX idx_path_ext      ON path(ext);
CREATE INDEX idx_path_top      ON path(top_folder);

-- Per-resource, per-stage bookkeeping so each analysis stage is resumable and
-- re-runnable independently (select resources lacking a row at the current version).
CREATE TABLE resource_stage (
    resource_id INTEGER NOT NULL REFERENCES resource(id) ON DELETE CASCADE,
    stage       TEXT NOT NULL,
    version     INTEGER NOT NULL,
    done_at     TEXT NOT NULL,
    error       TEXT,
    PRIMARY KEY (resource_id, stage)
);
CREATE INDEX idx_stage_lookup ON resource_stage(stage, version);

-- One row per scan invocation (diagnostics / history).
CREATE TABLE scan_run (
    id             INTEGER PRIMARY KEY,
    root           TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    status         TEXT NOT NULL CHECK (status IN ('running','complete','interrupted','aborted')),
    limit_n        INTEGER,
    workers        INTEGER NOT NULL,
    discovered     INTEGER NOT NULL DEFAULT 0,
    unchanged      INTEGER NOT NULL DEFAULT 0,
    new_paths      INTEGER NOT NULL DEFAULT 0,
    changed        INTEGER NOT NULL DEFAULT 0,
    moved          INTEGER NOT NULL DEFAULT 0,
    hashed         INTEGER NOT NULL DEFAULT 0,
    marked_missing INTEGER NOT NULL DEFAULT 0,
    errors         INTEGER NOT NULL DEFAULT 0
);
"""

MIGRATIONS: list[tuple[int, str]] = [
    (1, MIGRATION_1),
]

LATEST_VERSION = MIGRATIONS[-1][0]
