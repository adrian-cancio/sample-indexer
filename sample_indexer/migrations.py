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

MIGRATION_2 = """
-- L3-3: cheap audio metadata (ffprobe, header/container level) + provisional path/name labels.
-- Probe columns: what the container header CLAIMS. `valid` / `valid_error` (full-decode
-- validity) are deliberately NOT written by this stage; they stay NULL until a decode stage.
ALTER TABLE resource ADD COLUMN container    TEXT;            -- ffprobe format_name (wav, aiff, ...)
ALTER TABLE resource ADD COLUMN codec_name   TEXT;            -- pcm_s24le, pcm_f32le, ...
ALTER TABLE resource ADD COLUMN sample_fmt   TEXT;            -- ffprobe decoder-side format (s16, s32, flt)
ALTER TABLE resource ADD COLUMN sample_type  TEXT CHECK (sample_type IS NULL OR sample_type IN ('int','float'));
ALTER TABLE resource ADD COLUMN bits_source  TEXT;            -- which ffprobe field `bits` came from
ALTER TABLE resource ADD COLUMN probe_status TEXT CHECK (probe_status IS NULL OR probe_status IN
    ('ok','empty','malformed','no_audio_stream','unreadable','stale','timeout','tool_error'));
ALTER TABLE resource ADD COLUMN probe_error  TEXT;
ALTER TABLE resource ADD COLUMN probe_scope  TEXT;            -- 'header_container' (never full decode)
ALTER TABLE resource ADD COLUMN probe_at     TEXT;

-- Provisional labels from folder/file names (hints, not DSP results). kind/category hold the
-- interpreted result; *_source says which hints produced it; label_evidence keeps the raw
-- source hints as JSON so later DSP can confirm/correct. label_basis marks who owns
-- kind/category: the labels stage only (re)writes them while it is NULL/'path_name_provisional'.
ALTER TABLE resource ADD COLUMN kind_source         TEXT;
ALTER TABLE resource ADD COLUMN kind_confidence     TEXT CHECK (kind_confidence IS NULL OR kind_confidence IN ('low','medium','high'));
ALTER TABLE resource ADD COLUMN category_source     TEXT;
ALTER TABLE resource ADD COLUMN category_confidence TEXT CHECK (category_confidence IS NULL OR category_confidence IN ('low','medium','high'));
ALTER TABLE resource ADD COLUMN label_basis         TEXT;
ALTER TABLE resource ADD COLUMN label_evidence      TEXT;
ALTER TABLE resource ADD COLUMN bpm_name_raw        TEXT;     -- matched token, e.g. '125BPM'
ALTER TABLE resource ADD COLUMN key_name_raw        TEXT;     -- matched token, e.g. 'Fmin'
ALTER TABLE resource ADD COLUMN beats_hint          REAL;     -- duration * bpm_name / 60 (derived, NOT measured; `beats` is for DSP)
ALTER TABLE resource ADD COLUMN bpm_check           TEXT CHECK (bpm_check IS NULL OR bpm_check IN ('whole_bars','whole_beats','off_grid'));
CREATE INDEX idx_resource_probe_status ON resource(probe_status);

-- Fingerprint of the inputs a stage row was computed from (labels: present paths + duration),
-- so path renames invalidate path-derived work although the content hash is unchanged.
ALTER TABLE resource_stage ADD COLUMN input_sig TEXT;

-- One row per `metadata` invocation (diagnostics / history; updated at every checkpoint).
CREATE TABLE metadata_run (
    id           INTEGER PRIMARY KEY,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL CHECK (status IN ('running','complete','interrupted','aborted')),
    limit_n      INTEGER,
    workers      INTEGER NOT NULL,
    mode         TEXT NOT NULL,                 -- e.g. 'default', 'force', 'retry-errors', '+stratified'
    selected     INTEGER NOT NULL DEFAULT 0,
    processed    INTEGER NOT NULL DEFAULT 0,
    probed       INTEGER NOT NULL DEFAULT 0,
    probe_ok     INTEGER NOT NULL DEFAULT 0,
    probe_errors INTEGER NOT NULL DEFAULT 0,
    labelled     INTEGER NOT NULL DEFAULT 0,
    elapsed_s    REAL
);
"""

MIGRATION_3 = """
-- L3-4: low-end deterministic DSP (kick / rumble / bass candidates) + the FULL-decode validity
-- verdict. `valid` = 1 only after ffmpeg decoded the entire stream with -xerror and exit 0;
-- `valid` = 0 only for a conclusive decode failure of the source (never timeout / missing
-- tool / stale file). Existing DSP columns (f0_hz, sub_peak_hz, band_*, crest, dc_offset,
-- clipping, stereo_corr_low, decay_ms) are filled by this stage for its candidates only;
-- units are documented in reports/L3-4.md and in `lowend_json`.
ALTER TABLE resource ADD COLUMN valid_scope     TEXT;            -- 'full_decode_ffmpeg_xerror'
ALTER TABLE resource ADD COLUMN decoded_frames  INTEGER;         -- PCM frames really decoded
ALTER TABLE resource ADD COLUMN lowend_status   TEXT CHECK (lowend_status IS NULL OR lowend_status IN
    ('ok','empty_audio','unresolved_cap','unresolved_metadata','decode_failed','decode_warnings',
     'analysis_error','unreadable','stale','timeout','tool_error'));
ALTER TABLE resource ADD COLUMN lowend_error    TEXT;
ALTER TABLE resource ADD COLUMN lowend_at       TEXT;
ALTER TABLE resource ADD COLUMN lowend_select   TEXT;            -- why selected (rules), provenance
ALTER TABLE resource ADD COLUMN lowend_json     TEXT;            -- full descriptor: settings, caveats, contours
ALTER TABLE resource ADD COLUMN peak            REAL;            -- sample peak, full scale = 1.0
ALTER TABLE resource ADD COLUMN rms             REAL;            -- full scale = 1.0
ALTER TABLE resource ADD COLUMN clipped_samples INTEGER;         -- count of samples at/near full scale
CREATE INDEX idx_resource_lowend_status ON resource(lowend_status);

-- One row per `lowend` invocation (diagnostics / history; updated at every checkpoint).
CREATE TABLE lowend_run (
    id           INTEGER PRIMARY KEY,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL CHECK (status IN ('running','complete','interrupted','aborted')),
    limit_n      INTEGER,
    workers      INTEGER NOT NULL,
    mode         TEXT NOT NULL,
    selected     INTEGER NOT NULL DEFAULT 0,
    processed    INTEGER NOT NULL DEFAULT 0,
    decoded      INTEGER NOT NULL DEFAULT 0,      -- ffmpeg subprocesses spawned
    ok           INTEGER NOT NULL DEFAULT 0,
    failed       INTEGER NOT NULL DEFAULT 0,      -- conclusive decode failures (valid=0)
    unresolved   INTEGER NOT NULL DEFAULT 0,      -- caps / missing metadata / inconclusive
    elapsed_s    REAL
);
"""

MIGRATION_4 = """
-- L3-5: loop tempo / beats / bars / key from full-decode DSP (numpy FFT). The pre-existing
-- bpm_est, beats and key_est columns are filled ONLY by this stage (loop candidates); the new
-- columns carry the evidence. Every value is a measurement with caveats, see reports/L3-5.md and
-- `loops_json` (all settings, candidates, 12-bin chroma). NULL + a *_reason means "not resolved".
ALTER TABLE resource ADD COLUMN loops_status    TEXT CHECK (loops_status IS NULL OR loops_status IN
    ('ok','empty_audio','unresolved_cap','unresolved_metadata','decode_failed','decode_warnings',
     'analysis_error','unreadable','stale','timeout','tool_error'));
ALTER TABLE resource ADD COLUMN loops_error     TEXT;
ALTER TABLE resource ADD COLUMN loops_at        TEXT;
ALTER TABLE resource ADD COLUMN loops_select    TEXT;            -- why selected (rules), provenance
ALTER TABLE resource ADD COLUMN loops_json      TEXT;            -- full descriptor: settings, candidates, chroma12
ALTER TABLE resource ADD COLUMN bpm_confidence  REAL;            -- autocorrelation peak prominence (ordinal, not a probability)
ALTER TABLE resource ADD COLUMN bpm_reason      TEXT;            -- why bpm_est is NULL (silence, weak_periodicity, ...)
ALTER TABLE resource ADD COLUMN bars_est        REAL;            -- duration * bpm_est / 60 / 4 (4/4 assumed)
ALTER TABLE resource ADD COLUMN bpm_hint_check  TEXT CHECK (bpm_hint_check IS NULL OR bpm_hint_check IN
    ('exact','tolerance','half_double','conflict','unavailable'));
ALTER TABLE resource ADD COLUMN length_check    TEXT;            -- whole_bars | bars_plus_silent_tail | whole_beats | truncated | noninteger | unavailable
ALTER TABLE resource ADD COLUMN key_score       REAL;            -- best Krumhansl-Schmuckler correlation
ALTER TABLE resource ADD COLUMN key_margin      REAL;            -- best minus second-best correlation
ALTER TABLE resource ADD COLUMN key_strength    REAL;            -- chroma tonal strength 0..1
ALTER TABLE resource ADD COLUMN key_reason      TEXT;            -- why key_est is NULL
ALTER TABLE resource ADD COLUMN tuning_cents    REAL;            -- estimated tuning offset vs A=440, -50..+50
ALTER TABLE resource ADD COLUMN key_hint_check  TEXT CHECK (key_hint_check IS NULL OR key_hint_check IN
    ('same','relative','parallel','mismatch','unavailable'));
CREATE INDEX idx_resource_loops_status ON resource(loops_status);

-- One row per `loops` invocation (diagnostics / history; updated at every checkpoint).
CREATE TABLE loops_run (
    id           INTEGER PRIMARY KEY,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL CHECK (status IN ('running','complete','interrupted','aborted')),
    limit_n      INTEGER,
    workers      INTEGER NOT NULL,
    mode         TEXT NOT NULL,
    selected     INTEGER NOT NULL DEFAULT 0,
    processed    INTEGER NOT NULL DEFAULT 0,
    decoded      INTEGER NOT NULL DEFAULT 0,      -- ffmpeg subprocesses spawned
    ok           INTEGER NOT NULL DEFAULT 0,
    failed       INTEGER NOT NULL DEFAULT 0,      -- conclusive decode failures
    unresolved   INTEGER NOT NULL DEFAULT 0,      -- caps / missing metadata / inconclusive
    elapsed_s    REAL
);
"""

MIGRATION_5 = """
-- General measurements remain method-isolated from earlier lowend/loops values.
CREATE TABLE audio_measurement (
    resource_id INTEGER PRIMARY KEY REFERENCES resource(id),
    status TEXT NOT NULL,
    error TEXT,
    measured_at TEXT NOT NULL,
    descriptor TEXT
);
"""

MIGRATION_6 = """
-- L3-7 parsed MIDI facts are isolated from audio measurements and name hints.
ALTER TABLE resource ADD COLUMN midi_ticks INTEGER;
ALTER TABLE resource ADD COLUMN midi_beats REAL;
ALTER TABLE resource ADD COLUMN midi_duration_s REAL;
ALTER TABLE resource ADD COLUMN midi_status TEXT;
CREATE TABLE midi_measurement (
    resource_id INTEGER PRIMARY KEY REFERENCES resource(id),
    status TEXT NOT NULL,
    error TEXT,
    measured_at TEXT NOT NULL,
    descriptor TEXT,
    failed_paths TEXT NOT NULL
);
CREATE TABLE midi_run (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    selected INTEGER NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0,
    parsed INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0
);
"""

MIGRATION_7 = """
-- Drop obsolete confidence column (superseded by kind_confidence and category_confidence in v2)
ALTER TABLE resource DROP COLUMN confidence;

-- Backfill general audio DSP metrics into resource from audio_measurement
UPDATE resource
SET centroid = json_extract(am.descriptor, '$.analysis.spectral_centroid_hz'),
    rolloff = json_extract(am.descriptor, '$.analysis.spectral_rolloff85_hz'),
    flatness = json_extract(am.descriptor, '$.analysis.spectral_flatness'),
    lufs = json_extract(am.descriptor, '$.analysis.integrated_lufs'),
    true_peak = json_extract(am.descriptor, '$.analysis.true_peak_dbtp')
FROM audio_measurement am
WHERE resource.id = am.resource_id AND am.status = 'ok';
"""

MIGRATIONS: list[tuple[int, str]] = [
    (1, MIGRATION_1),
    (2, MIGRATION_2),
    (3, MIGRATION_3),
    (4, MIGRATION_4),
    (5, MIGRATION_5),
    (6, MIGRATION_6),
    (7, MIGRATION_7),
]

LATEST_VERSION = MIGRATIONS[-1][0]
