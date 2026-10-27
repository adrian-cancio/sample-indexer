# sample-indexer

A SQLite index for a techno/schranz sample and MIDI library. Python 3.14 stdlib plus the
system `ffprobe`/`ffmpeg` and `numpy` (for L3-4 through L3-6 DSP). The library is **read-only** to this tool: files are opened
`O_RDONLY` and never modified, moved or deleted.

## Quick start

```
python3 -m sample_indexer init                         # create/migrate data/index.sqlite
python3 -m sample_indexer scan                         # stage 1: identity (SHA-256) + paths
python3 -m sample_indexer metadata --workers 4         # stage 2: ffprobe header metadata + labels
python3 -m sample_indexer lowend --workers 2           # stage 3: full decode + low-end DSP (kick/rumble/bass candidates)
python3 -m sample_indexer loops --workers 2            # stage 4: full decode + tempo/beats/bars/key DSP (loop candidates)
python3 -m sample_indexer audio --workers 2 --max-seconds 1200 # L3-6: general streaming timbre/levels/loudness
python3 -m sample_indexer midi                         # L3-7: MIDI event parsing (stdlib only)
python3 -m sample_indexer midi-coverage                # MIDI coverage and statuses
python3 -m sample_indexer coverage                     # what is done, distributions
python3 -m sample_indexer stats                        # counts by ext/folder, duplicates
python3 -m sample_indexer verify-manifest MANIFEST.csv # DB vs manifest (sha256,rel_path,size,mtime)
python3 -m sample_indexer query --folder Samples/Kicks --ext wav --limit 20
python3 -m sample_indexer backup data/index.backup.sqlite
python3 -m unittest discover                           # tests (temp trees only)
```

Global flag `--db PATH` (before the subcommand) or `SAMPLE_INDEXER_DB` selects another database;
`SAMPLE_INDEXER_ROOT` overrides the library root. Use a scratch DB for experiments:
`python3 -m sample_indexer --db $TMPDIR/pilot.sqlite ...`.

Exit codes: 0 ok, 1 differences / file or probe errors, 2 usage or DB error, 130 interrupted.

## Stages

| Stage | Command | Reads | Writes | Version |
|---|---|---|---|---|
| identity | `scan` | file bytes (hash), `stat` | `resource` (sha256, size, ext), `path` | – |
| probe | `metadata` | ONE `ffprobe` per audio resource | duration, sr, channels, bits, codec, … | `PROBE_VERSION = 1` |
| labels | `metadata` | nothing (paths/names only) | kind, category, bpm_name, key_name, evidence | `LABELS_VERSION = 1` |
| lowend | `lowend` | whole file, `ffmpeg -xerror` decode | `valid`/`valid_error`, peak/rms/crest/dc/clipping, bands, sub_peak_hz, f0_hz, decay_ms, stereo_corr_low, `lowend_json` (candidates only) | `LOWEND_VERSION = 1` (DSP `VERSION = 1`) |
| loops | `loops` | whole file, `ffmpeg -xerror` decode | `bpm_est`, `beats`, `bars_est`, `key_est` + confidence/reason/hint-comparison columns, `loops_json` (loop candidates only) | `LOOPS_VERSION = 1` (DSP `VERSION = 1`) |
| audio | `audio` | complete streaming PCM + loudnorm input meter | isolated `audio_measurement` descriptor; conclusive validity only | `AUDIO_VERSION = 1` (DSP `VERSION = 1`) |
| midi | `midi` | bounded SMF/RMID events | MIDI core fields and isolated `midi_measurement` provenance | `MIDI_VERSION = 1` |

`loops` (L3-5) selects loop candidates (folder OR `kind='loop'` OR an explicit `bpm_name`; unknown labels
are not excluded), decodes each file completely (cap 300 s, never truncated) and measures tempo
(spectral flux + autocorrelation, half/double alternatives kept), duration-derived beats/bars and key
(chroma + Krumhansl-Schmuckler, NULL with a reason when unreliable). Name hints are compared afterwards
and never used for the estimates. Settings, candidates and the 12-bin chroma are in `loops_json`; methods,
validation and limits are in `reports/L3-5.md`. Estimates are measurements with caveats, not truth.

`lowend` selects kick/rumble/bass candidates (category hint OR source folder, any label
confidence), decodes each file completely and writes conclusive `valid` values, as do
`loops` and `audio`. Methods, units, caps and limits are in `reports/L3-4.md`.

`audio` (L3-6) selects every present audio resource, streaming the complete native-channel
signal with bounded memory and chunked spectra, without a duration cap. General measurements
live in `audio_measurement.descriptor` with method/provenance and null reasons; earlier
lowend/loops measurements and provisional labels are not overwritten. LUFS and measured
true peak use ffmpeg input measurements, not normalized output predictions. Short one-shots
and silence may have undefined gated integrated loudness. See `reports/L3-6.md` for
execution evidence and exact limitations. Acoustic category validation remains later work.

`midi` (L3-7) parses actual SMF/RMID events without third-party MIDI dependencies.
It checkpoints each resource, supports `--limit`, `--force`, `--retry-errors` and
folder-stratified `--stratify --seed N` pilots. Errors count as done until explicitly
retried. JSON provenance and exact attempted/failed paths live in `midi_measurement`;
MIDI-only core columns and `midi_ticks`, `midi_beats`, `midi_duration_s` are queryable.
Audio descriptors, audio validity and filename hints are never overwritten. Format-2
sequences are independent, SMPTE is explicitly unsupported, bars may be unknown,
and key/scale estimates are ambiguous hypotheses, not ground truth. See `reports/L3-7.md`.

### scan: identity and renames

* Identity of a file is the SHA-256 of its content (`resource`); paths are attributes (`path`).
* A path with unchanged `(rel_path, size, mtime_ns)` is **not re-hashed**. Limitation: an edit
  that keeps both size and mtime identical goes undetected.
* Every **new or relocated** path is **always hashed**. `(size, mtime)` only nominates move
  candidates; a move is accepted only when the SHA-256 matches the vanished path's resource.
  A different file that merely shares size and mtime gets its own resource and inherits nothing.
* Vanished paths are flagged `missing=1` (rows are never deleted). `--limit` scans are partial
  and never mark anything missing.

### metadata: probe (header/container only)

For each audio resource with a present path, one `ffprobe` subprocess reads the container
header and records what it **claims**:

* `duration_s`, `sr`, `channels`, `container`, `codec_name`, `sample_fmt`.
* `bits` comes from `bits_per_raw_sample` if present (e.g. 24 for `pcm_s24le`, whose
  `sample_fmt` is `s32`), else `bits_per_sample`; `bits_source` records which. Both absent or 0
  gives `NULL`, never a guess.
* `sample_type` is `float` for `pcm_f*` codecs and `int` for `pcm_s*`/`pcm_u*`; `NULL` for
  anything else (e.g. compressed codecs).
* `probe_status`: `ok`, `empty` (0 bytes, no subprocess spawned), `malformed` (ffprobe could not
  parse it), `no_audio_stream`, `unreadable`, `stale` (file changed since the scan), `timeout`,
  `tool_error`. `probe_error` holds the message, `probe_scope` is always `header_container`.

**No full-decode guarantee.** A successful probe means the header parsed, not that the audio
decodes: a truncated file whose header is intact probes `ok`. `resource.valid` and
`valid_error` are deliberately left `NULL` (unknown, not invalid) until a later decode stage.
Probe failure statuses describe the probe, not a verdict on the audio.

MIDI is never passed to `ffprobe`: `kind='midi'` is set from the extension, everything else
stays `NULL` until the MIDI stage.

### metadata: provisional path/name labels

Labels come only from folder and file names. They are **hints, not sonic truth**:

* `kind` (`one_shot` | `loop` | `midi` | `unknown`) and `category` with `kind_source` /
  `category_source` (which hint produced it) and `*_confidence` (`low` | `medium` | `high`,
  ordinal, not a probability).
* Mapped explicitly: `Drums - One Shots/*` → one_shot; `Loops/*` → loop; `Synths/Synth Shots`
  → one_shot; `Synths/Synth Loops` → loop. General synth and FX folders, `Unsorted`, `Vocals`
  and `Bonus` do **not** establish a kind.
* Folder categories are used where the folder name is specific (Kicks, Snares, Hi-Hats Open …);
  broad ones stay broad (`drums`, `synth`, `fx`, `cymbal`). Generic `Hi-Hats` is `hihat`, never
  open/closed. Dirty/Fat/Layer/Groove/Modular/Schranz/Sidechained loops and `Unsorted` give
  `unknown` category unless the file name says something that does not contradict the folder.
* Conflicting evidence gives `unknown`: a "Loop" file name under One Shots, "no kick" in a Kick
  Loops folder, a Snare name under Kicks, an explicit BPM/duration contradiction (a "loop" shorter
  than one beat, a "one shot" of 4+ beats at its BPM), or several paths of one resource that
  disagree (ambiguity is preserved, not resolved).
* `label_evidence` (JSON) keeps the raw source hints per path (matched folder rules, name
  tokens, rejected tokens) separately from the interpreted result, so later DSP can confirm or
  correct. `label_basis='path_name_provisional'` marks who owns `kind`/`category`: the labels
  stage only rewrites them while that marker (or NULL) is present; a later stage that sets another
  basis is never overwritten.
* Renaming a file changes the stage `input_sig` (present paths + duration), so path-derived labels
  are recomputed on the next `metadata` run even though the hash is unchanged. No re-probe happens.
* **Classification accuracy is unvalidated** (planned for L3-8). Do not read the labels as measured.

`bpm_name` is set only from an explicit marker: a number 60–240 (optional decimals) followed by
`BPM` in any case (`125BPM`, `140 BPM`, `128bpm`) or `BPM` followed by a number (`BPM_150`). Bare
numbers (`Kick 160`, `TR909`, `SVE2 Perc 39`, `GV_VocPad_120-Fmin`) are never tempo. Several
different explicit values: none stored. The raw token is kept in `bpm_name_raw`.

`key_name` only accepts an upper-case note letter (`A`–`G`), optional `#`/`b`, and a delimited
`min|minor|maj|major` token (`Fmin`, `F#min`, `Bbmajor`). Chord/variant tokens (`Cmin7th`, `Emaj7min9`,
`Cmin2`), lone letters (`A`, `B`, `C5`) and ordinary words are rejected; rejected look-alikes are kept
in the evidence. The raw token is in `key_name_raw`.

`beats_hint = duration_s * bpm_name / 60` (2 decimals) is a **derived** figure, separate from the
reserved `beats` column. `bpm_check` (`whole_bars` | `whole_beats` | `off_grid`) compares the duration
to a 4/4 grid with a tolerance of `0.02 s + 0.1 % of duration`. It is only a plausibility hint. There
are no BPM/key estimates and no audio analysis here; `bpm_est`, `key_est`, `beats` and every other DSP
column stay untouched.

## Skip, force and retry semantics

A stage row in `resource_stage(resource_id, stage, version, done_at, error, input_sig)` marks work done.

* **Default**: a resource is selected only if its `probe` row is missing or older than
  `PROBE_VERSION`, or its `labels` row is missing, older, or has a different `input_sig`. A second
  `metadata` run therefore spawns zero ffprobe processes.
* **Errors count as done.** A recorded probe error at the current version is not retried
  automatically (no retry storm), and it does not block other resources.
* `--retry-errors` re-probes exactly those resources whose recorded probe error is at the current version.
* `--force` redoes everything selected regardless of version (combine with `--limit` to bound it).
* A transient outcome (`timeout`, `tool_error`, `stale`, `unreadable`) records the status but keeps
  any earlier conclusive header facts.
* A resource is analysed once even with several paths (probes the first present, unchanged path).
  Missing paths are never probed.

### Resume, limits, interrupts

* `--limit N` processes at most N resources; run again to continue. `--stratify [--seed S]` picks
  round-robin across folder × extension strata instead of by id (use it for pilots: plain
  `--limit` selects by id, so it is **not** representative).
* `--workers N` bounds concurrent ffprobe processes (default 4); `--timeout S` (default 30 s) kills a
  hung probe. Commands are argv lists, files are passed as `file:<path>`, no shell is involved.
* Only the main thread writes to SQLite; results are committed every `--batch-size` (default 200)
  as a checkpoint. Ctrl-C (or `--max-seconds N`) kills in-flight probes, commits finished work and
  exits (130 for Ctrl-C); run the same command to resume.
* Every invocation is recorded in `metadata_run`.

## Database safety

Opening an existing populated database at an older schema (`init`, `scan`) first writes a
consistent copy `<db>.bak-v<old>-<UTC>` with the SQLite backup API, then migrates. `backup DEST` makes
a copy at any time and works on any schema version. Migrations are append-only.

## Practical queries (L3-9)

The CLI `query` subcommand and Python `reports.query(conn, ...)` provide parameterized technical queries returning present paths and metadata. By default, known corrupted/unusable files (`valid=0`, probe errors, or MIDI errors) are excluded. Use `--include-errors` to opt in or `--only-errors` to isolate corrupted sources. Parameterized numeric bounds exclude NULLs explicitly.

```bash
# 1. Kick one-shots with sub frequency peak in 45-65 Hz range
python3 -m sample_indexer query --folder "Samples/Drums - One Shots/Kicks" --sub-peak-min 45 --sub-peak-max 65 --limit 10

# 2. Fast loops (>= 140 BPM) with measured tempo confidence >= 0.20
python3 -m sample_indexer query --bpm-min 140 --bpm-confidence-min 0.20 --limit 10

# 3. Stereo loops with high mono compatibility in the low end (stereo_corr_low >= 0.85)
python3 -m sample_indexer query --stereo-corr-low-min 0.85 --limit 10

# 4. Short percussion/hi-hat one-shots (duration <= 0.15s, sr=44100)
python3 -m sample_indexer query --duration-max 0.15 --sr 44100 --limit 10

# 5. Long atmospheric textures (duration >= 10.0s)
python3 -m sample_indexer query --duration-min 10.0 --limit 10

# 6. Dense MIDI files (density >= 6.0 notes/beat)
python3 -m sample_indexer query --ext mid --midi-density-min 6.0 --limit 10

# 7. Long MIDI arrangements with 100+ note events
python3 -m sample_indexer query --ext mid --midi-notes-min 100 --limit 10

# 8. Samples matching key 'Fmin' (from filename hint or measured chroma)
python3 -m sample_indexer query --key Fmin --limit 10

# 9. Basslines with fundamental pitch f0 between 40 and 90 Hz
python3 -m sample_indexer query --f0-min 40 --f0-max 90 --limit 10

# 10. Locate known corrupted or unparseable files
python3 -m sample_indexer query --only-errors

# Output structured JSON including isolated audio descriptors
python3 -m sample_indexer query --folder "Samples/Loops/Bass Loops" --bpm-min 130 --bpm-max 140 --json --limit 5
```

## Limitations

* Full-decode validity is a decoding verdict, not a musical-quality or category verdict; unresolved outcomes retain explicit reasons.
* Labels are path/name hints with unvalidated accuracy; folder conventions are the curator's, not acoustic.
* Unchanged paths are not re-hashed (same size + mtime edit goes unnoticed).
* ffprobe's `duration` is header-derived; a file whose header lies reports the claimed duration.
* `bpm_check` assumes 4/4 and is a hint only.
* ffprobe/ffmpeg must be installed (`sudo dnf install ffmpeg-free` or the RPM Fusion `ffmpeg`).
