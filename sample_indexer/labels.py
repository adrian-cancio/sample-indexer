"""Provisional path/name labels (stage `labels`). No audio is read here.

Everything produced by this module is a HINT derived from folder names and file names:

* folder conventions are the library curator's filing choices, not sonic truth;
* `kind` / `category` are interpreted results, `label_evidence` keeps the raw source hints
  (matched folder rule, matched name tokens) so a later DSP stage can confirm or correct them;
* confidence is ordinal text (low | medium | high), deliberately not a probability;
* `bpm_name` / `key_name` only exist for explicit, bounded tokens ("125BPM", "Fmin");
  `beats_hint` is duration * bpm_name / 60 (a derived figure, never a measured one).

Pure functions only (no I/O), so the rules are unit-testable without a database.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

LABELS_VERSION = 1
LABEL_BASIS = "path_name_provisional"

BPM_MIN, BPM_MAX = 60.0, 240.0
# bpm_check tolerance: |duration - n * beat_length| <= BPM_TOL_ABS_S + BPM_TOL_REL * duration
BPM_TOL_ABS_S = 0.02
BPM_TOL_REL = 0.001  # 0.1%: a looser bound let 72.67 beats pass as "whole beats" (pilot finding)
BEATS_PER_BAR = 4  # 4/4 is assumed; the check is a plausibility hint only
ONE_SHOT_MAX_BEATS = 4.0  # a "one shot" at an explicit BPM spanning >= 1 bar is contradictory
LOOP_MIN_BEATS = 1.0  # a "loop" at an explicit BPM shorter than 1 beat is contradictory

CONF_ORDER = {"low": 0, "medium": 1, "high": 2}

# --------------------------------------------------------------------------- folder rules

# kind rules, keyed by casefolded (top, group, sub); sub None = any sub-folder of the group.
_FOLDER_KIND: dict[tuple[str, str, str | None], str] = {
    ("samples", "drums - one shots", None): "one_shot",
    ("samples", "loops", None): "loop",
    ("samples", "synths", "synth shots"): "one_shot",
    ("samples", "synths", "synth loops"): "loop",
}
# NOT mapped on purpose: other Synths folders and every FX folder (their names do not
# establish one_shot/loop), Unsorted, Vocals, Bonus.

_BROAD = {"drums", "fx", "synth", "cymbal"}

# category by (top, group, sub); (top, group, None) is the group-level default.
_FOLDER_CATEGORY: dict[tuple[str, str, str | None], str | None] = {
    ("samples", "drums - one shots", None): "drums",
    ("samples", "drums - one shots", "claps"): "clap",
    ("samples", "drums - one shots", "hi-hats"): "hihat",  # generic: not open/closed
    ("samples", "drums - one shots", "hi-hats closed"): "hihat_closed",
    ("samples", "drums - one shots", "hi-hats open"): "hihat_open",
    ("samples", "drums - one shots", "kicks"): "kick",
    ("samples", "drums - one shots", "misc"): "drums",
    ("samples", "drums - one shots", "percussion"): "percussion",
    ("samples", "drums - one shots", "rides"): "ride",
    ("samples", "drums - one shots", "rumble"): "rumble",
    ("samples", "drums - one shots", "snares"): "snare",
    ("samples", "fx & atmo", None): "fx",
    ("samples", "fx & atmo", "atmosphere"): "atmosphere",
    ("samples", "fx & atmo", "crashes"): "crash",
    ("samples", "fx & atmo", "impacts"): "impact",
    ("samples", "fx & atmo", "risers"): "riser",
    ("samples", "fx & atmo", "textures"): "texture",
    ("samples", "fx & atmo", "white noise"): "noise",
    ("samples", "loops", None): None,
    ("samples", "loops", "bass loops"): "bass",
    ("samples", "loops", "beat loops"): "drums",
    ("samples", "loops", "clap loops"): "clap",
    ("samples", "loops", "drum loops"): "drums",
    ("samples", "loops", "hi-hat loops"): "hihat",
    ("samples", "loops", "kick loops"): "kick",
    ("samples", "loops", "perc loops"): "percussion",
    ("samples", "loops", "ride & cymbal loops"): "cymbal",
    ("samples", "loops", "top loops"): "drums",
    # Dirty / Fat / Layer / Groove / Modular / Schranz / Sidechained loops: content unknown.
    ("samples", "synths", None): "synth",
    ("samples", "synths", "basslines"): "bass",
    ("samples", "synths", "leads"): "lead",
    ("samples", "vocals", None): "vocal",
    ("midi", "arps", None): "arp",
    ("midi", "basslines", None): "bass",
    ("midi", "drums", None): "drums",
    ("midi", "leads & melodies", None): "lead",
    ("midi", "pads & chords", None): "chords",
    ("midi", "sequences & hooks", None): "sequence",
}

# child -> parent: a more specific name token may refine a broader folder category.
_PARENT = {
    "kick": "drums", "snare": "drums", "clap": "drums", "hihat": "drums", "percussion": "drums",
    "tom": "drums", "rumble": "drums", "cymbal": "drums",
    "hihat_open": "hihat", "hihat_closed": "hihat",
    "ride": "cymbal", "crash": "cymbal",
    "atmosphere": "fx", "impact": "fx", "riser": "fx", "texture": "fx", "noise": "fx",
    "lead": "synth",
}


def _ancestors(cat: str) -> list[str]:
    out = []
    while cat in _PARENT:
        cat = _PARENT[cat]
        out.append(cat)
    return out


def _folder_parts(rel_path: str) -> tuple[str, str | None, str | None]:
    dirs = [p.casefold() for p in rel_path.split("/")[:-1]]
    top = dirs[0] if dirs else ""
    group = dirs[1] if len(dirs) > 1 else None
    sub = dirs[2] if len(dirs) > 2 else None
    return top, group, sub


def folder_hint(rel_path: str) -> dict:
    """Source hints from the folder path only: {'kind', 'category', 'rules'}."""
    top, group, sub = _folder_parts(rel_path)
    kind = category = None
    rules: list[str] = []
    if group is not None:
        for key in ((top, group, sub), (top, group, None)):
            if key in _FOLDER_KIND and kind is None:
                kind = _FOLDER_KIND[key]
                rules.append(f"kind:{'/'.join(p for p in key if p)}")
        for key in ((top, group, sub), (top, group, None)):
            if key in _FOLDER_CATEGORY:
                category = _FOLDER_CATEGORY[key]
                if category is not None:
                    rules.append(f"category:{'/'.join(p for p in key if p)}")
                break
    return {"kind": kind, "category": category, "rules": rules}


# --------------------------------------------------------------------------- name hints

_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
_NEGATORS = {"no", "without", "minus", "wo", "non"}
_KIND_TOKENS = {
    "loop": "loop", "loops": "loop",
    "shot": "one_shot", "shots": "one_shot", "oneshot": "one_shot", "oneshots": "one_shot",
}
_CATEGORY_TOKENS = {
    "kick": "kick", "kicks": "kick",
    "snare": "snare", "snares": "snare",
    "clap": "clap", "claps": "clap",
    "hat": "hihat", "hats": "hihat", "hihat": "hihat", "hihats": "hihat",
    "ride": "ride", "rides": "ride",
    "crash": "crash",
    "tom": "tom", "toms": "tom",
    "perc": "percussion", "percussion": "percussion",
    "bass": "bass", "bassline": "bass",
    "vocal": "vocal", "vocals": "vocal", "vox": "vocal",
}


def tokenize(stem: str) -> list[str]:
    """Lower-case alphabetic word tokens; splits on punctuation, digits and camelCase."""
    return [t.lower() for t in re.findall(r"[A-Za-z]+", _CAMEL.sub(" ", stem))]


def name_hint(filename: str) -> dict:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    tokens = tokenize(stem)
    kinds: dict[str, str] = {}
    cats: dict[str, str] = {}
    negated: list[str] = []
    negated_cats: set[str] = set()
    for i, tok in enumerate(tokens):
        if i > 0 and tokens[i - 1] in _NEGATORS and (tok in _CATEGORY_TOKENS or tok in _KIND_TOKENS):
            negated.append(tok)
            if tok in _CATEGORY_TOKENS:
                negated_cats.add(_CATEGORY_TOKENS[tok])
            continue
        if tok in _KIND_TOKENS:
            kinds[tok] = _KIND_TOKENS[tok]
        if tok in _CATEGORY_TOKENS:
            cats[tok] = _CATEGORY_TOKENS[tok]
    # "one shot" written as two words
    for i in range(len(tokens) - 1):
        if tokens[i] == "one" and tokens[i + 1] == "shot":
            kinds["one shot"] = "one_shot"
    cat_values = set(cats.values())
    if "hihat" in cat_values:
        if "open" in tokens and "closed" not in tokens:
            cat_values = (cat_values - {"hihat"}) | {"hihat_open"}
        elif "closed" in tokens and "open" not in tokens:
            cat_values = (cat_values - {"hihat"}) | {"hihat_closed"}
    return {
        "kinds": sorted(set(kinds.values())),
        "kind_tokens": sorted(kinds),
        "categories": sorted(cat_values),
        "category_tokens": sorted(cats),
        "negated_tokens": negated,
        "negated_categories": sorted(negated_cats),
    }


# --------------------------------------------------------------------------- bpm / key

_BPM_AFTER = re.compile(r"(?<![0-9])(\d{2,3})(?:[.,](\d{1,2}))?\s*[-_ ]?bpm(?![A-Za-z])", re.I)
_BPM_BEFORE = re.compile(r"(?<![A-Za-z])bpm[-_ ]?(\d{2,3})(?![0-9])", re.I)


def parse_bpm(filename: str) -> dict:
    """Explicit '<60-240> BPM' markers only (e.g. '125BPM', '128.5 bpm', 'BPM_140').

    Bare numbers ('Kick 160', 'TR909', 'SVE2 Perc 39') are never read as tempo.
    Returns {'value', 'raw', 'rejected', 'ambiguous'}; value is None unless exactly one
    distinct in-range explicit value was found.
    """
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    found: list[tuple[float, str]] = []
    rejected: list[str] = []
    for m in _BPM_AFTER.finditer(stem):
        value = float(m.group(1) + ("." + m.group(2) if m.group(2) else ""))
        (found if BPM_MIN <= value <= BPM_MAX else rejected).append((value, m.group(0)))
    for m in _BPM_BEFORE.finditer(stem):
        value = float(m.group(1))
        (found if BPM_MIN <= value <= BPM_MAX else rejected).append((value, m.group(0)))
    values = {v for v, _ in found}
    return {
        "value": found[0][0] if len(values) == 1 else None,
        "raw": found[0][1] if len(values) == 1 else None,
        "matches": [raw for _, raw in found],
        "rejected": [raw for _, raw in rejected],
        "ambiguous": len(values) > 1,
    }


# Uppercase note letter, optional lower-case 'b' or '#', explicit mode word, delimited on both
# sides by a non-alphanumeric character (or string edge). Chords/variants ("Cmin7th",
# "Emaj7min9", "Cmin2", "Gmin9") fail the right boundary and are NOT reported as keys.
_KEY = re.compile(r"(?<![A-Za-z0-9])([A-G])([#b]?)((?i:min(?:or)?|maj(?:or)?))(?![A-Za-z0-9])")
_KEY_LOOSE = re.compile(r"(?<![A-Za-z0-9])[A-G][#b]?(?i:min|maj)[A-Za-z0-9]+")


def parse_key(filename: str) -> dict:
    """Unambiguous '<Note>[#|b](min|minor|maj|major)' tokens, upper-case note letter."""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    found: list[tuple[str, str]] = []
    for m in _KEY.finditer(stem):
        mode = "min" if m.group(3).lower().startswith("min") else "maj"
        found.append((f"{m.group(1)}{m.group(2)}{mode}", m.group(0)))
    values = {v for v, _ in found}
    rejected = [m.group(0) for m in _KEY_LOOSE.finditer(stem) if not _KEY.fullmatch(m.group(0))]
    return {
        "value": found[0][0] if len(values) == 1 else None,
        "raw": found[0][1] if len(values) == 1 else None,
        "matches": [raw for _, raw in found],
        "rejected": rejected,
        "ambiguous": len(values) > 1,
    }


def bpm_duration_check(duration_s: float | None, bpm: float | None) -> tuple[float | None, str | None]:
    """(beats_hint, bpm_check). beats_hint = duration * bpm / 60, rounded to 2 decimals.

    bpm_check is a plausibility hint (4/4 assumed, absolute+relative time tolerance):
    'whole_bars' | 'whole_beats' | 'off_grid'. Not a measurement of the audio's tempo.
    """
    if not duration_s or not bpm or duration_s <= 0:
        return None, None
    beat_s = 60.0 / bpm
    beats = duration_s / beat_s
    tol = BPM_TOL_ABS_S + BPM_TOL_REL * duration_s
    check = "off_grid"
    n_beats = round(beats)
    if n_beats >= 1 and abs(duration_s - n_beats * beat_s) <= tol:
        check = "whole_bars" if n_beats % BEATS_PER_BAR == 0 else "whole_beats"
    return round(beats, 2), check


# --------------------------------------------------------------------------- interpretation


@dataclass
class LabelResult:
    kind: str | None = None
    kind_source: str | None = None
    kind_confidence: str | None = None
    category: str | None = None
    category_source: str | None = None
    category_confidence: str | None = None
    bpm_name: float | None = None
    bpm_name_raw: str | None = None
    key_name: str | None = None
    key_name_raw: str | None = None
    beats_hint: float | None = None
    bpm_check: str | None = None
    evidence: dict = field(default_factory=dict)

    def evidence_json(self) -> str:
        return json.dumps(self.evidence, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _interpret_kind(folder_kind: str | None, name_kinds: list[str]) -> tuple[str, str, str | None]:
    """(kind, source, confidence) for ONE path."""
    if len(name_kinds) > 1:
        return "unknown", "name_conflict", None
    name_kind = name_kinds[0] if name_kinds else None
    if folder_kind and name_kind:
        if folder_kind == name_kind:
            return folder_kind, "folder+name", "high"
        return "unknown", "folder_name_conflict", None
    if folder_kind:
        return folder_kind, "folder", "medium"
    if name_kind:
        return name_kind, "name", "low"
    return "unknown", "none", None


def _interpret_category(
    folder_cat: str | None, name_cats: list[str], negated_cats: list[str] | None = None
) -> tuple[str, str, str | None]:
    folder_conf = "low" if folder_cat in _BROAD else "medium"
    if folder_cat is not None and folder_cat in (negated_cats or []):
        # e.g. "Kick Loops/... no kick.wav": the name explicitly denies the folder's category.
        return "unknown", "folder_name_conflict", None
    if folder_cat is None:
        if len(name_cats) == 1:
            return name_cats[0], "name", "low"
        if len(name_cats) > 1:
            return "unknown", "name_ambiguous", None
        return "unknown", "none", None
    if not name_cats:
        return folder_cat, "folder", folder_conf
    if set(name_cats) == {folder_cat}:
        return folder_cat, "folder+name", "high" if folder_cat not in _BROAD else "medium"
    if folder_cat in name_cats:
        return folder_cat, "folder", "low"  # name also mentions other things: weaker hint
    # Related = name token is a descendant or ancestor of the folder category. The folder
    # (broad) category is kept: a name token never narrows it (the token stays in evidence).
    related = all(
        folder_cat in _ancestors(n) or n in _ancestors(folder_cat) for n in name_cats
    )
    if related:
        return folder_cat, "folder", folder_conf
    return "unknown", "folder_name_conflict", None


def _min_conf(confs: list[str | None]) -> str | None:
    confs = [c for c in confs if c]
    return min(confs, key=CONF_ORDER.__getitem__) if confs else None


def _merge(items: list[tuple[str, str, str | None]], what: str) -> tuple[str, str, str | None, str | None]:
    """Merge per-path (value, source, confidence). Returns value, source, conf, conflict note."""
    conflict_items = [i for i in items if i[0] == "unknown" and i[1] not in ("none",)]
    concrete = {i[0] for i in items if i[0] != "unknown"}
    if conflict_items:
        return "unknown", conflict_items[0][1], None, f"{what}: per-path conflict ({conflict_items[0][1]})"
    if len(concrete) > 1:
        return "unknown", "path_conflict", None, f"{what}: paths disagree {sorted(concrete)}"
    if not concrete:
        return "unknown", "none", None, None
    value = next(iter(concrete))
    chosen = [i for i in items if i[0] == value]
    sources = sorted({i[1] for i in chosen})
    return value, "+".join(sources) if len(sources) > 1 else sources[0], _min_conf([i[2] for i in chosen]), None


def interpret_resource(rel_paths: list[str], ext: str, duration_s: float | None) -> LabelResult:
    """Provisional labels for one resource from all of its present paths."""
    paths = sorted(rel_paths)
    res = LabelResult()
    per_path: list[dict] = []
    kinds: list[tuple[str, str, str | None]] = []
    cats: list[tuple[str, str, str | None]] = []
    bpms: list[dict] = []
    keys: list[dict] = []
    for rel in paths:
        filename = rel.rsplit("/", 1)[-1]
        fh, nh = folder_hint(rel), name_hint(filename)
        bpm, key = parse_bpm(filename), parse_key(filename)
        if ext == ".mid":
            kind = ("midi", "ext", "high")
        else:
            kind = _interpret_kind(fh["kind"], nh["kinds"])
        cat = _interpret_category(fh["category"], nh["categories"], nh["negated_categories"])
        kinds.append(kind)
        cats.append(cat)
        bpms.append(bpm)
        keys.append(key)
        entry = {
            "path": rel,
            "folder": {k: v for k, v in fh.items() if v},
            "name": {k: v for k, v in nh.items() if v},
            "kind": list(kind[:2]),
            "category": list(cat[:2]),
        }
        if bpm["matches"] or bpm["rejected"]:
            entry["bpm"] = {k: v for k, v in bpm.items() if v and k != "value"}
        if key["matches"] or key["rejected"]:
            entry["key"] = {k: v for k, v in key.items() if v and k != "value"}
        per_path.append(entry)

    notes: list[str] = []
    res.kind, res.kind_source, res.kind_confidence, note = _merge(kinds, "kind")
    if note:
        notes.append(note)
    res.category, res.category_source, res.category_confidence, note = _merge(cats, "category")
    if note:
        notes.append(note)

    bpm_values = {b["value"] for b in bpms if b["value"] is not None}
    if len(bpm_values) == 1 and not any(b["ambiguous"] for b in bpms):
        chosen = next(b for b in bpms if b["value"] is not None)
        res.bpm_name, res.bpm_name_raw = chosen["value"], chosen["raw"]
    elif len(bpm_values) > 1 or any(b["ambiguous"] for b in bpms):
        notes.append("bpm: ambiguous explicit markers, none stored")
    key_values = {k["value"] for k in keys if k["value"] is not None}
    if len(key_values) == 1 and not any(k["ambiguous"] for k in keys):
        chosen = next(k for k in keys if k["value"] is not None)
        res.key_name, res.key_name_raw = chosen["value"], chosen["raw"]
    elif len(key_values) > 1 or any(k["ambiguous"] for k in keys):
        notes.append("key: ambiguous explicit tokens, none stored")

    res.beats_hint, res.bpm_check = bpm_duration_check(duration_s, res.bpm_name)

    # Explicit duration/BPM contradiction -> kind unknown (only when kind came from hints).
    if res.beats_hint is not None and res.kind in ("loop", "one_shot"):
        if res.kind == "loop" and res.beats_hint < LOOP_MIN_BEATS:
            notes.append(f"kind: loop hint but beats_hint {res.beats_hint} < {LOOP_MIN_BEATS}")
            res.kind, res.kind_source, res.kind_confidence = "unknown", "bpm_duration_conflict", None
        elif res.kind == "one_shot" and res.beats_hint >= ONE_SHOT_MAX_BEATS:
            notes.append(f"kind: one_shot hint but beats_hint {res.beats_hint} >= {ONE_SHOT_MAX_BEATS}")
            res.kind, res.kind_source, res.kind_confidence = "unknown", "bpm_duration_conflict", None

    res.evidence = {
        "basis": LABEL_BASIS,
        "version": LABELS_VERSION,
        "paths": per_path,
        "duration_s": duration_s,
        "conflicts": notes,
    }
    if res.bpm_name is not None:
        res.evidence["bpm_check_tolerance"] = {
            "abs_s": BPM_TOL_ABS_S, "rel": BPM_TOL_REL, "beats_per_bar": BEATS_PER_BAR,
        }
    return res
