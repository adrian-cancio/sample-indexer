"""L3-5 loop DSP: tempo, beats/bars and key from decoded PCM (numpy FFT only).

Pure functions over an in-memory array: no I/O, no database, no filename/folder hints are
used for the *estimates* (hints are only compared afterwards, see `compare_tempo`/`compare_key`).
Every threshold lives in `DEFAULTS` and is echoed under `analysis["settings"]`.

Confidence numbers here are ordinal evidence (autocorrelation height/prominence, key-profile
correlation), NOT probabilities, and nothing here claims musical ground truth.

Tempo
  mono mix -> STFT (Hann, ~46 ms, hop = window/4) -> log-compressed magnitude -> positive
  spectral flux summed over bins -> local-mean-detrended onset envelope -> FFT autocorrelation
  -> local maxima inside [bpm_min, bpm_max] with parabolic lag refinement, height and
  topographic prominence. Candidates are ranked by a harmonic comb (autocorrelation summed at 1x, 2x
  and 4x the beat lag: a real beat also repeats every 2 and 4 beats, a spurious sub-beat peak does
  not). The comb was adopted after a 40-file pilot (see reports/L3-5.md); the gate (`tempo_min_peak`,
  `tempo_min_prominence`) still uses the best candidate's own height and prominence, so weak
  periodicity gives a NULL tempo
  with a reason. Half/double candidates stay in `tempo_candidates` and `tempo_octave_ambiguous`
  flags the case where one is nearly as strong. Beats = duration * bpm / 60 (estimated).

Length
  4/4 consistency of the decoded duration against a tempo (estimated or a hint), with a
  tolerance, trailing-silence evidence (10 ms RMS blocks, -60 dB relative to the loudest block)
  and the BPM implied by the duration for 1/2/4/8 bars.

Key
  Long Hann frames (~0.74 s) -> band 55..2000 Hz -> frames that are silent, noise-like (spectral
  flatness) or without concentrated spectral peaks are dropped -> interpolated spectral peaks ->
  tuning offset (circular, weighted, in cents against A=440) -> 12-bin chroma (per-frame normalised)
  -> Krumhansl-Schmuckler 24-profile correlation. NULL with a reason when silence, noise, a
  single/two pitch classes (typical for bass/kick material), flat chroma, low profile correlation
  or a small margin between the two best profiles.
"""

from __future__ import annotations

import math
import re

import numpy as np

VERSION = 1
MAX_SECONDS = 300.0
MAX_BYTES = 512 * 1024 * 1024  # the DSP refuses larger input arrays on its own

DEFAULTS: dict = {
    # --- tempo
    "bpm_min": 60.0, "bpm_max": 200.0,
    "tempo_window_s": 0.0464, "tempo_hop_div": 8, "flux_log_gamma": 1000.0, "flux_detrend_s": 0.4,
    "tempo_min_duration_s": 2.0,
    "tempo_min_peak": 0.15,          # min normalised autocorrelation height of the best peak
    "tempo_min_prominence": 0.10,    # min topographic prominence of the best peak
    "tempo_max_candidates": 8,
    "tempo_comb_harmonics": [1, 2, 4],  # candidate ranking: autocorrelation summed at these lag multiples
    "octave_ratio_tol": 0.03, "octave_min_relative_peak": 0.7,
    # --- hint comparison
    "hint_exact_bpm": 0.5, "hint_tolerance": 0.02,
    # --- length / grid
    "beats_per_bar": 4, "length_tol_abs_s": 0.03, "length_tol_rel": 0.002,
    "tail_block_s": 0.01, "tail_silence_db": -60.0, "implied_bars": [1, 2, 4, 8],
    # --- key
    "key_window_s": 0.74, "key_band_hz": [55.0, 2000.0], "reference_a_hz": 440.0,
    "key_quiet_frame_rel": 0.01, "key_silence_abs": 1e-4,
    "key_max_flatness": 0.06, "key_peak_rel": 0.10,
    "key_weight_exponent": 0.5, "key_min_tonal_frame_share": 0.15,
    "key_min_pitch_classes": 2.5,    # effective pitch classes (exp of chroma entropy)
    "key_min_strength_flat": 0.12, "key_min_strength": 0.25,
    "key_min_score": 0.70, "key_min_margin": 0.05, "key_low_end_hz": 120.0,
}

# Krumhansl-Schmuckler probe-tone profiles (major, minor), tonic first.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACC = {"": 0, "#": 1, "b": -1}
# max norm of (chroma - 1/12) for a unit-sum chroma (all mass on one class)
_MAX_DEV = math.sqrt((11 / 12) ** 2 + 11 * (1 / 12) ** 2)


def _profiles() -> np.ndarray:
    prof = np.array([np.roll(p, k) for p in (KS_MAJOR, KS_MINOR) for k in range(12)], dtype=float)
    prof -= prof.mean(axis=1, keepdims=True)
    prof /= np.linalg.norm(prof, axis=1, keepdims=True)
    return prof


_PROFILES = _profiles()


def _num(v, nd: int = 6):
    if v is None:
        return None
    v = float(v)
    return round(v, nd) if math.isfinite(v) else None


def _pow2(seconds: float, sr: float) -> int:
    return int(2 ** round(math.log2(max(seconds * sr, 16.0))))


# ----------------------------------------------------------------------------- hint parsing


def key_parts(key: str | None):
    """(pitch class 0..11, 'maj'|'min') from names like 'Fmin', 'Bbmaj', 'C#min'; else None."""
    m = re.fullmatch(r"([A-G])([#b]?)(maj|min)", key or "")
    if not m:
        return None
    return ((_PC[m[1]] + _ACC[m[2]]) % 12, m[3])


def compare_key(est: str | None, hint: str | None) -> str:
    """same | relative | parallel | mismatch | unavailable. Enharmonic spellings are equal."""
    a, b = key_parts(est), key_parts(hint)
    if a is None or b is None:
        return "unavailable"
    if a == b:
        return "same"
    if a[0] == b[0]:
        return "parallel"
    if a[1] != b[1]:
        minor_pc, major_pc = (a[0], b[0]) if a[1] == "min" else (b[0], a[0])
        if (minor_pc + 3) % 12 == major_pc:
            return "relative"
    return "mismatch"


def compare_tempo(est: float | None, hint: float | None, s: dict | None = None) -> str:
    """exact | tolerance | half_double | conflict | unavailable (est vs a name/folder hint)."""
    s = s or DEFAULTS
    if est is None or hint is None or hint <= 0:
        return "unavailable"
    tol = s["hint_tolerance"] * hint
    d = abs(est - hint)
    if d <= s["hint_exact_bpm"]:
        return "exact"
    if d <= tol:
        return "tolerance"
    if abs(est * 2 - hint) <= tol or abs(est / 2 - hint) <= tol:
        return "half_double"
    return "conflict"


# ----------------------------------------------------------------------------- helpers


def mono_mix(samples: np.ndarray) -> tuple[np.ndarray, str]:
    """float64 mono. Stereo uses the mid signal unless it cancels (anti-phase), then the loudest
    channel; the choice is reported, never silent."""
    ch = samples.shape[1]
    if ch == 1:
        return samples[:, 0].astype(np.float64), "mono"
    left = samples[:, 0].astype(np.float64)
    right = samples[:, 1].astype(np.float64)
    mid = 0.5 * (left + right)
    e_mid = float(np.mean(mid * mid))
    e_ch = 0.5 * (float(np.mean(left * left)) + float(np.mean(right * right)))
    if e_ch > 0 and e_mid < 0.25 * e_ch:
        return (left, "loudest_channel_0") if np.mean(left * left) >= np.mean(right * right) else (right, "loudest_channel_1")
    return mid, "mid"


def _stft_mag_chunks(x: np.ndarray, n: int, hop: int, chunk: int = 256):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    win = np.hanning(n)
    view = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    for i in range(0, len(view), chunk):
        yield np.abs(np.fft.rfft(view[i:i + chunk] * win, axis=1))


# ----------------------------------------------------------------------------- tempo


def onset_envelope(x: np.ndarray, sr: float, s: dict) -> tuple[np.ndarray, float]:
    n = _pow2(s["tempo_window_s"], sr)
    hop = max(1, n // int(s["tempo_hop_div"]))
    scale = 4.0 / n  # full-scale sine peak -> 1.0
    prev = None
    flux: list[np.ndarray] = []
    for mag in _stft_mag_chunks(x, n, hop):
        lm = np.log1p(s["flux_log_gamma"] * mag * scale)
        stacked = lm if prev is None else np.vstack([prev, lm])
        d = np.maximum(np.diff(stacked, axis=0), 0.0).sum(axis=1)
        flux.append(d if prev is not None else np.concatenate(([0.0], d)))
        prev = lm[-1:]
    env = np.concatenate(flux)
    rate = sr / hop
    w = max(3, int(round(s["flux_detrend_s"] * rate)))
    c = np.concatenate(([0.0], np.cumsum(env)))
    idx = np.arange(len(env))
    lo = np.clip(idx - w // 2, 0, len(env))
    hi = np.clip(idx + w // 2 + 1, 0, len(env))
    local = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return np.maximum(env - local, 0.0), rate


def _prominence(ac: np.ndarray, i: int, upto: int) -> float:
    peak = ac[i]
    left_min = peak
    for j in range(i - 1, 0, -1):
        if ac[j] > peak:
            break
        left_min = min(left_min, ac[j])
    else:
        pass
    right_min = peak
    for j in range(i + 1, upto):
        if ac[j] > peak:
            break
        right_min = min(right_min, ac[j])
    return float(peak - max(left_min, right_min))


def analyze_tempo(x: np.ndarray, sr: float, s: dict) -> dict:
    out = {"bpm_est": None, "tempo_confidence": None, "tempo_correlation": None, "tempo_reason": None,
           "tempo_candidates": [], "tempo_octave_ambiguous": False, "onset_frames": 0, "onset_rate_hz": None}
    if len(x) / sr < s["tempo_min_duration_s"]:
        out["tempo_reason"] = "too_short"
        return out
    if float(np.max(np.abs(x))) < s["key_silence_abs"]:
        out["tempo_reason"] = "silence"
        return out
    env, rate = onset_envelope(x, sr, s)
    n = len(env)
    out["onset_frames"], out["onset_rate_hz"] = n, _num(rate, 4)
    if float(env.max()) <= 1e-9:
        out["tempo_reason"] = "no_onsets"
        return out
    z = env - env.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    ac = np.fft.irfft(np.abs(np.fft.rfft(z, nfft)) ** 2)[:n]
    if ac[0] <= 0:
        out["tempo_reason"] = "no_onsets"
        return out
    ac = ac / ac[0]
    lag_lo = max(2, int(math.floor(rate * 60.0 / s["bpm_max"])) - 1)
    lag_hi = int(math.ceil(rate * 60.0 / s["bpm_min"])) + 1
    upto = min(n - 1, 2 * lag_hi + 2)
    if lag_hi >= n - 2:
        out["tempo_reason"] = "too_short"
        return out
    peaks = []
    for i in range(lag_lo, lag_hi + 1):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1]:
            den = ac[i - 1] - 2 * ac[i] + ac[i + 1]
            off = float(np.clip(0.5 * (ac[i - 1] - ac[i + 1]) / den, -0.5, 0.5)) if den else 0.0
            height = float(ac[i] - 0.25 * (ac[i - 1] - ac[i + 1]) * off)
            bpm = 60.0 * rate / (i + off)
            if s["bpm_min"] <= bpm <= s["bpm_max"]:
                lag = i + off
                comb = 0.0
                for k in s["tempo_comb_harmonics"]:  # periodicity at 1x, 2x, 4x the beat lag
                    j = int(round(lag * k))
                    if j < n:
                        comb += max(float(ac[max(j - 1, 0):j + 2].max()), 0.0)
                peaks.append({"bpm": bpm, "correlation": height, "comb_score": comb,
                              "prominence": _prominence(ac, i, upto)})
    peaks.sort(key=lambda p: -p["comb_score"])
    peaks = peaks[: s["tempo_max_candidates"]]
    if not peaks:
        out["tempo_reason"] = "no_periodic_peak"
        return out
    best = peaks[0]
    for p in peaks:
        ratio = p["bpm"] / best["bpm"]
        if abs(ratio - 2.0) <= 2.0 * s["octave_ratio_tol"]:
            rel = "double"
        elif abs(ratio - 0.5) <= 0.5 * s["octave_ratio_tol"]:
            rel = "half"
        elif p is best:
            rel = "best"
        else:
            rel = "other"
        p["relation_to_best"] = rel
        p["bpm"], p["correlation"], p["prominence"] = _num(p["bpm"], 3), _num(p["correlation"]), _num(p["prominence"])
        p["comb_score"] = _num(p["comb_score"])
    out["tempo_candidates"] = peaks
    out["tempo_correlation"] = best["correlation"]
    out["tempo_confidence"] = _num(max(best["prominence"], 0.0))
    out["tempo_octave_ambiguous"] = any(
        p["relation_to_best"] in ("half", "double") and p["correlation"] >= s["octave_min_relative_peak"] * best["correlation"]
        for p in peaks[1:])
    if best["correlation"] < s["tempo_min_peak"] or best["prominence"] < s["tempo_min_prominence"]:
        out["tempo_reason"] = "weak_periodicity"
        return out
    out["bpm_est"] = best["bpm"]
    return out


# ----------------------------------------------------------------------------- length


def trailing_silence(x: np.ndarray, sr: float, s: dict) -> dict:
    block = max(1, int(round(s["tail_block_s"] * sr)))
    nb = len(x) // block
    if nb == 0:
        return {"leading_silence_s": None, "trailing_silence_s": None}
    rms = np.sqrt(np.mean(x[: nb * block].reshape(nb, block) ** 2, axis=1))
    peak = float(rms.max())
    if peak <= 0:
        return {"leading_silence_s": len(x) / sr, "trailing_silence_s": len(x) / sr}
    active = np.where(rms > peak * 10 ** (s["tail_silence_db"] / 20.0))[0]
    lead = active[0] * block / sr
    tail = len(x) / sr - (active[-1] + 1) * block / sr
    return {"leading_silence_s": _num(lead, 4), "trailing_silence_s": _num(max(tail, 0.0), 4)}


def length_check(duration: float, bpm: float | None, tail_s: float, s: dict) -> dict:
    """4/4 consistency of `duration` with `bpm`: whole_bars | bars_plus_silent_tail | whole_beats |
    truncated | noninteger | unavailable. Duration-derived: it does not count beats in the audio."""
    if not bpm or bpm <= 0 or duration <= 0:
        return {"status": "unavailable", "bpm": bpm, "beats": None, "bars": None}
    beat = 60.0 / bpm
    bar = beat * s["beats_per_bar"]
    beats = duration / beat
    bars = duration / bar
    tol = s["length_tol_abs_s"] + s["length_tol_rel"] * duration
    nbars = round(bars)
    delta = duration - nbars * bar
    nbeats = round(beats)
    if nbars >= 1 and abs(delta) <= tol:
        status = "whole_bars"
    elif nbars >= 1 and delta > tol and (tail_s or 0.0) >= delta - tol:
        status = "bars_plus_silent_tail"
    elif nbeats >= 1 and abs(duration - nbeats * beat) <= tol:
        status = "whole_beats"
    elif nbars >= 1 and -beat < delta < -tol:
        status = "truncated"
    else:
        status = "noninteger"
    return {"status": status, "bpm": _num(bpm, 3), "beats": _num(beats, 4), "bars": _num(bars, 4),
            "nearest_bars": nbars, "grid_error_s": _num(delta, 5), "tolerance_s": _num(tol, 5),
            "assumption": "4/4, duration-derived (beats are not counted in the audio)"}


def implied_bpm(duration: float, s: dict) -> list[dict]:
    out = []
    for bars in s["implied_bars"]:
        bpm = s["beats_per_bar"] * 60.0 * bars / duration
        out.append({"bars": bars, "bpm": _num(bpm, 3), "in_range": bool(s["bpm_min"] <= bpm <= s["bpm_max"])})
    return out


# ----------------------------------------------------------------------------- key


def analyze_key(x: np.ndarray, sr: float, s: dict) -> dict:
    n = _pow2(s["key_window_s"], sr)
    hop = n // 2
    freq = np.fft.rfftfreq(n, 1.0 / sr)
    lo_hz, hi_hz = s["key_band_hz"]
    band = np.where((freq >= lo_hz) & (freq <= hi_hz))[0]
    f = freq[band]
    df = sr / n
    out = {"key_est": None, "key_reason": None, "key_score": None, "key_margin": None, "tonal_strength": None,
           "chroma12": [0.0] * 12, "tuning_cents": None, "tuning_concentration": None, "key_candidates": [],
           "low_end_fraction": None,
           "key_frames": {"total": 0, "silent": 0, "noisy": 0, "unpeaked": 0, "tonal": 0}}
    if float(np.max(np.abs(x))) < s["key_silence_abs"]:
        out["key_reason"] = "silence"
        return out
    xp = x if len(x) >= n else np.pad(x, (0, n - len(x)))
    csum = np.concatenate(([0.0], np.cumsum(xp * xp)))
    starts = np.arange(0, len(xp) - n + 1, hop)
    frame_rms = np.sqrt((csum[starts + n] - csum[starts]) / n)
    quiet = max(s["key_silence_abs"], s["key_quiet_frame_rel"] * float(frame_rms.max()))
    counts = out["key_frames"]
    counts["total"] = int(len(starts))
    per_frame: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    idx0 = 0
    for chunk in _stft_mag_chunks(xp, n, hop):
        for row in chunk:
            fi = idx0
            idx0 += 1
            if frame_rms[fi] < quiet:
                counts["silent"] += 1
                continue
            mag = row[band]
            total = float(mag.sum())
            if total <= 0:
                counts["silent"] += 1
                continue
            # spectral flatness of the POWER spectrum in the key band: ~1 for white noise,
            # near 0 for a few strong partials (measured on real loops: pads/saws 0.004-0.05,
            # drum breaks and crackle 0.04-0.21)
            pw = mag * mag
            flat = float(np.exp(np.mean(np.log(pw + 1e-30))) / pw.mean())
            if flat > s["key_max_flatness"]:
                counts["noisy"] += 1
                continue
            loc = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]))[0] + 1
            loc = loc[mag[loc] >= s["key_peak_rel"] * mag.max()]
            if len(loc) == 0:
                counts["unpeaked"] += 1
                continue
            a, b, c = (np.log(mag[loc - 1] + 1e-30), np.log(mag[loc] + 1e-30), np.log(mag[loc + 1] + 1e-30))
            den = a - 2 * b + c
            off = np.clip(np.divide(0.5 * (a - c), den, out=np.zeros(len(loc)), where=den != 0), -0.5, 0.5)
            hz = f[loc] + off * df
            midi = 69.0 + 12.0 * np.log2(hz / s["reference_a_hz"])
            w = mag[loc] ** s["key_weight_exponent"]
            per_frame.append((midi, w, hz))
            counts["tonal"] += 1
    if counts["tonal"] == 0:
        out["key_reason"] = "silence" if counts["silent"] == counts["total"] else "noise_or_percussive"
        return out
    midi_all = np.concatenate([p[0] for p in per_frame])
    w_all = np.concatenate([p[1] for p in per_frame])
    z = np.sum(w_all * np.exp(2j * np.pi * (midi_all - np.round(midi_all))))
    tune = float(np.angle(z) / (2 * np.pi))  # semitones, in [-0.5, 0.5]
    out["tuning_cents"] = _num(tune * 100.0, 3)
    out["tuning_concentration"] = _num(abs(z) / w_all.sum())
    chroma = np.zeros(12)
    low = tot = 0.0
    for midi, w, hz in per_frame:
        cf = np.zeros(12)
        np.add.at(cf, np.round(midi - tune).astype(int) % 12, w)
        chroma += cf / cf.sum()
        low += float(w[hz < s["key_low_end_hz"]].sum())
        tot += float(w.sum())
    chroma /= chroma.sum()
    out["low_end_fraction"] = _num(low / tot if tot else 0.0)
    out["chroma12"] = [_num(v) for v in chroma]
    strength = float(np.linalg.norm(chroma - 1.0 / 12) / _MAX_DEV)
    out["tonal_strength"] = _num(strength)
    cen = chroma - chroma.mean()
    scores = _PROFILES @ cen / max(float(np.linalg.norm(cen)), 1e-15)
    order = np.argsort(-scores)
    best = int(order[0])
    margin = float(scores[best] - scores[order[1]])
    name = lambda i: NOTE_NAMES[int(i) % 12] + ("maj" if i < 12 else "min")  # noqa: E731
    out["key_score"], out["key_margin"] = _num(scores[best]), _num(margin)
    out["key_candidates"] = [{"key": name(i), "score": _num(scores[i])} for i in order[:4]]
    nonsilent = counts["total"] - counts["silent"]
    # effective number of pitch classes = exp(entropy): 1 for one note, ~3 for a triad, 12 for flat
    classes = float(math.exp(-float(np.sum(chroma[chroma > 0] * np.log(chroma[chroma > 0])))))
    out["effective_pitch_classes"] = _num(classes, 3)
    reason = None
    if counts["tonal"] < max(1.0, s["key_min_tonal_frame_share"] * nonsilent):
        reason = "too_few_tonal_frames"
    elif classes < s["key_min_pitch_classes"]:
        reason = "single_pitch_low_end" if out["low_end_fraction"] >= 0.8 else "single_pitch"
    elif strength < s["key_min_strength_flat"]:
        reason = "flat_chroma"
    elif strength < s["key_min_strength"] or scores[best] < s["key_min_score"]:
        reason = "low_tonal_strength"
    elif margin < s["key_min_margin"]:
        reason = "ambiguous_profile"
    out["key_reason"] = reason
    if reason is None:
        out["key_est"] = name(best)
    return out


# ----------------------------------------------------------------------------- public API


def analyze(samples: np.ndarray, sr: float, *, bpm_hint: float | None = None, key_hint: str | None = None,
            overrides: dict | None = None) -> dict:
    """Analyse (frames, channels) float PCM. Raises ValueError for unusable/over-cap input."""
    s = dict(DEFAULTS)
    s.update(overrides or {})
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] < 1 or not sr or sr <= 0:
        raise ValueError("invalid PCM array")
    if samples.nbytes > MAX_BYTES or samples.shape[0] / float(sr) > MAX_SECONDS + 1e-6:
        raise ValueError("input exceeds caps (%.0f s, %d bytes)" % (MAX_SECONDS, MAX_BYTES))
    if not np.isfinite(samples).all():
        raise ValueError("non-finite samples")
    x, mix = mono_mix(samples)
    sr = float(sr)
    duration = len(x) / sr
    res = {"duration_s": _num(duration, 6), "mono_mix": mix}
    res.update(analyze_tempo(x, sr, s))
    res.update(analyze_key(x, sr, s))
    res.update(trailing_silence(x, sr, s))
    tail = res.get("trailing_silence_s") or 0.0
    est = res["bpm_est"]
    res["beats"] = _num(duration * est / 60.0, 4) if est else None
    res["bars_est"] = _num(duration * est / 60.0 / s["beats_per_bar"], 4) if est else None
    res["length_vs_est"] = length_check(duration, est, tail, s)
    res["implied_bpm"] = implied_bpm(duration, s)
    cmp_t = {"hint": bpm_hint, "relation": compare_tempo(est, bpm_hint, s)}
    if bpm_hint:
        cmp_t["length_vs_hint"] = length_check(duration, bpm_hint, tail, s)
        cmp_t["candidate_matching_hint"] = next(
            (c["bpm"] for c in res["tempo_candidates"] if compare_tempo(c["bpm"], bpm_hint, s) in ("exact", "tolerance")), None)
    res["tempo_hint_comparison"] = cmp_t
    res["key_hint_comparison"] = {"hint": key_hint, "relation": compare_key(res["key_est"], key_hint)}
    res["settings"] = dict(s, sample_rate_hz=sr, frames=int(len(x)), channels=int(samples.shape[1]),
                           max_seconds=MAX_SECONDS, max_bytes=MAX_BYTES)
    res["provenance"] = {"module": "sample_indexer.loops_dsp", "version": VERSION, "numpy": np.__version__,
                         "scope": "in_memory_pure_measurement"}
    res["caveats"] = [
        "Tempo is the strongest onset-autocorrelation peak; half/double candidates are kept, not resolved.",
        "Confidence values are ordinal evidence, not probabilities.",
        "Beats/bars are duration * estimated bpm (4/4 assumed); beats are not counted in the audio.",
        "Key profiles (Krumhansl-Schmuckler) assume Western tonal music; NULL is preferred to a guess.",
        "Hints (file/folder names) are compared after the fact and never used for the estimates.",
    ]
    return res
