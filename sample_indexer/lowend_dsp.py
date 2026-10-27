"""L3-4 low-end DSP: deterministic, pure measurements on an in-memory float64 array.

``analyze(samples, sr)`` takes ``samples`` as a ``numpy`` float64 array shaped
``(frames, channels)`` and returns a JSON-safe dict (no NaN/inf; undefined values are
``None`` with an accompanying reason). The input is never modified; no I/O is done here.

Caller contract: decode and cap the array (<= 120 s and <= 128 MiB; larger input raises
``ValueError``). Everything is chunked: no FFT is longer than ``nfft`` (<= 131072 points).

Method summary (all values are measurements, never labels):

* Time domain: sample peak, RMS (all channels), crest factor, DC (mean absolute
  per-channel mean), clipping = samples with ``|x| >= clip_threshold`` that sit in a run
  of >= 2 consecutive such samples (a single full-scale peak is not clipping).
* Bands: Welch average (Hann, 50 % overlap, ``seg`` bounded) of per-channel power spectra,
  averaged over channels (anti-phase channels cannot cancel). Values are mean-square
  (full scale 1.0 sine = 0.5) summed over bins in ``[20,60) [60,120) [120,2000) [2000,nyq]``.
  Content below 20 Hz is in no band. ``sub_peak_hz`` is the spectral maximum in 20..120 Hz
  (quadratic interpolation in log power); it is NOT the fundamental.
* Low stereo: the same STFT restricted to 20..120 Hz bins of channels 0 and 1. Correlation
  is the normalised in-band cross-spectrum (== correlation of the band-limited signals);
  ``None`` if either channel has no low-band energy. ``low_mono_loss`` is
  ``1 - E(L+R)/2 / mean(E(L), E(R))`` (0 = mono-safe, 1 = fully cancels).
* f0: per 0.4 s Hann frame, spectral peaks in 20..300 Hz (plus /2../4 sub-multiples) are
  scored with a harmonic comb over 20..2000 Hz, ``score = (comb_ratio - duty) / (1 - duty)``
  so noise scores ~0. A frame is accepted at ``score >= 0.8``; the highest f whose score is
  within 0.1 of the best is chosen (avoids octave-down errors). The file-level f0 is the
  median of the loud frames (within 6 dB of the loudest) and is reported only if >= half of
  them are valid and their spread is <= 15 %. A sweeping pitch gives ``f0_hz=None`` with
  reason ``pitch_unstable`` but a full contour (+ start/tail summary).
* Decay: amplitude envelope (20 ms RMS, 5 ms hop) from its peak to the first sample at
  -60 dB that stays below for >= 50 ms (or to the end if >= 20 ms remain). ``None`` when the
  level is never reached/sustained; the file length is never substituted.
"""
import math

import numpy as np

VERSION = 1

MAX_SECONDS = 120.0
MAX_BYTES = 128 * 1024 * 1024
MIN_SR = 2000

BANDS = (("band_sub", 20.0, 60.0), ("band_low", 60.0, 120.0),
         ("band_mid", 120.0, 2000.0), ("band_high", 2000.0, None))
STEREO_BAND = (20.0, 120.0)
SUB_PEAK_RANGE = (20.0, 120.0)

SETTINGS = {
    "clip_threshold": 0.999, "clip_min_run": 2,
    "spectrum_resolution_hz": 1.5, "spectrum_max_nfft": 65536,
    "silence_ms": 1e-12, "chunk_rows": 1 << 20,
    "f0_range_hz": (20.0, 300.0), "f0_analysis_hz": (20.0, 2000.0),
    "f0_frame_s": 0.4, "f0_hop_s": 0.1, "f0_min_frame_s": 0.1, "f0_max_candidates": 6,
    "f0_peak_floor_db": -25.0, "f0_min_score": 0.8, "f0_octave_margin": 0.1,
    "f0_max_duty": 0.5, "f0_loud_db": -6.0, "f0_quiet_db": -40.0,
    "f0_min_valid_fraction": 0.5, "f0_max_spread": 1.15,
    "contour_max_points": 64, "contour_head_points": 32,
    "decay_db": -60.0, "env_window_ms": 20.0, "env_hop_ms": 5.0,
    "decay_sustain_ms": 50.0, "decay_min_tail_ms": 20.0,
}


def _validate(samples, sr):
    if isinstance(sr, bool) or not isinstance(sr, (int, float)) or not math.isfinite(sr) or sr < MIN_SR:
        raise ValueError("sr must be a finite number >= %d" % MIN_SR)
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise ValueError("samples must be a 2-D numpy array (frames, channels)")
    if samples.dtype != np.float64:
        raise ValueError("samples must be float64")
    n, ch = samples.shape
    if n == 0 or ch == 0:
        raise ValueError("samples is empty")
    if samples.nbytes > MAX_BYTES or n / float(sr) > MAX_SECONDS + 1e-6:
        raise ValueError("input exceeds caps (%.0f s, %d bytes)" % (MAX_SECONDS, MAX_BYTES))


def _fnum(v, nd=6):
    """JSON-safe float (None for NaN/inf)."""
    v = float(v)
    return round(v, nd) if math.isfinite(v) else None


def _db(ms, floor=SETTINGS["silence_ms"]):
    return _fnum(10.0 * math.log10(ms)) if ms > floor else None


def _time_stats(x):
    """One chunked pass: peak, sum of squares, per-channel sums, clipped samples."""
    n, ch = x.shape
    step = SETTINGS["chunk_rows"]
    thr = SETTINGS["clip_threshold"]
    peak, sumsq, sums, clipped = 0.0, 0.0, np.zeros(ch), 0
    chan_sq = np.zeros(ch)
    for a in range(0, n, step):
        b = min(a + step, n)
        c = x[a:b]
        if not np.isfinite(c).all():
            raise ValueError("samples contain NaN or inf")
        peak = max(peak, float(np.abs(c).max()))
        sq = np.einsum("ij,ij->j", c, c)
        chan_sq += sq
        sums += c.sum(axis=0)
        lo, hi = max(a - 1, 0), min(b + 1, n)
        m = np.abs(x[lo:hi]) >= thr
        prev = np.zeros_like(m)
        nxt = np.zeros_like(m)
        prev[1:] = m[:-1]
        nxt[:-1] = m[1:]
        run = m & (prev | nxt)
        clipped += int(run[a - lo:b - lo].sum())
    return peak, chan_sq, sums, clipped


def _hann(length):
    if length < 2:
        return np.ones(length)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)


def _pow2_ge(v):
    return 1 << max(int(math.ceil(v)) - 1, 0).bit_length()


def _spectral_pass(x, sr):
    """Welch band powers, sub peak and low-band stereo cross-spectrum (bounded FFTs)."""
    n, ch = x.shape
    target = min(_pow2_ge(sr / SETTINGS["spectrum_resolution_hz"]), SETTINGS["spectrum_max_nfft"])
    seg = min(max(n, 4), target)
    seg -= seg % 4  # hop = seg/4 must be an integer for the constant-overlap property
    nfft = seg if seg == target else _pow2_ge(seg)
    hop = seg // 4
    win = _hann(seg)
    wsum = float(np.sum(win * win))
    # Zero-pad seg-hop samples on both sides so EVERY real sample is covered by exactly four
    # 75 %-overlapping Hann frames: sum of w^2 over the four shifts is constant (= 4*wsum/4 ... =
    # wsum*seg/hop/seg*seg), hence the band powers are an exact Parseval mean-square of the whole
    # file (a transient at the very start is not discounted by the window edge).
    pad = seg - hop
    starts = list(range(0, n + pad - hop + 1, hop))  # frame starts in padded coordinates
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    scale = np.full(freqs.shape, 2.0)
    scale[0] = 1.0
    if nfft % 2 == 0:
        scale[-1] = 1.0
    # sum over frames of windowed energy = (wsum/hop) * sum(x^2)  ->  mean-square = that * hop/(wsum*n)
    norm = scale * hop / (nfft * wsum * n)
    spec = np.zeros(len(freqs))
    slo, shi = STEREO_BAND
    sbins = (freqs >= slo) & (freqs < shi)
    cross = ll = rr = 0.0
    padded = np.zeros((n + 2 * pad, ch))  # one bounded copy: <= file size (+ small pad)
    padded[pad:pad + n] = x
    idx_row = np.arange(seg)
    batch = max(1, (1 << 20) // seg)
    for i in range(0, len(starts), batch):
        st = np.asarray(starts[i:i + batch])
        frames = padded[st[:, None] + idx_row[None, :]] * win[None, :, None]
        spc = np.fft.rfft(frames, n=nfft, axis=1)
        spec += (spc.real ** 2 + spc.imag ** 2).mean(axis=2).sum(axis=0)
        if ch >= 2:
            a, b = spc[:, sbins, 0], spc[:, sbins, 1]
            cross += float(np.sum((a * np.conj(b)).real * norm[sbins]))
            ll += float(np.sum((a.real ** 2 + a.imag ** 2) * norm[sbins]))
            rr += float(np.sum((b.real ** 2 + b.imag ** 2) * norm[sbins]))
    power = spec * norm
    return freqs, power, (cross, ll, rr) if ch >= 2 else None, nfft, seg


def _band_powers(freqs, power, nyq):
    out = {}
    for name, lo, hi in BANDS:
        if lo >= nyq:
            out[name] = None
            continue
        sel = (freqs >= lo) & ((freqs < hi) if hi is not None else (freqs <= nyq))
        out[name] = _fnum(power[sel].sum(), 9)
    return out


def _sub_peak(freqs, power):
    lo, hi = SUB_PEAK_RANGE
    idx = np.nonzero((freqs >= lo) & (freqs < hi))[0]
    if len(idx) == 0 or power[idx].max() <= 1e-12:
        return None, None
    # Only a genuine LOCAL maximum counts (>= both neighbours): the skirt of a strong component
    # just outside the range (e.g. 19 Hz leaking into the 20 Hz bin) is not a peak in range.
    inner = idx[(idx > 0) & (idx < len(power) - 1)]
    local = inner[(power[inner] >= power[inner - 1]) & (power[inner] >= power[inner + 1])]
    if len(local) == 0:
        return None, None
    k = int(local[np.argmax(power[local])])
    f = float(freqs[k])
    # interpolate only for an interior maximum: at a range edge the true peak may lie outside
    # [20,120) and the estimate must not leave the stated range
    if idx[0] < k < idx[-1] and power[k - 1] > 0 and power[k + 1] > 0:
        a, b, c = (math.log(power[k - 1]), math.log(power[k]), math.log(power[k + 1]))
        den = a - 2 * b + c
        if den < 0:
            f += 0.5 * (a - c) / den * float(freqs[1] - freqs[0])
    return _fnum(f, 3), _db(float(power[k]) + 0.0)


def _stereo_low(cross):
    """(mode-independent) low-band stereo metrics from (cross, ll, rr) spectra sums."""
    c, ll, rr = cross
    floor = 1e-12
    if ll <= floor or rr <= floor:
        return {"stereo_corr_low": None, "low_mono_loss": None, "low_side_energy": None,
                "low_side_ratio": None,
                "stereo_reason": "no_low_band_energy_in_channel"}
    mid = (ll + rr + 2 * c) / 4.0
    side = (ll + rr - 2 * c) / 4.0
    corr = max(-1.0, min(1.0, c / math.sqrt(ll * rr)))
    loss = max(0.0, min(1.0, 1.0 - mid / ((ll + rr) / 2.0)))
    return {"stereo_corr_low": _fnum(corr), "low_mono_loss": _fnum(loss),
            "low_side_energy": _fnum(max(side, 0.0), 9),
            "low_side_ratio": _fnum(max(side, 0.0) / (mid + max(side, 0.0))),
            "stereo_reason": None}


# ---------------------------------------------------------------- pitch

def _local_peaks(power, freqs, lo, hi, floor_rel, count):
    idx = np.nonzero((freqs >= lo) & (freqs <= hi))[0]
    idx = idx[(idx > 0) & (idx < len(power) - 1)]
    if len(idx) == 0:
        return []
    p = power[idx]
    ok = (p >= power[idx - 1]) & (p > power[idx + 1]) & (p >= p.max() * floor_rel) & (p > 0)
    cand = idx[ok]
    cand = cand[np.argsort(-power[cand], kind="stable")][:count]
    out = []
    for k in cand:
        a, b, c = (math.log(power[k - 1] + 1e-30), math.log(power[k]), math.log(power[k + 1] + 1e-30))
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den < 0 else 0.0
        out.append(float(freqs[k]) + max(-1.0, min(1.0, off)) * float(freqs[1] - freqs[0]))
    return out


def _comb_quality(cs, df, nbins, f, tol, fmax):
    """Best (quality, f) on a +/-1 % grid. quality = (comb - anti_comb) / total power."""
    grid = f * np.linspace(0.99, 1.01, 21)
    h = np.arange(1, int(fmax / f) + 2)
    best = (-1.0, f)
    for g in grid:
        hit = _window_sum(cs, df, nbins, h * g, tol)
        anti = _window_sum(cs, df, nbins, (h - 0.5) * g, tol)
        q = (hit - anti) / cs[-1]
        if q > best[0]:
            best = (float(q), float(g))
    return best


def _window_sum(cs, df, nbins, centres, tol):
    lo = np.clip(np.ceil((centres - tol) / df).astype(int), 0, nbins)
    hi = np.clip(np.floor((centres + tol) / df).astype(int) + 1, 0, nbins)
    return float(np.sum(cs[np.maximum(hi, lo)] - cs[lo]))


def _frame_pitch(power, freqs, tol):
    s = SETTINGS
    df = float(freqs[1] - freqs[0])
    lo_f, hi_f = s["f0_range_hz"]
    top = s["f0_analysis_hz"][1]
    peaks = _local_peaks(power, freqs, lo_f, hi_f, 10 ** (s["f0_peak_floor_db"] / 10),
                         s["f0_max_candidates"])
    if not peaks:
        return None, None, "no_low_peak"
    cs = np.concatenate(([0.0], np.cumsum(power)))
    cands = {}
    for p in peaks:
        for k in (1, 2, 3, 4):
            f = p / k
            if f >= lo_f and f > 4 * tol:
                cands[round(f, 1)] = f
    results = [(_comb_quality(cs, df, len(power), f, tol, top), f) for f in cands.values()]
    if not results:
        return None, None, "too_short_for_resolution"
    best_q = max(r[0][0] for r in results)
    q, f = max(((r[0][0], r[0][1]) for r in results if r[0][0] >= best_q - s["f0_octave_margin"]),
               key=lambda t: t[1])
    if q < s["f0_min_score"]:
        return None, _fnum(max(q, 0.0), 3), "weak_harmonic_evidence"
    return _fnum(f, 3), _fnum(min(q, 1.0), 3), None


def _pitch_track(x, sr):
    """Returns (contour, summary, f0, confidence, reason)."""
    s = SETTINGS
    n, ch = x.shape
    fl = int(round(s["f0_frame_s"] * sr))
    if n < fl:
        fl = n
    if fl < int(round(s["f0_min_frame_s"] * sr)):
        return [], _empty_summary(), None, None, "too_short"
    maxp = s["contour_max_points"]
    starts = list(range(0, n - fl + 1, max(int(round(s["f0_hop_s"] * sr)), -(-(n - fl) // (maxp - 1)) if n > fl else 1)))
    if starts[-1] != n - fl and len(starts) < maxp:
        starts.append(n - fl)
    ms = np.array([float(np.mean(x[a:a + fl] ** 2)) for a in starts])
    ref = float(ms.max())
    if ref <= s["silence_ms"]:
        return [], _empty_summary(), None, None, "silent"
    win = _hann(fl)
    nfft = _pow2_ge(2 * fl)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    tol = 1.25 * sr / fl
    fmask = (freqs >= s["f0_analysis_hz"][0]) & (freqs <= s["f0_analysis_hz"][1])
    quiet = ref * 10 ** (s["f0_quiet_db"] / 10)
    loud = ref * 10 ** (s["f0_loud_db"] / 10)
    contour, frames = [], []
    for a, m in zip(starts, ms):
        pt = {"t_ms": _fnum((a + fl / 2) * 1000.0 / sr, 1), "rms_db": _db(m, 1e-12),
              "f0_hz": None, "confidence": None}
        f = q = None
        if m < quiet:
            pt["reason"] = "quiet"
        else:
            seg = x[a:a + fl] - x[a:a + fl].mean(axis=0)
            spc = np.fft.rfft(seg * win[:, None], n=nfft, axis=0)
            p = ((spc.real ** 2 + spc.imag ** 2).mean(axis=1)) * fmask
            f, q, why = _frame_pitch(p, freqs, tol)
            pt["f0_hz"], pt["confidence"] = f, q
            if why:
                pt["reason"] = why
        contour.append(pt)
        frames.append((f, q, m >= loud))
    valid = [(f, q) for f, q, _ in frames if f is not None]
    loud_all = [(f, q) for f, q, is_loud in frames if is_loud]
    loud_valid = [(f, q) for f, q, is_loud in frames if is_loud and f is not None]
    summary = _contour_summary(contour)
    if not loud_valid:
        return contour, summary, None, None, "weak_harmonic_evidence"
    if len(loud_valid) < s["f0_min_valid_fraction"] * len(loud_all):
        return contour, summary, None, None, "too_few_valid_frames"
    fs = [f for f, _ in loud_valid]
    if max(fs) / min(fs) > s["f0_max_spread"]:
        return contour, summary, None, None, "pitch_unstable"
    mq = float(np.median([q for _, q in loud_valid]))
    if mq < s["f0_min_score"]:
        return contour, summary, None, None, "weak_harmonic_evidence"
    conf = mq * len(loud_valid) / len(loud_all)
    return contour, summary, _fnum(np.median(fs), 3), _fnum(conf, 3), "harmonic_evidence"


def _empty_summary():
    return {"pitch_start_hz": None, "pitch_tail_hz": None, "pitch_glide_semitones": None,
            "valid_frames": 0, "frames": 0}


def _contour_summary(contour):
    v = [p["f0_hz"] for p in contour if p["f0_hz"] is not None]
    out = {"valid_frames": len(v), "frames": len(contour), "pitch_start_hz": None,
           "pitch_tail_hz": None, "pitch_glide_semitones": None}
    if v:
        out["pitch_start_hz"] = _fnum(np.median(v[:3]), 3)
        out["pitch_tail_hz"] = _fnum(np.median(v[-3:]), 3)
        if len(v) >= 2:
            out["pitch_glide_semitones"] = _fnum(12 * math.log2(out["pitch_tail_hz"] / out["pitch_start_hz"]), 3)
    return out


# ---------------------------------------------------------------- decay

def _envelope(x, sr):
    """Mean-square envelope (all channels), window/hop from settings; bounded memory."""
    s = SETTINGS
    n = x.shape[0]
    win = max(int(round(sr * s["env_window_ms"] / 1000.0)), 1)
    hop = max(int(round(sr * s["env_hop_ms"] / 1000.0)), 1)
    if n < win:
        win = n
    nf = (n - win) // hop + 1
    out = np.empty(nf)
    block = 4096
    for f0 in range(0, nf, block):
        f1 = min(f0 + block, nf)
        a, b = f0 * hop, (f1 - 1) * hop + win
        sq = np.mean(x[a:b] ** 2, axis=1)
        cs = np.concatenate(([0.0], np.cumsum(sq)))
        k = np.arange(f1 - f0) * hop
        out[f0:f1] = (cs[k + win] - cs[k]) / win
    return out, win, hop


def _decay(x, sr):
    """Peak -> first sustained -60 dB amplitude crossing. Returns (ms, peak_ms, reason)."""
    s = SETTINGS
    env, win, hop = _envelope(x, sr)
    if float(env.max()) <= 1e-12:
        return None, None, "silent"
    p = int(np.argmax(env))
    peak_ms = (p * hop + win / 2) * 1000.0 / sr
    thr = env[p] * 10 ** (s["decay_db"] / 10)
    below = env <= thr
    sustain = max(int(round(s["decay_sustain_ms"] / s["env_hop_ms"])), 1)
    need_tail = max(int(round(s["decay_min_tail_ms"] / s["env_hop_ms"])), 1)
    # first index after the peak that starts a below-threshold run lasting `sustain` hops,
    # or a run that reaches the end of the file with at least `need_tail` hops
    n_env = len(env)
    above = np.concatenate(([0], np.cumsum(~below)))  # above[i] = #non-below in env[:i]
    cross = None
    seen_below = False
    for i in range(p + 1, n_env):
        if not below[i]:
            continue
        seen_below = True
        end = min(i + sustain, n_env)
        if above[end] - above[i] == 0 and (end - i >= sustain or end - i >= need_tail):
            cross = i
            break
    if cross is None:
        return None, _fnum(peak_ms, 1), "level_not_sustained" if seen_below else "level_not_reached"
    t = (cross * hop + win / 2) * 1000.0 / sr
    return _fnum(t - peak_ms, 1), _fnum(peak_ms, 1), None


# ---------------------------------------------------------------- public API

def analyze(samples, sr):
    """Measure low-end properties of ``samples`` (float64, frames x channels) at ``sr`` Hz."""
    _validate(samples, sr)
    sr = float(sr)
    n, ch = samples.shape
    peak, chan_sq, sums, clipped = _time_stats(samples)
    rms = math.sqrt(float(chan_sq.sum()) / (n * ch))
    means = sums / n
    dc = float(means[int(np.argmax(np.abs(means)))])
    freqs, power, cross, nfft, seg = _spectral_pass(samples, sr)
    nyq = sr / 2.0
    res = {"sample_peak": _fnum(peak), "rms": _fnum(rms, 9),
           "crest_db": _fnum(20 * math.log10(peak / rms)) if rms > 1e-12 else None,
           "dc_offset": _fnum(dc, 9), "clipped_samples": clipped,
           "clipping_fraction": _fnum(clipped / float(n * ch), 9)}
    res.update(_band_powers(freqs, power, nyq))
    res["sub_peak_hz"], res["sub_peak_db"] = _sub_peak(freqs, power)
    if ch >= 2:
        res["stereo_mode"] = "stereo" if ch == 2 else "multichannel_first_two"
        res.update(_stereo_low(cross))
    else:
        res["stereo_mode"] = "mono"
        res.update({"stereo_corr_low": None, "low_mono_loss": None, "low_side_energy": None,
                    "low_side_ratio": None, "stereo_reason": "mono"})
    contour, summary, f0, conf, why = _pitch_track(samples, sr)
    res.update({"f0_hz": f0, "f0_confidence": conf, "f0_reason": why,
                "pitch_contour": contour, "pitch_contour_summary": summary})
    d, pk, dwhy = _decay(samples, sr)
    res.update({"decay_ms": d, "decay_peak_ms": pk, "decay_reason": dwhy})
    res["settings"] = dict(
        {k: (list(v) if isinstance(v, tuple) else v) for k, v in SETTINGS.items()},
        sr=sr, frames=n, channels=ch, spectrum_nfft=nfft, spectrum_segment=seg,
        bands_hz=[[nm, lo, hi] for nm, lo, hi in BANDS], stereo_band_hz=list(STEREO_BAND),
        sub_peak_range_hz=list(SUB_PEAK_RANGE), max_seconds=MAX_SECONDS, max_bytes=MAX_BYTES)
    res["provenance"] = {"module": "sample_indexer.lowend_dsp", "version": VERSION,
                         "numpy": np.__version__, "scope": "in_memory_pure_measurement"}
    return res
