"""L3-5 DSP tests. Signals are synthesised with numpy, written as real WAV files and decoded by the
REAL ffmpeg (same path as the stage), so decode + DSP are exercised together."""

import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from sample_indexer import loops_dsp as L
from sample_indexer.lowend import ffmpeg_decode

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
SR = 44100


def write_wav16(path, data, sr=SR):
    data = np.asarray(data)
    ch = 1 if data.ndim == 1 else data.shape[1]
    pcm = np.round(np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def click_train(bpm, seconds, sr=SR, freq=1500.0, period_beats=1.0):
    n = int(round(sr * seconds))
    x = np.zeros(n)
    t = np.arange(int(0.01 * sr)) / sr
    burst = np.sin(2 * np.pi * freq * t) * np.exp(-t * 400)
    step = 60.0 / bpm * sr * period_beats
    k = 0
    while int(round(k * step)) + len(burst) <= n:
        i = int(round(k * step))
        x[i:i + len(burst)] += burst
        k += 1
    return 0.8 * x


def kick_hat(bpm, bars, sr=SR):
    """4-on-the-floor kick (sine sweep with decay) + off-beat hat, exactly `bars` bars long."""
    beat = 60.0 / bpm
    n = int(round(bars * 4 * beat * sr))
    x = np.zeros(n)
    kt = np.arange(int(0.18 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 * kt + 60 * (1 - np.exp(-kt * 30)) / 30)) * np.exp(-kt * 18)
    ht = np.arange(int(0.04 * sr)) / sr
    hat = np.random.default_rng(3).standard_normal(len(ht)) * np.exp(-ht * 120) * 0.3
    for b in range(bars * 4):
        i = int(round(b * beat * sr))
        k = max(0, min(len(kick), n - i))
        x[i:i + k] += kick[:k]
        j = int(round((b + 0.5) * beat * sr))
        m = max(0, min(len(hat), n - j))
        x[j:j + m] += hat[:m]
    return 0.7 * x


def tones(midis, seconds=6.0, cents=0.0, sr=SR, gain=0.07):
    t = np.arange(int(sr * seconds)) / sr
    x = np.zeros_like(t)
    for m in midis:
        f = 440.0 * 2 ** ((m - 69 + cents / 100.0) / 12)
        x += np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
    return gain * x


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class RealFfmpegDspTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sample-indexer-loopsdsp-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.n = 0

    def run_dsp(self, data, sr=SR, **kw):
        self.n += 1
        p = self.tmp / f"t{self.n}.wav"
        write_wav16(p, data, sr)
        d = ffmpeg_decode(str(p), 1 if np.ndim(data) == 1 else np.shape(data)[1])
        self.assertEqual(d.status, "ok", d.error)
        return L.analyze(d.samples, sr, **kw)

    # ---- tempo
    def test_click_trains_recover_tempo_within_lag_resolution(self):
        # tolerance: one autocorrelation lag at the envelope rate is 60/(rate*lag) of the tempo,
        # about 0.8 % at 160 BPM (hop 64 samples at 11.025 kHz); parabolic refinement must beat 0.5 %.
        for bpm in (120, 128, 155, 160):
            r = self.run_dsp(click_train(bpm, 10.0))
            self.assertIsNotNone(r["bpm_est"], bpm)
            self.assertLess(abs(r["bpm_est"] - bpm) / bpm, 0.005, (bpm, r["bpm_est"]))
            rels = {c["relation_to_best"] for c in r["tempo_candidates"]}
            self.assertIn("half", rels, "octave alternative must be kept")
            self.assertTrue(r["tempo_octave_ambiguous"])

    def test_double_time_pulse_keeps_slow_octave_candidate(self):
        # clicks every beat at 80 BPM plus off-beat clicks: the best may be 160, the 80 stays listed
        x = click_train(80, 12.0) + 0.9 * np.roll(click_train(80, 12.0), int(0.375 * SR))
        r = self.run_dsp(x)
        bpms = [c["bpm"] for c in r["tempo_candidates"]]
        self.assertTrue(any(abs(b - 80) < 1.5 for b in bpms), bpms)
        self.assertTrue(any(abs(b - 160) < 2.5 for b in bpms), bpms)

    def test_kick_hat_158_is_an_exact_whole_number_of_bars(self):
        r = self.run_dsp(kick_hat(158, 4))
        self.assertLess(abs(r["bpm_est"] - 158) / 158, 0.005, r["bpm_est"])
        lc = r["length_vs_est"]
        # the estimate carries its own error (<0.5 %), so the grid test must tolerate it:
        # 0.5 % of 6.08 s is 0.03 s, equal to the absolute tolerance
        self.assertIn(lc["status"], ("whole_bars",), lc)
        self.assertEqual(lc["nearest_bars"], 4)
        exact = L.length_check(r["duration_s"], 158.0, 0.0, L.DEFAULTS)
        self.assertEqual(exact["status"], "whole_bars")
        self.assertAlmostEqual(exact["bars"], 4.0, places=2)
        self.assertAlmostEqual(r["beats"], 16.0, delta=0.15)

    def test_noise_gives_low_confidence_and_no_tempo(self):
        r = self.run_dsp(0.2 * np.random.default_rng(5).standard_normal(SR * 8))
        self.assertIsNone(r["bpm_est"])
        self.assertEqual(r["tempo_reason"], "weak_periodicity")
        self.assertLess(r["tempo_confidence"], L.DEFAULTS["tempo_min_prominence"] + 0.05)

    def test_silence_gives_null_tempo_and_key(self):
        r = self.run_dsp(np.zeros((SR * 6, 2)))
        self.assertIsNone(r["bpm_est"])
        self.assertEqual(r["tempo_reason"], "silence")
        self.assertIsNone(r["key_est"])
        self.assertEqual(r["key_reason"], "silence")
        self.assertEqual(r["length_vs_est"]["status"], "unavailable")

    def test_too_short_for_tempo(self):
        r = self.run_dsp(click_train(120, 1.0))
        self.assertIsNone(r["bpm_est"])
        self.assertEqual(r["tempo_reason"], "too_short")

    # ---- key
    def test_c_major_and_a_minor_triads(self):
        r = self.run_dsp(tones([60, 64, 67]))
        self.assertEqual(r["key_est"], "Cmaj", r["key_candidates"])
        self.assertIsNone(r["key_reason"])
        self.assertEqual(len(r["chroma12"]), 12)
        self.assertGreater(r["chroma12"][0], 0.2)
        r = self.run_dsp(tones([57, 60, 64]))
        self.assertEqual(r["key_est"], "Amin", r["key_candidates"])
        self.assertGreater(r["key_margin"], L.DEFAULTS["key_min_margin"])

    def test_detuned_triad_reports_tuning_offset_and_same_key(self):
        for cents in (30.0, -30.0):
            r = self.run_dsp(tones([60, 64, 67], cents=cents))
            self.assertAlmostEqual(r["tuning_cents"], cents, delta=3.0)
            self.assertEqual(r["key_est"], "Cmaj")

    def test_chromatic_cluster_gives_no_key(self):
        r = self.run_dsp(tones(range(60, 72), gain=0.03))
        self.assertIsNone(r["key_est"])
        self.assertIn(r["key_reason"], ("flat_chroma", "low_tonal_strength", "ambiguous_profile", "noise_or_percussive"))

    def test_noise_gives_no_key(self):
        r = self.run_dsp(0.2 * np.random.default_rng(9).standard_normal(SR * 6))
        self.assertIsNone(r["key_est"])
        self.assertEqual(r["key_reason"], "noise_or_percussive")

    def test_percussion_gives_no_key(self):
        r = self.run_dsp(kick_hat(150, 4))
        self.assertIsNone(r["key_est"])
        self.assertIsNotNone(r["key_reason"])

    def test_single_low_end_pitch_gives_no_key_with_reason(self):
        t = np.arange(SR * 6) / SR
        r = self.run_dsp(0.4 * np.sin(2 * np.pi * 110.0 * t))
        self.assertIsNone(r["key_est"])
        self.assertEqual(r["key_reason"], "single_pitch_low_end")
        self.assertGreater(r["low_end_fraction"], 0.8)

    def test_single_pitch_mid_range_gives_no_key(self):
        t = np.arange(SR * 6) / SR
        r = self.run_dsp(0.4 * np.sin(2 * np.pi * 440.0 * t))
        self.assertIsNone(r["key_est"])
        self.assertEqual(r["key_reason"], "single_pitch")

    def test_antiphase_stereo_does_not_cancel(self):
        x = click_train(120, 8.0)
        r = self.run_dsp(np.stack([x, -x], axis=1))
        self.assertTrue(r["mono_mix"].startswith("loudest_channel"))
        self.assertLess(abs(r["bpm_est"] - 120), 1.0)

    def test_json_contains_settings_candidates_chroma_and_caveats(self):
        import json
        r = self.run_dsp(tones([60, 64, 67]))
        s = json.loads(json.dumps(r, allow_nan=False))
        for k in ("bpm_min", "bpm_max", "tempo_min_peak", "tempo_min_prominence", "key_band_hz", "reference_a_hz",
                  "key_window_s", "tempo_window_s"):
            self.assertIn(k, s["settings"])
        self.assertEqual(s["settings"]["key_band_hz"], [55.0, 2000.0])
        self.assertEqual(s["settings"]["reference_a_hz"], 440.0)
        self.assertEqual(len(s["chroma12"]), 12)
        self.assertTrue(s["caveats"])
        self.assertEqual(s["provenance"]["version"], L.VERSION)
        self.assertIn("tempo_candidates", s)


class PureTests(unittest.TestCase):
    def test_tempo_comparison_classes(self):
        c = L.compare_tempo
        self.assertEqual(c(128.0, 128.0), "exact")
        self.assertEqual(c(128.3, 128.0), "exact")
        self.assertEqual(c(129.5, 128.0), "tolerance")      # 1.2 % off
        self.assertEqual(c(64.0, 128.0), "half_double")
        self.assertEqual(c(256.2, 128.0), "half_double")
        self.assertEqual(c(100.0, 128.0), "conflict")
        self.assertEqual(c(None, 128.0), "unavailable")
        self.assertEqual(c(128.0, None), "unavailable")

    def test_key_comparison_classes(self):
        k = L.compare_key
        self.assertEqual(k("Cmaj", "Cmaj"), "same")
        self.assertEqual(k("A#min", "Bbmin"), "same")       # enharmonic
        self.assertEqual(k("Cmaj", "Amin"), "relative")
        self.assertEqual(k("Amin", "Cmaj"), "relative")
        self.assertEqual(k("D#min", "F#maj"), "relative")
        self.assertEqual(k("Cmaj", "Cmin"), "parallel")
        self.assertEqual(k("Cmaj", "Dmin"), "mismatch")
        self.assertEqual(k("Cmaj", "Emaj"), "mismatch")
        self.assertEqual(k(None, "Cmaj"), "unavailable")
        self.assertEqual(k("Cmaj", "weird"), "unavailable")

    def test_length_exact_truncated_noninteger(self):
        s = L.DEFAULTS
        bar = 240.0 / 128
        self.assertEqual(L.length_check(4 * bar, 128.0, 0.0, s)["status"], "whole_bars")
        self.assertEqual(L.length_check(4 * bar - 0.12, 128.0, 0.0, s)["status"], "truncated")
        self.assertEqual(L.length_check(4 * bar + 0.7, 128.0, 0.0, s)["status"], "noninteger")
        self.assertEqual(L.length_check(4 * bar + 0.7, 128.0, 1.0, s)["status"], "bars_plus_silent_tail")
        # 7 beats: whole beats but not a whole number of bars
        self.assertEqual(L.length_check(7 * 60 / 128, 128.0, 0.0, s)["status"], "whole_beats")
        self.assertEqual(L.length_check(3.3, None, 0.0, s)["status"], "unavailable")

    def test_implied_bpm_for_bar_counts(self):
        d = {r["bars"]: r["bpm"] for r in L.implied_bpm(7.5, L.DEFAULTS)}
        self.assertAlmostEqual(d[4], 128.0, places=2)
        self.assertAlmostEqual(d[2], 64.0, places=2)
        self.assertAlmostEqual(d[8], 256.0, places=2)
        self.assertAlmostEqual(d[1], 32.0, places=2)

    def test_invalid_and_over_cap_input(self):
        with self.assertRaises(ValueError):
            L.analyze(np.zeros((0, 1)), SR)
        with self.assertRaises(ValueError):
            L.analyze(np.zeros(100), SR)
        with self.assertRaises(ValueError):
            L.analyze(np.full((100, 1), np.nan), SR)
        with self.assertRaises(ValueError):
            L.analyze(np.zeros((int(L.MAX_SECONDS * 8000) + 8000, 1), dtype=np.float32), 8000)


if __name__ == "__main__":
    unittest.main()
