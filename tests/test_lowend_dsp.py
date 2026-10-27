"""Synthetic, memory-only checks for the L3-4 measurements."""
import json
import unittest

import numpy as np

from sample_indexer.lowend_dsp import analyze


class LowendTests(unittest.TestCase):
    sr = 8000

    def tone(self, hz=50, seconds=2, amplitude=0.5):
        t = np.arange(int(self.sr * seconds)) / self.sr
        return (amplitude * np.sin(2 * np.pi * hz * t))[:, None]

    def test_sines(self):
        for hz in (50, 60):
            with self.subTest(hz=hz):
                x = self.tone(hz)
                before = x.copy()
                r = analyze(x, self.sr)
                self.assertAlmostEqual(r['rms'], np.sqrt(0.125), places=5)
                self.assertAlmostEqual(r['sample_peak'], 0.5, places=3)
                self.assertAlmostEqual(r['crest_db'], 3.0103, places=2)
                self.assertAlmostEqual(r['f0_hz'], hz, delta=2)
                self.assertAlmostEqual(r['sub_peak_hz'], hz, delta=2)
                self.assertEqual(r['stereo_mode'], 'mono')
                self.assertIsNone(r['stereo_corr_low'])
                np.testing.assert_array_equal(x, before)
                json.dumps(r, allow_nan=False)
        self.assertGreater(analyze(self.tone(50), self.sr)['band_sub'], 0.11)
        self.assertGreater(analyze(self.tone(90), self.sr)['band_low'], 0.11)

    def test_silence_noise(self):
        for x in (np.zeros((16000, 2)), np.random.default_rng(10).normal(0, .1, (16000, 2))):
            r = analyze(x, self.sr)
            self.assertIsNone(r['f0_hz'])
            self.assertTrue(r['f0_reason'])
            json.dumps(r, allow_nan=False)
        r = analyze(np.zeros((1000, 2)), self.sr)
        self.assertIsNone(r['stereo_corr_low'])
        self.assertIsNone(r['decay_ms'])

    def test_stereo(self):
        x = self.tone()
        same = analyze(np.column_stack((x[:, 0], x[:, 0])), self.sr)
        anti = analyze(np.column_stack((x[:, 0], -x[:, 0])), self.sr)
        self.assertAlmostEqual(same['stereo_corr_low'], 1)
        self.assertAlmostEqual(anti['stereo_corr_low'], -1)
        self.assertAlmostEqual(same['band_sub'], anti['band_sub'])
        self.assertAlmostEqual(anti['low_mono_loss'], 1)
        self.assertAlmostEqual(same['low_mono_loss'], 0)
        high = self.tone(1000)
        r = analyze(np.column_stack((x[:, 0] + high[:, 0], x[:, 0] - high[:, 0])), self.sr)
        self.assertGreater(r['stereo_corr_low'], .99)
        self.assertGreater(r['band_mid'], .11)

    def test_decay(self):
        t = np.arange(3 * self.sr) / self.sr
        r = analyze(self.tone(seconds=3) * np.exp(-t[:, None] / .2), self.sr)
        self.assertAlmostEqual(r['decay_ms'], 1000 * .2 * np.log(1000), delta=70)
        self.assertIsNone(analyze(self.tone(), self.sr)['decay_ms'])

    def test_chirp(self):
        t = np.arange(3 * self.sr) / self.sr
        x = (.5 * np.sin(2 * np.pi * (100 * t - 10 * t*t)))[:, None]
        r = analyze(x, self.sr)
        valid = [p for p in r['pitch_contour'] if p['f0_hz'] is not None]
        self.assertGreater(len(valid), 3)
        self.assertGreater(valid[0]['f0_hz'], valid[-1]['f0_hz'] + 30)
        self.assertLessEqual(len(r['pitch_contour']), 64)

    def test_inharmonic_distorted_transient(self):
        x = self.tone(53) + self.tone(89) + self.tone(137)
        self.assertIsNone(analyze(x, self.sr)['f0_hz'])
        # tone buried in broadband noise (aperiodic distortion): no f0 claim
        noise = np.random.default_rng(3).normal(0, .3, (2 * self.sr, 1))
        r = analyze(0.1 * self.tone() + noise, self.sr)
        self.assertIsNone(r['f0_hz'])
        self.assertEqual(r['f0_reason'], 'weak_harmonic_evidence')
        # single click
        x = np.zeros((self.sr, 1)); x[0] = 1
        self.assertIsNone(analyze(x, self.sr)['f0_hz'])

    def test_clipped_periodic_keeps_f0_and_reports_clipping(self):
        # a hard-clipped sine is still strictly periodic: f0 stays correct, clipping is counted
        r = analyze(np.clip(10 * self.tone(), -1, 1), self.sr)
        self.assertAlmostEqual(r['f0_hz'], 50, delta=2)
        self.assertGreater(r['clipped_samples'], 0.5 * 2 * self.sr)
        self.assertGreater(r['clipping_fraction'], .5)
        # a lone full-scale sample is not a clipping run
        x = self.tone(); x[100] = 1.0
        self.assertEqual(analyze(x, self.sr)['clipped_samples'], 0)

    def test_missing_fundamental_vs_sub_peak(self):
        t = np.arange(2 * self.sr) / self.sr
        x = sum(.2 * np.sin(2 * np.pi * 50 * k * t) for k in (2, 3, 4, 5))[:, None]
        r = analyze(x, self.sr)
        self.assertAlmostEqual(r['sub_peak_hz'], 100, delta=2)   # spectral maximum in 20..120
        self.assertAlmostEqual(r['f0_hz'], 50, delta=2)          # periodicity, not the max

    def test_decay_retrigger_and_short(self):
        t = np.arange(3 * self.sr) / self.sr
        env = np.exp(-t / .1)
        env[int(1.5 * self.sr):] = 0.5            # later hit above -60 dB: crossing is the first sustained one
        r = analyze(self.tone(seconds=3) * env[:, None], self.sr)
        self.assertAlmostEqual(r['decay_ms'], 100 * np.log(1000), delta=60)
        # never reaches -60 dB within a short tail: unknown, not the file length
        r = analyze(self.tone(seconds=.3) * np.exp(-np.arange(int(.3 * self.sr)) / self.sr / .2)[:, None], self.sr)
        self.assertIsNone(r['decay_ms'])
        self.assertEqual(r['decay_reason'], 'level_not_reached')

    def test_unbounded_inputs_rejected(self):
        with self.assertRaises(ValueError):
            analyze(np.zeros((121 * self.sr, 1)), self.sr)

    def test_band_powers_sum_to_parseval_mean_square_for_transients(self):
        # Regression (found by the pilot's independent check): a Hann-windowed Welch average
        # under-weights samples near the file edges, so a kick whose energy sits at t=0 lost
        # most of its power. Band powers must now add up to the full mean-square (Parseval).
        rng = np.random.default_rng(3)
        t = np.arange(int(0.3 * 44100)) / 44100
        kick = np.exp(-t / 0.05) * np.sin(2 * np.pi * (50 + 120 * np.exp(-t / 0.02)) * t)
        for x in (kick[:, None], np.stack([kick, 0.5 * kick], axis=1), 0.2 * rng.normal(size=(5000, 1))):
            r = analyze(x, 44100)
            total = sum(r[k] for k in ('band_sub', 'band_low', 'band_mid', 'band_high'))
            ms = float((x ** 2).mean())
            sub20 = ms - total  # only content below 20 Hz (and DC) may be outside the bands
            self.assertGreaterEqual(sub20, -1e-6 * ms - 1e-9)
            self.assertLess(sub20, 0.15 * ms)  # <20 Hz share of these signals is small
        # whole-file Parseval for a pure 50 Hz tone: ms = A^2/2, all in band_sub
        r = analyze(self.tone(50, seconds=1.0), self.sr)
        self.assertAlmostEqual(r['band_sub'], 0.125, delta=0.125 * 0.03)

    def test_transient_at_file_start_keeps_its_band_power(self):
        x = np.zeros((4000, 1))
        x[:400, 0] = 0.5 * np.sin(2 * np.pi * 90 * np.arange(400) / self.sr)  # starts at sample 0
        r = analyze(x, self.sr)
        total = sum(r[k] for k in ('band_sub', 'band_low', 'band_mid', 'band_high'))
        self.assertAlmostEqual(total, float((x ** 2).mean()), delta=0.25 * float((x ** 2).mean()))

    def test_sub_peak_stays_inside_its_range(self):
        # energy peaking below 20 Hz must not be reported as a sub peak outside [20,120)
        r = analyze(self.tone(8, seconds=2) + 0.001 * self.tone(60, seconds=2), self.sr)
        self.assertTrue(r['sub_peak_hz'] is None or 20 <= r['sub_peak_hz'] < 120)

    def test_sub_peak_ignores_leakage_skirt_of_a_component_below_the_range(self):
        # Regression (real file BassStation 06.wav): a strong 19 Hz component leaks through the
        # Hann window into the 20 Hz bin; the reported peak must be the real in-range one (43 Hz).
        t = np.arange(8 * self.sr) / self.sr
        x = (1.0 * np.sin(2 * np.pi * 19 * t) + 0.1 * np.sin(2 * np.pi * 43 * t))[:, None]
        r = analyze(x, self.sr)
        self.assertAlmostEqual(r['sub_peak_hz'], 43.0, delta=2.0)

    def test_invalid(self):
        for x in (np.empty((0, 1)), np.array([[float('nan')]]), np.zeros(10)):
            with self.assertRaises(ValueError):
                analyze(x, self.sr)
        with self.assertRaises(ValueError):
            analyze(self.tone(), 0)


if __name__ == '__main__':
    unittest.main()
