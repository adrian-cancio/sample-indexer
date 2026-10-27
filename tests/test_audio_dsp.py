"""Independent numerical checks for bounded L3-6 streaming DSP."""
import json
import math
import unittest

import numpy as np

from sample_indexer.audio_dsp import AudioAccumulator, analyze


class AudioDSPTests(unittest.TestCase):
    sr = 8192

    def tone(self, hz, amplitude=0.5):
        return amplitude * np.sin(2 * np.pi * hz * np.arange(self.sr) / self.sr)

    def test_tone_numeric(self):
        x = self.tone(80)
        r = analyze(x[:, None], self.sr, nfft=4096)
        self.assertAlmostEqual(r['rms'], .5 / math.sqrt(2), places=12)
        self.assertAlmostEqual(r['sample_peak'], .5, places=12)
        self.assertAlmostEqual(r['crest_db'], 10 * math.log10(2), places=10)
        self.assertAlmostEqual(r['band_low'], .125, delta=.001)
        self.assertAlmostEqual(r['spectral_centroid_hz'], 80, delta=.5)
        self.assertAlmostEqual(r['spectral_rolloff85_hz'], 82, delta=2)
        self.assertLess(r['spectral_flatness'], .01)
        self.assertEqual(r['null_reasons']['stereo_corr'], 'mono')

    def test_noise_and_brightness(self):
        x = np.random.default_rng(73).normal(0, .1, (65536, 3))
        r = analyze(x, self.sr, nfft=2048)
        self.assertAlmostEqual(r['rms'], .1, delta=.001)
        self.assertAlmostEqual(sum(r[b] for b in ('band_sub', 'band_low', 'band_mid', 'band_high')), .01, delta=.0002)
        self.assertAlmostEqual(r['spectral_centroid_hz'], self.sr / 4, delta=40)
        self.assertAlmostEqual(r['spectral_rolloff85_hz'], .85 * self.sr / 2, delta=50)
        self.assertGreater(r['spectral_flatness'], .95)
        self.assertAlmostEqual(r['stereo_corr'], 0, delta=.02)
        dark = analyze(self.tone(80)[:, None], self.sr, nfft=2048)
        bright = analyze(self.tone(3000)[:, None], self.sr, nfft=2048)
        self.assertGreater(bright['spectral_centroid_hz'], dark['spectral_centroid_hz'] * 20)
        self.assertGreater(bright['band_high'], .12)

    def test_silence_and_no_low_energy(self):
        r = analyze(np.zeros((100, 2)), self.sr, nfft=1024)
        self.assertEqual(r['rms'], 0)
        self.assertEqual(r['band_low'], 0)
        self.assertEqual(r['null_reasons']['spectral_flatness'], 'silent')
        json.dumps(r, allow_nan=False)
        # A silent channel has no low energy even when its partner is loud.
        r = analyze(np.column_stack((self.tone(80), np.zeros(self.sr))), self.sr, nfft=2048)
        self.assertIsNone(r['stereo_corr_low'])
        self.assertEqual(r['null_reasons']['stereo_corr_low'], 'no_low_band_energy_in_channel')

    def test_antiphase(self):
        x = self.tone(80)
        r = analyze(np.column_stack((x, -x)), self.sr, nfft=4096)
        self.assertAlmostEqual(r['stereo_corr'], -1, places=12)
        self.assertAlmostEqual(r['stereo_corr_low'], -1, places=12)
        self.assertAlmostEqual(r['band_low'], .125, delta=.001)

    def test_stream_invariance_and_clipping(self):
        x = np.random.default_rng(5).uniform(-.3, .3, (5001, 4))
        x[0:3, 0] = 1
        x[16:18, 1] = -1
        x[-1, 2] = 1  # isolated peak is not clipping
        expected = analyze(x, self.sr, nfft=1024)
        acc = AudioAccumulator(self.sr, 4, nfft=1024)
        for start in range(0, len(x), 17):
            acc.update(x[start:start + 17])
            self.assertEqual(acc.buffer.shape, (1024, 4))
        result = acc.finish()
        self.assertEqual(result['clipped_samples'], 5)
        self.assertEqual(result['clipping_fraction'], 5 / x.size)
        self.assertAlmostEqual(result['dc_offset'], float(x.mean(axis=0)[np.argmax(np.abs(x.mean(axis=0)))]), places=14)
        for key, value in expected.items():
            if isinstance(value, float):
                self.assertAlmostEqual(result[key], value, places=12, msg=key)
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            acc.update(x)

    def test_parseval_including_endpoints(self):
        x = np.zeros((1003, 2))
        x[0] = [1, -1]
        x[-1] = [.5, .25]
        r = analyze(x, self.sr, nfft=1024)
        # Independent time-domain identity includes DC and sub-20 Hz omitted by bands.
        acc = AudioAccumulator(self.sr, 2, nfft=1024).update(x)
        acc.finish()
        power = acc.power * acc.hop / (acc.nfft * np.dot(acc.window, acc.window) * len(x))
        self.assertAlmostEqual(float(power.sum()), float(np.mean(x ** 2)), places=14)
        self.assertAlmostEqual(r['rms'], float(np.sqrt(np.mean(x ** 2))), places=14)

    def test_short_stream_parseval_public_bands(self):
        # An impulse has a flat two-sided spectrum. At sr=8192, nfft=1024,
        # the excluded <20 Hz bins are DC (weight 1), 8 and 16 Hz (weight 2).
        # Thus public band sums contain exactly 1019/1024 of its mean square,
        # independently of the implementation's window and normalization state.
        for frames in (1, 37, 256):  # single frame, sub-hop, exact hop
            with self.subTest(frames=frames):
                x = np.zeros((frames, 2))
                x[0] = [.5, -.25]
                expected_ms = (.5 ** 2 + .25 ** 2) / (2 * frames)
                acc = AudioAccumulator(self.sr, 2, nfft=1024)
                for row in x:
                    acc.update(row[None, :])
                result = acc.finish()
                bands = sum(result[name] for name in
                            ('band_sub', 'band_low', 'band_mid', 'band_high'))
                below20 = expected_ms * 5 / 1024
                self.assertAlmostEqual(bands + below20, expected_ms, places=14)
                self.assertAlmostEqual(result['rms'] ** 2, expected_ms, places=14)
                # Flat one-sided power has centroid exactly half Nyquist.
                self.assertAlmostEqual(result['spectral_centroid_hz'], 2048, places=10)
                self.assertEqual(result['clipped_samples'], 0)
                json.dumps(result, allow_nan=False)

    def test_clipping_one_frame_updates(self):
        # Runs cross every update boundary; isolated first/last peaks must not
        # count, and adjacent clipping in different channels is not a run.
        x = np.array([[1., 0.], [0., -1.], [1., -1.], [1., 0.],
                      [1., 0.], [0., 1.], [0., 0.], [1., -1.]])
        acc = AudioAccumulator(self.sr, 2, nfft=1024)
        for row in x:
            acc.update(row[None, :])
        result = acc.finish()
        self.assertEqual(result['clipped_samples'], 5)
        self.assertEqual(result['clipping_fraction'], 5 / 16)
        self.assertEqual(result['sample_peak'], 1)
        self.assertAlmostEqual(result['rms'] ** 2, 9 / 16, places=14)
        json.dumps(result, allow_nan=False)

    def test_validation(self):
        for kwargs in ({'sr': 0}, {'channels': 0}, {'nfft': 100}):
            args = dict(sr=self.sr, channels=2)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                AudioAccumulator(**args)
        acc = AudioAccumulator(self.sr, 2)
        with self.assertRaises(ValueError):
            acc.update(np.array([[float('nan'), 0.]]))
        self.assertEqual(acc.frames, 0)
        with self.assertRaises(ValueError):
            acc.finish()


if __name__ == '__main__':
    unittest.main()
