"""L3-6 bounded, native-channel streaming measurements (no I/O).

Feed finite float arrays shaped (frames, channels) to AudioAccumulator.update;
finish once. Working memory is O(nfft * channels), independent of duration and
caller chunk size. Spectra use periodic Hann windows, 75% overlap and zero
padding at both ends: integrated power is whole-stream mean square (Parseval).
Centroid/rolloff/flatness describe the channel-mean POWER spectrum, including DC.
Clipping counts samples in runs of at least two, independently per channel.
"""
import math

import numpy as np

from .lowend_dsp import BANDS, STEREO_BAND

VERSION = 1


class AudioAccumulator:
    """Streaming DSP; first two channels define stereo, all channels define power."""

    def __init__(self, sr, channels, *, nfft=32768, clip_threshold=0.999):
        if isinstance(sr, bool) or not isinstance(sr, (int, float)) or not math.isfinite(sr) or sr < 2000:
            raise ValueError("sr must be finite and >= 2000")
        if isinstance(channels, bool) or not isinstance(channels, int) or not 1 <= channels <= 256:
            raise ValueError("channels must be an integer in [1, 256]")
        if isinstance(nfft, bool) or not isinstance(nfft, int) or not 16 <= nfft <= 131072 or nfft & (nfft - 1):
            raise ValueError("nfft must be a power of two in [16, 131072]")
        if not math.isfinite(clip_threshold) or clip_threshold <= 0:
            raise ValueError("clip_threshold must be finite and positive")
        self.sr, self.channels, self.nfft = float(sr), channels, nfft
        self.clip_threshold = float(clip_threshold)
        self.hop = nfft // 4
        self.window = np.hanning(nfft + 1)[:-1]
        self.buffer = np.zeros((nfft, channels))
        self.used = nfft - self.hop
        self.frames = self.clipped = 0
        self.peak = self.cross = 0.0
        self.sums = np.zeros(channels)
        self.squares = np.zeros(channels)
        self.last_clip = np.zeros(channels, dtype=bool)
        self.pending = np.zeros(channels, dtype=bool)
        self.freqs = np.fft.rfftfreq(nfft, 1 / self.sr)
        self.power = np.zeros(len(self.freqs))
        self.low = np.zeros(3)
        self.closed = False

    def _fft(self):
        z = np.fft.rfft(self.buffer * self.window[:, None], axis=0)
        scale = np.full(len(z), 2.0)
        scale[[0, -1]] = 1.0
        p = (z.real ** 2 + z.imag ** 2) * scale[:, None]
        self.power += p.mean(axis=1)
        if self.channels >= 2:
            mask = (self.freqs >= STEREO_BAND[0]) & (self.freqs < STEREO_BAND[1])
            self.low += [float(p[mask, 0].sum()), float(p[mask, 1].sum()),
                         float(((z[mask, 0] * z[mask, 1].conj()).real * scale[mask]).sum())]
        self.buffer[:-self.hop] = self.buffer[self.hop:]
        self.used -= self.hop

    def update(self, chunk):
        """Consume without retaining or modifying caller memory; empty chunks are OK."""
        if self.closed:
            raise ValueError("accumulator is finished")
        if not isinstance(chunk, np.ndarray) or chunk.ndim != 2 or chunk.shape[1] != self.channels or chunk.dtype.kind != 'f':
            raise ValueError("chunk must be a floating array (frames, channels)")
        # Validate the entire update before mutating state, with bounded temporaries.
        for start in range(0, len(chunk), self.nfft):
            c = chunk[start:start + self.nfft]
            if not np.isfinite(c).all() or (np.abs(c.astype(np.float64, copy=False)) > 1e100).any():
                raise ValueError("samples must be finite and bounded by 1e100")
        for start in range(0, len(chunk), self.hop):
            c = np.asarray(chunk[start:start + self.hop], dtype=np.float64)
            self.frames += len(c)
            self.peak = max(self.peak, float(np.abs(c).max()))
            self.sums += c.sum(axis=0)
            self.squares += np.einsum('ij,ij->j', c, c)
            if self.channels >= 2:
                self.cross += float(np.dot(c[:, 0], c[:, 1]))
            m = np.abs(c) >= self.clip_threshold
            prev = np.vstack((self.last_clip, m[:-1]))
            nxt = np.vstack((m[1:], np.zeros((1, self.channels), dtype=bool)))
            self.clipped += int((m & (prev | nxt)).sum()) + int((self.pending & m[0]).sum())
            self.pending = m[-1] & ~prev[-1]
            self.last_clip = m[-1].copy()
            pos = 0
            while pos < len(c):
                take = min(self.nfft - self.used, len(c) - pos)
                self.buffer[self.used:self.used + take] = c[pos:pos + take]
                self.used += take
                pos += take
                if self.used == self.nfft:
                    self._fft()
        return self

    def finish(self):
        """Finalize and return JSON-finite metrics, provenance and per-field null reasons."""
        if self.closed or not self.frames:
            raise ValueError("accumulator finished or empty")
        self.closed = True
        # Flush every frame overlapping real data, including the trailing Hann skirt.
        remaining = self.used
        while remaining > 0:
            self.buffer[self.used:] = 0
            self._fft()
            remaining -= self.hop
        norm = self.hop / (self.nfft * float(np.dot(self.window, self.window)) * self.frames)
        p = self.power * norm
        total = float(p.sum())
        ms = float(self.squares.mean() / self.frames)
        rms = math.sqrt(ms)
        means = self.sums / self.frames
        out = dict(sample_peak=self.peak, rms=rms,
                   crest_db=20 * math.log10(self.peak / rms) if rms else None,
                   dc_offset=float(means[np.argmax(np.abs(means))]),
                   dc_per_channel=means.tolist(), clipped_samples=self.clipped,
                   clipping_fraction=self.clipped / (self.frames * self.channels))
        reasons = {}
        if not rms:
            reasons['crest_db'] = 'silent'
        for name, lo, hi in BANDS:
            mask = (self.freqs >= lo) & (self.freqs < hi if hi is not None else self.freqs <= self.sr / 2)
            out[name] = float(p[mask].sum()) if mask.any() else None
            if out[name] is None:
                reasons[name] = 'no_bins_at_sample_rate_and_resolution'
        out['spectral_centroid_hz'] = float(np.dot(self.freqs, p) / total) if total else None
        out['spectral_rolloff85_hz'] = float(self.freqs[min(int(np.searchsorted(np.cumsum(p), .85 * total)), len(p) - 1)]) if total else None
        out['spectral_flatness'] = (float(math.exp(float(np.log(p).mean())) / p.mean()) if np.all(p > 0) else 0.0) if total else None
        if not total:
            for key in ('spectral_centroid_hz', 'spectral_rolloff85_hz', 'spectral_flatness'):
                reasons[key] = 'silent'
        for key, energy, cross in (
                ('stereo_corr', self.squares[:2] / self.frames, self.cross / self.frames),
                ('stereo_corr_low', self.low[:2] * norm, self.low[2] * norm)):
            reason = 'mono' if self.channels == 1 else None
            if reason is None and min(energy) <= 1e-12:
                reason = 'no_low_band_energy_in_channel' if key.endswith('_low') else 'no_energy_in_channel'
            out[key] = None if reason else max(-1.0, min(1.0, float(cross / math.sqrt(float(energy[0]) * float(energy[1])))))
            if reason:
                reasons[key] = reason
        out['null_reasons'] = reasons
        out['provenance'] = dict(module=__name__, version=VERSION, numpy=np.__version__,
                                 scope='complete_native_channel_stream', frames=self.frames,
                                 channels=self.channels, sr=self.sr, nfft=self.nfft, hop=self.hop,
                                 window='periodic_hann', spectrum='channel_mean_power_including_dc',
                                 stereo='first_two_uncentered_normalized_cross_power',
                                 clipping_threshold=self.clip_threshold, clipping_min_run=2,
                                 bands_hz=[list(b) for b in BANDS], low_band_hz=list(STEREO_BAND))
        return out


# Public runner-facing name; retain the descriptive accumulator name as well.
StreamingAnalyzer = AudioAccumulator


def analyze(samples, sr, **kwargs):
    """Convenience array API; streaming callers should use AudioAccumulator directly."""
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise ValueError("samples must be a 2-D numpy array")
    return AudioAccumulator(sr, samples.shape[1], **kwargs).update(samples).finish()
