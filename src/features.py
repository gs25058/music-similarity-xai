"""Deterministic DSP baseline; pitch estimates are mixed-audio proxies."""
import numpy as np
from scipy.ndimage import maximum_filter1d

FPS = 20
MIDI_MIN, MIDI_MAX = 24, 107
PITCH_BINS = 85  # 84 pitches plus unvoiced
INPUT_DIM = 86
FACTORS = ("pitch_intervals", "rhythm", "harmony", "accompaniment_bass")
ALL_FACTORS = (*FACTORS, "phrase_structure", "melody", "timbre")


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-8)


def frame_features(samples, valid_samples=None):
    """20 Hz centered FFT frames, 84 MIDI energy bins + onset + energy."""
    valid_samples = len(samples) if valid_samples is None else valid_samples
    nfft, hop = 2048, 1200
    centers = np.arange(0, len(samples), hop)
    padded = np.pad(samples, (nfft // 2, nfft // 2))
    frames = np.stack([padded[c:c+nfft] for c in centers])
    spec = np.abs(np.fft.rfft(frames * np.hanning(nfft), axis=1))
    frequencies = 440 * 2 ** ((np.arange(MIDI_MIN, MIDI_MAX+1) - 69) / 12)
    bins = frequencies / 24000 * nfft
    # Interpolate energies, with local spectral maxima for low FFT resolution.
    local = maximum_filter1d(spec, size=3, axis=1)
    energies = np.stack([np.interp(bins, np.arange(local.shape[1]), row) for row in local])
    energy = np.sqrt(np.mean(frames ** 2, axis=1))
    onset = np.maximum(0, np.diff(energy, prepend=energy[0]))
    onset /= max(float(onset.max()), 1e-6)
    log = np.log1p(energies)
    log /= max(float(log.max()), 1e-6)
    valid = centers < valid_samples
    features = np.c_[log, onset, np.clip(energy * 5, 0, 1)].astype(np.float32)
    features[~valid] = 0
    chroma = np.stack([energies[:, (np.arange(84)+MIDI_MIN) % 12 == pc].sum(1)
                       for pc in range(12)], axis=1)
    chroma /= np.maximum(chroma.max(axis=1, keepdims=True), 1e-6)
    return features, valid, chroma.astype(np.float32)


def _resize(x, length):
    return np.interp(np.linspace(0, len(x)-1, length), np.arange(len(x)), x)


def _intervals(pitch):
    differences = np.clip(np.diff(pitch), -12, 12)
    return np.bincount((differences+12).astype(int), minlength=25)


def dsp_encode(window):
    frames, valid, chroma = frame_features(window.samples, window.valid_samples)
    f = frames[valid]
    if len(f) < 2 or np.max(f[:, -1]) < 1e-4:
        return {}, {"status": "unavailable", "reason": "silence_or_too_short"}
    high = f[:, 31:67].argmax(axis=1) + 55
    low = f[:, :30].argmax(axis=1) + 24
    onset = f[:, -2]
    corr = np.correlate(onset-onset.mean(), onset-onset.mean(), mode="full")[len(onset)-1:]
    rhythmic = np.r_[_resize(corr, 60), _resize(onset, 40)]
    # Cosine over temporal chroma, not a claimed chord transcriber.
    harmony = np.concatenate([_resize(chroma[valid, pc], 20) for pc in range(12)])
    emb = dict(pitch_intervals=unit(_intervals(high)), rhythm=unit(rhythmic),
               harmony=unit(harmony), accompaniment_bass=unit(np.r_[_intervals(low), _resize(onset, 40)]))
    info = dict(status="proxy", pitch_source="mixed_audio_dominant_high_register",
                accompaniment_source="mixed_audio_low_register_proxy_not_isolated_bass",
                harmony_source="spectral_chroma_not_chord_annotation",
                chroma_mean=chroma[valid].mean(0).round(4).tolist(),
                dominant_pitch_midi_median=float(np.median(high)),
                low_register_midi_median=float(np.median(low)),
                onset_times_seconds=(np.flatnonzero(onset > .5) / FPS + window.start).round(3).tolist(),
                missing=["isolated_main_melody", "verified_chords", "isolated_bass", "beat_labels"])
    return emb, info
