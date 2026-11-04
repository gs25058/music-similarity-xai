from dataclasses import dataclass
from math import gcd
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 24000


@dataclass
class Audio:
    path: str
    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self):
        return len(self.samples) / self.sample_rate


@dataclass
class Window:
    index: int
    start: float
    end: float
    samples: np.ndarray
    valid_samples: int

    def metadata(self):
        return dict(index=self.index, start=self.start, end=self.end,
                    actual_duration=self.end - self.start,
                    padded_duration=len(self.samples) / SAMPLE_RATE,
                    padding_samples=len(self.samples) - self.valid_samples)


def load_audio(path) -> Audio:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        x, sr = sf.read(path, always_2d=True, dtype="float32")
        x = x.mean(axis=1)
        if sr != SAMPLE_RATE:
            g = gcd(sr, SAMPLE_RATE)
            x = resample_poly(x, SAMPLE_RATE // g, sr // g)
    except (RuntimeError, sf.LibsndfileError if hasattr(sf, 'LibsndfileError') else RuntimeError):
        # ffmpeg is an explicit optional decoder for compressed formats.
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
                                 "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
                                check=True, capture_output=True)
        x = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    return Audio(str(path), np.asarray(x, dtype=np.float32))


def windows(audio: Audio, seconds=10.0, hop=5.0):
    if not 0 < hop <= seconds:
        raise ValueError("Require 0 < hop <= window duration")
    size, step = round(seconds * SAMPLE_RATE), round(hop * SAMPLE_RATE)
    if size < 2048 or step < 1:
        raise ValueError("Window too short")
    for index, start in enumerate(range(0, len(audio.samples), step)):
        end = min(start + size, len(audio.samples))
        chunk = audio.samples[start:end]
        yield Window(index, start / SAMPLE_RATE, end / SAMPLE_RATE,
                     np.pad(chunk, (0, size - len(chunk))), len(chunk))
        if end == len(audio.samples):
            break
