"""Make fake hums with known notes, for testing the tools before any real recordings."""
from __future__ import annotations

import numpy as np

DEMO_MELODIES = [
    [(60, 0.4), (63, 0.4), (67, 0.8), (65, 0.4), (63, 0.4), (60, 1.0)],
    [(55, 0.25), (58, 0.25), (60, 0.25), (63, 0.5), (60, 0.25), (58, 0.25), (55, 0.75)],
    [(67, 1.2), (70, 0.6), (72, 1.2), (67, 1.0)],
]


def make_hum(melody, sr: int = 44100, seed: int = 0, gap: float = 0.08):
    """Render (midi_pitch, seconds) pairs with a pitch scoop, vibrato, harmonics and noise."""
    rng = np.random.default_rng(seed)
    lead_in = 0.3
    truth = []
    pieces = [np.zeros(int(lead_in * sr))]
    now = lead_in
    for pitch, seconds in melody:
        n = int(seconds * sr)
        t = np.arange(n) / sr
        scoop = -0.5 * np.exp(-t / 0.02)
        vibrato = 0.15 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.3, 0, 1)
        drift = rng.uniform(-0.12, 0.12)
        hz = 440.0 * 2 ** ((pitch + drift + scoop + vibrato - 69) / 12)
        phase = 2 * np.pi * np.cumsum(hz) / sr
        tone = sum(np.sin(k * phase) / k ** 1.5 for k in range(1, 7))
        envelope = np.clip(t / 0.015, 0, 1) * np.clip((seconds - t) / 0.03, 0, 1)
        pieces.append(0.3 * tone * envelope)
        truth.append({"start": round(now, 4), "end": round(now + seconds, 4), "pitch": pitch})
        now += seconds + gap
        pieces.append(np.zeros(int(gap * sr)))
    pieces.append(np.zeros(int(0.3 * sr)))
    audio = np.concatenate(pieces)
    audio += rng.normal(0, 0.002, audio.size)
    return audio.astype(np.float32), truth
