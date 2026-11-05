"""Make a first guess at the notes in a hum, so correcting takes less time than typing.

This is only a starting point for the correction page. It is not the note engine.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.signal import medfilt

from .align import align

SR = 22050
HOP = 256
MIN_NOTE_SECONDS = 0.07
MAX_GAP_SECONDS = 0.03


def analyze(path: Path, melody: list[tuple[int, float]] | None = None) -> dict:
    """Return the clip length, its pitch curve and draft notes."""
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return analyze_audio(audio, melody)


def analyze_audio(audio: np.ndarray, melody: list[tuple[int, float]] | None = None) -> dict:
    """Analyze audio at SR. With a melody, the notes come from it instead of a guess."""
    f0, voiced, _ = librosa.pyin(
        audio, fmin=65.0, fmax=1047.0, sr=SR, frame_length=2048, hop_length=HOP
    )
    midi = np.full(f0.shape, np.nan)
    midi[voiced] = librosa.hz_to_midi(f0[voiced])
    times = librosa.frames_to_time(np.arange(midi.size), sr=SR, hop_length=HOP)
    result = {
        "duration": round(audio.size / SR, 4),
        "curve": {
            "hop": HOP / SR,
            "midi": [None if np.isnan(m) else round(float(m), 2) for m in midi],
        },
    }
    if melody is None:
        result["notes"] = segment(times, midi)
    else:
        rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=HOP)[0]
        aligned = align(midi, rms, HOP / SR, melody)
        result["notes"] = aligned.pop("notes")
        result["check"] = aligned
    return result


def segment(times: np.ndarray, midi: np.ndarray) -> list[dict]:
    """Group a pitch curve into notes, folding very short blips into their neighbours."""
    if midi.size == 0:
        return []
    hop = float(times[1] - times[0]) if times.size > 1 else HOP / SR
    voiced = ~np.isnan(midi)
    filled = np.where(voiced, midi, 0.0)
    smooth = medfilt(filled, kernel_size=5)
    rounded = np.where(voiced & (smooth > 0), np.rint(smooth), -1).astype(int)

    runs = []
    start = 0
    for i in range(1, rounded.size + 1):
        if i == rounded.size or rounded[i] != rounded[start]:
            runs.append([int(rounded[start]), start, i])
            start = i

    min_frames = int(MIN_NOTE_SECONDS / hop)
    merged = []
    for pitch, a, b in runs:
        if pitch >= 0 and b - a < min_frames and merged and merged[-1][0] >= 0:
            merged[-1][2] = b
        elif merged and merged[-1][0] == pitch:
            merged[-1][2] = b
        else:
            merged.append([pitch, a, b])

    notes = []
    max_gap = int(MAX_GAP_SECONDS / hop)
    for pitch, a, b in merged:
        if pitch < 0 or b - a < min_frames:
            continue
        if notes and notes[-1]["pitch"] == pitch and a - notes[-1]["_b"] <= max_gap:
            notes[-1]["_b"] = b
            continue
        notes.append({"pitch": pitch, "_a": a, "_b": b})
    return [
        {
            "start": round(float(n["_a"] * hop), 4),
            "end": round(float(n["_b"] * hop), 4),
            "pitch": n["pitch"],
        }
        for n in notes
    ]
