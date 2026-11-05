"""Work out which natural minor key a hum is in, and move its notes onto that scale.

Pitches come from the hum itself rather than rounded notes, so someone who hums slightly
sharp or flat throughout still lands on the notes they meant.
"""
from __future__ import annotations

import numpy as np

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NATURAL_MINOR = (0, 2, 3, 5, 7, 8, 10)
# How strongly each scale step suggests a minor key, from Krumhansl and Kessler's listening tests.
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PROFILE_WEIGHT = 0.25
CLEAR_LEAD = 0.05


def parse_tonic(text: str) -> int:
    """Read a key such as 'C', 'Cm', 'F#' or 'A minor' as a pitch class."""
    name = text.strip().lower().removesuffix("minor").strip().removesuffix("m").strip()
    name = name[:1].upper() + name[1:]
    flats = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    name = flats.get(name, name)
    if name not in NAMES:
        raise ValueError(f"Unknown key {text!r}. Try something like C, F# or Am.")
    return NAMES.index(name)


def hummed_pitches(notes: list[dict], curve: dict) -> np.ndarray:
    """The pitch actually hummed on each note, as a decimal note number."""
    hop = curve["hop"]
    midi = curve["midi"]
    pitches = []
    for note in notes:
        frames = midi[int(note["start"] / hop) : int(note["end"] / hop) + 1]
        voiced = [m for m in frames if m is not None]
        pitches.append(float(np.median(voiced)) if voiced else float(note["pitch"]))
    return np.array(pitches)


def tuning_offset(pitches: np.ndarray, durations: np.ndarray) -> float:
    """How far sharp or flat the whole hum sits from the standard note grid, in semitones."""
    angles = 2 * np.pi * pitches
    return float(np.angle(np.sum(durations * np.exp(1j * angles))) / (2 * np.pi))


def detect_tonic(pitches: np.ndarray, durations: np.ndarray) -> tuple[int, bool]:
    """Return the best natural minor tonic, and whether it clearly beat the next best."""
    weights = np.zeros(12)
    np.add.at(weights, np.rint(pitches).astype(int) % 12, durations)
    total = weights.sum()
    scores = np.zeros(12)
    for tonic in range(12):
        rotated = np.roll(weights, -tonic)
        fit = rotated[list(NATURAL_MINOR)].sum() / total
        match = np.corrcoef(rotated, _MINOR_PROFILE)[0, 1] if np.ptp(rotated) > 0 else 0.0
        scores[tonic] = fit + _PROFILE_WEIGHT * match
    ranked = np.argsort(scores)[::-1]
    return int(ranked[0]), bool(scores[ranked[0]] - scores[ranked[1]] >= CLEAR_LEAD)


def snap_pitch(pitch: float, tonic: int) -> int:
    """The natural minor note nearest to a hummed pitch."""
    center = int(round(pitch))
    candidates = [c for c in range(center - 2, center + 3) if (c - tonic) % 12 in NATURAL_MINOR]
    return min(candidates, key=lambda c: abs(c - pitch))


def snap_notes(notes: list[dict], curve: dict, tonic: int | None = None) -> tuple[list[dict], dict]:
    """Move notes onto a natural minor scale, detecting the key unless one is given."""
    if not notes:
        return [], {"tonic": tonic, "name": None, "clear": False, "moved": 0}
    durations = np.array([n["end"] - n["start"] for n in notes])
    pitches = hummed_pitches(notes, curve)
    in_tune = pitches - tuning_offset(pitches, durations)
    clear = True
    if tonic is None:
        tonic, clear = detect_tonic(in_tune, durations)
    snapped = [{**note, "pitch": snap_pitch(p, tonic)} for note, p in zip(notes, in_tune)]
    moved = sum(new["pitch"] != int(round(p)) for new, p in zip(snapped, in_tune))
    return snapped, {"tonic": tonic, "name": f"{NAMES[tonic]} minor", "clear": clear, "moved": moved}
