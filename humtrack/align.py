"""Line up a hum with the melody it copied.

The melody supplies the correct notes, so this only has to find when each one was hummed
and whether the hum really landed on it. The hum can be in any key.

Repeated notes sound the same, so pitch can't separate them. They are found as one group
first, then split at the quiet moments between them.
"""
from __future__ import annotations

import numpy as np

OFF_LIMIT = 0.6
MIN_NOTE_SECONDS = 0.06
MIN_SHARE_OF_EXPECTED = 0.15
MAX_GAP_SECONDS = 0.08
_CAP = 3.0
_UNVOICED_COST = 0.05
_STEP_COST = 0.3


def _folded(difference: np.ndarray) -> np.ndarray:
    """Pitch difference ignoring octaves, since pitch trackers often jump by one."""
    return (difference + 6.0) % 12.0 - 6.0


def _dips(rms: np.ndarray, size: int, window: int = 15) -> np.ndarray:
    """Frames noticeably quieter than their surroundings, where a repeated note can restart."""
    rms = np.resize(rms, size) if rms.size != size else rms
    padded = np.pad(rms, window, mode="edge")
    peaks = np.lib.stride_tricks.sliding_window_view(padded, 2 * window + 1).max(axis=1)
    return rms < 0.6 * peaks


def _path(midi: np.ndarray, pitches: np.ndarray) -> tuple[float, np.ndarray]:
    """Best group for each frame, moving through the melody in order without skipping."""
    voiced = ~np.isnan(midi)
    cost = np.full((midi.size, pitches.size), _UNVOICED_COST)
    cost[voiced] = np.minimum(np.abs(_folded(midi[voiced, None] - pitches[None, :])), _CAP)

    best = np.full(pitches.size, np.inf)
    best[0] = cost[0, 0]
    moved = np.zeros(cost.shape, dtype=bool)
    for i in range(1, midi.size):
        advance = np.concatenate(([np.inf], best[:-1])) + _STEP_COST
        moved[i] = advance < best
        best = np.minimum(best, advance) + cost[i]

    path = np.zeros(midi.size, dtype=int)
    k = pitches.size - 1
    for i in range(midi.size - 1, -1, -1):
        path[i] = k
        if moved[i, k]:
            k -= 1
    return float(best[-1]), path


def _main_run(frames: np.ndarray, max_gap: int) -> np.ndarray:
    """The longest stretch of frames, so a stray blip doesn't stretch the note."""
    if frames.size == 0:
        return frames
    runs = np.split(frames, np.where(np.diff(frames) > max_gap)[0] + 1)
    return max(runs, key=len)


def _groups(reference: list[int]) -> list[list[int]]:
    """Indexes of the melody's notes, with neighbouring equal pitches grouped together."""
    groups = [[0]]
    for k in range(1, len(reference)):
        if reference[k] == reference[k - 1]:
            groups[-1].append(k)
        else:
            groups.append([k])
    return groups


def _quiet_regions(quiet: np.ndarray, first: int, last: int) -> list[tuple[int, int]]:
    """Stretches of quiet frames strictly inside first..last, as (start, end) pairs."""
    inside = quiet[first : last + 1].astype(int)
    edges = np.diff(np.concatenate(([0], inside, [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(first + a, first + b) for a, b in zip(starts, ends) if a > 0 and first + b <= last]


def _split_group(frames, quiet, lengths) -> list[np.ndarray]:
    """Divide a group's frames into its repeated notes, or leave a note empty if inseparable."""
    if len(lengths) == 1 or frames.size == 0:
        return [frames] + [frames[:0]] * (len(lengths) - 1)
    first, last = int(frames[0]), int(frames[-1])
    regions = _quiet_regions(quiet, first, last)
    scale = (last - first + 1) / sum(lengths)
    reach = 0.5 * min(lengths) * scale

    pieces, start, position, used = [], first, first, -1
    for length in lengths[:-1]:
        position += length * scale
        choices = [
            (abs((a + b) / 2 - position), i)
            for i, (a, b) in enumerate(regions)
            if i > used and abs((a + b) / 2 - position) <= reach
        ]
        if not choices:
            pieces.append(frames[:0])
            continue
        used = min(choices)[1]
        a, b = regions[used]
        pieces.append(frames[(frames >= start) & (frames < a)])
        start = b
    pieces.append(frames[frames >= start])
    return pieces


def align(midi: np.ndarray, rms: np.ndarray, hop: float, melody: list[tuple[int, float]]) -> dict:
    """Return timed notes in the hummed key, and which melody notes were off or missing."""
    reference = [pitch for pitch, _ in melody]
    lengths = [seconds for _, seconds in melody]
    voiced = ~np.isnan(midi)
    groups = _groups(reference)
    if midi.size < len(groups) or not voiced.any():
        return {"notes": [], "off": [], "missing": list(range(1, len(reference) + 1)), "shift": 0}

    pitches = np.array([reference[group[0]] for group in groups], dtype=float)
    center = int(round(np.median(midi[voiced]) - np.median(reference)))
    candidates = [(_path(midi, pitches + shift), shift) for shift in range(center - 6, center + 6)]
    (_, path), shift = min(candidates, key=lambda c: (c[0][0], abs(c[1] - center)))

    quiet = ~voiced | _dips(rms, midi.size)
    voiced_frames = np.where(voiced)[0]
    frames_per_second = (voiced_frames[-1] - voiced_frames[0] + 1) / sum(lengths)
    max_gap = max(1, int(MAX_GAP_SECONDS / hop))

    notes, deviations, missing = [], {}, []
    for g, group in enumerate(groups):
        frames = np.where((path == g) & voiced)[0]
        frames = _main_run(frames, max_gap) if len(group) == 1 else frames
        pieces = _split_group(frames, quiet, [lengths[k] for k in group])
        for k, piece in zip(group, pieces):
            run = _main_run(piece, max_gap)
            needed = max(MIN_NOTE_SECONDS / hop, MIN_SHARE_OF_EXPECTED * lengths[k] * frames_per_second)
            if run.size < needed:
                missing.append(k + 1)
                continue
            notes.append({
                "start": round(float(run[0] * hop), 4),
                "end": round(float((run[-1] + 1) * hop), 4),
                "pitch": int(reference[k] + shift),
            })
            deviations[k + 1] = float(np.median(_folded(midi[run] - (reference[k] + shift))))

    tuning = float(np.median(list(deviations.values()))) if deviations else 0.0
    off = [number for number, value in deviations.items() if abs(value - tuning) > OFF_LIMIT]
    return {"notes": notes, "off": off, "missing": missing, "shift": shift}
