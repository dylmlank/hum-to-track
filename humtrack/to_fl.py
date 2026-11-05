"""Send a recorded hum to FL Studio as rough piano roll notes.

The notes are the first guess from draft.py, not the finished note engine. FL is reached
through StudioKit, which has its own Python, so that part runs as a separate program.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from statistics import median

from . import draft
from .clips import CLIP_IDS, split_of
from .key import NAMES, snap_notes

STUDIOKIT_DIR = Path(os.environ.get("STUDIOKIT_DIR", "E:/studiokit"))
SENDER = Path(__file__).with_name("studiokit_send.py")
SONG = "hum-to-track"
GRID_BEATS = 0.25
SLOWEST_BPM = 70.0
FLICKER_SECONDS = 0.12
FLICKER_GAP_SECONDS = 0.03
REPEAT_GAP_SECONDS = 0.08
STRAY_SECONDS = 0.25
STRAY_JUMP = 7


def find_recording(clip: str, data_dir: Path) -> tuple[Path, str]:
    """Turn a clip number or .wav path into a file and a part name."""
    if clip.isdigit():
        number = int(clip)
        if not 1 <= number <= len(CLIP_IDS):
            raise ValueError(f"Clip must be between 1 and {len(CLIP_IDS)}.")
        clip_id = CLIP_IDS[number - 1]
        if split_of(clip_id) == "hidden":
            raise ValueError(f"Clip {number} is a hidden test clip. It stays unseen until the engine is finished.")
        path, name = data_dir / "audio" / f"{clip_id}.wav", clip_id
    else:
        path, name = Path(clip), Path(clip).stem
    if not path.exists():
        raise ValueError(f"No recording at {path}.")
    return path, name


def merge_repeats(notes: list[dict]) -> list[dict]:
    """Join pieces of one held note that ended up on the same pitch with almost no gap."""
    merged: list[dict] = []
    for note in notes:
        if merged and merged[-1]["pitch"] == note["pitch"] and note["start"] - merged[-1]["end"] <= REPEAT_GAP_SECONDS:
            merged[-1]["end"] = max(merged[-1]["end"], note["end"])
        else:
            merged.append(dict(note))
    return merged


def drop_strays(notes: list[dict]) -> list[dict]:
    """Remove short blips far from the notes around them, usually a voice trailing off."""
    kept = []
    for i, note in enumerate(notes):
        around = notes[max(i - 1, 0) : i] + notes[i + 1 : i + 2]
        far = around and all(abs(other["pitch"] - note["pitch"]) >= STRAY_JUMP for other in around)
        if not (far and note["end"] - note["start"] < STRAY_SECONDS):
            kept.append(note)
    return kept


def merge_flickers(notes: list[dict]) -> list[dict]:
    """Fold very short pieces into the neighbouring note they are a wobble of."""
    merged = [dict(note) for note in notes]
    changed = True
    while changed:
        changed = False
        for i, note in enumerate(merged):
            if note["end"] - note["start"] >= FLICKER_SECONDS:
                continue
            neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(merged) and _touching(merged[min(i, j)], merged[max(i, j)])]
            neighbours = [j for j in neighbours if abs(merged[j]["pitch"] - note["pitch"]) <= 1]
            if not neighbours:
                continue
            target = merged[max(neighbours, key=lambda j: merged[j]["end"] - merged[j]["start"])]
            target["start"] = min(target["start"], note["start"])
            target["end"] = max(target["end"], note["end"])
            del merged[i]
            changed = True
            break
    return merged


def _touching(first: dict, second: dict) -> bool:
    return second["start"] - first["end"] <= FLICKER_GAP_SECONDS


def beat_seconds(notes: list[dict]) -> float:
    """Estimate the hum's beat from how far apart its notes start, at 70 to 140 BPM."""
    gaps = [b["start"] - a["start"] for a, b in zip(notes, notes[1:])]
    beat = median(gaps) if gaps else 60.0 / 120.0
    while 60.0 / beat >= 2 * SLOWEST_BPM:
        beat *= 2
    while 60.0 / beat < SLOWEST_BPM:
        beat /= 2
    return beat


def to_beats(notes: list[dict], beat: float, velocity: float = 0.8) -> list[dict]:
    """Snap timed notes to a quarter-beat grid, starting at beat 0, with no overlaps."""
    if not notes:
        return []
    origin = notes[0]["start"]
    snapped: list[dict] = []
    for note in notes:
        start = round((note["start"] - origin) / beat / GRID_BEATS) * GRID_BEATS
        end = round((note["end"] - origin) / beat / GRID_BEATS) * GRID_BEATS
        end = max(end, start + GRID_BEATS)
        if snapped and snapped[-1]["start"] == start:
            if end - start <= snapped[-1]["length"]:
                continue
            snapped.pop()
        if snapped:
            previous = snapped[-1]
            previous["length"] = min(previous["length"], start - previous["start"])
        snapped.append({"pitch": note["pitch"], "start": start, "length": end - start, "velocity": velocity})
    return snapped


def prepare(path: Path, tonic: int | None = None, snap: bool = True) -> tuple[list[dict], float, dict | None]:
    """Return a recording's notes in beats, the tempo it was hummed at, and its key."""
    analysis = draft.analyze(path)
    notes = drop_strays(merge_flickers(analysis["notes"]))
    if not notes:
        raise ValueError(f"No notes found in {path}. Check the recording isn't silent.")
    key = None
    if snap:
        notes, key = snap_notes(notes, analysis["curve"], tonic)
        notes = merge_repeats(notes)
    beat = beat_seconds(notes)
    return to_beats(notes, beat), round(60.0 / beat, 1), key


def send(part: str, notes: list[dict], bpm: float, channel: str | None, match_tempo: bool, key: dict | None = None) -> dict:
    """Have StudioKit put the notes in a new FL pattern. Raises RuntimeError if it can't."""
    python = STUDIOKIT_DIR / "venv" / "Scripts" / "python.exe"
    if not python.exists():
        raise RuntimeError(f"Can't find StudioKit at {STUDIOKIT_DIR}. Set STUDIOKIT_DIR to its folder.")
    payload = {
        "studiokit": str(STUDIOKIT_DIR),
        "song": SONG,
        "part": part,
        "notes": notes,
        "bpm": bpm,
        "channel": channel,
        "match_tempo": match_tempo,
        "key": NAMES[key["tonic"]] if key else None,
    }
    done = subprocess.run(
        [str(python), str(SENDER)], input=json.dumps(payload), text=True, capture_output=True, timeout=60
    )
    if done.returncode != 0:
        lines = done.stdout.strip() or done.stderr.strip() or "StudioKit stopped without saying why."
        raise RuntimeError(lines.splitlines()[-1])
    return json.loads(done.stdout)
