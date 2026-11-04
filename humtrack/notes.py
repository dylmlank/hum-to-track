"""Read and write note lists. A note is {"start": s, "end": s, "pitch": 0-127}."""
from __future__ import annotations

import json
from pathlib import Path

import mido


def clean_notes(raw, duration: float | None = None) -> list[dict]:
    """Validate notes, sort them and trim overlaps so each hum stays one note at a time."""
    if not isinstance(raw, list):
        raise ValueError("notes must be a list")
    notes = []
    for item in raw:
        try:
            start, end = float(item["start"]), float(item["end"])
            pitch = item["pitch"]
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"bad note: {item!r}") from e
        if isinstance(pitch, bool) or not isinstance(pitch, int) or not 0 <= pitch <= 127:
            raise ValueError(f"bad pitch: {pitch!r}")
        if not (0 <= start < end):
            raise ValueError(f"bad note times: {start}..{end}")
        if duration is not None:
            if start >= duration:
                raise ValueError(f"note starts after the clip ends: {start}")
            end = min(end, duration)
        notes.append({"start": round(start, 4), "end": round(end, 4), "pitch": pitch})

    notes.sort(key=lambda n: n["start"])
    kept = []
    for note in notes:
        if kept and kept[-1]["end"] > note["start"]:
            kept[-1]["end"] = note["start"]
            if kept[-1]["end"] <= kept[-1]["start"]:
                kept.pop()
        kept.append(note)
    return kept


def save_notes(path: Path, notes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"notes": notes}, indent=1), encoding="utf-8")


def load_notes(path: Path) -> list[dict]:
    if path.suffix.lower() in (".mid", ".midi"):
        return _load_midi(path)
    return json.loads(path.read_text(encoding="utf-8"))["notes"]


def _load_midi(path: Path) -> list[dict]:
    now = 0.0
    held: dict[int, float] = {}
    notes = []
    for msg in mido.MidiFile(path):
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            held.setdefault(msg.note, now)
        elif msg.type in ("note_on", "note_off") and msg.note in held:
            start = held.pop(msg.note)
            if now > start:
                notes.append({"start": start, "end": now, "pitch": msg.note})
    return sorted(notes, key=lambda n: n["start"])
