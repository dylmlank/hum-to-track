"""The recording list and the fixed split between tuning clips and hidden clips.

Every third clip is hidden. Because the list runs category by category, this keeps
the same mix of hum types in both groups (20 tuning, 10 hidden).

Most clips copy a played melody, so their correct notes are known. The rest are your own
melodies, checked only by ear, and are scored separately.
"""
from __future__ import annotations

import re

from .melodies import MELODIES, PROMPTS

CLIP_IDS = [f"clip_{i:03d}" for i in range(1, len(PROMPTS) + 1)]
_ID_RE = re.compile(r"^clip_(\d{3})$")


def is_clip_id(clip_id: str) -> bool:
    return bool(_ID_RE.match(clip_id))


def prompt_for(clip_id: str) -> str:
    match = _ID_RE.match(clip_id)
    number = int(match.group(1)) if match else 0
    return PROMPTS[number - 1] if 1 <= number <= len(PROMPTS) else ""


def melody_for(clip_id: str) -> list[tuple[int, float]] | None:
    """The (pitch, seconds) notes this clip copies, or None for your own melodies."""
    match = _ID_RE.match(clip_id)
    number = int(match.group(1)) if match else 0
    return MELODIES[number - 1] if 1 <= number <= len(MELODIES) else None


def split_of(clip_id: str) -> str:
    match = _ID_RE.match(clip_id)
    if not match:
        raise ValueError(f"not a clip id: {clip_id}")
    return "hidden" if int(match.group(1)) % 3 == 0 else "tune"
