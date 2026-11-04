"""The recording list and the fixed split between tuning clips and hidden clips.

Every third clip is hidden. Because the list runs category by category, this keeps
the same mix of hum types in both groups (20 tuning, 10 hidden).
"""
from __future__ import annotations

import re

_OWN = "Your own idea: hum a melody you could build a beat around."
_PROMPTS = [
    *[_OWN + hint for hint in (
        " Keep it slow and simple.",
        " Use a short hook that repeats.",
        " Make it dark and low.",
        " Make it bright and high.",
        " Sing it with 'da da da'.",
        " Sing it with 'mmm'.",
        " Leave long gaps between phrases.",
        " Repeat the same note a few times in it.",
        " Use big jumps between notes.",
        " Keep the notes close together.",
        " Hum it quietly.",
        " Hum it with energy.",
    )],
    "Known tune: Twinkle Twinkle Little Star.",
    "Known tune: Ode to Joy.",
    "Known tune: Happy Birthday.",
    "Known tune: Mary Had a Little Lamb.",
    "Known tune: Amazing Grace.",
    "Known tune: Greensleeves.",
    *[f"Fast: {hint}" for hint in (
        "a quick trap-style hook.",
        "a fast run of short notes going up and down.",
        "a bouncy rhythm on just two or three notes.",
        "a rapid repeated note, then a change.",
        "the fastest melody you can hum cleanly.",
        "a syncopated, off-beat pattern.",
    )],
    *[f"Expressive: {hint}" for hint in (
        "long held notes.",
        "slide slowly between notes.",
        "add vibrato on the long notes.",
        "hum as low as is comfortable.",
        "hum as high as is comfortable.",
        "fall off the end of each phrase.",
    )],
]

CLIP_IDS = [f"clip_{i:03d}" for i in range(1, len(_PROMPTS) + 1)]
_ID_RE = re.compile(r"^clip_(\d{3})$")


def is_clip_id(clip_id: str) -> bool:
    return bool(_ID_RE.match(clip_id))


def prompt_for(clip_id: str) -> str:
    match = _ID_RE.match(clip_id)
    number = int(match.group(1)) if match else 0
    return _PROMPTS[number - 1] if 1 <= number <= len(_PROMPTS) else ""


def split_of(clip_id: str) -> str:
    match = _ID_RE.match(clip_id)
    if not match:
        raise ValueError(f"not a clip id: {clip_id}")
    return "hidden" if int(match.group(1)) % 3 == 0 else "tune"
