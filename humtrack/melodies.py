"""The melodies played for hum-back clips. Their notes are the answer key.

Written as "C3:2" for note and length in beats, with one beat as the default.
"""
from __future__ import annotations

_NAMES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def _parse(bpm: int, text: str) -> list[tuple[int, float]]:
    melody = []
    for token in text.split():
        name, _, beats = token.partition(":")
        pitch = _NAMES[name[:-1]] + 12 * (int(name[-1]) + 1)
        melody.append((pitch, float(beats or 1) * 60 / bpm))
    return melody


# Prompt, tempo and notes. Clips run category by category, in groups of three, so that
# every third clip being hidden keeps the same mix in the tuning and hidden sets.
_LIST = [
    ("a slow scale going up (do re mi fa so la ti do).", 60, "C3 D3 E3 F3 G3 A3 B3 C4"),
    ("a slow scale going down.", 70, "C4 B3 A3 G3 F3 E3 D3 C3"),
    ("a short phrase down and back up, on 'da'.", 72, "A3 G3 F3 E3 D3 E3 F3 G3 A3:2"),
    ("a simple repeating hook, on 'mmm'.", 80, "C3 D3 E3 D3 C3 D3 E3 D3 C3:2"),
    ("a dark, low phrase.", 72, "C3 D#3 F3 G3 F3 D#3 C3 G2:2"),
    ("a phrase with repeated notes. Separate each one.", 80, "E3 E3 E3 F3 G3 G3 G3 F3 E3:2"),
    ("big jumps between notes.", 66, "C3 G3 C3 G3 E3 C4 G3 C3:2"),
    ("big jumps, on 'da'.", 66, "E3 A3 C4 A3 E3 A3 G3 E3:2"),
    ("wide jumps going up, then a walk down.", 66, "D3 A3 F3 D4 C4 G3 E3 C3:2"),
    ("Twinkle Twinkle Little Star.", 90, "C3 C3 G3 G3 A3 A3 G3:2 F3 F3 E3 E3 D3 D3 C3:2"),
    ("Ode to Joy.", 100, "E3 E3 F3 G3 G3 F3 E3 D3 C3 C3 D3 E3 E3:1.5 D3:.5 D3:2"),
    ("Happy Birthday.", 90, "C3:.75 C3:.25 D3 C3 F3 E3:2 C3:.75 C3:.25 D3 C3 G3 F3:2"),
    ("Mary Had a Little Lamb.", 100, "E3 D3 C3 D3 E3 E3 E3:2 D3 D3 D3:2 E3 G3 G3:2"),
    ("Amazing Grace.", 80, "C3 F3:2 A3:.5 F3:.5 A3:2 G3 F3:2 D3 C3:2 C3 F3:2 A3:.5 F3:.5 A3:2 G3 C4:3"),
    ("Greensleeves.", 120, "E3 G3:2 A3 B3:1.5 C4:.5 B3 A3:2 F#3 D3:1.5 E3:.5 F#3 G3:2 E3 E3:1.5 D#3:.5 E3 F#3:2 D#3 B2:2"),
    ("a quick hook, twice through.", 120, "C3:.5 D#3:.5 F3:.5 G3:.5 F3:.5 D#3:.5 C3:.5 D#3:.5 C3:.5 D#3:.5 F3:.5 G3:.5 F3:.5 D#3:.5 C3"),
    ("a trap-style hook, twice through.", 130, "C3:.5 C3:.5 D#3:.5 C3:.5 G3 F3:.5 D#3:.5 C3:.5 C3:.5 D#3:.5 C3:.5 G3 F3:.5 C3"),
    ("a fast run up and down.", 120, "C3:.5 D3:.5 E3:.5 F3:.5 G3:.5 A3:.5 B3:.5 C4:.5 B3:.5 A3:.5 G3:.5 F3:.5 E3:.5 D3:.5 C3"),
    ("a bouncy rhythm on three notes.", 120, "E3:.5 G3:.5 E3:.5 A3:.5 E3:.5 G3:.5 E3 E3:.5 G3:.5 E3:.5 A3:.5 G3:.5 E3 E3"),
    ("a rapid repeated note, then a change.", 120, "G3:.5 G3:.5 G3:.5 G3:.5 A#3 G3:.5 G3:.5 G3:.5 G3:.5 F3 D#3 C3"),
    ("an off-beat pattern.", 110, "C3:.5 D#3 C3:.5 F3 D#3:.5 C3:1.5 C3:.5 D#3 C3:.5 G3 F3:.5 D#3:1.5"),
    ("long held notes, with vibrato if you can.", 50, "C3:2 G3:2 F3:2 E3:2 C3:2"),
    ("hold each note, sliding gently into the next.", 50, "E3:2 G3:2 A3:2 G3:2 E3:2"),
    ("let the end of each phrase fall away.", 60, "C4 A3 G3:2 A3 G3 E3:2 G3 E3 D3 C3:2"),
]

_OWN = "Your own melody: "
_OWN_PROMPTS = [
    "keep it slow and simple.",
    "use a short hook that repeats.",
    "make it dark and low.",
    "make it bright and high.",
    "make it quick and bouncy.",
    "use long, expressive notes.",
]

PROMPTS = [f"Hum it back: {text}" for text, _, _ in _LIST] + [_OWN + text for text in _OWN_PROMPTS]
MELODIES = [_parse(bpm, notes) for _, bpm, notes in _LIST] + [None] * len(_OWN_PROMPTS)
