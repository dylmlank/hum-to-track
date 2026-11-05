# Hum to Track

Hum a melody and get a multi-part beat in FL Studio, with a measured accuracy score to
prove the notes are right.

**Status:** milestone 1, the test set and scoring tools. The note engine isn't built yet.

## Setup

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Building the test set

**1. Record 30 hums.** No music training needed. For 24 clips, the recorder plays a melody
and you hum it back in any key, so the correct notes are already known. It checks your pitch
and tells you if a take needs redoing. The last 6 are your own melodies. Wear headphones.

```bash
.venv/Scripts/python.exe ht.py record
```

**2. Check the notes.** On hum-back clips, the page opens with the melody's notes lined up to
your hum, so you only check the timing. On your own melodies it opens with a first guess:
fix the notes to what you meant to hum.

```bash
.venv/Scripts/python.exe ht.py review
```

Then open http://127.0.0.1:9791. To try the page without recordings, add `--demo`.

**3. Score a method.** Put its output in a folder as `clip_001.json` or `clip_001.mid`.

```bash
.venv/Scripts/python.exe ht.py score path/to/predictions
```

## How the score stays honest

- **Standard grading:** `mir_eval` counts a note when the pitch is right and it starts
  within 50 ms of the true note. F1 combines missed notes and false notes into one number.
- **Hidden clips:** every third clip is kept out of tuning, and is only scored with
  `--final`. Using those results to tune the engine would make the final number meaningless.
- **No skipping:** a clip with no prediction scores zero.
- **Known answers:** hum-back clips take their notes from the melody, not from anyone's ear.
- **Separate score:** your own melodies are checked by ear only, so they get their own line.
  They could be nudged toward what a pitch tracker hears, and copying a melody can also sound
  a little different from humming your own idea. Report both numbers.
- **Timing:** note starts on every clip are lined up automatically, then checked by ear.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests -q
```

The tests use fake hums with known notes. They show the tools work, not how accurate
anything is on real humming.

Recordings in `data/audio/` aren't committed to git.
