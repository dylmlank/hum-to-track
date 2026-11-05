"""Guided recording: walks through the clip list, checks each take and saves the keepers."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from . import draft
from .clips import CLIP_IDS, melody_for, prompt_for
from .synth import render_melody

SR = 44100
MIN_SECONDS = 3.0
MAX_SECONDS = 45.0


def check_take(audio: np.ndarray, sr: int = SR) -> list[str]:
    """Return plain-language problems with a take. An empty list means it's usable."""
    problems = []
    seconds = audio.size / sr
    if seconds < MIN_SECONDS:
        problems.append(f"Too short ({seconds:.1f}s). Hum the whole melody.")
    if seconds > MAX_SECONDS:
        problems.append(f"Too long ({seconds:.0f}s). One time through is enough.")
    if audio.size:
        peak = float(np.max(np.abs(audio)))
        if peak < 0.03:
            problems.append("Very quiet. Move closer or check the right mic is selected.")
        if np.mean(np.abs(audio) >= 0.99) > 0.001:
            problems.append("Too loud, the sound is distorting. Move back a little.")
    return problems


def check_melody(audio: np.ndarray, melody, sr: int = SR) -> list[str]:
    """Compare a hum to the melody it copied. An empty list means every note matched."""
    resampled = librosa.resample(audio, orig_sr=sr, target_sr=draft.SR)
    check = draft.analyze_audio(resampled, melody)["check"]
    problems = []
    if check["missing"]:
        problems.append(f"Couldn't find note(s) {_numbers(check['missing'])}. Redo this one.")
    if check["off"]:
        problems.append(f"Note(s) {_numbers(check['off'])} landed on a different pitch. Redo this one.")
    return problems


def _numbers(values: list[int]) -> str:
    return ", ".join(str(value) for value in values)


def _listen(melody) -> None:
    import sounddevice as sd

    while True:
        sd.play(render_melody(melody, SR), SR)
        sd.wait()
        if input("  Enter = record your hum, l = listen again: ").strip().lower() != "l":
            return


def save_take(data_dir: Path, clip_id: str, audio: np.ndarray, sr: int = SR) -> Path:
    """Save a take and clear the draft and corrected notes made from any older take."""
    path = data_dir / "audio" / f"{clip_id}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")
    for stale in (data_dir / "drafts" / f"{clip_id}.json", data_dir / "truth" / f"{clip_id}.json"):
        stale.unlink(missing_ok=True)
    return path


def _capture(device) -> np.ndarray:
    import sounddevice as sd

    chunks = []

    def on_audio(indata, frames, time, status):
        chunks.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=device, callback=on_audio):
        input("  Recording. Press Enter to stop.")
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def run(data_dir: Path, device=None, clip: int | None = None) -> None:
    import sounddevice as sd

    print(f"Mic: {sd.query_devices(device, kind='input')['name']}")
    if clip is not None:
        if not 1 <= clip <= len(CLIP_IDS):
            raise SystemExit(f"--clip must be between 1 and {len(CLIP_IDS)}")
        queue = [CLIP_IDS[clip - 1]]
        existing = data_dir / "audio" / f"{queue[0]}.wav"
        if existing.exists():
            answer = input(f"{queue[0]} exists. Keeping a new take erases its corrected notes. Continue? [y/N] ")
            if answer.strip().lower() != "y":
                return
    else:
        queue = [c for c in CLIP_IDS if not (data_dir / "audio" / f"{c}.wav").exists()]
    if not queue:
        print("All clips are recorded.")
        return

    print(f"{len(queue)} to record. Wear headphones so the mic can't hear the melody. Ctrl+C to pause anytime.")
    print("Copy each melody in any key that's comfortable. Your speed doesn't need to match.")
    try:
        for clip_id in queue:
            while True:
                print(f"\n{clip_id}: {prompt_for(clip_id)}")
                melody = melody_for(clip_id)
                if melody:
                    input("  Press Enter to hear the melody.")
                    _listen(melody)
                else:
                    input("  Press Enter to start.")
                audio = _capture(device)
                problems = check_take(audio)
                if melody and audio.size:
                    print("  Checking your notes...")
                    problems += check_melody(audio, melody)
                    if not problems:
                        print(f"  All {len(melody)} notes matched.")
                for problem in problems:
                    print(f"  ! {problem}")
                answer = input("  Enter = keep, r = redo, p = play it back: ").strip().lower()
                while answer == "p":
                    sd.play(audio, SR)
                    sd.wait()
                    answer = input("  Enter = keep, r = redo, p = play it back: ").strip().lower()
                if answer != "r":
                    print(f"  Saved {save_take(data_dir, clip_id, audio)}")
                    break
    except KeyboardInterrupt:
        print("\nPaused. Run the same command to pick up where you left off.")
