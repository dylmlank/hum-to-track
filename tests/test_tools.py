import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from humtrack import clips, draft
from humtrack.notes import clean_notes, load_notes, save_notes
from humtrack.align import align
from humtrack.record import check_melody, check_take, save_take
from humtrack.review import Store, make_server
from humtrack.score import format_report, score_clip, score_folder
from humtrack.synth import DEMO_MELODIES, make_hum, render_melody
from humtrack.key import NATURAL_MINOR, detect_tonic, parse_tonic, snap_pitch, tuning_offset
from humtrack.to_fl import beat_seconds, drop_strays, find_recording, merge_flickers, merge_repeats, prepare, to_beats

TRUTH = [
    {"start": 0.0, "end": 0.5, "pitch": 60},
    {"start": 0.6, "end": 1.0, "pitch": 63},
    {"start": 1.2, "end": 2.0, "pitch": 67},
]


def shifted(semitones=0, seconds=0.0):
    return [
        {"start": n["start"] + seconds, "end": n["end"] + seconds, "pitch": n["pitch"] + semitones}
        for n in TRUTH
    ]


# Scoring
def test_perfect_prediction_scores_one():
    assert score_clip(TRUTH, TRUTH) == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "f1_with_end": 1.0}


def test_wrong_pitch_scores_zero():
    assert score_clip(TRUTH, shifted(semitones=1))["f1"] == 0.0


def test_timing_tolerance_is_50_ms():
    assert score_clip(TRUTH, shifted(seconds=0.04))["f1"] == 1.0
    assert score_clip(TRUTH, shifted(seconds=0.06))["f1"] == 0.0


def test_extra_and_missing_notes_lower_precision_and_recall():
    result = score_clip(TRUTH, TRUTH[:2])
    assert result["precision"] == 1.0
    assert result["recall"] == pytest.approx(2 / 3)
    extra = score_clip(TRUTH, TRUTH + [{"start": 3.0, "end": 3.5, "pitch": 70}])
    assert extra["precision"] == pytest.approx(3 / 4)
    assert extra["recall"] == 1.0


def test_empty_prediction_scores_zero():
    assert score_clip(TRUTH, [])["f1"] == 0.0


def test_missing_prediction_counts_as_zero_and_splits_stay_apart(tmp_path):
    truth_dir, pred_dir = tmp_path / "truth", tmp_path / "pred"
    save_notes(truth_dir / "clip_001.json", TRUTH)
    save_notes(truth_dir / "clip_002.json", TRUTH)
    save_notes(truth_dir / "clip_003.json", TRUTH)
    save_notes(pred_dir / "clip_001.json", TRUTH)
    save_notes(pred_dir / "clip_003.json", TRUTH)

    tune = score_folder(pred_dir, truth_dir, "tune")
    assert tune["clips"] == 2
    assert tune["mean"]["f1"] == pytest.approx(0.5)
    assert tune["missing"] == ["clip_002"]
    assert "clip_003" not in format_report(tune)

    hidden = score_folder(pred_dir, truth_dir, "hidden")
    assert [r["clip"] for r in hidden["rows"]] == ["clip_003"]


def test_split_is_20_tune_and_10_hidden():
    splits = [clips.split_of(c) for c in clips.CLIP_IDS]
    assert len(splits) == 30
    assert splits.count("hidden") == 10
    assert all(clips.prompt_for(c) for c in clips.CLIP_IDS)


# Notes
def test_midi_predictions_load(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("note_on", note=60, velocity=90, time=480))
    track.append(mido.Message("note_off", note=60, time=480))
    path = tmp_path / "clip_001.mid"
    midi.save(path)
    assert load_notes(path) == [{"start": pytest.approx(0.5), "end": pytest.approx(1.0), "pitch": 60}]


# Hum-back clips
def test_clip_list_mixes_copied_and_own_melodies():
    melodies = [clips.melody_for(c) for c in clips.CLIP_IDS]
    assert sum(m is not None for m in melodies) == 24
    hidden = [m for c, m in zip(clips.CLIP_IDS, melodies) if clips.split_of(c) == "hidden"]
    assert sum(m is None for m in hidden) == 2
    for melody in filter(None, melodies):
        assert 3.0 <= sum(seconds for _, seconds in melody) <= 45.0
        assert render_melody(melody, 8000).size


def fake_take(melody, shift=0, seed=1):
    audio, truth = make_hum([(p + shift, s) for p, s in melody], sr=draft.SR, seed=seed)
    return draft.analyze_audio(audio, melody), truth


@pytest.mark.parametrize("number", [1, 6, 10, 12, 16, 20])
@pytest.mark.parametrize("shift", [0, 5])
def test_hum_back_finds_timing_in_any_key(number, shift):
    result, truth = fake_take(clips.melody_for(f"clip_{number:03d}"), shift)
    assert result["check"]["missing"] == [] and result["check"]["off"] == []
    assert score_clip(truth, result["notes"])["f1"] >= 0.9


@pytest.mark.parametrize("semitones", [1, 2, -3])
def test_hum_back_flags_a_wrong_note(semitones):
    melody = clips.melody_for("clip_001")
    wrong = list(melody)
    wrong[3] = (wrong[3][0] + semitones, wrong[3][1])
    audio, _ = make_hum(wrong, sr=draft.SR, seed=1)
    check = draft.analyze_audio(audio, melody)["check"]
    assert 4 in check["off"] + check["missing"]


def test_hum_back_reports_repeated_notes_run_together():
    melody = [(60, 0.5), (60, 0.5), (60, 0.5), (64, 0.5)]
    audio, _ = make_hum([(60, 1.5), (64, 0.5)], sr=draft.SR, seed=1)
    assert draft.analyze_audio(audio, melody)["check"]["missing"]


def test_hum_back_ignores_a_stray_blip_after_the_end():
    melody = clips.melody_for("clip_001")
    audio, truth = make_hum(melody + [(71, 0.1)], sr=draft.SR, seed=1, gap=0.25)
    result = draft.analyze_audio(audio, melody)
    assert result["notes"][-1]["end"] == pytest.approx(truth[-2]["end"], abs=0.1)


def test_silent_take_is_reported_not_crashed():
    result = align(np.full(500, np.nan), np.zeros(500), 0.01, [(60, 1.0), (62, 1.0)])
    assert result["notes"] == [] and result["missing"] == [1, 2]
    silence = np.zeros(44100, dtype=np.float32)
    assert any("Couldn't find" in p for p in check_melody(silence, clips.melody_for("clip_001")))


def test_own_melodies_score_separately(tmp_path):
    truth_dir, pred_dir = tmp_path / "truth", tmp_path / "pred"
    save_notes(truth_dir / "clip_001.json", TRUTH)
    save_notes(truth_dir / "clip_025.json", TRUTH)
    save_notes(pred_dir / "clip_001.json", TRUTH)
    result = score_folder(pred_dir, truth_dir, "tune")
    assert result["mean"]["f1"] == 1.0
    assert result["mean_own"]["f1"] == 0.0
    assert "OWN*" in format_report(result)


def test_clean_notes_sorts_and_trims_overlaps():
    cleaned = clean_notes([
        {"start": 1.0, "end": 2.0, "pitch": 62},
        {"start": 0.0, "end": 1.5, "pitch": 60},
    ])
    assert cleaned == [
        {"start": 0.0, "end": 1.0, "pitch": 60},
        {"start": 1.0, "end": 2.0, "pitch": 62},
    ]


@pytest.mark.parametrize("bad", [
    [{"start": 1.0, "end": 0.5, "pitch": 60}],
    [{"start": 0.0, "end": 0.5, "pitch": 128}],
    [{"start": 0.0, "end": 0.5, "pitch": "C4"}],
    [{"start": 0.0, "end": 0.5}],
    "notes",
])
def test_clean_notes_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        clean_notes(bad)


# Recorder
def test_take_checks():
    sr = 1000
    good = 0.3 * np.sin(np.arange(10 * sr) / 5)
    assert check_take(good, sr) == []
    assert any("Too short" in p for p in check_take(good[:sr], sr))
    assert any("Very quiet" in p for p in check_take(good * 0.01, sr))
    assert any("Too loud" in p for p in check_take(np.clip(good * 20, -1, 1), sr))


def test_new_take_clears_stale_notes(tmp_path):
    save_notes(tmp_path / "truth" / "clip_001.json", TRUTH)
    save_notes(tmp_path / "drafts" / "clip_001.json", TRUTH)
    path = save_take(tmp_path, "clip_001", np.zeros(4410, dtype=np.float32))
    assert path.exists()
    assert not (tmp_path / "truth" / "clip_001.json").exists()
    assert not (tmp_path / "drafts" / "clip_001.json").exists()


# First guess
@pytest.mark.parametrize("seed", range(len(DEMO_MELODIES)))
def test_first_guess_finds_known_notes_in_fake_hums(tmp_path, seed):
    audio, truth = make_hum(DEMO_MELODIES[seed], seed=seed + 1)
    path = tmp_path / "hum.wav"
    sf.write(path, audio, 44100)
    result = draft.analyze(path)
    assert score_clip(truth, result["notes"])["f1"] >= 0.8


# Sending to FL
def timed(*items):
    return [{"start": s, "end": e, "pitch": p} for s, e, p in items]


def test_flickers_fold_into_the_note_they_wobble_from():
    notes = timed((0.0, 0.08, 49), (0.08, 0.5, 48), (0.6, 1.0, 50))
    assert merge_flickers(notes) == timed((0.0, 0.5, 48), (0.6, 1.0, 50))


def test_short_notes_that_are_really_different_stay():
    notes = timed((0.0, 0.08, 60), (0.08, 0.5, 67), (0.8, 0.9, 67), (1.5, 2.0, 67))
    assert merge_flickers(notes) == notes


@pytest.mark.parametrize("spacing, bpm", [(0.5, 120), (0.25, 120), (1.0, 120), (0.75, 80)])
def test_beat_stays_in_a_usable_tempo_range(spacing, bpm):
    notes = timed(*[(i * spacing, (i + 1) * spacing, 60) for i in range(4)])
    assert 60 / beat_seconds(notes) == pytest.approx(bpm)


def test_to_beats_starts_at_zero_and_snaps_to_the_grid():
    notes = timed((1.0, 1.48, 60), (1.51, 2.02, 62))
    assert to_beats(notes, 0.5, velocity=0.5) == [
        {"pitch": 60, "start": 0.0, "length": 1.0, "velocity": 0.5},
        {"pitch": 62, "start": 1.0, "length": 1.0, "velocity": 0.5},
    ]


def test_to_beats_never_overlaps_or_stacks_notes():
    notes = timed((0.0, 0.5, 60), (0.02, 0.1, 61), (0.4, 1.0, 62))
    beats = to_beats(notes, 0.5)
    assert [n["pitch"] for n in beats] == [60, 62]
    assert beats[0]["start"] + beats[0]["length"] <= beats[1]["start"]


def test_find_recording_blocks_hidden_clips(tmp_path):
    with pytest.raises(ValueError, match="hidden"):
        find_recording("3", tmp_path)
    with pytest.raises(ValueError, match="between"):
        find_recording("31", tmp_path)
    with pytest.raises(ValueError, match="No recording"):
        find_recording("1", tmp_path)
    path = tmp_path / "idea.wav"
    path.touch()
    assert find_recording(str(path), tmp_path) == (path, "idea")


def test_prepare_turns_a_fake_hum_into_beats(tmp_path):
    audio, truth = make_hum(DEMO_MELODIES[0], seed=1)
    path = tmp_path / "hum.wav"
    sf.write(path, audio, 44100)
    notes, bpm, _ = prepare(path, snap=False)
    assert [n["pitch"] for n in notes] == [n["pitch"] for n in truth]
    assert notes[0]["start"] == 0.0
    assert 70 <= bpm < 140


# Finding the key
MINOR_TUNE = [0, 3, 5, 7, 8, 7, 5, 3, 0, 10, 0, 2, 3, 0]


def tune_in(tonic, offset=0.0):
    pitches = np.array([48 + tonic + step + offset for step in MINOR_TUNE], dtype=float)
    return pitches, np.full(len(MINOR_TUNE), 0.5)


@pytest.mark.parametrize("tonic", range(12))
def test_detects_the_minor_key_whatever_key_you_hum_in(tonic):
    pitches, durations = tune_in(tonic)
    assert detect_tonic(pitches, durations)[0] == tonic


@pytest.mark.parametrize("offset", [-0.4, -0.2, 0.2, 0.4])
def test_humming_sharp_or_flat_throughout_keeps_the_right_notes(offset):
    pitches, durations = tune_in(5, offset)
    corrected = pitches - tuning_offset(pitches, durations)
    assert detect_tonic(corrected, durations)[0] == 5
    assert [snap_pitch(p, 5) for p in corrected] == [48 + 5 + s for s in MINOR_TUNE]


def test_snap_moves_only_notes_outside_the_scale():
    assert snap_pitch(60.0, 0) == 60
    assert snap_pitch(63.2, 0) == 63
    assert snap_pitch(63.7, 0) == 63
    assert snap_pitch(64.3, 0) == 65
    assert snap_pitch(61.4, 0) == 62
    assert snap_pitch(66.3, 0) == 67
    assert snap_pitch(69.2, 0) == 70
    assert all((snap_pitch(p / 2, 9) - 9) % 12 in NATURAL_MINOR for p in range(80, 160))


@pytest.mark.parametrize("text, tonic", [("C", 0), ("cm", 0), ("F#", 6), ("Bb", 10), ("A minor", 9), ("Am", 9)])
def test_parse_tonic(text, tonic):
    assert parse_tonic(text) == tonic


def test_parse_tonic_rejects_nonsense():
    with pytest.raises(ValueError, match="Unknown key"):
        parse_tonic("H")


def test_short_blips_far_from_their_neighbours_are_dropped():
    notes = timed((0.0, 0.5, 60), (0.6, 1.0, 62), (1.2, 1.4, 73))
    assert drop_strays(notes) == notes[:2]


def test_real_jumps_and_long_notes_are_kept():
    jump = timed((0.0, 0.5, 60), (0.6, 0.8, 72), (0.9, 1.3, 71))
    assert drop_strays(jump) == jump
    held = timed((0.0, 0.5, 60), (0.6, 1.2, 72))
    assert drop_strays(held) == held


def test_merge_repeats_joins_only_touching_notes():
    notes = timed((0.0, 0.5, 60), (0.5, 0.8, 60), (1.5, 2.0, 60), (2.0, 2.5, 62))
    assert merge_repeats(notes) == timed((0.0, 0.8, 60), (1.5, 2.0, 60), (2.0, 2.5, 62))


def test_fake_hum_lands_in_the_detected_key(tmp_path):
    melody = [(48 + 5 + step, 0.45) for step in MINOR_TUNE]
    melody[3] = (melody[3][0] + 1, 0.45)
    audio, _ = make_hum(melody, seed=1)
    path = tmp_path / "hum.wav"
    sf.write(path, audio, 44100)
    notes, _, key = prepare(path)
    assert key["name"] == "F minor"
    assert all((n["pitch"] - 5) % 12 in NATURAL_MINOR for n in notes)


def test_prepare_can_use_a_chosen_key_or_no_scale(tmp_path):
    audio, truth = make_hum(DEMO_MELODIES[0], seed=1)
    path = tmp_path / "hum.wav"
    sf.write(path, audio, 44100)
    notes, _, key = prepare(path, tonic=2)
    assert key["name"] == "D minor"
    assert all((n["pitch"] - 2) % 12 in NATURAL_MINOR for n in notes)
    raw, _, none = prepare(path, snap=False)
    assert none is None and [n["pitch"] for n in raw] == [n["pitch"] for n in truth]


# Correction page server
@pytest.fixture
def server(tmp_path):
    audio, _ = make_hum(DEMO_MELODIES[0], seed=1)
    path = tmp_path / "audio" / "clip_001.wav"
    path.parent.mkdir()
    sf.write(path, audio, 44100)
    httpd = make_server(Store(tmp_path), 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()
    httpd.server_close()


def post(url, payload, content_type="application/json"):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": content_type}
    )
    return urllib.request.urlopen(request)


def test_server_saves_corrected_notes(server):
    base, data = server
    detail = json.load(urllib.request.urlopen(base + "/api/clip/clip_001"))
    assert detail["done"] is False and detail["notes"]
    post(base + "/api/clip/clip_001", {"notes": TRUTH})
    assert load_notes(data / "truth" / "clip_001.json") == TRUTH
    detail = json.load(urllib.request.urlopen(base + "/api/clip/clip_001"))
    assert detail["done"] is True and detail["notes"] == TRUTH


@pytest.mark.parametrize("path", ["/api/clip/..%2f..%2fsecret", "/audio/../../secret", "/api/clip/clip_999"])
def test_server_rejects_unknown_clips(server, path):
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(server[0] + path)
    assert caught.value.code == 404


def test_server_rejects_bad_saves(server):
    base, data = server
    for payload, content_type, status in (
        ({"notes": []}, "application/json", 400),
        ({"notes": [{"start": 2, "end": 1, "pitch": 60}]}, "application/json", 400),
        ({"notes": TRUTH}, "text/plain", 415),
    ):
        with pytest.raises(urllib.error.HTTPError) as caught:
            post(base + "/api/clip/clip_001", payload, content_type)
        assert caught.value.code == status
    assert not (data / "truth" / "clip_001.json").exists()
