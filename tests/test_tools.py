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
from humtrack.record import check_take, save_take
from humtrack.review import Store, make_server
from humtrack.score import format_report, score_clip, score_folder
from humtrack.synth import DEMO_MELODIES, make_hum

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
