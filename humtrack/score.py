"""Grade predicted notes against the corrected notes with mir_eval.

A predicted note counts when it has the right pitch and starts within 50 ms of the true
note (the usual transcription standard). A clip with no prediction scores zero, so a
method can't look better by skipping the hard clips.
"""
from __future__ import annotations

from pathlib import Path

import mir_eval
import numpy as np

from .clips import is_clip_id, melody_for, split_of
from .notes import load_notes

ONSET_TOLERANCE = 0.05


def _arrays(notes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.array([[n["start"], n["end"]] for n in notes], dtype=float).reshape(-1, 2)
    hz = np.array([440.0 * 2 ** ((n["pitch"] - 69) / 12) for n in notes], dtype=float)
    return intervals, hz


def score_clip(truth: list[dict], pred: list[dict]) -> dict:
    """Return note precision, recall and F1, plus F1 that also requires the right note end."""
    if not truth:
        raise ValueError("truth has no notes")
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f1_with_end": 0.0}
    ref_i, ref_p = _arrays(truth)
    est_i, est_p = _arrays(pred)
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p,
        onset_tolerance=ONSET_TOLERANCE, pitch_tolerance=50.0, offset_ratio=None,
    )
    _, _, f1_end, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p,
        onset_tolerance=ONSET_TOLERANCE, pitch_tolerance=50.0, offset_ratio=0.2,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f1_with_end": float(f1_end),
    }


def _find_prediction(pred_dir: Path, clip_id: str) -> Path | None:
    for suffix in (".json", ".mid", ".midi"):
        candidate = pred_dir / f"{clip_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _mean(rows: list[dict]) -> dict:
    keys = ("precision", "recall", "f1", "f1_with_end")
    return {k: float(np.mean([r[k] for r in rows])) if rows else 0.0 for k in keys}


def score_folder(pred_dir: Path, truth_dir: Path, split: str = "tune") -> dict:
    """Score every corrected clip in one split against a folder of predictions."""
    rows = []
    missing = []
    for truth_path in sorted(truth_dir.glob("*.json")):
        clip_id = truth_path.stem
        if not is_clip_id(clip_id) or split_of(clip_id) != split:
            continue
        pred_path = _find_prediction(pred_dir, clip_id)
        if pred_path is None:
            missing.append(clip_id)
        pred = load_notes(pred_path) if pred_path else []
        rows.append({"clip": clip_id, **score_clip(load_notes(truth_path), pred)})
    copied = [r for r in rows if melody_for(r["clip"])]
    own = [r for r in rows if not melody_for(r["clip"])]
    return {
        "split": split,
        "clips": len(rows),
        "missing": missing,
        "mean": _mean(copied),
        "mean_own": _mean(own) if own else None,
        "rows": rows,
    }


def format_report(result: dict) -> str:
    lines = [f"{'clip':<10} {'precision':>9} {'recall':>7} {'F1':>6} {'F1+end':>7}"]
    for row in result["rows"]:
        lines.append(
            f"{row['clip']:<10} {row['precision']:>9.3f} {row['recall']:>7.3f} "
            f"{row['f1']:>6.3f} {row['f1_with_end']:>7.3f}"
        )
    means = [("MEAN", result["mean"])]
    if result["mean_own"]:
        means.append(("OWN*", result["mean_own"]))
    for label, mean in means:
        lines.append(
            f"{label:<10} {mean['precision']:>9.3f} {mean['recall']:>7.3f} "
            f"{mean['f1']:>6.3f} {mean['f1_with_end']:>7.3f}"
        )
    if result["mean_own"]:
        lines.append("* Your own melodies, checked by ear only. MEAN covers hum-back clips.")
    lines.append(f"{result['clips']} {result['split']} clips scored")
    if result["missing"]:
        lines.append(f"Scored as zero, no prediction: {', '.join(result['missing'])}")
    return "\n".join(lines)
