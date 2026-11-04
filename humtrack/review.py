"""Local server for the note correction page. Only reachable from this computer."""
from __future__ import annotations

import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import soundfile as sf

from . import draft
from .clips import CLIP_IDS, is_clip_id, prompt_for, split_of
from .notes import clean_notes, load_notes, save_notes
from .synth import DEMO_MELODIES, make_hum

PAGE = Path(__file__).with_name("review.html")
MAX_BODY_BYTES = 2_000_000


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.prompts: dict[str, str] = {}

    def audio_path(self, clip_id: str) -> Path:
        return self.data_dir / "audio" / f"{clip_id}.wav"

    def truth_path(self, clip_id: str) -> Path:
        return self.data_dir / "truth" / f"{clip_id}.json"

    def prompt(self, clip_id: str) -> str:
        return self.prompts.get(clip_id) or prompt_for(clip_id)

    def list_clips(self) -> dict:
        clips = [
            {
                "id": clip_id,
                "prompt": self.prompt(clip_id),
                "split": split_of(clip_id),
                "done": self.truth_path(clip_id).exists(),
            }
            for clip_id in CLIP_IDS
            if self.audio_path(clip_id).exists()
        ]
        return {"clips": clips, "planned": len(self.prompts) or len(CLIP_IDS)}

    def analysis(self, clip_id: str) -> dict:
        cache = self.data_dir / "drafts" / f"{clip_id}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        result = draft.analyze(self.audio_path(clip_id))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result), encoding="utf-8")
        return result

    def clip_detail(self, clip_id: str) -> dict:
        result = self.analysis(clip_id)
        truth = self.truth_path(clip_id)
        return {
            "id": clip_id,
            "prompt": self.prompt(clip_id),
            "split": split_of(clip_id),
            "duration": result["duration"],
            "curve": result["curve"],
            "notes": load_notes(truth) if truth.exists() else result["notes"],
            "done": truth.exists(),
        }

    def save_truth(self, clip_id: str, raw_notes) -> int:
        notes = clean_notes(raw_notes, self.analysis(clip_id)["duration"])
        if not notes:
            raise ValueError("a clip needs at least one note")
        save_notes(self.truth_path(clip_id), notes)
        return len(notes)


def make_demo_store() -> Store:
    """A throwaway set of fake hums, for trying the page without touching real data."""
    store = Store(Path(tempfile.mkdtemp(prefix="hum-to-track-demo-")))
    for i, melody in enumerate(DEMO_MELODIES, start=1):
        clip_id = f"clip_{i:03d}"
        audio, _ = make_hum(melody, seed=i)
        path = store.audio_path(clip_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, 44100, subtype="PCM_16")
        store.prompts[clip_id] = f"Demo clip {i}: a fake hum made by the test tools."
    return store


def make_server(store: Store, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _clip_id(self, value: str) -> str | None:
            exists = is_clip_id(value) and store.audio_path(value).exists()
            if not exists:
                self._json(404, {"error": "no such clip"})
            return value if exists else None

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            elif path == "/api/clips":
                self._json(200, store.list_clips())
            elif path.startswith("/api/clip/"):
                clip_id = self._clip_id(path.removeprefix("/api/clip/"))
                if clip_id:
                    self._json(200, store.clip_detail(clip_id))
            elif path.startswith("/audio/"):
                clip_id = self._clip_id(path.removeprefix("/audio/"))
                if clip_id:
                    self._send(200, store.audio_path(clip_id).read_bytes(), "audio/wav")
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if not path.startswith("/api/clip/"):
                return self._json(404, {"error": "not found"})
            clip_id = self._clip_id(path.removeprefix("/api/clip/"))
            if not clip_id:
                return
            if self.headers.get("Content-Type") != "application/json":
                return self._json(415, {"error": "expected JSON"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                return self._json(413, {"error": "too large"})
            try:
                body = json.loads(self.rfile.read(length))
                count = store.save_truth(clip_id, body["notes"])
            except (ValueError, KeyError, TypeError) as e:
                return self._json(400, {"error": str(e)})
            self._json(200, {"saved": count})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
