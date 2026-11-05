"""Runs inside StudioKit's Python: saves a part and imports it into a new FL pattern.

Reads its job as JSON on stdin and prints the result as JSON. Called by to_fl.py.
"""
from __future__ import annotations

import json
import sys
import time


def main() -> int:
    job = json.load(sys.stdin)
    sys.path.insert(0, job["studiokit"])
    from core import tools
    from core.bridge import FLBridge

    bridge = FLBridge()
    try:
        deadline = time.time() + 5.0
        while not bridge.connected() and time.time() < deadline:
            time.sleep(0.1)
        if not bridge.connected():
            print("FL Studio isn't connected. Start loopMIDI, then FL, and enable the StudioKit script.")
            return 1

        project = bridge.request("project")
        existing = bridge.request("patterns")
        pattern = max((p["i"] for p in existing), default=0) + 1
        channel = job["channel"] if job["channel"] is not None else project["selected_channel"]

        scale = "minor" if job["key"] else None
        tools.song_plan(bridge, job["song"], bpm=job["bpm"], key=job["key"], scale=scale)
        tools.write_part(bridge, job["song"], job["part"], job["notes"], role="melody")
        if job["match_tempo"]:
            tools.set_tempo(bridge, job["bpm"])
        message = tools.import_part(bridge, job["song"], job["part"], channel=channel, pattern=pattern)
        print(json.dumps({
            "message": message,
            "imported": message.startswith("Imported"),
            "pattern": pattern,
            "channel": channel,
            "project_bpm": project["bpm"],
        }))
        return 0
    finally:
        bridge.close()


if __name__ == "__main__":
    sys.exit(main())
