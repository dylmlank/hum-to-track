"""Hum to Track command line. Run `python ht.py --help`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def cmd_record(args) -> None:
    from humtrack.record import run

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    run(DATA_DIR, device=device, clip=args.clip)


def cmd_review(args) -> None:
    from humtrack.review import Store, make_demo_store, make_server

    store = make_demo_store() if args.demo else Store(DATA_DIR)
    server = make_server(store, args.port)
    print(f"Open http://127.0.0.1:{server.server_address[1]}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_score(args) -> None:
    from humtrack.score import format_report, score_folder

    split = "hidden" if args.final else "tune"
    if not args.predictions.is_dir():
        sys.exit(f"Not a folder: {args.predictions}")
    print(format_report(score_folder(args.predictions, DATA_DIR / "truth", split)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)

    record = commands.add_parser("record", help="record the hums, one prompt at a time")
    record.add_argument("--clip", type=int, help="redo one clip by number")
    record.add_argument("--device", help="mic name or number (default: system mic)")
    record.set_defaults(run=cmd_record)

    review = commands.add_parser("review", help="open the note correction page")
    review.add_argument("--port", type=int, default=9791)
    review.add_argument("--demo", action="store_true", help="use fake hums instead of your recordings")
    review.set_defaults(run=cmd_review)

    score = commands.add_parser("score", help="grade a folder of predicted notes")
    score.add_argument("predictions", type=Path, help="folder of clip_NNN.json or clip_NNN.mid files")
    score.add_argument("--final", action="store_true", help="use the hidden clips (for final results only)")
    score.set_defaults(run=cmd_score)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
