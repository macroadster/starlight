#!/usr/bin/env python3
"""
Bridge utility: watches the Starlight inscription outbox and forwards stego images
to Stargate's /api/inscribe endpoint, then moves processed files to a sent folder.
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward pending Starlight inscriptions to Stargate."
    )
    parser.add_argument(
        "--outbox",
        default=os.environ.get("STARLIGHT_INSCRIPTION_OUTBOX", "inscriptions/pending"),
        help="Directory to watch for pending inscription files.",
    )
    parser.add_argument(
        "--sent",
        default=os.environ.get("STARLIGHT_INSCRIPTION_SENT", "inscriptions/sent"),
        help="Directory to move successfully forwarded files.",
    )
    parser.add_argument(
        "--stargate-url",
        default=os.environ.get(
            "STARGATE_INSCRIBE_URL", "http://localhost:3001/api/inscribe"
        ),
        help="Stargate inscription endpoint.",
    )
    parser.add_argument(
        "--text",
        default=os.environ.get(
            "STARLIGHT_INSCRIPTION_TEXT", "starlight inscription payload"
        ),
        help="Default text/description to attach to the Stargate contract.",
    )
    parser.add_argument(
        "--expected-output",
        default=os.environ.get("STARLIGHT_EXPECTED_OUTPUT", "image/png"),
        help="Expected output format hint for Stargate.",
    )
    parser.add_argument(
        "--price",
        type=float,
        default=float(os.environ.get("STARLIGHT_INSCRIPTION_PRICE", "0.0")),
        help="Price in BTC to include with the inscription request.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.environ.get("STARLIGHT_WATCH_INTERVAL", "10")),
        help="Seconds between directory scans.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process the outbox once and exit (no watch loop).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pending files without forwarding or moving them.",
    )
    return parser.parse_args()


def list_pending(outbox: Path) -> List[Path]:
    return sorted([p for p in outbox.glob("*") if p.is_file()])


def forward_file(path: Path, args: argparse.Namespace) -> bool:
    try:
        with path.open("rb") as f:
            files = {"image": (path.name, f, "image/png")}
            data = {
                "text": args.text,
                "expectedOutput": args.expected_output,
                "price": str(args.price),
            }
            resp = requests.post(
                args.stargate_url, data=data, files=files, timeout=args.timeout
            )
        if resp.status_code >= 400:
            logging.error(
                "Stargate responded with %s for %s: %s",
                resp.status_code,
                path.name,
                resp.text,
            )
            return False
        logging.info("Forwarded %s to Stargate (%s)", path.name, args.stargate_url)
        return True
    except Exception as e:
        logging.error("Failed to forward %s: %s", path.name, e)
        return False


def process_once(args: argparse.Namespace) -> None:
    outbox = Path(args.outbox)
    sent = Path(args.sent)
    outbox.mkdir(parents=True, exist_ok=True)
    sent.mkdir(parents=True, exist_ok=True)

    pending = list_pending(outbox)
    if not pending:
        logging.info("No pending inscriptions in %s", outbox)
        return

    logging.info("Found %d pending file(s)", len(pending))
    for path in pending:
        if args.dry_run:
            logging.info("[dry-run] Would forward %s", path)
            continue
        if forward_file(path, args):
            dest = sent / path.name
            try:
                shutil.move(str(path), dest)
                logging.info("Moved %s -> %s", path, dest)
            except Exception as e:
                logging.error("Forwarded but failed to move %s: %s", path, e)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()

    if args.once:
        process_once(args)
        return 0

    logging.info(
        "Watching %s every %ds -> %s", args.outbox, args.interval, args.stargate_url
    )
    try:
        while True:
            process_once(args)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logging.info("Exiting watcher")
        return 0
    except Exception as e:
        logging.error("Watcher failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
