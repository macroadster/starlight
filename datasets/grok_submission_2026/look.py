#!/usr/bin/env python3
"""Track B Look Protocol: official detector + spec extractors → observation block.

This is a preprocessor for llama.cpp. It is not a second language model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHOD_TO_EXTRACTOR = {
    "alpha": "alpha",
    "palette": "palette",
    "lsb.rgb": "lsb",
    "lsb": "lsb",
    "exif": "exif",
    "raw": "eoi",
    "eoi": "eoi",
}


def default_detector() -> Path:
    pth = ROOT / "models" / "detector_balanced.pth"
    if pth.is_file():
        return pth
    gguf = ROOT / "models" / "starlight.gguf"
    return gguf


def _scan(image: Path, detector: Path) -> Dict[str, Any]:
    import scanner

    scanner.init_worker(str(detector))
    return scanner._scan_logic(str(image), scanner.SESSION, extract_message=True)


def _extract_any(image: Path, hinted: Optional[str], bit_order: str) -> Dict[str, Any]:
    from scripts.starlight_extractor import extraction_functions

    order = []
    if hinted:
        order.append(METHOD_TO_EXTRACTOR.get(hinted, hinted))
    for name in ("alpha", "exif", "eoi", "lsb", "palette"):
        if name not in order:
            order.append(name)
    last_err = None
    for name in order:
        fn = extraction_functions.get(name)
        if fn is None:
            continue
        try:
            msg, _ = fn(str(image), bit_order=bit_order)
        except TypeError:
            try:
                msg, _ = fn(str(image))
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                continue
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            continue
        if msg:
            method = {
                "alpha": "alpha",
                "palette": "palette",
                "lsb": "lsb.rgb",
                "exif": "exif",
                "eoi": "raw",
            }.get(name, name)
            return {"method": method, "payload": str(msg), "error": None}
    return {"method": hinted or "none", "payload": "", "error": last_err}


def _meaningful(text: str) -> bool:
    s = text.replace("\x00", "").strip()
    if len(s) < 4:
        return False
    if len(set(s)) < 3:
        return False
    if s.isspace() or s.endswith("?"):
        return False
    return True


def observe(image: Path, detector: Optional[Path] = None) -> Dict[str, Any]:
    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    det = detector or default_detector()
    scan: Dict[str, Any] = {}
    scan_error = None
    scan_path = det
    if det.suffix.lower() == ".gguf":
        pth = ROOT / "models" / "detector_balanced.pth"
        if pth.is_file():
            scan_path = pth
        else:
            scan_error = (
                "starlight.gguf is the Trin detector container; "
                "look.py runs models/detector_balanced.pth (same weights) when present"
            )
    if scan_path.is_file() and scan_path.suffix == ".pth":
        try:
            scan = _scan(image, scan_path)
        except Exception as exc:  # noqa: BLE001
            scan_error = str(exc)
    hinted = scan.get("stego_type") if scan.get("is_stego") else None
    bit_order = scan.get("bit_order") or "none"
    extracted = _extract_any(image, hinted, bit_order)
    payload = scan.get("extracted_message") or extracted.get("payload") or ""
    payload = str(payload).replace("\x00", "").strip()
    if not _meaningful(payload):
        payload = ""
    is_stego = bool(scan.get("is_stego")) or bool(payload)
    method = extracted.get("method") if payload else (hinted or "none")
    if method == "lsb":
        method = "lsb.rgb"
    if method == "eoi":
        method = "raw"
    status = "stego" if is_stego else "clean"
    if scan_error and not payload and not scan:
        status = "unknown"
    conf = scan.get("stego_probability")
    if conf is None:
        conf = 1.0 if payload else 0.0
    detector_id = (
        scan_path.name if scan_path.is_file() else (det.name if det.is_file() else "missing")
    )
    notes = []
    if det.suffix.lower() == ".gguf" and scan_path.suffix == ".pth":
        notes.append("ran detector_balanced.pth (Trin GGUF is not a Python CNN)")
    if scan_error:
        notes.append(f"detector: {scan_error}")
    if extracted.get("error") and not payload:
        notes.append(f"extract: {extracted['error']}")
    return {
        "status": status,
        "method": method or "none",
        "confidence": float(conf),
        "bit_order": bit_order if is_stego else "none",
        "detector": detector_id,
        "payload": payload,
        "notes": "; ".join(notes),
        "scan": {k: v for k, v in scan.items() if k != "extracted_message"} or None,
    }


def format_block(obs: Dict[str, Any]) -> str:
    payload = obs.get("payload") or ""
    notes = obs.get("notes") or ""
    return (
        "<starlight_observation>\n"
        f"status: {obs['status']}\n"
        f"method: {obs['method']}\n"
        f"confidence: {obs['confidence']:.3f}\n"
        f"bit_order: {obs['bit_order']}\n"
        f"detector: {obs['detector']}\n"
        f'payload: """{payload}"""\n'
        f"notes: {notes}\n"
        "</starlight_observation>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Image to look at")
    parser.add_argument(
        "--detector",
        default=None,
        help="Path to detector_balanced.pth (default: models/detector_balanced.pth)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the block")
    parser.add_argument(
        "--block",
        action="store_true",
        help="Print only the Look Protocol block (for llama-cli -sys)",
    )
    args = parser.parse_args()
    det = Path(args.detector) if args.detector else None
    obs = observe(Path(args.image), detector=det)
    if args.json:
        print(json.dumps(obs, indent=2, default=str))
    else:
        text = format_block(obs)
        if args.block:
            sys_path = HERE / "model" / "system_prompt.txt"
            if sys_path.is_file():
                print(sys_path.read_text(encoding="utf-8").rstrip())
                print()
            print(text, end="")
        else:
            print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
