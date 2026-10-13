#!/usr/bin/env python3
"""Opus Track B Look Protocol: official detector + spec extractors -> observation.

Runs the official BalancedStarlightDetector (models/detector_balanced.pth, the
same weights as the Trin container models/starlight.gguf) for detection, and a
set of STEGO_FORMAT_SPEC v2.1 extractors that read the payload bytes. Emits the
<starlight_observation> block, JSON, and a GBNF "quote lock" grammar that makes
llama.cpp copy the payload verbatim. This is a preprocessor, not a language model.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AI42 = b"AI42"
MAX_PAYLOAD_BYTES = 64 * 1024
# Detector-only stego claims (nothing decoded). The detector's own operating point;
# the highest clean-cover probability in eval/bench_look.py --detector was 0.363.
DETECT_ONLY_THRESHOLD = 0.5
EXIF_IFD = 0x8769
USER_COMMENT = 0x9286
XP_COMMENT = 0x9C9C

METHOD_LABEL = {
    "alpha": "alpha-channel LSB",
    "palette": "palette-index LSB",
    "lsb.rgb": "RGB LSB",
    "exif": "EXIF UserComment",
    "raw": "data appended after the end-of-image marker",
}


@dataclass
class Candidate:
    method: str
    payload: str
    bit_order: str
    rank: int  # 3 = AI42 marker, 2 = container field, 1 = bare pixel stream
    source: str


# ---------------------------------------------------------------- text checks


def _printable(text: str) -> bool:
    for ch in text:
        if ch in "\n\r\t":
            continue
        if ch == "\ufffd" or unicodedata.category(ch)[0] == "C":
            return False
    return True


def _plausible_bare(text: str) -> bool:
    """Bare LSB streams have no marker, so demand text that looks written."""
    s = text.strip()
    if len(s) < 4 or len(set(s)) < 3:
        return False
    if len(s) < 12 and not any(ch.isspace() for ch in s):
        return False
    letters = sum(ch.isalpha() for ch in s)
    spaces = sum(ch.isspace() for ch in s)
    if letters < 3 or (letters + spaces) < 0.6 * len(s):
        return False
    return True


def _decode_utf8(raw: bytes) -> Optional[str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    text = text.strip()
    if not text or not _printable(text):
        return None
    return text


def _until_nul(raw: bytes) -> Tuple[bytes, bool]:
    end = raw.find(b"\x00")
    if end < 0:
        return raw, False
    return raw[:end], True


# ------------------------------------------------------------ pixel streams


def _pixel_streams(img: Image.Image, fmt: str) -> Iterable[Tuple[str, str, np.ndarray]]:
    """Yield (method, variant, lsb bit array) in spec order."""
    if fmt in {"JPEG", "MPO"}:
        return
    mode = img.mode
    if mode == "P" or (mode == "L" and fmt == "GIF"):
        yield "palette", "index", np.asarray(img, dtype=np.uint8).ravel() & 1
        # Some generators flip the LSB of each pixel's palette colour instead of its index.
        yield "palette", "color", np.asarray(img.convert("RGB"), dtype=np.uint8)[..., 0].ravel() & 1
        return
    has_alpha = mode in {"RGBA", "LA", "PA", "RGBa", "La"} or "transparency" in img.info
    rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)
    if has_alpha:
        yield "alpha", "a", rgba[..., 3].ravel() & 1
    yield "lsb.rgb", "rgb", rgba[..., :3].ravel() & 1
    if has_alpha:
        yield "lsb.rgb", "rgba", rgba.ravel() & 1


def _stream_candidates(method: str, variant: str, bits: np.ndarray) -> List[Candidate]:
    out: List[Candidate] = []
    n = min(bits.size, 8 * (MAX_PAYLOAD_BYTES + len(AI42) + 1))
    n -= n % 8
    if n < 16:
        return out
    bits = bits[:n]
    for order, np_order in (("lsb-first", "little"), ("msb-first", "big")):
        raw = np.packbits(bits, bitorder=np_order).tobytes()
        marker = raw.startswith(AI42)
        body, terminated = _until_nul(raw[len(AI42):] if marker else raw)
        if not terminated or not body:
            continue
        text = _decode_utf8(body)
        if text is None:
            continue
        if not marker and not _plausible_bare(text):
            continue
        out.append(
            Candidate(
                method=method,
                payload=text,
                bit_order=order,
                rank=3 if marker else 1,
                source=f"{method}:{variant}:{order}{':AI42' if marker else ''}",
            )
        )
    return out


# ---------------------------------------------------------------- metadata


def _decode_user_comment(raw: Any) -> Optional[str]:
    if isinstance(raw, str):
        return _decode_utf8(raw.encode("utf-8"))
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        return None
    raw = bytes(raw)
    head, body = raw[:8], raw[8:]
    if head == b"UNICODE\x00":
        if body[:2] in (b"\xff\xfe", b"\xfe\xff"):
            text = body.decode("utf-16", errors="ignore")
        else:
            be = body.decode("utf-16-be", errors="ignore")
            le = body.decode("utf-16-le", errors="ignore")
            text = be if sum(c.isascii() for c in be) >= sum(c.isascii() for c in le) else le
        text = text.split("\x00", 1)[0].strip()
        return text if text and _printable(text) else None
    if head == b"JIS\x00\x00\x00\x00\x00":
        text = _until_nul(body)[0].decode("shift_jis", errors="ignore").strip()
        return text if text and _printable(text) else None
    if head in (b"ASCII\x00\x00\x00", b"\x00" * 8):
        raw = body
    return _decode_utf8(_until_nul(raw)[0])


def _exif_candidates(img: Image.Image, data: bytes) -> List[Candidate]:
    comments: List[Any] = []
    try:
        exif = img.getexif()
        comments.append(exif.get_ifd(EXIF_IFD).get(USER_COMMENT))
        xp = exif.get(XP_COMMENT)
        if isinstance(xp, (bytes, bytearray, tuple)):
            xp = bytes(xp).decode("utf-16-le", errors="ignore").split("\x00", 1)[0]
            comments.append(xp)
    except Exception:  # noqa: BLE001
        pass
    if not any(comments):
        try:
            import piexif

            src = img.info.get("exif") or data
            d = piexif.load(src)
            comments.append(d.get("Exif", {}).get(piexif.ExifIFD.UserComment))
        except Exception:  # noqa: BLE001
            pass
    out = []
    for raw in comments:
        text = _decode_user_comment(raw)
        if text:
            out.append(Candidate("exif", text, "none", 2, "exif:UserComment"))
            break
    return out


# ------------------------------------------------------------ trailing bytes


def _jpeg_end(data: bytes) -> Optional[int]:
    pos = data.rfind(b"\xff\xd9")
    return pos + 2 if pos >= 0 else None


def _png_end(data: bytes) -> Optional[int]:
    pos = 8
    while pos + 12 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        pos += 12 + length
        if ctype == b"IEND":
            return pos
    return None


def _gif_end(data: bytes) -> Optional[int]:
    if len(data) < 13:
        return None
    flags = data[10]
    pos = 13 + (3 * (2 ** ((flags & 7) + 1)) if flags & 0x80 else 0)

    def skip_sub_blocks(p: int) -> int:
        while p < len(data):
            size = data[p]
            p += 1
            if size == 0:
                return p
            p += size
        return p

    while pos < len(data):
        tag = data[pos]
        if tag == 0x3B:
            return pos + 1
        if tag == 0x21:
            pos = skip_sub_blocks(pos + 2)
        elif tag == 0x2C:
            if pos + 10 > len(data):
                return None
            lflags = data[pos + 9]
            pos += 10
            if lflags & 0x80:
                pos += 3 * (2 ** ((lflags & 7) + 1))
            pos = skip_sub_blocks(pos + 1)
        else:
            return None
    return None


def _riff_end(data: bytes) -> Optional[int]:
    (size,) = struct.unpack("<I", data[4:8])
    end = 8 + size
    return end + (end & 1) if end <= len(data) else None


def _bmp_end(data: bytes) -> Optional[int]:
    (size,) = struct.unpack("<I", data[2:6])
    return size if 0 < size <= len(data) else None


def _container_end(data: bytes) -> Optional[int]:
    if data[:2] == b"\xff\xd8":
        return _jpeg_end(data)
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _png_end(data)
    if data[:4] == b"GIF8":
        return _gif_end(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _riff_end(data)
    if data[:2] == b"BM":
        return _bmp_end(data)
    return None


def _trailing_candidates(data: bytes) -> List[Candidate]:
    end = _container_end(data)
    if end is None or end >= len(data):
        return []
    tail = data[end:]
    if tail.startswith(AI42):
        tail = tail[len(AI42):]
    text = _decode_utf8(_until_nul(tail)[0])
    if not text:
        return []
    return [Candidate("raw", text, "none", 2, "raw:after-end-marker")]


# ---------------------------------------------------------------- extraction


def extract_candidates(image: Path) -> Tuple[List[Candidate], Optional[str]]:
    data = image.read_bytes()
    cands: List[Candidate] = []
    error = None
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format or ""
        img.seek(0)
        img.load()
    except Exception as exc:  # noqa: BLE001
        img, fmt, error = None, "", f"decode: {exc}"
    if img is not None:
        cands.extend(_exif_candidates(img, data))
        for method, variant, bits in _pixel_streams(img, fmt):
            cands.extend(_stream_candidates(method, variant, bits))
    cands.extend(_trailing_candidates(data))
    return cands, error


def _bit_error_marks(text: str) -> int:
    """Digits wedged between letters ("circui4s") betray flipped bits."""
    return sum(
        text[i].isdigit() and text[i - 1].isalpha() and text[i + 1].isalpha()
        for i in range(1, len(text) - 1)
    )


def choose(cands: List[Candidate], hinted: Optional[str]) -> Optional[Candidate]:
    if not cands:
        return None

    def key(c: Candidate) -> Tuple[int, int, int, int]:
        spec_order = int(c.method != "alpha" or c.bit_order == "lsb-first")
        return (c.rank, int(c.method == hinted), spec_order, len(c.payload) - 20 * _bit_error_marks(c.payload))

    return max(cands, key=key)


# ------------------------------------------------------------------ detector


def default_detector() -> Path:
    pth = ROOT / "models" / "detector_balanced.pth"
    return pth if pth.is_file() else ROOT / "models" / "starlight.gguf"


def run_detector(image: Path, detector: Path) -> Dict[str, Any]:
    """Raw BalancedStarlightDetector output (scanner heuristics disabled)."""
    import contextlib

    with contextlib.redirect_stdout(sys.stderr):
        import scanner

        scanner.NO_HEURISTICS = True
        if scanner.SESSION is None:
            scanner.init_worker(str(detector))
        res = scanner._scan_logic(str(image), scanner.SESSION, extract_message=False)
    if "error" in res:
        raise RuntimeError(res["error"])
    return {
        "stego_probability": float(res["stego_probability"]),
        "method": scanner.METHOD_MAP.get(res.get("method_id"), "none"),
        "bit_order": res.get("bit_order") or "none",
    }


# ------------------------------------------------------------------- observe


def observe(image: Path, detector: Optional[Path] = None, use_detector: bool = True) -> Dict[str, Any]:
    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    det = detector or default_detector()
    run_path = det
    if det.suffix.lower() == ".gguf":
        run_path = det.with_name("detector_balanced.pth")
    notes: List[str] = []
    scan: Optional[Dict[str, Any]] = None
    if use_detector:
        if run_path.is_file():
            try:
                scan = run_detector(image, run_path)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"detector error: {exc}")
        else:
            notes.append(f"detector missing: {run_path.name}")

    cands, decode_err = extract_candidates(image)
    if decode_err:
        notes.append(decode_err)
    hinted = scan["method"] if scan else None
    best = choose(cands, hinted)
    p = scan["stego_probability"] if scan else None

    if best is not None:
        status, method, bit_order, payload = "stego", best.method, best.bit_order, best.payload
        confidence = 1.0
        notes.append(f"decoded {best.source}")
    elif p is not None and p >= DETECT_ONLY_THRESHOLD:
        status, method, payload = "stego", scan["method"], ""
        bit_order = scan["bit_order"] if method in {"alpha", "palette", "lsb.rgb"} else "none"
        confidence = p
        notes.append("detector flags a payload; no extractor could decode it")
    elif p is not None or not use_detector:
        status, method, bit_order, payload = "clean", "none", "none", ""
        confidence = p if p is not None else 0.0
    else:
        status, method, bit_order, payload = "unknown", "none", "none", ""
        confidence = 0.0
    if scan:
        notes.insert(0, f"detector p={p:.3f} predicts {scan['method']}/{scan['bit_order']}")

    return {
        "status": status,
        "method": method,
        "confidence": round(float(confidence), 3),
        "bit_order": bit_order,
        "detector": run_path.name if scan else ("none" if not use_detector else det.name),
        "payload": payload,
        "notes": "; ".join(notes),
        "scan": scan,
        "candidates": [c.__dict__ for c in cands],
    }


def format_block(obs: Dict[str, Any]) -> str:
    notes = (obs.get("notes") or "").replace("\n", " ")
    return (
        "<starlight_observation>\n"
        f"status: {obs['status']}\n"
        f"method: {obs['method']}\n"
        f"confidence: {obs['confidence']:.3f}\n"
        f"bit_order: {obs['bit_order']}\n"
        f"detector: {obs['detector']}\n"
        f'payload: """{obs.get("payload") or ""}"""\n'
        f"notes: {notes}\n"
        "</starlight_observation>\n"
    )


# ------------------------------------------------------------- quote lock


def _gbnf_literal(text: str) -> str:
    out = ['"']
    for ch in text:
        cp = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif cp < 0x20 or cp == 0x7F:
            out.append(f"\\x{cp:02X}")
        elif cp > 0x7E:
            out.append(f"\\u{cp:04X}" if cp <= 0xFFFF else f"\\U{cp:08X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def answer_lead(obs: Dict[str, Any]) -> Tuple[str, str]:
    """Fixed facts the reply must state, split around the verbatim quote."""
    status, method = obs["status"], obs["method"]
    order = obs["bit_order"]
    where = METHOD_LABEL.get(method, method)
    order_txt = f", {order}" if order != "none" else ""
    if status == "stego" and obs.get("payload"):
        return (
            f"Yes, this picture hides a secret ({where}; method: {method}{order_txt}). "
            "The recovered payload reads: \"",
            '"',
        )
    if status == "stego":
        return (
            f"A hidden payload appears to be present (predicted method: {method}{order_txt}; "
            f"detector confidence {obs['confidence']:.2f}), but its bytes could not be decoded, "
            "so I have no secret text to quote.",
            "",
        )
    if status == "clean":
        return (
            "No secret found. The Starlight detector reads this picture as clean and no "
            "extractor recovered a payload, so there is nothing to quote and I will not invent one.",
            "",
        )
    return (
        "I could not complete the look on this picture, so I cannot say whether anything is "
        "hidden, and I will not guess a payload.",
        "",
    )


def format_grammar(obs: Dict[str, Any]) -> str:
    """GBNF that pins the factual lead and the exact payload, then frees one sentence."""
    lead, close = answer_lead(obs)
    lines = [f"root ::= lead{' payload close' if close else ''} tail"]
    lines.append(f"lead ::= {_gbnf_literal(lead)}")
    if close:
        lines.append(f"payload ::= {_gbnf_literal(obs['payload'])}")
        lines.append(f"close ::= {_gbnf_literal(close)}")
    lines.append('tail ::= "" | " " [A-Za-z] [^"\\n\\x60:<>]{0,180}')
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Image to look at")
    parser.add_argument("--detector", default=None, help="detector_balanced.pth or starlight.gguf")
    parser.add_argument("--json", action="store_true", help="Print the observation as JSON")
    parser.add_argument(
        "--block",
        action="store_true",
        help="Print system_prompt.txt followed by the observation (for llama-cli -sys)",
    )
    parser.add_argument("--gbnf", action="store_true", help="Print the quote-lock grammar only")
    parser.add_argument("--grammar-out", default=None, help="Also write the quote-lock grammar here")
    parser.add_argument("--no-detector", action="store_true", help="Extractors only (benchmarks)")
    args = parser.parse_args()
    obs = observe(
        Path(args.image),
        detector=Path(args.detector) if args.detector else None,
        use_detector=not args.no_detector,
    )
    if args.grammar_out:
        Path(args.grammar_out).write_text(format_grammar(obs), encoding="utf-8")
    if args.json:
        print(json.dumps(obs, indent=2, ensure_ascii=False, default=str))
    elif args.gbnf:
        print(format_grammar(obs), end="")
    else:
        if args.block:
            sys_path = HERE / "model" / "system_prompt.txt"
            if sys_path.is_file():
                print(sys_path.read_text(encoding="utf-8").rstrip())
                print()
        print(format_block(obs), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
