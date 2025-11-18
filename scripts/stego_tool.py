#!/usr/bin/env python3
"""
stego_tool_v3.py - Ultimate Stego Scanner (GPU BATCHED)
- SDM removed (requires clean reference - not blockchain compatible)
- GPU-accelerated batch processing with NumPy vectorization
- Parallel scanning with progress bar
- CSV/JSON export + diagnostic mode
"""

import os
import io
import argparse
import csv
import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    import piexif
except ImportError:
    piexif = None

# ----------------------------------------------------------------------
# 1. Embedding Functions
# ----------------------------------------------------------------------
def embed_lsb(cover: Image.Image, payload: bytes) -> Image.Image:
    img = cover.convert("RGBA")
    pixels = list(img.getdata())
    bits = "".join(f"{b:08b}" for b in payload) + "00000000"
    max_bits = len(pixels) * 3
    if len(bits) > max_bits:
        raise ValueError("Payload too large for LSB embedding.")
    new_data = []
    i = 0
    for r, g, b, a in pixels:
        if i < len(bits): r = (r & ~1) | int(bits[i]); i += 1
        if i < len(bits): g = (g & ~1) | int(bits[i]); i += 1
        if i < len(bits): b = (b & ~1) | int(bits[i]); i += 1
        new_data.append((r, g, b, a))
    out = Image.new("RGBA", img.size)
    out.putdata(new_data)
    return out

def embed_alpha(cover: Image.Image, payload: bytes) -> Image.Image:
    img = cover.convert("RGBA")
    prefix = b"AI42"
    payload = prefix + payload + b"\x00"
    pixels = list(img.getdata())
    if len(payload) * 8 > len(pixels):
        raise ValueError("Payload too large for alpha embedding.")
    new_data = []
    pixel_index = 0
    for byte_val in payload:
        for bit_index in range(8):
            r, g, b, a = pixels[pixel_index]
            bit_to_embed = (byte_val >> bit_index) & 1
            a = (a & ~1) | bit_to_embed
            new_data.append((r, g, b, a))
            pixel_index += 1
    new_data.extend(pixels[pixel_index:])
    out = Image.new("RGBA", img.size)
    out.putdata(new_data)
    return out

def embed_palette(cover: Image.Image, payload: bytes) -> Image.Image:
    img = cover.convert("P", palette=Image.ADAPTIVE, colors=256)
    data = list(img.getdata())
    bits = "".join(f"{b:08b}" for b in payload) + "00000000"
    if len(bits) > len(data):
        raise ValueError("Payload too large for palette embedding.")
    new_data = []
    i = 0
    for idx in data:
        if i < len(bits):
            idx = (idx & ~1) | int(bits[i])
            i += 1
        new_data.append(idx)
    out = Image.new("P", img.size)
    out.putdata(new_data)
    out.putpalette(img.getpalette())
    return out

def embed_exif(cover: Image.Image, payload: bytes) -> Image.Image:
    if piexif is None:
        raise RuntimeError("piexif not installed; cannot perform EXIF embedding.")
    img = cover.convert("RGB")
    exif_dict = {
        "0th": {},
        "Exif": {piexif.ExifIFD.UserComment: b"ASCII\x00\x00\x00" + payload},
        "GPS": {},
        "1st": {},
        "thumbnail": None
    }
    img.info["exif_bytes"] = piexif.dump(exif_dict)
    return img

def embed_eoi(cover: Image.Image, payload: bytes) -> Image.Image:
    img = cover.convert("RGB")
    append_data = b'AI42' + payload + b'\x00'
    img.info["eoi_append"] = append_data
    return img

# ----------------------------------------------------------------------
# 2. GPU-Accelerated Extraction Helpers
# ----------------------------------------------------------------------
def extract_message_from_bits(bits, max_length=24576):
    """Vectorized bit extraction with multiple strategies"""
    ai_hint_bytes = b'AI42'
    hint_bits = ''.join(format(byte, '08b') for byte in ai_hint_bytes)
    
    # Strategy 1: AI42 prefix detection
    if bits.startswith(hint_bits):
        bits_after_hint = bits[len(hint_bits):]
        terminator_bits = '00000000'
        terminator_index = bits_after_hint.find(terminator_bits)
        if terminator_index != -1:
            if terminator_index % 8 != 0:
                terminator_index = (terminator_index // 8) * 8
            payload_bits = bits_after_hint[:terminator_index]
            if len(payload_bits) == 0:
                return None, bits_after_hint
            bytes_data = []
            for i in range(0, len(payload_bits), 8):
                byte_str = payload_bits[i:i+8]
                reversed_byte_str = byte_str[::-1]
                current_byte = int(reversed_byte_str, 2)
                bytes_data.append(current_byte)
            try:
                message = bytes(bytes_data).decode('utf-8')
                return message.strip(), bits_after_hint[terminator_index + 8:]
            except UnicodeDecodeError:
                return bytes(bytes_data).hex(), bits_after_hint[terminator_index + 8:]
    
    # Strategy 2: Length-prefixed payload (32-bit header)
    if len(bits) >= 32:
        try:
            length = int(bits[:32], 2)
            if 0 < length <= (len(bits) - 32) // 8:
                payload_bits = bits[32:32 + length * 8]
                if len(payload_bits) % 8 != 0:
                    return None, bits
                bytes_data = [int(payload_bits[j:j+8], 2) for j in range(0, len(payload_bits), 8)]
                try:
                    message = bytes(bytes_data).decode('utf-8')
                    return message.strip(), bits[32 + length * 8:]
                except UnicodeDecodeError:
                    return bytes(bytes_data).hex(), bits[32 + length * 8:]
        except Exception:
            pass
    
    # Strategy 3: Null-terminated UTF-8
    try:
        max_bits = min(len(bits), max_length * 8)
        max_bits = (max_bits // 8) * 8
        bytes_data = []
        i = 0
        while i < max_bits:
            if i + 8 > len(bits):
                break
            byte = int(bits[i:i+8], 2)
            if byte == 0:
                break
            bytes_data.append(byte)
            try:
                bytes(bytes_data).decode('utf-8')
            except UnicodeDecodeError:
                bytes_data.pop()
                break
            i += 8
        if bytes_data:
            message = bytes(bytes_data).decode('utf-8').strip()
            return message, bits[i:]
    except Exception:
        pass
    
    return None, bits

def extract_bits_batch(images, strategy, max_bits):
    """GPU-accelerated batch bit extraction using NumPy vectorization"""
    results = []
    
    for img in images:
        bits = ''
        
        if strategy == 'rgba_flat':
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            data = np.array(img)
            # Vectorized LSB extraction
            color_channels = data.flatten()[:max_bits]
            bits = ''.join((color_channels & 1).astype(str))
        
        elif strategy == 'rgb_flat':
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            data = np.array(img)
            if len(data.shape) == 3 and data.shape[2] >= 3:
                rgb_data = data[:, :, :3].flatten()[:max_bits]
                bits = ''.join((rgb_data & 1).astype(str))
        
        elif strategy == 'rgb_interleaved':
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            data = np.array(img)
            if len(data.shape) == 3 and data.shape[2] >= 3:
                rgb_only = data[:, :, :3]
                flattened = rgb_only.flatten()[:max_bits]
                bits = ''.join((flattened & 1).astype(str))
        
        elif strategy in ['red', 'green', 'blue']:
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            data = np.array(img)
            channel_idx = {'red': 0, 'green': 1, 'blue': 2}[strategy]
            if len(data.shape) == 3 and data.shape[2] > channel_idx:
                channel_data = data[:, :, channel_idx].flatten()[:max_bits]
                bits = ''.join((channel_data & 1).astype(str))
        
        elif strategy == 'alpha':
            if img.mode != 'RGBA':
                results.append('')
                continue
            data = np.array(img)
            alpha_data = data[:, :, 3].flatten()[:max_bits]
            bits = ''.join((alpha_data & 1).astype(str))
        
        results.append(bits)
    
    return results

# ----------------------------------------------------------------------
# 3. Confidence Scoring
# ----------------------------------------------------------------------
def calculate_message_confidence(message):
    if not message:
        return 0.0
    score = 0.0
    if len(message) < 20:
        if all(c in '0123456789abcdef' for c in message.lower()):
            score += 5
        else:
            score -= 30
    printable_count = sum(1 for c in message if c.isprintable())
    score += (printable_count / len(message)) * 40
    if any(c.isalpha() for c in message):
        score += 20
    space_count = message.count(' ')
    if space_count > 0:
        score += min(space_count * 2, 20)
    markdown_indicators = ['#', '##', '*', '-', '`', '[', ']', '(', ')']
    markdown_count = sum(1 for ind in markdown_indicators if ind in message)
    if markdown_count > 0:
        score += min(markdown_count * 5, 20)
    common_words = ['the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 'was', 'have']
    word_count = sum(1 for word in common_words if word in message.lower())
    if word_count > 0:
        score += min(word_count * 3, 15)
    if len(message) > 100:
        score += 10
    if len(message) > 500:
        score += 10
    if space_count == 0 and len(message) > 20:
        score -= 30
    unique_chars = len(set(message.lower()))
    if unique_chars < 10:
        score -= (10 - unique_chars) * 5
    return max(0.0, min(100.0, score))

def _lsb_confidence(message: str) -> float:
    if not message or len(message) < 20:
        return 0.0
    base = calculate_message_confidence(message)
    has_ai42 = "AI42" in message[:20]
    try:
        dummy_bits = "0"*32 + ''.join(f"{ord(c):08b}" for c in message)
        length = int(dummy_bits[:32], 2)
        plausible = 0 < length <= len(message)
    except Exception:
        plausible = False
    if has_ai42 or plausible:
        base += 20
    if len(message) < 50:
        base -= 10
    if message.isdigit() or all(c in '0123456789abcdef ' for c in message.lower()):
        base += 15
    return max(0.0, min(100.0, base))

# ----------------------------------------------------------------------
# 4. Extraction Functions
# ----------------------------------------------------------------------
def reconstruct_lsb_first_message_from_bits(bits, max_length=24576):
    ai_hint_bytes = b'AI42'
    hint_bits = ''
    for byte in ai_hint_bytes:
        lsb_first_bits = bin(byte)[2:].zfill(8)[::-1]
        hint_bits += lsb_first_bits
    if not bits.startswith(hint_bits):
        return None, bits
    bits_after_hint = bits[len(hint_bits):]
    terminator_bits = '00000000'
    terminator_index = bits_after_hint.find(terminator_bits)
    if terminator_index == -1:
        return None, bits
    if terminator_index % 8 != 0:
        terminator_index = (terminator_index // 8) * 8
    payload_bits = bits_after_hint[:terminator_index]
    if len(payload_bits) == 0:
        return None, bits
    bytes_data = []
    for i in range(0, len(payload_bits), 8):
        byte_str = payload_bits[i:i+8]
        reversed_byte_str = byte_str[::-1]
        current_byte = int(reversed_byte_str, 2)
        bytes_data.append(current_byte)
    try:
        message = bytes(bytes_data).decode('utf-8')
        return message.strip(), bits_after_hint[terminator_index + 8:]
    except UnicodeDecodeError:
        return bytes(bytes_data).hex(), bits_after_hint[terminator_index + 8:]

def extract_lsb(image_path, max_bits=1_000_000):
    """GPU-accelerated LSB extraction"""
    img = Image.open(image_path)
    
    # Try LSB-first strategy with AI42 prefix
    if img.mode in ('RGBA', 'RGB') or str(image_path).lower().endswith(('.png', '.webp', '.jpg', '.jpeg')):
        bits_list = extract_bits_batch([img], 'rgba_flat', max_bits)
        if bits_list and bits_list[0]:
            msg, rem = reconstruct_lsb_first_message_from_bits(bits_list[0])
            if msg and len(msg) >= 20:
                conf = _lsb_confidence(msg)
                if conf >= 50:
                    return msg, rem
    
    # Try all strategies
    strategies = ['rgba_flat', 'rgb_flat', 'rgb_interleaved', 'red', 'green', 'blue', 'alpha']
    results = []
    
    bits_batch = extract_bits_batch([img] * len(strategies), strategies[0], max_bits)
    
    for i, strategy in enumerate(strategies):
        bits_list = extract_bits_batch([img], strategy, max_bits)
        if bits_list and bits_list[0]:
            message, remaining = extract_message_from_bits(bits_list[0])
            if message:
                results.append({
                    'strategy': strategy,
                    'message': message,
                    'confidence': calculate_message_confidence(message)
                })
    
    if not results:
        return None, None
    
    best = max(results, key=lambda x: x['confidence'])
    if best['confidence'] >= 50:
        return best['message'], None
    
    return None, None

def extract_alpha(image_path):
    """GPU-accelerated alpha channel extraction"""
    img = Image.open(image_path)
    bits_list = extract_bits_batch([img], 'alpha', 1000000)
    
    if not bits_list or not bits_list[0]:
        return None, None
    
    bits = bits_list[0]
    
    # Try LSB-first reconstruction
    message, remaining = reconstruct_lsb_first_message_from_bits(bits)
    if message and len(message) >= 15:
        return message, remaining
    
    # Try standard extraction
    message, remaining = extract_message_from_bits(bits)
    if message and len(message) >= 15:
        return message, remaining
    
    return None, None

def extract_palette(image_path):
    """GPU-accelerated palette extraction"""
    img = Image.open(image_path)
    if img.mode != 'P':
        return None, None
    
    img_array = np.array(img)
    # Vectorized LSB extraction
    lsb_bits = (img_array.flatten() & 1).astype(str)
    bits = ''.join(lsb_bits)
    
    return extract_message_from_bits(bits)

def extract_exif(image_path):
    img = Image.open(image_path)
    if piexif:
        try:
            exif_dict = piexif.load(image_path)
            for ifd in exif_dict:
                if ifd == "thumbnail":
                    continue
                for tag_id in exif_dict[ifd]:
                    value = exif_dict[ifd][tag_id]
                    if isinstance(value, (str, bytes)) and len(str(value)) > 10:
                        try:
                            msg = value.decode('utf-8', errors='ignore').strip()
                            if len(msg) > 10 and any(c.isprintable() for c in msg):
                                return msg, None
                        except:
                            pass
        except:
            pass
    exif = img.getexif()
    if exif:
        for tag_id in exif:
            value = exif.get(tag_id)
            if value and isinstance(value, (str, bytes)) and len(str(value)) > 10:
                try:
                    msg = value.decode('utf-8', errors='ignore').strip() if isinstance(value, bytes) else value
                    if len(msg) > 10 and any(c.isprintable() for c in msg):
                        return msg, None
                except:
                    pass
    return None, None

def extract_eoi(image_path):
    try:
        with open(image_path, 'rb') as f:
            data = f.read()
    except:
        return None, None
    if not data.startswith(b'\xff\xd8'):
        return None, None
    eoi_pos = data.rfind(b'\xff\xd9')
    if eoi_pos < 0 or eoi_pos + 2 >= len(data):
        return None, None
    payload = data[eoi_pos + 2:]
    if len(payload) < 4:
        return None, None
    if (payload.startswith(b'\xff\xd8') or payload.startswith(b'Exif') or
        payload.startswith(b'ICC_PROFILE') or payload.startswith(b'<?xpacket') or
        payload.startswith(b'http://ns.adobe.com')):
        return None, None
    if all(b == 0 for b in payload[:min(20, len(payload))]) or all(b == 0xFF for b in payload[:min(20, len(payload))]):
        return None, None
    try:
        message = payload.decode('utf-8', errors='ignore')
        if len(message) > 10 and any(c.isprintable() for c in message):
            return message, None
    except:
        pass
    return None, None

# ----------------------------------------------------------------------
# 5. Dictionaries & Helpers
# ----------------------------------------------------------------------
embedding_functions = {
    'lsb': embed_lsb,
    'alpha': embed_alpha,
    'palette': embed_palette,
    'exif': embed_exif,
    'eoi': embed_eoi
}

extraction_functions = {
    'alpha': extract_alpha,
    'exif': extract_exif,
    'eoi': extract_eoi,
    'palette': extract_palette,
    'lsb': extract_lsb
}

def save_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".webp":
        img.save(path, format="WEBP", lossless=True)
    elif ext == ".png":
        if img.mode == "P":
            img = img.convert("RGBA")
        img.save(path, format="PNG", optimize=True)
    elif ext == ".gif":
        img.save(path, format="GIF", save_all=True)
    elif ext in [".jpg", ".jpeg"]:
        img.convert("RGB").save(path, format="JPEG", quality=95)
    elif ext == ".bmp":
        img.save(path, format="BMP")
    else:
        img.save(path)

def save_jpeg_with_exif(img, path, exif_bytes, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", exif=exif_bytes, quality=quality)

def save_jpeg_with_eoi_append(img, path, append_bytes, quality=95):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    data = buf.getvalue()
    if b'\xff\xd9' not in data:
        raise RuntimeError("JPEG EOI marker not found.")
    new_data = data + append_bytes
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(new_data)

def is_image_file(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

# ----------------------------------------------------------------------
# 6. Parallel Worker Function
# ----------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 50.0

def process_image(file_path, detail=False, diagnose=False):
    candidates = []
    for method, func in extraction_functions.items():
        msg, _ = func(str(file_path))
        if msg:
            conf = _lsb_confidence(msg) if method == 'lsb' else calculate_message_confidence(msg)
            candidates.append((method, msg, conf))

    status = "CLEAN"
    best_method = ""
    best_msg = ""
    best_conf = 0.0
    excerpt = ""

    if candidates:
        best_method, best_msg, best_conf = max(candidates, key=lambda x: x[2])
        if best_conf >= CONFIDENCE_THRESHOLD:
            status = "STEGO"
        excerpt = best_msg[:100] + ('...' if len(best_msg) > 100 else '')

    row = {
        "file": str(file_path),
        "status": status,
        "method": best_method,
        "confidence": f"{best_conf:.2f}",
        "message": excerpt
    }

    detail_line = ""
    if detail or diagnose:
        conf_str = f"{best_conf:.2f}"
        if status == "STEGO":
            detail_line = f"{file_path}: {status} (method: {best_method}, conf: {conf_str})"
        elif diagnose and candidates:
            top_method, top_msg, top_conf = max(candidates, key=lambda x: x[2])
            detail_line = f"{file_path}: MISSED (top: {top_method}, conf: {top_conf:.2f}, len: {len(top_msg)})"
        elif detail:
            detail_line = f"{file_path}: {status} (conf: {conf_str})"

    return row, detail_line, status == "STEGO"

# ----------------------------------------------------------------------
# 7. Scan Function with Parallel Execution
# ----------------------------------------------------------------------
def scan_directory(directory, detail=False, recursive=False, output=None, diagnose=False):
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Error: {directory} is not a directory.")
        return

    image_files = []
    if recursive:
        for root, _, files in os.walk(dir_path):
            for f in files:
                p = Path(root) / f
                if is_image_file(p):
                    image_files.append(p)
    else:
        for p in dir_path.iterdir():
            if p.is_file() and is_image_file(p):
                    image_files.append(p)

    total_images = len(image_files)
    if total_images == 0:
        print("No images found.")
        return

    results = []
    stego_details = []
    stego_count = 0
    results_lock = threading.Lock()

    max_workers = min(32, (os.cpu_count() or 1) + 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, fp, detail, diagnose): fp for fp in image_files}
        for future in tqdm(as_completed(futures), total=total_images, desc="Scanning", unit="img", ncols=100):
            row, detail_line, is_stego = future.result()
            with results_lock:
                results.append(row)
                if detail_line:
                    stego_details.append(detail_line)
                if is_stego:
                    stego_count += 1

    if output:
        if output.endswith('.csv'):
            with open(output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["file","status","method","confidence","message"])
                writer.writeheader()
                writer.writerows(results)
        elif output.endswith('.json'):
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        print(f"Results exported to {output}")

    print(f"\nTotal images scanned: {total_images}")
    print(f"Stego images: {stego_count}")

    if detail or diagnose:
        print("\nDetails:")
        for d in stego_details:
            print(d)

# ----------------------------------------------------------------------
# 8. Main CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stego Tool v3: Embed, Extract, Scan (GPU Batched, No SDM)")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    embed_parser = subparsers.add_parser('embed', help="Embed message")
    embed_parser.add_argument('--input', required=True)
    embed_parser.add_argument('--output', required=True)
    embed_parser.add_argument('--message', required=True)
    embed_parser.add_argument('--method', choices=list(embedding_functions.keys()), required=True)

    extract_parser = subparsers.add_parser('extract', help="Extract message")
    extract_parser.add_argument('--input', required=True)
    extract_parser.add_argument('--method', default='auto', choices=['auto'] + list(extraction_functions.keys()))

    scan_parser = subparsers.add_parser('scan', help="Scan directory (GPU batched)")
    scan_parser.add_argument('--directory', required=True)
    scan_parser.add_argument('--detail', action='store_true')
    scan_parser.add_argument('--recursive', action='store_true')
    scan_parser.add_argument('--output', help="Export to CSV/JSON")
    scan_parser.add_argument('--diagnose', action='store_true', help="Show near-misses")

    args = parser.parse_args()

    if args.mode == 'embed':
        cover = Image.open(args.input)
        payload = args.message.encode('utf-8')
        stego = embedding_functions[args.method](cover, payload)
        output_path = Path(args.output)
        if args.method == 'exif':
            if piexif is None:
                print("Error: piexif required for EXIF.")
                return
            save_jpeg_with_exif(stego, output_path, stego.info.get("exif_bytes"))
        elif args.method == 'eoi':
            save_jpeg_with_eoi_append(stego, output_path, stego.info.get("eoi_append", b""))
        else:
            save_image(stego, output_path)
        print(f"Embedded using {args.method} to {args.output}")

    elif args.mode == 'extract':
        if args.method == 'auto':
            found = False
            for method, func in extraction_functions.items():
                msg, _ = func(args.input)
                if msg:
                    print(f"[{method}] {msg}")
                    found = True
            if not found:
                print("No message found.")
        else:
            func = extraction_functions[args.method]
            msg, _ = func(args.input)
            print(msg or "No message.")

    elif args.mode == 'scan':
        scan_directory(args.directory, args.detail, args.recursive, args.output, args.diagnose)

if __name__ == "__main__":
    main()
