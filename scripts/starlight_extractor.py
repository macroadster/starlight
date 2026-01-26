#!/usr/bin/env python3
"""
Steganography Extractor - ALIGNED WITH STEGO_FORMAT_SPEC.md

Key Format Rules:
1. Alpha: LSB-first (byte-reversed) bit order with AI42 hint
2. LSB: Can use various strategies, may have AI42 hint
3. Palette: MSB-first, null-terminated
4. EXIF: UserComment with encoding headers
5. EOI: Raw append after JPEG EOI marker
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse

try:
    import piexif
except ImportError:
    piexif = None
    print("Warning: piexif not installed; EXIF extraction may be limited.")


def extract_message_lsb_first(bits, max_length=24576):
    """
    Extract message from LSB-first (byte-reversed) bit stream.
    Used by Alpha channel method.
    """
    # Build AI42 marker in LSB-first order
    ai_hint_bytes = b"AI42"
    hint_bits = ""
    for byte in ai_hint_bytes:
        # Reverse the bits of each byte (LSB-first)
        lsb_first_bits = format(byte, "08b")[::-1]
        hint_bits += lsb_first_bits

    if not bits.startswith(hint_bits):
        return None, bits

    bits_after_hint = bits[len(hint_bits) :]

    # Look for null terminator (0x00 in LSB-first = '00000000')
    terminator_bits = "00000000"
    terminator_index = -1
    for i in range(0, len(bits_after_hint), 8):
        if bits_after_hint[i : i + 8] == terminator_bits:
            terminator_index = i
            break

    if terminator_index == -1:
        return None, bits

    payload_bits = bits_after_hint[:terminator_index]

    if len(payload_bits) == 0:
        return None, bits

    # Decode bytes in LSB-first order (reverse each byte)
    bytes_data = []
    for i in range(0, len(payload_bits), 8):
        byte_str = payload_bits[i : i + 8]
        if len(byte_str) == 8:
            # Reverse the bits to get MSB-first byte value
            reversed_byte_str = byte_str[::-1]
            bytes_data.append(int(reversed_byte_str, 2))

    try:
        message = bytes(bytes_data).decode("utf-8")
        return message.strip(), bits_after_hint[terminator_index + 8 :]
    except UnicodeDecodeError:
        return bytes(bytes_data).hex(), bits_after_hint[terminator_index + 8 :]


def extract_message_msb_first(bits, max_length=24576):
    """
    Extract message from MSB-first bit stream.
    """
    # Strategy 1: AI42 hint with null terminator
    ai_hint_bytes = b"AI42"
    hint_bits = "".join(format(byte, "08b") for byte in ai_hint_bytes)

    if bits.startswith(hint_bits):
        bits_after_hint = bits[len(hint_bits) :]
        terminator_bits = "00000000"
        terminator_index = bits_after_hint.find(terminator_bits)

        if terminator_index != -1:
            if terminator_index % 8 != 0:
                terminator_index = (terminator_index // 8) * 8

            payload_bits = bits_after_hint[:terminator_index]

            if len(payload_bits) > 0:
                bytes_data = []
                for i in range(0, len(payload_bits), 8):
                    byte_str = payload_bits[i : i + 8]
                    if len(byte_str) == 8:
                        bytes_data.append(int(byte_str, 2))

                try:
                    message = bytes(bytes_data).decode("utf-8")
                    return message.strip(), bits_after_hint[terminator_index + 8 :]
                except UnicodeDecodeError:
                    return (
                        bytes(bytes_data).hex(),
                        bits_after_hint[terminator_index + 8 :],
                    )

    # Strategy 2: Length Prefix
    if len(bits) >= 32:
        try:
            length = int(bits[:32], 2)
            if 0 < length <= (len(bits) - 32) // 8:
                payload_bits = bits[32 : 32 + length * 8]
                if len(payload_bits) % 8 == 0:
                    bytes_data = [
                        int(payload_bits[j : j + 8], 2)
                        for j in range(0, len(payload_bits), 8)
                    ]
                    try:
                        message = bytes(bytes_data).decode("utf-8")
                        return message.strip(), bits[32 + length * 8 :]
                    except UnicodeDecodeError:
                        return bytes(bytes_data).hex(), bits[32 + length * 8 :]
        except Exception:
            pass

    # Strategy 3: Null-Terminated
    try:
        max_bits = min(len(bits), max_length * 8)
        max_bits = (max_bits // 8) * 8
        bytes_data = []
        i = 0
        while i < max_bits:
            if i + 8 > len(bits):
                break
            byte = int(bits[i : i + 8], 2)
            if byte == 0:  # Null terminator
                break
            bytes_data.append(byte)
            i += 8

        if bytes_data:
            try:
                message = bytes(bytes_data).decode("utf-8").strip()
                return message, bits[i:]
            except UnicodeDecodeError:
                pass
    except Exception:
        pass

    return None, bits


def extract_message_best_effort(bits, bit_order="msb-first"):
    """
    Decodes an entire bitstream and returns the longest valid UTF-8 prefix.
    """
    bytes_data = []
    if bit_order == "lsb-first":
        for i in range(0, len(bits), 8):
            byte_str = bits[i : i + 8]
            if len(byte_str) == 8:
                reversed_byte_str = byte_str[::-1]
                bytes_data.append(int(reversed_byte_str, 2))
    else:  # msb-first
        for i in range(0, len(bits), 8):
            byte_str = bits[i : i + 8]
            if len(byte_str) == 8:
                bytes_data.append(int(byte_str, 2))

    if not bytes_data:
        return None, bits

    try:
        message = bytes(bytes_data).decode("utf-8")
        return message.strip(), ""
    except UnicodeDecodeError as e:
        valid_bytes = bytes_data[: e.start]
        if not valid_bytes:
            return None, bits
        try:
            message = bytes(valid_bytes).decode("utf-8")
            return message.strip(), ""
        except Exception:
            return None, bits


def extract_alpha(image_path, bit_order="lsb-first"):
    """
    Extract alpha channel steganography.
    """
    img = Image.open(image_path)
    if img.mode != "RGBA":
        return None, None

    pixels = list(img.getdata())
    bits = "".join(str(pixel[3] & 1) for pixel in pixels)

    # Prioritize bit_order hint
    if bit_order == "lsb-first":
        message, remaining = extract_message_lsb_first(bits)
        if message:
            return message, remaining
        message, remaining = extract_message_msb_first(bits)
    else:
        message, remaining = extract_message_msb_first(bits)
        if message:
            return message, remaining
        message, remaining = extract_message_lsb_first(bits)

    return message, remaining


def extract_bits_by_strategy(img, strategy, max_bits):
    """Extract bits using various strategies for generic LSB."""
    bits = ""

    if strategy == "rgba_flat":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        data = np.array(img)
        color_channels = data.flatten()
        for i, val in enumerate(color_channels):
            if i >= max_bits:
                break
            bits += str(val & 1)

    elif strategy == "rgb_flat":
        if img.mode not in ["RGB", "RGBA"]:
            img = img.convert("RGB")
        data = np.array(img)
        if len(data.shape) == 3 and data.shape[2] >= 3:
            rgb_data = data[:, :, :3].flatten()
            for i, val in enumerate(rgb_data):
                if i >= max_bits:
                    break
                bits += str(val & 1)

    elif strategy == "rgb_interleaved":
        if img.mode not in ["RGB", "RGBA"]:
            img = img.convert("RGB")
        data = np.array(img)
        if len(data.shape) == 3 and data.shape[2] >= 3:
            count = 0
            for row in data:
                for pixel in row:
                    for c in pixel[:3]:
                        bits += str(c & 1)
                        count += 1
                        if count >= max_bits:
                            break
                    if count >= max_bits:
                        break
                if count >= max_bits:
                    break

    elif strategy in ["red", "green", "blue"]:
        if img.mode not in ["RGB", "RGBA"]:
            img = img.convert("RGB")
        data = np.array(img)
        channel_idx = {"red": 0, "green": 1, "blue": 2}[strategy]
        if len(data.shape) == 3 and data.shape[2] > channel_idx:
            channel_data = data[:, :, channel_idx].flatten()
            for i, val in enumerate(channel_data):
                if i >= max_bits:
                    break
                bits += str(val & 1)

    return bits


def extract_lsb(image_path, max_bits=1000000, bit_order="none"):
    """
    Extract generic LSB steganography.
    """
    img = Image.open(image_path)
    strategies = ["rgba_flat", "rgb_flat", "rgb_interleaved", "red", "green", "blue"]

    best_message = None
    best_confidence = 0

    # Determine bit orders to try based on hint
    bit_orders = ["lsb-first", "msb-first"]
    if bit_order in bit_orders:
        bit_orders.remove(bit_order)
        bit_orders.insert(0, bit_order)  # Put hint first

    for strategy in strategies:
        bits = extract_bits_by_strategy(img, strategy, max_bits)
        if not bits:
            continue

        for bo in bit_orders:
            # 1. Try with markers
            if bo == "lsb-first":
                message, _ = extract_message_lsb_first(bits)
            else:
                message, _ = extract_message_msb_first(bits)

            if message and len(message) > 5:
                confidence = calculate_message_confidence(message)
                if confidence > best_confidence:
                    best_message = message
                    best_confidence = confidence

            # 2. Try best effort (no markers)
            message, _ = extract_message_best_effort(bits, bit_order=bo)
            if message and len(message) > 10:
                confidence = calculate_message_confidence(message)
                if confidence > best_confidence:
                    best_message = message
                    best_confidence = confidence

    return best_message, None


def calculate_message_confidence(message):
    """Calculate confidence score for extracted message."""
    if not message:
        return 0.0

    score = 0.0
    if len(message) < 20:
        if all(c in "0123456789abcdef" for c in message.lower()):
            score += 5
        else:
            score -= 30

    printable_count = sum(1 for c in message if c.isprintable())
    score += (printable_count / len(message)) * 40
    if any(c.isalpha() for c in message):
        score += 20
    space_count = message.count(" ")
    if space_count > 0:
        score += min(space_count * 2, 20)
    if len(message) > 100:
        score += 10

    return max(0.0, min(100.0, score))


def extract_palette(image_path, bit_order="msb-first"):
    """
    Extract palette-based steganography.
    """
    img = Image.open(image_path)
    if img.mode != "P":
        img = img.convert("L")

    img_array = np.array(img)
    bits = "".join(str(pixel & 1) for pixel in img_array.flatten())

    bit_orders = ["lsb-first", "msb-first"]
    if bit_order in bit_orders:
        bit_orders.remove(bit_order)
        bit_orders.insert(0, bit_order)

    for bo in bit_orders:
        if bo == "lsb-first":
            message, _ = extract_message_lsb_first(bits)
        else:
            message, _ = extract_message_msb_first(bits)

        if message:
            return message, None

        message, _ = extract_message_best_effort(bits, bit_order=bo)
        if message:
            return message, None

    return None, None


def extract_exif(image_path, bit_order=None):
    """
    Extract data hidden in EXIF metadata.
    """
    img = Image.open(image_path)
    message = None
    exif_dict = None

    if piexif:
        try:
            if "exif" in img.info:
                exif_dict = piexif.load(img.info["exif"])
            else:
                exif_dict = piexif.load(image_path)
        except Exception:
            pass

    if exif_dict:
        try:
            user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
            if user_comment and isinstance(user_comment, bytes):
                if user_comment.startswith(b"ASCII\x00\x00\x00"):
                    message = user_comment[8:].decode("ascii", errors="ignore").strip()
                elif user_comment.startswith(b"UNICODE\x00"):
                    message = user_comment[8:].decode("utf-16", errors="ignore").strip()
                elif user_comment.startswith(b"JIS\x00\x00\x00\x00\x00"):
                    message = (
                        user_comment[8:].decode("shift_jis", errors="ignore").strip()
                    )
                else:
                    try:
                        message = user_comment.decode("utf-8", errors="ignore").strip()
                    except:
                        message = user_comment.hex()

                if message:
                    return message, None

            for ifd_name, tag_id in [
                ("0th", piexif.ImageIFD.ImageDescription),
                ("0th", piexif.ImageIFD.Software),
            ]:
                value = exif_dict.get(ifd_name, {}).get(tag_id)
                if value:
                    if isinstance(value, bytes):
                        message = value.decode("utf-8", errors="ignore").strip()
                    elif isinstance(value, str):
                        message = value.strip()
                    if message:
                        return message, None
        except Exception:
            pass

    return message, None


def extract_eoi(image_path, bit_order=None):
    """
    Extract data hidden after image end marker.
    """
    try:
        img = Image.open(image_path)
        with open(image_path, "rb") as f:
            data = f.read()

        if data.startswith(b"\xff\xd8"):  # JPEG
            end_pos = data.rfind(b"\xff\xd9")
            if end_pos >= 0:
                payload = data[end_pos + 2 :]
            else:
                return None, None
        elif data.startswith(b"\x89PNG"):  # PNG
            end_pos = data.rfind(b"IEND")
            if end_pos >= 0:
                payload = data[end_pos + 12 :]
            else:
                return None, None
        else:
            return None, None

        if len(payload) < 4:
            return None, None

        if payload.startswith(b"AI42"):
            payload = payload[4:]
        terminator_pos = payload.find(b"\x00")
        if terminator_pos != -1:
            payload = payload[:terminator_pos]

        try:
            message = payload.decode("utf-8")
            return message, None
        except UnicodeDecodeError:
            return payload.hex(), None
    except Exception:
        return None, None


# Export extraction functions
extraction_functions = {
    "alpha": extract_alpha,
    "palette": extract_palette,
    "lsb": extract_lsb,
    "exif": extract_exif,
    "eoi": extract_eoi,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract steganography from images")
    parser.add_argument("image", help="Image file to extract from")
    parser.add_argument(
        "--method",
        choices=["alpha", "palette", "lsb", "exif", "eoi", "auto"],
        default="auto",
        help="Extraction method",
    )
    parser.add_argument(
        "--bit-order",
        choices=["lsb-first", "msb-first", "none"],
        default="none",
        help="Bit order hint",
    )
    args = parser.parse_args()

    if args.method == "auto":
        for method_name, extractor in extraction_functions.items():
            message, _ = extractor(args.image, bit_order=args.bit_order)
            if message:
                print(f"[{method_name.upper()}] Found message:")
                print(message)
                break
        else:
            print("No steganography detected")
    else:
        extractor = extraction_functions[args.method]
        message, _ = extractor(args.image, bit_order=args.bit_order)
        if message:
            print(f"[{args.method.upper()}] Extracted message:")
            print(message)
        else:
            print(f"No message found using {args.method} method")
