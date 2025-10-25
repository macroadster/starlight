#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse

# Import the two-stage model and features from starlight_model.py
from starlight_model import transform_val as transform_rgba_val, extract_features

try:
    import piexif
except ImportError:
    piexif = None
    print("Warning: piexif not installed; EXIF extraction may be limited.")

def extract_message_from_bits(bits, max_length=24576):
    """
    Extract message from bit string with multiple format support.
    
    FIXED: The hint detection block now correctly handles the LSB-first 
    (byte-reversed) bit order used by the 'alpha' embedding algorithm.
    """
    # ----------------------------------------------------------------------
    # --- Strategy 1: AI Hint (for LSB-first Alpha/LSB embedding) ---
    # ----------------------------------------------------------------------
    ai_hint_bytes = b'AI42'
    hint_bits = ''.join(format(byte, '08b') for byte in ai_hint_bytes)
    
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
        else:
            max_bits = min(len(bits_after_hint), max_length * 8)
            max_bits = (max_bits // 8) * 8
            return None, bits_after_hint

    # ----------------------------------------------------------------------
    # --- Strategy 2: Length Prefix (Unchanged, uses MSB-first decoding) ---
    # ----------------------------------------------------------------------
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
    
    # ----------------------------------------------------------------------
    # --- Strategy 3: Null-Terminated ASCII/UTF-8 (Unchanged) ---
    # ----------------------------------------------------------------------
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

def extract_bits_by_strategy(img, strategy, max_bits):
    """Extract bits using a specific strategy."""
    bits = ''
    
    if strategy == 'rgba_flat':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        data = np.array(img)
        color_channels = data.flatten()
        for i, val in enumerate(color_channels):
            if i >= max_bits:
                break
            bits += str(val & 1)
    
    elif strategy == 'rgb_flat':
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        data = np.array(img)
        if len(data.shape) == 3 and data.shape[2] >= 3:
            rgb_data = data[:, :, :3].flatten()
            for i, val in enumerate(rgb_data):
                if i >= max_bits:
                    break
                bits += str(val & 1)
    
    elif strategy == 'rgb_interleaved':
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
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
    
    elif strategy in ['red', 'green', 'blue']:
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        data = np.array(img)
        channel_idx = {'red': 0, 'green': 1, 'blue': 2}[strategy]
        if len(data.shape) == 3 and data.shape[2] > channel_idx:
            channel_data = data[:, :, channel_idx].flatten()
            for i, val in enumerate(channel_data):
                if i >= max_bits:
                    break
                bits += str(val & 1)

    elif strategy == 'alpha':
        if img.mode != 'RGBA':
            return ''
        data = np.array(img)
        alpha_data = data[:, :, 3].flatten()
        for i, val in enumerate(alpha_data):
           if i >= max_bits:
                break
           bits += str(val & 1)
 
    return bits

def calculate_message_confidence(message):
    """Calculate confidence score for extracted message."""
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
    
    return max(0.0, min(100.0, score))

def extract_lsb_strategy(image_path, strategy='auto', channel='all', max_bits=1000000):
    """Extract LSB data using different strategies."""
    img = Image.open(image_path)
    
    strategies_to_try = []
    
    if strategy == 'auto':
        strategies_to_try = ['rgba_flat', 'rgb_flat', 'rgb_interleaved', 'red', 'green', 'blue', 'alpha']
    elif strategy == 'channel_specific':
        strategies_to_try = [channel]
    else:
        strategies_to_try = [strategy]
    
    results = []
    
    for strat in strategies_to_try:
        bits = extract_bits_by_strategy(img, strat, max_bits)
        if bits:
            message, remaining = extract_message_from_bits(bits)
            if message:
                results.append({
                    'strategy': strat,
                    'message': message,
                    'bits_used': len(bits) - len(remaining) if remaining else len(bits),
                    'confidence': calculate_message_confidence(message)
                })
    
    if not results:
        return None, None
    
    best = max(results, key=lambda x: x['confidence'])
    return best['message'], None

def reconstruct_lsb_first_message_from_bits(bits, max_length=24576):
    """
    Reconstructs bytes from a bit stream where bits are embedded LSB-first.
    This logic matches the verification in data_generator.py (Hint + Payload + 0x00).
    """
    ai_hint_bytes = b'AI42'
    
    hint_bits = ''
    for byte in ai_hint_bytes:
        lsb_first_bits = bin(byte)[2:].zfill(8)[::-1]
        hint_bits += lsb_first_bits
    
    if not bits.startswith(hint_bits):
        print(f"  [DEBUG] Expected Hint Bits (LSB-first): {hint_bits[:32]}")
        print(f"  [DEBUG] Found Bits: {bits[:32]}")
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

def extract_lsb(image_path, channel='all', max_bits=1000000):
    """Extract LSB steganography data."""
    img = Image.open(image_path)
    
    # Priority 1: Check for the LSB-first Alpha/RGBA format
    if img.mode == 'RGBA' or 'png' in image_path.lower():
        bits = extract_bits_by_strategy(img, 'rgba_flat', max_bits)
        message, remaining = reconstruct_lsb_first_message_from_bits(bits)
        
        if message and len(message) > 10:
            print(f"  [DEBUG] LSB extraction found LSB-first format. Message length: {len(message)}")
            return message, remaining

    # Priority 2: Use the robust LSB strategy extraction
    return extract_lsb_strategy(image_path, strategy='auto', channel=channel, max_bits=max_bits)

def extract_alpha(image_path):
    img = Image.open(image_path)
    bits = extract_bits_by_strategy(img, 'alpha', 1000000)
    
    if not bits:
        return None, None

    # Strategy A: Try LSB-First (AI42-Hinted)
    message, remaining = reconstruct_lsb_first_message_from_bits(bits)
    
    if message:
        print(f"  [DEBUG] Alpha extraction successful (LSB-First/AI42). Message length: {len(message)}")
        return message, remaining
    
    print(f"  [DEBUG] LSB-First/AI42 extraction failed. Attempting general decoders...")
    
    # Strategy B: Try General MSB-First Decoders
    message, remaining = extract_message_from_bits(bits)
    
    if message:
        print(f"  [DEBUG] Alpha extraction successful (General MSB-First). Message length: {len(message)}")
        return message, remaining

    print(f"  [DEBUG] Alpha extraction failed. Bits extracted: {len(bits)}. No message found.")
    return None, None

def extract_palette(image_path):
    img = Image.open(image_path)
    if img.mode != 'P':
        return None, None
    img_array = np.array(img)
    bits = ''.join(str(pixel & 1) for pixel in img_array.flatten())
    return extract_message_from_bits(bits)

def extract_sdm(image_path, clean_path=None):
    """
    Extract data from PNG SDM embedding.
    """
    stego_img = Image.open(image_path).convert('RGB')
    stego_array = np.array(stego_img, dtype=np.float32)
    
    block_size = 8
    height, width = stego_array.shape[:2]
    
    # MODE 1: With clean image (most accurate)
    if clean_path and os.path.exists(clean_path):
        print(f"  [DEBUG] SDM extraction using clean image comparison")
        orig_img = Image.open(clean_path).convert('RGB')
        orig_array = np.array(orig_img, dtype=np.float32)
        
        usable_blocks = []
        
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                if y + block_size <= height and x + block_size <= width:
                    mid_y, mid_x = block_size // 2, block_size // 2
                    orig_val = orig_array[y + mid_y, x + mid_x, 0]
                    
                    if 15 <= orig_val <= 240:
                        usable_blocks.append((y, x))
        
        bits = []
        for y, x in usable_blocks:
            mid_y, mid_x = block_size // 2, block_size // 2
            stego_val = stego_array[y + mid_y, x + mid_x, 0]
            orig_val = orig_array[y + mid_y, x + mid_x, 0]
            diff = stego_val - orig_val
            
            if diff > 7.5:
                bits.append('1')
            elif diff < -7.5:
                bits.append('0')
            else:
                bits.append('0')
    
    # MODE 2: Without clean image (pattern detection)
    else:
        print(f"  [DEBUG] SDM extraction without clean image - using pattern detection")
        
        block_centers = []
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                if y + block_size <= height and x + block_size <= width:
                    mid_y, mid_x = block_size // 2, block_size // 2
                    center_val = stego_array[y + mid_y, x + mid_x, 0]
                    
                    if 0 <= center_val <= 255:
                        block_centers.append((y, x, center_val))
        
        if len(block_centers) < 100:
            print(f"  [DEBUG] SDM extraction failed: too few blocks ({len(block_centers)})")
            return None, None
        
        bits = []
        for y, x, center_val in block_centers:
            mid_y, mid_x = block_size // 2, block_size // 2
            
            block = stego_array[y:y+block_size, x:x+block_size, 0]
            
            surrounding_pixels = []
            for dy in range(block_size):
                for dx in range(block_size):
                    if dy != mid_y or dx != mid_x:
                        surrounding_pixels.append(block[dy, dx])
            
            local_avg = np.mean(surrounding_pixels)
            diff_from_local = center_val - local_avg
            
            if diff_from_local > 10:
                bits.append('1')
            elif diff_from_local < -10:
                bits.append('0')
            else:
                if center_val < 30:
                    bits.append('0')
                elif center_val > 225:
                    bits.append('1')
                else:
                    bits.append('0')
    
    if len(bits) < 32:
        print(f"  [DEBUG] SDM extraction failed: only {len(bits)} bits extracted (need at least 32)")
        return None, None
    
    bits_str = ''.join(bits)
    
    try:
        length = int(bits_str[:32], 2)
        
        if length > 100000 or length <= 0:
            print(f"  [DEBUG] SDM extraction failed: invalid length {length}")
            return None, None
        
        total_bits_needed = 32 + length * 8
        
        if len(bits_str) < total_bits_needed:
            print(f"  [DEBUG] SDM extraction failed: need {total_bits_needed} bits, have {len(bits_str)}")
            return None, None
        
        payload_bits = bits_str[32:total_bits_needed]
        payload_bytes = bytes([int(payload_bits[i:i+8], 2) 
                               for i in range(0, len(payload_bits), 8)])
        
        try:
            message = payload_bytes.decode('utf-8')
            print(f"  [DEBUG] SDM extraction successful. Message length: {len(message)} chars")
            return message, None
        except UnicodeDecodeError:
            return payload_bytes.hex(), None
            
    except Exception as e:
        print(f"  [DEBUG] SDM extraction error: {e}")
        return None, None

def extract_exif(image_path):
    img = Image.open(image_path)
    message = None
    raw_bytes = None
    if piexif:
        try:
            exif_dict = piexif.load(image_path)
            user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
            if user_comment:
                if isinstance(user_comment, bytes):
                    raw_bytes = user_comment
                    if user_comment.startswith(b'ASCII\x00\x00\x00'):
                        message = user_comment[8:].decode('ascii', errors='ignore').strip()
                    else:
                        message = user_comment.decode('utf-8', errors='ignore').strip()
                if message:
                    return message, None
        except Exception:
            pass
    exif = img.getexif()
    if exif:
        tags = [270, 306, 315, 36867, 37510]
        for tag in tags:
            value = exif.get(tag)
            if value:
                if tag == 37510 and isinstance(value, bytes):
                    raw_bytes = value
                    try:
                        if value.startswith(b'ASCII\x00\x00\x00'):
                            message = value[8:].decode('ascii', errors='ignore').strip()
                        else:
                            message = value.decode('utf-8', errors='ignore').strip()
                    except UnicodeDecodeError:
                        message = value.hex()
                elif isinstance(value, str) and value.strip():
                    message = value
                if message:
                    break
    if not message and raw_bytes:
        message = raw_bytes.hex()
    return message, None

def extract_eoi(image_path):
    """
    Extract data hidden after JPEG End-of-Image (EOI) marker.
    CRITICAL FIX: Only processes JPEG files and filters legitimate metadata.
    """
    try:
        img = Image.open(image_path)
        if img.format != 'JPEG':
            return None, None
    except Exception:
        return None, None
    
    with open(image_path, 'rb') as f:
        data = f.read()
    
    if not data.startswith(b'\xff\xd8'):
        return None, None
    
    eoi_pos = data.rfind(b'\xff\xd9')
    
    if eoi_pos < 0:
        return None, None
    
    if eoi_pos + 2 >= len(data):
        return None, None
    
    payload = data[eoi_pos + 2:]
    
    # Additional validation: Skip legitimate JPEG metadata
    if len(payload) < 4:
        return None, None
    
    # Skip known legitimate formats
    if (payload.startswith(b'\xff\xd8') or      # JPEG thumbnail
        payload.startswith(b'Exif') or           # EXIF data
        payload.startswith(b'ICC_PROFILE') or    # ICC profile
        payload.startswith(b'<?xpacket') or      # XMP metadata
        payload.startswith(b'http://ns.adobe.com')):  # Adobe metadata
        print(f"  [DEBUG] EOI: Skipping legitimate metadata ({payload[:20]})")
        return None, None
    
    # Skip if it's just padding (all zeros or all 0xFF)
    if all(b == 0 for b in payload[:min(20, len(payload))]):
        print(f"  [DEBUG] EOI: Skipping null padding")
        return None, None
    
    if all(b == 0xFF for b in payload[:min(20, len(payload))]):
        print(f"  [DEBUG] EOI: Skipping 0xFF padding")
        return None, None
    
    # Check for the AI42 hint marker (our stego format)
    if payload.startswith(b'0xAI42'):
        payload = payload[6:]
        terminator_pos = payload.find(b'\x00')
        if terminator_pos != -1:
            payload = payload[:terminator_pos]
    elif payload.startswith(b'AI42'):
        payload = payload[4:]
        terminator_pos = payload.find(b'\x00')
        if terminator_pos != -1:
            payload = payload[:terminator_pos]
    else:
        # No marker found - be cautious
        # Check if payload looks like text vs binary
        try:
            # Try to decode as text
            test_decode = payload.decode('utf-8', errors='strict')
            # If it decodes cleanly and is mostly printable, might be stego
            printable_ratio = sum(c.isprintable() or c in '\n\r\t' for c in test_decode) / len(test_decode)
            if printable_ratio < 0.8:
                print(f"  [DEBUG] EOI: Low printable ratio ({printable_ratio:.2f}), likely not text stego")
                return None, None
        except UnicodeDecodeError:
            # Binary data without marker - probably not our stego
            print(f"  [DEBUG] EOI: Binary data without AI42 marker, skipping")
            return None, None
    
    # Try to decode as UTF-8
    try:
        message = payload.decode('utf-8')
        # Additional validation: message should be mostly printable
        printable_ratio = sum(c.isprintable() or c in '\n\r\t' for c in message) / len(message)
        if printable_ratio > 0.8:
            return message, None
        else:
            return payload.hex(), None
    except UnicodeDecodeError:
        return payload.hex(), None

extraction_functions = {
    'alpha': extract_alpha,
    'palette': extract_palette,
    'sdm': extract_sdm,
    'lsb': extract_lsb,
    'exif': extract_exif,
    'eoi': extract_eoi
}

