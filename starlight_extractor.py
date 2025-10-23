#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import argparse
import math
from scipy.stats import entropy
from scipy.fftpack import dct
from PIL.ExifTags import TAGS

# Import the two-stage model and the corresponding validation transform from trainer.py
from trainer import StarlightTwoStage, transform_val as transform_rgba_val 

try:
    import piexif
except ImportError:
    piexif = None
    print("Warning: piexif not installed; EXIF extraction may be limited.")

def extract_features(path, img):
    """
    Extracts 15 statistical/structural features from the image file.
    This function is updated to match the 15 features in trainer.py.
    """
    width, height = img.size
    area = width * height if width * height > 0 else 1.0
    
    file_size = os.path.getsize(path) / 1024.0
    file_size_norm = file_size / area
    
    exif_bytes = img.info.get('exif')
    exif_present = 1.0 if exif_bytes else 0.0
    exif_length = len(exif_bytes) if exif_bytes else 0.0
    exif_length_norm = min(exif_length / area, 1.0)
    
    comment_length = 0.0
    exif_entropy = 0.0
    if exif_bytes:
        try:
            exif_dict = img.getexif()
            tag_values = []
            for tag_id, value in exif_dict.items():
                tag = TAGS.get(tag_id, tag_id)
                # Feature: UserComment length
                if tag == 'UserComment' and isinstance(value, bytes):
                    comment_length = min(len(value) / area, 1.0)
                if isinstance(value, (bytes, str)):
                    tag_values.append(value if isinstance(value, bytes) else value.encode('utf-8'))
            # Feature: EXIF entropy
            if tag_values:
                lengths = [len(v) for v in tag_values]
                max_len = max(lengths or [1])
                hist = np.histogram(lengths, bins=10, range=(0, max_len))[0]
                exif_entropy = entropy(hist + 1e-10) / area if any(hist) else 0.0
        except:
            comment_length = 0.0
            exif_entropy = 0.0
    
    # Features: Palette presence, length, and entropy
    palette_present = 1.0 if img.mode == 'P' else 0.0
    palette = img.getpalette()
    palette_length = len(palette) / 3 if palette else 0.0
    if palette_present:
        hist = img.histogram()
        palette_entropy_value = entropy([h + 1 for h in hist if h > 0]) if any(hist) else 0.0
    else:
        palette_entropy_value = 0.0
    
    # Feature: EOF length (for JPEG EOI)
    with open(path, 'rb') as f:
        data = f.read()
    if img.format == 'JPEG':
        eoi_pos = data.rfind(b'\xff\xd9')
        eof_length = len(data) - (eoi_pos + 2) if eoi_pos >= 0 else 0.0
    else:
        eof_length = 0.0
    eof_length_norm = min(eof_length / area, 1.0)
    
    # Features: Alpha channel metrics
    has_alpha = 1.0 if img.mode in ('RGBA', 'LA', 'PA') else 0.0
    alpha_variance = 0.0
    alpha_mean = 0.5 # Default to 0.5 for non-RGBA
    alpha_unique_ratio = 0.0
    
    if has_alpha and img.mode == 'RGBA':
        img_array = np.array(img)
        if img_array.shape[-1] == 4:
            alpha_channel = img_array[:, :, 3].astype(float)
            alpha_variance = np.var(alpha_channel) / 65025.0 # Max variance is 255^2 = 65025
            alpha_mean = np.mean(alpha_channel) / 255.0
            unique_alphas = len(np.unique(alpha_channel))
            total_pixels = alpha_channel.size
            alpha_unique_ratio = unique_alphas / min(total_pixels, 256)
    
    # Features: File format indicators
    is_jpeg = 1.0 if img.format == 'JPEG' else 0.0
    is_png = 1.0 if img.format == 'PNG' else 0.0
    
    return np.array([
        file_size_norm, exif_present, exif_length_norm, comment_length, 
        exif_entropy, palette_present, palette_length, palette_entropy_value, 
        eof_length_norm, has_alpha, alpha_variance, alpha_mean, alpha_unique_ratio, 
        is_jpeg, is_png # Total 15 features
    ], dtype=np.float32)

def extract_message_from_bits(bits, max_length=24576):
    """Extract message from bit string with multiple format support."""
    hint_bits = ''.join(format(byte, '08b') for byte in b'0xAI42')
    if bits.startswith(hint_bits):
        bits_after_hint = bits[len(hint_bits):]
        terminator_bits = '00000000'
        terminator_index = bits_after_hint.find(terminator_bits)
        
        if terminator_index != -1:
            if terminator_index % 8 != 0:
                terminator_index = (terminator_index // 8) * 8
            
            payload_bits = bits_after_hint[:terminator_index]
            num_bytes = len(payload_bits) // 8
            
            if len(payload_bits) % 8 != 0:
                payload_bits = payload_bits[:num_bytes * 8]
            
            if len(payload_bits) == 0:
                return None, bits_after_hint
            
            bytes_data = [int(payload_bits[j:j+8], 2) for j in range(0, len(payload_bits), 8)]
            try:
                message = bytes(bytes_data).decode('utf-8')
                return message.strip(), bits_after_hint[terminator_index + 8:]
            except UnicodeDecodeError:
                return bytes(bytes_data).hex(), bits_after_hint[terminator_index + 8:]
        else:
            max_bits = min(len(bits_after_hint), max_length * 8)
            max_bits = (max_bits // 8) * 8
            bytes_data = [int(bits_after_hint[j:j+8], 2) for j in range(0, max_bits, 8)]
            return None, bits_after_hint
    
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

def extract_lsb(image_path, channel='all', max_bits=1000000):
    """Extract LSB steganography data."""
    img = Image.open(image_path)
    
    if img.mode == 'RGBA' or 'png' in image_path.lower():
        bits = extract_bits_by_strategy(img, 'rgba_flat', max_bits)
        message, remaining = extract_message_from_bits(bits)
        if message and len(message) > 10:
            return message, remaining

    # Use the robust strategy extraction for auto-detection
    return extract_lsb_strategy(image_path, strategy='auto', channel=channel, max_bits=max_bits)

def extract_alpha(image_path):
    return extract_lsb_strategy(image_path, strategy='alpha')

def extract_palette(image_path):
    img = Image.open(image_path)
    if img.mode != 'P':
        return None, None
    img_array = np.array(img)
    bits = ''.join(str(pixel & 1) for pixel in img_array.flatten())
    return extract_message_from_bits(bits)

def extract_dct(image_path, clean_path=None):
    img = np.array(Image.open(image_path).convert('YCbCr'))[:, :, 0].astype(float)
    dct_coeffs = dct(dct(img.T, norm='ortho').T, norm='ortho')
    if clean_path and os.path.exists(clean_path):
        clean_img = np.array(Image.open(clean_path).convert('YCbCr'))[:, :, 0].astype(float)
        clean_dct = dct(dct(clean_img.T, norm='ortho').T, norm='ortho')
        diffs = dct_coeffs - clean_dct
        bits = ''
        block_size = 8
        height, width = img.shape
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                if y + block_size <= height and x + block_size <= width:
                    mid_y, mid_x = block_size // 2, block_size // 2
                    current_val = clean_img[y:y+block_size, x:x+block_size][mid_y, mid_x]
                    if 15 <= current_val <= 240:
                        diff = diffs[y + mid_y, x + mid_x]
                        if abs(diff) > 10:
                            bits += '1' if diff > 0 else '0'
    else:
        coeffs_flat = dct_coeffs.flatten()[1:]
        bits = ''.join(bin(int(coeff))[-1] for coeff in coeffs_flat if abs(coeff) > 10)
    return extract_message_from_bits(bits)

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
    with open(image_path, 'rb') as f:
        data = f.read()
    eoi_pos = data.rfind(b'\xff\xd9')
    if eoi_pos >= 0:
        payload = data[eoi_pos + 2:]
        if payload.startswith(b'0xAI42'):
            payload = payload[6:]
            terminator_pos = payload.find(b'\x00')
            if terminator_pos != -1:
                payload = payload[:terminator_pos]
        try:
            message = payload.decode('utf-8')
            return message, None
        except UnicodeDecodeError:
            return payload.hex(), None
    return None, None

extraction_functions = {
    'alpha': extract_alpha,
    'palette': extract_palette,
    'dct': extract_dct,
    'lsb': extract_lsb,
    'exif': extract_exif,
    'eoi': extract_eoi
}

def predict(model, device, image_path, class_map):
    """
    Predicts the steganography type using the StarlightTwoStage model.
    """
    model.eval()
    
    # Open image and convert to RGBA for feature extraction and new model input
    img = Image.open(image_path).convert('RGBA')
    
    # Extract the 15 features
    features = extract_features(image_path, img)
    
    # Apply the RGBA transform
    img_tensor = transform_rgba_val(img).unsqueeze(0).to(device)
    features_tensor = torch.tensor(features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # The StarlightTwoStage model returns: (all_logits, normality_score, stego_type_logits)
        all_logits, normality_score, stego_type_logits = model(img_tensor, features_tensor)
        
        # We use all_logits (7 classes: clean + 6 stego) for the final prediction
        outputs = all_logits 
        
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
        pred_label = class_map.get(pred_class, 'unknown')
        confidence = probs[pred_class]
        
    return pred_label, confidence, probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steganography Extractor using Starlight Model")
    parser.add_argument('image_paths', nargs='+', help="Paths to the image files to analyze")
    parser.add_argument('--model_path', default='best_model.pth', help="Path to the trained model file")
    parser.add_argument('--channel', default='all', choices=['all', 'red', 'green', 'blue'], help="Channel for LSB extraction")
    parser.add_argument('--clean_path', default=None, help="Path to the corresponding clean image for DCT extraction")
    # Added new parameter for explicit algorithm choice
    parser.add_argument('--extract_algo', default=None, choices=list(extraction_functions.keys()), help="Explicitly specify the stego algorithm to attempt extraction for, overriding the model's prediction.")
    args = parser.parse_args()

    class_map = {0: 'clean', 1: 'alpha', 2: 'palette', 3: 'dct', 4: 'lsb', 5: 'eoi', 6: 'exif'}

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Instantiate the new StarlightTwoStage model. num_stego_classes=6 and feature_dim=15.
    model = StarlightTwoStage(num_stego_classes=6, feature_dim=15)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"Error: Failed to load model from {args.model_path}: {str(e)}")
        exit(1)
    model.to(device)

    for image_path in args.image_paths:
        if not os.path.exists(image_path):
            print(f"Error: File not found - {image_path}")
            continue
        
        clean_path = args.clean_path
        if clean_path is None:
            # Attempt to find a corresponding clean image automatically
            clean_path = image_path.replace('/stego/', '/clean/')
            if not os.path.exists(clean_path):
                clean_path = None
        
        pred_label, confidence, probs = predict(model, device, image_path, class_map)
        
        # Determine the extraction algorithm to use: explicitly specified or predicted
        algo_to_try = args.extract_algo if args.extract_algo else pred_label

        print(f"\nAnalysis for {image_path}:")
        
        # Print prediction results
        if pred_label == 'clean':
            print("  Predicted: Clean (no steganography detected)")
            print(f"  Confidence: {confidence:.4f}")
        else:
            print(f"  Predicted Stego Algorithm: {pred_label}")
            print(f"  Confidence: {confidence:.4f}")
        
        print("  All Probabilities:")
        for idx, label in class_map.items():
            print(f"    {label}: {probs[idx]:.4f}")

        # Only attempt extraction if the determined algorithm is a stego algorithm
        if algo_to_try != 'clean' and algo_to_try in extraction_functions:
            extractor = extraction_functions.get(algo_to_try)
            
            # Print which algorithm is being used
            if args.extract_algo:
                print(f"  Attempting extraction with explicitly specified algorithm: {algo_to_try}")
            else:
                print(f"  Attempting extraction with predicted algorithm: {algo_to_try}")
                
            if extractor:
                # Call the specific extractor function
                if algo_to_try == 'lsb':
                    message, bits = extractor(image_path, channel=args.channel)
                elif algo_to_try == 'dct':
                    message, bits = extractor(image_path, clean_path=clean_path)
                else:
                    message, bits = extractor(image_path)
                
                if message:
                    print("  Extracted Message:")
                    print(f"    {message}")
                else:
                    print("  No message extracted or extraction failed.")
            else:
                print(f"  Extraction not implemented for algorithm: {algo_to_try}.")
        elif algo_to_try != 'clean' and algo_to_try not in extraction_functions:
             print(f"  Error: Explicitly specified algorithm '{algo_to_try}' is not a valid extraction type.")
        else:
            # Skip extraction for 'clean' prediction
            pass
