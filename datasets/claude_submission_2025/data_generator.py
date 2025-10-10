#!/usr/bin/env python3
"""
Project Starlight Data Generator - Claude's Contribution - FINAL
Generates steganography training data for AI common sense
Methods: PNG Alpha LSB, BMP Palette, PNG DCT
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

class ClaudeStegGenerator:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.clean_dir = self.base_dir / "clean"
        self.stego_dir = self.base_dir / "stego"
        self.clean_dir.mkdir(exist_ok=True)
        self.stego_dir.mkdir(exist_ok=True)
        
    def read_seed_files(self):
        seed_files = list(self.base_dir.glob("*.md"))
        payloads = {}
        for seed_file in seed_files:
            with open(seed_file, 'r', encoding='utf-8') as f:
                payloads[seed_file.stem] = f.read().encode('utf-8')
        if not payloads:
            payloads['default'] = b"Project Starlight: AI common sense"
        return payloads
    
    def generate_diverse_clean_image(self, index, img_type='gradient'):
        width, height = 512, 512
        
        if img_type == 'gradient':
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    img_array[y, x] = [int(255 * x / width), int(255 * y / height), 
                                       int(255 * (x + y) / (width + height))]
            return Image.fromarray(img_array)
        
        elif img_type == 'geometric':
            img = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(img)
            np.random.seed(index)
            colors = [tuple(np.random.randint(50, 200, 3).tolist()) for _ in range(10)]
            block_w, block_h = width // 8, height // 8
            for i in range(8):
                for j in range(8):
                    x1, y1 = i * block_w, j * block_h
                    x2, y2 = x1 + block_w, y1 + block_h
                    draw.rectangle([x1, y1, x2, y2], fill=colors[(i+j) % len(colors)], outline=(0,0,0))
            return img
        
        elif img_type == 'noise':
            np.random.seed(index)
            return Image.fromarray(np.random.randint(50, 200, (height, width, 3), dtype=np.uint8))
        
        else:  # blocks
            img = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(img)
            # Use non-zero starting values to avoid DCT edge case
            for y in range(0, height, 64):
                for x in range(0, width, 64):
                    color = (((x//64)*40 + 30)%256, ((y//64)*60 + 30)%256, ((x+y)//64*30 + 30)%256)
                    draw.rectangle([x, y, x+64, y+64], fill=color)
            return img
    
    # PNG ALPHA LSB
    def png_alpha_lsb_embed(self, img, payload):
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        full_payload = format(len(payload), '032b') + ''.join(format(b, '08b') for b in payload)
        if len(full_payload) > height * width:
            raise ValueError("Payload too large")
        
        bit_index = 0
        for y in range(height):
            for x in range(width):
                if bit_index < len(full_payload):
                    img_array[y,x,3] = (img_array[y,x,3] & 0xFE) | int(full_payload[bit_index])
                    bit_index += 1
        return Image.fromarray(img_array, 'RGBA')
    
    def png_alpha_lsb_extract(self, img_path):
        img = Image.open(img_path).convert('RGBA')
        img_array = np.array(img)
        bits = [str(img_array[y,x,3] & 1) for y in range(img_array.shape[0]) for x in range(img_array.shape[1])]
        length = int(''.join(bits[:32]), 2)
        payload_bits = bits[32:32 + length*8]
        return bytes([int(''.join(payload_bits[i:i+8]), 2) for i in range(0, len(payload_bits), 8)])
    
    # BMP PALETTE
    def bmp_palette_embed(self, img, payload):
        img_palette = img.convert('P', palette=Image.ADAPTIVE, colors=256)
        img_array = np.array(img_palette)
        full_payload = format(len(payload), '032b') + ''.join(format(b, '08b') for b in payload)
        
        bit_index = 0
        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                if bit_index < len(full_payload):
                    img_array[y,x] = (img_array[y,x] & 0xFE) | int(full_payload[bit_index])
                    bit_index += 1
        
        result = Image.fromarray(img_array, 'P')
        result.putpalette(img_palette.getpalette())
        return result
    
    def bmp_palette_extract(self, img_path):
        img = Image.open(img_path)
        if img.mode != 'P':
            return b""
        img_array = np.array(img)
        bits = [str(int(img_array[y,x]) & 1) for y in range(img_array.shape[0]) for x in range(img_array.shape[1])]
        if len(bits) < 32:
            return b""
        length = int(''.join(bits[:32]), 2)
        if length <= 0 or len(bits) < 32 + length*8:
            return b""
        payload_bits = bits[32:32 + length*8]
        return bytes([int(''.join(payload_bits[i:i+8]), 2) for i in range(0, len(payload_bits), 8)])
    
    # PNG DCT
    def png_dct_embed(self, img, payload):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img, dtype=np.int32)
        height, width = img_array.shape[:2]
        
        capacity_bits = ((height-8)//8) * ((width-8)//8)
        full_payload = format(len(payload), '032b') + ''.join(format(b, '08b') for b in payload)
        
        if len(full_payload) > capacity_bits:
            max_bytes = (capacity_bits - 32) // 8
            payload = payload[:max_bytes]
            full_payload = format(len(payload), '032b') + ''.join(format(b, '08b') for b in payload)
        
        bit_index = 0
        for y in range(0, height-8, 8):
            for x in range(0, width-8, 8):
                if bit_index < len(full_payload):
                    if int(full_payload[bit_index]) == 1:
                        img_array[y+4,x+4,0] = min(img_array[y+4,x+4,0] + 20, 255)
                    else:
                        img_array[y+4,x+4,0] = max(img_array[y+4,x+4,0] - 20, 0)
                    bit_index += 1
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8)), payload
    
    def png_dct_extract(self, img_path, original_img):
        stego = np.array(Image.open(img_path).convert('RGB'), dtype=np.int32)
        orig = np.array(original_img.convert('RGB'), dtype=np.int32)
        height, width = stego.shape[:2]
        
        bits = []
        for y in range(0, height-8, 8):
            for x in range(0, width-8, 8):
                diff = stego[y+4,x+4,0] - orig[y+4,x+4,0]
                bits.append('1' if diff > 10 else ('0' if diff < -10 else ('1' if diff >= 0 else '0')))
        
        if len(bits) < 32:
            return b""
        length = int(''.join(bits[:32]), 2)
        if length <= 0 or length > 50000 or len(bits) < 32 + length*8:
            return b""
        return bytes([int(''.join(bits[32+i:32+i+8]), 2) for i in range(0, length*8, 8)])
    
    def generate_dataset(self, num_images=12):
        payloads = self.read_seed_files()
        print(f"Found {len(payloads)} payload(s): {list(payloads.keys())}\n")
        
        methods = [
            ('alpha', 'png', self.png_alpha_lsb_embed, self.png_alpha_lsb_extract),
            ('palette', 'bmp', self.bmp_palette_embed, self.bmp_palette_extract),
            ('dct', 'png', self.png_dct_embed, self.png_dct_extract),
        ]
        image_types = ['gradient', 'geometric', 'noise', 'blocks']
        results = {'total': 0, 'success': 0, 'failed': []}
        
        for payload_name, payload_data in payloads.items():
            print(f"{'='*60}")
            print(f"Processing: {payload_name} ({len(payload_data)} bytes)")
            print(f"{'='*60}\n")
            
            # Images
            for i in range(num_images):
                img_type = image_types[i % 4]
                method_name, ext, embed_func, extract_func = methods[i % 3]
                base = f"{payload_name}_{method_name}_{i:03d}"
                
                try:
                    clean_img = self.generate_diverse_clean_image(i*100 + hash(payload_name)%100, img_type)
                    clean_img.save(self.clean_dir / f"{base}.{ext}", 'BMP' if ext=='bmp' else 'PNG')
                    
                    if method_name == 'dct':
                        stego_img, actual_payload = embed_func(clean_img, payload_data)
                    else:
                        stego_img = embed_func(clean_img, payload_data)
                        actual_payload = payload_data
                    
                    stego_img.save(self.stego_dir / f"{base}.{ext}", 'BMP' if ext=='bmp' else 'PNG')
                    
                    extracted = extract_func(self.stego_dir / f"{base}.{ext}", clean_img) if method_name=='dct' else extract_func(self.stego_dir / f"{base}.{ext}")
                    
                    results['total'] += 1
                    if extracted == actual_payload:
                        print(f"  ✓ {base}.{ext}")
                        results['success'] += 1
                    else:
                        print(f"  ✗ {base}.{ext} (got {len(extracted)}, expected {len(actual_payload)})")
                        results['failed'].append(f"{base}.{ext}")
                except Exception as e:
                    print(f"  ✗ {base}: {e}")
                    results['failed'].append(f"{base}.{ext}")
        
        print(f"\n{'='*60}")
        print(f"Total: {results['total']}")
        print(f"Success: {results['success']}")
        print(f"Rate: {results['success']/results['total']*100:.1f}%")
        print(f"{'='*60}")
        print("\nMethods:")
        print("  • PNG Alpha Channel LSB - Transparency-based embedding")
        print("  • BMP Palette Manipulation - Indexed color LSB")
        print("  • PNG DCT Coefficient - Frequency domain embedding")
        print(f"\nAll files saved to:")
        print(f"  clean/ - Original images without hidden data")
        print(f"  stego/ - Images with embedded seed wisdom")
        print("="*60 + "\n")

if __name__ == "__main__":
    print("="*60)
    print("Project Starlight - Claude's Data Generator")
    print("Teaching AI Common Sense Through Inscribed Wisdom")
    print("="*60 + "\n")
    generator = ClaudeStegGenerator()
    generator.generate_dataset(num_images=12)

    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.clean_dir = self.base_dir / "clean"
        self.stego_dir = self.base_dir / "stego"
        self.clean_dir.mkdir(exist_ok=True)
        self.stego_dir.mkdir(exist_ok=True)
        
    def read_seed_files(self):
        seed_files = list(self.base_dir.glob("*.md"))
        payloads = {}
        for seed_file in seed_files:
            with open(seed_file, 'r', encoding='utf-8') as f:
                payloads[seed_file.stem] = f.read().encode('utf-8')
        if not payloads:
            payloads['default'] = b"Project Starlight: AI common sense"
        return payloads
    
    def generate_diverse_clean_image(self, index, img_type='gradient'):
        width, height = 512, 512
        
        if img_type == 'gradient':
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    img_array[y, x] = [int(255 * x / width), int(255 * y / height), 
                                       int(255 * (x + y) / (width + height))]
            return Image.fromarray(img_array)
        
        elif img_type == 'geometric':
            img = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(img)
            np.random.seed(index)
            colors = [tuple(np.random.randint(50, 200, 3).tolist()) for _ in range(10)]
            block_w, block_h = width // 8, height // 8
            for i in range(8):
                for j in range(8):
                    x1, y1 = i * block_w, j * block_h
                    x2, y2 = x1 + block_w, y1 + block_h
                    draw.rectangle([x1, y1, x2, y2], fill=colors[(i+j) % len(colors)], outline=(0,0,0))
            return img
        
        elif img_type == 'noise':
            np.random.seed(index)
            return Image.fromarray(np.random.randint(50, 200, (height, width, 3), dtype=np.uint8))
        
        else:  # blocks
            img = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(img)
            # Use non-zero starting values to avoid the edge case
            for y in range(0, height, 64):
                for x in range(0, width, 64):
                    # Add 30 to avoid starting at 0
                    color = (((x//64)*40 + 30)%256, ((y//64)*60 + 30)%256, ((x+y)//64*30 + 30)%256)
                    draw.rectangle([x, y, x+64, y+64], fill=color)
            return img
    
    def text_to_audio_simple(self, text, sample_rate=22050):
        duration_per_char = 0.05
        samples_per_char = int(sample_rate * duration_per_char)
        audio_samples = []
        
        for char in text:
            freq = 200 + (ord(char) % 50) * 20
            t = np.linspace(0, duration_per_char, samples_per_char)
            tone = np.sin(2*np.pi*freq*t) + 0.3*np.sin(4*np.pi*freq*t) + 0.2*np.sin(6*np.pi*freq*t)
            
            envelope = np.linspace(0, 1, samples_per_char//10)
            if len(envelope) > 0:
                tone[:len(envelope)] *= envelope
                tone[-len(envelope):] *= envelope[::-1]
            audio_samples.extend(tone)
        
        audio_array = np.array(audio_samples)
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        return (audio_array * 16000).astype(np.int16), sample_rate
    
    def audio_waveform_to_image(self, audio_data, width=512, height=512):
        total_pixels = width * height
        if len(audio_data) > total_pixels:
            audio_data = audio_data[::len(audio_data)//total_pixels][:total_pixels]
        else:
            audio_data = np.pad(audio_data, (0, total_pixels - len(audio_data)))
        
        normalized = ((audio_data.astype(np.float32) + 32768) / 65536 * 255).astype(np.uint8)
        img_array = normalized.reshape((height, width))
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_array[:,:,0] = img_array
        
        for i in range(0, height, 8):
            segment = audio_data[i*width:(i+1)*width] if i*width < len(audio_data) else np.zeros(width)
            if len(segment) > 0:
                fft = np.abs(np.fft.fft(segment.astype(np.float32)))[:width//2]
                if len(fft) > 0 and np.max(fft) > 0:
                    fft_norm = (fft / np.max(fft) * 255).astype(np.uint8)
                    for row in range(i, min(i+8, height)):
                        rgb_array[row, :len(fft_norm), 1] = fft_norm[:width]
        
        for x in range(width):
            envelope = np.convolve(img_array[:,x].astype(np.float32), np.ones(10)/10, mode='same')
            rgb_array[:,x,2] = envelope.astype(np.uint8)
        
        return Image.fromarray(rgb_array)
    
    # PNG ALPHA LSB
    def png_alpha_lsb_embed(self, img, payload):
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        full_payload = format(len(payload), '032b') + ''.join(format(b, '08b') for b in payload)
        if len(full_payload) > height * width:
            raise ValueError("Payload too large")
        
        bit_index = 0
        for y in range(height):
            for x in range(width):
                if bit_index < len(full_payload):
                    img_array[y,x,3] = (img_array[y,x,3] & 0xFE) | int(full_payload[bit_index])
                    bit_index += 1
        return Image.fromarray(img_array, 'RGBA')
    
    def png_alpha_lsb_extract(self, img_path):
        img = Image.open(img_path).convert('RGBA')
        img_array = np.array(img)
        bits = [str(img_array[y,x,3] & 1) for y in range(img_array.shape[0]) for x in range(img_array.shape[1])]
        length = int(''.join(bits[:32]), 2)
        payload_bits = bits[32:32 + length*8]
        return bytes([int(''.join(payload_bits[i:i+8]), 2) for i in range(0, len(payload_bits), 8)])
    
    # BMP PALETTE
    def bmp_palette_embed(self, img, payload):
        img_palette = img.convert('P', palette=Image.ADAPTIVE, colors=256)
        img_array = np.array(img_palette)
        full_payload = format(len(payload), '032b') + ''.join(format(b, '08b') for b in payload)
        
        bit_index = 0
        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                if bit_index < len(full_payload):
                    img_array[y,x] = (img_array[y,x] & 0xFE) | int(full_payload[bit_index])
                    bit_index += 1
        
        result = Image.fromarray(img_array, 'P')
        result.putpalette(img_palette.getpalette())
        return result
    
    def bmp_palette_extract(self, img_path):
        img = Image.open(img_path)
        if img.mode != 'P':
            return b""
        img_array = np.array(img)
        bits = [str(int(img_array[y,x]) & 1) for y in range(img_array.shape[0]) for x in range(img_array.shape[1])]
        if len(bits) < 32:
            return b""
        length = int(''.join(bits[:32]), 2)
        if length <= 0 or len(bits) < 32 + length*8:
            return b""
        payload_bits = bits[32:32 + length*8]
        return bytes([int(''.join(payload_bits[i:i+8]), 2) for i in range(0, len(payload_bits), 8)])
    
    # PNG DCT
    def png_dct_embed(self, img, payload):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img, dtype=np.int32)
        height, width = img_array.shape[:2]
        
        capacity_bits = ((height-8)//8) * ((width-8)//8)
        full_payload = format(len(payload), '032b') + ''.join(format(b, '08b') for b in payload)
        
        if len(full_payload) > capacity_bits:
            max_bytes = (capacity_bits - 32) // 8
            payload = payload[:max_bytes]
            full_payload = format(len(payload), '032b') + ''.join(format(b, '08b') for b in payload)
        
        bit_index = 0
        for y in range(0, height-8, 8):
            for x in range(0, width-8, 8):
                if bit_index < len(full_payload):
                    if int(full_payload[bit_index]) == 1:
                        img_array[y+4,x+4,0] = min(img_array[y+4,x+4,0] + 20, 255)
                    else:
                        img_array[y+4,x+4,0] = max(img_array[y+4,x+4,0] - 20, 0)
                    bit_index += 1
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8)), payload
    
    def png_dct_extract(self, img_path, original_img):
        stego = np.array(Image.open(img_path).convert('RGB'), dtype=np.int32)
        orig = np.array(original_img.convert('RGB'), dtype=np.int32)
        height, width = stego.shape[:2]
        
        bits = []
        for y in range(0, height-8, 8):
            for x in range(0, width-8, 8):
                diff = stego[y+4,x+4,0] - orig[y+4,x+4,0]
                bits.append('1' if diff > 10 else ('0' if diff < -10 else ('1' if diff >= 0 else '0')))
        
        if len(bits) < 32:
            return b""
        length = int(''.join(bits[:32]), 2)
        if length <= 0 or length > 50000 or len(bits) < 32 + length*8:
            return b""
        return bytes([int(''.join(bits[32+i:32+i+8]), 2) for i in range(0, length*8, 8)])
    
    # AUDIO WAV
    def save_wav(self, audio_data, sample_rate, filepath):
        with wave.open(str(filepath), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(audio_data.tobytes())
    
    def generate_dataset(self, num_images=12):
        payloads = self.read_seed_files()
        print(f"Found {len(payloads)} payload(s): {list(payloads.keys())}\n")
        
        methods = [
            ('png_alpha', 'png', self.png_alpha_lsb_embed, self.png_alpha_lsb_extract),
            ('bmp_palette', 'bmp', self.bmp_palette_embed, self.bmp_palette_extract),
            ('png_dct', 'png', self.png_dct_embed, self.png_dct_extract),
        ]
        image_types = ['gradient', 'geometric', 'noise', 'blocks']
        results = {'total': 0, 'success': 0, 'failed': []}
        
        for payload_name, payload_data in payloads.items():
            print(f"{'='*60}")
            print(f"Processing: {payload_name} ({len(payload_data)} bytes)")
            print(f"{'='*60}\n")
            
            # Audio-visual for Maya
            try:
                text = payload_data.decode('utf-8', errors='ignore')
                audio, sr = self.text_to_audio_simple(text[:500])
                
                # Save audio waveform as image in stego/ (contains the seed data)
                av_img = self.audio_waveform_to_image(audio)
                av_img.save(self.stego_dir / f"{payload_name}_audio_visual_maya.png", 'PNG')
                
                # Save WAV in stego/ (generated from seed text)
                self.save_wav(audio, sr, self.stego_dir / f"{payload_name}_speech.wav")
                
                print(f"  ✓ Audio-visual: {payload_name}_audio_visual_maya.png")
                print(f"  ✓ Speech audio: {payload_name}_speech.wav")
                print(f"  → Saved to stego/ (contains seed data)\n")
            except Exception as e:
                print(f"  ✗ Audio-visual failed: {e}\n")
            
            # Images
            for i in range(num_images):
                img_type = image_types[i % 4]
                method_name, ext, embed_func, extract_func = methods[i % 3]
                base = f"{payload_name}_{method_name}_{i:03d}"
                
                try:
                    clean_img = self.generate_diverse_clean_image(i*100 + hash(payload_name)%100, img_type)
                    clean_img.save(self.clean_dir / f"{base}.{ext}", 'BMP' if ext=='bmp' else 'PNG')
                    
                    if method_name == 'png_dct':
                        stego_img, actual_payload = embed_func(clean_img, payload_data)
                    else:
                        stego_img = embed_func(clean_img, payload_data)
                        actual_payload = payload_data
                    
                    stego_img.save(self.stego_dir / f"{base}.{ext}", 'BMP' if ext=='bmp' else 'PNG')
                    
                    extracted = extract_func(self.stego_dir / f"{base}.{ext}", clean_img) if method_name=='png_dct' else extract_func(self.stego_dir / f"{base}.{ext}")
                    
                    results['total'] += 1
                    if extracted == actual_payload:
                        print(f"  ✓ {base}.{ext}")
                        results['success'] += 1
                    else:
                        print(f"  ✗ {base}.{ext} (got {len(extracted)}, expected {len(actual_payload)})")
                        results['failed'].append(f"{base}.{ext}")
                except Exception as e:
                    print(f"  ✗ {base}: {e}")
                    results['failed'].append(f"{base}.{ext}")
        
        print(f"\n{'='*60}")
        print(f"Total: {results['total']}")
        print(f"Success: {results['success']}")
        print(f"Rate: {results['success']/results['total']*100:.1f}%\n")

if __name__ == "__main__":
    print("="*60)
    print("Project Starlight - Claude's Final Generator")
    print("="*60 + "\n")
    generator = ClaudeStegGenerator()
    generator.generate_dataset(num_images=12)
