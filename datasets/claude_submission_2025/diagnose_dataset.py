#!/usr/bin/env python3
"""
Dataset diagnostic script - helps understand why model isn't learning
"""

import numpy as np
from PIL import Image
from pathlib import Path
import json

def analyze_image_statistics(img_path):
    """Analyze pixel statistics of an image"""
    img = Image.open(img_path)
    
    # Convert to RGB
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    arr = np.array(img)
    
    return {
        'mean': arr.mean(),
        'std': arr.std(),
        'min': arr.min(),
        'max': arr.max(),
        'shape': arr.shape
    }

def compare_clean_stego_pair(clean_path, stego_path):
    """Compare a clean/stego pair"""
    clean_img = Image.open(clean_path)
    stego_img = Image.open(stego_path)
    
    print(f"\n  Original modes: clean={clean_img.mode}, stego={stego_img.mode}")
    
    # For RGBA images, analyze alpha channel separately
    if clean_img.mode == 'RGBA' and stego_img.mode == 'RGBA':
        clean_alpha = np.array(clean_img.split()[3]).astype(np.float32)
        stego_alpha = np.array(stego_img.split()[3]).astype(np.float32)
        
        alpha_diff = np.abs(clean_alpha - stego_alpha)
        print(f"  Alpha channel analysis:")
        print(f"    Max diff: {alpha_diff.max()}")
        print(f"    Mean diff: {alpha_diff.mean():.6f}")
        print(f"    Pixels changed: {(alpha_diff > 0).sum()} / {alpha_diff.size}")
        print(f"    Percent changed: {(alpha_diff > 0).sum() / alpha_diff.size * 100:.2f}%")
    
    # Convert both to RGB for RGB comparison
    if clean_img.mode == 'RGBA':
        bg = Image.new('RGB', clean_img.size, (0, 0, 0))
        bg.paste(clean_img, mask=clean_img.split()[3])
        clean_img = bg
    elif clean_img.mode != 'RGB':
        clean_img = clean_img.convert('RGB')
    
    if stego_img.mode == 'RGBA':
        bg = Image.new('RGB', stego_img.size, (0, 0, 0))
        bg.paste(stego_img, mask=stego_img.split()[3])
        stego_img = bg
    elif stego_img.mode != 'RGB':
        stego_img = stego_img.convert('RGB')
    
    clean_arr = np.array(clean_img).astype(np.float32)
    stego_arr = np.array(stego_img).astype(np.float32)
    
    diff = np.abs(clean_arr - stego_arr)
    
    print(f"  RGB comparison (after conversion):")
    return {
        'max_diff': diff.max(),
        'mean_diff': diff.mean(),
        'pixels_changed': (diff > 0).sum(),
        'total_pixels': diff.size,
        'percent_changed': (diff > 0).sum() / diff.size * 100
    }

def main():
    clean_dir = Path('clean')
    stego_dir = Path('stego')
    
    print("="*60)
    print("Dataset Diagnostic Report")
    print("="*60)
    
    # Count files
    clean_files = sorted(list(clean_dir.glob('*.png')) + list(clean_dir.glob('*.bmp')))
    stego_files = sorted(list(stego_dir.glob('*.png')) + list(stego_dir.glob('*.bmp')))
    
    print(f"\nFile counts:")
    print(f"  Clean: {len(clean_files)}")
    print(f"  Stego: {len(stego_files)}")
    
    # Analyze a few images
    print(f"\nSample clean image statistics:")
    if clean_files:
        stats = analyze_image_statistics(clean_files[0])
        print(f"  {clean_files[0].name}")
        for k, v in stats.items():
            print(f"    {k}: {v}")
    
    print(f"\nSample stego image statistics:")
    if stego_files:
        stats = analyze_image_statistics(stego_files[0])
        print(f"  {stego_files[0].name}")
        for k, v in stats.items():
            print(f"    {k}: {v}")
    
    # Compare pairs
    print(f"\nClean vs Stego comparison:")
    
    # Find matching pairs
    pairs_found = 0
    total_diff_stats = []
    
    for stego_file in stego_files[:5]:  # Check first 5
        # Try to find matching clean file
        base_name = stego_file.stem
        for clean_file in clean_files:
            if clean_file.stem == base_name:
                print(f"\n  Pair: {base_name}")
                diff_stats = compare_clean_stego_pair(clean_file, stego_file)
                for k, v in diff_stats.items():
                    print(f"    {k}: {v}")
                total_diff_stats.append(diff_stats)
                pairs_found += 1
                break
    
    if not total_diff_stats:
        print("  No matching clean/stego pairs found!")
        print("  Files should have same base name in clean/ and stego/")
    else:
        print(f"\n  Total pairs analyzed: {pairs_found}")
        avg_percent = sum(s['percent_changed'] for s in total_diff_stats) / len(total_diff_stats)
        print(f"  Average pixels changed: {avg_percent:.2f}%")
    
    # Check JSON metadata
    print(f"\nMetadata check:")
    json_files = list(stego_dir.glob('*.json'))
    print(f"  JSON sidecars: {len(json_files)}")
    
    if json_files:
        with open(json_files[0]) as f:
            metadata = json.load(f)
        print(f"  Sample metadata: {json_files[0].name}")
        print(f"    {json.dumps(metadata, indent=6)}")
    
    print("\n" + "="*60)
    print("Diagnostic complete")
    print("="*60)
    
    # Recommendations
    print("\nRecommendations:")
    if not total_diff_stats:
        print("  ⚠ No matching pairs found - check filename alignment")
    elif avg_percent < 0.1:
        print("  ⚠ Very few pixels changed - steganography may be too subtle")
        print("  → Try methods with stronger artifacts")
    elif len(clean_files) + len(stego_files) < 40:
        print("  ⚠ Small dataset - consider generating more images")
        print("  → Aim for 100+ pairs for better training")
    else:
        print("  ✓ Dataset looks reasonable for training")

if __name__ == "__main__":
    main()
