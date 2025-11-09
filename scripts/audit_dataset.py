import os
from PIL import Image
from pathlib import Path
import argparse
from collections import defaultdict
import json

def audit_dataset(clean_dir, stego_dir):
    """
    Performs a health check on a clean/stego image dataset using JSON sidecars for pairing.
    Checks for:
    1. Stego images missing JSON or clean_file references.
    2. Clean files mentioned in JSON but not found.
    3. Corrupt images that cannot be opened.
    4. Mismatched image dimensions between pairs.
    5. Mismatched image modes (e.g., RGB vs. RGBA) between pairs.
    """
    print("="*60)
    print(f"Starting JSON-driven audit for dataset:")
    print(f"  Clean dir: {clean_dir}")
    print(f"  Stego dir: {stego_dir}")
    print("="*60)

    clean_path_obj = Path(clean_dir)
    stego_path_obj = Path(stego_dir)
    issues = defaultdict(list)
    
    image_extensions = {'.jpg', '.png', '.gif', '.webp', '.bmp', '.jpeg'}
    stego_images = [f for f in stego_path_obj.iterdir() if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(stego_images)} potential stego images to audit...")
    
    pairs_to_check = []

    # First pass: identify all valid pairs from JSON files
    for stego_filepath in stego_images:
        json_path = stego_filepath.with_suffix(stego_filepath.suffix + '.json')
        
        if not json_path.exists():
            issues["stego_missing_json"].append(stego_filepath.name)
            continue
        
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            clean_filename = metadata.get('clean_file')
            if not clean_filename:
                issues["json_missing_clean_file_ref"].append(json_path.name)
                continue
            
            clean_filepath = clean_path_obj / clean_filename
            if not clean_filepath.exists():
                issues["clean_file_not_found"].append(f"{clean_filename} (referenced by {stego_filepath.name})")
                continue
            
            pairs_to_check.append((clean_filepath, stego_filepath))

        except (json.JSONDecodeError, KeyError) as e:
            issues["json_read_error"].append(f"{json_path.name} (Error: {e})")

    # Second pass: check integrity and attributes of the valid pairs
    print(f"Found {len(pairs_to_check)} valid pairs to analyze...")
    for i, (clean_img_path, stego_img_path) in enumerate(pairs_to_check):
        if (i + 1) % 50 == 0:
            print(f"  ...analyzed {i+1}/{len(pairs_to_check)} pairs...")

        try:
            clean_img = Image.open(clean_img_path)
            clean_img.load()
        except Exception as e:
            issues["corrupt_clean_image"].append(f"{clean_img_path.name} (Error: {e})")
            continue

        try:
            stego_img = Image.open(stego_img_path)
            stego_img.load()
        except Exception as e:
            issues["corrupt_stego_image"].append(f"{stego_img_path.name} (Error: {e})")
            continue

        if clean_img.size != stego_img.size:
            issues["dimension_mismatch"].append(f"{stego_img_path.name} (Clean: {clean_img.size}, Stego: {stego_img.size})")
        
        if clean_img.mode != stego_img.mode:
            issues["mode_mismatch"].append(f"{stego_img_path.name} (Clean: {clean_img.mode}, Stego: {stego_img.mode})")

    print("\n...Audit complete.")
    print("\n" + "="*60)
    print("AUDIT SUMMARY")
    print("="*60)

    if not issues:
        print("\n✅ No issues found. The dataset appears to be structurally sound based on JSON pairing.")
    else:
        for issue_type, file_list in issues.items():
            print(f"\n❌ Found {len(file_list)} issues of type: '{issue_type}'")
            for item in file_list[:10]:
                print(f"  - {item}")
            if len(file_list) > 10:
                print(f"  ... and {len(file_list) - 10} more.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit a steganography image dataset using JSON sidecars for pairing.")
    parser.add_argument("--clean_dir", required=True, help="Directory containing clean images.")
    parser.add_argument("--stego_dir", required=True, help="Directory containing stego images.")
    args = parser.parse_args()

    audit_dataset(args.clean_dir, args.stego_dir)
