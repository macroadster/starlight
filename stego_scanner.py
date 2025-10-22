#!/usr/bin/env python3
import os
import sys
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.stats import entropy
import argparse
from PIL.ExifTags import TAGS

# Updated transform for RGBA input, matching trainer.py
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
])

def load_model_class(trainer_path='trainer.py'):
    """Dynamically load the StarlightTwoStage model class from trainer.py."""
    spec = importlib.util.spec_from_file_location("trainer", trainer_path)
    if spec is None:
        raise ImportError(f"Could not load trainer.py from {trainer_path}")
    trainer_module = importlib.util.module_from_spec(spec)
    sys.modules["trainer"] = trainer_module
    spec.loader.exec_module(trainer_module)
    return trainer_module.StarlightTwoStage

def extract_features(path, img):
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
                if tag == 'UserComment' and isinstance(value, bytes):
                    comment_length = min(len(value) / area, 1.0)
                if isinstance(value, (bytes, str)):
                    tag_values.append(value if isinstance(value, bytes) else value.encode('utf-8'))
            if tag_values:
                lengths = [len(v) for v in tag_values]
                hist = np.histogram(lengths, bins=10, range=(0, max(lengths or [1])))[0]
                exif_entropy = entropy(hist + 1e-10) / area if any(hist) else 0.0
        except:
            comment_length = 0.0
            exif_entropy = 0.0
    
    palette_present = 1.0 if img.mode == 'P' else 0.0
    palette = img.getpalette()
    palette_length = len(palette) / 3 if palette else 0.0
    if palette_present:
        hist = img.histogram()
        palette_entropy_value = entropy([h + 1 for h in hist if h > 0]) if any(hist) else 0.0
    else:
        palette_entropy_value = 0.0
    
    with open(path, 'rb') as f:
        data = f.read()
    if img.format == 'JPEG':
        eoi_pos = data.rfind(b'\xff\xd9')
        eof_length = len(data) - (eoi_pos + 2) if eoi_pos >= 0 else 0.0
    else:
        eof_length = 0.0
    eof_length_norm = min(eof_length / area, 1.0)
    
    has_alpha = 1.0 if img.mode in ('RGBA', 'LA', 'PA') else 0.0
    alpha_variance = 0.0
    alpha_mean = 0.5
    alpha_unique_ratio = 0.0
    
    if has_alpha and img.mode == 'RGBA':
        img_array = np.array(img)
        if img_array.shape[-1] == 4:
            alpha_channel = img_array[:, :, 3].astype(float)
            alpha_variance = np.var(alpha_channel) / 65025.0
            alpha_mean = np.mean(alpha_channel) / 255.0
            unique_alphas = len(np.unique(alpha_channel))
            total_pixels = alpha_channel.size
            alpha_unique_ratio = unique_alphas / min(total_pixels, 256)
    
    is_jpeg = 1.0 if img.format == 'JPEG' else 0.0
    is_png = 1.0 if img.format == 'PNG' else 0.0
    
    return np.array([
        file_size_norm, exif_present, exif_length_norm, comment_length,
        exif_entropy, palette_present, palette_length, palette_entropy_value,
        eof_length_norm, has_alpha, alpha_variance, alpha_mean, alpha_unique_ratio,
        is_jpeg, is_png
    ], dtype=np.float32)

def scan_directory(model, device, directory, clean_threshold, class_map, batch_size=32):
    """Scan a directory of images for steganography."""
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print(f"No images found in {directory}")
        return []
    
    results = []
    discrepancies = {'clean_low_prob': 0, 'stego_high_prob': 0}
    model.eval()
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_features = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                # Open and preserve original format
                img_original = Image.open(path)
                features = extract_features(path, img_original)
                
                # Convert to RGBA for model
                img_rgba = img_original.convert('RGBA')
                img_tensor = transform(img_rgba)
                
                batch_images.append(img_tensor)
                batch_features.append(torch.tensor(features))
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Stack batch
        batch_images = torch.stack(batch_images).to(device)
        batch_features = torch.stack(batch_features).to(device)
        
        with torch.no_grad():
            all_logits, normality_score, stego_type_logits = model(batch_images, batch_features)
            probs = F.softmax(all_logits, dim=1).cpu().numpy()
            normality_probs = normality_score.cpu().numpy()
        
        # Process results
        for path, prob, norm_prob in zip(valid_paths, probs, normality_probs):
            clean_prob = norm_prob.item()  # Use normality score for clean probability
            pred_class = np.argmax(prob)
            pred_label = class_map[pred_class]
            
            # Flag as stego only if the predicted class is not clean
            is_stego = pred_class != 0
            
            # Track discrepancies
            if is_stego and clean_prob > clean_threshold:
                discrepancies['stego_high_prob'] += 1
            elif not is_stego and clean_prob < clean_threshold:
                discrepancies['clean_low_prob'] += 1
            
            results.append({
                'path': path,
                'is_stego': is_stego,
                'clean_prob': clean_prob,
                'pred_class': pred_label,
                'pred_conf': prob[pred_class],
                'all_probs': prob
            })
        
        print(f"\rProcessed batch {(i // batch_size) + 1} of {(len(image_paths) + batch_size - 1) // batch_size}", end='')
    
    print()  # New line after progress
    return results, discrepancies

def main():
    parser = argparse.ArgumentParser(description="Project Starlight - Steganography Scanner")
    parser.add_argument('directory', help="Directory containing images to scan")
    parser.add_argument('--model', default='best_model.pth', help="Path to trained model")
    parser.add_argument('--threshold', type=float, default=0.8, 
                       help="Clean probability threshold (default: 0.8)")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for processing")
    parser.add_argument('--verbose', action='store_true', default=False, 
                       help="Print detailed results for all images (default: False, shows only summary)")
    parser.add_argument('--trainer-path', default='trainer.py', 
                       help="Path to trainer.py containing the model definition (default: trainer.py)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Project Starlight - Steganography Scanner")
    print("Two-Stage Detection & Algorithm Classification")
    print("=" * 80)
    
    # Load model class from trainer.py
    print(f"Loading model class from {args.trainer_path}...")
    try:
        StarlightTwoStage = load_model_class(args.trainer_path)
    except Exception as e:
        print(f"Error loading model class from {args.trainer_path}: {e}")
        sys.exit(1)
    
    # Load model weights
    print(f"Loading model weights from {args.model}...")
    class_map = {0: 'clean', 1: 'alpha', 2: 'palette', 3: 'dct', 4: 'lsb', 5: 'eoi', 6: 'exif'}
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    try:
        model = StarlightTwoStage(num_stego_classes=6, feature_dim=15)
        model.load_state_dict(torch.load(args.model, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    print(f"Model: StarlightTwoStage (7 classes)")
    print(f"Model classes: {', '.join(class_map.values())}")
    print(f"Clean Probability Threshold: {args.threshold}")
    
    # Scan directory
    results, discrepancies = scan_directory(model, device, args.directory, args.threshold, class_map, args.batch_size)
    
    if not results:
        print("\nNo images processed.")
        return
    
    # Summary
    total = len(results)
    flagged = sum(1 for r in results if r['is_stego'])
    
    print("\n--- SCAN SUMMARY ---")
    print(f"Total Images Scanned: {total}")
    print(f"Images Flagged as Stego: {flagged} (Clean Prob. Threshold: {args.threshold:.2f})")
    print(f"Discrepancies Detected:")
    print(f"  Predicted Clean with Low Clean Probability (< {args.threshold:.2f}): {discrepancies['clean_low_prob']}")
    print(f"  Predicted Stego with High Clean Probability (> {args.threshold:.2f}): {discrepancies['stego_high_prob']}")
    print("-" * 20)
    
    # Detailed results only if verbose is True
    if args.verbose:
        print(f"\n--- Detailed Results for All Images ({total} images) ---\n")
        for idx, result in enumerate(results, 1):
            status = "🚨 STEGO" if result['is_stego'] else "✓ CLEAN"
            print(f"[{idx}] {os.path.basename(result['path'])}")
            print(f"  Status: {status}")
            print(f"  Overall Prediction: {result['pred_class'].upper()} (Conf: {result['pred_conf']:.4f})")
            
            if result['is_stego']:
                # Find highest non-clean class
                stego_probs = result['all_probs'][1:]
                stego_idx = np.argmax(stego_probs) + 1
                stego_type = class_map[stego_idx]
                print(f"  Most Likely Stego Type: {stego_type.upper()} (Conf: {result['all_probs'][stego_idx]:.4f})")
            else:
                print(f"  Clean Confidence: {result['clean_prob']:.4f}")
            print()

if __name__ == "__main__":
    main()
