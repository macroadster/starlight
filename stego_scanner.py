#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse

# Import the model class, feature extractor, and transform from the centralized file
from starlight_model import StarlightTwoStage, extract_features, transform_val as transform

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
                # Use imported feature extractor
                features = extract_features(path, img_original)
                
                # Convert to RGBA for model
                img_rgba = img_original.convert('RGBA')
                # Use imported transform
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
            normality_probs = normality_score.squeeze().cpu().numpy()
        
        # Process results
        for path, prob, norm_prob in zip(valid_paths, probs, normality_probs):
            clean_prob = norm_prob.item() # Normality score is the clean probability estimate
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
    args = parser.parse_args()
    
    print("=" * 80)
    print("Project Starlight - Steganography Scanner")
    print("Two-Stage Detection & Algorithm Classification")
    print("=" * 80)
    
    class_map = {0: 'clean', 1: 'alpha', 2: 'palette', 3: 'dct', 4: 'lsb', 5: 'eoi', 6: 'exif'}
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Loading model weights from {args.model}...")
    try:
        # StarlightTwoStage is imported from starlight_model.py
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
