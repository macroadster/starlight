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
                # Open and preserve original format for feature extraction
                img_original = Image.open(path)
                
                # Extract 24 features from ORIGINAL image
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
            normality_probs = normality_score.flatten().cpu().numpy()
        
        # Process results
        for path, prob, norm_prob in zip(valid_paths, probs, normality_probs):
            clean_prob = norm_prob.item() if isinstance(norm_prob, np.ndarray) else norm_prob
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
    
    print(f"Loading model from {args.model}...")
    try:
        # Load model with 24 features (upgraded from 15)
        model = StarlightTwoStage(num_stego_classes=6, feature_dim=24)
        
        # Handle both checkpoint formats
        checkpoint = torch.load(args.model, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with metadata
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            bal_acc = checkpoint.get('balanced_acc', 'unknown')
            print(f"Model loaded successfully (Epoch: {epoch}, Balanced Acc: {bal_acc})")
        else:
            # Old format - direct state_dict
            model.load_state_dict(checkpoint)
            print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    print(f"Model: StarlightTwoStage (7 classes, 24 features)")
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
    
    print("\n" + "=" * 80)
    print("SCAN SUMMARY")
    print("=" * 80)
    print(f"Total Images Scanned: {total}")
    print(f"Images Flagged as Stego: {flagged}")
    print(f"Clean Probability Threshold: {args.threshold:.2f}")
    
    # Breakdown by stego type
    if flagged > 0:
        print(f"\nDetected Stego Types:")
        stego_counts = {}
        for r in results:
            if r['is_stego']:
                stego_type = r['pred_class']
                stego_counts[stego_type] = stego_counts.get(stego_type, 0) + 1
        
        for stego_type in sorted(stego_counts.keys()):
            count = stego_counts[stego_type]
            print(f"  {stego_type.upper()}: {count} ({100*count/flagged:.1f}%)")
    
    print(f"\nDiscrepancies (Stage 1 vs Stage 2):")
    print(f"  Predicted Clean but Low Normality Score (< {args.threshold:.2f}): {discrepancies['clean_low_prob']}")
    print(f"  Predicted Stego but High Normality Score (> {args.threshold:.2f}): {discrepancies['stego_high_prob']}")
    print("=" * 80)
    
    # Detailed results only if verbose is True
    if args.verbose:
        print(f"\n--- Detailed Results for All Images ({total} images) ---\n")
        for idx, result in enumerate(results, 1):
            status = "🚨 STEGO" if result['is_stego'] else "✓ CLEAN"
            print(f"[{idx}] {os.path.basename(result['path'])}")
            print(f"  Status: {status}")
            print(f"  Overall Prediction: {result['pred_class'].upper()} (Conf: {result['pred_conf']:.4f})")
            print(f"  Stage 1 Normality Score: {result['clean_prob']:.4f}")
            
            if result['is_stego']:
                # Show all stego type probabilities
                print(f"  Stego Type Probabilities:")
                for class_idx in range(1, 7):
                    class_name = class_map[class_idx]
                    class_prob = result['all_probs'][class_idx]
                    print(f"    {class_name}: {class_prob:.4f}")
            
            # Flag discrepancies
            if result['is_stego'] and result['clean_prob'] > args.threshold:
                print(f"  ⚠️  DISCREPANCY: Predicted stego but high normality score!")
            elif not result['is_stego'] and result['clean_prob'] < args.threshold:
                print(f"  ⚠️  DISCREPANCY: Predicted clean but low normality score!")
            
            print()

if __name__ == "__main__":
    main()
