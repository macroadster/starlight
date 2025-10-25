#!/usr/bin/env python3
"""
Starlight CNN Inference with Directory Scanning
Uses single CNN model trained with RF teacher starlight
Handles: clean + 4 pixel-based stego types
NEW: Can scan entire directories and provide summary statistics
"""
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
import pickle
from pathlib import Path
from collections import Counter, defaultdict

from starlight_model import extract_features, transform_val, StarlightCNN
from starlight_extractor import extract_alpha, extract_palette, extract_sdm, extract_lsb, extract_eoi, extract_exif

def predict_starlight(model, rf_model, device, image_path):
    """
    Prediction with starlight-trained CNN:
    1. Check if EXIF/EOI using RF features (fast)
    2. Use CNN for clean/pixel-based stego classification
    3. Extract message if stego detected
    """
    # Extract features
    img = Image.open(image_path)
    features = extract_features(image_path, img)
    
    # Quick RF check for file-level stego (EXIF/EOI)
    rf_pred = rf_model.predict([features])[0]
    rf_probs = rf_model.predict_proba([features])[0]
    
    class_names_7 = ['clean', 'alpha', 'palette', 'sdm', 'lsb', 'eoi', 'exif']
    rf_prediction = class_names_7[rf_pred]
    
    # If EXIF or EOI, return immediately (RF is authoritative for these)
    if rf_prediction in ['eoi', 'exif']:
        return {
            'type': rf_prediction,
            'confidence': rf_probs[rf_pred],
            'method': 'RF (file-level)',
            'is_clean': False
        }
    
    # Use CNN for clean/pixel-based classification
    original_mode = img.mode
    if original_mode in ('RGBA', 'LA', 'PA'):
        img_processed = img.convert('RGBA')
    else:
        img_rgb = img.convert('RGB')
        img_processed = Image.new('RGBA', img_rgb.size, (0, 0, 0, 255))
        img_processed.paste(img_rgb, (0, 0))
    
    img_tensor = transform_val(img_processed).unsqueeze(0).to(device)
    features_tensor = torch.tensor(features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor, features_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    
    class_names_5 = ['clean', 'alpha', 'palette', 'sdm', 'lsb']
    pred_class = np.argmax(probs)
    pred_type = class_names_5[pred_class]
    pred_conf = probs[pred_class]
    
    return {
        'type': pred_type,
        'confidence': pred_conf,
        'method': 'CNN (starlight-trained)',
        'is_clean': pred_class == 0,
        'all_probs': {class_names_5[i]: probs[i] for i in range(5)}
    }

def scan_directory(model, rf_model, device, directory, recursive=False, min_confidence=0.5):
    """
    Scan a directory for images and classify them
    
    Args:
        model: Starlight CNN model
        rf_model: Random Forest model
        device: Torch device
        directory: Directory path to scan
        recursive: If True, scan subdirectories
        min_confidence: Minimum confidence to count as detection
    
    Returns:
        dict with scan results and statistics
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    # Collect image files
    image_files = []
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            fpath = os.path.join(directory, file)
            if os.path.isfile(fpath) and Path(file).suffix.lower() in supported_extensions:
                image_files.append(fpath)
    
    if not image_files:
        return {
            'total_images': 0,
            'error': 'No image files found in directory'
        }
    
    # Scan all images
    results = []
    errors = []
    type_counts = Counter()
    confidence_by_type = defaultdict(list)
    
    print(f"\nScanning {len(image_files)} images...")
    for i, image_path in enumerate(image_files, 1):
        try:
            result = predict_starlight(model, rf_model, device, image_path)
            result['path'] = image_path
            result['filename'] = os.path.basename(image_path)
            results.append(result)
            
            # Count types (only if confidence meets threshold)
            if result['confidence'] >= min_confidence:
                type_counts[result['type']] += 1
                confidence_by_type[result['type']].append(result['confidence'])
            
            # Progress indicator
            if i % 50 == 0 or i == len(image_files):
                print(f"  Progress: {i}/{len(image_files)} ({100*i/len(image_files):.1f}%)", end='\r')
        
        except Exception as e:
            errors.append({'path': image_path, 'error': str(e)})
    
    print()  # New line after progress
    
    # Calculate statistics
    total_images = len(image_files)
    clean_count = type_counts.get('clean', 0)
    stego_count = total_images - clean_count - len(errors)
    
    # Average confidences per type
    avg_confidence = {}
    for stype, confs in confidence_by_type.items():
        avg_confidence[stype] = sum(confs) / len(confs) if confs else 0.0
    
    # Group by stego type
    stego_by_type = {k: v for k, v in type_counts.items() if k != 'clean'}
    
    return {
        'total_images': total_images,
        'clean_count': clean_count,
        'stego_count': stego_count,
        'errors': len(errors),
        'type_counts': dict(type_counts),
        'stego_by_type': stego_by_type,
        'avg_confidence': avg_confidence,
        'results': results,
        'error_details': errors,
        'clean_percentage': 100.0 * clean_count / total_images if total_images > 0 else 0,
        'stego_percentage': 100.0 * stego_count / total_images if total_images > 0 else 0
    }

def print_scan_summary(scan_results):
    """Print a formatted summary of scan results"""
    print("\n" + "="*80)
    print("DIRECTORY SCAN SUMMARY")
    print("="*80)
    
    print(f"\nTotal Images Scanned: {scan_results['total_images']}")
    if scan_results['errors'] > 0:
        print(f"Errors: {scan_results['errors']} (see details below)")
    
    print(f"\nOverall Classification:")
    print(f"  Clean Images:      {scan_results['clean_count']:4d} ({scan_results['clean_percentage']:.1f}%)")
    print(f"  Stego Images:      {scan_results['stego_count']:4d} ({scan_results['stego_percentage']:.1f}%)")
    
    if scan_results['stego_by_type']:
        print(f"\nSteganography Types Detected:")
        for stype in ['alpha', 'palette', 'sdm', 'lsb', 'eoi', 'exif']:
            count = scan_results['stego_by_type'].get(stype, 0)
            if count > 0:
                avg_conf = scan_results['avg_confidence'].get(stype, 0)
                pct = 100.0 * count / scan_results['stego_count'] if scan_results['stego_count'] > 0 else 0
                print(f"  {stype.upper():8s}: {count:4d} ({pct:5.1f}% of stego) - Avg Conf: {avg_conf:.2%}")
    
    if scan_results['avg_confidence']:
        print(f"\nAverage Confidence by Type:")
        for stype, conf in sorted(scan_results['avg_confidence'].items(), key=lambda x: -x[1]):
            print(f"  {stype:8s}: {conf:.2%}")
    
    if scan_results['error_details']:
        print(f"\nErrors ({len(scan_results['error_details'])}):")
        for err in scan_results['error_details'][:10]:  # Show first 10
            print(f"  {os.path.basename(err['path'])}: {err['error']}")
        if len(scan_results['error_details']) > 10:
            print(f"  ... and {len(scan_results['error_details']) - 10} more")
    
    print("="*80)

def print_detailed_results(scan_results, show_clean=False, min_confidence=0.0):
    """Print detailed results for each image"""
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    # Sort by type, then confidence
    sorted_results = sorted(
        scan_results['results'],
        key=lambda x: (x['type'], -x['confidence'])
    )
    
    current_type = None
    for result in sorted_results:
        # Skip clean images if not requested
        if not show_clean and result['is_clean']:
            continue
        
        # Skip low confidence results
        if result['confidence'] < min_confidence:
            continue
        
        # Print type header
        if result['type'] != current_type:
            current_type = result['type']
            print(f"\n{current_type.upper()} Images:")
            print("-" * 80)
        
        # Print result
        print(f"  {result['filename']:50s} {result['confidence']:6.2%} ({result['method']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starlight CNN Inference with Directory Scanning")
    parser.add_argument('image_paths', nargs='+', help="Images or directories to analyze")
    parser.add_argument('--cnn_model', default='starlight_cnn.pth', help="Starlight CNN model")
    parser.add_argument('--rf_model', default='starlight_rf.pkl', help="RF model (for EXIF/EOI)")
    parser.add_argument('--extract', action='store_true', help="Extract messages")
    parser.add_argument('--clean_path', default=None, help="Clean image for SDM")
    parser.add_argument('--recursive', '-r', action='store_true', help="Scan directories recursively")
    parser.add_argument('--detailed', '-d', action='store_true', help="Show detailed results for directory scans")
    parser.add_argument('--show_clean', action='store_true', help="Show clean images in detailed results")
    parser.add_argument('--min_confidence', type=float, default=0.5, help="Minimum confidence threshold (default: 0.5)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("STARLIGHT CNN INFERENCE")
    print("Single model handles: clean + alpha/palette/sdm/lsb")
    print("RF used only for EXIF/EOI (file-level)")
    print("=" * 80)
    
    # Load models
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"\nLoading models...")
    
    # Load starlight CNN
    model = StarlightCNN(num_classes=5, feature_dim=24)
    checkpoint = torch.load(args.cnn_model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✓ Starlight CNN loaded (Balanced Acc: {checkpoint.get('balanced_acc', 'N/A'):.4f})")
    print(f"  Clean Recall: {checkpoint.get('clean_recall', 'N/A'):.4f}")
    
    # Load RF for EXIF/EOI
    with open(args.rf_model, 'rb') as f:
        rf_model = pickle.load(f)
    rf_model.verbose = 0
    print(f"✓ RF loaded (for EXIF/EOI detection)")
    
    print(f"\nUsing device: {device}")
    print(f"Minimum confidence threshold: {args.min_confidence:.2%}\n")
    
    extraction_functions = {
        'alpha': extract_alpha,
        'palette': extract_palette,
        'sdm': extract_sdm,
        'lsb': extract_lsb,
        'eoi': extract_eoi,
        'exif': extract_exif
    }
    
    # Separate directories from files
    directories = []
    files = []
    
    for path in args.image_paths:
        if os.path.isdir(path):
            directories.append(path)
        elif os.path.isfile(path):
            files.append(path)
        else:
            print(f"Warning: Path not found - {path}")
    
    # Process directories first
    if directories:
        for directory in directories:
            print(f"\n{'='*80}")
            print(f"SCANNING DIRECTORY: {directory}")
            if args.recursive:
                print("(Recursive mode enabled)")
            print(f"{'='*80}")
            
            scan_results = scan_directory(
                model, rf_model, device, directory, 
                recursive=args.recursive,
                min_confidence=args.min_confidence
            )
            
            if 'error' in scan_results:
                print(f"\nError: {scan_results['error']}")
                continue
            
            print_scan_summary(scan_results)
            
            if args.detailed:
                print_detailed_results(
                    scan_results, 
                    show_clean=args.show_clean,
                    min_confidence=args.min_confidence
                )
    
    # Process individual files
    if files:
        for image_path in files:
            print(f"\n{'='*80}")
            print(f"Analysis: {os.path.basename(image_path)}")
            print(f"{'='*80}")
            
            result = predict_starlight(model, rf_model, device, image_path)
            
            print(f"\nDetection Result:")
            print(f"  Type: {result['type'].upper()}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Method: {result['method']}")
            
            if result.get('all_probs'):
                print(f"\n  All Probabilities:")
                for name, prob in result['all_probs'].items():
                    print(f"    {name}: {prob:.2%}")
            
            # Extract message if requested and not clean
            if args.extract and not result['is_clean']:
                stego_type = result['type']
                
                if stego_type in extraction_functions:
                    print(f"\nExtracting message ({stego_type})...")
                    extractor = extraction_functions[stego_type]
                    
                    if stego_type == 'lsb':
                        message, _ = extractor(image_path, channel='all')
                    elif stego_type == 'sdm':
                        message, _ = extractor(image_path, clean_path=args.clean_path)
                    else:
                        message, _ = extractor(image_path)
                    
                    if message:
                        print(f"\n✓ MESSAGE EXTRACTED:")
                        print(f"{'='*80}")
                        print(message)
                        print(f"{'='*80}")
                    else:
                        print(f"✗ No message extracted")
            
            print()
