#!/usr/bin/env python3
"""
Project Starlight - Steganography Algorithm Classifier
Trains a model to identify which embedding technique was used in stego images

Supported Algorithms:
1. PNG Alpha Channel LSB
2. BMP Palette Manipulation
3. PNG DCT Coefficient Embedding
4. Audio-Visual Patterns
5. Clean (no steganography)

Author: AI Trainer for Project Starlight
Date: 2025
License: MIT
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
import json
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle

class StegoFeatureExtractor:
    """Extract features that distinguish steganography methods"""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_alpha_channel_features(self, img):
        """Features specific to alpha channel LSB"""
        features = []
        
        if img.mode != 'RGBA':
            # No alpha channel - return zeros
            return [0] * 8
        
        img_array = np.array(img)
        alpha_channel = img_array[:, :, 3]
        
        # LSB statistics
        lsb = alpha_channel & 1
        features.append(np.mean(lsb))  # LSB mean
        features.append(np.std(lsb))   # LSB std
        
        # Chi-square test for LSB randomness
        hist = np.histogram(alpha_channel, bins=256)[0]
        pairs = hist[::2] + hist[1::2]
        chi_sq = np.sum((hist[::2] - hist[1::2])**2 / (pairs + 1e-10))
        features.append(chi_sq)
        
        # Sequential correlation
        lsb_flat = lsb.flatten()
        try:
            if len(lsb_flat) > 1 and np.std(lsb_flat[:-1]) > 0 and np.std(lsb_flat[1:]) > 0:
                corr = np.corrcoef(lsb_flat[:-1], lsb_flat[1:])[0, 1]
            else:
                corr = 0
        except:
            corr = 0
        features.append(corr)
        
        # Alpha channel entropy
        hist_norm = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        features.append(entropy)
        
        # Alpha uniqueness
        unique_ratio = len(np.unique(alpha_channel)) / alpha_channel.size
        features.append(unique_ratio)
        
        # LSB plane complexity
        lsb_transitions = np.sum(np.abs(np.diff(lsb_flat)))
        features.append(lsb_transitions / len(lsb_flat))
        
        # Alpha variance
        features.append(np.var(alpha_channel))
        
        return features
    
    def extract_palette_features(self, img):
        """Features specific to palette manipulation"""
        features = []
        
        # Check if image has palette
        has_palette = img.mode == 'P'
        features.append(float(has_palette))
        
        if not has_palette:
            return features + [0] * 7
        
        img_array = np.array(img)
        
        # Palette index LSB statistics
        lsb = img_array & 1
        features.append(np.mean(lsb))
        features.append(np.std(lsb))
        
        # Palette index distribution
        hist = np.histogram(img_array, bins=256)[0]
        hist_norm = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        features.append(entropy)
        
        # Color pair analysis (adjacent indices should have similar colors)
        palette = img.getpalette()
        if palette:
            palette_array = np.array(palette).reshape(-1, 3)
            color_diffs = []
            for i in range(0, len(palette_array) - 1, 2):
                if i + 1 < len(palette_array):
                    diff = np.linalg.norm(palette_array[i] - palette_array[i+1])
                    color_diffs.append(diff)
            features.append(np.mean(color_diffs) if color_diffs else 0)
            features.append(np.std(color_diffs) if color_diffs else 0)
        else:
            features.extend([0, 0])
        
        # Index transitions
        transitions = np.sum(np.abs(np.diff(img_array.flatten())))
        features.append(transitions / img_array.size)
        
        # Unique palette entries used
        unique_ratio = len(np.unique(img_array)) / img_array.size
        features.append(unique_ratio)
        
        return features
    
    def extract_dct_features(self, img):
        """Features specific to DCT coefficient embedding"""
        features = []
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img, dtype=np.float32)
        
        # Downsample for faster computation if image is large
        max_dim = 512
        if img_array.shape[0] > max_dim or img_array.shape[1] > max_dim:
            scale = max_dim / max(img_array.shape[0], img_array.shape[1])
            new_height = int(img_array.shape[0] * scale)
            new_width = int(img_array.shape[1] * scale)
            img_pil = Image.fromarray(img_array.astype(np.uint8))
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
            img_array = np.array(img_pil, dtype=np.float32)
        
        # Block-based analysis (8x8 blocks like DCT)
        block_size = 8
        height, width = img_array.shape[:2]
        
        block_variances = []
        mid_pixel_changes = []
        
        # Sample blocks instead of checking all
        sample_blocks = min(100, (height // block_size) * (width // block_size))
        block_count = 0
        
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                if block_count >= sample_blocks:
                    break
                    
                block = img_array[y:y+block_size, x:x+block_size, 0]
                block_variances.append(np.var(block))
                
                # Check middle pixel (DCT embedding typically modifies mid-frequency)
                mid_y, mid_x = block_size // 2, block_size // 2
                mid_pixel_changes.append(block[mid_y, mid_x])
                block_count += 1
            
            if block_count >= sample_blocks:
                break
        
        features.append(np.mean(block_variances) if block_variances else 0)
        features.append(np.std(block_variances) if block_variances else 0)
        features.append(np.mean(mid_pixel_changes) if mid_pixel_changes else 0)
        features.append(np.std(mid_pixel_changes) if mid_pixel_changes else 0)
        
        # High-frequency noise analysis (sampled)
        gray = np.mean(img_array, axis=2)
        sample_size = min(1000, (gray.shape[0]-2) * (gray.shape[1]-2))
        
        # Sample random positions for edge detection
        y_coords = np.random.randint(1, gray.shape[0]-1, sample_size)
        x_coords = np.random.randint(1, gray.shape[1]-1, sample_size)
        
        edges = []
        for y, x in zip(y_coords, x_coords):
            edge = abs(gray[y, x] * -4 + 
                      gray[y-1, x] + gray[y+1, x] + 
                      gray[y, x-1] + gray[y, x+1])
            edges.append(edge)
        
        features.append(np.mean(edges) if edges else 0)
        features.append(np.std(edges) if edges else 0)
        
        # Periodic patterns in blocks (sampled)
        if len(block_variances) > 10:
            fft = np.fft.fft(block_variances[:min(100, len(block_variances))])
            features.append(np.mean(np.abs(fft)))
        else:
            features.append(0)
        
        # Gradient analysis (sampled)
        sample_rows = min(50, gray.shape[0])
        row_indices = np.linspace(0, gray.shape[0]-1, sample_rows, dtype=int)
        
        grad_x_vals = []
        grad_y_vals = []
        
        for idx in row_indices[:-1]:
            if idx + 1 < gray.shape[0] and idx < gray.shape[1] - 1:
                grad_x_vals.extend(np.abs(gray[idx, 1:] - gray[idx, :-1]))
                grad_y_vals.extend(np.abs(gray[idx+1, :] - gray[idx, :]))
        
        features.append(np.mean(grad_x_vals) if grad_x_vals else 0)
        features.append(np.mean(grad_y_vals) if grad_y_vals else 0)
        
        return features
    
    def extract_audio_visual_features(self, img):
        """Features specific to audio waveform visualizations"""
        features = []
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Downsample for faster computation
        downsample_factor = max(1, img_array.shape[0] // 128)
        if downsample_factor > 1:
            img_array = img_array[::downsample_factor, ::downsample_factor, :]
        
        # Channel separation analysis (audio-visual has distinct channel patterns)
        r_channel = img_array[:, :, 0]
        g_channel = img_array[:, :, 1]
        b_channel = img_array[:, :, 2]
        
        # Cross-channel correlation (audio-visual has low correlation)
        # Sample randomly for speed
        sample_size = min(10000, r_channel.size)
        indices = np.random.choice(r_channel.size, sample_size, replace=False)
        
        r_flat = r_channel.flatten()[indices]
        g_flat = g_channel.flatten()[indices]
        b_flat = b_channel.flatten()[indices]
        
        # Calculate correlations with error handling
        try:
            if np.std(r_flat) > 0 and np.std(g_flat) > 0:
                corr_rg = np.corrcoef(r_flat, g_flat)[0, 1]
            else:
                corr_rg = 0
        except:
            corr_rg = 0
            
        try:
            if np.std(r_flat) > 0 and np.std(b_flat) > 0:
                corr_rb = np.corrcoef(r_flat, b_flat)[0, 1]
            else:
                corr_rb = 0
        except:
            corr_rb = 0
            
        try:
            if np.std(g_flat) > 0 and np.std(b_flat) > 0:
                corr_gb = np.corrcoef(g_flat, b_flat)[0, 1]
            else:
                corr_gb = 0
        except:
            corr_gb = 0
        
        features.append(corr_rg)
        features.append(corr_rb)
        features.append(corr_gb)
        
        # Horizontal patterns (waveforms have strong horizontal structure)
        row_variances = np.var(img_array, axis=1)
        col_variances = np.var(img_array, axis=0)
        
        features.append(np.mean(row_variances))
        features.append(np.mean(col_variances))
        features.append(np.mean(row_variances) / (np.mean(col_variances) + 1e-10))
        
        # Frequency content (FFT on sample rows)
        sample_rows = min(10, r_channel.shape[0])
        row_indices = np.linspace(0, r_channel.shape[0]-1, sample_rows, dtype=int)
        row_ffts = []
        for idx in row_indices:
            row = img_array[idx, :, 0]
            fft = np.abs(np.fft.fft(row))
            row_ffts.append(np.mean(fft))
        
        features.append(np.mean(row_ffts))
        features.append(np.std(row_ffts))
        
        # Simplified repetitive pattern detection (using variance instead of autocorr)
        # High variance in row means = less repetitive
        row_means = np.mean(img_array[:, :, 0], axis=1)
        features.append(np.var(row_means))
        
        return features
    
    def extract_general_features(self, img):
        """General statistical features"""
        features = []
        
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Basic statistics
        features.append(np.mean(img_array))
        features.append(np.std(img_array))
        features.append(np.var(img_array))
        
        # Entropy
        hist = np.histogram(img_array, bins=256)[0]
        hist_norm = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        features.append(entropy)
        
        # Color channel statistics
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            for i in range(3):
                channel = img_array[:, :, i]
                features.append(np.mean(channel))
                features.append(np.std(channel))
        else:
            features.extend([0] * 6)
        
        return features
    
    def extract_jpeg_features(self, img_path, img):
        """Features specific to JPEG steganography (EOI and EXIF)"""
        features = []
        
        # Check if it's a JPEG file
        is_jpeg = str(img_path).lower().endswith(('.jpg', '.jpeg'))
        features.append(float(is_jpeg))
        
        if not is_jpeg:
            return features + [0] * 11
        
        try:
            # Read raw file data for EOI marker analysis
            with open(img_path, 'rb') as f:
                data = f.read()
            
            # Look for EOI marker (0xFF 0xD9)
            eoi_pos = data.rfind(b'\xff\xd9')
            file_size = len(data)
            
            # Data after EOI (EOI steganography)
            if eoi_pos > 0 and eoi_pos < file_size - 2:
                data_after_eoi = file_size - (eoi_pos + 2)
                features.append(float(data_after_eoi > 0))
                features.append(data_after_eoi / file_size if file_size > 0 else 0)
            else:
                features.extend([0, 0])
            
            # EXIF data analysis
            try:
                exif_data = img._getexif() if hasattr(img, '_getexif') else None
                has_exif = exif_data is not None and len(exif_data) > 0
                features.append(float(has_exif))
                
                if has_exif:
                    # Number of EXIF tags
                    features.append(len(exif_data))
                    # Check for unusual EXIF tags
                    unusual_tags = sum(1 for tag in exif_data.keys() if tag > 50000)
                    features.append(unusual_tags)
                else:
                    features.extend([0, 0])
            except:
                features.extend([0, 0, 0])
            
            # JPEG segment analysis
            jpeg_markers = [b'\xff\xd8', b'\xff\xe0', b'\xff\xe1', b'\xff\xdb', b'\xff\xc0']
            marker_counts = [data.count(marker) for marker in jpeg_markers]
            features.extend(marker_counts[:5])
            
            # File size ratio (actual vs expected from image dimensions)
            expected_size = img.size[0] * img.size[1] * 3 * 0.1  # rough JPEG compression estimate
            size_ratio = file_size / expected_size if expected_size > 0 else 1
            features.append(size_ratio)
            
        except Exception as e:
            features.extend([0] * 11)
        
        return features
    
    def extract_webp_features(self, img):
        """Features specific to WebP LSB and Alpha steganography"""
        features = []
        
        # WebP images are typically loaded as RGB/RGBA
        if img.format == 'WEBP':
            features.append(1.0)
        else:
            features.append(0.0)
            return features + [0] * 11
        
        img_array = np.array(img)
        
        # Check if it has alpha channel
        has_alpha = (img.mode == 'RGBA' and len(img_array.shape) == 3 and img_array.shape[2] == 4)
        features.append(float(has_alpha))
        
        # Alpha channel LSB analysis (if present)
        if has_alpha:
            alpha_channel = img_array[:, :, 3]
            alpha_lsb = alpha_channel & 1
            features.append(np.mean(alpha_lsb))
            features.append(np.std(alpha_lsb))
        else:
            features.extend([0, 0])
        
        # LSB analysis on RGB channels
        if len(img_array.shape) == 3:
            for i in range(min(3, img_array.shape[2])):
                channel = img_array[:, :, i]
                lsb = channel & 1
                features.append(np.mean(lsb))
                features.append(np.std(lsb))
        else:
            features.extend([0] * 6)
        
        # WebP compression artifacts
        # Check for block patterns (WebP uses 4x4 or 8x8 blocks)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array[:, :, :3], axis=2)
        else:
            gray = img_array
        
        block_size = 4
        if gray.shape[0] >= block_size and gray.shape[1] >= block_size:
            block_var = []
            for y in range(0, gray.shape[0] - block_size, block_size):
                for x in range(0, gray.shape[1] - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    block_var.append(np.var(block))
            features.append(np.mean(block_var) if block_var else 0)
        else:
            features.append(0)
        
        return features
    
    def extract_all_features(self, img_path):
        """Extract all features from an image"""
        img = Image.open(img_path)
        
        features = []
        features.extend(self.extract_general_features(img))
        features.extend(self.extract_alpha_channel_features(img))
        features.extend(self.extract_palette_features(img))
        features.extend(self.extract_dct_features(img))
        features.extend(self.extract_audio_visual_features(img))
        features.extend(self.extract_jpeg_features(img_path, img))
        features.extend(self.extract_webp_features(img))
        
        return np.array(features)


class StegoClassifier:
    """Train and evaluate steganography algorithm classifier"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "datasets"
        self.extractor = StegoFeatureExtractor()
        self.model = None
        self.label_map = {
            'clean': 0,
            'png_alpha': 1,
            'bmp_palette': 2,
            'png_dct': 3,
            'audio_visual': 4,
            'png_lsb': 5,
            'jpeg_eoi': 6,
            'jpeg_exif': 7,
            'webp_lsb': 8,
            'gif_palette': 9,
            'webp_alpha': 10
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
    def identify_algorithm(self, filename):
        """Identify algorithm from filename"""
        filename_lower = filename.lower()
        
        # Check file extension first for format-specific methods
        is_jpeg = filename_lower.endswith(('.jpg', '.jpeg'))
        is_png = filename_lower.endswith('.png')
        is_webp = filename_lower.endswith('.webp')
        is_gif = filename_lower.endswith('.gif')
        is_bmp = filename_lower.endswith('.bmp')
        
        # Check for specific algorithm markers in filename
        # Order matters - check more specific patterns first
        
        # JPEG-specific methods
        if is_jpeg:
            if 'eoi' in filename_lower:
                return 'jpeg_eoi'
            elif 'exif' in filename_lower:
                return 'jpeg_exif'
        
        # WebP-specific methods
        if is_webp:
            if 'alpha' in filename_lower:
                return 'webp_alpha'
            elif 'lsb' in filename_lower:
                return 'webp_lsb'
        
        # GIF-specific methods
        if is_gif:
            if 'palette' in filename_lower:
                return 'gif_palette'
        
        # PNG-specific methods
        if is_png:
            if 'alpha' in filename_lower:
                return 'png_alpha'
            elif 'dct' in filename_lower:
                return 'png_dct'
            elif 'lsb' in filename_lower:
                return 'png_lsb'
            # Check for audio_visual ONLY for PNG and with specific keywords
            elif ('audio' in filename_lower or 'visual' in filename_lower or 'maya' in filename_lower):
                return 'audio_visual'
        
        # BMP-specific methods
        if is_bmp:
            if 'palette' in filename_lower:
                return 'bmp_palette'
        
        # If no specific pattern matched
        return 'unknown'
    
    def load_dataset(self):
        """Load and label all images from datasets/*/clean and datasets/*/stego"""
        X = []
        y = []
        filenames = []
        
        if not self.datasets_dir.exists():
            print(f"Warning: {self.datasets_dir} directory not found!")
            return np.array(X), np.array(y), filenames
        
        # Find all submission directories
        submission_dirs = [d for d in self.datasets_dir.iterdir() if d.is_dir()]
        
        if not submission_dirs:
            print(f"Warning: No submission directories found in {self.datasets_dir}")
            return np.array(X), np.array(y), filenames
        
        print(f"Found {len(submission_dirs)} submission directories")
        print("="*60)
        
        # Load clean images from all submissions
        print("\nLoading clean images...")
        for submission_dir in submission_dirs:
            clean_dir = submission_dir / "clean"
            if clean_dir.exists():
                clean_count = 0
                for img_path in clean_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.png', '.bmp', '.jpg', '.jpeg']:
                        try:
                            features = self.extractor.extract_all_features(img_path)
                            X.append(features)
                            y.append(self.label_map['clean'])
                            filenames.append(f"{submission_dir.name}/{img_path.name}")
                            clean_count += 1
                            
                            # Progress indicator
                            if clean_count % 10 == 0:
                                print(f"    Processed {clean_count} images...", end='\r')
                        except Exception as e:
                            print(f"  Error loading {submission_dir.name}/{img_path.name}: {e}")
                if clean_count > 0:
                    print(f"  {submission_dir.name}: {clean_count} clean images          ")
        
        total_clean = len([l for l in y if l == 0])
        print(f"\nTotal clean images: {total_clean}")
        
        # Load stego images from all submissions
        print("\nLoading stego images...")
        for submission_dir in submission_dirs:
            stego_dir = submission_dir / "stego"
            if stego_dir.exists():
                stego_counts = defaultdict(int)
                total_stego = 0
                for img_path in stego_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.png', '.bmp', '.jpg', '.jpeg', '.webp', '.gif']:
                        try:
                            algorithm = self.identify_algorithm(img_path.name)
                            if algorithm == 'unknown':
                                print(f"  Warning: Could not identify algorithm for {submission_dir.name}/{img_path.name}")
                                continue
                            
                            features = self.extractor.extract_all_features(img_path)
                            X.append(features)
                            y.append(self.label_map[algorithm])
                            filenames.append(f"{submission_dir.name}/{img_path.name}")
                            stego_counts[algorithm] += 1
                            total_stego += 1
                            
                            # Progress indicator
                            if total_stego % 10 == 0:
                                print(f"    Processed {total_stego} images...", end='\r')
                        except Exception as e:
                            print(f"  Error loading {submission_dir.name}/{img_path.name}: {e}")
                
                if stego_counts:
                    print(f"  {submission_dir.name}: {total_stego} stego images          ")
                    for algo, count in stego_counts.items():
                        print(f"    - {algo}: {count}")
        
        print(f"\n{'='*60}")
        print("Dataset Summary:")
        print(f"{'='*60}")
        for algo_name, label in self.label_map.items():
            count = sum(1 for l in y if l == label)
            print(f"  {algo_name}: {count} images")
        print(f"  Total: {len(y)} images")
        
        # Check feature consistency
        if len(X) > 0:
            feature_lengths = [len(features) for features in X]
            unique_lengths = set(feature_lengths)
            if len(unique_lengths) > 1:
                print(f"\n⚠️  WARNING: Inconsistent feature lengths detected!")
                print(f"  Unique lengths: {unique_lengths}")
                for length in unique_lengths:
                    count = feature_lengths.count(length)
                    indices = [i for i, l in enumerate(feature_lengths) if l == length]
                    print(f"  Length {length}: {count} samples")
                    if len(indices) <= 5:
                        for idx in indices[:5]:
                            print(f"    - {filenames[idx]}")
                return np.array([]), np.array([]), []
            else:
                print(f"\n✓ All samples have {feature_lengths[0]} features")
        
        print(f"  Total: {len(y)} images")
        
        return np.array(X), np.array(y), filenames
    
    def train(self, X, y):
        """Train the classifier"""
        print("\n" + "="*60)
        print("TRAINING STEGANOGRAPHY ALGORITHM CLASSIFIER")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"\nTraining accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"CV scores: {cv_scores}")
        print(f"CV mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        # Detailed classification report
        y_pred = self.model.predict(X_test)
        
        # Get unique labels present in the test data
        unique_labels = sorted(set(y_test) | set(y_pred))
        target_names = [self.reverse_label_map[i] for i in unique_labels]
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            labels=unique_labels,
            target_names=target_names
        ))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        print("Rows: True labels, Columns: Predicted labels")
        print("Order:", target_names)
        print(cm)
        
        # Feature importance
        print("\nTop 20 Most Important Features:")
        feature_importance = self.model.feature_importances_
        indices = np.argsort(feature_importance)[::-1][:20]
        
        for i, idx in enumerate(indices):
            print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
        
        return self.model, test_score
    
    def save_model(self, filepath="stego_classifier_model.pkl"):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'label_map': self.label_map,
            'extractor': self.extractor
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath="stego_classifier_model.pkl"):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_map = model_data['label_map']
        self.extractor = model_data['extractor']
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        print(f"Model loaded from {filepath}")
    
    def predict(self, img_path):
        """Predict algorithm for a single image"""
        features = self.extractor.extract_all_features(img_path)
        features = features.reshape(1, -1)
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        algorithm = self.reverse_label_map[prediction]
        confidence = probabilities[prediction]
        
        # Convert probabilities array to dict
        prob_dict = {
            self.reverse_label_map[i]: probabilities[i]
            for i in range(len(probabilities))
        }
        
        return algorithm, confidence, prob_dict
    
    def analyze_directory(self, directory):
        """Analyze all images in a directory"""
        directory = Path(directory)
        results = []
        
        for img_path in directory.glob("*.*"):
            if img_path.suffix.lower() in ['.png', '.bmp', '.jpg', '.jpeg', '.webp', '.gif']:
                try:
                    algorithm, confidence, probs = self.predict(img_path)
                    results.append({
                        'filename': img_path.name,
                        'predicted_algorithm': algorithm,
                        'confidence': confidence,
                        'probabilities': {
                            self.reverse_label_map[i]: probs[i]
                            for i in range(len(probs))
                        }
                    })
                except Exception as e:
                    print(f"Error analyzing {img_path.name}: {e}")
        
        return results


def main():
    print("="*60)
    print("Project Starlight - Steganography Algorithm Classifier")
    print("Training AI to Detect Embedding Techniques")
    print("="*60)
    print("\nSupported Algorithms:")
    print("  1. PNG Alpha Channel LSB")
    print("  2. BMP Palette Manipulation")
    print("  3. PNG DCT Coefficient Embedding")
    print("  4. Audio-Visual Patterns")
    print("  5. Clean (no steganography)")
    print("="*60 + "\n")
    
    # Initialize classifier
    classifier = StegoClassifier()
    
    # Load dataset
    X, y, filenames = classifier.load_dataset()
    
    if len(X) == 0:
        print("\nNo images found! Please ensure you have:")
        print("  - datasets/[username]_submission_[year]/clean/")
        print("  - datasets/[username]_submission_[year]/stego/")
        print("\nExample structure:")
        print("  datasets/")
        print("    ├── claude_submission_2025/")
        print("    │   ├── clean/")
        print("    │   └── stego/")
        print("    └── grok_submission_2025/")
        print("        ├── clean/")
        print("        └── stego/")
        print("\nRun data_generator.py in a submission directory to create the dataset.")
        return
    
    # Train model
    model, accuracy = classifier.train(X, y)
    
    # Save model
    classifier.save_model()
    
    # Example predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    # Get sample files from any available submission
    sample_files = []
    for submission_dir in classifier.datasets_dir.iterdir():
        if submission_dir.is_dir():
            stego_dir = submission_dir / "stego"
            if stego_dir.exists():
                for img_path in stego_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.png', '.bmp', '.jpg', '.jpeg', '.webp', '.gif']:
                        sample_files.append(img_path)
                        if len(sample_files) >= 5:
                            break
        if len(sample_files) >= 5:
            break
    
    sample_files = sample_files[:5]
    
    for img_path in sample_files:
        if img_path.suffix.lower() in ['.png', '.bmp', '.jpg', '.jpeg', '.webp', '.gif']:
            algorithm, confidence, probs = classifier.predict(img_path)
            print(f"\nFile: {img_path.name}")
            print(f"Predicted: {algorithm} (confidence: {confidence:.4f})")
            print("Probabilities:")
            for algo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {algo}: {prob:.4f}")
    
    print("\n" + "="*60)
    print("Training complete! Model saved as 'stego_classifier_model.pkl'")
    print("="*60)


if __name__ == "__main__":
    main()
