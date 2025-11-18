#!/usr/bin/env python3
"""
Dataset Repair Pipeline - Week 1, Day 2
Fixes fundamental dataset quality issues

Run: python scripts/dataset_repair.py --datasets datasets --output datasets/grok_submission_2025/training/v3_repaired --dry-run
"""

import os
import shutil
import argparse
import json
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
import hashlib

@dataclass
class RepairAction:
    """Represents a repair action to be taken"""
    image_path: str
    action_type: str  # 'remove_label', 'change_label', 'remove_image', 'keep'
    old_label: Optional[str]
    new_label: Optional[str]
    reason: str

class DatasetRepairer:
    """Fix fundamental dataset quality issues"""
    
    def __init__(self, source_dir: Path, output_dir: Path, dry_run: bool = True):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.dry_run = dry_run
        self.repair_actions: List[RepairAction] = []
        self.stats = {
            'total_processed': 0,
            'labels_removed': 0,
            'labels_changed': 0,
            'images_removed': 0,
            'kept_unchanged': 0
        }
    
    def validate_alpha_label(self, img_path: Path, label: str) -> Optional[str]:
        """
        Validate alpha steganography label
        
        Returns:
            None if label is invalid, otherwise returns the label
        """
        if label != 'alpha':
            return label
        
        try:
            with Image.open(img_path) as img:
                mode = img.mode
                
                # Alpha steganography requires alpha channel
                if mode not in ['RGBA', 'LA', 'PA']:
                    self.repair_actions.append(
                        RepairAction(
                            image_path=str(img_path),
                            action_type='remove_label',
                            old_label='alpha',
                            new_label=None,
                            reason=f"Alpha label on {mode} image (no alpha channel)"
                        )
                    )
                    return None
                
                # Additional check: verify alpha channel has variation
                if mode == 'RGBA':
                    # Sample the alpha channel
                    img_array = list(img.getdata())
                    alpha_values = [pixel[3] for pixel in img_array[:1000]]  # Sample first 1000
                    
                    # Check if alpha is uniform (all same value)
                    unique_alphas = set(alpha_values)
                    if len(unique_alphas) == 1:
                        self.repair_actions.append(
                            RepairAction(
                                image_path=str(img_path),
                                action_type='remove_label',
                                old_label='alpha',
                                new_label=None,
                                reason=f"Alpha channel is uniform (value={alpha_values[0]}), no hidden data"
                            )
                        )
                        return None
                
                return label
                
        except Exception as e:
            self.repair_actions.append(
                RepairAction(
                    image_path=str(img_path),
                    action_type='remove_image',
                    old_label=label,
                    new_label=None,
                    reason=f"Failed to open image: {str(e)}"
                )
            )
            return None
    
    def validate_palette_label(self, img_path: Path, label: str) -> Optional[str]:
        """Validate palette steganography label"""
        if label != 'palette':
            return label
        
        try:
            with Image.open(img_path) as img:
                mode = img.mode
                
                # Palette steganography requires palette mode
                if mode not in ['P', 'PA']:
                    # Check if this might actually be LSB
                    if mode in ['RGB', 'RGBA']:
                        self.repair_actions.append(
                            RepairAction(
                                image_path=str(img_path),
                                action_type='change_label',
                                old_label='palette',
                                new_label='lsb',
                                reason=f"Palette label on {mode} image - likely should be LSB"
                            )
                        )
                        return 'lsb'
                    else:
                        self.repair_actions.append(
                            RepairAction(
                                image_path=str(img_path),
                                action_type='remove_label',
                                old_label='palette',
                                new_label=None,
                                reason=f"Palette label on {mode} image (no palette)"
                            )
                        )
                        return None
                
                return label
                
        except Exception as e:
            self.repair_actions.append(
                RepairAction(
                    image_path=str(img_path),
                    action_type='remove_image',
                    old_label=label,
                    new_label=None,
                    reason=f"Failed to open image: {str(e)}"
                )
            )
            return None
    
    def verify_extraction(self, img_path: Path, method: str) -> bool:
        """
        Verify that steganography can actually be extracted
        
        Note: This is a placeholder. In production, you'd call actual
        extraction functions here.
        """
        # TODO: Implement actual extraction verification
        # For now, just check basic format compatibility
        
        try:
            with Image.open(img_path) as img:
                mode = img.mode
                img_format = img.format
                
                # Check format compatibility
                if method == 'lsb' and img_format == 'JPEG':
                    # JPEG is lossy, LSB likely corrupted
                    self.repair_actions.append(
                        RepairAction(
                            image_path=str(img_path),
                            action_type='remove_label',
                            old_label=method,
                            new_label=None,
                            reason="LSB on lossy JPEG - data likely corrupted"
                        )
                    )
                    return False
                
                # More checks would go here with actual extraction
                return True
                
        except:
            return False
    
    def balance_format_distribution(self, clean_dir: Path, stego_dir: Path) -> Dict[str, int]:
        """
        Ensure clean images match stego format distribution
        
        Returns:
            Dictionary of formats needing more clean images
        """
        def count_formats(directory: Path) -> Counter:
            """Count image formats in directory"""
            format_counts = Counter()
            
            if not directory.exists():
                return format_counts
            
            for img_path in directory.rglob('*'):
                if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}:
                    try:
                        with Image.open(img_path) as img:
                            format_counts[img.format] += 1
                    except:
                        pass
            
            return format_counts
        
        stego_formats = count_formats(stego_dir)
        clean_formats = count_formats(clean_dir)
        
        shortage = {}
        for fmt, stego_count in stego_formats.items():
            clean_count = clean_formats.get(fmt, 0)
            if clean_count < stego_count:
                shortage[fmt] = stego_count - clean_count
        
        return shortage
    
    def process_dataset(self):
        """Process and repair entire dataset"""
        print(f"Processing dataset: {self.source_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Dry run: {self.dry_run}")
        
        # Find dataset directories
        dataset_dirs = list(self.source_dir.glob("*_submission_*"))
        if not dataset_dirs:
            dataset_dirs = [self.source_dir]
        
        for dataset_dir in dataset_dirs:
            print(f"\nProcessing {dataset_dir.name}...")
            
            # Process stego images
            stego_dir = dataset_dir / "stego"
            if stego_dir.exists():
                self._process_stego_directory(stego_dir)
            
            # Process clean images
            clean_dir = dataset_dir / "clean"
            if clean_dir.exists():
                self._process_clean_directory(clean_dir)
            
            # Check format balance
            if stego_dir.exists() and clean_dir.exists():
                shortage = self.balance_format_distribution(clean_dir, stego_dir)
                if shortage:
                    print(f"\n⚠️  Format imbalance detected:")
                    for fmt, count in shortage.items():
                        print(f"  Need {count} more clean {fmt} images")
        
        self._print_summary()
        
        if not self.dry_run:
            self._apply_repairs()
    
    def _process_stego_directory(self, stego_dir: Path):
        """Process stego images and validate labels"""
        methods = ['alpha', 'lsb', 'palette', 'exif', 'eoi']
        
        for method in methods:
            method_dir = stego_dir / method
            if not method_dir.exists():
                continue
            
            print(f"  Processing {method} images...")
            
            for img_path in method_dir.rglob('*'):
                if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}:
                    self.stats['total_processed'] += 1
                    
                    # Validate label based on method
                    if method == 'alpha':
                        new_label = self.validate_alpha_label(img_path, method)
                    elif method == 'palette':
                        new_label = self.validate_palette_label(img_path, method)
                    else:
                        new_label = method
                    
                    # Verify extraction (placeholder for now)
                    if new_label and not self.verify_extraction(img_path, new_label):
                        new_label = None
                    
                    # Record statistics
                    if new_label is None:
                        self.stats['labels_removed'] += 1
                    elif new_label != method:
                        self.stats['labels_changed'] += 1
                    else:
                        self.stats['kept_unchanged'] += 1
    
    def _process_clean_directory(self, clean_dir: Path):
        """Process clean images"""
        print(f"  Processing clean images...")
        
        for img_path in clean_dir.rglob('*'):
            if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}:
                self.stats['total_processed'] += 1
                self.stats['kept_unchanged'] += 1
    
    def _apply_repairs(self):
        """Apply all repair actions to create repaired dataset"""
        print(f"\nApplying repairs...")
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and repair files
        for action in self.repair_actions:
            src_path = Path(action.image_path)
            
            if action.action_type == 'remove_image':
                # Don't copy this image
                continue
            
            elif action.action_type == 'remove_label':
                # Move to clean directory
                rel_path = src_path.relative_to(self.source_dir)
                dst_path = self.output_dir / 'clean' / rel_path.name
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
            
            elif action.action_type == 'change_label':
                # Move to new label directory
                rel_path = src_path.relative_to(self.source_dir)
                dst_path = self.output_dir / 'stego' / action.new_label / rel_path.name
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
            
            elif action.action_type == 'keep':
                # Copy unchanged
                rel_path = src_path.relative_to(self.source_dir)
                dst_path = self.output_dir / rel_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
        
        # Write manifest
        self._write_manifest()
        
        print(f"✅ Repairs applied to {self.output_dir}")
    
    def _write_manifest(self):
        """Write repair manifest with all actions taken"""
        manifest = {
            'source_directory': str(self.source_dir),
            'output_directory': str(self.output_dir),
            'statistics': self.stats,
            'actions': [asdict(action) for action in self.repair_actions]
        }
        
        manifest_path = self.output_dir / 'repair_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Manifest written to {manifest_path}")
    
    def _print_summary(self):
        """Print repair summary"""
        print(f"\n{'='*70}")
        print("REPAIR SUMMARY")
        print("="*70)
        print(f"\nTotal Images Processed: {self.stats['total_processed']}")
        print(f"  Kept Unchanged: {self.stats['kept_unchanged']}")
        print(f"  Labels Removed: {self.stats['labels_removed']}")
        print(f"  Labels Changed: {self.stats['labels_changed']}")
        print(f"  Images Removed: {self.stats['images_removed']}")
        
        if self.repair_actions:
            print(f"\nTop Issues:")
            issue_counts = Counter(action.reason for action in self.repair_actions)
            for reason, count in issue_counts.most_common(5):
                print(f"  {count:4d}x {reason}")
        
        print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description='Repair dataset quality issues'
    )
    parser.add_argument(
        '--datasets',
        required=True,
        help='Source datasets directory'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for repaired dataset'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    repairer = DatasetRepairer(
        source_dir=Path(args.datasets),
        output_dir=Path(args.output),
        dry_run=args.dry_run
    )
    
    repairer.process_dataset()
    
    if args.dry_run:
        print("\n💡 This was a dry run. Use --no-dry-run to apply changes.")
    
    return 0

if __name__ == '__main__':
    exit(main())
