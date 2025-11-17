#!/usr/bin/env python3
"""
Dataset Analysis Script - Week 1, Day 1
Analyzes current datasets to identify fundamental quality issues

Run: python scripts/analyze_datasets.py --output docs/claude/dataset_audit.json
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

@dataclass
class ImageAnalysis:
    """Detailed analysis of a single image"""
    path: str
    mode: str  # RGB, RGBA, P (palette), L (grayscale)
    size: Tuple[int, int]
    format: str  # PNG, JPEG, GIF, etc.
    has_alpha: bool
    has_palette: bool
    label: Optional[str]  # Method label if stego
    is_valid: bool  # Whether label matches image properties
    issues: List[str]  # List of detected problems

@dataclass
class DatasetStats:
    """Overall dataset statistics"""
    total_images: int
    clean_count: int
    stego_count: int
    mode_distribution: Dict[str, int]
    format_distribution: Dict[str, int]
    method_distribution: Dict[str, int]
    invalid_labels: int
    missing_alpha_channel: int  # Alpha labels on RGB images
    format_mismatches: int
    issues_summary: Dict[str, int]

class DatasetAnalyzer:
    """Analyzes steganography datasets for quality issues"""
    
    STEGO_METHODS = ['alpha', 'lsb', 'palette', 'exif', 'eoi']
    
    def __init__(self, datasets_dir: str):
        self.datasets_dir = Path(datasets_dir)
        self.analyses: List[ImageAnalysis] = []
        self.stats = None
    
    def analyze_image(self, img_path: Path, label: Optional[str] = None) -> ImageAnalysis:
        """Analyze a single image for quality issues"""
        issues = []
        
        try:
            img = Image.open(img_path)
            mode = img.mode
            size = img.size
            img_format = img.format or 'UNKNOWN'
            
            # Check for alpha channel
            has_alpha = mode in ['RGBA', 'LA', 'PA']
            has_palette = mode in ['P', 'PA']
            
            # Validate label against image properties
            is_valid = True
            
            if label == 'alpha':
                if not has_alpha:
                    is_valid = False
                    issues.append(f"INVALID_LABEL: Alpha steganography labeled on {mode} image without alpha channel")
            
            if label == 'palette':
                if not has_palette:
                    is_valid = False
                    issues.append(f"INVALID_LABEL: Palette steganography labeled on {mode} image without palette")
            
            # Check for suspicious characteristics
            if mode == 'P' and label == 'lsb':
                issues.append("WARNING: LSB on palette image may be unreliable")
            
            if img_format == 'JPEG' and label in ['lsb', 'alpha']:
                issues.append("WARNING: JPEG lossy compression may corrupt LSB/alpha data")
            
            # Check image size
            if size[0] < 10 or size[1] < 10:
                issues.append("WARNING: Suspiciously small image dimensions")
            
            return ImageAnalysis(
                path=str(img_path.relative_to(self.datasets_dir)),
                mode=mode,
                size=size,
                format=img_format,
                has_alpha=has_alpha,
                has_palette=has_palette,
                label=label,
                is_valid=is_valid,
                issues=issues
            )
            
        except Exception as e:
            return ImageAnalysis(
                path=str(img_path.relative_to(self.datasets_dir)),
                mode='ERROR',
                size=(0, 0),
                format='ERROR',
                has_alpha=False,
                has_palette=False,
                label=label,
                is_valid=False,
                issues=[f"ERROR: Failed to open image - {str(e)}"]
            )
    
    def scan_directory(self, dir_path: Path, label: Optional[str] = None) -> List[ImageAnalysis]:
        """Scan a directory for images and analyze them"""
        analyses = []
        
        if not dir_path.exists():
            print(f"Warning: Directory {dir_path} does not exist")
            return analyses
        
        # Common image extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        
        for img_path in dir_path.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                analysis = self.analyze_image(img_path, label)
                analyses.append(analysis)
        
        return analyses
    
    def infer_label_from_path(self, path: Path) -> Optional[str]:
        """Infer steganography method from directory structure"""
        path_parts = path.parts
        
        for method in self.STEGO_METHODS:
            if method in path_parts:
                return method
        
        return None
    
    def analyze_all(self) -> DatasetStats:
        """Analyze all datasets and generate statistics"""
        print(f"Scanning datasets in: {self.datasets_dir}")
        
        # Find all dataset directories
        dataset_dirs = list(self.datasets_dir.glob("*_submission_*"))
        
        if not dataset_dirs:
            print("Warning: No dataset directories found matching pattern '*_submission_*'")
            dataset_dirs = [self.datasets_dir]
        
        print(f"Found {len(dataset_dirs)} dataset directories")
        
        for dataset_dir in dataset_dirs:
            print(f"\nAnalyzing dataset: {dataset_dir.name}")
            
            # Analyze clean images
            clean_dir = dataset_dir / "clean"
            if clean_dir.exists():
                clean_analyses = self.scan_directory(clean_dir, label=None)
                self.analyses.extend(clean_analyses)
                print(f"  Clean images: {len(clean_analyses)}")
            
            # Analyze stego images (check for method subdirectories)
            stego_dir = dataset_dir / "stego"
            if stego_dir.exists():
                # Check if there are method subdirectories
                method_dirs = [d for d in stego_dir.iterdir() if d.is_dir()]
                
                if method_dirs:
                    # Organized by method
                    for method_dir in method_dirs:
                        method = method_dir.name
                        if method in self.STEGO_METHODS:
                            stego_analyses = self.scan_directory(method_dir, label=method)
                            self.analyses.extend(stego_analyses)
                            print(f"  Stego images ({method}): {len(stego_analyses)}")
                else:
                    # Flat structure - infer from filenames
                    stego_analyses = self.scan_directory(stego_dir, label='unknown')
                    self.analyses.extend(stego_analyses)
                    print(f"  Stego images (flat): {len(stego_analyses)}")
        
        # Generate statistics
        self.stats = self._generate_stats()
        return self.stats
    
    def _generate_stats(self) -> DatasetStats:
        """Generate summary statistics from analyses"""
        mode_dist = Counter()
        format_dist = Counter()
        method_dist = Counter()
        issues_summary = defaultdict(int)
        
        clean_count = 0
        stego_count = 0
        invalid_labels = 0
        missing_alpha = 0
        format_mismatches = 0
        
        for analysis in self.analyses:
            mode_dist[analysis.mode] += 1
            format_dist[analysis.format] += 1
            
            if analysis.label:
                method_dist[analysis.label] += 1
                stego_count += 1
                
                if not analysis.is_valid:
                    invalid_labels += 1
                
                # Check for specific issues
                if analysis.label == 'alpha' and not analysis.has_alpha:
                    missing_alpha += 1
            else:
                clean_count += 1
            
            # Categorize issues
            for issue in analysis.issues:
                if issue.startswith('INVALID_LABEL'):
                    issues_summary['invalid_labels'] += 1
                elif issue.startswith('WARNING'):
                    issue_type = issue.split(':')[0]
                    issues_summary[issue_type] += 1
                elif issue.startswith('ERROR'):
                    issues_summary['errors'] += 1
        
        return DatasetStats(
            total_images=len(self.analyses),
            clean_count=clean_count,
            stego_count=stego_count,
            mode_distribution=dict(mode_dist),
            format_distribution=dict(format_dist),
            method_distribution=dict(method_dist),
            invalid_labels=invalid_labels,
            missing_alpha_channel=missing_alpha,
            format_mismatches=format_mismatches,
            issues_summary=dict(issues_summary)
        )
    
    def export_report(self, output_path: Path):
        """Export detailed analysis report as JSON"""
        report = {
            'summary': asdict(self.stats) if self.stats else None,
            'analyses': [asdict(a) for a in self.analyses],
            'critical_issues': self._get_critical_issues()
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport exported to: {output_path}")
    
    def _get_critical_issues(self) -> Dict[str, List[str]]:
        """Extract critical issues for immediate attention"""
        critical = defaultdict(list)
        
        for analysis in self.analyses:
            for issue in analysis.issues:
                if 'INVALID_LABEL' in issue:
                    critical['invalid_labels'].append(
                        f"{analysis.path}: {issue}"
                    )
        
        return dict(critical)
    
    def print_summary(self):
        """Print human-readable summary"""
        if not self.stats:
            print("No statistics available. Run analyze_all() first.")
            return
        
        print("\n" + "="*70)
        print("DATASET ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\nTotal Images: {self.stats.total_images}")
        print(f"  Clean: {self.stats.clean_count}")
        print(f"  Stego: {self.stats.stego_count}")
        
        print(f"\nImage Mode Distribution:")
        for mode, count in sorted(self.stats.mode_distribution.items()):
            pct = (count / self.stats.total_images) * 100
            print(f"  {mode:8s}: {count:6d} ({pct:5.1f}%)")
        
        print(f"\nImage Format Distribution:")
        for fmt, count in sorted(self.stats.format_distribution.items()):
            pct = (count / self.stats.total_images) * 100
            print(f"  {fmt:8s}: {count:6d} ({pct:5.1f}%)")
        
        print(f"\nSteganography Method Distribution:")
        for method, count in sorted(self.stats.method_distribution.items()):
            pct = (count / self.stats.stego_count) * 100 if self.stats.stego_count > 0 else 0
            print(f"  {method:8s}: {count:6d} ({pct:5.1f}%)")
        
        print(f"\n{'='*70}")
        print("CRITICAL ISSUES")
        print("="*70)
        
        print(f"\nInvalid Labels: {self.stats.invalid_labels}")
        print(f"  - Alpha labels on RGB: {self.stats.missing_alpha_channel}")
        
        if self.stats.issues_summary:
            print(f"\nIssues Summary:")
            for issue_type, count in sorted(self.stats.issues_summary.items()):
                print(f"  {issue_type}: {count}")
        
        print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(
        description='Analyze steganography datasets for quality issues'
    )
    parser.add_argument(
        '--datasets-dir',
        default='datasets',
        help='Directory containing datasets (default: datasets)'
    )
    parser.add_argument(
        '--output',
        default='docs/claude/dataset_audit.json',
        help='Output path for JSON report (default: docs/claude/dataset_audit.json)'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(args.datasets_dir)
    
    # Run analysis
    print("Starting dataset analysis...")
    stats = analyzer.analyze_all()
    
    # Print summary to console
    analyzer.print_summary()
    
    # Export detailed report
    analyzer.export_report(Path(args.output))
    
    # Exit with error code if critical issues found
    if stats.invalid_labels > 0:
        print(f"\n⚠️  WARNING: {stats.invalid_labels} invalid labels detected!")
        print("Run dataset repair pipeline to fix these issues.")
        sys.exit(1)
    else:
        print("\n✅ No critical issues detected!")
        sys.exit(0)

if __name__ == '__main__':
    main()
