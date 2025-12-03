#!/usr/bin/env python3
"""
Dataset Quality Audit Script
Systematically scans all submission datasets to identify quality issues

Author: Claude (Anthropic)
Date: December 2, 2025
"""

import os
import json
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
import sys

class DatasetQualityAuditor:
    """Comprehensive dataset quality auditing tool"""
    
    def __init__(self, base_path="datasets"):
        self.base_path = Path(base_path)
        self.issues = defaultdict(list)
        self.stats = defaultdict(lambda: defaultdict(int))
        
    def scan_all_datasets(self):
        """Scan all submission datasets"""
        from scripts.starlight_utils import get_submission_dirs
        print("="*80)
        print("DATASET QUALITY AUDIT")
        print("="*80)
        print()
        
        # Find all submission datasets
        submission_dirs = get_submission_dirs(self.base_path)
        
        print(f"Found {len(submission_dirs)} submission datasets:")
        for d in submission_dirs:
            print(f"  - {d.name}")
        print()
        
        # Scan each dataset
        for dataset_dir in submission_dirs:
            self.scan_dataset(dataset_dir)
        
        # Generate report
        self.generate_report()
    
    def scan_dataset(self, dataset_dir):
        """Scan a single dataset for quality issues"""
        dataset_name = dataset_dir.name
        print(f"Scanning {dataset_name}...")
        
        stego_dir = dataset_dir / "stego"
        clean_dir = dataset_dir / "clean"
        
        if not stego_dir.exists():
            self.issues['missing_directories'].append({
                'dataset': dataset_name,
                'missing': 'stego'
            })
            return
        
        if not clean_dir.exists():
            self.issues['missing_directories'].append({
                'dataset': dataset_name,
                'missing': 'clean'
            })
            return
        
        # Scan stego images
        stego_files = list(stego_dir.glob("*.png")) + list(stego_dir.glob("*.bmp"))
        clean_files = list(clean_dir.glob("*.png")) + list(clean_dir.glob("*.bmp"))
        
        self.stats[dataset_name]['total_stego'] = len(stego_files)
        self.stats[dataset_name]['total_clean'] = len(clean_files)
        
        print(f"  Stego: {len(stego_files)}, Clean: {len(clean_files)}")
        
        # Check for JSON sidecars and validate labels
        for stego_file in stego_files:
            self.validate_stego_file(dataset_name, stego_file)
        
        # Check clean image formats
        self.check_format_distribution(dataset_name, clean_files, stego_files)
        
        print()
    
    def validate_stego_file(self, dataset_name, stego_file):
        """Validate a single stego file and its metadata"""
        json_file = stego_file.with_suffix(stego_file.suffix + '.json')
        
        # Check for JSON sidecar
        if not json_file.exists():
            self.issues['missing_json'].append({
                'dataset': dataset_name,
                'file': str(stego_file.name)
            })
            self.stats[dataset_name]['missing_json'] += 1
            return
        
        # Load and validate JSON
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            self.issues['invalid_json'].append({
                'dataset': dataset_name,
                'file': str(stego_file.name),
                'error': str(e)
            })
            self.stats[dataset_name]['invalid_json'] += 1
            return
        
        # Get technique from metadata
        technique = metadata.get('embedding', {}).get('technique')
        if not technique:
            self.issues['missing_technique'].append({
                'dataset': dataset_name,
                'file': str(stego_file.name)
            })
            self.stats[dataset_name]['missing_technique'] += 1
            return
        
        # Load image to check mode
        try:
            img = Image.open(stego_file)
            img_mode = img.mode
        except Exception as e:
            self.issues['image_load_error'].append({
                'dataset': dataset_name,
                'file': str(stego_file.name),
                'error': str(e)
            })
            self.stats[dataset_name]['image_load_error'] += 1
            return
        
        # Critical check: Alpha technique on non-RGBA images
        if technique == 'alpha' and img_mode != 'RGBA':
            self.issues['invalid_alpha_label'].append({
                'dataset': dataset_name,
                'file': str(stego_file.name),
                'mode': img_mode,
                'technique': technique
            })
            self.stats[dataset_name]['invalid_alpha_label'] += 1
        
        # Check palette technique on non-palette images
        if technique == 'palette' and img_mode != 'P':
            self.issues['invalid_palette_label'].append({
                'dataset': dataset_name,
                'file': str(stego_file.name),
                'mode': img_mode,
                'technique': technique
            })
            self.stats[dataset_name]['invalid_palette_label'] += 1
        
        # Track technique distribution
        self.stats[dataset_name][f'technique_{technique}'] += 1
    
    def check_format_distribution(self, dataset_name, clean_files, stego_files):
        """Check if clean format distribution matches stego"""
        clean_formats = Counter()
        stego_formats = Counter()
        
        for f in clean_files:
            try:
                img = Image.open(f)
                clean_formats[img.mode] += 1
            except:
                pass
        
        for f in stego_files:
            try:
                img = Image.open(f)
                stego_formats[img.mode] += 1
            except:
                pass
        
        # Store format distributions
        self.stats[dataset_name]['clean_formats'] = dict(clean_formats)
        self.stats[dataset_name]['stego_formats'] = dict(stego_formats)
        
        # Check for mismatches
        for mode, count in stego_formats.items():
            clean_count = clean_formats.get(mode, 0)
            if clean_count < count * 0.5:  # Clean has <50% of stego count
                self.issues['format_imbalance'].append({
                    'dataset': dataset_name,
                    'mode': mode,
                    'stego_count': count,
                    'clean_count': clean_count
                })
    
    def generate_report(self):
        """Generate comprehensive quality report"""
        print("\n" + "="*80)
        print("AUDIT RESULTS")
        print("="*80)
        print()
        
        # Summary statistics
        print("SUMMARY STATISTICS")
        print("-" * 80)
        total_stego = sum(stats['total_stego'] for stats in self.stats.values())
        total_clean = sum(stats['total_clean'] for stats in self.stats.values())
        print(f"Total stego images: {total_stego}")
        print(f"Total clean images: {total_clean}")
        print()
        
        # Per-dataset stats
        print("PER-DATASET STATISTICS")
        print("-" * 80)
        for dataset_name, stats in sorted(self.stats.items()):
            print(f"\n{dataset_name}:")
            print(f"  Stego: {stats['total_stego']}, Clean: {stats['total_clean']}")
            
            # Technique distribution
            techniques = {k.replace('technique_', ''): v 
                         for k, v in stats.items() if k.startswith('technique_')}
            if techniques:
                print(f"  Techniques: {techniques}")
            
            # Format distribution
            if 'clean_formats' in stats:
                print(f"  Clean formats: {stats['clean_formats']}")
            if 'stego_formats' in stats:
                print(f"  Stego formats: {stats['stego_formats']}")
        
        print()
        
        # Critical issues
        print("\nCRITICAL ISSUES")
        print("-" * 80)
        
        critical_count = 0
        
        if 'invalid_alpha_label' in self.issues:
            count = len(self.issues['invalid_alpha_label'])
            critical_count += count
            print(f"\n⚠️  INVALID ALPHA LABELS: {count} files")
            print("    Alpha technique labeled on non-RGBA images")
            if count <= 10:
                for issue in self.issues['invalid_alpha_label']:
                    print(f"      - {issue['dataset']}/{issue['file']} (mode: {issue['mode']})")
            else:
                print(f"      Showing first 10 of {count}:")
                for issue in self.issues['invalid_alpha_label'][:10]:
                    print(f"      - {issue['dataset']}/{issue['file']} (mode: {issue['mode']})")
        
        if 'invalid_palette_label' in self.issues:
            count = len(self.issues['invalid_palette_label'])
            critical_count += count
            print(f"\n⚠️  INVALID PALETTE LABELS: {count} files")
            print("    Palette technique labeled on non-palette images")
            if count <= 10:
                for issue in self.issues['invalid_palette_label']:
                    print(f"      - {issue['dataset']}/{issue['file']} (mode: {issue['mode']})")
            else:
                print(f"      Showing first 10 of {count}:")
                for issue in self.issues['invalid_palette_label'][:10]:
                    print(f"      - {issue['dataset']}/{issue['file']} (mode: {issue['mode']})")
        
        if critical_count == 0:
            print("\n✅ No critical issues found")

        # Warnings
        print("\n\nWARNINGS")
        print("-" * 80)

        warning_count = 0

        if 'format_imbalance' in self.issues:
            count = len(self.issues['format_imbalance'])
            warning_count += count
            print(f"\n🟡  FORMAT IMBALANCES: {count} cases")
            print("    Clean format distribution may not match stego (can be normal).")
            # This can be a normal side-effect of steganography, e.g.,
            # applying an alpha technique to an RGB image results in an RGBA stego image.
            for issue in self.issues['format_imbalance']:
                print(f"      - {issue['dataset']}: {issue['mode']} "
                      f"(stego: {issue['stego_count']}, clean: {issue['clean_count']})")

        if warning_count == 0:
            print("\n  No warnings found")

        # Other issues
        print("\n\nOTHER ISSUES")
        print("-" * 80)
        
        other_count = 0
        
        if 'missing_json' in self.issues:
            count = len(self.issues['missing_json'])
            other_count += count
            print(f"\n  Missing JSON sidecars: {count} files")
        
        if 'invalid_json' in self.issues:
            count = len(self.issues['invalid_json'])
            other_count += count
            print(f"  Invalid JSON files: {count} files")
        
        if 'missing_technique' in self.issues:
            count = len(self.issues['missing_technique'])
            other_count += count
            print(f"  Missing technique field: {count} files")
        
        if 'image_load_error' in self.issues:
            count = len(self.issues['image_load_error'])
            other_count += count
            print(f"  Image load errors: {count} files")
        
        if other_count == 0:
            print("\n  No other issues found")
        
        print()
        print("="*80)
        print(f"TOTAL CRITICAL ISSUES: {critical_count}")
        print(f"TOTAL WARNINGS: {warning_count}")
        print(f"TOTAL OTHER ISSUES: {other_count}")
        print("="*80)
        
        # Save detailed report
        self.save_json_report()
    
    def save_json_report(self):
        """Save detailed report as JSON"""
        report = {
            'statistics': dict(self.stats),
            'issues': dict(self.issues)
        }
        
        output_dir = Path("docs/claude")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "dataset_quality_audit.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {output_file}")


if __name__ == "__main__":
    auditor = DatasetQualityAuditor()
    auditor.scan_all_datasets()
