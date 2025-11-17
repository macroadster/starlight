#!/usr/bin/env python3
"""
Label Validation Script - Week 1, Day 1
Checks for impossible label assignments (e.g., alpha on RGB images)

Run: python scripts/validate_labels.py --datasets "datasets/*_submission_*/stego" --report docs/claude/invalid_labels.md
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple
from dataclasses import dataclass
import glob

@dataclass
class InvalidLabel:
    """Represents an invalid label assignment"""
    image_path: str
    assigned_label: str
    image_mode: str
    reason: str
    suggestion: str

class LabelValidator:
    """Validates steganography method labels against image properties"""
    
    # Define valid image modes for each steganography method
    VALID_MODES = {
        'alpha': {'RGBA', 'LA', 'PA'},  # Requires alpha channel
        'lsb': {'RGB', 'RGBA', 'L'},    # Works on any pixel-based format
        'palette': {'P', 'PA'},          # Requires palette/indexed color
        'exif': {'RGB', 'RGBA', 'L', 'P'},  # Works with JPEG metadata
        'eoi': {'RGB', 'RGBA', 'L', 'P'}    # Works with JPEG end-of-image
    }
    
    VALID_FORMATS = {
        'alpha': {'PNG', 'TIFF', 'WEBP'},  # Formats supporting alpha
        'lsb': {'PNG', 'BMP', 'TIFF'},     # Lossless formats
        'palette': {'PNG', 'GIF'},          # Formats with palette support
        'exif': {'JPEG', 'TIFF'},           # EXIF metadata support
        'eoi': {'JPEG'}                     # JPEG-specific
    }
    
    def __init__(self):
        self.invalid_labels: List[InvalidLabel] = []
    
    def validate_image(self, img_path: Path, method: str) -> bool:
        """
        Validate that the steganography method is compatible with the image
        
        Returns:
            True if valid, False if invalid
        """
        try:
            with Image.open(img_path) as img:
                mode = img.mode
                img_format = img.format or 'UNKNOWN'
                
                is_valid = True
                reasons = []
                suggestions = []
                
                # Check mode compatibility
                valid_modes = self.VALID_MODES.get(method, set())
                if valid_modes and mode not in valid_modes:
                    is_valid = False
                    reasons.append(
                        f"Method '{method}' requires image mode in {valid_modes}, but image has mode '{mode}'"
                    )
                    
                    # Provide specific suggestions
                    if method == 'alpha' and mode == 'RGB':
                        suggestions.append(
                            "Either: (1) Remove this label - RGB images cannot use alpha steganography, "
                            "or (2) Verify if image was mislabeled and should be 'lsb' instead"
                        )
                    elif method == 'palette' and mode in {'RGB', 'RGBA'}:
                        suggestions.append(
                            "Either: (1) Remove this label - true-color images don't use palette steganography, "
                            "or (2) Verify if this should be labeled as 'lsb'"
                        )
                    else:
                        suggestions.append(f"Verify label or convert image to compatible mode")
                
                # Check format compatibility
                valid_formats = self.VALID_FORMATS.get(method, set())
                if valid_formats and img_format not in valid_formats:
                    # Warning rather than hard error for format
                    reasons.append(
                        f"WARNING: Method '{method}' typically uses formats {valid_formats}, "
                        f"but image is '{img_format}'"
                    )
                    
                    if method in ['lsb', 'alpha'] and img_format == 'JPEG':
                        suggestions.append(
                            "JPEG's lossy compression may corrupt LSB/alpha data. "
                            "Consider removing or re-verifying this sample."
                        )
                
                # Check specific constraints
                if method == 'alpha':
                    # Verify alpha channel actually exists
                    if not img.mode.endswith('A'):
                        is_valid = False
                        reasons.append(
                            f"Alpha steganography labeled but image mode '{mode}' has no alpha channel"
                        )
                        suggestions.append("This is the most critical type of error - must be fixed")
                
                if method == 'palette':
                    # Verify palette exists
                    if not hasattr(img, 'palette') or img.palette is None:
                        is_valid = False
                        reasons.append(
                            f"Palette steganography labeled but image has no palette data"
                        )
                        suggestions.append("Remove this label or verify image format")
                
                if not is_valid or reasons:
                    self.invalid_labels.append(
                        InvalidLabel(
                            image_path=str(img_path),
                            assigned_label=method,
                            image_mode=mode,
                            reason="; ".join(reasons),
                            suggestion="; ".join(suggestions) if suggestions else "Verify label"
                        )
                    )
                
                return is_valid
                
        except Exception as e:
            self.invalid_labels.append(
                InvalidLabel(
                    image_path=str(img_path),
                    assigned_label=method,
                    image_mode='ERROR',
                    reason=f"Failed to open/analyze image: {str(e)}",
                    suggestion="Check if file is corrupted or in unsupported format"
                )
            )
            return False
    
    def validate_directory(self, dir_path: Path, method: str = None) -> Tuple[int, int]:
        """
        Validate all images in a directory
        
        Args:
            dir_path: Directory containing images
            method: Steganography method (if None, inferred from directory name)
        
        Returns:
            Tuple of (total_images, invalid_count)
        """
        if not dir_path.exists():
            print(f"Warning: Directory {dir_path} does not exist")
            return 0, 0
        
        # Infer method from directory name if not provided
        if method is None:
            method = dir_path.name
            if method not in self.VALID_MODES:
                print(f"Warning: Could not infer method from directory '{method}'")
                return 0, 0
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        
        total = 0
        invalid = 0
        
        for img_path in dir_path.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                total += 1
                if not self.validate_image(img_path, method):
                    invalid += 1
        
        return total, invalid
    
    def validate_datasets(self, patterns: List[str]) -> Dict[str, Tuple[int, int]]:
        """
        Validate multiple dataset directories
        
        Args:
            patterns: List of glob patterns for dataset directories
        
        Returns:
            Dictionary mapping directory path to (total, invalid) counts
        """
        results = {}
        
        for pattern in patterns:
            for dir_path in glob.glob(pattern):
                dir_path = Path(dir_path)
                
                # Check if this is a method-specific directory
                if dir_path.name in self.VALID_MODES:
                    method = dir_path.name
                    total, invalid = self.validate_directory(dir_path, method)
                    results[str(dir_path)] = (total, invalid)
                    print(f"Validated {total} images in {dir_path}, found {invalid} invalid labels")
                else:
                    # Check for subdirectories by method
                    for method in self.VALID_MODES.keys():
                        method_dir = dir_path / method
                        if method_dir.exists():
                            total, invalid = self.validate_directory(method_dir, method)
                            results[str(method_dir)] = (total, invalid)
                            print(f"Validated {total} images in {method_dir}, found {invalid} invalid labels")
        
        return results
    
    def generate_report(self, output_path: Path):
        """Generate markdown report of invalid labels"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# Invalid Label Report\n\n")
            f.write(f"**Generated:** {Path.cwd()}\n\n")
            f.write(f"**Total Invalid Labels:** {len(self.invalid_labels)}\n\n")
            
            f.write("---\n\n")
            
            if not self.invalid_labels:
                f.write("✅ **No invalid labels found!**\n\n")
                f.write("All steganography method labels are compatible with their image formats.\n")
                return
            
            # Group by method
            by_method = {}
            for invalid in self.invalid_labels:
                method = invalid.assigned_label
                if method not in by_method:
                    by_method[method] = []
                by_method[method].append(invalid)
            
            f.write("## Summary by Method\n\n")
            for method, invalids in sorted(by_method.items()):
                f.write(f"- **{method}**: {len(invalids)} invalid labels\n")
            f.write("\n---\n\n")
            
            # Detailed listings
            f.write("## Detailed Findings\n\n")
            
            for method, invalids in sorted(by_method.items()):
                f.write(f"### Method: `{method}`\n\n")
                f.write(f"**Invalid Labels:** {len(invalids)}\n\n")
                
                for i, invalid in enumerate(invalids, 1):
                    f.write(f"#### {i}. `{invalid.image_path}`\n\n")
                    f.write(f"- **Image Mode:** `{invalid.image_mode}`\n")
                    f.write(f"- **Assigned Label:** `{invalid.assigned_label}`\n")
                    f.write(f"- **Issue:** {invalid.reason}\n")
                    f.write(f"- **Suggestion:** {invalid.suggestion}\n\n")
                
                f.write("\n")
            
            # Action items
            f.write("---\n\n")
            f.write("## Recommended Actions\n\n")
            
            alpha_rgb_count = sum(
                1 for inv in self.invalid_labels 
                if inv.assigned_label == 'alpha' and 'RGB' in inv.image_mode
            )
            
            if alpha_rgb_count > 0:
                f.write(f"### 🚨 CRITICAL: {alpha_rgb_count} Alpha Labels on RGB Images\n\n")
                f.write("These are impossible and must be fixed:\n\n")
                f.write("1. **Option A**: Remove the labels entirely (recommended)\n")
                f.write("2. **Option B**: Verify if images were mislabeled as 'alpha' when they should be 'lsb'\n")
                f.write("3. **Option C**: Remove images from dataset if they cannot be verified\n\n")
            
            palette_errors = sum(
                1 for inv in self.invalid_labels 
                if inv.assigned_label == 'palette' and 'RGB' in inv.image_mode
            )
            
            if palette_errors > 0:
                f.write(f"### ⚠️  {palette_errors} Palette Labels on True-Color Images\n\n")
                f.write("These should be reviewed:\n\n")
                f.write("1. Verify if these should be labeled as 'lsb' instead\n")
                f.write("2. Check if palette indices were embedded during conversion\n\n")
            
            f.write("### Next Steps\n\n")
            f.write("1. Run `scripts/dataset_repair.py` to automatically fix invalid labels\n")
            f.write("2. Manually review suggested changes before applying\n")
            f.write("3. Re-run validation to confirm all issues resolved\n\n")
        
        print(f"\nReport generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Validate steganography labels against image properties'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='Glob patterns for dataset directories (e.g., "datasets/*_submission_*/stego")'
    )
    parser.add_argument(
        '--report',
        default='docs/claude/invalid_labels.md',
        help='Output path for markdown report'
    )
    
    args = parser.parse_args()
    
    validator = LabelValidator()
    
    print("Validating labels...")
    results = validator.validate_datasets(args.datasets)
    
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print("="*70)
    
    total_images = sum(r[0] for r in results.values())
    total_invalid = sum(r[1] for r in results.values())
    
    print(f"\nTotal Images Validated: {total_images}")
    print(f"Total Invalid Labels: {total_invalid}")
    
    if total_invalid > 0:
        print(f"\n⚠️  Found {total_invalid} invalid labels ({(total_invalid/total_images)*100:.1f}%)")
    else:
        print("\n✅ All labels are valid!")
    
    # Generate report
    validator.generate_report(Path(args.report))
    
    # Return error code if issues found
    return 1 if total_invalid > 0 else 0

if __name__ == '__main__':
    exit(main())
