#!/usr/bin/env python3
"""
Advanced Steganography Checker
Properly detects all steganography methods including:
- PNG alpha channel LSB
- JPEG EXIF metadata
- JPEG EOI trailing data
- GIF/BMP palette indexing
"""

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

class AdvancedStegoChecker:
    """Check for steganography using multiple detection methods"""
    
    def __init__(self, dataset_root='datasets'):
        self.dataset_root = Path(dataset_root)
    
    def check_file_metadata(self, clean_path, stego_path):
        """Check file-level differences (size, EXIF, EOI)"""
        differences = {}
        
        # File size comparison
        clean_size = clean_path.stat().st_size
        stego_size = stego_path.stat().st_size
        differences['file_size_diff'] = stego_size - clean_size
        
        # For JPEG: check EXIF and EOI
        if clean_path.suffix.lower() in ['.jpg', '.jpeg']:
            clean_exif = self._get_exif_size(clean_path)
            stego_exif = self._get_exif_size(stego_path)
            differences['exif_size_diff'] = stego_exif - clean_exif
            
            # Check for EOI trailing data
            clean_eoi = self._check_eoi_data(clean_path)
            stego_eoi = self._check_eoi_data(stego_path)
            differences['clean_has_eoi_data'] = clean_eoi
            differences['stego_has_eoi_data'] = stego_eoi
        
        return differences
    
    def _get_exif_size(self, img_path):
        """Get EXIF data size"""
        try:
            with Image.open(img_path) as img:
                exif = img.getexif()
                if exif:
                    # Estimate EXIF size by serializing
                    exif_bytes = img.info.get('exif', b'')
                    return len(exif_bytes)
        except:
            pass
        return 0
    
    def _check_eoi_data(self, img_path):
        """Check if JPEG has data after EOI marker (0xFFD9)"""
        try:
            with open(img_path, 'rb') as f:
                data = f.read()
                # Find last occurrence of EOI marker
                eoi_marker = b'\xFF\xD9'
                eoi_pos = data.rfind(eoi_marker)
                
                if eoi_pos != -1:
                    # Check if there's data after EOI (not just whitespace)
                    after_eoi = data[eoi_pos + 2:]
                    # Ignore trailing nulls/whitespace
                    after_eoi = after_eoi.strip(b'\x00\x20\x0a\x0d')
                    return len(after_eoi) > 0
        except:
            pass
        return False
    
    def check_alpha_channel(self, clean_img, stego_img):
        """Check for alpha channel steganography"""
        if clean_img.mode != 'RGBA' or stego_img.mode != 'RGBA':
            return None
        
        clean_array = np.array(clean_img)
        stego_array = np.array(stego_img)
        
        # Compare alpha channels only
        clean_alpha = clean_array[:, :, 3]
        stego_alpha = stego_array[:, :, 3]
        
        alpha_diff = np.abs(clean_alpha.astype(np.int32) - stego_alpha.astype(np.int32))
        
        return {
            'alpha_changed_pixels': int(np.sum(alpha_diff > 0)),
            'alpha_mean_diff': float(np.mean(alpha_diff)),
            'alpha_max_diff': int(np.max(alpha_diff)),
            'alpha_percent_changed': float(np.sum(alpha_diff > 0) / alpha_diff.size * 100)
        }
    
    def check_palette_indices(self, clean_path, stego_path):
        """Check for palette index steganography (GIF/BMP)"""
        try:
            clean_img = Image.open(clean_path)
            stego_img = Image.open(stego_path)
            
            if clean_img.mode != 'P' or stego_img.mode != 'P':
                return None
            
            # Get palette indices directly
            clean_indices = np.array(clean_img)
            stego_indices = np.array(stego_img)
            
            index_diff = np.abs(clean_indices.astype(np.int32) - stego_indices.astype(np.int32))
            
            return {
                'palette_changed_indices': int(np.sum(index_diff > 0)),
                'palette_mean_diff': float(np.mean(index_diff)),
                'palette_max_diff': int(np.max(index_diff)),
                'palette_percent_changed': float(np.sum(index_diff > 0) / index_diff.size * 100)
            }
        except:
            return None
    
    def comprehensive_check(self, clean_path, stego_path):
        """Comprehensive steganography detection"""
        result = {
            'clean_path': str(clean_path),
            'stego_path': str(stego_path),
            'has_steganography': False,
            'stego_methods': [],
            'details': {}
        }
        
        try:
            # 1. File-level checks (EXIF, EOI, file size)
            file_diff = self.check_file_metadata(clean_path, stego_path)
            result['details']['file_metadata'] = file_diff
            
            if file_diff.get('file_size_diff', 0) != 0:
                result['has_steganography'] = True
                
                if file_diff.get('exif_size_diff', 0) > 0:
                    result['stego_methods'].append('JPEG EXIF metadata')
                
                if file_diff.get('stego_has_eoi_data', False) and not file_diff.get('clean_has_eoi_data', False):
                    result['stego_methods'].append('JPEG EOI trailing data')
            
            # 2. Load images for pixel-level checks
            clean_img = Image.open(clean_path)
            stego_img = Image.open(stego_path)
            
            # 3. Check alpha channel (PNG alpha LSB)
            alpha_result = self.check_alpha_channel(clean_img, stego_img)
            if alpha_result and alpha_result['alpha_changed_pixels'] > 0:
                result['has_steganography'] = True
                result['stego_methods'].append('PNG alpha channel LSB')
                result['details']['alpha'] = alpha_result
            
            # 4. Check palette indices (GIF/BMP)
            palette_result = self.check_palette_indices(clean_path, stego_path)
            if palette_result and palette_result['palette_changed_indices'] > 0:
                result['has_steganography'] = True
                result['stego_methods'].append('GIF/BMP palette indexing')
                result['details']['palette'] = palette_result
            
            # 5. Check RGB pixels (standard LSB/DCT)
            clean_array = np.array(clean_img.convert('RGB'))
            stego_array = np.array(stego_img.convert('RGB'))
            
            if clean_array.shape == stego_array.shape:
                pixel_diff = np.abs(clean_array.astype(np.int32) - stego_array.astype(np.int32))
                mean_diff = np.mean(pixel_diff)
                max_diff = np.max(pixel_diff)
                changed_pixels = np.sum(pixel_diff > 0)
                
                result['details']['pixels'] = {
                    'mean_diff': float(mean_diff),
                    'max_diff': int(max_diff),
                    'changed_pixels': int(changed_pixels),
                    'percent_changed': float(changed_pixels / pixel_diff.size * 100)
                }
                
                if changed_pixels > 0:
                    result['has_steganography'] = True
                    
                    if max_diff <= 1:
                        result['stego_methods'].append('RGB LSB')
                    elif max_diff <= 20:
                        result['stego_methods'].append('RGB DCT/frequency domain')
                    else:
                        result['stego_methods'].append('RGB unknown method')
            
            # If no steganography detected but files are different
            if not result['has_steganography'] and file_diff.get('file_size_diff', 0) != 0:
                result['has_steganography'] = True
                result['stego_methods'].append('Unknown file-level steganography')
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def check_submission(self, submission_dir):
        """Check all pairs in a submission"""
        clean_dir = submission_dir / 'clean'
        stego_dir = submission_dir / 'stego'
        
        if not clean_dir.exists() or not stego_dir.exists():
            return None
        
        results = {
            'submission': submission_dir.name,
            'pairs': [],
            'statistics': {
                'total': 0,
                'with_steganography': 0,
                'without_steganography': 0,
                'errors': 0,
                'methods': {}
            }
        }
        
        print(f"\nChecking {submission_dir.name}...")
        clean_files = list(clean_dir.glob('*.*'))
        
        for clean_path in tqdm(clean_files):
            results['statistics']['total'] += 1
            
            # Find corresponding stego
            stego_path = None
            for ext in ['.png', '.bmp', '.jpg', '.jpeg', '.webp', '.gif']:
                test_path = stego_dir / (clean_path.stem + ext)
                if test_path.exists():
                    stego_path = test_path
                    break
            
            if stego_path is None:
                results['statistics']['errors'] += 1
                continue
            
            # Comprehensive check
            check_result = self.comprehensive_check(clean_path, stego_path)
            results['pairs'].append(check_result)
            
            if check_result.get('error'):
                results['statistics']['errors'] += 1
            elif check_result['has_steganography']:
                results['statistics']['with_steganography'] += 1
                
                # Count methods
                for method in check_result['stego_methods']:
                    results['statistics']['methods'][method] = \
                        results['statistics']['methods'].get(method, 0) + 1
            else:
                results['statistics']['without_steganography'] += 1
        
        return results
    
    def check_all(self):
        """Check entire dataset"""
        print("="*60)
        print("Advanced Steganography Checker - Project Starlight")
        print("="*60)
        
        submission_dirs = [d for d in self.dataset_root.glob('*') 
                          if d.is_dir() and not d.name.startswith('_')]
        
        all_results = []
        
        for submission_dir in submission_dirs:
            result = self.check_submission(submission_dir)
            if result:
                all_results.append(result)
        
        # Print summary
        self.print_summary(all_results)
        
        # Save detailed report
        self.save_report(all_results)
        
        return all_results
    
    def print_summary(self, results):
        """Print summary of findings"""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        total_pairs = 0
        total_with_stego = 0
        total_without_stego = 0
        total_errors = 0
        all_methods = {}
        
        for sub_result in results:
            stats = sub_result['statistics']
            
            print(f"\n{sub_result['submission']}:")
            print(f"  Total pairs: {stats['total']}")
            print(f"  With steganography: {stats['with_steganography']} ({stats['with_steganography']/max(stats['total'],1)*100:.1f}%)")
            print(f"  WITHOUT steganography (IDENTICAL): {stats['without_steganography']} ({stats['without_steganography']/max(stats['total'],1)*100:.1f}%)")
            print(f"  Errors: {stats['errors']}")
            
            if stats['methods']:
                print(f"  Methods detected:")
                for method, count in stats['methods'].items():
                    print(f"    - {method}: {count}")
                    all_methods[method] = all_methods.get(method, 0) + count
            
            total_pairs += stats['total']
            total_with_stego += stats['with_steganography']
            total_without_stego += stats['without_steganography']
            total_errors += stats['errors']
        
        print("\n" + "="*60)
        print("OVERALL STATISTICS")
        print("="*60)
        print(f"Total pairs: {total_pairs}")
        print(f"Valid steganography: {total_with_stego} ({total_with_stego/max(total_pairs,1)*100:.1f}%)")
        print(f"IDENTICAL (no stego): {total_without_stego} ({total_without_stego/max(total_pairs,1)*100:.1f}%)")
        print(f"Errors: {total_errors}")
        
        print("\n🔍 Steganography Methods Found:")
        for method, count in sorted(all_methods.items(), key=lambda x: -x[1]):
            print(f"  • {method}: {count} pairs")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if total_without_stego > 0:
            print(f"\n⚠️  {total_without_stego} pairs are IDENTICAL - no steganography detected!")
            print("   These should be removed or regenerated.")
        
        if total_with_stego >= 1000:
            print(f"\n✓ {total_with_stego} valid steganography pairs - excellent!")
        elif total_with_stego >= 500:
            print(f"\n✓ {total_with_stego} valid steganography pairs - good for training")
        else:
            print(f"\n⚠️  Only {total_with_stego} valid pairs - need more data")
    
    def save_report(self, results, filename='advanced_stego_check.json'):
        """Save detailed report"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Detailed report saved to: {filename}")


def main():
    checker = AdvancedStegoChecker(dataset_root='datasets')
    results = checker.check_all()


if __name__ == "__main__":
    main()
