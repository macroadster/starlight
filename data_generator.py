#!/usr/bin/env python3
"""
Top-level data_generator.py

Orchestrates dataset generation across all submissions.
"""

import os
import shutil
import subprocess
import argparse
from pathlib import Path
from PIL import Image

def count_clean_images(clean_dir):
    count = 0
    for p in clean_dir.iterdir():
        if p.is_file() and p.name != '.DS_Store':
            try:
                with Image.open(p) as img:
                    img.verify()
                count += 1
            except:
                pass
    return count

def main():
    parser = argparse.ArgumentParser(description="Top-level dataset generator")
    parser.add_argument('--limit', type=int, default=10, help='Percentage for sample and val limits, or number for other submissions (default 10)')
    args = parser.parse_args()

    # Clean up datasets/*_submission_*/clean and stego, except datasets/sample_submission_2025/clean and datasets/maya_submission_2025
    datasets_dir = Path('datasets')
    for sub_dir in datasets_dir.iterdir():
        if sub_dir.is_dir() and sub_dir.name.endswith('_submission_2025'):
            stego_dir = sub_dir / 'stego'
            if stego_dir.exists():
                shutil.rmtree(stego_dir)
            if sub_dir.name != 'sample_submission_2025' and sub_dir.name != 'maya_submission_2025':
                clean_dir = sub_dir / 'clean'
                if clean_dir.exists():
                    shutil.rmtree(clean_dir)

    # Run data_generator.py --limit N for each submission directory
    for sub_dir in datasets_dir.iterdir():
        if sub_dir.is_dir() and sub_dir.name.endswith('_submission_2025') and sub_dir.name != 'sample_submission_2025':
            os.chdir(sub_dir)
            if sub_dir.name == 'maya_submission_2025':
                subprocess.run(['python', 'data_generator.py'])
            else:
                subprocess.run(['python', 'data_generator.py', '--limit', str(args.limit)])
            os.chdir('../..')

    # Clean up val/stego
    val_stego = Path('datasets/val/stego')
    if val_stego.exists():
        shutil.rmtree(val_stego)

    # Clean up sample stego
    sample_stego = Path('datasets/sample_submission_2025/stego')
    if sample_stego.exists():
        shutil.rmtree(sample_stego)

    # Run data_generator.py from sample_submission_2025 for sample
    sample_clean_dir = Path('datasets/sample_submission_2025/clean')
    if sample_clean_dir.exists():
        limit_sample = int(count_clean_images(sample_clean_dir) * (args.limit / 100.0))
    else:
        limit_sample = args.limit
    os.chdir('datasets/sample_submission_2025')
    subprocess.run([
        'python', 'data_generator.py',
        '--clean_source', 'clean',
        '--output_stego', 'stego',
        '--seeds_dir', 'seeds',
        '--limit', str(limit_sample)
    ])
    os.chdir('../..')

    # Run data_generator.py from sample_submission_2025 for val
    val_clean_dir = Path('datasets/val/clean')
    if val_clean_dir.exists():
        limit_val = int(count_clean_images(val_clean_dir) * (args.limit / 100.0))
    else:
        limit_val = args.limit
    os.chdir('datasets/sample_submission_2025')
    subprocess.run([
        'python', 'data_generator.py',
        '--clean_source', '../val/clean',
        '--output_stego', '../val/stego',
        '--seeds_dir', '../val/seeds',
        '--limit', str(limit_val)
    ])
    os.chdir('../..')

if __name__ == "__main__":
    main()
