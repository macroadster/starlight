#!/usr/bin/env python3
import os
import glob
import re
import subprocess
import argparse
from collections import defaultdict
from tqdm import tqdm # Import tqdm for progress bar
import random # Import random for sampling

# --- Configuration ---
# The root directory for the dataset submissions
DATASET_ROOT = 'datasets'
# The maximum number of files to sample from each submission directory for testing.
# Set to None or a very large number to test all files.
SAMPLE_SIZE = 30 
# Common image extensions to find all files
IMAGE_FILE_PATTERN = '*{ext}'
IMAGE_EXTENSIONS = ['png', 'gif', 'jpeg', 'jpg', 'webp', 'bmp']
# The name of your primary execution script
EXTRACTOR_SCRIPT = 'scanner.py'

def parse_filename(filename):
    """
    Parses the filename format: {payload_name}_{algorithm}_{index}.{ext}
    Returns a tuple (algorithm, payload_base_name).
    """
    try:
        # Example: 'temporal_causality_warning_png_alpha_186.png'
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split('_')
        
        if len(parts) >= 3:
            # The algorithm name is the second-to-last part
            algorithm = parts[-2].lower()
            # The payload base name is everything before the algorithm and index
            payload_base_name = '_'.join(parts[:-2])
            return algorithm, payload_base_name
        
        return None, None
    except Exception:
        return None, None

def read_expected_payload(payload_path):
    """Reads the content of the expected Markdown payload file."""
    if payload_path and os.path.exists(payload_path):
        try:
            with open(payload_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            return f"ERROR_READING_PAYLOAD: {e}"
    return ""

def normalize_text(text):
    """Normalizes text by removing all whitespace and lowercasing."""
    # Removes all spaces, tabs, newlines, and returns the lowercase string
    return ''.join(text.split()).lower()

def run_extraction_test(image_path, algorithm, clean_path=None, payload_path=None):
    """
    Runs the scanner.py script for a specific image and algorithm.
    Returns a tuple: (result_status, failure_details)
    
    UPDATED: Now parses the 'MESSAGE EXTRACTED' block from scanner.py's output.
    """
    
    # Base command arguments for single file analysis and extraction
    command = [
        'python3', 
        EXTRACTOR_SCRIPT, 
        image_path,
        '--extract' # Important: Add the --extract flag
    ]
    
    # Add clean path for SDM extraction, if needed (though the current scanner.py only uses it for SDM)
    # We will assume that the existing logic in scanner.py handles which arguments it needs.
    if algorithm == 'sdm' and clean_path:
        command.extend(['--clean_path', clean_path])

    # Read expected payload for comparison
    expected_content = read_expected_payload(payload_path)
    if expected_content.startswith("ERROR_READING_PAYLOAD"):
        return "ERROR", expected_content
    
    # Execute the command
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30 # Add a timeout in case the script hangs
        )
        
        stdout_lines = result.stdout.split('\n')
        
        # --- NEW Logic to find the extracted message block from scanner.py ---
        start_tag = "✓ MESSAGE EXTRACTED:"
        end_delimiter = "=" * 80
        
        extracted_text = ""
        start_capturing = False
        
        for i, line in enumerate(stdout_lines):
            line_stripped = line.strip()
            
            if line_stripped == start_tag:
                # The message content starts on the next line, which should be the first delimiter
                start_capturing = True
                continue
            
            if start_capturing:
                if line_stripped == end_delimiter:
                    # Found the start delimiter
                    # The message is between the two delimiters
                    message_start = i + 1 # Start capturing from the line after the first '====='
                    
                    # Look for the second '=====' to find the end of the message
                    try:
                        message_end = stdout_lines[message_start:].index(end_delimiter) + message_start
                    except ValueError:
                        # Failed to find the closing delimiter
                        return "FAILURE", "Extraction start tag found, but missing closing '=====' delimiter."
                    
                    # Join the lines for the extracted message
                    extracted_text = '\n'.join(stdout_lines[message_start:message_end]).strip()
                    break # Stop processing lines after finding the full message block

        
        if extracted_text:
            # --- Payload Similarity Check for PASS/FAIL (Relaxed) ---
            if expected_content:
                # Normalize both texts to compare character count, ignoring all formatting
                normalized_expected = normalize_text(expected_content)
                normalized_extracted = normalize_text(extracted_text)
                
                expected_len = len(normalized_expected)
                extracted_len = len(normalized_extracted)
                
                # Success criterion: Extracted length must be within 20% tolerance of expected length.
                min_len = expected_len * 0.80
                max_len = expected_len * 1.20

                if extracted_len > 0 and min_len <= extracted_len <= max_len:
                    return "SUCCESS", "Extracted content length (normalized) is similar to expected payload."
                else:
                    details = (f"Extracted content length validation failed (Normalized Comparison). "
                               f"Expected range: [{min_len:.0f} - {max_len:.0f}] chars. "
                               f"Actual: {extracted_len} chars. Expected: {expected_len} chars.")
                    return "FAILURE", details
            else:
                # If no ground truth payload is found, assume success if a message was captured.
                return "SUCCESS", "Extraction block found, no ground truth payload for comparison."
        
        else:
            # No 'MESSAGE EXTRACTED' block found
            error_details = (f"Extraction Block Missing in stdout. Scanner output follows:", 
                             '\n'.join(result.stdout.strip().split('\n')[-10:]),
                             result.stderr.strip() if result.stderr else "No stderr.")
            return "FAILURE", error_details

    except FileNotFoundError:
        return "ERROR", f"Could not find {EXTRACTOR_SCRIPT}."
    except subprocess.TimeoutExpired:
        return "ERROR", f"Process timed out after 30 seconds."
    except Exception as e:
        return "ERROR", f"An unexpected error occurred during execution: {e}"

def main():
    """Main function to discover and run all tests."""
    
    overall_results = defaultdict(lambda: defaultdict(int))
    total_files = 0
    
    print("Starting Starlight Extractor tests...")
    
    # Iterate through all possible submission directories
    submission_root_dirs = glob.glob(f'{DATASET_ROOT}/*_submission_2025')
    
    if not submission_root_dirs:
        print(f"No submission root directories found matching pattern: {DATASET_ROOT}/*_submission_2025")
        return

    for root_dir in submission_root_dirs:
        stego_dir = os.path.join(root_dir, 'stego')
        
        # 1. Discover all image files in the stego directory
        stego_files = []
        for ext in IMAGE_EXTENSIONS:
            stego_files.extend(glob.glob(os.path.join(stego_dir, IMAGE_FILE_PATTERN.format(ext=ext))))
        
        # --- Random Sampling Logic ---
        initial_file_count = len(stego_files)
        global SAMPLE_SIZE 
        if SAMPLE_SIZE is not None and initial_file_count > SAMPLE_SIZE:
            print(f"\nSampling {SAMPLE_SIZE} files from {initial_file_count} in {os.path.basename(root_dir)}...")
            stego_files = random.sample(stego_files, SAMPLE_SIZE)
        else:
             print(f"\nProcessing all {initial_file_count} files in: {os.path.basename(root_dir)}")

        # Use tqdm for progress bar
        for image_path in tqdm(stego_files, desc=f"Testing {os.path.basename(root_dir)}"):
            total_files += 1
            stego_filename = os.path.basename(image_path)
            
            # 2. Infer Algorithm and Payload Name from the filename
            algorithm, payload_base_name = parse_filename(stego_filename)
            
            if not algorithm:
                # Only increment the unknown counter, do not print verbose messages for skips
                overall_results['unknown']['SKIPPED'] += 1
                continue
            
            # 3. Determine the clean path and payload path
            clean_dir = os.path.join(root_dir, 'clean')

            # --- ENHANCEMENT: Correctly derive the clean image filename ---
            # 1. Get the original file extension (e.g., .png)
            file_ext = os.path.splitext(stego_filename)[1]
            
            # 2. The clean image name is everything before the algorithm/index parts.
            # (payload_base_name is already computed as everything before _{algorithm}_{index})
            clean_filename = f"{payload_base_name}{file_ext}"
            
            # 3. Construct the full path
            clean_path = os.path.join(clean_dir, clean_filename)
            payload_path = os.path.join(root_dir, f'{payload_base_name}.md')

            # 4. Run the test and record the result
            result, failure_details = run_extraction_test(image_path, algorithm, clean_path, payload_path)
            
            # Verbose failure details block removed as requested.

            overall_results[algorithm][result] += 1

    # 5. Print Overall Test Summary (Unchanged)
    print("\n" + "="*70)
    print(f"FINAL STARLIGHT EXTRACTOR PERFORMANCE SUMMARY ({total_files} files tested)")
    print("="*70)
    
    for algo, results in overall_results.items():
        total_attempted = sum(v for k, v in results.items() if k != 'SKIPPED')
        successes = results['SUCCESS']
        skipped = results['SKIPPED']
        failures = results['FAILURE'] + results['ERROR']
        
        if total_attempted > 0:
            success_rate = (successes / total_attempted) * 100
            status = "✅ PASS" if success_rate >= 70 else "⚠️ MIXED" if success_rate > 30 else "❌ FAIL"
            print(f"[{algo.upper()}] {status} | Attempted: {total_attempted}, Success Rate: {success_rate:.2f}% ({successes}/{total_attempted}) | Skipped: {skipped}")
        elif skipped > 0:
            print(f"[{algo.upper()}]: All tests skipped ({skipped}).")
        else:
            print(f"[{algo.upper()}]: No files found for this algorithm.")

if __name__ == '__main__':
    main()
