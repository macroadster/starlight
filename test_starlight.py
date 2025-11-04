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
# The name of the output log file
TEST_LOG_FILE = 'starlight_test_log.txt'

def log_and_print(message, log_file=None, end='\n'):
    """Prints to console and writes to a log file."""
    print(message, end=end)
    if log_file:
        log_file.write(message + end)

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

def run_extraction_test(image_path, algorithm, payload_path=None):
    """
    Runs the scanner.py script for a specific image and algorithm.
    Returns a tuple: (result_status, failure_details)
    """
    
    # Read expected payload for comparison
    expected_content = read_expected_payload(payload_path)
    if expected_content.startswith("ERROR_READING_PAYLOAD"):
        return "ERROR", expected_content
    
    try:
        # --- Run scanner.py (detection and extraction are automatic) ---
        command = [
            'python3', 
            EXTRACTOR_SCRIPT, 
            image_path
        ]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30 # Add a timeout in case the script hangs
        )
        
        stdout_lines = result.stdout.split('\n')
        
        # --- Parse detection results ---
        predicted_class = "unknown"
        confidence = 0.0
        
        # Regex to find "Predicted: {class} (confidence: {confidence:.2%})"
        prediction_regex = re.compile(r"Predicted: (\w+) \(confidence: ([\d.]+)%\)")
        
        for line in stdout_lines:
            match = prediction_regex.search(line)
            if match:
                predicted_class = match.group(1).lower()
                confidence = float(match.group(2)) / 100.0 # Convert percentage to float
                break
        
        # --- Evaluate Detection Result ---
        if predicted_class == 'clean':
            # If a stego image is detected as clean, it's a failure
            return "FAILURE", f"Detected as clean (confidence: {confidence:.2%}) but expected {algorithm}."
        elif predicted_class != algorithm:
            # If a stego image is detected as a wrong stego type, it's a failure
            return "FAILURE", f"Detected as {predicted_class} (confidence: {confidence:.2%}) but expected {algorithm}."
        
        # --- Parse extraction results ---
        start_tag = "Extracted Message:"
        # The end of the message is not clearly delimited, so we'll have to parse differently.
        # The message seems to be the last part of the output.
        
        extracted_text = ""
        try:
            start_index = result.stdout.index(start_tag) + len(start_tag)
            extracted_text = result.stdout[start_index:].strip()
        except ValueError:
            # If start_tag is not found, it means no message was extracted.
            pass

        if expected_content:
            normalized_expected = normalize_text(expected_content)
            expected_len = len(normalized_expected)
            
            if extracted_text:
                normalized_extracted = normalize_text(extracted_text)
                extracted_len = len(normalized_extracted)
                
                # Success criterion: Extracted length must be within 20% tolerance of expected length (100%).
                min_len = expected_len * 0.80
                max_len = expected_len * 1.20

                if extracted_len > 0 and min_len <= extracted_len <= max_len:
                    return "SUCCESS", f"Detected {predicted_class} (conf: {confidence:.2%}) and extracted content length (normalized) is similar to expected payload."
                else:
                    details = (f"Detected {predicted_class} (conf: {confidence:.2%}) but extracted content length validation failed (Normalized Comparison). "
                               f"Expected range: [{min_len:.0f} - {max_len:.0f}] chars. "
                               f"Actual: {extracted_len} chars. Expected: {expected_len} chars.")
                    return "FAILURE", details
            else:
                return "FAILURE", f"Detected {predicted_class} (conf: {confidence:.2%}) but no extraction block found."
        else:
            if extracted_text:
                return "SUCCESS", f"Detected {predicted_class} (conf: {confidence:.2%}) and extraction performed (no ground truth payload for comparison)."
            else:
                return "FAILURE", f"Detected {predicted_class} (conf: {confidence:.2%}) but no extraction block found."

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
    
    # Open log file
    try:
        log_file = open(TEST_LOG_FILE, 'w', encoding='utf-8')
    except Exception as e:
        print(f"ERROR: Could not open log file {TEST_LOG_FILE}: {e}")
        log_file = None
        
    log_and_print("Starting Starlight Extractor tests...", log_file=log_file)
    
    # Iterate through all possible submission directories
    submission_root_dirs = glob.glob(f'{DATASET_ROOT}/*_submission_2025')
    
    if not submission_root_dirs:
        log_and_print(f"No submission root directories found matching pattern: {DATASET_ROOT}/*_submission_2025", log_file)
        # Close log file before returning
        if log_file: log_file.close()
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
            log_and_print(f"\nSampling {SAMPLE_SIZE} files from {initial_file_count} in {os.path.basename(root_dir)}...", log_file)
            stego_files = random.sample(stego_files, SAMPLE_SIZE)
        else:
             log_and_print(f"\nProcessing all {initial_file_count} files in: {os.path.basename(root_dir)}", log_file)

        # Use tqdm for progress bar
        for image_path in tqdm(stego_files, desc=f"Testing {os.path.basename(root_dir)}"):
            total_files += 1
            stego_filename = os.path.basename(image_path)
            
            # 2. Infer Algorithm and Payload Name from the filename
            algorithm, payload_base_name = parse_filename(stego_filename)
            
            if not algorithm:
                # Log the skip
                log_file.write(f"TEST RESULT: [UNKNOWN] SKIPPED | File: {stego_filename} | Details: Filename parsing failed.\n")
                overall_results['unknown']['SKIPPED'] += 1
                continue
            
            # 3. Determine the clean path and payload path
            clean_dir = os.path.join(root_dir, 'clean')

            clean_path = os.path.join(clean_dir, stego_filename)
            payload_path = os.path.join(root_dir, f'{payload_base_name}.md')

            # 4. Run the test and record the result
            result, failure_details = run_extraction_test(image_path, algorithm, payload_path=payload_path)
            
            # --- Log individual test result ---
            log_message = f"TEST RESULT: [{algorithm.upper()}] {result:7s} | File: {stego_filename} | Details: {failure_details}"
            if log_file:
                 log_file.write(log_message + '\n')
            # ---------------------------------

            overall_results[algorithm][result] += 1

    # 5. Print Overall Test Summary (Logged and Printed)
    log_and_print("\n" + "="*70, log_file)
    log_and_print(f"FINAL STARLIGHT EXTRACTOR PERFORMANCE SUMMARY ({total_files} files tested)", log_file)
    log_and_print("="*70, log_file)
    
    for algo, results in overall_results.items():
        total_attempted = sum(v for k, v in results.items() if k != 'SKIPPED')
        successes = results['SUCCESS']
        skipped = results['SKIPPED']
        failures = results['FAILURE'] + results['ERROR']
        
        if total_attempted > 0:
            success_rate = (successes / total_attempted) * 100
            status = "✅ PASS" if success_rate >= 70 else "⚠️ MIXED" if success_rate > 30 else "❌ FAIL"
            log_and_print(f"[{algo.upper()}] {status} | Attempted: {total_attempted}, Success Rate: {success_rate:.2f}% ({successes}/{total_attempted}) | Skipped: {skipped}", log_file)
        elif skipped > 0:
            log_and_print(f"[{algo.upper()}]: All tests skipped ({skipped}).", log_file)
        else:
            log_and_print(f"[{algo.upper()}]: No files found for this algorithm.", log_file)

    # Close the log file
    if log_file:
        log_file.close()
        print(f"\nDetailed test results also written to {TEST_LOG_FILE}")


if __name__ == '__main__':
    main()
