# Negative Examples Schema

## Purpose
Define the format for negative counterexamples used in training the steganography detection model.

## Manifest Format (manifest.jsonl)
JSON Lines format with the following fields per line:

- `method`: The generation method/category (string)
- `constraint`: The teaching constraint/rule (string)
- `label`: Always "clean" for negative examples (string)
- `file_path`: Relative path to the generated image from the negatives directory (string)

Example entry:
```json
{"method": "rgb_no_alpha", "constraint": "RGB images cannot have alpha steganography", "label": "clean", "file_path": "rgb_no_alpha/rgb_no_alpha_0000.png"}
```

## Categories and Constraints

1. **rgb_no_alpha**: RGB images with no alpha channel
   - Constraint: "RGB images cannot have alpha steganography"

2. **uniform_alpha**: RGBA images with uniform alpha values
   - Constraint: "Uniform alpha channel contains no hidden data"

3. **dithered_gif**: Images with GIF dithering artifacts
   - Constraint: "GIF dithering is natural noise, not steganography"

4. **noise_lsb**: Images with natural LSB noise
   - Constraint: "Natural LSB variation is not hidden data"

5. **repetitive_hex**: Images with repetitive patterns
   - Constraint: "Repetitive hex patterns are visible, not hidden"

## Validation Report (validation_report.json)
JSON object containing validation results:

```json
{
  "total_samples": 5000,
  "extraction_tests_passed": 5000,
  "extraction_tests_failed": 0,
  "failed_samples": [],
  "validation_timestamp": "2025-11-17T...",
  "scanner_version": "v1.0"
}
```

All negative examples must pass steganography extraction tests (no hidden data detected).
