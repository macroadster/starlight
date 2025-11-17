# Negative Examples Specification

## Purpose
Teach the model what steganography is NOT by providing diverse examples of clean images that should never be classified as containing hidden data.

## Categories Generated

### 1. rgb_no_alpha/ (200 images)
**Teaching:** RGB images cannot have alpha steganography
- Diverse RGB content with gradients, patterns, and textures
- No alpha channel (mode: RGB)
- Should always be classified as clean
- File formats: PNG

### 2. uniform_alpha/ (200 images)
**Teaching:** Uniform alpha = no hidden data
- RGBA images with alpha channel set to uniform values (0, 128, or 255)
- Alpha channel contains no variation or hidden information
- Should be classified as clean despite having alpha channel
- File formats: PNG

### 3. natural_noise/ (200 images)
**Teaching:** Natural noise ≠ steganography
- RGB images with natural-looking noise patterns
- Noise simulates sensor noise or compression artifacts
- No systematic patterns that could indicate hidden data
- Should be classified as clean
- File formats: PNG

### 4. patterns/ (200 images)
**Teaching:** Regular patterns ≠ steganography
- Images with geometric patterns, gradients, and textures
- Patterns are visible and intentional, not hidden
- Should be classified as clean
- File formats: PNG

### 5. special_cases/ (100 images)
**Teaching:** Edge cases that should be clean
- Small images (64x64, 128x128)
- Grayscale images (mode: L)
- Palette images (mode: P)
- Single-color images
- Should be classified as clean
- File formats: PNG, JPG

## Quality Guarantees
- All negative examples are verified to contain no steganography
- Diverse content covering edge cases and common confounders
- Balanced distribution across categories
- Proper image modes and formats for each category

## Usage in Training
- Label: 0 (clean) for all negative examples
- Can be mixed with regular clean examples
- Helps model distinguish between clean images and actual steganography
- Reduces false positives on legitimate image variations

## Integration Notes
- Total: 900 negative examples
- All images are 256x256 except special_cases
- Naming convention: {category}_{index:04d}.{ext}
- Ready for inclusion in training pipeline