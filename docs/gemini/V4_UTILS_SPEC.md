# V4 Utilities Specification

**Version:** 1.0  
**Date:** 2025-11-18  
**Author:** Gemini

## 1. Overview

This document specifies the data loading and preprocessing utilities for the V4 unified architecture. It defines the `load_unified_input` function, which provides a 6-stream tensor input for the model.

## 2. `load_unified_input` Function

The `load_unified_input` function is located in `scripts/starlight_utils.py`. It takes a single argument, the path to an image file, and returns a tuple of 7 tensors.

### 2.1. Returned Tensors

The function returns the following tensors in this order:

1.  `pixel_tensor`: The augmented image data.
2.  `meta_tensor`: The metadata from the image file, including EXIF and EOI (End of Image) tail data.
3.  `alpha_tensor`: The alpha channel of the image, if present.
4.  `lsb_tensor`: The least significant bits of the image's color channels.
5.  `palette_tensor`: The image's color palette, if present.
6.  `format_tensor`: A tensor of features describing the image format.
7.  `content_features`: Features derived from the content of LSB and Alpha channels.

### 2.2. Tensor Shapes and Types

| Tensor | Shape | Type |
|---|---|---|
| `pixel_tensor` | `(3, 256, 256)` | `torch.float32` |
| `meta_tensor` | `(2048,)` | `torch.float32` |
| `alpha_tensor` | `(1, 256, 256)` | `torch.float32` |
| `lsb_tensor` | `(3, 256, 256)` | `torch.float32` |
| `palette_tensor` | `(768,)` | `torch.float32` |
| `format_tensor` | `(6,)` | `torch.float32` |
| `content_features` | `(6,)` | `torch.float32` |

### 2.3. Data Preprocessing Details

This section details the computation and extraction logic for each of the 7 tensors returned by `load_unified_input`.

#### 2.3.1. `pixel_tensor`

The `pixel_tensor` represents the core image data.
-   The input image is first converted to RGB format.
-   A `CenterCrop` operation is applied to resize the image to `(256, 256)`.
-   The cropped image is then converted into a `torch.float32` tensor, with pixel values normalized to the range [0, 1].

#### 2.3.2. `meta_tensor`

The `meta_tensor` captures critical metadata and tail data from the image file.
-   **EXIF Data:** Extracted using `img.info.get("exif")`, which leverages Pillow's built-in EXIF parsing capabilities for various formats, including JPEG, PNG, and WebP.
-   **EOI (End of Image) Tail Data:** Extracted using the `extract_post_tail` utility function, which intelligently identifies and extracts data appended after the official end-of-image markers for various formats (JPEG: `\xFF\xD9`, PNG: `IEND` chunk, GIF: trailer `;`, WEBP: RIFF chunk size).
-   The combined EXIF and EOI tail bytes are buffered, then padded with zeros or truncated to a fixed size of 2048 bytes.
-   The resulting byte array is converted to a `torch.float32` tensor, with values normalized to the range [0, 1].

#### 2.3.3. `alpha_tensor`

The `alpha_tensor` represents the alpha channel information.
-   If the input image is in `RGBA` mode, the alpha plane is extracted.
-   The Least Significant Bit (LSB) of each pixel in the alpha plane is calculated.
-   This LSB data is then `CenterCrop`ped to `(256, 256)` and converted to a `torch.float32` tensor.
-   If the image does not have an alpha channel, a tensor of zeros with shape `(1, 256, 256)` is returned.

#### 2.3.4. `lsb_tensor`

The `lsb_tensor` captures the Least Significant Bits of the image's color channels.
-   **Critical Fix:** LSB extraction is performed on the *original, un-augmented* RGB image data (after an initial `CenterCrop` to `(256, 256)`). This ensures that the steganographic signal, if present, is preserved before any data augmentation.
-   The LSB for each of the R, G, and B channels is extracted independently.
-   These three LSB planes are stacked and converted to a `torch.float32` tensor.

#### 2.3.5. `palette_tensor`

The `palette_tensor` provides information about the image's color palette.
-   If the input image is in `P` (palette) mode, its color palette is extracted.
-   The palette data is padded with zeros to a fixed size of 768 bytes.
-   The resulting byte array is converted to a `torch.float32` tensor, with values normalized to the range [0, 1].
-   If the image is not in palette mode, a tensor of zeros with shape `(768,)` is returned.

#### 2.3.6. `format_tensor`

The `format_tensor` encodes various high-level features describing the image's format.
-   It is a `torch.float32` tensor containing the following 6 features:
    -   `has_alpha`: Binary indicator (1.0 if `RGBA` mode, 0.0 otherwise).
    -   `alpha_std_dev`: Standard deviation of the alpha channel (normalized to [0, 1]), 0.0 if no alpha.
    -   `is_palette`: Binary indicator (1.0 if `P` mode, 0.0 otherwise).
    -   `is_rgb`: Binary indicator (1.0 if `RGB` mode, 0.0 otherwise).
    -   `width_norm`: Image width normalized by 256.
    -   `height_norm`: Image height normalized by 256.

#### 2.3.7. `content_features`

The `content_features` tensor provides statistical features derived from the LSB and Alpha channels.
-   It is a concatenation of `lsb_content_features` and `alpha_content_features`.
-   Each set of content features is calculated using the `_calculate_content_features` helper function, which computes:
    -   `uniqueness_ratio`: Ratio of unique bytes to total bytes.
    -   `printable_char_ratio`: Ratio of printable ASCII characters to total bytes.
    -   `most_common_char_ratio`: Ratio of the most common byte's count to total bytes.

## 3. Validation and Regression Testing

A validation script, `tests/test_unified_pipeline.py`, has been created to verify the output of the `load_unified_input` function.

A regression test was performed on the clean dataset using the `experiments/validate_unified_pipeline.py` script and the `models/detector_conservative.onnx` model. The test revealed a high false positive rate of 45.5%, significantly above the 0.37% target.

### 3.1. False Positive Analysis

The analysis of the false positives revealed that the vast majority were of type "raw" and "exif", with a confidence of 1.0. This suggests that the `detector_conservative.onnx` model is overly sensitive to the presence of any data in the `meta` stream.

When the `meta` stream was zeroed out, the false positive rate dropped to 0.69%, with the single remaining false positive being of type "palette". This confirms that the `meta` stream is the primary cause of the regression.

## 4. Recommendations

Based on the validation results, it is recommended that the `detector_conservative.onnx` model be re-trained. The high false positive rate is due to two main factors:

1.  **Model Sensitivity:** The model is overly sensitive to the presence of any data in the `meta` stream, causing it to incorrectly flag clean images as steganographic.
2.  **Lack of Special Case Handling:** The validation script (`experiments/validate_unified_pipeline.py`) does not include the special case handling and strong validation logic present in the more robust `scanner.py` implementation. This logic is crucial for filtering out false positives.

The next agent should consider two possible paths forward:

1.  **Re-train the model with special case handling:** Re-train the model and continue to use the special case handling from `scanner.py` to achieve a low false positive rate.
2.  **Build a model capable of generalization:** Build a new model that is capable of generalization and does not require rule-based "special cases". This aligns with the overall goal of Project Starlight.

The `load_unified_input` function is now ready for use in the training pipeline.
