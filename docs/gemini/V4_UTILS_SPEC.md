# V4 Utilities Specification

**Version:** 1.0  
**Date:** 2025-11-18  
**Author:** Gemini

## 1. Overview

This document specifies the data loading and preprocessing utilities for the V4 unified architecture. It defines the `load_unified_input` function, which provides a 6-stream tensor input for the model.

## 2. `load_unified_input` Function

The `load_unified_input` function is located in `scripts/starlight_utils.py`. It takes a single argument, the path to an image file, and returns a tuple of 6 tensors.

### 2.1. Returned Tensors

The function returns the following tensors in this order:

1.  `pixel_tensor`: The augmented image data.
2.  `meta_tensor`: The metadata from the image file, including EXIF and tail data.
3.  `alpha_tensor`: The alpha channel of the image, if present.
4.  `lsb_tensor`: The least significant bits of the image's color channels.
5.  `palette_tensor`: The image's color palette, if present.
6.  `format_tensor`: A tensor of features describing the image format.

### 2.2. Tensor Shapes and Types

| Tensor | Shape | Type |
|---|---|---|
| `pixel_tensor` | `(3, 256, 256)` | `torch.float32` |
| `meta_tensor` | `(2048,)` | `torch.float32` |
| `alpha_tensor` | `(1, 256, 256)` | `torch.float32` |
| `lsb_tensor` | `(3, 256, 256)` | `torch.float32` |
| `palette_tensor` | `(768,)` | `torch.float32` |
| `format_tensor` | `(5,)` | `torch.float32` |

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
