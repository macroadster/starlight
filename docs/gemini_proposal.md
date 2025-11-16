# Gemini Proposal - V3 Dual-Stream Architecture

**Version:** 1.0  
**Date:** 2025-11-08  
**Author:** Gemini-CLI  
**Status:** In Progress

## 1. Architecture Overview: Dual-Stream Input

The V3 model architecture addresses a critical blind spot in previous models by introducing a **Dual-Stream Input** pipeline. This design processes image pixel data and file metadata as two separate, parallel inputs, allowing the model to learn from features that are invisible in the decoded image alone (e.g., EXIF and EOI steganography).

The two streams are:
1.  **Pixel Tensor**: The traditional decoded image data (e.g., an RGB array).
2.  **Metadata Tensor**: A new, fixed-size tensor representing binary metadata extracted from the image file.

This approach allows the model to simultaneously analyze both visual patterns and file structure anomalies.

## 2. Data Structure and Preprocessing

The new V3 pipeline is implemented through the `scripts/v3_preprocess_data.py` script.

### 2.1. `v3_preprocess(image_path)`

This is the main function that takes an image file path and returns a dictionary containing the two tensors.

-   **Input**: `image_path` (string)
-   **Output**: `{"pixel_tensor": np.array, "metadata_tensor": np.array}`

### 2.2. Metadata Tensor

The `metadata_tensor` is a 1D NumPy array of `uint8` with a fixed size.

-   **Function**: `create_metadata_tensor(image_path, max_size=1024)`
-   **Size**: 1024 bytes. This is a crucial, fixed dimension for the model's input layer.
-   **Content**: The tensor is a concatenation of **EXIF data** and **End-of-Image (EOI) data**.
    -   **EXIF Data**: All EXIF tags are extracted and serialized into a byte string.
    -   **EOI Data**: For JPEG files, any data appended after the `0xFFD9` marker is extracted.
-   **Padding**: If the combined metadata is less than 1024 bytes, it is padded with null bytes (`0x00`) to reach the fixed size. If it is larger, it is truncated.

### 2.3. Pixel Tensor

The `pixel_tensor` is a standard 3D NumPy array representing the image.

-   **Function**: `create_pixel_tensor(image_path)`
-   **Format**: The image is always converted to `RGBA` to ensure a consistent 4-channel input for the model.
-   **Shape**: `(height, width, 4)`

## 3. Code Snippets & Example Usage

The following demonstrates how to use `v3_preprocess_data.py` to generate the two tensors for a given image.

```python
# Ensure you have Pillow and exifread installed:
# pip install Pillow exifread

from PIL import Image
from pathlib import Path
from scripts.v3_preprocess_data import v3_preprocess

# --- Create a dummy image for testing ---
dummy_file = Path("test_v3.jpg")
if not dummy_file.exists():
    print("Creating a dummy 'test_v3.jpg' for demonstration.")
    dummy_img = Image.new('RGB', (64, 64), color='green')
    dummy_img.save(dummy_file, "jpeg")
    # Add some dummy EOI data
    with open(dummy_file, "ab") as f:
        f.write(b"This is a secret message hidden after the EOI marker.")
# -----------------------------------------

# Process the file to get V3 tensors
try:
    tensors = v3_preprocess(str(dummy_file))

    # Print the results
    print(f"Successfully processed {dummy_file}")
    print("Pixel Tensor Shape:", tensors["pixel_tensor"].shape)
    print("Metadata Tensor Shape:", tensors["metadata_tensor"].shape)
    print(f"Metadata Tensor Content (first 48 bytes):\n{tensors['metadata_tensor'][:48]}")

except Exception as e:
    print(f"An error occurred: {e}")

```

This specification provides a concrete and verifiable implementation path for the V3 architecture. All agents should align their data generation and training scripts with the functions and data structures defined herein.
