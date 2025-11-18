# Coordination with Grok: Hugging Face Pipeline Compatibility for `inference.py`

## Subject: Refactoring `inference.py` for Hugging Face Pipeline Specification

This document outlines the current state of `inference.py` and the necessary steps to refactor it for compatibility with the Hugging Face `transformers` library's pipeline specification. This refactoring will enable easier integration, deployment, and sharing of our Starlight models within the Hugging Face ecosystem.

### Current State of `inference.py`

The existing `inference.py` script utilizes `onnxruntime` directly to perform inference with our Starlight models. It defines a `StarlightModel` class with `__init__` and `predict` methods. The `predict` method currently encapsulates all three stages of a typical inference workflow:
1.  **Preprocessing**: Loading and transforming image data using `starlight_utils.load_unified_input`.
2.  **Forward Pass (Inference)**: Executing the ONNX model via `self.detector.run()`.
3.  **Postprocessing**: Converting raw model outputs (logits) into structured predictions.

While functional, this structure does not align with the standardized Hugging Face `Pipeline` API.

### Benefits of Hugging Face Pipeline Compatibility

Refactoring `inference.py` to conform to the Hugging Face pipeline specification offers several advantages:
*   **Standardization**: Adherence to a widely adopted standard for ML model inference.
*   **Easier Integration**: Seamless integration with other Hugging Face tools and libraries.
*   **Simplified Deployment**: Potentially simpler deployment to platforms that support Hugging Face models.
*   **Community Sharing**: Enables easier sharing of our models on the Hugging Face Hub.

### Refactoring Status

The refactoring of `inference.py` to achieve Hugging Face pipeline compatibility has been successfully completed and tested. The following steps have been implemented:

1.  **Import `Pipeline`**: `from transformers import Pipeline` has been added to `inference.py`.
2.  **Custom Pipeline Class**: A new class, `StarlightSteganographyDetectionPipeline`, inheriting from `Pipeline`, has been defined.
3.  **`_sanitize_parameters` Implemented**: This method handles and validates pipeline-specific parameters.
4.  **`preprocess(self, image_path)` Implemented**: This method now calls `starlight_utils.load_unified_input(image_path, fast_mode=True)` and converts tensors to NumPy arrays with correct batch dimensions and channel ordering.
5.  **`_forward(self, model_inputs)` Implemented**: This method executes the ONNX model inference using `self.model.run(None, model_inputs)`. The ONNX session loading logic is integrated into the pipeline's `__init__`.
6.  **`postprocess(self, model_outputs)` Implemented**: This method performs sigmoid calculation, `argmax` for method prediction, and formats the final output dictionary, utilizing the `id2label` mapping from `config.json`.
7.  **Configuration File (`config.json`) Created**: A `config.json` file has been created in the `model/` directory, defining model architecture, labels, and other metadata, including the `id2label` mapping for steganography methods.
8.  **Testing Completed**: The refactored `inference.py` has been successfully tested with sample images, correctly detecting steganography and returning structured predictions.

### Current Usage

The refactored pipeline can be easily instantiated and used via the `get_starlight_pipeline()` convenience function in `inference.py`.

```python
from inference import get_starlight_pipeline

# Instantiate the pipeline
starlight_pipeline = get_starlight_pipeline()

# Example usage with an image path
image_path = "path/to/your/image.png"
result = starlight_pipeline(image_path)

print(result)
# Expected output format:
# [{'score': 0.9876, 'label': 'lsb.rgb'}]
```

### Next Steps

The `inference.py` script is now compatible with the Hugging Face `transformers` pipeline specification, enabling easier integration and deployment. The next phase involves testing the integrated pipeline within the broader training environment and preparing for model sharing on the Hugging Face Hub.