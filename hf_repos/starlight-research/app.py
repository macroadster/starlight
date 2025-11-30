#!/usr/bin/env python3
"""
Gradio demo for Starlight Research (Experimental)
"""

import gradio as gr
from inference import detect_steganography
import tempfile
import os

def detect_image(image):
    """Detect steganography in uploaded image"""
    if image is None:
        return "Please upload an image"

    # Save uploaded image to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Run detection
        result = detect_steganography(tmp_path)

        # Format output
        prob = result.get('stego_probability', 0.0)
        predicted = result.get('predicted', False)
        method = result.get('method', 'unknown')

        status = "STEGANOGRAPHY DETECTED" if predicted else "CLEAN IMAGE"
        confidence = f"{prob:.3f}"

        warning = """
⚠️ **EXPERIMENTAL MODEL** - Results may be unreliable
This is a research version under active development.
Use production model for reliable detection.
        """

        output = f"""
{warning}

**Detection Result:** {status}
**Probability:** {confidence}
**Method:** {method}
**Image:** {result.get('image_path', 'N/A')}
        """

        return output.strip()

    except Exception as e:
        return f"Error during detection: {str(e)}"
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Create Gradio interface
iface = gr.Interface(
    fn=detect_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Detection Results"),
    title="Starlight Research Detector (Experimental)",
    description="⚠️ EXPERIMENTAL - Upload an image to detect steganography. This is a development version, results are not reliable.",
    examples=[
        ["https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"]
    ]
)

if __name__ == "__main__":
    iface.launch()