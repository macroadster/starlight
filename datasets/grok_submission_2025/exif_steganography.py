# exif_steganography.py - EXIF steganography detection and extraction
import os
import piexif
import numpy as np
from PIL import Image


def detect_exif_stego(image_path):
    """
    Detect if an image contains EXIF steganography.
    Analyzes EXIF metadata for suspicious UserComment fields and other indicators.

    Args:
        image_path (str): Path to the image file

    Returns:
        dict: Detection results with probability and indicators
    """
    try:
        # Check if file exists and is JPEG
        if not os.path.exists(image_path):
            return {
                "stego_probability": 0.0,
                "indicators": [],
                "error": "File not found",
            }

        if not image_path.lower().endswith((".jpg", ".jpeg")):
            return {
                "stego_probability": 0.0,
                "indicators": ["Not JPEG format"],
                "error": None,
            }

        # Load EXIF data
        try:
            exif_dict = piexif.load(image_path)
        except Exception as e:
            return {
                "stego_probability": 0.1,
                "indicators": ["No EXIF data"],
                "error": str(e),
            }

        indicators = []
        stego_score = 0.0

        # Check for UserComment field (primary EXIF stego carrier)
        if "Exif" in exif_dict:
            exif_ifd = exif_dict["Exif"]

            # UserComment detection (main indicator)
            if piexif.ExifIFD.UserComment in exif_ifd:
                user_comment = exif_ifd[piexif.ExifIFD.UserComment]
                if isinstance(user_comment, bytes) and len(user_comment) > 8:
                    # Check for ASCII header pattern
                    if user_comment.startswith(b"ASCII\0\0\0"):
                        payload_length = len(user_comment) - 8
                        if payload_length > 50:  # Significant payload
                            stego_score += 0.8
                            indicators.append(
                                f"Large UserComment payload ({payload_length} bytes)"
                            )
                        elif payload_length > 10:
                            stego_score += 0.4
                            indicators.append(
                                f"UserComment payload ({payload_length} bytes)"
                            )

                    # Check for random-looking data (high entropy)
                    if len(user_comment) > 20:
                        payload = (
                            user_comment[8:]
                            if user_comment.startswith(b"ASCII\0\0\0")
                            else user_comment
                        )
                        entropy = calculate_entropy(payload)
                        if (
                            entropy > 6.0
                        ):  # High entropy indicates encrypted/compressed data
                            stego_score += 0.3
                            indicators.append(f"High entropy payload ({entropy:.2f})")

            # Check for unusual EXIF fields
            unusual_fields = []
            for tag, value in exif_ifd.items():
                tag_info = piexif.TAGS["Exif"].get(tag, ["Unknown"])
                tag_name = tag_info[0] if isinstance(tag_info, list) else str(tag_info)
                if tag_name not in common_exif_tags():
                    if isinstance(value, bytes) and len(value) > 100:
                        unusual_fields.append(f"{tag_name}: {len(value)} bytes")
                        stego_score += 0.1

            if unusual_fields:
                indicators.append(
                    f"Unusual EXIF fields: {', '.join(unusual_fields[:3])}"
                )

        # Check overall EXIF size (large EXIF can hide data)
        exif_size = 0
        try:
            exif_bytes = piexif.dump(exif_dict)
            exif_size = len(exif_bytes)
            if exif_size > 65535:  # Large EXIF segment
                stego_score += 0.2
                indicators.append(f"Large EXIF segment ({exif_size} bytes)")
        except:
            pass

        # Check for multiple EXIF segments (rare, suspicious)
        img = Image.open(image_path)
        if hasattr(img, "info") and img.info:
            # Additional checks can be added here
            pass

        # Normalize score
        stego_probability = min(stego_score, 1.0)
        final_exif_size = exif_size if "exif_size" in locals() else 0

        return {
            "stego_probability": stego_probability,
            "indicators": indicators,
            "exif_size": final_exif_size,
            "error": None,
        }

    except Exception as e:
        return {"stego_probability": 0.0, "indicators": [], "error": str(e)}


def extract_exif_payload(image_path):
    """
    Extract hidden payload from EXIF UserComment field.

    Args:
        image_path (str): Path to the JPEG image

    Returns:
        dict: Extraction results with payload and metadata
    """
    try:
        if not os.path.exists(image_path):
            return {"payload": "", "error": "File not found"}

        if not image_path.lower().endswith((".jpg", ".jpeg")):
            return {"payload": "", "error": "Not JPEG format"}

        # Load EXIF data
        exif_dict = piexif.load(image_path)

        if "Exif" not in exif_dict:
            return {"payload": "", "error": "No EXIF data found"}

        exif_ifd = exif_dict["Exif"]

        # Extract UserComment
        if piexif.ExifIFD.UserComment not in exif_ifd:
            return {"payload": "", "error": "No UserComment field found"}

        user_comment = exif_ifd[piexif.ExifIFD.UserComment]

        if not isinstance(user_comment, bytes) or len(user_comment) <= 8:
            return {"payload": "", "error": "Empty or invalid UserComment"}

        # Parse ASCII header
        if user_comment.startswith(b"ASCII\0\0\0"):
            payload = user_comment[8:]  # Skip 8-byte header
        else:
            payload = user_comment

        # Try to decode as ASCII
        try:
            decoded_payload = payload.decode("ascii", errors="replace")
            return {
                "payload": decoded_payload,
                "raw_payload": payload.hex(),
                "payload_size": len(payload),
                "encoding": "ASCII",
                "error": None,
            }
        except UnicodeDecodeError:
            # If not ASCII, return as hex
            return {
                "payload": "",
                "raw_payload": payload.hex(),
                "payload_size": len(payload),
                "encoding": "binary",
                "error": "Payload not ASCII encoded",
            }

    except Exception as e:
        return {"payload": "", "error": f"Extraction failed: {str(e)}"}


def analyze_exif_structure(image_path):
    """
    Analyze EXIF structure for steganography indicators.

    Args:
        image_path (str): Path to the image file

    Returns:
        dict: Detailed EXIF analysis
    """
    try:
        if not os.path.exists(image_path):
            return {"error": "File not found"}

        # Basic image info
        img = Image.open(image_path)
        img_info = {
            "format": getattr(img, "format", "Unknown"),
            "mode": getattr(img, "mode", "Unknown"),
            "size": getattr(img, "size", (0, 0)),
            "file_size": os.path.getsize(image_path),
        }

        # EXIF analysis
        try:
            exif_dict = piexif.load(image_path)
            exif_size = len(piexif.dump(exif_dict)) if exif_dict else 0
            exif_analysis = {
                "has_exif": True,
                "exif_size": exif_size,
                "ifd_count": len(
                    [
                        k
                        for k in exif_dict.keys()
                        if k in ["0th", "Exif", "1st", "GPS", "Interop"]
                    ]
                ),
                "total_tags": sum(
                    len(v) for v in exif_dict.values() if isinstance(v, dict)
                ),
            }

            # Analyze each IFD
            ifd_details = {}
            for ifd_name, ifd_data in exif_dict.items():
                if isinstance(ifd_data, dict):
                    ifd_details[ifd_name] = {
                        "tag_count": len(ifd_data),
                        "tags": list(ifd_data.keys())[:10],  # First 10 tags
                    }

            exif_analysis["ifd_details"] = ifd_details

        except Exception as e:
            exif_analysis = {"has_exif": False, "error": str(e)}

        result = {"image_info": img_info, "exif_analysis": exif_analysis, "error": None}
        return result

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


def calculate_entropy(data):
    """Calculate Shannon entropy of byte data."""
    if not data:
        return 0.0

    # Count byte frequencies
    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)

    # Calculate probabilities
    probabilities = byte_counts / len(data)
    probabilities = probabilities[probabilities > 0]  # Remove zero probabilities

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def common_exif_tags():
    """Return list of common EXIF tags that are usually not suspicious."""
    return [
        "DateTime",
        "DateTimeOriginal",
        "DateTimeDigitized",
        "Make",
        "Model",
        "Software",
        "ExifVersion",
        "Flash",
        "FNumber",
        "ExposureTime",
        "ISOSpeedRatings",
        "FocalLength",
        "WhiteBalance",
        "MeteringMode",
        "ColorSpace",
        "SensingMethod",
        "FileSource",
        "SceneType",
        "CustomRendered",
        "ExposureMode",
        "WhiteBalance",
        "DigitalZoomRatio",
        "SceneCaptureType",
        "GainControl",
        "Contrast",
        "Saturation",
        "Sharpness",
        "SubjectDistanceRange",
        "ImageWidth",
        "ImageLength",
        "BitsPerSample",
        "Compression",
        "PhotometricInterpretation",
        "Orientation",
        "SamplesPerPixel",
        "PlanarConfiguration",
        "YCbCrPositioning",
        "XResolution",
        "YResolution",
        "ResolutionUnit",
    ]


def batch_detect_exif(image_paths, threshold=0.5):
    """
    Detect EXIF steganography in multiple images.

    Args:
        image_paths (list): List of image file paths
        threshold (float): Probability threshold for stego detection

    Returns:
        dict: Batch detection results
    """
    results = {
        "total_images": len(image_paths),
        "stego_detected": 0,
        "clean_images": 0,
        "errors": 0,
        "details": [],
    }

    for img_path in image_paths:
        detection = detect_exif_stego(img_path)

        if detection["error"]:
            results["errors"] += 1
            status = "error"
        elif detection["stego_probability"] >= threshold:
            results["stego_detected"] += 1
            status = "stego"
        else:
            results["clean_images"] += 1
            status = "clean"

        results["details"].append(
            {
                "path": img_path,
                "status": status,
                "probability": detection["stego_probability"],
                "indicators": detection["indicators"],
            }
        )

    return results


# Demo and testing functions
if __name__ == "__main__":
    # Generate some test images with EXIF steganography
    print("=== EXIF Steganography Detection Demo ===")

    # Test with existing stego images if available
    test_images = [
        "./stego/seed1_lsb_000.png",  # This is PNG, should be negative
        "../sample_submission_2025/stego/seed1_lsb_004.png",  # Also PNG
    ]

    # Look for JPEG images
    jpeg_images = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                jpeg_images.append(os.path.join(root, file))

    if jpeg_images:
        test_images.extend(jpeg_images[:3])  # Test first 3 JPEGs

    print(f"Testing {len(test_images)} images...")

    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n--- Analyzing: {img_path} ---")

            # Detection
            detection = detect_exif_stego(img_path)
            print(f"Stego Probability: {detection['stego_probability']:.3f}")
            if detection["indicators"]:
                print("Indicators:", ", ".join(detection["indicators"]))

            # Structure analysis
            analysis = analyze_exif_structure(img_path)
            if analysis.get("error") is None:
                img_info = analysis.get("image_info", {})
                exif_info = analysis.get("exif_analysis", {})
                print(f"Format: {img_info.get('format', 'Unknown')}")
                print(f"File Size: {img_info.get('file_size', 0)} bytes")
                if exif_info.get("has_exif", False):
                    print(f"EXIF Size: {exif_info.get('exif_size', 0)} bytes")
                    print(f"Total Tags: {exif_info.get('total_tags', 0)}")

            # Extraction if high probability
            if detection["stego_probability"] > 0.3:
                extraction = extract_exif_payload(img_path)
                if extraction["payload"]:
                    print(
                        f"Extracted Payload ({len(extraction['payload'])} chars): {extraction['payload'][:100]}..."
                    )
                elif extraction["error"]:
                    print(f"Extraction Error: {extraction['error']}")
        else:
            print(f"File not found: {img_path}")

    print("\n=== Summary ===")
    print("EXIF steganography detection system ready!")
    print("Features:")
    print("- UserComment payload detection")
    print("- Entropy analysis")
    print("- Unusual EXIF field detection")
    print("- Batch processing support")
