# test_exif.py - Test EXIF steganography detection and extraction
import os
import sys

# Add current directory to path for imports
sys.path.append(".")
from exif_steganography import (
    detect_exif_stego,
    extract_exif_payload,
    batch_detect_exif,
)


def test_exif_detection():
    """Test EXIF steganography detection on sample images."""
    print("=== EXIF Steganography Detection Test ===")

    # Find JPEG images in stego directory
    jpeg_images = []
    for file in os.listdir("./stego"):
        if file.lower().endswith((".jpg", ".jpeg")):
            jpeg_images.append(f"./stego/{file}")

    if not jpeg_images:
        print("No JPEG images found in ./stego/ directory")
        return

    print(f"Found {len(jpeg_images)} JPEG images to test...")

    # Test each image
    for img_path in jpeg_images[:5]:  # Test first 5
        print(f"\n--- Testing: {img_path} ---")

        # Detection
        detection = detect_exif_stego(img_path)
        print(f"Stego Probability: {detection['stego_probability']:.3f}")

        if detection["indicators"]:
            print("Indicators:")
            for indicator in detection["indicators"]:
                print(f"  - {indicator}")

        if detection["error"]:
            print(f"Error: {detection['error']}")
            continue

        # Extraction if high probability
        if detection["stego_probability"] > 0.3:
            extraction = extract_exif_payload(img_path)
            if extraction["payload"]:
                print(f"Extracted Payload ({len(extraction['payload'])} chars):")
                print(f"  {extraction['payload'][:100]}...")
                print(f"  Encoding: {extraction.get('encoding', 'unknown')}")
            elif extraction["error"]:
                print(f"Extraction Error: {extraction['error']}")
        else:
            print("Low stego probability - skipping extraction")


def test_batch_detection():
    """Test batch EXIF detection."""
    print("\n=== Batch Detection Test ===")

    # Collect all images
    all_images = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                all_images.append(os.path.join(root, file))

    if not all_images:
        print("No JPEG images found")
        return

    print(f"Testing {len(all_images)} images with batch detection...")

    # Batch detection
    results = batch_detect_exif(all_images, threshold=0.5)

    print("\nBatch Results:")
    print(f"  Total Images: {results['total_images']}")
    print(f"  Stego Detected: {results['stego_detected']}")
    print(f"  Clean Images: {results['clean_images']}")
    print(f"  Errors: {results['errors']}")

    # Show details for detected stego images
    stego_details = [d for d in results["details"] if d["status"] == "stego"]
    if stego_details:
        print("\nStego Images Detected:")
        for detail in stego_details:
            print(f"  - {detail['path']}: {detail['probability']:.3f}")


def test_exif_structure_analysis():
    """Test EXIF structure analysis."""
    print("\n=== EXIF Structure Analysis Test ===")

    # Find a JPEG image
    jpeg_images = []
    for file in os.listdir("./stego"):
        if file.lower().endswith((".jpg", ".jpeg")):
            jpeg_images.append(f"./stego/{file}")

    if not jpeg_images:
        print("No JPEG images found for structure analysis")
        return

    # Analyze first JPEG
    img_path = jpeg_images[0]
    print(f"Analyzing structure of: {img_path}")

    from exif_steganography import analyze_exif_structure

    analysis = analyze_exif_structure(img_path)

    if "error" in analysis and analysis["error"]:
        print(f"Analysis Error: {analysis['error']}")
        return

    # Display image info
    img_info = analysis.get("image_info", {})
    print("\nImage Information:")
    print(f"  Format: {img_info.get('format', 'Unknown')}")
    print(f"  Mode: {img_info.get('mode', 'Unknown')}")
    print(f"  Size: {img_info.get('size', 'Unknown')}")
    print(f"  File Size: {img_info.get('file_size', 0)} bytes")

    # Display EXIF info
    exif_info = analysis.get("exif_analysis", {})
    print("\nEXIF Information:")
    print(f"  Has EXIF: {exif_info.get('has_exif', False)}")
    if exif_info.get("has_exif", False):
        print(f"  EXIF Size: {exif_info.get('exif_size', 0)} bytes")
        print(f"  IFD Count: {exif_info.get('ifd_count', 0)}")
        print(f"  Total Tags: {exif_info.get('total_tags', 0)}")

        # Show IFD details
        ifd_details = exif_info.get("ifd_details", {})
        for ifd_name, ifd_data in ifd_details.items():
            print(f"    {ifd_name}: {ifd_data.get('tag_count', 0)} tags")


if __name__ == "__main__":
    print("EXIF Steganography Detection and Extraction Test Suite")
    print("=" * 60)

    # Run all tests
    test_exif_detection()
    test_batch_detection()
    test_exif_structure_analysis()

    print("\n" + "=" * 60)
    print("EXIF steganography testing completed!")
    print("\nFeatures demonstrated:")
    print("- Individual image detection")
    print("- Payload extraction")
    print("- Batch processing")
    print("- EXIF structure analysis")
    print("- Steganography probability scoring")
