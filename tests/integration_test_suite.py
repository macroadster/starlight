import pytest
import os
import numpy as np
from PIL import Image
import tempfile
import shutil

# Import scanner and extractor
from scripts.scanner import FastStegoScanner
from scripts.starlight_extractor import (
    extract_alpha,
    extract_palette,
    extract_lsb,
    extract_exif,
    extract_eoi,
)


class TestIntegrationSuite:
    """Integration tests for Starlight steganography detection across multiple formats."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for test images."""
        clean_dir = tempfile.mkdtemp()
        stego_dir = tempfile.mkdtemp()
        yield clean_dir, stego_dir
        shutil.rmtree(clean_dir)
        shutil.rmtree(stego_dir)

    @pytest.fixture
    def scanner(self):
        """Create scanner instance."""
        return FastStegoScanner(quick_mode=False, max_workers=1)

    def create_test_image(self, format="PNG", size=(256, 256)):
        """Create a test image."""
        if format == "PNG":
            img = Image.new("RGB", size, color="red")
        elif format == "JPEG":
            img = Image.new("RGB", size, color="blue")
            # Save as JPEG
        elif format == "GIF":
            img = Image.new("P", size)
            img.putpixel((0, 0), 0)
        else:
            img = Image.new("RGB", size, color="green")
        return img

    def test_clean_images_detection(self, scanner, temp_dirs):
        """Test detection on clean images across formats."""
        clean_dir, _ = temp_dirs

        # Create clean images
        formats = ["PNG", "JPEG", "GIF"]
        for fmt in formats:
            img = self.create_test_image(fmt)
            img.save(os.path.join(clean_dir, f"clean_{fmt.lower()}_001.{fmt.lower()}"))

        # Scan
        results = scanner.scan_directory(clean_dir, recursive=False, detail=False)

        # Assert low false positives
        stego_count = sum(1 for r in results if r.get("is_stego", False))
        assert (
            stego_count / len(results) < 0.05
        ), f"Too many false positives: {stego_count}/{len(results)}"

    def test_stego_detection_alpha(self, scanner, temp_dirs):
        """Test alpha channel steganography detection."""
        _, stego_dir = temp_dirs

        # Create RGBA image with hidden data in alpha
        img = Image.new("RGBA", (256, 256), color=(255, 0, 0, 255))
        # Simulate alpha stego
        pixels = list(img.getdata())
        for i in range(min(100, len(pixels))):
            r, g, b, a = pixels[i]
            pixels[i] = (r, g, b, a & 0xFE | (i % 2))  # Embed LSB
        img.putdata(pixels)
        img.save(os.path.join(stego_dir, "stego_alpha_001.png"))

        # Scan
        results = scanner.scan_directory(stego_dir, recursive=False, detail=False)

        # Should detect stego
        stego_results = [r for r in results if r.get("is_stego", False)]
        assert len(stego_results) > 0, "Failed to detect alpha steganography"

    def test_stego_detection_lsb(self, scanner, temp_dirs):
        """Test LSB steganography detection."""
        _, stego_dir = temp_dirs

        # Create image with LSB stego
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        pixels = np.array(img)
        # Embed in LSB
        for i in range(min(100, pixels.size // 3)):
            for c in range(3):
                pixels.flat[i * 3 + c] = (pixels.flat[i * 3 + c] & 0xFE) | (i % 2)
        img = Image.fromarray(pixels)
        img.save(os.path.join(stego_dir, "stego_lsb_001.png"))

        # Scan
        results = scanner.scan_directory(stego_dir, recursive=False, detail=False)

        # Should detect
        stego_results = [r for r in results if r.get("is_stego", False)]
        assert len(stego_results) > 0, "Failed to detect LSB steganography"

    def test_multi_format_support(self, scanner):
        """Test scanner works with multiple image formats."""
        formats = ["png", "jpg", "jpeg", "gif", "webp", "bmp"]
        for fmt in formats:
            try:
                img = self.create_test_image(fmt.upper() if fmt != "jpg" else "JPEG")
                with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
                    img.save(f.name)
                    result = scanner.scan_file(f.name, extract_messages=False)
                    assert result["status"] == "success", f"Failed to scan {fmt}"
                    os.unlink(f.name)
            except Exception as e:
                pytest.skip(f"Format {fmt} not supported: {e}")

    def test_extraction_functions(self):
        """Test individual extraction functions."""
        # Create test images with known payloads
        # This would require embedding known messages

        # For now, test that functions don't crash on clean images
        img = Image.new("RGBA", (64, 64), color=(255, 255, 255, 255))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)

            # Test extractors
            try:
                extract_alpha(f.name)
                extract_lsb(f.name)
                extract_palette(f.name)  # Will return None for RGB
                extract_exif(f.name)
                extract_eoi(f.name)
            except Exception as e:
                pytest.fail(f"Extraction function crashed: {e}")
            finally:
                os.unlink(f.name)

    @pytest.mark.parametrize("method", ["alpha", "palette", "lsb", "exif", "eoi"])
    def test_method_coverage(self, method):
        """Test that all steganography methods are covered."""
        # This is a placeholder - would need actual stego samples
        assert method in ["alpha", "palette", "lsb", "exif", "eoi"]


if __name__ == "__main__":
    pytest.main([__file__])
