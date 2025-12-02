import unittest
import torch
from scripts.starlight_utils import load_unified_input
import os

class TestUnifiedPipeline(unittest.TestCase):

    def test_load_unified_input(self):
        # Path to a test image
        test_image_path = 'datasets/val/clean/clean-0405.png'

        # Check if the test image exists
        self.assertTrue(os.path.exists(test_image_path), f"Test image not found at {test_image_path}")

        # Load the unified input
        pixel_tensor, meta_tensor, alpha_tensor, lsb_tensor, palette_tensor, palette_lsb_tensor, format_tensor, content_features = load_unified_input(test_image_path)

        # --- Check Tensor Shapes ---
        self.assertEqual(pixel_tensor.shape, (3, 256, 256))
        self.assertEqual(meta_tensor.shape, (2048,))
        self.assertEqual(alpha_tensor.shape, (1, 256, 256))
        self.assertEqual(lsb_tensor.shape, (3, 256, 256))
        self.assertEqual(palette_tensor.shape, (768,))
        self.assertEqual(palette_lsb_tensor.shape, (1, 256, 256))
        self.assertEqual(format_tensor.shape, (6,))
        self.assertEqual(content_features.shape, (6,))

        # --- Check Tensor Types ---
        self.assertEqual(pixel_tensor.dtype, torch.float32)
        self.assertEqual(meta_tensor.dtype, torch.float32)
        self.assertEqual(alpha_tensor.dtype, torch.float32)
        self.assertEqual(lsb_tensor.dtype, torch.float32)
        self.assertEqual(palette_tensor.dtype, torch.float32)
        self.assertEqual(palette_lsb_tensor.dtype, torch.float32)
        self.assertEqual(format_tensor.dtype, torch.float32)
        self.assertEqual(content_features.dtype, torch.float32)

        # --- Check Tensor Value Ranges (Optional but Recommended) ---
        self.assertTrue(pixel_tensor.min() >= 0.0 and pixel_tensor.max() <= 1.0)
        self.assertTrue(meta_tensor.min() >= 0.0 and meta_tensor.max() <= 1.0)
        self.assertTrue(alpha_tensor.min() >= 0.0 and alpha_tensor.max() <= 1.0)
        self.assertTrue(lsb_tensor.min() >= 0.0 and lsb_tensor.max() <= 1.0)
        self.assertTrue(palette_tensor.min() >= 0.0 and palette_tensor.max() <= 1.0)
        self.assertTrue(palette_lsb_tensor.min() >= 0.0 and palette_lsb_tensor.max() <= 1.0)
        self.assertTrue(format_tensor.min() >= 0.0 and format_tensor.max() <= 1.0)
        self.assertTrue(content_features.min() >= 0.0 and content_features.max() <= 1.0)

    def test_load_steganographic_input(self):
        # Path to a steganographic test image
        test_image_path = '/Users/eric/sandbox/starlight/datasets/val/stego/humanity_bridge_alpha_001.png'

        # Check if the test image exists
        self.assertTrue(os.path.exists(test_image_path), f"Test image not found at {test_image_path}")

        # Load the unified input
        _, _, alpha_tensor, _, _, _, _, _ = load_unified_input(test_image_path)

        # Check for the AI42 prefix in the alpha tensor
        # The prefix is LSB-first, so we need to check for the reversed bit pattern
        ai42_prefix = torch.tensor([1,0,0,0,0,0,1,0, 1,0,0,1,0,0,1,0, 0,0,1,0,1,1,0,0, 0,1,0,0,1,1,0,0], dtype=torch.float32)
        self.assertTrue(torch.equal(alpha_tensor.flatten()[:32], ai42_prefix))

if __name__ == '__main__':
    unittest.main()
