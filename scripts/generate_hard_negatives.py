import os
import shutil
from tqdm import tqdm
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from scanner import _scan_logic, init_worker
from scripts.starlight_utils import extract_post_tail
import onnxruntime


def generate_hard_negatives(clean_dir, hard_negatives_dir, model_path):
    """
    Generates a dataset of hard negatives by running the original scanner on clean images.
    """
    if not os.path.exists(hard_negatives_dir):
        os.makedirs(hard_negatives_dir)

    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(clean_dir)
        for file in files
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"))
    ]

    session = onnxruntime.InferenceSession(model_path)

    hard_negatives_count = 0
    for image_path in tqdm(image_paths, desc="Generating hard negatives"):
        try:
            # This is the model prediction part from the old scanner
            from trainer import load_enhanced_multi_input
            import torch

            meta, alpha, lsb, palette = load_enhanced_multi_input(image_path)
            meta = meta.unsqueeze(0)
            alpha = alpha.unsqueeze(0)
            lsb = lsb.unsqueeze(0)
            palette = palette.unsqueeze(0)
            bit_order = torch.tensor([[0.0, 1.0, 0.0]])
            input_feed = {
                "meta": meta.numpy(),
                "alpha": alpha.numpy(),
                "lsb": lsb.numpy(),
                "palette": palette.numpy(),
                "bit_order": bit_order.numpy(),
            }
            stego_logits, _, _, _, _ = session.run(None, input_feed)
            logit = stego_logits[0][0]
            if logit >= 0:
                stego_prob = 1 / (1 + np.exp(-logit))
            else:
                stego_prob = np.exp(logit) / (1 + np.exp(logit))

            # Now, we check if the model prediction is positive
            if stego_prob > 0.5:
                # Now run the full _scan_logic from the original scanner
                # which includes the special cases
                result = _scan_logic(image_path, session)
                if not result.get("is_stego"):
                    # This is a hard negative!
                    shutil.copy(
                        image_path,
                        os.path.join(hard_negatives_dir, os.path.basename(image_path)),
                    )
                    hard_negatives_count += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(f"Generated {hard_negatives_count} hard negatives.")


if __name__ == "__main__":
    # We need to use the model that the original scanner was using.
    # scanner.py defaults to models/detector_balanced.onnx

    # I will use all the clean datasets to generate hard negatives
    clean_dirs = [
        "datasets/val/clean",
        "datasets/chatgpt_submission_2025/clean",
        "datasets/claude_submission_2025/clean",
        "datasets/gemini_submission_2025/clean",
        "datasets/grok_submission_2025/clean",
        "datasets/maya_submission_2025/clean",
        "datasets/sample_submission_2025/clean",
    ]

    for clean_dir in clean_dirs:
        generate_hard_negatives(
            clean_dir, "datasets/hard_negatives/clean", "models/detector_balanced.onnx"
        )
