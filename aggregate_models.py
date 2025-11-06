#!/usr/bin/env python3
"""
aggregate_models.py - Create ensemble super model from multiple steganography detectors
Following the Starlight architecture proposal
"""

import os
import json
import sys
from typing import List, Dict

# Dynamic model loading function
def load_model_from_path(model_path, model_name):
    """Dynamically load a model from a given path"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            f"{model_name}_inference", 
            os.path.join(model_path, "inference.py")
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, 'StarlightModel', None)
    except Exception as e:
        print(f"Warning: Could not load {model_name} model: {e}")
    return None

# Load models dynamically
ChatGPTModel = load_model_from_path(
    os.path.join(os.path.dirname(__file__), 'datasets', 'chatgpt_submission_2025', 'model'),
    'chatgpt'
)

GrokModel = load_model_from_path(
    os.path.join(os.path.dirname(__file__), 'datasets', 'grok_submission_2025', 'model'),
    'grok'
)


class SuperStarlightDetector:
    """
    Ensemble detector that combines multiple steganography detection methods
    """

    def __init__(self, model_configs: List[Dict]):
        """
        Initialize ensemble with multiple model configurations

        Args:
            model_configs: List of model configs with weights and methods
        """
        self.model_configs = model_configs
        self.weights = [config.get("weight", 1.0) for config in model_configs]

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def predict(self, img_path: str) -> Dict:
        """
        Make ensemble prediction

        Args:
            img_path: Path to image file

        Returns:
            Dictionary with ensemble results
        """
        if not os.path.exists(img_path):
            return {"error": "Image file not found"}

        individual_results = []
        weighted_probs = []

        for i, config in enumerate(self.model_configs):
            try:
                # Choose model based on source
                source = config.get("source", "grok")
                if source == "chatgpt" and ChatGPTModel:
                    model_path = os.path.join(os.path.dirname(__file__), 'datasets', 'chatgpt_submission_2025', 'model', 'detector.onnx')
                    model = ChatGPTModel(model_path)
                elif source == "grok" and GrokModel:
                    model_path = os.path.join(os.path.dirname(__file__), 'datasets', 'grok_submission_2025', 'model', 'detector.onnx')
                    model = GrokModel(model_path, task="detect")
                else:
                    # Fallback to generic model
                    model = None
                    result = {"error": f"Model {source} not available"}
                    individual_results.append(result)
                    weighted_probs.append(0.0)
                    continue

                result = model.predict(img_path)
                individual_results.append(result)

                # Extract probability based on model method
                method = config.get("method", "neural")
                if method == "neural":
                    prob = result.get("nn_probability", result.get("stego_probability", 0.0))
                elif method == "lsb":
                    prob = result.get("lsb_probability", 0.0)
                elif method == "exif":
                    prob = result.get("exif_probability", 0.0)
                else:
                    prob = result.get("stego_probability", 0.0)

                weighted_probs.append(prob * self.weights[i])

            except Exception as e:
                print(f"Error in model {i}: {e}")
                individual_results.append({"error": str(e)})
                weighted_probs.append(0.0)

        # Calculate ensemble probability
        ensemble_prob = sum(weighted_probs)

        # Determine stego type based on strongest signal
        stego_type = "none"
        max_signal = 0.0

        for result in individual_results:
            if "lsb_probability" in result and result["lsb_probability"] > max_signal:
                max_signal = result["lsb_probability"]
                stego_type = "lsb"
            if "exif_probability" in result and result["exif_probability"] > max_signal:
                max_signal = result["exif_probability"]
                stego_type = "exif"

        return {
            "image_path": img_path,
            "ensemble_probability": ensemble_prob,
            "predicted": ensemble_prob > 0.5,
            "stego_type": stego_type,
            "individual_results": individual_results,
            "model_weights": self.weights,
        }


def calculate_model_weights(model_cards: List[Dict]) -> List[float]:
    """
    Calculate weights for models based on their performance metrics

    Args:
        model_cards: List of model card dictionaries

    Returns:
        List of calculated weights
    """
    weights = []

    for card in model_cards:
        base_weight = 1.0

        # AUC-based weighting
        auc = card.get("performance", {}).get("AUC-ROC", 0.5)
        if auc >= 0.99:
            base_weight *= 1.5
        elif auc >= 0.95:
            base_weight *= 1.2

        # Coverage bonus
        coverage = card.get("steganography_coverage", [])
        if len(coverage) >= 3:
            base_weight *= 1.1

        # Speed bonus
        speed = card.get("inference_speed", {}).get("GPU", 100)
        if speed < 5:
            base_weight *= 1.1

        # Extraction accuracy bonus
        ber = card.get("performance", {}).get("Extraction BER", 1.0)
        if ber < 0.01:
            base_weight *= 1.3

        weights.append(base_weight)

    return weights


def load_model_card(model_path: str) -> Dict:
    """Load model card from markdown file"""
    card_path = os.path.join(model_path, "model_card.md")
    if not os.path.exists(card_path):
        return {}

    # Simple parsing - in production, use proper markdown parser
    card = {}
    with open(card_path, "r") as f:
        content = f.read()

    # Extract performance metrics
    if "AUC-ROC" in content:
        for line in content.split("\n"):
            if "AUC-ROC" in line and "|" in line:
                try:
                    auc = float(line.split("|")[2].strip())
                    card["performance"] = card.get("performance", {})
                    card["performance"]["AUC-ROC"] = auc
                except (ValueError, IndexError):
                    pass

    # Extract coverage
    if "Steganography Coverage" in content:
        coverage_start = content.find("Steganography Coverage")
        coverage_section = content[coverage_start : coverage_start + 200]
        if "`" in coverage_section:
            coverage_text = coverage_section.split("`")[1]
            card["steganography_coverage"] = [
                c.strip() for c in coverage_text.split(",")
            ]

    # Extract speed
    if "GPU:" in content:
        for line in content.split("\n"):
            if "GPU:" in line:
                try:
                    speed_str = line.split("GPU:")[1].split("ms")[0].strip()
                    card["inference_speed"] = card.get("inference_speed", {})
                    card["inference_speed"]["GPU"] = float(speed_str)
                except (ValueError, IndexError):
                    pass

    return card


def create_ensemble():
    """Create and test the ensemble model"""
    print("=== Creating Super Starlight Ensemble ===")

    # Model configurations for ensemble - aggregating ChatGPT and Grok models
    model_configs = [
        {
            "task": "detect",
            "method": "neural",
            "source": "chatgpt",
            "weight": 1.0,
            "description": "ChatGPT neural network detector",
        },
        {
            "task": "detect", 
            "method": "neural",
            "source": "grok",
            "weight": 1.0,
            "description": "Grok neural network detector",
        },
        {
            "task": "detect",
            "method": "lsb",
            "source": "grok",
            "weight": 1.0,
            "description": "Grok LSB statistical detector",
        },
        {
            "task": "detect",
            "method": "exif",
            "source": "grok",
            "weight": 1.0,
            "description": "Grok EXIF metadata detector",
        },
    ]

    # Load model cards and calculate weights
    chatgpt_card = load_model_card("datasets/chatgpt_submission_2025/model")
    grok_card = load_model_card("datasets/grok_submission_2025/model")
    
    model_cards = [card for card in [chatgpt_card, grok_card] if card]
    if model_cards:
        weights = calculate_model_weights(model_cards)
        # Apply weights to corresponding models
        if chatgpt_card and len(weights) > 0:
            model_configs[0]["weight"] = weights[0] if len(weights) > 0 else 1.0
        if grok_card and len(weights) > 1:
            for i in range(1, min(4, len(weights) + 1)):
                if i < len(model_configs):
                    model_configs[i]["weight"] = weights[1]

    print("Model configurations:")
    for i, config in enumerate(model_configs):
        print(f"  {i + 1}. {config['description']} (weight: {config['weight']:.2f})")

    # Create ensemble
    ensemble = SuperStarlightDetector(model_configs)

    # Test on sample images from both datasets
    print("\n=== Testing Ensemble ===")

    test_images = [
        ("datasets/chatgpt_submission_2025/clean/sample_seed_alpha_000.png", "ChatGPT Clean"),
        ("datasets/chatgpt_submission_2025/stego/sample_seed_alpha_000.png", "ChatGPT Stego"),
        ("datasets/grok_submission_2025/clean/README_lsb_000.png", "Grok Clean LSB"),
        ("datasets/grok_submission_2025/stego/README_lsb_000.png", "Grok Stego LSB"),
    ]

    results = []
    for img_path, label in test_images:
        if os.path.exists(img_path):
            result = ensemble.predict(img_path)
            results.append(
                {
                    "label": label,
                    "path": img_path,
                    "probability": float(result["ensemble_probability"]),
                    "predicted": bool(result["predicted"]),
                    "type": result["stego_type"],
                }
            )

            print(
                f"{label:20s} - Probability: {result['ensemble_probability']:.3f}, "
                f"Predicted: {result['predicted']}, Type: {result['stego_type']}"
            )
        else:
            print(f"Image not found: {img_path}")

    # Save ensemble results
    output = {
        "ensemble_config": model_configs,
        "test_results": results,
        "performance": {
            "models_count": len(model_configs),
            "ensemble_weights": ensemble.weights,
        },
    }

    os.makedirs("model", exist_ok=True)
    with open("model/ensemble_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Ensemble Complete ===")
    print("Results saved to model/ensemble_results.json")

    return ensemble


if __name__ == "__main__":
    ensemble = create_ensemble()
