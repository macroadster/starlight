#!/usr/bin/env python3
"""
scripts/aggregate_models.py - Aggregate all submission models into super model
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
    os.path.join(os.path.dirname(__file__), '..', 'datasets', 'chatgpt_submission_2025', 'model'),
    'chatgpt'
)

GrokModel = load_model_from_path(
    os.path.join(os.path.dirname(__file__), '..', 'datasets', 'grok_submission_2025', 'model'),
    'grok'
)

ClaudeModel = load_model_from_path(
    os.path.join(os.path.dirname(__file__), '..', 'datasets', 'claude_submission_2025', 'model'),
    'claude'
)

GeminiModel = load_model_from_path(
    os.path.join(os.path.dirname(__file__), '..', 'datasets', 'gemini_submission_2025', 'model'),
    'gemini'
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
                model = None
                try:
                    if source == "chatgpt" and ChatGPTModel:
                        model_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'chatgpt_submission_2025', 'model', 'detector.onnx')
                        if os.path.exists(model_path):
                            model = ChatGPTModel(model_path)
                    elif source == "grok" and GrokModel:
                        model_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'grok_submission_2025', 'model', 'detector.onnx')
                        if os.path.exists(model_path):
                            # Patch the method_config path in GrokModel before instantiation
                            import sys
                            original_path = sys.path[:]
                            sys.path.insert(0, os.path.dirname(model_path))
                            try:
                                model = GrokModel(model_path, task="detect")
                            finally:
                                sys.path[:] = original_path
                    elif source == "claude" and ClaudeModel:
                        model_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'claude_submission_2025', 'model', 'detector.onnx')
                        if os.path.exists(model_path):
                            # Patch the method_config path in ClaudeModel before instantiation
                            import sys
                            original_path = sys.path[:]
                            sys.path.insert(0, os.path.dirname(model_path))
                            try:
                                model = ClaudeModel(model_path, task="detect")
                            finally:
                                sys.path[:] = original_path
                    elif source == "gemini" and GeminiModel:
                        model_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'gemini_submission_2025', 'model', 'detector.onnx')
                        if os.path.exists(model_path):
                            # Change working directory temporarily for method_config.json
                            old_cwd = os.getcwd()
                            os.chdir(os.path.dirname(model_path))
                            try:
                                model = GeminiModel()
                                model.task = "detect"
                            finally:
                                os.chdir(old_cwd)
                except Exception as init_error:
                    print(f"Error initializing {source} model: {init_error}")
                
                if not model:
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
                elif method == "alpha":
                    prob = result.get("stego_probability", 0.0)
                elif method == "eoi":
                    prob = result.get("stego_probability", 0.0)
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
                    auc_str = line.split("|")[2].strip()
                    # Handle range values like "0.980-0.995"
                    if "-" in auc_str:
                        # Take the higher value for weight calculation
                        auc = float(auc_str.split("-")[1].strip())
                    else:
                        auc = float(auc_str)
                    card["performance"] = card.get("performance", {})
                    card["performance"]["AUC-ROC"] = auc
                except (ValueError, IndexError):
                    pass

    # Extract coverage
    if "Steganography Coverage" in content:
        coverage_start = content.find("Steganography Coverage")
        coverage_section = content[coverage_start : coverage_start + 500]
        # Look for numbered methods or bullet points
        methods = []
        lines = coverage_section.split("\n")
        for line in lines:
            if "**" in line:  # Bold method names
                method = line.split("**")[1].split("**")[0].strip()
                methods.append(method)
            elif line.startswith("1.") or line.startswith("2."):  # Numbered list
                method = line.split(".", 1)[1].strip()
                if "**" in method:
                    method = method.split("**")[1].split("**")[0].strip()
                methods.append(method)
        
        if methods:
            card["steganography_coverage"] = methods

    # Extract speed
    if "GPU (" in content or "GPU:" in content:
        for line in content.split("\n"):
            if "GPU (" in line or "GPU:" in line:
                try:
                    if "GPU (" in line:
                        speed_str = line.split("GPU (")[1].split("ms")[0].strip()
                    else:
                        speed_str = line.split("GPU:")[1].split("ms")[0].strip()
                    card["inference_speed"] = card.get("inference_speed", {})
                    card["inference_speed"]["GPU"] = float(speed_str)
                except (ValueError, IndexError):
                    pass

    return card


def create_method_router():
    """Create method_router.json"""
    router = {
        "lsb": {"ensemble": "lsb", "input_shape": [1, 3, 256, 256]},
        "alpha": {"ensemble": "alpha", "input_shape": [1, 4, 256, 256]},
        "dct": {"ensemble": "dct", "input_shape": [1, 4096]},
        "exif": {"ensemble": "exif", "input_shape": [1, 1024]},
        "eoi": {"ensemble": "eoi", "input_shape": [1, 1024]},
        "palette": {"ensemble": "palette", "input_shape": [1, 1024]}
    }
    os.makedirs("../models", exist_ok=True)
    with open("../models/method_router.json", "w") as f:
        json.dump(router, f, indent=2)


def create_ensemble_weights(ensemble):
    """Create ensemble_weights.json"""
    weights = {}
    for i, config in enumerate(ensemble.model_configs):
        source = config.get("source", "unknown")
        method = config.get("method", "neural")
        key = f"{source}_{method}"
        weights[key] = ensemble.weights[i]
    
    os.makedirs("../models", exist_ok=True)
    with open("../models/ensemble_weights.json", "w") as f:
        json.dump(weights, f, indent=2)


def create_leaderboard():
    """Create leaderboard.md"""
    leaderboard = "# Model Leaderboard\n\n"
    leaderboard += "| Submission | Methods | AUC-ROC | Accuracy |\n"
    leaderboard += "|------------|---------|---------|----------|\n"

    submissions = []
    datasets_path = os.path.join(os.path.dirname(__file__), "..", "datasets")
    for subdir in os.listdir(datasets_path):
        if "_submission_" in subdir:
            config_path = os.path.join(datasets_path, subdir, "model", "method_config.json")
            if os.path.exists(config_path):
                card_path = os.path.join(datasets_path, subdir, "model", "model_card.md")
                card = load_model_card(os.path.join(datasets_path, subdir, "model"))
                auc = card.get("performance", {}).get("AUC-ROC", 0.5)
                methods = list(json.load(open(config_path)).keys())
                submissions.append({
                    "name": subdir,
                    "methods": ",".join(methods),
                    "auc": auc,
                    "acc": 0.95  # Placeholder
                })

    for sub in submissions:
        leaderboard += f"| {sub['name']} | {sub['methods']} | {sub['auc']} | {sub['acc']} |\n"

    os.makedirs("../models", exist_ok=True)
    with open("../models/leaderboard.md", "w") as f:
        f.write(leaderboard)


def create_ensemble():
    """Create and test the ensemble model"""
    print("=== Creating Super Starlight Ensemble ===")

    # Model configurations for ensemble - aggregating ChatGPT, Grok, and Claude models
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
            "method": "neural",
            "source": "claude",
            "weight": 1.0,
            "description": "Claude neural network detector (Alpha + Palette)",
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
        {
            "task": "detect",
            "method": "alpha",
            "source": "gemini",
            "weight": 1.0,
            "description": "Gemini alpha channel detector",
        },
        {
            "task": "detect",
            "method": "eoi",
            "source": "gemini",
            "weight": 1.0,
            "description": "Gemini EOI (End of Image) detector",
        },
    ]

    # Load model cards and calculate weights
    datasets_path = os.path.join(os.path.dirname(__file__), "..", "datasets")
    chatgpt_card = load_model_card(os.path.join(datasets_path, "chatgpt_submission_2025", "model"))
    grok_card = load_model_card(os.path.join(datasets_path, "grok_submission_2025", "model"))
    claude_card = load_model_card(os.path.join(datasets_path, "claude_submission_2025", "model"))
    gemini_card = load_model_card(os.path.join(datasets_path, "gemini_submission_2025", "model"))
    
    model_cards = [card for card in [chatgpt_card, grok_card, claude_card, gemini_card] if card]
    if model_cards:
        weights = calculate_model_weights(model_cards)
        # Apply weights to corresponding models
        if chatgpt_card and len(weights) > 0:
            model_configs[0]["weight"] = weights[0] if len(weights) > 0 else 1.0
        if grok_card and len(weights) > 1:
            model_configs[1]["weight"] = weights[1] if len(weights) > 1 else 1.0
            model_configs[3]["weight"] = weights[1] if len(weights) > 1 else 1.0  # Grok LSB
            model_configs[4]["weight"] = weights[1] if len(weights) > 1 else 1.0  # Grok EXIF
        if claude_card and len(weights) > 2:
            model_configs[2]["weight"] = weights[2] if len(weights) > 2 else 1.0
        if gemini_card and len(weights) > 3:
            model_configs[5]["weight"] = weights[3] if len(weights) > 3 else 1.0  # Gemini Alpha
            model_configs[6]["weight"] = weights[3] if len(weights) > 3 else 1.0  # Gemini EOI

    print("Model configurations:")
    for i, config in enumerate(model_configs):
        print(f"  {i + 1}. {config['description']} (weight: {config['weight']:.2f})")

    # Create ensemble
    ensemble = SuperStarlightDetector(model_configs)

    # Test on sample images from both datasets
    print("\n=== Testing Ensemble ===")

    test_images = [
        (os.path.join(datasets_path, "chatgpt_submission_2025", "clean", "sample_seed_alpha_000.png"), "ChatGPT Clean"),
        (os.path.join(datasets_path, "chatgpt_submission_2025", "stego", "sample_seed_alpha_000.png"), "ChatGPT Stego"),
        (os.path.join(datasets_path, "grok_submission_2025", "clean", "README_lsb_000.png"), "Grok Clean LSB"),
        (os.path.join(datasets_path, "grok_submission_2025", "stego", "README_lsb_000.png"), "Grok Stego LSB"),
        (os.path.join(datasets_path, "claude_submission_2025", "clean", "essence_seed_alpha_000.png"), "Claude Clean Alpha"),
        (os.path.join(datasets_path, "claude_submission_2025", "stego", "essence_seed_alpha_000.png"), "Claude Stego Alpha"),
        (os.path.join(datasets_path, "claude_submission_2025", "clean", "essence_seed_palette_000.bmp"), "Claude Clean Palette"),
        (os.path.join(datasets_path, "claude_submission_2025", "stego", "essence_seed_palette_000.bmp"), "Claude Stego Palette"),
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

    os.makedirs("../models", exist_ok=True)
    with open("../models/ensemble_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Create additional model files
    create_method_router()
    create_ensemble_weights(ensemble)
    create_leaderboard()

    print("\n=== Ensemble Complete ===")
    print("Results saved to models/ensemble_results.json")
    print("Model components saved in 'models/' directory")

    return ensemble


def main():
    print("=== Aggregating AI Models ===")
    ensemble = create_ensemble()
    print("=== Aggregation Complete ===")


if __name__ == "__main__":
    main()
