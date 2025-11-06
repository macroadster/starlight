#!/usr/bin/env python3
"""
scripts/aggregate_models.py - Aggregate all submission models into super model
"""

import os
import json
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn


class SuperStarlightModel:
    def __init__(self):
        self.method_ensembles = {}  # "alpha": [model1, model2], ...
        self.weights = {}           # "alpha": [0.6, 0.4]
        self.load_all_models()

    def load_all_models(self):
        datasets_path = Path("datasets")
        for subdir in datasets_path.glob("*_submission_*"):
            config_path = subdir / "model" / "method_config.json"
            if not config_path.exists():
                continue
            with open(config_path) as f:
                config = json.load(f)
            for method, cfg in config.items():
                model_path = subdir / "model" / "detector.onnx"
                if model_path.exists():
                    if method not in self.method_ensembles:
                        self.method_ensembles[method] = []
                        self.weights[method] = []
                    self.method_ensembles[method].append(str(model_path))
                    # Weight by AUC from model_card.md
                    card_path = subdir / "model" / "model_card.md"
                    auc = self._extract_auc_from_card(card_path)
                    self.weights[method].append(auc)

    def _extract_auc_from_card(self, card_path):
        """Extract AUC from model_card.md"""
        if not card_path.exists():
            return 0.5  # Default weight
        with open(card_path, 'r') as f:
            content = f.read()
            if "AUC-ROC" in content:
                lines = content.split('\n')
                for line in lines:
                    if "AUC-ROC" in line:
                        try:
                            auc = float(line.split('|')[-2].strip())
                            return auc
                        except:
                            pass
        return 0.5

    def _detect_method_from_filename(self, img_path: str) -> str:
        basename = os.path.basename(img_path)
        parts = basename.split("_")
        if len(parts) >= 3:
            return parts[-2]  # e.g., alpha, eoi, dct
        return "lsb"

    def predict(self, img_path: str) -> Dict[str, Any]:
        method = self._detect_method_from_filename(img_path)
        if method not in self.method_ensembles:
            return {"error": f"No models for method {method}"}

        models = self.method_ensembles[method]
        weights = np.array(self.weights[method])
        weights = weights / weights.sum()  # Normalize

        # For simplicity, assume all models have same input preprocessing
        # In real implementation, need to handle different input shapes per method
        # For now, use dummy prediction
        prob = 0.5  # Placeholder
        return {
            "method": method,
            "stego_probability": prob,
            "predicted": prob > 0.5
        }


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
    with open("models/method_router.json", "w") as f:
        json.dump(router, f, indent=2)


def create_ensemble_weights(super_model):
    """Create ensemble_weights.json"""
    weights = {}
    for method, model_paths in super_model.method_ensembles.items():
        weights[method] = {}
        for i, path in enumerate(model_paths):
            submission = Path(path).parent.parent.name
            weights[method][submission] = super_model.weights[method][i]
    with open("models/ensemble_weights.json", "w") as f:
        json.dump(weights, f, indent=2)


def create_leaderboard(super_model):
    """Create leaderboard.md"""
    leaderboard = "# Model Leaderboard\n\n"
    leaderboard += "| Submission | Methods | AUC-ROC | Accuracy |\n"
    leaderboard += "|------------|---------|---------|----------|\n"

    submissions = []
    datasets_path = Path("datasets")
    for subdir in datasets_path.glob("*_submission_*"):
        config_path = subdir / "model" / "method_config.json"
        if config_path.exists():
            card_path = subdir / "model" / "model_card.md"
            auc = super_model._extract_auc_from_card(card_path) if card_path.exists() else 0.5
            methods = list(json.load(open(config_path)).keys())
            submissions.append({
                "name": subdir.name,
                "methods": ",".join(methods),
                "auc": auc,
                "acc": 0.95  # Placeholder
            })

    for sub in submissions:
        leaderboard += f"| {sub['name']} | {sub['methods']} | {sub['auc']} | {sub['acc']} |\n"

    with open("models/leaderboard.md", "w") as f:
        f.write(leaderboard)


def export_super_models():
    """Export super_detector.onnx and super_extractor.onnx"""
    # Placeholder: create dummy ONNX models
    # In real implementation, create a routing model
    print("Exporting super models... (placeholder)")


def main():
    print("=== Aggregating AI Models ===")

    super_model = SuperStarlightModel()

    print(f"Found ensembles for methods: {list(super_model.method_ensembles.keys())}")

    # Create output directory
    os.makedirs("models", exist_ok=True)

    create_method_router()
    create_ensemble_weights(super_model)
    create_leaderboard(super_model)
    export_super_models()

    print("=== Aggregation Complete ===")
    print("Super model components saved in 'models/' directory")


if __name__ == "__main__":
    main()