import os
import sys
import csv
import json
from pathlib import Path
import argparse

# Add scripts to path
sys.path.append('scripts')

def load_model_from_submission(submission_path):
    """Load model from a submission directory"""
    model_dir = Path(submission_path) / 'model'
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Try to import the inference module
    sys.path.insert(0, str(model_dir))
    try:
        import inference
        # Assume it has a class or function
        if hasattr(inference, 'StarlightModel'):
            model = inference.StarlightModel()
        elif hasattr(inference, 'get_starlight_pipeline'):
            model = inference.get_starlight_pipeline()
        else:
            raise AttributeError("No suitable model class found")
    except ImportError as e:
        raise ImportError(f"Could not load model from {model_dir}: {e}")
    finally:
        sys.path.pop(0)

    return model

def run_inference_on_dataset(model, dataset_path, output_csv):
    """Run inference on all images in dataset and save to CSV"""
    image_paths = list(Path(dataset_path).glob('**/*.png'))
    if not image_paths:
        print(f"No PNG files found in {dataset_path}")
        return

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'stego_probability', 'predicted_method', 'is_steganography']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for img_path in image_paths:
            try:
                result = model.predict(str(img_path))
                if 'error' in result:
                    print(f"Error on {img_path}: {result['error']}")
                    continue
                writer.writerow({
                    'image_path': str(img_path),
                    'stego_probability': result.get('stego_probability', 0),
                    'predicted_method': result.get('predicted_method', 'unknown'),
                    'is_steganography': result.get('is_steganography', False)
                })
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

def calculate_metrics(predictions_csv, ground_truth_dir):
    """Calculate FPR and detection rates from predictions CSV"""
    predictions = {}
    with open(predictions_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions[row['image_path']] = {
                'prob': float(row['stego_probability']),
                'is_stego': row['is_steganography'].lower() == 'true'
            }

    # Assume clean images are in clean/ subdir, stego in others
    clean_paths = list(Path(ground_truth_dir).glob('clean/**/*.png'))
    stego_paths = []
    methods = {}
    for method_dir in Path(ground_truth_dir).iterdir():
        if method_dir.is_dir() and method_dir.name != 'clean':
            stego_paths.extend(list(method_dir.glob('**/*.png')))
            methods[method_dir.name] = list(method_dir.glob('**/*.png'))

    # FPR
    fp_count = 0
    for path in clean_paths:
        if str(path) in predictions and predictions[str(path)]['is_stego']:
            fp_count += 1
    fpr = (fp_count / len(clean_paths)) * 100 if clean_paths else 0

    # Detection rates per method
    detection_rates = {}
    for method, paths in methods.items():
        tp_count = 0
        for path in paths:
            if str(path) in predictions and predictions[str(path)]['is_stego']:
                tp_count += 1
        detection_rates[method] = (tp_count / len(paths)) * 100 if paths else 0

    return {'fpr': fpr, 'detection_rates': detection_rates}

def main():
    parser = argparse.ArgumentParser(description='Generate baseline predictions and metrics')
    parser.add_argument('--submission', required=True, help='Path to submission directory (e.g., datasets/grok_submission_2025)')
    parser.add_argument('--test-dataset', default='val', help='Path to test dataset')
    parser.add_argument('--output-dir', default='results/baselines', help='Output directory')

    args = parser.parse_args()

    submission_name = Path(args.submission).name
    output_csv = f"{args.output_dir}/{submission_name}_predictions.csv"
    metrics_json = f"{args.output_dir}/{submission_name}_metrics.json"

    # Load model
    model = load_model_from_submission(args.submission)

    # Run inference
    run_inference_on_dataset(model, args.test_dataset, output_csv)

    # Calculate metrics
    metrics = calculate_metrics(output_csv, args.test_dataset)

    # Save metrics
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Baseline generated for {submission_name}")
    print(f"Predictions: {output_csv}")
    print(f"Metrics: {metrics_json}")

if __name__ == '__main__':
    main()