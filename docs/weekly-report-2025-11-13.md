# Starlight Steganography Scanner Performance Report

## Introduction
This report analyzes the performance of the Starlight steganography scanner on generated datasets, focusing on false positives (clean images incorrectly flagged as stego) and false negatives (stego images not detected). Scans were conducted on clean and stego image directories from various submissions using the `scanner.py` script with JSON output.

## Methodology
- **Datasets Scanned**: Clean and stego images from `datasets/*_submission_2025` and `val` (generated via data generators).
- **Tools**: `scanner.py` with `--json` flag for batch scanning.
- **Metrics**:
  - False positives: `is_stego: true` in clean results.
  - False negatives: `is_stego: false` in stego results.
- **Analysis**: Counts by detected type (for false positives) and embedded algorithm (for false negatives).

## Results

### False Positives (Clean Images Detected as Stego)
High rates on synthetic clean images, indicating sensitivity to generation patterns (gradients, circles).

| Dataset | Total Clean Images | False Positives | Rate | Types (Count) |
|---------|-------------------|-----------------|------|---------------|
| chatgpt_submission_2025 | 2800 | 707 | 25.3% | lsb.rgb (307), palette (400) |
| sample_submission_2025 | 311 | 84 | 27.0% | raw (48), palette (34), lsb.rgb (2) |
| val | 8 | 1 | 12.5% | lsb.rgb (1) |
| Others (claude, gemini, grok, maya) | 0-500 | 0 | 0% | N/A |

### False Negatives (Stego Images Not Detected)
Low rates overall, with highest in `val` dataset.

| Dataset | Total Stego Images | False Negatives | Rate | Algorithms (Count) |
|---------|-------------------|-----------------|------|-------------------|
| chatgpt_submission_2025 | 2800 | 43 | 1.5% | alpha (43) |
| val | 2484 | 196 | 7.9% | lsb (112), eoi (66), alpha (18) |
| Others (claude, gemini, grok, maya, sample) | 0-1500 | 0 | 0% | N/A |

## Discussion
- **False Positives**: Predominantly on generated clean images with artificial patterns. Types like `palette` and `lsb.rgb` are over-triggered, possibly due to model training on similar data. `raw` false positives in `sample` suggest sensitivity to file appends.
- **False Negatives**: Rare, but `lsb` and `eoi` embeddings in `val` are missed, indicating weaker signals or threshold issues. `alpha` failures in `chatgpt` may relate to uniform alpha channels.
- **Overall Performance**: High precision on actual stego (low false negatives), but low specificity on clean images (high false positives). Balanced for stego detection but prone to alarms on synthetic data.

## Recommendations
- **Threshold Tuning**: Adjust probabilities/thresholds for `lsb.rgb`, `palette`, and `raw` to reduce false positives.
- **Training Data**: Include more diverse clean images to improve generalization.
- **Validation**: Test on real-world images beyond generated sets.
- **Further Analysis**: Investigate embedding strengths and model confidence scores for missed detections.

This report is based on automated scans; manual verification of samples recommended for deeper insights.