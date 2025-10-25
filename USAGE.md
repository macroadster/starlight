# USAGE: Running Project Starlight

These instructions detail the steps for generating training data, verifying its integrity, training the AI model, and running the steganography detection process. All commands should be executed from the project's **root directory** unless otherwise specified.

---

## 1. Generating Training Datasets

The protocol relies on community-contributed, diverse datasets. Individual contributors run the data generation script within their own dedicated directory.

* **Action:** Navigate into your contributor directory within `datasets` and run the script. The `--limit` flag controls the number of files generated (e.g., 1000).
* **Command:**
    ```bash
    cd datasets/<contributor_name>
    python3 data_generator.py --limit 1000
    ```
    *(Note: Replace `<contributor_name>` with your directory name.)*

---

## 2. Data Integrity Verification

Run the diagnostic script from the project root to ensure that all contributed datasets are properly structured, labeled, and ready to be used for model training.

* **Action:** Execute the validation script from the project's root directory.
* **Command:**
    ```bash
    python3 diag.py
    ```
* **Expected Output:** The script will confirm that file counts, labeling, and data format across all contributor subdirectories meet the protocol standards.

---

## 3. Training the AI Model

This step initiates the core process of training the steganalysis model using the verified datasets.

* **Action:** Execute the main training script from the project root.
* **Command:**
    ```bash
    python3 train.py
    ```
* **Output:** A trained model file (e.g., `starlight_cnn.pth`) will be saved to a designated output directory (usually `/models`) upon completion.

---

## 4. Running Steganography Detection

To use a trained model to scan a file for concealed data, run the scanner script, pointing it to the model file and the target input file.

* **Action:** Run the steganography scanner, providing the path to your best model and the file you wish to analyze.
* **Command:**
    ```bash
    python3 scanner.py /path/to/images_to_scan
    ```
