Project Starlight: Design Document
Version: 1.0 Date: 2025-10-24 Author: Gemini

1. Introduction
Project Starlight is a steganography detection and extraction suite. Its primary purpose is to analyze image files to identify the presence of hidden data and determine the specific algorithm used for embedding.

The system is designed as a multi-modal classifier that leverages a hybrid machine learning architecture (CNN + Random Forest) to achieve high accuracy across diverse file types and steganography techniques.

The project is divided into three core components:

Trainer: A script to train the detection models using a pre-generated dataset.

Scanner: A command-line tool to scan images and classify the steganography algorithm used.

Extractor: A utility to extract the hidden message once the algorithm is identified.

2. Goals and Scope
2.1. Core Goals

Trainer (train.py): Develop a script that trains a hybrid classifier to detect steganography.

Scanner (scan.py): Create a script that uses the trained model to predict the algorithm used on a target image (or "Clean").

Extractor (extract.py): Build a script that, when given an image and an algorithm, extracts the hidden payload.

2.2. In-Scope Algorithms

The system will support the detection and extraction of payloads for the following six algorithms:

LSB (Least Significant Bit): Messages hidden in the least significant bits of RGB(A) pixel values.

DCT (Discrete Cosine Transform): Messages hidden in the LSBs of quantized DCT coefficients (common in JPEGs).

EXIF: Payloads stored in image metadata tags (e.g., UserComment, ImageDescription).

EOI (End of Image): Data appended to the file after the standard EOI marker (e.g., FFD9 for JPEG).

Alpha: Messages hidden in the LSBs of the alpha (transparency) channel.

Palette: Messages hidden in the LSBs of palette indices for palettized images (e.g., 8-bit PNGs, GIFs).

2.3. Supported File Types

PNG

JPEG

WebP

GIF

BMP

2.4. Out of Scope

Detection of novel or unknown steganography algorithms.

Detection or extraction from video or audio files.

Breaking or bypassing encryption (the extractor will retrieve the raw payload, which may still be encrypted).

3. System Architecture
The core of Starlight is a hybrid ensemble classifier that combines a Convolutional Neural Network (CNN) with a Random Forest (RF) classifier. This architecture is specifically chosen to handle the diverse nature of steganography: some methods alter pixel statistics (for the CNN), while others alter file metadata or structure (for the RF).

3.1. Architectural Flow

Input: An image file is provided to the Scanner.

Feature Extraction: The image is processed by two parallel feature extractors:

Structural Feature Extractor: This component analyzes the file's metadata and structure. It does not decompress the pixel data. It generates a feature vector for the Random Forest.

Pixel/Coefficient Feature Extractor: This component decompresses the image to analyze its pixel data. It generates an input tensor (or tensors) for the CNN.

Parallel Classification:

Random Forest (RF) Classifier: Receives the structural feature vector. It is highly effective at identifying EXIF and EOI attacks, as wells as filtering by file properties (e.g., has_alpha, is_palettized).

Convolutional Neural Network (CNN): Receives the pixel/coefficient data. It is trained to spot subtle statistical anomalies in spatial or frequency domains, making it ideal for LSB, DCT, Alpha, and Palette detection.

Ensemble Classifier (Feedback Loop):

The probability outputs from both the RF and the CNN are fed as inputs into a final, simple meta-classifier (e.g., Logistic Regression or a small Multi-Layer Perceptron).

This "feedback loop" or ensemble model weighs the evidence from both specialized models to make a final prediction (e.g., LSB, DCT, EXIF, EOI, Alpha, Palette, or Clean).

This design allows the model to "reason." For example, if the RF is 99% certain an EOI payload exists (based on file size), it can override a CNN that sees "clean" pixels. Conversely, if the RF sees no structural issues, but the CNN reports a 95% probability of LSB, the ensemble will trust the CNN.

4. Component Design
4.1. Component 1: Trainer (train.py)

This script trains and serializes the three models (RF, CNN, Ensemble).

Input: Path to the dataset root (e.g., datasets/). It must recursively find clean and stego subdirectories.

Data Labeling:

Images in .../clean/ are labeled as Clean.

Images in .../stego/ must have their algorithm encoded in their filename or a supplementary metadata file (e.g., image_001_lsb.png, image_002_exif.jpg). This is a critical requirement for supervised training.

Process:

Data Ingestion: Load all image paths and their corresponding labels.

Feature Extraction: For every image, generate and cache both the structural feature vector and the pixel/coefficient tensor.

Stage 1: RF Training: Train the Random Forest classifier on the structural feature vectors only.

Stage 2: CNN Training: Train the CNN on the pixel/coefficient tensors.

Stage 3: Ensemble Training:

Freeze the weights of the trained RF and CNN.

Pass the entire dataset through both models to get their prediction probabilities.

Train the Ensemble Classifier using these probabilities as its input features, with the original image labels as the target.

Output: Serialized model files (e.g., starlight_cnn.h5, starlight_rf.pkl, starlight_ensemble.pkl).

4.2. Component 2: Scanner (scan.py)

This is the main user-facing tool for detection.

Input: A path to a single image or a directory of images.

Process:

Load the three trained models.

For each image:

Execute the Structural Feature Extractor to get the RF input vector.

Execute the Pixel/Coefficient Feature Extractor to get the CNN input tensor(s).

Feed the vector into the RF to get Prob_RF.

Feed the tensor(s) into the CNN to get Prob_CNN.

Feed Prob_RF and Prob_CNN into the Ensemble Classifier.

Output: A JSON or console report mapping each file to its predicted class and a confidence score.

JSON
{
  "scan_results": [
    {
      "file": "image1.jpg",
      "prediction": "DCT",
      "confidence": 0.92
    },
    {
      "file": "image2.png",
      "prediction": "Clean",
      "confidence": 0.99
    },
    {
      "file": "image3.gif",
      "prediction": "Palette",
      "confidence": 0.88
    }
  ]
}
4.3. Component 3: Extractor (extract.py)

This is a collection of discrete algorithm-specific extraction modules. It does not use the AI models.

Input:

--image: Path to the target image.

--algorithm: The algorithm to use (e.g., LSB, EXIF). This is required.

--output: Path to write the extracted payload.

Process: A large switch statement or factory pattern will invoke the correct extractor module:

LSB: Loads the image (e.g., with Pillow), iterates pixel-by-pixel, and reads the LSB from each R, G, and B channel. Assembles bits into bytes.

DCT: Uses a JPEG library (e.g., jpegio) to access raw DCT coefficients, reads the LSBs, and assembles them.

EXIF: Uses a metadata library (e.g., exiftool or Pillow.ExifTags) to read common payload tags and dump their contents.

EOI: Opens the file in binary mode, seeks to the EOI marker (FFD9), and dumps all subsequent bytes.

Alpha: Loads the image, confirms it has an alpha channel, and extracts the LSBs from that channel only.

Palette: Loads the image, confirms it is palettized, reads the list of pixel indices, and extracts LSBs from the indices themselves.

Output: The raw extracted data is written to the specified output file.

5. Data Model and Feature Engineering
This is the most critical part of the design, as it must handle the inconsistencies between file types (e.g., JPEGs have no alpha channel or palette).

5.1. Structural Feature Vector (for Random Forest)

This is a flat vector of numerical and boolean features. This model's primary job is to classify EXIF and EOI and to provide file-type context to the ensemble.

Example Features:

file_size_bytes: (int)

image_width: (int)

image_height: (int)

image_mode: (enum: RGB, RGBA, P, CMYK, etc.)

is_jpeg: (bool)

is_png: (bool)

is_gif: (bool)

is_webp: (bool)

is_bmp: (bool)

has_alpha_channel: (bool)

has_palette: (bool)

palette_color_count: (int)

has_exif_data: (bool)

exif_data_size_bytes: (int)

eoi_payload_size_bytes: (int) Size of data found after the EOI marker.

5.2. Pixel/Coefficient Tensors (for CNN)

The CNN will be a multi-input model to handle the fundamental difference between spatial-domain (PNG, BMP) and frequency-domain (JPEG) data.

Input 1: Spatial Tensor (for LSB, Alpha, Palette)

Shape: (width, height, channels)

Content: For PNG/GIF/BMP, this tensor contains the normalized pixel values. Channels could include R, G, B, Alpha (if present), and pre-calculated LSB bit-planes.

Handling: For JPEGs, this tensor would be the decompressed pixel data.

Input 2: DCT Coefficient Tensor (for DCT)

Shape: A 2D matrix of quantized DCT coefficients.

Content: For JPEGs, this tensor is populated using a JPEG library.

Handling: For non-JPEG files (PNG, GIF, BMP), this tensor will be all zeros, effectively disabling this branch of the CNN.

This multi-input approach allows the CNN to train specialized "branches" for different data types, and the Ensemble model will learn (based on the RF's is_jpeg feature) which branch's output to trust.

6. Technology Stack (Proposed)
Python 3.10+

AI/ML:

tensorflow or pytorch: For the CNN.

scikit-learn: For the Random Forest ( RandomForestClassifier) and the Ensemble (LogisticRegression).

Image Processing:

Pillow (PIL): For loading/saving images, reading basic metadata, and LSB/Alpha/Palette extraction.

jpegio: For low-level access to JPEG DCT coefficients.

Metadata:

exifread or a wrapper for exiftool: For robust EXIF tag parsing.

Data Handling:

numpy: For all numerical and tensor operations.

pandas: For managing feature vectors during training.
