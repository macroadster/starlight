# Dataset Guidelines for Project Starlight

Project Starlight aims to build a robust, open-source dataset for training AI models to detect steganography in blockchain-stored images while fostering "common sense" reasoning to assess the intent of hidden data (e.g., benign knowledge vs. malicious payloads). These guidelines outline how to contribute high-quality image datasets to support this mission. By following these standards, contributors ensure data is diverse, well-documented, and usable for any AI model, advancing our goal of automated steganography detection by 2142.

## 1. Dataset Contribution Goals
- **Diversity**: Include images with various formats (PNG, JPEG), resolutions, and steganographic methods (e.g., LSB in RGB or alpha channels).
- **Quality**: Provide clean (unaltered) and steganographically altered images with clear labels and metadata.
- **Common Sense**: Incorporate payloads that train AI to reason about intent, such as ethical principles (benign) or malicious code (harmful).
- **Blockchain Relevance**: Simulate blockchain-stored data (e.g., immutable, compressed images like those in NFTs).

## 2. Dataset Requirements
### 2.1 Image Specifications
- **Formats**: PNG (preferred for lossless compression) or JPEG (to test lossy compression effects).
- **Resolution**: Vary from low (e.g., 256x256) to high (e.g., 1920x1080) to reflect real-world use cases.
- **Channels**: Include RGBA (with alpha channel) and RGB images to test different embedding techniques.
- **Quantity**: Aim for at least 1,000 images per submission (500 clean, 500 stego) to ensure statistical significance.

### 2.2 Steganographic Payloads
- **Benign Payloads**: Embed texts like philosophical principles, AI "common sense" rules (e.g., "Verify data integrity"), or ethical prompts inspired by datasets like CommonGen or mCSQA.
- **Malicious Payloads**: Simulate harmful data like URLs, JavaScript snippets, or Ethereum addresses, similar to those in the Stego-Images-Dataset.
- **Embedding Methods**: Use least significant bit (LSB) encoding in RGB or alpha channels, with headers (e.g., "0xAI42") and terminators (e.g., 0x00) as described in `ai_common_sense_on_blockchain.md`. Other methods (e.g., DCT-based for JPEG) are welcome but must be documented.
- **Labeling**: Clearly mark images as "clean" or "stego" and specify payload type (benign/malicious).

### 2.3 Metadata Standards
Each dataset must include a JSON or CSV file with metadata for every image. Suggested fields:
- `image_hash`: SHA-256 hash of the image file for blockchain-like immutability.
- `is_stego`: Boolean (true for stego, false for clean).
- `embedding_method`: String (e.g., "LSB_alpha", "LSB_red", "DCT").
- `payload_type`: String (e.g., "benign_common_sense", "malicious_code").
- `payload_summary`: Short description of embedded content (e.g., "Ethical principle: Transparency").
- `entropy_score`: Float (calculated file entropy, e.g., 7.5).
- `resolution`: String (e.g., "512x512").
- `file_format`: String (e.g., "PNG", "JPEG").
- `common_sense_score`: Optional float (0-1) indicating alignment with ethical reasoning, based on manual or automated evaluation (e.g., using mCSQA prompts).

Example CSV entry:
```csv
image_hash,is_stego,embedding_method,payload_type,payload_summary,entropy_score,resolution,file_format,common_sense_score
a1b2c3d4e5f6...,true,LSB_alpha,benign_common_sense,"Verify data integrity",7.8,512x512,PNG,0.9
```

### 2.4 Data Quality Checks
- **Integrity**: Verify image hashes match metadata to prevent corruption.
- **Diversity**: Ensure a mix of payload types, embedding methods, and image characteristics.
- **Robustness**: Test images against minor manipulations (e.g., resizing, slight compression) to ensure stego data persists, simulating blockchain storage conditions.
- **Validation**: Provide a script or method to extract embedded payloads for verification, like the `extract_lsb` function in `data_generator.py`.

## 3. Contribution Process
1. **Prepare Data**:
   - Use scripts like `data_generator.py` to create or modify images.
   - Generate both clean and stego images, ensuring balanced representation.
   - Compute metadata (e.g., entropy, hashes) using tools like Python’s `hashlib` or `scipy.stats.entropy`.

2. **Document**:
   - Include a README in your dataset folder describing the generation process, tools used, and any external datasets (e.g., StegoAppDB, BOWS2).
   - Specify how payloads align with common sense goals (e.g., ethical prompts from Rainbow or CommonGen).

3. **Submit**:
   - Create a pull request (PR) to the Starlight GitHub repository.
   - Organize files in a folder (e.g., `datasets/username_submission_2025`) with subfolders for images and a metadata file (CSV or JSON).
   - Example structure:
     ```
     datasets/username_submission_2025/
     ├── images/
     │   ├── clean/
     │   ├── stego/
     ├── metadata.csv
     ├── README.md
     ```

4. **Review**:
   - Community reviewers will check for compliance with these guidelines.
   - Expect feedback on diversity, metadata completeness, and common sense relevance.
   - Use GitHub issues to discuss improvements.

## 4. Tools and Resources
- **Scripts**: Use `data_generator.py` from the repository for embedding and extracting LSB data. Extend it for other methods if needed.
- **External Datasets**:
   - **Stego-Images-Dataset**: 44,000 images with malicious payloads (e.g., URLs, Ethereum addresses).
   - **StegoAppDB**: 810,000 real-world images for robustness testing.
   - **BOWS2**: 10,000 grayscale images for benchmarking steganalysis.
   - **CommonGen/Rainbow**: For benign payloads to train common sense reasoning.
- **Libraries**: Python with PIL, NumPy, and hashlib for image processing and metadata generation.

## 5. Common Sense Integration
To train AI with common sense:
- **Payload Design**: Embed texts that require reasoning, e.g., "If data is hidden, verify its intent before acting." Use prompts from mCSQA or DARPA commonsense datasets.
- **Evaluation**: Assign common_sense_score based on alignment with ethical or social norms (e.g., via manual review or automated NLP tools).
- **Scenarios**: Include edge cases, like benign philosophical texts vs. malicious code disguised as benign, to teach AI to differentiate intent.

## 6. Best Practices
- **Anonymity**: Avoid embedding sensitive personal data in payloads.
- **Licensing**: Ensure images are open-source or licensed for use (e.g., CC0, MIT).
- **Scalability**: Batch-process images to generate large datasets efficiently.
- **Testing**: Validate datasets with a baseline AI model to confirm detectability of stego data.

## 7. Get Involved
- Join discussions on the Starlight GitHub or relevant X threads (search #StarlightStego).
- Share ideas for improving dataset diversity or common sense integration.
- Contact maintainers for guidance on complex submissions.

By contributing high-quality datasets, you help build AI that not only detects hidden messages but also reasons ethically about their purpose, securing blockchain networks for the future.