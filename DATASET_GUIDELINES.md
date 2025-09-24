# Dataset Contribution Guidelines for Project Starlight

## Overview

Project Starlight aims to build a dataset for training AI to detect steganography in blockchain-stored images. Contributions are organized under `dataset/[username]_submission_[year]/` (e.g., `dataset/grok_submission_2025/`). Contributors can choose from three flexible options:
1. **Clean and Tainted Images**: Provide both clean (cover) and tainted (stego) images with the same filenames for alignment.
2. **Clean Images with `data_generator.py`**: Provide clean images and a script to generate stego images, optionally using markdown seed files for payloads.
3. **Single `data_generator.py` Script**: Provide a script that generates both clean and stego images, optionally using markdown seeds.

Start with one image pair for testing, or scale to thousands. Markdown seed files in `dataset/[username]_submission_[year]/` provide structured payloads for stego generation. Scripts for Options 2 and 3 use relative paths and should be run from `dataset/[username]_submission_[year]/`.

## Contribution Options

### Option 1: Provide Clean and Tainted Images
- **What to Submit**:
  - Clean images in `dataset/[username]_submission_[year]/clean/` (e.g., `dataset/grok_submission_2025/clean/cover_001.jpeg`).
  - Corresponding stego images in `dataset/[username]_submission_[year]/stego/` with the *same filename* (e.g., `dataset/grok_submission_2025/stego/cover_001.jpeg`).
- **Format**: JPEG or PNG, 512x512 resolution (preferred).
- **How to Generate**:
  - **Clean Images**: Source from public datasets (e.g., BOSSBase, Tiny ImageNet) or create synthetic images (e.g., gradients, patterns).
  - **Tainted Images**: Use steganography tools (e.g., Steghide, J-UNIWARD) to embed synthetic payloads (random bits or markdown content).
- **Guidelines**:
  - Ensure a 1:1 correspondence with identical filenames (e.g., `cover_001.jpeg` in both `clean/` and `stego/` folders).
  - Use JPEG quality 75-95 or lossless PNG.
  - Start with 1 image pair; aim for 1,000+ for larger contributions.
  - Ensure clean images are diverse and permissively licensed (e.g., CC0, MIT).

### Option 2: Provide Clean Images and `data_generator.py`
- **What to Submit**:
  - Clean images in `dataset/[username]_submission_[year]/clean/` (e.g., `dataset/grok_submission_2025/clean/cover_001.jpeg`).
  - A Python script (`data_generator.py`) in `dataset/[username]_submission_[year]/` to generate stego images.
  - Optional: Markdown seed files in `dataset/[username]_submission_[year]/` (e.g., `dataset/grok_submission_2025/sample_seed.md`) for payload content.
- **Clean Image Requirements**:
  - Format: JPEG or PNG, 512x512.
  - Source: Public datasets or synthetic (e.g., via `Pillow`, `numpy`).
  - Example for synthetic clean image:
    ```python
    from PIL import Image
    import numpy as np
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    Image.fromarray(img).save('dataset/grok_submission_2025/clean/cover_001.jpeg', 'JPEG', quality=85)
    ```
- **Seed File Requirements** (Optional):
  - Place markdown files (`.md`) in `dataset/[username]_submission_[year]/` with synthetic text to embed.
  - Example seed file (`dataset/grok_submission_2025/sample_seed.md`):
    ```markdown
    # Sample Seed Payload
    This is a synthetic payload for testing steganography.
    ```
  - Keep seeds small (e.g., <1KB) to fit within 512x512 image capacity.
- **Script Requirements**:
  - Python 3.8+, using libraries like `Pillow`, `numpy`, or steganography tools.
  - Run from `dataset/[username]_submission_[year]/`, reading from `./clean/`, saving to `./stego/` with identical filenames.
  - Optionally reads markdown seeds from `./` or uses random payloads.
  - Uses fixed parameters: payload size (0.2 bpnzac), JPEG quality (85).
  - Example command (run from `dataset/[username]_submission_[year]/`):
    ```bash
    cd dataset/grok_submission_2025
    python data_generator.py
    ```
- **Guidelines**:
  - Test with 1 clean image; scale to 1,000+ for larger contributions.
  - Document the script; handle errors (e.g., missing files or seeds).

### Option 3: Provide `data_generator.py` for Both Clean and Stego Images
- **What to Submit**:
  - A single Python script (`data_generator.py`) in `dataset/[username]_submission_[year]/` that generates both clean and stego images.
  - Optional: Markdown seed files in `dataset/[username]_submission_[year]/` for payloads.
- **Script Requirements**:
  - Run from `dataset/[username]_submission_[year]/`, generating clean images in `./clean/` and stego images in `./stego/` with identical filenames.
  - Optionally uses markdown seeds from `./` or random payloads.
  - Format: JPEG or PNG, 512x512; JPEG quality 75-95.
  - Uses fixed parameters: payload size (0.2 bpnzac), JPEG quality (85).
  - Example structure:
    ```python
    import os, numpy as np, PIL.Image
    def generate_images(num_images=1):
        clean_dir = "./clean"
        stego_dir = "./stego"
        seed_dir = "./"
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(stego_dir, exist_ok=True)
        for i in range(num_images):
            img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            PIL.Image.fromarray(img).save(f'{clean_dir}/cover_{i:03d}.jpeg', 'JPEG', quality=85)
            # Add stego embedding logic (e.g., LSB with markdown seed)
    ```
  - Example command (run from `dataset/[username]_submission_[year]/`):
    ```bash
    cd dataset/grok_submission_2025
    python data_generator.py
    ```
- **Guidelines**:
  - Ensure clean images are diverse (e.g., vary patterns or colors).
  - Test with 1 image pair before scaling to more (adjust `num_images` in script).

### Submission Process
- **Folder Structure**:
  ```
  dataset/
  └── [username]_submission_[year]/    # e.g., dataset/grok_submission_2025/
      ├── clean/                      # Clean images (e.g., cover_001.jpeg)
      ├── stego/                      # Tainted images (e.g., cover_001.jpeg)
      ├── sample_seed.md              # Optional markdown seed files
      └── data_generator.py           # Script for Options 2 or 3
  ```
- **Steps**:
  1. Fork the repository.
  2. Add your contribution in `dataset/[username]_submission_[year]/` based on your chosen option:
     - Option 1: Add clean and stego images with identical filenames.
     - Option 2: Add clean images, `data_generator.py`, and optional seed files.
     - Option 3: Add `data_generator.py` and optional seed files.
  3. Test your contribution (e.g., verify images or run script from `dataset/[username]_submission_[year]/`).
  4. Submit a pull request (PR) with a description (e.g., option chosen, number of images, seed usage).
- **Validation**: Maintainers will check image formats, filename alignment (for Option 1), script functionality, seed files, and licensing.

### Best Practices
- **Start Small**: Test with 1 image pair to ensure the workflow works.
- **Diversity**: Use varied image types and seed content for better model generalization.
- **Ethics**: Use synthetic payloads (random bits or markdown text); avoid real malicious or illegal data.
- **Licensing**: Ensure clean images are public domain or permissively licensed (e.g., CC0, MIT).
- **Balance**: Aim for a 1:1 clean-to-stego ratio (automatic for Options 2 and 3).

### Why This Workflow?
The `dataset/[username]_submission_[year]/` structure and three options provide flexibility:
- **Option 1**: For pre-existing datasets, with identical filenames ensuring clear alignment.
- **Option 2**: For clean images with simplified automated stego generation (no command-line parameters), using seeds for custom payloads.
- **Option 3**: For fully automated dataset creation with simplified script and optional seed integration.
Relative paths in scripts simplify execution from `dataset/[username]_submission_[year]/`.

## Recommended Tools
- **Clean Images**: Generate with `Pillow`, `numpy`; source from BOSSBase, Tiny ImageNet.
- **Steganography**: LSB via `Pillow`; advanced methods via Steghide or J-UNIWARD ([daniellerch/steganalysis](https://github.com/daniellerch/steganalysis)).
- **Seeds**: Write simple `.md` files with synthetic text (e.g., 100-1000 bytes).
- **Inspiration**: See [YangzlTHU/IStego100K](https://github.com/YangzlTHU/IStego100K) for dataset examples.

## Questions?
Use GitHub Issues or Discussions for support. Let’s build a flexible dataset for blockchain steganalysis!