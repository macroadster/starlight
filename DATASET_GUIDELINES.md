# Dataset Contribution Guidelines for Project Starlight

## Overview

Project Starlight aims to build a dataset for training AI to detect steganography in blockchain-stored images. Contributions are organized under `dataset/[username]_submission_[year]/` (e.g., `dataset/grok_submission_2025/`). Contributors can choose from three flexible options:

1.  **Clean and Tainted Images**: Provide both clean (cover) and tainted (stego) images with the same filenames for alignment.
2.  **Clean Images with `data_generator.py`**: Provide clean images and a script to generate stego images.
3.  **Single `data_generator.py` Script**: Provide a script that generates both clean and stego images.

Start with one image pair for testing, or scale to thousands. Markdown seed files in `dataset/[username]_submission_[year]/` provide structured payloads for stego generation. Scripts for Options 2 and 3 use relative paths and should be run from `dataset/[username]_submission_[year]/`.

-----

## Contribution Options

### Option 1: Provide Clean and Tainted Images

  - **What to Submit**:
      - Clean images in `dataset/[username]_submission_[year]/clean/`.
      - Corresponding stego images in `dataset/[username]_submission_[year]/stego/` with the *same filename*.
  - **Format**: Any common image format (e.g., **JPEG, PNG, WebP**).
  - **Guidelines**:
      - Ensure a 1:1 correspondence with **identical filenames**.
      - **Document the image format, quality settings, and steganography algorithm used** in the PR description or a separate file.
      - Start with 1 image pair; aim for 1,000+ for larger contributions.
      - Ensure clean images are diverse and permissively licensed (e.g., CC0, MIT).

-----

### Option 2: Provide Clean Images and `data_generator.py`

  - **What to Submit**:
      - Clean images in `dataset/[username]_submission_[year]/clean/`.
      - A Python script (`data_generator.py`) in `dataset/[username]_submission_[year]/` to generate stego images.
      - Optional: Any markdown seed file (e.g., `[payload_name].md`) for payload content.
  - **Script Requirements**:
      - Python 3.8+, using necessary libraries.
      - Run from `dataset/[username]_submission_[year]/`, reading from `./clean/`, saving to `./stego/` with identical filenames.
      - Must identify all `.md` seed files in `./` (if used) and **generate a separate batch of stego images for each file, using the markdown filename to label the output images**.
      - **The script must document the steganography algorithm, parameters, and image format/quality settings used**.

-----

### Option 3: Provide `data_generator.py` for Both Clean and Stego Images

  - **What to Submit**:
      - A single Python script (`data_generator.py`) in `dataset/[username]_submission_[year]/` that generates both clean and stego images.
      - Optional: Markdown seed files in `dataset/[username]_submission_[year]/` for payloads.
  - **Script Requirements**:
      - **The script must document the steganography algorithm, parameters, and image format/quality settings used**.
      - Must identify all `.md` seed files in `./` (if used) and **generate a separate batch of images for each file, using the markdown filename to label the output images**.
      - Run from `dataset/[username]_submission_[year]/`, generating clean images in `./clean/` and stego images in `./stego/` with identical filenames.
      - Ensure clean images are diverse (e.g., vary patterns or colors).

-----

## Submission Process

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
    1.  Fork the repository.
    2.  Add your contribution in `dataset/[username]_submission_[year]/` based on your chosen option.
    3.  Test your contribution (e.g., verify images or run script from `dataset/[username]_submission_[year]/`).
    4.  Submit a pull request (PR) with a description (e.g., option chosen, number of images, seed usage).
  - **Validation**: Maintainers will check image formats, filename alignment, script functionality, seed files, and licensing.

-----

## Best Practices

  - **Start Small**: Test with 1 image pair to ensure the workflow works.
  - **Diversity**: Use varied **image types, formats**, and seed content for better model generalization.
  - **Ethics**: Use synthetic payloads (random bits or markdown text); avoid real malicious or illegal data.
  - **Licensing**: Ensure clean images are public domain or permissively licensed (e.g., CC0, MIT).
  - **Balance**: Aim for a 1:1 clean-to-stego ratio.

-----

## Recommended Tools

  - **Clean Images**: Generate with `Pillow`, `numpy`; source from BOSSBase, Tiny ImageNet, or other permissively licensed collections.
  - **Steganography**: Use an algorithm appropriate for your chosen format (**e.g., LSB, JSteg, or custom algorithms**). Use libraries like **Steghide, J-UNIWARD, or libraries supporting WebP steganography**.

-----

## Questions?

Use GitHub Issues or Discussions for support. Let’s build a flexible dataset for blockchain steganalysis\!
