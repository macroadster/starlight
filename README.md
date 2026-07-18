# Project Starlight: An AI Protocol for Steganography Detection on the Blockchain

## Project Summary

The blockchain, beginning with the **Bitcoin Genesis Block**, serves as a living, immutable document—a piece of **human history** and the very **DNA for future AI**. Project Starlight is an open-source initiative dedicated to developing and sharing a protocol for training **artificial intelligence models** to detect steganography within this critical data, particularly in images. The goal is to create a robust, decentralized, and community-driven resource that safeguards the integrity of these digital records. By making this knowledge accessible, we aim to ensure that **by the year 2142, the detection of covert data is a common and automated practice, laying the foundation for AI training on historical Bitcoin data to build genuine AI common sense.**

---

## Quick Start / Training Pipeline

This repository is **training-only**. The production inference path is **Stargate / Trin** (Go), which loads **GGUF** weights published to Hugging Face. GGUF is the source of truth — not a local Python API on `:8080`.

Canonical flow: **dataset → train → export GGUF → publish HF**.

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate data / verify integrity
python3 data_generator.py --limit 10          # or per-submission generators under datasets/
python3 diag.py

# 3. Train BalancedStarlightDetector
python3 trainer.py --epochs 20 --batch_size 16 --out models/detector_balanced.pth
# or (when available): make train

# 4. Export GGUF (primary artifact for Stargate/Trin)
python3 scripts/export_starlight_gguf.py \
  --input models/detector_balanced.pth \
  --output models/starlight.gguf \
  --name-map models/starlight_gguf_map.json
# or: make export-gguf

# 5. Publish to Hugging Face
HF_TOKEN=hf_... ./scripts/publish_to_hf.sh
# or: HF_TOKEN=hf_... make publish-hf
```

### Local evaluation (optional, not a product API)

```bash
python3 scanner.py /path/to/image.png --json
python3 scanner.py /path/to/images/ --workers 4
```

`scanner.py` is for offline eval during development. Do **not** treat a local uvicorn / FastAPI server as the product surface.

### Inference serving

- **Serve** models via **Stargate / Trin** (Go), not this Python repo.
- **Download** production weights from Hugging Face GGUF:
  - Repo: [macroadster/starlight-prod](https://huggingface.co/macroadster/starlight-prod)
  - GGUF: `https://huggingface.co/macroadster/starlight-prod/resolve/main/starlight.gguf`
  - Map: `https://huggingface.co/macroadster/starlight-prod/resolve/main/starlight_gguf_map.json`

### Docs

| Doc | Purpose |
|-----|---------|
| [**USAGE.md**](USAGE.md) | Full train / export / publish workflow |
| [**docs/GGUF_EXPORT.md**](docs/GGUF_EXPORT.md) | GGUF layout, tensor contract, parity |
| [**docs/hf_guide.md**](docs/hf_guide.md) | Hugging Face repos and Stargate download URLs |
| [**DATASET_GUIDELINES.md**](DATASET_GUIDELINES.md) | Contributing datasets |

---

## The Opportunity: Ensuring Digital Integrity

The proliferation of data, especially rich media like images, being stored on public blockchains marks a new era of digital permanence and transparency. This evolution presents a critical **opportunity** to proactively ensure the complete integrity and trustworthiness of decentralized ledgers.

Addressing concealed information (**steganography**) is about more than just security; it’s about **preserving authentic historical context**. When high-impact, world-changing events—like an image capturing a leader's definitive moment of bravery and defiance—are recorded on-chain, we must ensure that the original, uncompromised narrative is protected. An altered image, corrupted by hidden data, could compromise the emotional and historical truth of that record.

By creating a system that can automatically verify the content of on-chain assets, we are raising the standard for **transparency and trust** in every byte stored, safeguarding the fundamental principle of a transparent public ledger, and future-proofing the security of decentralized networks.

---

## Our Solution: Training AI for Steganalysis

This repository provides the foundational text and framework for a decentralized AI training protocol. Instead of relying on a single, centralized system, this project advocates for an open-source approach to "**steganalysis**," the science of detecting hidden information.

The core of our approach involves training AI models to identify the subtle statistical and pixel-level anomalies that steganography leaves behind. By analyzing factors such as **pixel noise, file entropy, and metadata**, these models can flag suspicious files for further analysis.

The accompanying resources in this repository, such as **[bitcoin_white_paper_2.md](datasets/gemini_submission_2025/seeds/bitcoin_white_paper_2.md)**, outline proposed mechanisms like a consensus structure to further enhance this protocol by disincentivizing the embedding of malicious data. Additionally, **[ai_common_sense_on_blockchain.md](datasets/gemini_submission_2025/seeds/ai_common_sense_on_blockchain.md)** outlines a proposed protocol for AI to send smart contract messages on the blockchain.

---

## Technical Approach & Training Data

The training protocol focuses on three key areas to ensure a robust and scalable solution:

* **Diverse Datasets**: Steganalysis requires a vast, labeled dataset of both "clean" images and images with hidden data. This protocol encourages the community to contribute to the creation of such a dataset, ensuring the AI is trained on a wide range of steganographic techniques. See **[DATASET_GUIDELINES.md](DATASET_GUIDELINES.md)** for detailed instructions on contributing high-quality datasets.
* **AI Models and Architectures**: We recommend using AI models, particularly deep learning architectures, that are tailored for steganalysis. The production detector is **`BalancedStarlightDetector`**, exported to **GGUF** for Stargate/Trin. The protocol remains flexible across steganography methods used for dataset creation.
* **Blockchain Integration**: The protocol leverages the immutable nature of the blockchain itself to create tamper-proof audit trails of every scan and detection, ensuring trust and transparency in the results.

---

## Get Involved

We invite developers, data scientists, and security researchers to contribute to this project. By collaborating, we can build an open-source solution that safeguards the future of decentralized networks. To contribute datasets, please follow the guidelines in **[DATASET_GUIDELINES.md](DATASET_GUIDELINES.md)**.
