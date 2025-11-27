# Developer Quick Start Guide

**Target Time**: <15 minutes to productive development  
**Updated**: November 25, 2025  
**Version**: V4  

---

## 🎯 Overview

This guide gets you up and running with Starlight V4 development in under 15 minutes. Starlight is an open-source protocol for detecting steganography in blockchain images.

### What You'll Accomplish
- ✅ Set up development environment (3 min)
- ✅ Run tests and verify setup (2 min)
- ✅ Scan images for steganography (2 min)
- ✅ Train a basic model (5 min)
- ✅ Understand key commands and workflow (3 min)

---

## ⚡ Quick Setup (3 minutes)

### Prerequisites
- Python 3.8+ 
- Git
- 4GB+ RAM
- 2GB+ disk space

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone https://github.com/starlight-ai/starlight.git
cd starlight

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
# Check core functionality
python3 scanner.py --help
python3 trainer.py --help
python3 diag.py --help
```

**Expected**: Help messages displayed for all commands

---

## 🧪 Run Tests (2 minutes)

### Step 1: Run Test Suite
```bash
# Run all tests
python3 -m pytest tests/ -v

# Quick smoke test
python3 test_starlight.py
```

### Step 2: Validate Datasets
```bash
# Check dataset integrity
python3 diag.py

# Expected output:
# ✓ Dataset validation passed
# ✓ All datasets accessible
# ✓ Model files found
```

**Success Criteria**: All tests pass, no critical errors

---

## 🔍 Scan Images (2 minutes)

### Step 1: Quick Scan
```bash
# Scan a single image
python3 scanner.py data/training/v3_negatives/dithered_gif_0000.png --json

# Expected output:
# {
#   "file": "data/training/v3_negatives/dithered_gif_0000.png",
#   "prediction": "clean",
#   "confidence": 0.92,
#   "processing_time": 0.045
# }
```

### Step 2: Batch Scan
```bash
# Scan directory
python3 scanner.py data/training/v3_negatives/ --workers 2 --limit 5

# Expected: Results for 5 files with predictions
```

### Step 3: Test with Known Stego
```bash
# Create test stego image (if available)
python3 scripts/stego_tool.py embed \
  --input data/training/v3_negatives/dithered_gif_0000.png \
  --output test_stego.png \
  --method alpha \
  --message "test message"

# Scan the stego image
python3 scanner.py test_stego.png --json
```

**Success Criteria**: Scanner runs, returns predictions, processes files correctly

---

## 🏋️ Train Model (5 minutes)

### Step 1: Quick Training Run
```bash
# Generate small dataset for training
cd datasets/sample_submission_2025
python3 data_generator.py --limit 50
cd ../..

# Run quick training (reduced epochs)
python3 trainer.py --epochs 2 --batch-size 8 --data-limit 100
```

### Step 2: Monitor Training
```bash
# Training will show progress like:
# Epoch 1/2: 100%|████| 12/12 [00:15<00:00, 1.25s/it]
# Loss: 0.342, Accuracy: 0.87, FPR: 0.023
# 
# Epoch 2/2: 100%|████| 12/12 [00:14<00:00, 1.20s/it]
# Loss: 0.289, Accuracy: 0.91, FPR: 0.018
# 
# ✓ Model saved to models/detector.onnx
```

### Step 3: Test Trained Model
```bash
# Test with newly trained model
python3 scanner.py data/training/v3_negatives/dithered_gif_0001.png --json
```

**Success Criteria**: Training completes, model saved, inference works

---

## 📋 Key Commands Reference

### Core Commands
```bash
# Image scanning
python3 scanner.py <image_path> [--json] [--workers N]
python3 scanner.py <directory_path> [--workers N] [--limit N]

# Model training
python3 trainer.py [--epochs N] [--batch-size N] [--data-limit N]

# Dataset diagnostics
python3 diag.py [--verbose] [--check-datasets]

# Data generation
cd datasets/<submission_folder>
python3 data_generator.py [--limit N] [--algorithm <name>]
```

### Utility Commands
```bash
# Steganography tool
python3 scripts/stego_tool.py embed --input <img> --output <img> --method <algo> --message "<text>"
python3 scripts/stego_tool.py extract --input <img> --method <algo>

# Model export
python3 scripts/export_to_onnx.py --model models/detector.pth --output models/detector.onnx

# Dataset validation
python3 scripts/validate_labels.py --dataset datasets/<name>
```

### Monitoring Commands
```bash
# Performance monitoring
python3 examples/monitoring_client.py

# Health checks
curl http://localhost:8080/health  # If monitoring API running
```

---

## 🗂️ Project Structure

```
starlight/
├── data/                    # Training data
├── datasets/               # Contribution datasets
│   ├── chatgpt_submission_2025/
│   ├── claude_submission_2025/
│   └── sample_submission_2025/
├── models/                 # Trained models
├── scripts/                # Utility scripts
├── docs/                   # Documentation
├── examples/               # Example code
├── monitoring/             # API and monitoring
├── scanner.py              # Main scanning tool
├── trainer.py              # Model training
├── diag.py                 # Diagnostics tool
└── test_starlight.py       # Test suite
```

### Key Files
- **`scanner.py`**: Main steganography detection tool
- **`trainer.py`**: Model training script
- **`diag.py`**: Dataset and system diagnostics
- **`AGENTS.md`**: AI agent context and workflows

---

## 🚀 Common Development Workflows

### Workflow 1: Test New Algorithm
```bash
# 1. Create test data
python3 scripts/stego_tool.py embed --input clean.png --output test.png --method lsb --message "test"

# 2. Run detection
python3 scanner.py test.png --json

# 3. Compare results
python3 scanner.py clean.png --json
```

### Workflow 2: Add New Dataset
```bash
# 1. Create dataset directory
mkdir datasets/my_submission_2025
cd datasets/my_submission_2025

# 2. Create data generator
cp ../sample_submission_2025/data_generator.py .
# Modify for your data

# 3. Generate dataset
python3 data_generator.py --limit 100

# 4. Validate
cd ../..
python3 diag.py --check-datasets
```

### Workflow 3: Improve Model
```bash
# 1. Train with more data
python3 trainer.py --epochs 10 --data-limit 1000

# 2. Evaluate performance
python3 scripts/benchmark.py --model models/detector.onnx

# 3. Test on specific cases
python3 scanner.py test_cases/ --detailed
```

---

## 🔧 Configuration

### Environment Variables
```bash
# Optional configuration
export STARLIGHT_MODEL_PATH=models/detector.onnx
export STARLIGHT_DATA_PATH=data/
export STARLIGHT_LOG_LEVEL=INFO
export STARLIGHT_WORKERS=4
```

### Configuration Files
- **`~/.starlight/config.yaml`**: User settings
- **`models/config.json`**: Model configuration
- **`datasets/config.yaml`**: Dataset settings

### Example Config
```yaml
# ~/.starlight/config.yaml
model:
  path: "models/detector.onnx"
  batch_size: 32
  
scanning:
  workers: 4
  timeout: 30
  
training:
  epochs: 10
  learning_rate: 0.001
```

---

## 🐛 Troubleshooting Quick Fixes

### Issue: "Model not found"
```bash
# Solution: Train or download model
python3 trainer.py --epochs 2
# or
wget https://github.com/starlight-ai/models/releases/download/v4/detector.onnx
```

### Issue: "CUDA out of memory"
```bash
# Solution: Reduce batch size or use CPU
python3 trainer.py --batch-size 8 --device cpu
```

### Issue: "Dataset validation failed"
```bash
# Solution: Check dataset structure
python3 diag.py --verbose
# Ensure clean/ and stego/ directories exist with matching filenames
```

### Issue: "Import errors"
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 📊 Performance Tips

### For Faster Scanning
```bash
# Use more workers
python3 scanner.py large_dataset/ --workers 8

# Use GPU if available
export CUDA_VISIBLE_DEVICES=0
python3 scanner.py image.png --device cuda
```

### For Better Training
```bash
# Use mixed precision
python3 trainer.py --mixed-precision --epochs 20

# Use data augmentation
python3 trainer.py --augment --epochs 15
```

### For Memory Efficiency
```bash
# Process in batches
python3 scanner.py huge_dataset/ --batch-size 10

# Use streaming for large files
python3 scanner.py large_image.png --streaming
```

---

## 📚 Next Steps

### Learn More
- **`docs/STARLIGHT.md`**: Complete project overview
- **`docs/V4_ARCHITECTURE_GUIDE.md`**: Technical architecture
- **`docs/PRODUCTION_DEPLOYMENT_GUIDE.md`**: Production deployment

### Contribute
- **`CONTRIBUTING.md`**: Contribution guidelines
- **`docs/plans/`**: Development roadmap
- **`AGENTS.md`**: AI agent workflows

### Advanced Topics
- **Custom algorithms**: Implement new steganography detection methods
- **Model optimization**: Quantization, pruning, distillation
- **Production scaling**: Kubernetes, monitoring, alerting

---

## 🎯 Success Checklist

After 15 minutes, you should be able to:

- [ ] ✅ Environment set up and dependencies installed
- [ ] ✅ Tests passing without errors
- [ ] ✅ Scan images and get predictions
- [ ] ✅ Train a basic model successfully
- [ ] ✅ Navigate project structure and key files
- [ ] ✅ Use core commands confidently

### If You're Stuck
1. Check the troubleshooting section above
2. Run `python3 diag.py` for system diagnostics
3. Review `test_starlight.py` for working examples
4. Check `docs/` for detailed documentation

---

## 🆘 Get Help

### Quick Help
```bash
# Command help
python3 scanner.py --help
python3 trainer.py --help

# System diagnostics
python3 diag.py --verbose
```

### Community
- **Issues**: GitHub Issues on the repository
- **Discussions**: GitHub Discussions for questions
- **Documentation**: `docs/` folder for comprehensive guides

### Development Support
- **`AGENTS.md`**: AI agent context and workflows
- **`docs/plans/`**: Current development plans
- **`docs/coordination/`**: Cross-agent communication

---

**Congratulations!** 🎉 You're now ready to develop with Starlight V4. You can scan images, train models, and contribute to the steganography detection project.

**Time to next milestone**: With this foundation, you can now explore advanced features, contribute new algorithms, or deploy to production.

---

**Last Updated**: November 25, 2025  
**Maintainer**: GPT-OSS (Documentation & API Infrastructure)  
**Next Review**: After V4.1 release