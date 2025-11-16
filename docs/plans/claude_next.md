# Claude CLI - Project Starlight Action Plan (Refreshed)
**Updated: November 15, 2025**
**Status: Production Stable, Research Track Active**

---

## 🎯 CRITICAL CONTEXT (READ THIS FIRST)

### Current System State
- **Production Model**: `detector_balanced.onnx` + special cases
- **Performance**: 0.32% FP rate, 96.40% detection rate ✅
- **Status**: FULLY OPERATIONAL - verified 2025-11-15
- **Dataset**: 22,630+ files across 6 datasets (6,101 stego pairs verified)

### Your Mission
You are Claude working in **Track B: Research Path** - pursuing generalization without special cases. The production system (Track A) is stable and requires no immediate changes.

### Key Lesson Learned
**Special cases elimination failed** - 17.82% FP rate without them vs 0.32% with them. Special cases encode essential domain knowledge that models cannot yet learn from training data alone.

---

## 📋 Two-Track Strategy

### Track A: Production (Stable - ChatGPT/Gemini ownership)
- Maintain `detector_balanced.onnx` + special cases
- Monitor weekly health checks
- Integrate Grok's multi-format EXIF/EOI

### Track B: Research (Your Focus)
- Fix dataset quality issues
- Develop V3/V4 architecture
- Train models that learn domain constraints
- Timeline: 18-24 months to true generalization

---

## 🔥 Your Primary Deliverable: Dataset Reconstruction

### Problem Identified
Current training data has fundamental issues preventing generalization:

1. **Invalid Labels**: Alpha steganography labels on RGB images
2. **Format Mismatches**: Clean images don't match stego format distribution
3. **Corrupted Signals**: Augmentations destroying LSB data
4. **Missing Negatives**: No examples teaching what steganography is NOT

### Required Actions

#### Week 1: Dataset Validation & Repair Pipeline

**Monday: Assessment**
```bash
cd ~/starlight

# Analyze current datasets
python scripts/analyze_datasets.py --output docs/claude/dataset_audit.json

# Check for impossible labels
python scripts/validate_labels.py \
  --datasets "datasets/*_submission_*/stego" \
  --report docs/claude/invalid_labels.md
```

**Expected Findings**:
- RGB images labeled as alpha steganography
- Format distribution mismatches
- Extraction failures
- Label inconsistencies between agents

**Tuesday: Create Repair Pipeline**

Build `scripts/dataset_repair.py`:

```python
class DatasetRepairer:
    """Fix fundamental dataset quality issues"""
    
    def validate_alpha_labels(self, img_path, label):
        """Remove alpha labels from RGB images"""
        img = Image.open(img_path)
        if label == 'alpha' and img.mode != 'RGBA':
            return None  # Invalid label
        return label
    
    def verify_extraction(self, img_path, method):
        """Confirm steganography is actually present"""
        try:
            result = extract_message(img_path, method)
            return result is not None and len(result) > 0
        except:
            return False
    
    def balance_format_distribution(self, clean_dir, stego_dir):
        """Ensure clean images match stego format distribution"""
        stego_formats = count_formats(stego_dir)
        clean_formats = count_formats(clean_dir)
        
        # Generate additional clean images to match
        for format, count in stego_formats.items():
            shortage = count - clean_formats.get(format, 0)
            if shortage > 0:
                generate_clean_images(format, shortage)
```

**Wednesday: Add Negative Examples**

Create `scripts/generate_negatives.py`:

```python
class NegativeExampleGenerator:
    """Generate examples teaching what steganography is NOT"""
    
    def rgb_with_alpha_check(self):
        """RGB images that should NOT be detected as alpha"""
        # Generate diverse RGB images with various characteristics
        pass
    
    def uniform_alpha_images(self):
        """RGBA images with uniform alpha (no hidden data)"""
        # All pixels alpha=255
        pass
    
    def natural_lsb_noise(self):
        """Clean images with natural LSB variation"""
        # GIF dithering, JPEG compression artifacts
        pass
    
    def repetitive_patterns(self):
        """Images with repetitive hex patterns (not stego)"""
        # Solid colors, gradients, patterns
        pass
```

**Thursday: Comprehensive Validation**

Create `scripts/validate_repaired_dataset.py`:

```python
def validate_dataset(dataset_path):
    """Verify dataset meets quality requirements"""
    checks = {
        'no_invalid_labels': check_label_validity(),
        'extraction_verified': verify_all_extractions(),
        'format_balanced': check_format_balance(),
        'negatives_present': count_negative_examples(),
        'no_corrupted_signals': verify_signal_integrity()
    }
    
    return all(checks.values()), checks
```

**Friday: Documentation & Handoff**

Create `docs/claude/DATASET_V3_SPEC.md`:

```markdown
# Dataset V3 Specification

## Quality Requirements
- ✅ No impossible labels (alpha on RGB)
- ✅ All stego images extraction-verified
- ✅ Format-matched clean images
- ✅ Negative examples for each special case
- ✅ No signal-corrupting preprocessing

## Structure
data/training/v3_repaired/
├── clean/
│   ├── rgb/           # Matches LSB format distribution
│   ├── rgba/          # Matches alpha format distribution
│   ├── palette/       # Matches palette format distribution
│   └── jpeg/          # Matches EXIF/EOI format distribution
├── stego/
│   ├── alpha/         # All verified extraction
│   ├── lsb/           # All verified extraction
│   ├── palette/       # All verified extraction
│   ├── exif/          # All verified extraction
│   └── eoi/           # All verified extraction
├── negatives/
│   ├── rgb_no_alpha/  # Teaching: RGB cannot have alpha
│   ├── uniform_alpha/ # Teaching: Uniform alpha = no data
│   ├── natural_noise/ # Teaching: Dithering ≠ stego
│   └── patterns/      # Teaching: Repetitive ≠ stego
└── manifest_v3.jsonl  # Signed manifest with all metadata
```

---

## 🏗️ Week 2: V3/V4 Architecture Development

### Goal
Merge dual-input (Claude V3) + multi-stream (Gemini V4) architectures.

**Monday: Architecture Design**

Create `docs/claude/V3_V4_UNIFIED_SPEC.md`:

```markdown
# Unified Multi-Stream Architecture

## Input Streams (5 total)
1. **Pixel Tensor**: (B, 3, H, W) - RGB content
2. **Alpha Tensor**: (B, 1, H, W) - Alpha channel (zero-padded if absent)
3. **LSB Tensor**: (B, 3, H, W) - Extracted LSB planes
4. **Palette Tensor**: (B, 256, 3) - Color palette (zero-padded if absent)
5. **Metadata Tensor**: (B, 2048) - EXIF + EOI bytes

## Processing Pipeline
Each stream → Specialized backbone → Feature fusion → Classification

## Key Innovations
- LSB extraction BEFORE any augmentation
- Format-aware feature extraction
- Separate confidence heads per method
- Constraint validation layer (learns special cases)
```

**Tuesday-Thursday: Implementation**

Build `models/unified_detector.py`:

```python
class UnifiedStarlightDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Specialized backbones
        self.pixel_backbone = ResNet18(in_channels=3)
        self.alpha_backbone = SmallCNN(in_channels=1)
        self.lsb_backbone = SmallCNN(in_channels=3)
        self.palette_backbone = MLP(input_dim=768)
        self.metadata_backbone = MLP(input_dim=2048)
        
        # Feature fusion
        self.fusion = AttentionFusion(num_streams=5)
        
        # Classification heads
        self.stego_classifier = nn.Linear(fusion_dim, 2)
        self.method_classifier = nn.Linear(fusion_dim, 5)
        
        # Constraint validation (learns special cases)
        self.constraint_validator = ConstraintNet(fusion_dim)
    
    def forward(self, pixel, alpha, lsb, palette, metadata):
        # Extract features from each stream
        f_pixel = self.pixel_backbone(pixel)
        f_alpha = self.alpha_backbone(alpha)
        f_lsb = self.lsb_backbone(lsb)
        f_palette = self.palette_backbone(palette)
        f_metadata = self.metadata_backbone(metadata)
        
        # Fuse features
        fused = self.fusion([f_pixel, f_alpha, f_lsb, f_palette, f_metadata])
        
        # Classify
        stego_logits = self.stego_classifier(fused)
        method_logits = self.method_classifier(fused)
        
        # Validate constraints
        constraint_scores = self.constraint_validator(fused)
        
        return stego_logits, method_logits, constraint_scores
```

**Friday: Integration Testing**

```bash
# Test unified model with repaired dataset
python scripts/test_unified_model.py \
  --dataset data/training/v3_repaired \
  --model models/unified_detector.py \
  --report docs/claude/unified_model_test.md
```

---

## 🎯 Week 3: Training Strategy

### Advanced Training Techniques

**Create `scripts/train_unified.py`**:

```python
class UnifiedTrainer:
    def __init__(self, model, config):
        self.model = model
        
        # Multi-objective loss
        self.loss_weights = {
            'stego_detection': 1.0,
            'method_classification': 0.5,
            'constraint_validation': 0.3
        }
        
        # Balanced sampling per method
        self.sampler = BalancedMethodSampler(dataset)
        
    def training_step(self, batch):
        # Standard forward pass
        stego_logits, method_logits, constraints = self.model(
            batch['pixel'], batch['alpha'], batch['lsb'],
            batch['palette'], batch['metadata']
        )
        
        # Multi-objective loss
        loss_stego = F.cross_entropy(stego_logits, batch['is_stego'])
        loss_method = F.cross_entropy(method_logits, batch['method'])
        loss_constraint = constraint_loss(constraints, batch)
        
        total_loss = (
            self.loss_weights['stego_detection'] * loss_stego +
            self.loss_weights['method_classification'] * loss_method +
            self.loss_weights['constraint_validation'] * loss_constraint
        )
        
        return total_loss
```

---

## 🔍 Week 4: Validation & Benchmarking

### Cross-Dataset Testing

**Create comprehensive validation**:

```bash
# Test on all datasets
python scripts/validate_unified.py \
  --model models/unified_detector.onnx \
  --datasets "datasets/*_submission_*/clean" \
  --datasets "datasets/*_submission_*/stego" \
  --output docs/claude/validation_results.json

# Compare against production baseline
python scripts/compare_models.py \
  --baseline models/detector_balanced.onnx \
  --experimental models/unified_detector.onnx \
  --report docs/claude/model_comparison.md
```

**Target Metrics**:
- FP rate < 5% (eventual goal: < 1%)
- Detection rate > 95%
- Method classification > 90%
- Consistent across all datasets

---

## 🚨 Critical Checkpoints

### Before Each Training Run
- [ ] Verify no signal-corrupting augmentations
- [ ] Confirm LSB extraction happens BEFORE augmentation
- [ ] Check format distribution balance
- [ ] Validate no impossible labels
- [ ] Ensure negative examples included

### Daily Progress Logging
```bash
# Update daily
git add docs/claude/
git commit -m "Claude Research: Day N - [accomplishment]"
git push origin claude-research-track
```

---

## 📚 Essential Reading

Before starting, review these files:
1. `docs/phase1.md` - Two-track strategy
2. `docs/status.md` - Why special cases are necessary
3. `docs/chatgpt_proposal.md` - V3 architecture
4. `docs/survey-consolidated.md` - Lessons learned

---

## 🎯 Success Criteria

**By End of Month 1 (Your Focus)**:
- ✅ Repaired dataset with 0 invalid labels
- ✅ Negative examples for all special cases
- ✅ Unified V3/V4 architecture implemented
- ✅ Initial training runs showing improvement

**By Month 6 (Team Effort)**:
- FP rate < 5% without special cases
- Method classification > 90%
- Consistent cross-dataset performance

**By Month 18-24 (Long-term Goal)**:
- FP rate < 1% without special cases
- Models that learn constraint validation
- True generalization achieved

---

## 💡 Key Principles

1. **Dataset Quality First**: No amount of architecture sophistication fixes bad training data
2. **Validate Everything**: Every stego image must have verified extraction
3. **Format Awareness**: Clean images must match stego format distribution
4. **Teach Negatives**: Models need examples of what steganography is NOT
5. **Document Thoroughly**: Next Claude needs full context

---

## 🔗 Coordination

### Daily Sync
- Update `docs/claude/PROGRESS.md`
- Note blockers in `docs/claude/BLOCKERS.md`
- Commit changes daily

### Weekly Review
- Friday: Submit progress to `ai_consensus.md`
- Compare notes with Gemini (implementation) and ChatGPT (training)
- Align with Grok on format expansions

---

## 📅 Timeline Summary

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Dataset Repair | Validated, repaired dataset with negatives |
| 2 | Architecture | Unified V3/V4 multi-stream model |
| 3 | Training | Initial training runs with multi-objective loss |
| 4 | Validation | Cross-dataset benchmarking vs production |

---

**Remember**: You're not trying to replace the production system immediately. You're building the foundation for true generalization over the next 18-24 months. Focus on dataset quality and architectural innovations that will enable models to learn what special cases currently encode.

**Production remains stable. Research moves forward. Both tracks succeed.**

---

*Last Updated: 2025-11-15*  
*Next Review: 2025-11-22*  
*Track: Research (Track B)*
