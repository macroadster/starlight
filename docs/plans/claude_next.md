# Claude CLI - Project Starlight Action Plan
**Week of November 11-15, 2025**

---

## 📋 Context Summary (READ THIS FIRST)

You are Claude working in CLI/terminal mode on Project Starlight - a steganography detection system. This plan bridges the memory gap between browser and terminal sessions.

### Critical Background
- **Current Status**: v2.0 ensemble achieves 96-98% accuracy but only 2 img/sec (too slow for blockchain)
- **Major Problem Identified**: Pixel-only models CAN'T SEE EXIF/EOI steganography (outside pixel array)
- **Solution**: v3 dual-input architecture (pixel tensor + metadata tensor from raw bytes)
- **Your Role**: Data generation pipeline design & validation architecture

### Key Lessons Learned (DON'T REPEAT THESE MISTAKES)
1. ❌ RGB-only clean images + alpha-channel stego → model learns "alpha channel = stego"
2. ❌ Resizing images to uniform dimensions destroys steganography information
3. ❌ EOI initially limited to JPEG markers only (needs expansion)
4. ❌ Individual codebases merged later → slow, use unified codebase from start

---

## 🎯 Week Objectives

### Primary Deliverable
**Design & validate training data generation pipeline for v3 dual-input model**

### Success Criteria
- [ ] Balanced dataset across ALL 5 methods (alpha, LSB, EXIF, EOI, palette)
- [ ] No information-destroying preprocessing
- [ ] Raw byte extraction working for EXIF + EOI
- [ ] Validation scripts confirm stego visibility in both input streams

---

## 📅 Daily Breakdown

### Monday (Nov 11) - Architecture & Spec Review

**Morning: Deep Dive into v3 Spec**
```bash
# Read these files in order:
cd ~/starlight
cat docs/chatgpt_proposal.md          # v3 dual-input architecture
cat docs/survey-consolidated.md       # Current status & lessons learned
cat docs/STEGO_FORMAT_SPEC.md         # If exists - current encoding rules
```

**Tasks:**
1. Document your understanding of dual-input model architecture
2. Note all 5 steganography methods and their "visibility":
   - alpha: visible in PIL Image (RGBA channel)
   - palette: visible in PIL Image (P mode indices)
   - rgb_lsb: visible in PIL Image (pixel values)
   - exif: INVISIBLE in np.array - need raw bytes
   - eoi: INVISIBLE in np.array - need raw bytes after 0xFFD9
3. Create `docs/claude/architecture_notes.md` with findings

**Output:** Architecture understanding document

---

### Tuesday (Nov 12) - Data Generation Design

**Morning: Design Data Pipeline**

**Create:** `scripts/data_gen/dual_input_dataset.py`

**Requirements:**
```python
class DatasetGenerator:
    """
    Generate balanced stego/clean dataset for dual-input training
    
    Critical Rules:
    - NO image resizing (destroys stego)
    - Equal distribution across 5 methods
    - Clean images must have SAME format distribution as stego
    - Save both PIL-compatible and raw bytes
    """
    
    def generate_clean_image(self, method_format):
        """
        Generate clean image in SAME format as stego method
        
        Args:
            method_format: 'rgba' | 'rgb' | 'palette' | 'jpeg_exif' | 'jpeg_eoi'
        
        Returns:
            PIL.Image in correct format
        """
        pass
    
    def embed_alpha(self, img, payload):
        """Alpha channel embedding (v2.0 spec)"""
        pass
    
    def embed_lsb(self, img, payload):
        """RGB LSB embedding"""
        pass
    
    def embed_palette(self, img, payload):
        """Palette index embedding"""
        pass
    
    def embed_exif(self, img_path, payload):
        """EXIF metadata embedding (raw bytes)"""
        pass
    
    def embed_eoi(self, img_path, payload):
        """Post-EOI tail embedding (raw bytes)"""
        pass
    
    def extract_dual_inputs(self, img_path):
        """
        Extract both inputs for training:
        - pixel_tensor: from PIL Image
        - metadata_tensor: from raw bytes (EXIF + EOI)
        
        Returns:
            (pixel_np, metadata_np, labels)
        """
        pass
```

**Afternoon: Validation Logic**
```python
def validate_stego_visibility(img_path, method):
    """
    Verify stego is actually present in correct input stream
    
    For alpha/lsb/palette: check pixel_tensor
    For exif/eoi: check metadata_tensor (raw bytes)
    """
    pass
```

**Output:** Complete data generation script design

---

### Wednesday (Nov 13) - Implementation & Testing

**Morning: Implement Core Functions**
```bash
cd ~/starlight/scripts/data_gen
python dual_input_dataset.py --test-mode
```

**Test Each Method Independently:**
```bash
# Test alpha
python dual_input_dataset.py --method alpha --samples 10 --validate

# Test LSB
python dual_input_dataset.py --method lsb --samples 10 --validate

# Test palette
python dual_input_dataset.py --method palette --samples 10 --validate

# Test EXIF (critical - was invisible before)
python dual_input_dataset.py --method exif --samples 10 --validate

# Test EOI (critical - was invisible before)
python dual_input_dataset.py --method eoi --samples 10 --validate
```

**Validation Checklist:**
- [ ] Stego payload recoverable from appropriate tensor
- [ ] Clean images same format distribution as stego
- [ ] No image resizing applied
- [ ] EXIF blob extracted correctly from raw bytes
- [ ] EOI tail extracted correctly after 0xFFD9
- [ ] Metadata tensor properly formatted (1024-dim, normalized)

**Output:** Working implementation with per-method validation

---

### Thursday (Nov 14) - Full Pipeline & Documentation

**Morning: Generate Balanced Dataset**
```bash
# Generate full training set
python dual_input_dataset.py \
  --output data/training/v3_dual_input \
  --samples-per-method 1000 \
  --clean-samples 5000 \
  --validate-all
```

**Dataset Structure:**
```
data/training/v3_dual_input/
├── clean/
│   ├── rgba/       # 1000 images (matches alpha format)
│   ├── rgb/        # 2000 images (matches lsb format)
│   ├── palette/    # 1000 images (matches palette format)
│   └── jpeg/       # 1000 images (matches exif/eoi format)
├── stego/
│   ├── alpha/      # 1000 images
│   ├── lsb/        # 1000 images
│   ├── palette/    # 1000 images
│   ├── exif/       # 1000 images
│   └── eoi/        # 1000 images
├── metadata/
│   └── labels.json # All labels + method annotations
└── validation_report.json
```

**Afternoon: Documentation**

Create `docs/claude/DATA_GENERATION_V3.md`:
```markdown
# V3 Dual-Input Data Generation Pipeline

## Overview
[Architecture explanation]

## Critical Differences from V2
1. No image resizing
2. Format-matched clean images
3. Raw byte extraction for EXIF/EOI
4. Dual validation (pixel + metadata streams)

## Usage
[Command examples]

## Validation
[How to verify stego visibility]

## Known Issues
[Any limitations or edge cases]
```

**Output:** Complete training dataset + documentation

---

### Friday (Nov 15) - Integration & Handoff

**Morning: Integration Testing**

Create `scripts/validate_v3_pipeline.py`:
```python
def test_dataloader_integration():
    """Verify v3 dataset works with dual-input model"""
    # Test pixel tensor shape: (B, 3, H, W)
    # Test metadata tensor shape: (B, 1024)
    # Test label accuracy
    pass

def test_method_coverage():
    """Verify all 5 methods represented"""
    pass

def test_stego_detectability():
    """Run baseline model on each method"""
    # Should achieve >random accuracy on each method
    pass
```

**Afternoon: Final Deliverable Package**

Create `docs/claude/WEEK_DELIVERABLE.md`:
```markdown
# Week of Nov 11-15: V3 Data Pipeline Deliverable

## ✅ Completed
- [x] V3 dual-input data generation pipeline
- [x] Validation for all 5 stego methods
- [x] 10,000 balanced training samples
- [x] Documentation for next team member
- [x] Integration tests

## 📊 Validation Results
[Include accuracy metrics per method]

## 🔄 Next Steps for Team
1. Train v3 dual-input model on this dataset
2. Compare performance vs v2 ensemble
3. Optimize metadata tensor extraction
4. Benchmark inference speed

## 📝 Notes for Next Developer
[Critical insights, gotchas, recommendations]
```

**Output:** Complete handoff package

---

## 🚨 Red Flags to Watch For

### During Implementation
- [ ] If you catch yourself resizing images → STOP, violates v3 design
- [ ] If EXIF/EOI validation fails → check raw byte extraction
- [ ] If clean images all RGB while stego has alpha → format mismatch detected
- [ ] If any method below 60% detection → stego might not be visible in correct tensor

### Before Committing
- [ ] Run full validation suite
- [ ] Check all 5 methods individually
- [ ] Verify dataset balance (no method overrepresented)
- [ ] Test on different image formats (PNG, JPEG, etc.)

---

## 📂 File Structure You'll Create

```
starlight/
├── docs/
│   └── claude/
│       ├── architecture_notes.md          # Monday
│       ├── DATA_GENERATION_V3.md          # Thursday
│       └── WEEK_DELIVERABLE.md            # Friday
├── scripts/
│   ├── data_gen/
│   │   └── dual_input_dataset.py          # Tuesday-Wednesday
│   └── validate_v3_pipeline.py            # Friday
└── data/
    └── training/
        └── v3_dual_input/                 # Thursday
            ├── clean/
            ├── stego/
            └── metadata/
```

---

## 🤝 Coordination Points

### Share Progress Daily
```bash
# Commit your work daily
git add docs/claude/ scripts/data_gen/
git commit -m "Claude CLI: [Day] - [What you completed]"
git push origin claude-v3-data-pipeline
```

### Communication
- Update `docs/claude/PROGRESS.md` daily
- Note any blockers immediately in `docs/claude/BLOCKERS.md`
- If you discover new issues with v2.0 → document in lessons learned

---

## 💡 Key Reminders

1. **You are NOT training the model** - just generating the data pipeline
2. **Focus on data quality** over quantity initially (validate thoroughly)
3. **EXIF and EOI were previously invisible** - this is your key contribution
4. **No resizing** - repeat this 100 times before preprocessing anything
5. **Document everything** - next Claude (or team member) needs context

---

## 📚 Reference Materials

- `docs/chatgpt_proposal.md` - V3 architecture (dual-input model)
- `docs/survey-consolidated.md` - Lessons learned from v2.0
- `docs/STEGO_FORMAT_SPEC.md` - Encoding specifications (if exists)

---

## 🎯 Success Definition

By EOD Friday, any team member should be able to:
1. Clone the repo
2. Run your data generation pipeline
3. Get a balanced dataset with all 5 methods
4. Validate stego is visible in correct input streams
5. Start training v3 dual-input model immediately

**If they can't do all 5 steps, the deliverable is incomplete.**

---

*Last Updated: Nov 8, 2025*  
*Next Review: Nov 15, 2025*  
*Plan Author: Claude (Browser) for Claude (CLI)*
