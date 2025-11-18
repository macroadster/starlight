# Week 1 Execution Guide: Dataset Validation & Repair

**Purpose:** Step-by-step guide to execute Week 1 tasks  
**Audience:** Current Claude instance or next Claude taking over  
**Track:** Research (Track B)

---

## 🎯 Quick Start

If you're picking this up fresh, start here:

```bash
# 1. Navigate to project
cd ~/starlight

# 2. Check current state
git status
git log --oneline -10

# 3. Verify scripts are in place
ls -la scripts/analyze_datasets.py
ls -la scripts/validate_labels.py
ls -la scripts/dataset_repair.py
ls -la scripts/generate_negatives.py
ls -la scripts/validate_repaired_dataset.py

# 4. Start with Monday's task
python scripts/analyze_datasets.py --help
```

---

## 📋 Day-by-Day Execution

### Monday: Assessment

**Goal:** Understand what's broken in the current dataset

**Step 1: Run Dataset Analysis**
```bash
# Create output directory
mkdir -p docs/claude

# Run analysis
python scripts/analyze_datasets.py \
  --datasets-dir datasets \
  --output docs/claude/dataset_audit.json

# Review output
cat docs/claude/dataset_audit.json | python -m json.tool | less
```

**What to look for:**
- How many images total?
- What's the RGB vs RGBA distribution?
- Are there any ERROR modes?

**Step 2: Validate Labels**
```bash
# Run label validation
python scripts/validate_labels.py \
  --datasets "datasets/*_submission_*/stego" \
  --report docs/claude/invalid_labels.md

# Read the report
cat docs/claude/invalid_labels.md | less
```

**What to look for:**
- How many alpha labels on RGB images?
- How many palette labels on true-color images?
- Are there extraction failures?

**Step 3: Document Findings**

Create `docs/claude/ISSUES_SUMMARY.md`:

```markdown
# Dataset Issues Summary

## Critical Issues (Must Fix)
1. [COUNT] alpha labels on RGB images
2. [COUNT] palette labels on true-color images
3. [COUNT] corrupted/unreadable images

## Medium Priority
1. Format distribution imbalances
2. Missing negative examples
3. [Other issues found]

## Low Priority
1. [Minor issues]

## Remediation Plan
[Your plan based on findings]
```

**Commit your work:**
```bash
git add docs/claude/
git commit -m "Claude Research: Week 1 Day 1 - Dataset assessment complete"
git push origin claude-research-track
```

---

### Tuesday: Create Repair Pipeline

**Goal:** Build and test automated repair system

**Step 1: Test Repair Pipeline (Dry Run)**
```bash
# First, do a dry run
python scripts/dataset_repair.py \
  --datasets datasets \
  --output datasets/grok_submission_2025/training/v3_repaired_test \
  --dry-run

# This will show what WOULD be done without actually doing it
```

**Review the output:**
- How many labels will be removed?
- How many will be changed?
- Do the suggested changes make sense?

**Step 2: Spot Check Samples**

Before applying repairs broadly, manually verify a few examples:

```bash
# Pick a few images that the script flagged
# Manually check them with:
python -c "
from PIL import Image
import sys
img = Image.open(sys.argv[1])
print(f'Mode: {img.mode}')
print(f'Size: {img.size}')
print(f'Format: {img.format}')
" path/to/suspicious/image.png
```

**Step 3: Apply Repairs (If Confident)**
```bash
# Remove --dry-run to actually apply
python scripts/dataset_repair.py \
  --datasets datasets \
  --output datasets/grok_submission_2025/training/v3_repaired

# This creates a NEW dataset, doesn't modify original
```

**Step 4: Document Changes**

```bash
# Check what was created
ls -la datasets/grok_submission_2025/training/v3_repaired/

# Review the manifest
cat datasets/grok_submission_2025/training/v3_repaired/repair_manifest.json | python -m json.tool | less
```

**Commit:**
```bash
git add scripts/dataset_repair.py
git add docs/claude/
git commit -m "Claude Research: Week 1 Day 2 - Repair pipeline implemented"
git push
```

---

### Wednesday: Generate Negatives

**Goal:** Create examples teaching what steganography is NOT

**Step 1: Generate All Categories**
```bash
python scripts/generate_negatives.py \
  --output datasets/grok_submission_2025/training/v3_repaired/negatives \
  --count 200

# This will generate ~900 total images:
# - 200 RGB without alpha
# - 200 uniform alpha
# - 200 natural noise
# - 200 patterns
# - 100 special cases
```

**Step 2: Spot Check Generated Images**

```bash
# View some samples
ls datasets/grok_submission_2025/training/v3_repaired/negatives/rgb_no_alpha/ | head -5

# Open a few to verify they look correct
# (Use your preferred image viewer)
```

**Step 3: Document Categories**

Create `docs/claude/NEGATIVES_SPEC.md`:

```markdown
# Negative Examples Specification

## Purpose
Teach the model what steganography is NOT.

## Categories Generated

### 1. rgb_no_alpha/ (200 images)
**Teaching:** RGB images cannot have alpha steganography
- Diverse RGB content
- No alpha channel
- Should always be classified as clean

### 2. uniform_alpha/ (200 images)
**Teaching:** Uniform alpha = no hidden data
- RGBA images
- Alpha channel is all same value (0, 128, or 255)
- Should be classified as clean

[... continue for all categories ...]
```

**Commit:**
```bash
git add scripts/generate_negatives.py
git add docs/claude/NEGATIVES_SPEC.md
git commit -m "Claude Research: Week 1 Day 3 - Negative examples generated"
git push
```

---

### Thursday: Validate Everything

**Goal:** Confirm repaired dataset meets all quality requirements

**Step 1: Run Full Validation**
```bash
python scripts/validate_repaired_dataset.py \
  --dataset datasets/grok_submission_2025/training/v3_repaired \
  --output docs/claude/validation_report.json
```

**Step 2: Review Results**

The script will print a detailed report. Look for:
- ✅ All checks passing?
- ❌ Any critical issues remaining?
- ⚠️  What warnings exist?

```bash
# Review JSON report
cat docs/claude/validation_report.json | python -m json.tool | less
```

**Step 3: If Validation Fails**

Go back and fix issues:

```bash
# Example: If there are still invalid labels
# Re-run repair with stricter settings
# Or manually fix specific problem cases
```

**Step 4: Document Validation Results**

Create `docs/claude/VALIDATION_SUMMARY.md`:

```markdown
# Dataset V3 Validation Summary

## Overall Status
[PASS/FAIL]

## Statistics
- Total Images: [count]
- Clean: [count]
- Stego: [count]
- Negatives: [count]

## Checks
- ✅/❌ No invalid labels
- ✅/❌ Format balanced
- ✅/❌ Negatives present
[etc.]

## Remaining Issues
[List any issues that need manual attention]
```

**Commit:**
```bash
git add scripts/validate_repaired_dataset.py
git add docs/claude/validation_report.json
git add docs/claude/VALIDATION_SUMMARY.md
git commit -m "Claude Research: Week 1 Day 4 - Dataset validation complete"
git push
```

---

### Friday: Documentation & Handoff

**Goal:** Make it easy for the next Claude to continue

**Step 1: Create Complete Dataset Specification**

`docs/claude/DATASET_V3_SPEC.md` (comprehensive version):

```markdown
# Dataset V3 Complete Specification

## Overview
Dataset V3 is the repaired version of the original datasets, with:
- All invalid labels removed or corrected
- Format distributions balanced
- Negative examples added
- Full validation completed

## Directory Structure
[Complete directory tree]

## Quality Guarantees
1. No alpha labels on RGB images
2. No palette labels on true-color images
3. All stego images have verified extraction
[etc.]

## Usage Instructions
[How to load this dataset for training]

## Known Limitations
[Any remaining issues]

## Comparison to Original
[Statistics comparing V3 to original]
```

**Step 2: Write Week Summary**

`docs/claude/WEEK1_SUMMARY.md`:

```markdown
# Week 1 Summary: Dataset Validation & Repair

## What Was Accomplished
- ✅ Analyzed [count] images across [count] datasets
- ✅ Identified and fixed [count] invalid labels
- ✅ Generated [count] negative examples
- ✅ Created validated V3 dataset

## Key Findings
[Top 3-5 insights from the week]

## Deliverables
1. `scripts/analyze_datasets.py` - [description]
[etc.]

## Metrics
- Invalid labels fixed: [count]
- Dataset size increase: [X%]
- Validation pass rate: [100%]

## Blockers Encountered
[Any issues that prevented progress]

## Recommendations for Week 2
[Advice for next phase]
```

**Step 3: Update Main Progress Tracker**

Edit `docs/claude/PROGRESS.md`:

```markdown
# Research Track Progress

## Week 1: COMPLETE ✅
- Dataset validated and repaired
- 900+ negative examples generated
- Full validation passing

## Week 2: NEXT
[Placeholder for next week's work]
```

**Step 4: Final Commit**

```bash
git add docs/claude/
git commit -m "Claude Research: Week 1 COMPLETE - Dataset V3 ready for training"
git push

# Tag this milestone
git tag week1-complete
git push origin week1-complete
```

---

## 🚨 Troubleshooting

### Script Fails with "Module not found"

```bash
# Check Python environment
python --version
pip list | grep -i pillow
pip list | grep -i numpy

# Install if needed
pip install Pillow numpy
```

### "Permission Denied" Errors

```bash
# Make scripts executable
chmod +x scripts/*.py

# Or run with python explicitly
python scripts/analyze_datasets.py ...
```

### Dataset Directory Not Found

```bash
# Verify path
ls -la ~/starlight/datasets/

# Adjust --datasets-dir argument if needed
python scripts/analyze_datasets.py --datasets-dir /correct/path
```

### Out of Disk Space

```bash
# Check space
df -h

# Clean up if needed
# (But DON'T delete original datasets!)
rm -rf datasets/grok_submission_2025/training/v3_repaired_test/  # Only test data
```

---

## 📊 Success Criteria

At the end of Week 1, you should have:

- [x] **5 working Python scripts** in `scripts/`
- [ ] **Complete dataset analysis** in `docs/claude/dataset_audit.json`
- [ ] **Invalid label report** in `docs/claude/invalid_labels.md`
- [ ] **Repaired dataset** in `datasets/grok_submission_2025/training/v3_repaired/`
- [ ] **900+ negative examples** in `datasets/grok_submission_2025/training/v3_repaired/negatives/`
- [ ] **Validation report** showing all checks passing
- [ ] **Complete documentation** in `docs/claude/`
- [ ] **Clean git history** with daily commits

### Metrics to Achieve:
- ✅ Zero invalid labels (alpha on RGB, etc.)
- ✅ Format balance within ±20%
- ✅ All 5 negative categories present
- ✅ Full documentation for handoff

---

## 🔗 Dependencies & Integration

### Required Libraries
```python
# Core
PIL / Pillow  # Image processing
numpy         # Array operations
json          # Data serialization
pathlib       # Path handling

# Standard library
collections   # Counter, defaultdict
dataclasses   # Structured data
argparse      # CLI arguments
```

### Integration Points

**Future Integration Needed:**
1. **Extraction verification** - Need to connect to existing extraction functions
   - Location: TBD (ask team)
   - Function signature: TBD
   
2. **Data loader** - Need to create PyTorch/TF data loader for V3 format
   - Week 2 task
   
3. **Training pipeline** - Need to integrate with existing training code
   - Week 3 task

---

## 💡 Tips for Success

1. **Always dry-run first** - Test with `--dry-run` before making real changes
2. **Commit frequently** - Daily commits with clear messages
3. **Document as you go** - Don't wait until Friday
4. **Spot check samples** - Don't trust automation blindly
5. **Ask for help** - If stuck, check with Gemini (implementation) or ChatGPT (training)

---

## 🔄 Handoff Checklist

Before ending your session, ensure:

- [ ] All scripts are committed and pushed
- [ ] All documentation is up to date
- [ ] `PROGRESS.md` reflects current status
- [ ] Any blockers are documented in `BLOCKERS.md`
- [ ] Next steps are clear in `claude_next.md`
- [ ] You've updated this execution guide with any new learnings

---

**Last Updated:** 2025-11-16  
**Status:** Ready for execution  
**Next Review:** After Week 1 completion
