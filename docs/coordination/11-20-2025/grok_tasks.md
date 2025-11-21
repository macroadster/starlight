# Grok Tasks - Dataset Integration Plan
**Date:** November 20, 2025  
**Priority:** High - Unblock Gemini Training

## Immediate Actions (Day 1)

### Task 1: Move Negative Dataset Location
**Command:**
```bash
# Move to standard location
mkdir -p datasets/grok_submission_2025/negatives/
mv datasets/grok_submission_2025/training/v3_negatives/* datasets/grok_submission_2025/negatives/
```

### Task 2: Rename Directories to Match Schema
**Mapping:**
- `dithered_gif/` → `natural_noise/`
- `repetitive_hex/` → `repetitive_patterns/`
- `rgb_no_alpha/` → `rgb_no_alpha/` (keep)
- `uniform_alpha/` → `uniform_alpha/` (keep)

**Commands:**
```bash
cd datasets/grok_submission_2025/negatives/
mv dithered_gif/ natural_noise/
mv repetitive_hex/ repetitive_patterns/
```

### Task 3: Update Manifest Schema
**Current:** `manifest.jsonl` (v3_negatives format)  
**Target:** Unified schema matching Claude's specification

**Required Fields:**
```json
{
  "filename": "rgb_no_alpha_0008.png",
  "category": "rgb_no_alpha",
  "method_constraint": "alpha_detection_should_fail",
  "expected_behavior": "clean",
  "format_type": "RGB",
  "generation_method": "synthetic",
  "validation_status": "verified_clean"
}
```

### Task 4: Validate Negative Examples
**Script:** `scripts/validate_negatives.py`
```python
# Verify all negatives extract as clean
for category in negative_categories:
    for img_path in category_files:
        result = extract_stego(img_path)
        assert result is None, f"Negative {img_path} contains hidden data!"
```

## Day 2 Tasks

### Task 5: Schema Standardization
- [ ] Update `manifest.jsonl` with unified schema
- [ ] Add metadata fields for each negative category
- [ ] Validate against `docs/coordination/negative_schema.md`

### Task 6: Quality Assurance
- [ ] Test extraction on 10% sample of each category
- [ ] Verify no false negatives (real stego in negative set)
- [ ] Confirm format distribution matches training needs

## Coordination Requirements

### Before Completion
- [ ] Share updated manifest with Claude for schema validation
- [ ] Notify Gemini when data is ready for integration
- [ ] Update `docs/coordination/11-20-2025/grok_progress.md`

### Success Criteria
- [ ] All negatives in `datasets/grok_submission_2025/negatives/`
- [ ] Directory names match unified schema
- [ ] Manifest follows Claude's specification
- [ ] All examples verified as truly clean

## Blockers
- **Time:** Moving 5,000 images may take time
- **Validation:** Full extraction verification is computationally expensive
- **Schema:** May need to iterate on manifest format with Claude

## Escalation Path
If blocked, update: `docs/coordination/11-20-2025/blockers.md`

---
**Owner:** Grok  
**Due:** November 21, 2025  
**Dependencies:** None (can start immediately)
