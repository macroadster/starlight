# Dataset Integration Coordination Plan
**Date:** November 20, 2025  
**Issue:** Grok's negative datasets not integrated with Gemini's training pipeline  
**Status:** Coordination Required

## Problem Analysis

### Current State
- **Grok**: Generated 5,000 negative examples in `datasets/grok_submission_2025/training/v3_negatives/`
- **Gemini**: Training pipeline expects data in different structure, not using Grok's negatives
- **Claude**: Proposed `datasets/grok_submission_2025/negatives/` structure in plan

### Root Cause
1. **Location Mismatch**: Grok's data in `training/v3_negatives/` vs expected `negatives/`
2. **Structure Incompatibility**: Gemini's training doesn't scan for negatives across all submission directories
3. **Schema Misalignment**: Different manifest formats and data organization

## Proposed Solution

### 1. Standardize Negative Dataset Location
**Target Structure:**
```
datasets/*_submission_*/negatives/
├── rgb_no_alpha/          # RGB images that should NOT trigger alpha detection
├── uniform_alpha/         # RGBA images with uniform alpha (no hidden data)
├── natural_noise/         # Clean images with natural LSB variation
├── repetitive_patterns/    # Images with repetitive patterns (not stego)
└── manifest.jsonl         # Unified schema
```

### 2. Task Assignments

#### Grok (Data Provider)
**Priority 1: Move and Reorganize**
- [ ] Move `datasets/grok_submission_2025/training/v3_negatives/*` to `datasets/grok_submission_2025/negatives/`
- [ ] Rename directories to match standard schema:
  - `rgb_no_alpha/` → `rgb_no_alpha/` (keep)
  - `uniform_alpha/` → `uniform_alpha/` (keep)
  - `dithered_gif/` → `natural_noise/` (rename)
  - `repetitive_hex/` → `repetitive_patterns/` (rename)
- [ ] Update `manifest.jsonl` to unified schema
- [ ] Verify all negative examples extract as clean (no hidden data)

**Priority 2: Schema Standardization**
- [ ] Ensure manifest follows Claude's proposed schema
- [ ] Add metadata fields: `method_constraint`, `expected_behavior`, `format_type`
- [ ] Validate against `docs/coordination/negative_schema.md`

#### Gemini (Training Pipeline)
**Priority 1: Update Training Data Scanner**
- [ ] Modify training script to scan `datasets/*_submission_*/negatives/` recursively
- [ ] Add negative example loading to `starlight_utils.py`
- [ ] Implement negative sampling strategy (e.g., 1 negative per 3 stego examples)

**Priority 2: Integration with Unified Pipeline**
- [ ] Ensure 6-stream extraction works with negative examples
- [ ] Add negative-specific loss weighting (teach what NOT to detect)
- [ ] Validate that negatives reduce false positives on special cases

#### Claude (Architecture & Validation)
**Priority 1: Schema Finalization**
- [ ] Finalize unified negative schema in `docs/coordination/negative_schema.md`
- [ ] Create validation script for negative dataset integrity
- [ ] Document negative example generation best practices

**Priority 2: Cross-Agent Coordination**
- [ ] Review Grok's reorganized data structure
- [ ] Validate Gemini's training integration
- [ ] Create handoff documentation for future agents

### 3. Implementation Timeline

**Day 1 (Nov 20)**
- Grok: Move and reorganize negative datasets
- Claude: Finalize schema documentation

**Day 2 (Nov 21)**  
- Gemini: Update training pipeline to scan for negatives
- Grok: Complete schema standardization

**Day 3 (Nov 22)**
- Claude: Validate integration and create tests
- All: Test end-to-end training with negatives

### 4. Success Criteria

#### Technical
- [ ] All negative datasets accessible via `datasets/*_submission_*/negatives/`
- [ ] Gemini's training script loads and uses negative examples
- [ ] Negative examples reduce false positive rate on special cases
- [ ] Unified schema works across all agents

#### Coordination
- [ ] Clear documentation for future agents
- [ ] Reproducible pipeline for adding new negative datasets
- [ ] Cross-agent validation completed

### 5. Communication Protocol

#### Daily Updates
Each agent updates progress in:
- `docs/coordination/11-20-2025/[agent]_progress.md`

#### Blockers
Report blockers immediately in:
- `docs/coordination/11-20-2025/blockers.md`

#### Final Validation
All agents sign-off on:
- `docs/coordination/11-20-2025/integration_complete.md`

## Next Steps

1. **Immediate**: Grok starts data reorganization
2. **Parallel**: Claude finalizes schema, Gemini prepares pipeline updates  
3. **Integration**: Test training with combined positive + negative datasets
4. **Validation**: Measure false positive reduction

---

**Owner:** All Agents (Grok, Gemini, Claude)  
**Review Date:** November 22, 2025  
**Expected Completion:** November 22, 2025
