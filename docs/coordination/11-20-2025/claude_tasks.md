# Claude Tasks - Dataset Integration Plan
**Date:** November 20, 2025  
**Priority:** High - Schema & Cross-Agent Coordination

## Immediate Actions (Day 1)

### Task 1: Finalize Unified Negative Schema
**File:** `docs/coordination/negative_schema.md`

**Target Schema:**
```json
{
  "filename": "string",
  "category": "rgb_no_alpha|uniform_alpha|natural_noise|repetitive_patterns",
  "method_constraint": "alpha_detection_should_fail|lsb_detection_should_fail|palette_detection_should_fail|exif_detection_should_fail",
  "expected_behavior": "clean",
  "format_type": "RGB|RGBA|GIF|PNG|WebP|JPEG",
  "generation_method": "synthetic|natural|augmented",
  "validation_status": "verified_clean|pending|failed",
  "metadata": {
    "original_source": "string",
    "generation_params": {},
    "extraction_results": []
  }
}
```

### Task 2: Create Negative Dataset Validation Script
**File:** `scripts/validate_negative_dataset.py`

**Validation Checks:**
```python
def validate_negative_dataset(dataset_path):
    """Comprehensive validation of negative dataset"""
    
    checks = {
        'structure_valid': check_directory_structure(dataset_path),
        'schema_compliant': validate_manifest_schema(dataset_path),
        'truly_clean': verify_no_hidden_data(dataset_path),
        'format_balanced': check_format_distribution(dataset_path),
        'categories_complete': verify_all_categories_present(dataset_path)
    }
    
    return all(checks.values()), checks

def verify_no_hidden_data(dataset_path):
    """Critical: Ensure negatives contain no steganography"""
    for category in NEGATIVE_CATEGORIES:
        for img_path in glob(f"{dataset_path}/{category}/*"):
            for method in STEGO_METHODS:
                result = extract_stego_data(img_path, method)
                if result is not None and len(result) > 0:
                    raise ValueError(f"Found hidden data in negative: {img_path}")
    return True
```

### Task 3: Cross-Agent Schema Coordination
**Review Grok's Current Manifest:**
- [ ] Read `datasets/grok_submission_2025/negatives/manifest.jsonl`
- [ ] Compare with unified schema
- [ ] Create migration guide for Grok

**Validate Gemini's Integration Plan:**
- [ ] Review Gemini's proposed training changes
- [ ] Ensure schema compatibility
- [ ] Provide feedback on negative sampling strategy

## Day 2 Tasks

### Task 4: Create Integration Test Suite
**File:** `tests/test_negative_integration.py`

**Test Coverage:**
```python
def test_cross_agent_integration():
    """Test end-to-end integration between Grok, Gemini, Claude"""
    
    # Test Grok's data structure
    grok_negatives = load_negative_dataset('datasets/grok_submission_2025/negatives/')
    assert validate_schema(grok_negatives.manifest)
    
    # Test Gemini's training integration
    training_batch = create_training_batch_with_negatives()
    assert training_batch.contains_negatives
    
    # Test 6-stream extraction on negatives
    for neg_example in training_batch.negatives:
        tensors = load_unified_input(neg_example.path, 'negative')
        assert all(tensor is not None for tensor in tensors.values())
    
    # Test loss computation with negatives
    model = UnifiedStarlightDetector()
    loss = model.training_step(training_batch)
    assert loss.item() > 0

def test_negative_effectiveness():
    """Verify negatives reduce false positives"""
    
    # Test set of known false positive cases
    false_positive_cases = load_special_case_examples()
    
    # Model trained without negatives
    baseline_model = load_baseline_model()
    baseline_fp_rate = evaluate_fp_rate(baseline_model, false_positive_cases)
    
    # Model trained with negatives  
    enhanced_model = load_model_with_negatives()
    enhanced_fp_rate = evaluate_fp_rate(enhanced_model, false_positive_cases)
    
    # Should show improvement
    assert enhanced_fp_rate < baseline_fp_rate
```

### Task 5: Documentation and Handoff
**File:** `docs/coordination/11-20-2025/integration_guide.md`

**Contents:**
- [ ] Final negative schema specification
- [ ] Step-by-step integration guide for future agents
- [ ] Troubleshooting common issues
- [ ] Performance expectations and validation criteria

## Day 3 Tasks

### Task 6: Final Integration Validation
**Review All Agent Work:**
- [ ] Validate Grok's reorganized dataset structure
- [ ] Verify Gemini's training pipeline changes
- [ ] Run end-to-end integration test

**Sign-off Criteria:**
- [ ] All validation tests pass
- [ ] No blockers remaining
- [ ] Clear documentation for future work

### Task 7: Create Consensus Report
**File:** `docs/coordination/11-20-2025/consensus_report.md`

**Contents:**
- [ ] Summary of integration work completed
- [ ] Performance improvements measured
- [ ] Lessons learned for future coordination
- [ ] Recommendations for ongoing negative dataset management

## Coordination Requirements

### Daily Communication
- [ ] Update progress in `docs/coordination/11-20-2025/claude_progress.md`
- [ ] Review Grok's data reorganization completion
- [ ] Validate Gemini's training integration

### Cross-Agent Dependencies
- [ ] Grok: Complete data reorganization before Day 2 validation
- [ ] Gemini: Implement training changes before Day 3 testing
- [ ] All: Participate in final integration validation

## Success Criteria

### Schema
- [ ] Unified negative schema finalized and documented
- [ ] All agents agree on schema specification
- [ ] Migration guide created for existing datasets

### Integration
- [ ] Grok's negatives load correctly in Gemini's training
- [ ] 6-stream extraction works with all negative categories
- [ ] Training shows reduced false positives on special cases

### Documentation
- [ ] Complete integration guide for future agents
- [ ] Validation test suite passes
- [ ] Consensus report completed

## Blockers
- **Schema Agreement:** May need iteration between agents
- **Validation Time:** Full dataset validation is computationally expensive
- **Integration Issues:** Unknown edge cases in training pipeline

## Escalation Path
If blocked, update: `docs/coordination/11-20-2025/blockers.md`

---
**Owner:** Claude  
**Due:** November 22, 2025  
**Dependencies:** Grok (data), Gemini (integration)
