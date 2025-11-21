# Gemini Tasks - Dataset Integration Plan
**Date:** November 20, 2025  
**Priority:** High - Integrate Grok's Negatives for Training

## Immediate Actions (Day 1-2)

### Task 1: Update Training Data Scanner
**Current:** Scans only `datasets/*_submission_*/clean/` and `datasets/*_submission_*/stego/`  
**Target:** Also scan `datasets/*_submission_*/negatives/`

**File to Modify:** `scripts/starlight_utils.py` or training script
```python
def load_training_datasets():
    datasets = {
        'clean': scan_directories('datasets/*_submission_*/clean/'),
        'stego': scan_directories('datasets/*_submission_*/stego/'),
        'negatives': scan_directories('datasets/*_submission_*/negatives/')  # NEW
    }
    return datasets
```

### Task 2: Integrate Negative Examples in Training Pipeline
**Location:** Training script (likely `trainer.py` or similar)

**Required Changes:**
```python
class UnifiedTrainer:
    def __init__(self):
        # Add negative sampling strategy
        self.negative_sampler = NegativeSampler(
            negative_ratio=0.33,  # 1 negative per 3 examples
            categories=['rgb_no_alpha', 'uniform_alpha', 'natural_noise', 'repetitive_patterns']
        )
    
    def load_batch(self):
        # Load balanced mix of clean, stego, and negatives
        batch = {
            'clean': self.load_clean_examples(batch_size // 3),
            'stego': self.load_stego_examples(batch_size // 3), 
            'negatives': self.load_negative_examples(batch_size // 3)
        }
        return batch
```

### Task 3: Update 6-Stream Extraction for Negatives
**File:** `scripts/starlight_utils.py`

**Ensure:** `load_unified_input()` handles negative examples correctly
```python
def load_unified_input(img_path, label_type):
    # label_type: 'clean', 'stego', 'negative'
    tensors = extract_six_streams(img_path)
    
    if label_type == 'negative':
        # Special handling for negative examples
        # Teach model what NOT to detect
        target_label = 0  # clean
        constraint_mask = generate_constraint_mask(img_path)
        return tensors, target_label, constraint_mask
    
    return tensors, target_label, None
```

## Day 2-3 Tasks

### Task 4: Implement Negative-Specific Loss Weighting
**Purpose:** Teach model to avoid false positives on special cases

```python
def compute_loss(predictions, targets, metadata):
    base_loss = F.cross_entropy(predictions, targets)
    
    if metadata['type'] == 'negative':
        # Higher weight for negative examples
        # Especially important for special case categories
        category_weight = get_category_weight(metadata['category'])
        loss = base_loss * category_weight  # e.g., 2.0x weight
    
    return loss
```

### Task 5: Test Integration with Grok's Data
**Validation Script:** `test_negative_integration.py`

```python
def test_training_with_negatives():
    # Test loading Grok's reorganized negatives
    negatives = load_negatives('datasets/grok_submission_2025/negatives/')
    
    # Verify 6-stream extraction works
    for neg_path in negatives[:10]:  # Sample test
        tensors = load_unified_input(neg_path, 'negative')
        assert tensors['pixel'] is not None
        assert tensors['alpha'] is not None
        # ... etc for all 6 streams
    
    # Test training step
    model = UnifiedStarlightDetector()
    batch = create_test_batch_with_negatives()
    loss = model.training_step(batch)
    assert loss.requires_grad
```

## Coordination Requirements

### Dependencies
- [ ] Wait for Grok to complete data reorganization
- [ ] Get Claude's final schema specification
- [ ] Verify negative dataset structure before integration

### Before Training
- [ ] Confirm Grok's data is in `datasets/grok_submission_2025/negatives/`
- [ ] Validate manifest schema matches expectations
- [ ] Test extraction on sample negative images

### Integration Testing
- [ ] Run small training test with 100 examples including negatives
- [ ] Verify model learns from negative examples
- [ ] Check for reduction in false positives on special cases

## Success Criteria

### Technical
- [ ] Training script loads negatives from all submission directories
- [ ] 6-stream extraction works with negative examples
- [ ] Loss function properly weights negative examples
- [ ] No training errors with mixed positive/negative batches

### Performance
- [ ] Model shows reduced false positives on special cases
- [ ] Training convergence is stable with negative examples
- [ ] Overall detection accuracy is maintained

## Blockers
- **Data Structure:** Waiting for Grok to reorganize data
- **Schema:** Need Claude's final specification
- **Extraction:** May need to handle edge cases in negative examples

## Escalation Path
If blocked, update: `docs/coordination/11-20-2025/blockers.md`

---
**Owner:** Gemini  
**Due:** November 22, 2025  
**Dependencies:** Grok (data reorganization), Claude (schema)
