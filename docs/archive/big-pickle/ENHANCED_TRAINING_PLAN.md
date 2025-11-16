# Enhanced Model Training Plan - Fix Performance Degradation

## Current Problem Analysis

**Baseline Performance**: 100% stego detection, 0% clean detection (perfect but may have false positives)
**Enhanced Model Performance**: 70.2% stego detection, 29.8% clean detection (significant degradation)

**Root Causes Identified**:
1. **Aggressive Format Constraint Loss**: The format constraint loss (weight 0.05) is too aggressive, interfering with primary stego detection
2. **Training Pipeline Issues**: Quality filtering removed too many samples, reducing training diversity
3. **Loss Component Imbalance**: Method loss and constraint loss dominating the primary stego detection task
4. **Learning Rate Scheduling**: Too aggressive LR reduction causing underfitting

## 1. Training Strategy

### Conservative Training Parameters
```
Epochs: 30 (reduced from 50 to prevent overfitting)
Batch Size: 16 (increased from 8 for better gradient stability)
Learning Rate: 2e-4 (higher initial rate for better convergence)
LR Scheduler: Cosine annealing with warm restarts (gentler than ReduceLROnPlateau)
Early Stopping: Patience 8 (more patient than current 5)
```

### Data Augmentation Strategy
```
Primary Augmentations:
- RandomHorizontalFlip: 50% probability
- RandomVerticalFlip: 20% probability (reduced from 50%)
- ColorJitter: brightness=0.05, contrast=0.05 (reduced from 0.1)
- RandomRotation: ±10 degrees (reduced from ±15)

NO aggressive augmentations that might destroy steganography signals
```

### Validation Strategy
```
Training Data: All submission datasets (chatgpt, claude, gemini, grok, maya)
Validation Data: Dedicated val/ dataset only
Validation Frequency: Every epoch
Validation Metrics: Stego accuracy, method accuracy, loss components
```

### Early Stopping Criteria
```
Primary Metric: Validation stego accuracy (must stay > 95%)
Secondary Metric: Validation loss (monitor for overfitting)
Stop Conditions:
- Val stego accuracy drops below 95% for 3 consecutive epochs
- No improvement in val loss for 8 epochs
- Training stego accuracy reaches 100% but val stego accuracy < 90%
```

## 2. Model Configuration

### Format Constraint Loss - DISABLED
```
Decision: DISABLE format constraint loss entirely
Reason: The constraint loss is interfering with primary stego detection
Alternative: Use gentle format-aware features without explicit constraints
```

### Loss Function Configuration
```
Primary Loss: BCEWithLogitsLoss for stego detection
Method Loss: CrossEntropyLoss with class balancing
Loss Weights:
- Stego loss: 1.0 (primary objective)
- Method loss: 0.005 (reduced from 0.01 to minimize interference)
- Format constraint: 0.0 (disabled)
```

### Model Architecture Adjustments
```
Metadata Weight: 0.2 (reduced from 0.3 to further reduce EXIF dominance)
Dropout: 0.2 (reduced from 0.3 for better learning)
Gradient Clipping: 1.0 (maintained for stability)
```

### Class Weighting Strategy
```
Compute inverse frequency weights for method classification
Apply only to method loss, not stego loss
Minimum weight: 0.5, Maximum weight: 2.0 (to prevent extreme imbalance)
```

## 3. Data Pipeline

### Quality Filtering - DISABLED
```
Decision: DISABLE aggressive quality filtering
Reason: Removing too many diverse samples, hurting generalization
Alternative: Use basic validation only (file existence, readable format)
```

### Data Balancing Strategy
```
Strategy: 'balanced_classes' for training
- Equal representation from each stego method
- Clean samples matched to total stego count
- Minimum 50 samples per method for stability

Validation Strategy: 'use_all_stego'
- All validation stego samples for comprehensive testing
- Clean samples oversampled to match
```

### Format Awareness Approach
```
Gentle Format Features Only:
- has_alpha, is_palette, is_rgb (binary flags)
- width_norm, height_norm (normalized dimensions)
- NO aggressive format constraints or penalties
- Features used for information, not constraint enforcement
```

### Dataset Creation Parameters
```
Training Patterns:
- Clean: "datasets/*_submission_*/clean"
- Stego: "datasets/*_submission_*/stego"

Validation Patterns:
- Clean: "datasets/val/clean"
- Stego: "datasets/val/stego"

Exclusions: None (use all available data)
```

## 4. Training Monitoring

### Metrics to Track
```
Primary Metrics:
- Training/Validation stego accuracy (target: >95%)
- Training/Validation stego loss
- Training/Validation method accuracy

Secondary Metrics:
- Loss component breakdown (stego, method)
- Learning rate schedule
- Gradient norms
- Per-class method accuracy
```

### Logging Frequency
```
Console Output: Every epoch
Detailed Metrics: Every epoch
Loss Components: Every epoch
Model Checkpoints: Best validation loss only
Learning Rate: Every epoch
```

### Checkpoint Strategy
```
Save Strategy:
- Best model based on validation stego accuracy
- Keep only best checkpoint (not all epochs)
- Final model export to ONNX format
- Training history saved to JSON
```

### Validation Monitoring
```
Alert Conditions:
- Val stego accuracy < 95%: WARNING
- Val stego accuracy < 90%: CRITICAL
- Val loss increasing for 3+ epochs: WARNING
- Training accuracy 100% but val accuracy < 90%: OVERFITTING
```

## 5. Verification Plan

### Model Testing Protocol
```
Test Datasets:
1. Training subset (20% held out)
2. Full validation dataset
3. Individual submission datasets (cross-validation)
4. Clean-only dataset (false positive test)
```

### Comparison Metrics vs Baseline
```
Primary Metrics:
- Stego detection rate (target: >95%, baseline: 100%)
- Clean detection rate (target: <5% false positives, baseline: 0%)
- Overall accuracy (target: >95%)

Secondary Metrics:
- Method classification accuracy (target: >70%)
- AUC-ROC score (target: >0.95)
- Precision/Recall balance
```

### Success Criteria
```
Minimum Requirements:
- Stego detection accuracy: ≥95%
- Clean false positive rate: ≤5%
- Overall accuracy: ≥95%
- Method classification accuracy: ≥70%

Target Goals:
- Stego detection accuracy: ≥98%
- Clean false positive rate: ≤2%
- Overall accuracy: ≥98%
- Method classification accuracy: ≥75%
```

### False Positive Rate Targets
```
Acceptable Range: 2-5% false positives on clean images
Target: <2% false positives
Maximum: 5% false positives (beyond this, model fails)
Test Dataset: Minimum 1000 clean images for statistical significance
```

## Implementation Steps

### Phase 1: Setup (Day 1)
1. Backup current enhanced model
2. Create new training script based on corrected_enhanced_trainer.py
3. Disable format constraint loss
4. Set conservative training parameters

### Phase 2: Training (Day 1-2)
1. Train with conservative parameters for 30 epochs
2. Monitor metrics closely
3. Early stop if criteria met
4. Save best model

### Phase 3: Validation (Day 2)
1. Test on validation dataset
2. Test on individual submissions
3. Measure false positive rate on clean images
4. Compare with baseline

### Phase 4: Iteration (Day 2-3)
1. If false positive rate >5%, adjust parameters
2. If stego detection <95%, increase training epochs
3. Fine-tune loss weights if needed
4. Retrain if necessary

### Phase 5: Final Testing (Day 3)
1. Comprehensive testing on all datasets
2. Generate performance report
3. Compare with baseline model
4. Deploy if success criteria met

## Risk Mitigation

### High-Risk Factors
1. **Overfitting to training data**: Mitigated by early stopping and validation monitoring
2. **Underfitting due to conservative parameters**: Mitigated by higher learning rate and adequate epochs
3. **Class imbalance**: Mitigated by balanced class sampling and class weights
4. **Format constraints interfering**: Mitigated by disabling constraint loss

### Fallback Plans
1. **If stego detection <90%**: Increase epochs to 50, reduce regularization
2. **If false positives >10%**: Increase format awareness gently, add more clean data
3. **If training unstable**: Reduce learning rate, increase batch size
4. **If no improvement**: Fall back to baseline model with minor improvements

## Expected Outcomes

### Conservative Estimate
- Stego detection: 95-98%
- Clean false positives: 3-5%
- Overall accuracy: 95-97%
- Method classification: 70-75%

### Optimistic Estimate
- Stego detection: 98-99%
- Clean false positives: 1-3%
- Overall accuracy: 97-99%
- Method classification: 75-80%

This plan prioritizes maintaining the excellent stego detection performance of the baseline while systematically reducing false positives through conservative, incremental improvements rather than aggressive changes that could degrade performance.