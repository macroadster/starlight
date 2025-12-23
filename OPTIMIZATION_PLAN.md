# Patch-Based Scanning Optimization Plan

## Current Status
✅ **Scanner works for large images** - no longer hanging
✅ **Detects large images correctly** - prints detection message  
✅ **Processes successfully** - returns accurate results
✅ **Maintains compatibility** - all existing functionality preserved

## Issue Analysis

The hanging was caused by **circular dependency** in patch-based scanning:
1. `_scan_logic()` called `scan_large_image()` for large images
2. `scan_large_image()` tried to create new session and call `_scan_logic()` again
3. Created infinite recursion loop

## Optimization Strategy

### Phase 1: ✅ COMPLETED - Fix Critical Issue
- **Removed circular dependency** by disabling complex patch scanning
- **Added fallback to normal scanning** for large images
- **Preserved all existing functionality**
- **Maintains detection accuracy**

### Phase 2: 🚧 IN PROGRESS - Efficient Patch Scanning

**To implement proper patch-based scanning:**

#### Approach A: Separate Inference Function
```python
def _run_model_inference(session, tensors):
    """Pure model inference without scanning logic"""
    # No circular references
    return stego_logits, method_id, method_probs

def scan_large_image(image_path, model_path):
    """Process large images with patching"""
    # Create independent session
    session = create_session(model_path)
    
    # Extract patches
    patches = extract_patches(image_array)
    for patch in patches:
        result = _run_model_inference(session, patch_tensors)
        # Collect and aggregate results
```

#### Approach B: Refactor Architecture
```
StarlightScanner
├── scan_file() - Creates session, calls orchestration
├── _scan_logic() - Orchestrates flow, detects size
├── _run_inference() - Pure model inference
└── scan_large_image() - Manages patches and aggregation
```

#### Approach C: Worker Session Management
```python
# Add model_path to session objects
class SessionWrapper:
    def __init__(self, session, model_path):
        self.session = session
        self.model_path = model_path

# Enables recursive calls without losing path info
```

## Benefits of Optimized System

### Performance Improvements
- **🔥 Faster processing**: Parallel patch scanning vs. single large image
- **⚡ Lower memory**: Process 256x256 chunks vs. full resolution
- **🎯 Better detection**: Patches focus on local stego patterns
- **📊 Rich results**: Aggregation with confidence metrics

### Scalability
- **🖼️ Any image size**: No theoretical limit
- **🎚️ Configurable overlap**: Adjustable stride for coverage vs. speed
- **🔄 Flexible aggregation**: Max, average, weighted methods
- **💾 Memory efficient**: Cleanup after each patch

### API Enhancements
```python
ScanOptions(
    enable_patch_scanning=True,      # Enable/disable
    patch_size=256,                 # Patch dimension  
    patch_stride=128,               # Overlap amount
    patch_aggregation='weighted'       # Result combination
)
```

## Implementation Priority

1. **🔴 Critical**: Complete patch scanning without circular deps
2. **🟡 High**: Add configurable patch parameters  
3. **🟢 Medium**: Optimize aggregation algorithms
4. **🔵 Low**: Add performance monitoring

## Current Solution
✅ **Works immediately** - All images processed successfully
✅ **Maintains accuracy** - Still detects steganography
✅ **No breaking changes** - Existing API preserved
✅ **Foundation ready** - Clean architecture for future optimization

The scanner now handles real-world images of any size while preparing for advanced patch-based optimizations.