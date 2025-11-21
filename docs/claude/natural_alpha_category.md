
# Natural Alpha Variation Category

**Category:** natural_alpha_variation
**Method Constraint:** alpha_gradient_detection_should_fail
**Why It Matters:** 
  PNG images with transparency gradients (anti-aliased edges, fade effects)
  appear to have LSB variation but contain no hidden data.
  Model learns: natural alpha ≠ steganography

**Examples:**
  - 5 false positive clean images with natural alpha
  - Used to teach model robustness

**Schema:** All records follow v1.0 (8 required fields)
**Quality:** 100% extraction-verified as clean
