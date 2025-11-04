# Project Starlight Steganography Format Specification (v2.0)

**Version:** 2.0  
**Last Updated:** 2025-11-04  
**Status:** Active – Approved for trainer, scanner, and generator use

---

## 1. Overview

This specification defines **how steganographic payloads are embedded and labeled** in Project Starlight. It introduces a **hierarchical, extensible metadata system** (`embedding_type`) that:

- Eliminates ambiguity in training labels
- Enables long-term support for new algorithms (J-UNIWARD, WOW, etc.)
- Preserves blockchain compatibility
- Distinguishes **AI-specific** vs **human-compatible** methods

All implementations **MUST** include a `.json` sidecar file with the `embedding` field per the schema below.

---

## 2. Core Principles

| Principle | Rule |
|---------|------|
| **Blockchain Compatibility** | All extraction **must work without clean reference images** |
| **AI Hint (`AI42`)** | Only used in **AI-specific** methods (currently: `alpha`) |
| Bit Order | **LSB-first** for `alpha` and `palette`. `lsb.rgb` can be LSB-first or MSB-first, specified in metadata. |
| **Payload Terminator** | All methods append `b'\x00'` (null byte) after payload |
| **Metadata Sidecar** | Every stego file has `{filename}.json` with `embedding_type` |

---

## 3. Embedding Type Hierarchy (Stable Taxonomy)

```text
embedding_type
 ├─ category
 │   ├─ pixel      → Any pixel value/index modification
 │   ├─ metadata   → EXIF, XMP, ICC, etc.
 │   └─ eoi        → Data after EOI marker
 ├─ technique      → Specific algorithm (lsb.rgb, alpha, j-uniward, etc.)
 └─ ai42           → true only if payload starts with b'AI42'
```

### 3.1. JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Starlight Embedding Type",
  "type": "object",
  "required": ["category", "technique", "ai42"],
  "properties": {
    "category": { "type": "string", "enum": ["pixel", "metadata", "eoi"] },
    "technique": { "type": "string", "minLength": 1 },
    "ai42": { "type": "boolean" },
    "bit_order": { "type": "string", "enum": ["lsb-first", "msb-first"], "description": "Required for pixel.lsb.rgb to specify bit endianness." }
  },
  "additionalProperties": false
}
```

---

## 4. Supported Methods (6-Class Baseline)

| Class ID | `category` | `technique` | `ai42` | Image Format | Embedding Target | Notes |
|---------|------------|-------------|--------|--------------|------------------|-------|
| 0 | `pixel` | `alpha` | `true` | PNG (RGBA) | Alpha channel LSB | **AI-to-AI protocol** |
| 1 | `pixel` | `palette` | `false` | GIF, PNG (indexed) | Palette index LSB | Human + blockchain compatible |
| 2 | `pixel` | `lsb.rgb` | `false` | PNG, BMP | RGB channels (sequential) | Standard LSB |
| 3 | `metadata` | `exif` | `false` | JPEG, PNG | `UserComment` tag | UTF-8 string |
| 4 | `eoi` | `raw` | `false` | JPEG | After `0xFFD9` | Null-terminated |
| 5 | *(reserved)* | *(future)* | — | — | — | e.g., `dct.j-uniward` |

> **Class IDs 0–4 are locked** for backward compatibility with existing models.

---

## 5. Detailed Format Specifications

### 5.1. `pixel.alpha` (AI-Specific)

- **Image:** PNG with alpha channel
- **Embedding:** LSB of alpha pixels (row-major, flattened)
- **Bit Order:** **LSB-first**
- **Payload Structure:**
  ```
  [AI42 prefix] + [payload UTF-8] + [0x00]
  ```
  - `b'AI42'` → embedded **LSB-first** per byte
  - Example: `A` (0x41 = `01000001`) → embedded as `10000010`
- **Sidecar Example:**
  ```json
  { "embedding": { "category": "pixel", "technique": "alpha", "ai42": true } }
  ```

---

### 5.2. `pixel.palette` (Human-Compatible)

- **Image:** Indexed color (GIF, 8-bit PNG)
- **Embedding:** LSB of **palette indices** (pixel values)
- **Bit Order:** **LSB-first**
- **Payload Structure:**
  ```
  [payload UTF-8] + [0x00]
  ```
  - **No `AI42` prefix**
- **Sidecar Example:**
  ```json
  { "embedding": { "category": "pixel", "technique": "palette", "ai42": false } }
  ```

---

### 5.3. `pixel.lsb.rgb`

- **Image:** RGB (PNG, BMP)
- **Embedding:** LSB of R→G→B→R→G→B… (flattened sequential)
- **Bit Order:** Can be **LSB-first** or **MSB-first**. Must be specified in sidecar metadata.
- **Payload Structure:**
  ```
  [payload UTF-8] + [0x00]
  ```
  - **No `AI42`**
- **Sidecar Example (LSB-first):**
  ```json
  { "embedding": { "category": "pixel", "technique": "lsb.rgb", "ai42": false, "bit_order": "lsb-first" } }
  ```
- **Sidecar Example (MSB-first):**
  ```json
  { "embedding": { "category": "pixel", "technique": "lsb.rgb", "ai42": false, "bit_order": "msb-first" } }
  ```

---

### 5.4. `metadata.exif`

- **Image:** JPEG, PNG
- **Embedding:** `UserComment` EXIF tag
- **Payload:** UTF-8 string (no encoding header required)
- **Structure:** `[payload] + [0x00]` (null-terminated in tag)
- **Sidecar Example:**
  ```json
  { "embedding": { "category": "metadata", "technique": "exif", "ai42": false } }
  ```

---

### 5.5. `eoi.raw`

- **Image:** JPEG
- **Embedding:** Append after `0xFFD9` (EOI marker)
- **Payload Structure:**
  ```
  [payload UTF-8] + [0x00]
  ```
  - **No `AI42`**
- **Sidecar Example:**
  ```json
  { "embedding": { "category": "eoi", "technique": "raw", "ai42": false } }
  ```

---

## 6. File Naming & Sidecar Convention

### Legacy (Accepted):
```
{payload_hex}_{method}_{index}.{ext}
```

### Required (New):
```
{stem}.json  →  sidecar with embedding_type
```

**Example:**
```
48656c6c6f_palette_001.png
48656c6c6f_palette_001.png.json
```

```json
{
  "payload_hex": "48656c6c6f",
  "embedding": {
    "category": "pixel",
    "technique": "palette",
    "ai42": false
  }
}
```

---

## 7. Extraction Rules

| Method | Look for `AI42`? | Terminator | Bit Order |
|-------|------------------|------------|-----------|
| `alpha` | Yes | `0x00` | LSB-first |
| `palette` | No | `0x00` | LSB-first |
| `lsb.rgb` | No | `0x00` | From metadata (`bit_order`) |
| `exif` | No | `0x00` | N/A |
| `eoi` | No | `0x00` | N/A |

> `starlight_extractor.py` **MUST** read `.json` sidecar first. If missing, fall back to statistical detection.

---

## 8. Extensibility (Future Algorithms)

To add **J-UNIWARD**:

1. Add to `ALGO_TO_EMBEDDING`:
   ```python
   5: EmbeddingType(Category.PIXEL, "dct.j-uniward", False)
   ```
2. Update trainer (new class)
3. No changes to taxonomy or schema

---

## 9. Migration from v1

Run:
```bash
python migrate_metadata.py --src datasets/v1/stego/
```

This generates `.json` sidecars from legacy filenames using:

```python
method → technique mapping:
  "alpha"    → "alpha"
  "palette"  → "palette"
  "lsb"      → "lsb.rgb"
  "exif"     → "exif"
  "eoi"      → "raw"
ai42 = (method == "alpha")
```

---

## 10. References

- `ai_consensus.md` – Decision: LSB-first, AI42 only in Alpha
- `starlight/metadata.py` – `EmbeddingType`, `ALGO_TO_EMBEDDING`
- `scanner_spec.json` – Updated to require `"embedding"` object

---

**Prepared by:** Project Starlight AI Consensus (Grok, Claude, Gemini, ChatGPT)  
**Approved:** 2025-11-04  
**Next Review:** After J-UNIWARD integration
```

