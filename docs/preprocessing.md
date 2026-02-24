# Image Preprocessing Pipeline

When `preprocess=true` (default), each uploaded image goes through the steps below before being sent to the model.

If **Detect Signature** is also enabled (`detect_signature=true`), an additional step runs *before* the pipeline below: Azure Document Intelligence analyses the image to detect and crop the signature region. See the [Signature Detection](#signature-detection) section at the end.

## Standard Pipeline

Each uploaded image goes through five steps:

| Step | What it does | Why |
|---|---|---|
| **1. EXIF orientation fix** | Rotates the image per EXIF metadata | Phone photos are often stored rotated |
| **2. Grayscale conversion** | Converts to single-channel luminance | Removes colour noise, preserves pressure info |
| **3. Gaussian blur** (σ=0.5) | Light smoothing pass | Reduces paper texture without erasing thin strokes |
| **4. Auto-crop** | Trims whitespace, adds 20px padding | Centres the signature, removes scanning artefacts |
| **5. Resize** | Scales to 1024px on longest edge | Keeps file size reasonable; never upscales |

---

## Implementation Details

- Located in `app/services/preprocessing.py`
- Uses **Pillow** (`PIL`) for all image operations
- Runs in a **thread pool** via `asyncio.run_in_executor` — the CPU-bound Pillow work doesn't block the event loop
- Both images in a pair are preprocessed together via `preprocess_signature_pair()`
- Output format is always PNG regardless of input format

---

## Signature Detection

When `detect_signature=true`, an additional step runs **before** the standard pipeline above.

Located in `app/services/signature_detection.py`, it uses Azure Document Intelligence's `prebuilt-layout` model with a 3-strategy cascade:

| Strategy | Method | When it fires |
|---|---|---|
| **A** | `page.signatures` field | When the API returns explicit signature objects (future API versions) |
| **B** | Handwriting style detection | When DI classifies content as `is_handwritten=True` — maps handwriting spans to word polygons to build a bounding box |
| **C** | Pillow ink-region detection | Fallback when DI returns no text or classifies content as printed — finds dark-ink-on-light (or light-on-dark) strokes via pixel analysis |

**Format handling:** Unsupported formats (e.g. WebP) are auto-converted to JPEG before sending to Document Intelligence.

**⚠ Limitations:**
- Strategy B treats all handwriting as a potential signature — it cannot distinguish a handwritten amount from a signature on the same document
- Strategy C is purely pixel-based with no semantic understanding
- **Recommended:** Provide pre-cropped signature images when possible; enable Detect Signature only for full documents/cheques/forms
