# Image Preprocessing Pipeline

When `preprocess=true` (default), each uploaded image goes through five steps before being sent to the model:

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
