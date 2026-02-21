"""
Image preprocessing utilities for the Signature Matching Agent.

Pipeline: Grayscale → Light denoise → Auto-crop whitespace → Re-encode as PNG bytes.

Design choices:
- Grayscale removes colour noise while preserving stroke thickness / pressure info.
- Gentle Gaussian blur (sigma=0.5) smooths paper texture without erasing thin strokes.
- NO binarisation — pressure patterns and line-weight variation are preserved.
- Auto-crop trims excess whitespace so the model focuses on the signature region.
"""

from __future__ import annotations

import io
from PIL import Image, ImageFilter, ImageOps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_signature(
    image_bytes: bytes,
    *,
    grayscale: bool = True,
    denoise: bool = True,
    denoise_radius: float = 0.5,
    autocrop: bool = True,
    autocrop_padding: int = 20,
    target_long_edge: int | None = 1024,
) -> bytes:
    """
    Preprocess a signature image and return cleaned PNG bytes.

    Parameters
    ----------
    image_bytes : raw bytes of the source image (any PIL-supported format).
    grayscale   : convert to grayscale (recommended).
    denoise     : apply light Gaussian blur to reduce paper texture.
    denoise_radius : sigma for the Gaussian blur (lower = gentler).
    autocrop    : trim surrounding whitespace.
    autocrop_padding : pixels of padding to keep around the cropped signature.
    target_long_edge : resize so the longest side is at most this many pixels.
                       Set to None to skip resizing.

    Returns
    -------
    PNG-encoded bytes of the preprocessed image.
    """
    img = Image.open(io.BytesIO(image_bytes))

    # 0. Fix EXIF orientation (phone photos may be stored rotated)
    img = ImageOps.exif_transpose(img)

    # 1. Grayscale
    if grayscale:
        img = ImageOps.grayscale(img)

    # 2. Light denoise (Gaussian blur with small sigma)
    if denoise:
        img = img.filter(ImageFilter.GaussianBlur(radius=denoise_radius))

    # 3. Auto-crop whitespace
    if autocrop:
        img = _autocrop(img, padding=autocrop_padding)

    # 4. Resize (keep aspect ratio)
    if target_long_edge and max(img.size) > target_long_edge:
        img.thumbnail((target_long_edge, target_long_edge), Image.LANCZOS)

    # Encode as PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def preprocess_signature_pair(
    img1_bytes: bytes,
    img2_bytes: bytes,
    **kwargs,
) -> tuple[bytes, bytes]:
    """Convenience wrapper – preprocess two images with identical settings."""
    return (
        preprocess_signature(img1_bytes, **kwargs),
        preprocess_signature(img2_bytes, **kwargs),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _autocrop(img: Image.Image, padding: int = 20) -> Image.Image:
    """
    Trim whitespace around the signature.

    Works by inverting the image (so ink becomes white), finding the bounding
    box of non-zero pixels, and cropping with a padding margin.
    """
    # Work on a grayscale copy for detection (source may already be L or RGB)
    detect = img.convert("L")
    inverted = ImageOps.invert(detect)
    bbox = inverted.getbbox()

    if bbox is None:
        # Entirely blank image – return as-is
        return img

    left, upper, right, lower = bbox
    left = max(0, left - padding)
    upper = max(0, upper - padding)
    right = min(img.width, right + padding)
    lower = min(img.height, lower + padding)

    return img.crop((left, upper, right, lower))


# ---------------------------------------------------------------------------
# Quick visual check (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python image_preprocessing.py <image_path> [output_path]")
        print("Example: python image_preprocessing.py Data/VR1.jpg Data/VR1_processed.png")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_processed.png"

    with open(src, "rb") as f:
        raw = f.read()

    processed = preprocess_signature(raw)

    with open(dst, "wb") as f:
        f.write(processed)

    raw_kb = len(raw) / 1024
    proc_kb = len(processed) / 1024
    print(f"Original : {raw_kb:,.1f} KB")
    print(f"Processed: {proc_kb:,.1f} KB  ({proc_kb/raw_kb*100:.0f}% of original)")
    print(f"Saved to : {dst}")
