"""
Signature detection and cropping using Azure AI Document Intelligence.

Uses the prebuilt-layout model with the **signatures** add-on feature to:
1. Detect whether an uploaded image actually contains a handwritten signature.
2. Extract the bounding region of the detected signature.
3. Crop the image to the signature area (with configurable padding).

This runs BEFORE grayscaling / LLM comparison so the downstream pipeline
only receives a tightly-cropped signature region.

Requires:
    pip install azure-ai-documentintelligence
    Environment variable: DOCUMENT_INTELLIGENCE_ENDPOINT
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field

from PIL import Image

from app.config import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SignatureDetectionResult:
    """Outcome of running Document Intelligence signature detection."""

    signature_found: bool
    confidence: float = 0.0
    cropped_bytes: bytes | None = None
    bbox: tuple[int, int, int, int] | None = None  # pixel coords (left, top, right, bottom)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def detect_and_crop_signature(
    image_bytes: bytes,
    *,
    confidence_threshold: float = 0.5,
    padding: int = 20,
) -> SignatureDetectionResult:
    """
    Analyse *image_bytes* with Azure Document Intelligence's prebuilt-layout
    model (signatures feature) and, if a signature is detected, crop the image
    to the signature bounding region.

    Parameters
    ----------
    image_bytes : raw bytes of the source image.
    confidence_threshold : minimum confidence to accept a detection.
    padding : extra pixels around the detected bounding box when cropping.

    Returns
    -------
    SignatureDetectionResult with ``signature_found`` flag, optional
    ``cropped_bytes`` (PNG), and pixel-level ``bbox``.
    """

    endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
    if not endpoint:
        logger.warning(
            "DOCUMENT_INTELLIGENCE_ENDPOINT not set — skipping signature detection"
        )
        return SignatureDetectionResult(signature_found=False)

    # Late imports so the rest of the app works even when the package is absent
    try:
        from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import (
            AnalyzeDocumentRequest,
        )
    except ImportError:
        logger.error(
            "azure-ai-documentintelligence is not installed — "
            "run: pip install azure-ai-documentintelligence"
        )
        return SignatureDetectionResult(signature_found=False)

    credential = _get_credential()

    try:
        # Convert unsupported formats (webp, bmp, etc.) to JPEG for DI
        di_bytes = _ensure_supported_format(image_bytes)
        b64_source = base64.b64encode(di_bytes).decode()

        async with DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=credential,
        ) as client:
            poller = await client.begin_analyze_document(
                "prebuilt-layout",
                body=AnalyzeDocumentRequest(bytes_source=b64_source),
            )
            result = await poller.result()

        # ----- Find the best signature across all pages ----- #
        if not result.pages:
            return SignatureDetectionResult(signature_found=False)

        # Debug: log what the API returned
        for page in result.pages:
            sigs = getattr(page, "signatures", None)
            logger.debug(
                "Page %s: unit=%s, width=%s, height=%s, signatures=%s, words=%d, lines=%d",
                getattr(page, "page_number", "?"),
                getattr(page, "unit", "?"),
                getattr(page, "width", "?"),
                getattr(page, "height", "?"),
                sigs,
                len(getattr(page, "words", None) or []),
                len(getattr(page, "lines", None) or []),
            )
        styles = getattr(result, "styles", None) or []
        for style in styles:
            logger.debug(
                "Style: is_handwritten=%s, confidence=%s, spans=%s",
                getattr(style, "is_handwritten", None),
                getattr(style, "confidence", None),
                getattr(style, "spans", None),
            )

        best_sig = None
        best_page = None
        best_confidence = 0.0

        for page in result.pages:
            for sig in getattr(page, "signatures", None) or []:
                kind = getattr(sig, "kind", "")
                conf = getattr(sig, "confidence", 0.0) or 0.0
                if kind == "signature" and conf > best_confidence:
                    best_sig = sig
                    best_page = page
                    best_confidence = conf

        # ----- Strategy A: page.signatures found a signature ----- #
        if best_sig is not None and best_confidence >= confidence_threshold:
            logger.info("Signature detected via page.signatures — confidence %.2f", best_confidence)

            bbox_page_units = _extract_bounding_box(best_sig, best_page, result)
            if bbox_page_units:
                crop_result = _crop_image(image_bytes, bbox_page_units, best_page, padding=padding)
                if crop_result:
                    cropped_bytes, pixel_bbox = crop_result
                    return SignatureDetectionResult(
                        signature_found=True,
                        confidence=best_confidence,
                        cropped_bytes=cropped_bytes,
                        bbox=pixel_bbox,
                    )
            # Found but couldn't crop
            return SignatureDetectionResult(signature_found=True, confidence=best_confidence)

        # ----- Strategy B: handwriting style detected (for standalone signature images) ----- #
        hw_confidence = 0.0
        styles = getattr(result, "styles", None) or []
        for style in styles:
            if getattr(style, "is_handwritten", False):
                hw_conf = getattr(style, "confidence", 0.0) or 0.0
                hw_confidence = max(hw_confidence, hw_conf)

        if hw_confidence >= confidence_threshold:
            logger.info("Handwritten content detected — confidence %.2f (using as signature signal)", hw_confidence)

            # Try to build a bounding box from handwriting spans → words
            first_page = result.pages[0]
            bbox_hw = _extract_handwriting_bbox(result, first_page)

            if bbox_hw:
                crop_result = _crop_image(image_bytes, bbox_hw, first_page, padding=padding)
                if crop_result:
                    cropped_bytes, pixel_bbox = crop_result
                    return SignatureDetectionResult(
                        signature_found=True,
                        confidence=hw_confidence,
                        cropped_bytes=cropped_bytes,
                        bbox=pixel_bbox,
                    )
            # Handwriting detected but couldn't crop — still mark as found
            return SignatureDetectionResult(signature_found=True, confidence=hw_confidence)

        # ----- Strategy C: image-based ink detection (Pillow fallback) ----- #
        # When DI can't classify the content as handwritten or returns 0 words,
        # use image processing to find dark-ink strokes on a light background.
        ink_result = _detect_ink_region(image_bytes, padding=padding)
        if ink_result is not None:
            ink_confidence, cropped_bytes, pixel_bbox = ink_result
            if ink_confidence >= confidence_threshold:
                logger.info(
                    "Ink-based detection succeeded — ink_density %.2f (using as signature signal)",
                    ink_confidence,
                )
                return SignatureDetectionResult(
                    signature_found=True,
                    confidence=ink_confidence,
                    cropped_bytes=cropped_bytes,
                    bbox=pixel_bbox,
                )

        # ----- No signature detected by any strategy ----- #
        logger.info(
            "No signature detected (best sig confidence: %.2f, handwriting confidence: %.2f, threshold: %.2f)",
            best_confidence, hw_confidence, confidence_threshold,
        )
        return SignatureDetectionResult(
            signature_found=False, confidence=max(best_confidence, hw_confidence)
        )

    except Exception as exc:
        logger.error("Document Intelligence signature detection failed: %s", exc)
        return SignatureDetectionResult(signature_found=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Formats accepted by Azure Document Intelligence
_SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "PDF"}


def _ensure_supported_format(image_bytes: bytes) -> bytes:
    """
    If *image_bytes* is in an unsupported format (e.g. WEBP), convert it to
    JPEG so that Document Intelligence can process it.  Returns the original
    bytes unchanged when the format is already supported.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = (img.format or "").upper()
        if fmt in _SUPPORTED_FORMATS:
            return image_bytes

        # Convert to JPEG (RGB — drop alpha if present)
        logger.info("Converting unsupported format '%s' → JPEG for Document Intelligence", fmt)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("Format detection/conversion failed (%s) — sending original bytes", exc)
        return image_bytes


def _detect_ink_region(
    image_bytes: bytes,
    *,
    padding: int = 20,
    min_ink_fraction: float = 0.002,
    max_ink_fraction: float = 0.60,
) -> tuple[float, bytes, tuple[int, int, int, int]] | None:
    """
    Pure-image fallback: detect dark ink strokes on a light background (or
    light ink on a dark background — auto-detected).

    1. Convert to grayscale.
    2. Detect background polarity (light vs dark) using the median pixel value.
    3. For light backgrounds, ``ink = pixel < threshold``.
       For dark backgrounds, ``ink = pixel > threshold``.
    4. Find bounding box of ink pixels.
    5. Calculate ink_density = ink_pixels_in_bbox / bbox_area → use as confidence.
    6. Return ``(confidence, cropped_png_bytes, pixel_bbox)`` or ``None``.

    Returns ``None`` when:
    * the image has no meaningful ink, or
    * the bounding box is degenerate (< 10 px in either dimension).
    """
    import numpy as np

    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    w, h = img.size
    total_pixels = w * h

    arr = np.array(img)
    median_val = int(np.median(arr))

    # Determine background polarity and threshold
    if median_val >= 128:
        # Light background → ink is dark (< threshold)
        ink_threshold = min(median_val - 30, 180)
        ink_mask = (arr < ink_threshold).astype(np.uint8)
        bg_label = "light"
    else:
        # Dark background → ink is light (> threshold)
        ink_threshold = max(median_val + 30, 75)
        ink_mask = (arr > ink_threshold).astype(np.uint8)
        bg_label = "dark"

    ink_pixel_count = int(ink_mask.sum())
    ink_fraction = ink_pixel_count / total_pixels

    logger.debug(
        "Ink fallback: median=%d, bg=%s, threshold=%d, ink_pixels=%d (%.4f of total)",
        median_val, bg_label, ink_threshold, ink_pixel_count, ink_fraction,
    )

    if ink_fraction < min_ink_fraction:
        logger.debug("Ink fallback: too few ink pixels (%.4f < %.4f)", ink_fraction, min_ink_fraction)
        return None

    if ink_fraction > max_ink_fraction:
        logger.debug(
            "Ink fallback: too many ink pixels (%.4f > %.4f) — likely not a clean signature",
            ink_fraction, max_ink_fraction,
        )
        return None

    # Bounding box of ink pixels
    rows = np.any(ink_mask, axis=1)
    cols = np.any(ink_mask, axis=0)
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return None

    top = int(row_indices[0])
    bottom = int(row_indices[-1])
    left = int(col_indices[0])
    right = int(col_indices[-1])

    if (right - left) < 10 or (bottom - top) < 10:
        return None

    # Ink density within the bounding box → use as confidence proxy
    bbox_area = (right - left + 1) * (bottom - top + 1)
    ink_in_bbox = int(ink_mask[top:bottom + 1, left:right + 1].sum())
    ink_density = ink_in_bbox / bbox_area

    # Confidence: scale ink density into [0, 1].
    # Typical signature ink density is 0.03–0.30.
    confidence = min(1.0, ink_density / 0.03)  # 3 % fill → 1.0

    # Apply padding and crop
    left_p = max(0, left - padding)
    top_p = max(0, top - padding)
    right_p = min(w, right + padding)
    bottom_p = min(h, bottom + padding)

    original = Image.open(io.BytesIO(image_bytes))
    cropped = original.crop((left_p, top_p, right_p, bottom_p))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")

    logger.info(
        "Ink fallback cropped: (%d,%d)–(%d,%d) from %dx%d → %dx%d  ink_density=%.3f  confidence=%.2f",
        left_p, top_p, right_p, bottom_p, w, h, cropped.width, cropped.height,
        ink_density, confidence,
    )

    return confidence, buf.getvalue(), (left_p, top_p, right_p, bottom_p)


def _get_credential():
    """Use Managed Identity in Azure App Service, fall back to CLI locally."""
    from azure.identity.aio import AzureCliCredential, ManagedIdentityCredential

    if os.getenv("WEBSITE_SITE_NAME"):
        return ManagedIdentityCredential()
    return AzureCliCredential()


def _extract_handwriting_bbox(
    result, page
) -> tuple[float, float, float, float] | None:
    """
    Build a bounding box from handwritten style spans mapped to word polygons.
    Used when ``page.signatures`` is empty but handwriting is detected.
    """
    styles = getattr(result, "styles", None) or []
    page_words = getattr(page, "words", None) or []
    if not page_words:
        return None

    all_xs: list[float] = []
    all_ys: list[float] = []

    for style in styles:
        if not getattr(style, "is_handwritten", False):
            continue
        for span in getattr(style, "spans", None) or []:
            s_off = getattr(span, "offset", span.get("offset", 0) if isinstance(span, dict) else 0)
            s_len = getattr(span, "length", span.get("length", 0) if isinstance(span, dict) else 0)
            s_end = s_off + s_len
            for word in page_words:
                w_span = getattr(word, "span", None)
                if w_span is None:
                    continue
                wo = getattr(w_span, "offset", w_span.get("offset", 0) if isinstance(w_span, dict) else 0)
                wl = getattr(w_span, "length", w_span.get("length", 0) if isinstance(w_span, dict) else 0)
                if wo < s_end and (wo + wl) > s_off:
                    poly = getattr(word, "polygon", None)
                    if poly:
                        _collect_polygon_coords(poly, all_xs, all_ys)

    if all_xs and all_ys:
        return (min(all_xs), min(all_ys), max(all_xs), max(all_ys))
    return None


def _extract_bounding_box(
    sig, page, result
) -> tuple[float, float, float, float] | None:
    """
    Return ``(left, top, right, bottom)`` in **page coordinate units** for a
    detected signature.  Tries multiple strategies.
    """

    # Strategy 1 — bounding_regions on the signature object itself
    bbox = _bbox_from_regions(getattr(sig, "bounding_regions", None))
    if bbox:
        return bbox

    # Strategy 2 — map signature spans → overlapping words (which have polygons)
    spans = getattr(sig, "spans", None) or []
    if spans and page:
        words = getattr(page, "words", None) or []
        if words:
            all_xs: list[float] = []
            all_ys: list[float] = []
            for span in spans:
                s_off = getattr(span, "offset", 0)
                s_len = getattr(span, "length", 0)
                s_end = s_off + s_len
                for word in words:
                    w_span = getattr(word, "span", None)
                    if w_span is None:
                        continue
                    wo = getattr(w_span, "offset", 0)
                    wl = getattr(w_span, "length", 0)
                    if wo < s_end and (wo + wl) > s_off:
                        poly = getattr(word, "polygon", None)
                        if poly:
                            _collect_polygon_coords(poly, all_xs, all_ys)
            if all_xs and all_ys:
                return (min(all_xs), min(all_ys), max(all_xs), max(all_ys))

    # Strategy 3 — look at handwritten styles and map to word bounding boxes
    styles = getattr(result, "styles", None) or []
    for style in styles:
        if not getattr(style, "is_handwritten", False):
            continue
        style_spans = getattr(style, "spans", None) or []
        if not style_spans:
            continue
        all_xs_hw: list[float] = []
        all_ys_hw: list[float] = []
        page_words = getattr(page, "words", None) or []
        for span in style_spans:
            s_off = getattr(span, "offset", 0)
            s_len = getattr(span, "length", 0)
            s_end = s_off + s_len
            for word in page_words:
                w_span = getattr(word, "span", None)
                if w_span is None:
                    continue
                wo = getattr(w_span, "offset", 0)
                wl = getattr(w_span, "length", 0)
                if wo < s_end and (wo + wl) > s_off:
                    poly = getattr(word, "polygon", None)
                    if poly:
                        _collect_polygon_coords(poly, all_xs_hw, all_ys_hw)
        if all_xs_hw and all_ys_hw:
            return (min(all_xs_hw), min(all_ys_hw), max(all_xs_hw), max(all_ys_hw))

    return None


def _bbox_from_regions(regions) -> tuple[float, float, float, float] | None:
    """Extract a bounding box from a list of BoundingRegion objects."""
    if not regions:
        return None
    for region in regions:
        polygon = getattr(region, "polygon", None)
        if not polygon or len(polygon) < 4:
            continue
        xs: list[float] = []
        ys: list[float] = []
        _collect_polygon_coords(polygon, xs, ys)
        if xs and ys:
            return (min(xs), min(ys), max(xs), max(ys))
    return None


def _collect_polygon_coords(
    polygon, xs: list[float], ys: list[float]
) -> None:
    """
    Append x/y values from *polygon* into *xs* / *ys*.

    Handles both flat lists ``[x1, y1, x2, y2, …]`` and lists of point
    objects with ``.x`` / ``.y`` attributes.
    """
    if not polygon:
        return
    first = polygon[0]
    if isinstance(first, (int, float)):
        # Flat list: [x1, y1, x2, y2, ...]
        xs.extend(polygon[i] for i in range(0, len(polygon), 2))
        ys.extend(polygon[i] for i in range(1, len(polygon), 2))
    else:
        # List of point objects or dicts
        for p in polygon:
            if isinstance(p, dict):
                xs.append(p.get("x", 0))
                ys.append(p.get("y", 0))
            else:
                xs.append(getattr(p, "x", 0))
                ys.append(getattr(p, "y", 0))


def _crop_image(
    image_bytes: bytes,
    bbox_page: tuple[float, float, float, float],
    page,
    *,
    padding: int = 20,
) -> tuple[bytes, tuple[int, int, int, int]] | None:
    """
    Crop *image_bytes* to *bbox_page* (in page-coordinate units) and return
    ``(png_bytes, pixel_bbox)`` or ``None`` if the crop is degenerate.
    """
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size

    page_width = getattr(page, "width", None) or w
    page_height = getattr(page, "height", None) or h
    page_unit = getattr(page, "unit", "pixel")

    # Scale factor: page-coord → pixel
    if page_unit == "inch":
        scale_x = w / page_width
        scale_y = h / page_height
    elif page_unit == "pixel":
        scale_x = 1.0
        scale_y = 1.0
    else:
        scale_x = w / page_width
        scale_y = h / page_height

    left = max(0, int(bbox_page[0] * scale_x) - padding)
    top = max(0, int(bbox_page[1] * scale_y) - padding)
    right = min(w, int(bbox_page[2] * scale_x) + padding)
    bottom = min(h, int(bbox_page[3] * scale_y) + padding)

    # Reject degenerate crops
    if (right - left) < 10 or (bottom - top) < 10:
        return None

    cropped = img.crop((left, top, right, bottom))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")

    logger.info(
        "Cropped signature region: (%d,%d)–(%d,%d) from %dx%d → %dx%d",
        left, top, right, bottom, w, h, cropped.width, cropped.height,
    )

    return buf.getvalue(), (left, top, right, bottom)


# ---------------------------------------------------------------------------
# Quick standalone test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import asyncio
    from dotenv import load_dotenv

    load_dotenv(override=True)

    # Minimal logger setup for standalone run
    import logging
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s] %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m app.services.signature_detection <image_path> [output_path]")
        print("Example: python -m app.services.signature_detection Data/RSP1.jpg Data/RSP1_cropped.png")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_sig_cropped.png"

    with open(src, "rb") as f:
        raw = f.read()

    print(f"Input : {src} ({len(raw) / 1024:.1f} KB)")
    print(f"Calling Document Intelligence...")

    result = asyncio.run(detect_and_crop_signature(raw))

    print(f"  signature_found : {result.signature_found}")
    print(f"  confidence      : {result.confidence:.2f}")
    print(f"  bbox (pixels)   : {result.bbox}")
    print(f"  cropped         : {'Yes' if result.cropped_bytes else 'No'}")

    if result.cropped_bytes:
        with open(dst, "wb") as f:
            f.write(result.cropped_bytes)
        print(f"  cropped size    : {len(result.cropped_bytes) / 1024:.1f} KB")
        print(f"  saved to        : {dst}")
    else:
        print("  No cropped image produced.")

