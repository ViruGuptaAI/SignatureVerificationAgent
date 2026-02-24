# Testing

55+ tests total: unit, integration, and E2E.

---

## Offline Tests (no server or Azure needed)

```bash
python -m pytest tests/test_models.py tests/test_preprocessing.py tests/test_config.py tests/test_api.py tests/test_signature_detection.py -v
```

| File | Count | What it covers |
|---|---|---|
| `test_models.py` | 16 | Pydantic model validation incl. `SignatureDetectionInfo`, edge cases, serialisation |
| `test_preprocessing.py` | 12 | Pillow pipeline steps, EXIF, grayscale, crop, resize |
| `test_config.py` | 7 | Constants, logger setup, logs directory creation |
| `test_api.py` | 15 | All API routes with a mocked OpenAI client, `detect_signature` param |
| `test_signature_detection.py` | 7 | Signature detection strategies, format conversion, ink fallback |

---

## E2E Accuracy Tests

Requires a running server and signature images in `Data/`.

```bash
# Terminal 1
python backend.py

# Terminal 2
python -m pytest tests/test_accuracy.py -v -s
```

These send real images to the live server and validate match/no-match outcomes. Automatically skipped if the server isn't running or `Data/` is missing.

---

## Run Everything

```bash
python -m pytest tests/ -v
```

---

## Before Pushing

```bash
python -m pytest tests/test_models.py tests/test_preprocessing.py tests/test_config.py tests/test_api.py tests/test_signature_detection.py -v
```
