"""
Accuracy verification script for the Signature Matching Agent.

Runs every image pair through the /compare endpoint twice:
  1. With preprocessing  (preprocess=true)
  2. Without preprocessing (preprocess=false)

Then prints a side-by-side comparison table so you can evaluate
whether preprocessing improves, hurts, or has no effect on results.

Prerequisites:
    pip install httpx rich
    The backend must be running:  python backend.py

Usage:
    python verify_accuracy.py                    # uses default Data/ folder pairs
    python verify_accuracy.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import httpx

# ---------------------------------------------------------------------------
# Test-case definitions
# ---------------------------------------------------------------------------
# Each tuple: (image1_path, image2_path, expected_match, label)
# Set expected_match to None if you don't know the ground truth yet.

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")

TEST_CASES: list[tuple[str, str, bool | None, str]] = [
    # Same person – should match
    (os.path.join(DATA_DIR, "VR1.jpg"), os.path.join(DATA_DIR, "VR2.jpg"), True,  "VR1 vs VR2 (same person)"),
    # Different person – should not match
    (os.path.join(DATA_DIR, "VR1.jpg"), os.path.join(DATA_DIR, "VF1.jpg"), False, "VR1 vs VF1 (different)"),
    (os.path.join(DATA_DIR, "VR1.jpg"), os.path.join(DATA_DIR, "AR1.jpg"), False, "VR1 vs AR1 (different)"),
    (os.path.join(DATA_DIR, "VF1.jpg"), os.path.join(DATA_DIR, "AR1.jpg"), False, "VF1 vs AR1 (different)"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def call_compare(
    client: httpx.AsyncClient,
    url: str,
    img1_path: str,
    img2_path: str,
    preprocess: bool,
) -> dict:
    """Call the /compare endpoint and return the parsed JSON response."""
    with open(img1_path, "rb") as f1, open(img2_path, "rb") as f2:
        files = [
            ("image1", (os.path.basename(img1_path), f1, "image/jpeg")),
            ("image2", (os.path.basename(img2_path), f2, "image/jpeg")),
        ]
        resp = await client.post(
            f"{url}/compare",
            files=files,
            params={"preprocess": str(preprocess).lower()},
            timeout=120.0,
        )
    resp.raise_for_status()
    return resp.json()


def verdict_emoji(matched: bool, expected: bool | None) -> str:
    if expected is None:
        return "?"
    return "PASS" if matched == expected else "FAIL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(url: str):
    print(f"\nSignature Matching – Accuracy Verification")
    print(f"Endpoint : {url}/compare")
    print(f"Test cases: {len(TEST_CASES)}\n")
    print("=" * 120)

    results: list[dict] = []

    async with httpx.AsyncClient() as client:
        for img1, img2, expected, label in TEST_CASES:
            if not os.path.isfile(img1) or not os.path.isfile(img2):
                print(f"  SKIP  {label}  (file not found)")
                continue

            print(f"\n  {label}")
            print(f"  {'─' * 80}")

            row: dict = {"label": label, "expected": expected}

            for preprocess in [False, True]:
                tag = "preprocessed" if preprocess else "raw"
                start = time.perf_counter()
                try:
                    data = await call_compare(client, url, img1, img2, preprocess)
                    elapsed = (time.perf_counter() - start) * 1000
                    res = data["result"]
                    matched = res["signature_matched"]
                    score = res["confidence_score"]
                    v = verdict_emoji(matched, expected)
                    tokens = data.get("usage", {}).get("total_tokens", "?")

                    print(f"    [{tag:>13}]  matched={matched!s:<6}  confidence={score:.2f}  "
                          f"tokens={tokens}  time={elapsed:.0f}ms  {v}")

                    row[tag] = {
                        "matched": matched,
                        "confidence": score,
                        "reasoning": res["reasoning"][:120] + "…" if len(res["reasoning"]) > 120 else res["reasoning"],
                        "tokens": tokens,
                        "elapsed_ms": round(elapsed, 1),
                        "verdict": v,
                    }
                except Exception as exc:
                    print(f"    [{tag:>13}]  ERROR: {exc}")
                    row[tag] = {"error": str(exc)}

            results.append(row)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 120)
    print(f"\n{'SUMMARY':^120}\n")
    header = f"{'Test Case':<35} | {'Expected':^8} | {'Raw Match':^9} {'Raw Conf':^8} {'Raw':^4} | {'Pre Match':^9} {'Pre Conf':^8} {'Pre':^4}"
    print(header)
    print("─" * len(header))

    raw_correct = 0
    pre_correct = 0
    total = 0

    for r in results:
        exp = r["expected"]
        exp_str = str(exp) if exp is not None else "?"

        raw = r.get("raw", {})
        pre = r.get("preprocessed", {})

        raw_m = str(raw.get("matched", "ERR"))
        raw_c = f'{raw.get("confidence", 0):.2f}' if "confidence" in raw else " ERR "
        raw_v = raw.get("verdict", "?")

        pre_m = str(pre.get("matched", "ERR"))
        pre_c = f'{pre.get("confidence", 0):.2f}' if "confidence" in pre else " ERR "
        pre_v = pre.get("verdict", "?")

        print(f"{r['label']:<35} | {exp_str:^8} | {raw_m:^9} {raw_c:^8} {raw_v:^4} | {pre_m:^9} {pre_c:^8} {pre_v:^4}")

        if exp is not None:
            total += 1
            if raw.get("matched") == exp:
                raw_correct += 1
            if pre.get("matched") == exp:
                pre_correct += 1

    if total:
        print(f"\nAccuracy  →  Raw: {raw_correct}/{total} ({raw_correct/total*100:.0f}%)   "
              f"Preprocessed: {pre_correct}/{total} ({pre_correct/total*100:.0f}%)")

    # Save full results to JSON
    out_path = os.path.join(os.path.dirname(__file__), "verification_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify signature matching accuracy")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Backend URL")
    args = parser.parse_args()
    asyncio.run(run(args.url))
