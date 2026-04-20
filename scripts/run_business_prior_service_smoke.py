#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", required=True)
    parser.add_argument("--product-title", required=True)
    parser.add_argument("--hint-phrase", action="append", default=[])
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"image does not exist: {image_path}")

    health = httpx.get(f"{args.base_url.rstrip('/')}/healthz", timeout=30.0)
    health.raise_for_status()

    with image_path.open("rb") as handle:
        response = httpx.post(
            f"{args.base_url.rstrip('/')}/generate/business-prior",
            data={
                "product_title": args.product_title,
                "hint_phrases": "|".join(args.hint_phrase),
            },
            files={"image": (image_path.name, handle, "image/png")},
            timeout=1800.0,
        )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(json.dumps(payload, indent=2, ensure_ascii=True))
    output_path = Path(str(payload.get("output_path", "")))
    if not output_path.exists():
        raise FileNotFoundError(f"service returned missing output path: {output_path}")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
