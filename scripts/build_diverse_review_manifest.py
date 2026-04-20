#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "data" / "generalization_diverse_v1" / "source_selection.json"
DEFAULT_EXISTING_MANIFEST = REPO_ROOT / "data" / "human_review_seed" / "review_seed_manifest.json"
DEFAULT_OUTPUT_MANIFEST = REPO_ROOT / "data" / "generalization_diverse_v1" / "review_seed_manifest.json"
DEFAULT_OUTPUT_CATALOG = REPO_ROOT / "data" / "generalization_diverse_v1" / "source_catalog.json"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _normalize_image_url(url: str) -> str:
    return str(url).split("?")[0].strip()


def _mirror_url(source_page_url: str) -> str:
    return "https://r.jina.ai/http://" + source_page_url


def _parse_title(page_markdown: str) -> str:
    match = re.search(r"^Title:\s*(.+)$", page_markdown, re.M)
    if not match:
        raise ValueError("Unable to parse page title from mirrored page.")
    title = match.group(1).strip()
    if title.endswith(" - Walmart.com"):
        title = title[: -len(" - Walmart.com")].strip()
    return title


def _extract_customer_photo_urls(page_markdown: str) -> list[str]:
    block_patterns = (
        re.compile(r"Customer photos\s+(.*?)Filtered and sorted results", re.S),
        re.compile(r"Customer images\s+(.*?)Filtered and sorted results", re.S),
        re.compile(r"Customer photos\s+(.*?)View all reviews", re.S),
        re.compile(r"Customer images\s+(.*?)View all reviews", re.S),
    )
    image_pattern = re.compile(r"https://i5\.walmartimages\.com/[^)\s]+")
    for pattern in block_patterns:
        match = pattern.search(page_markdown)
        if not match:
            continue
        urls = [_normalize_image_url(url) for url in image_pattern.findall(match.group(1))]
        deduped: list[str] = []
        seen: set[str] = set()
        for url in urls:
            if url in seen:
                continue
            deduped.append(url)
            seen.add(url)
        if deduped:
            return deduped
    raise ValueError("Unable to extract customer photo URLs from mirrored page.")


def _load_existing_manifest(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in rows}


def _load_cached_customer_photo_snapshot(*, slug: str, source_page_url: str) -> dict[str, object] | None:
    for catalog_path in sorted((REPO_ROOT / "data").glob("generalization_diverse_v*/source_catalog.json")):
        try:
            rows = json.loads(catalog_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("slug", "")) != slug:
                continue
            if str(row.get("source_page_url", "")) != source_page_url:
                continue
            customer_photo_urls = [str(url) for url in row.get("customer_photo_urls", []) if str(url).strip()]
            if not customer_photo_urls:
                continue
            return {
                "title": str(row.get("product_title", "")).strip() or slug,
                "customer_photo_urls": customer_photo_urls,
                "source_catalog_path": str(catalog_path),
            }
    return None


def _fetch_page_snapshot(source_page_url: str) -> dict[str, object]:
    response = requests.get(_mirror_url(source_page_url), timeout=60, headers=REQUEST_HEADERS)
    response.raise_for_status()
    page_markdown = response.text
    return {
        "title": _parse_title(page_markdown),
        "customer_photo_urls": _extract_customer_photo_urls(page_markdown),
    }


def build_manifest(
    *,
    config_path: Path,
    existing_manifest_path: Path,
    output_manifest_path: Path,
    output_catalog_path: Path,
) -> tuple[Path, Path]:
    config_path = config_path if config_path.is_absolute() else (REPO_ROOT / config_path)
    existing_manifest_path = (
        existing_manifest_path if existing_manifest_path.is_absolute() else (REPO_ROOT / existing_manifest_path)
    )
    output_manifest_path = (
        output_manifest_path if output_manifest_path.is_absolute() else (REPO_ROOT / output_manifest_path)
    )
    output_catalog_path = output_catalog_path if output_catalog_path.is_absolute() else (REPO_ROOT / output_catalog_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    existing_manifest = _load_existing_manifest(existing_manifest_path)
    capture_date = str(config["capture_date"])
    try:
        manifest_dir = output_manifest_path.parent.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("Output manifest path must live under the repository root.") from exc
    local_image_prefix = manifest_dir / "images"

    manifest_rows: list[dict[str, object]] = []
    catalog_rows: list[dict[str, object]] = []

    for control in config.get("controls", []):
        source_seed_id = str(control["source_seed_id"])
        source = existing_manifest[source_seed_id]
        output_id = str(control["id"])
        row = {
            "id": output_id,
            "platform": str(source["platform"]),
            "source_page_url": str(source["source_page_url"]),
            "source_image_url": str(source["source_image_url"]),
            "product_title": str(source["product_title"]),
            "hint_phrases": list(source.get("hint_phrases", [])),
            "capture_date": capture_date,
            "local_image_path": str(local_image_prefix / f"{output_id}.jpg"),
            "bucket": str(control.get("bucket", "control")),
            "stress_tags": list(control.get("stress_tags", [])),
            "source_seed_id": source_seed_id,
        }
        manifest_rows.append(row)
        catalog_rows.append(
            {
                "id": output_id,
                "source_type": "control",
                "source_seed_id": source_seed_id,
                "source_page_url": str(source["source_page_url"]),
                "source_image_url": str(source["source_image_url"]),
                "product_title": str(source["product_title"]),
                "bucket": row["bucket"],
                "stress_tags": row["stress_tags"],
            }
        )

    for product in config.get("products", []):
        source_page_url = str(product["source_page_url"])
        slug = str(product["slug"])
        snapshot = _load_cached_customer_photo_snapshot(slug=slug, source_page_url=source_page_url)
        if snapshot is None:
            try:
                snapshot = _fetch_page_snapshot(source_page_url)
            except Exception as exc:
                raise ValueError(f"Failed to fetch customer photos for {slug}: {source_page_url}") from exc
        title = str(snapshot["title"])
        customer_photo_urls = list(snapshot["customer_photo_urls"])
        bucket = str(product.get("bucket", "product"))
        hint_phrases = [str(item) for item in product.get("hint_phrases", [])]
        selected_photos = product.get("selected_photos", [])
        catalog_rows.append(
            {
                "slug": slug,
                "source_type": "mirrored_customer_photos",
                "source_page_url": source_page_url,
                "product_title": title,
                "bucket": bucket,
                "hint_phrases": hint_phrases,
                "customer_photo_urls": customer_photo_urls,
            }
        )
        for selection in selected_photos:
            index = int(selection["index"])
            if index < 1 or index > len(customer_photo_urls):
                raise IndexError(
                    f"Photo index {index} is out of range for {slug}; only {len(customer_photo_urls)} customer photos found."
                )
            output_id = str(selection["id"])
            manifest_rows.append(
                {
                    "id": output_id,
                    "platform": "walmart",
                    "source_page_url": source_page_url,
                    "source_image_url": customer_photo_urls[index - 1],
                    "product_title": title,
                    "hint_phrases": hint_phrases,
                    "capture_date": capture_date,
                    "local_image_path": str(local_image_prefix / f"{output_id}.jpg"),
                    "bucket": bucket,
                    "stress_tags": list(selection.get("stress_tags", [])),
                    "source_slug": slug,
                    "source_photo_index": index,
                }
            )

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    output_catalog_path.write_text(json.dumps(catalog_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    return output_manifest_path, output_catalog_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--existing-manifest", default=str(DEFAULT_EXISTING_MANIFEST))
    parser.add_argument("--output-manifest", default=str(DEFAULT_OUTPUT_MANIFEST))
    parser.add_argument("--output-catalog", default=str(DEFAULT_OUTPUT_CATALOG))
    args = parser.parse_args()

    manifest_path, catalog_path = build_manifest(
        config_path=Path(args.config),
        existing_manifest_path=Path(args.existing_manifest),
        output_manifest_path=Path(args.output_manifest),
        output_catalog_path=Path(args.output_catalog),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "catalog_path": str(catalog_path),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
