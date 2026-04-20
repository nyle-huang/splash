#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"

KEEP_OUTPUT_DIRS = {
    # Latest approved broad checkpoint
    "generalization_diverse_v6_approved_bundle",
    # Key approved milestone bundles
    "generalization_diverse_v2_new_categories_final_bundle",
    "generalization_diverse_v2_targeted_rootfix_v30_curated_final_bundle",
    "generalization_diverse_v3_targeted_repair_v5_final_bundle",
    "generalization_diverse_v4_targeted_repair_v8_curated_final_bundle",
    "generalization_diverse_v4_approved_bundle",
    "human_review_batch_v33_final_bundle",
    # Current in-progress tranche state
    "generalization_diverse_v5_localization_v1",
    "generalization_diverse_v5_upstream_review_v3",
    "generalization_diverse_v5_final_bundle_v1",
}

REMOVE_TOPLEVEL_DIRS = {
    REPO_ROOT / ".pytest_cache",
    REPO_ROOT / ".tmp",
}


def _human_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{int(num)}B"


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.lstat().st_size
        except FileNotFoundError:
            return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.lstat().st_size
        except FileNotFoundError:
            continue
    return total


def _iter_pycache_dirs(root: Path) -> list[Path]:
    excluded_roots = {
        REPO_ROOT / ".venv",
    }
    results: list[Path] = []
    for path in root.rglob("__pycache__"):
        if not path.is_dir():
            continue
        if any(excluded in path.parents for excluded in excluded_roots):
            continue
        results.append(path)
    return sorted(results)


def _iter_removals() -> list[Path]:
    removals: list[Path] = []
    if OUTPUTS_DIR.exists():
        for path in sorted(OUTPUTS_DIR.iterdir()):
            if path.name in KEEP_OUTPUT_DIRS:
                continue
            removals.append(path)
    removals.extend(path for path in REMOVE_TOPLEVEL_DIRS if path.exists())
    removals.extend(_iter_pycache_dirs(REPO_ROOT))
    return removals


def _wait_for_pid(pid: int, poll_seconds: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.wait_for_pid > 0:
        _wait_for_pid(args.wait_for_pid, max(1, args.poll_seconds))

    removals = _iter_removals()
    size_by_path = [(path, _path_size(path)) for path in removals]
    total = sum(size for _, size in size_by_path)

    print("Cleanup plan:")
    for path, size in size_by_path:
        rel = path.relative_to(REPO_ROOT)
        print(f"- {rel} ({_human_bytes(size)})")
    print(f"Total removable size: {_human_bytes(total)}")

    if not args.execute:
        print("Dry run only. Re-run with --execute to delete.")
        return 0

    for path, _ in size_by_path:
        if not path.exists():
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
