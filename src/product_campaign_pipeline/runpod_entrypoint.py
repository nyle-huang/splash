"""Selectable Runpod Serverless entrypoint.

The default mode runs the production generation worker. Diagnostic mode runs a
minimal ping worker so Runpod dispatch/logging can be tested without importing
the image-generation stack.
"""

from __future__ import annotations

import os

GENERATION_MODE = "generation"
PING_MODE = "ping"


def resolve_worker_mode(raw_mode: str | None) -> str:
    """Normalize the Runpod worker mode configured by the template env."""

    mode = (raw_mode or GENERATION_MODE).strip().lower()
    if mode in {GENERATION_MODE, "prod", "production"}:
        return GENERATION_MODE
    if mode in {PING_MODE, "diagnostic", "diagnostics", "minimal_ping"}:
        return PING_MODE
    raise ValueError(
        "Unsupported PCP_RUNPOD_WORKER_MODE. "
        f"Expected '{GENERATION_MODE}' or '{PING_MODE}', got {raw_mode!r}."
    )


def _start_generation_worker() -> None:
    print("PCP Runpod entrypoint importing generation worker.", flush=True)
    from product_campaign_pipeline.runpod_worker import start_runpod_worker

    print("PCP Runpod entrypoint starting generation worker.", flush=True)
    start_runpod_worker()


def _start_ping_worker() -> None:
    print("PCP Runpod entrypoint importing ping worker.", flush=True)
    from product_campaign_pipeline.runpod_ping_worker import start_runpod_ping_worker

    print("PCP Runpod entrypoint starting ping worker.", flush=True)
    start_runpod_ping_worker()


def main() -> None:
    """Start the configured Runpod worker mode."""

    print("PCP Runpod entrypoint starting.", flush=True)
    mode = resolve_worker_mode(os.getenv("PCP_RUNPOD_WORKER_MODE"))
    print(f"PCP Runpod entrypoint selected worker mode: {mode}", flush=True)
    if mode == PING_MODE:
        _start_ping_worker()
        return
    _start_generation_worker()


if __name__ == "__main__":  # pragma: no cover - deployment-only entrypoint
    main()
