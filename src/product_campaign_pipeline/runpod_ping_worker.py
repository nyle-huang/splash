"""Minimal Runpod Serverless worker for dispatch and logging diagnostics."""

from __future__ import annotations

import logging
import os
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

DEFAULT_WORKER_LOG_PATH = "/runpod-volume/logs/runpod_worker.log"
ENABLE_WORKER_FILE_LOG_ENV = "PCP_ENABLE_WORKER_FILE_LOG"

LOGGER = logging.getLogger(__name__)


def handle_ping_job(job: dict[str, Any]) -> dict[str, Any]:
    """Return immediately after proving that Runpod invoked the handler."""

    started_at = time.perf_counter()
    request_id = _request_id(job)
    input_keys = _input_keys(job)
    LOGGER.info(
        "Minimal Runpod ping handler entered. request_id=%s input_keys=%s",
        request_id,
        input_keys,
    )
    print(f"PCP_RUNPOD_PING_HANDLER_ENTERED request_id={request_id}", flush=True)

    payload = {
        "status": "succeeded",
        "request_id": request_id,
        "summary": "Minimal Runpod ping worker completed.",
        "worker_mode": "ping",
        "input_keys": input_keys,
        "versions": {
            "python": sys.version.split()[0],
            "runpod": _package_version("runpod"),
        },
        "paths": _diagnostic_paths(),
    }
    LOGGER.info(
        "Minimal Runpod ping handler completed. request_id=%s elapsed_seconds=%.3f",
        request_id,
        time.perf_counter() - started_at,
    )
    return payload


def _configure_ping_logging() -> None:
    print("PCP_RUNPOD_PING_LOGGING_CONFIGURE_START", flush=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    if not any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    ):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    if not _worker_file_logging_enabled():
        LOGGER.info("Minimal Runpod ping startup file logging disabled.")
        print("PCP_RUNPOD_PING_LOGGING_READY stream=stdout file=disabled", flush=True)
        return

    log_path = Path(os.getenv("PCP_WORKER_LOG_PATH", DEFAULT_WORKER_LOG_PATH))
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path
            for handler in root_logger.handlers
        ):
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        LOGGER.info("Minimal Runpod ping persistent log configured at %s", log_path)
    except OSError:
        LOGGER.exception("Failed to configure minimal Runpod ping log at %s", log_path)

    print(f"PCP_RUNPOD_PING_LOGGING_READY path={log_path}", flush=True)


def _worker_file_logging_enabled() -> bool:
    raw = os.getenv(ENABLE_WORKER_FILE_LOG_ENV)
    return raw is not None and raw.strip().lower() in {"1", "true", "yes", "on"}


def _diagnostic_paths() -> dict[str, Any]:
    log_path = Path(os.getenv("PCP_WORKER_LOG_PATH", DEFAULT_WORKER_LOG_PATH))
    return {
        "cwd": os.getcwd(),
        "hf_home": os.getenv("HF_HOME"),
        "output_root": os.getenv("PCP_OUTPUT_ROOT"),
        "worker_log_path": str(log_path),
        "worker_log_parent_exists": log_path.parent.exists(),
        "runpod_volume_exists": Path("/runpod-volume").exists(),
        "workspace_exists": Path("/workspace").exists(),
    }


def _input_keys(job: dict[str, Any]) -> list[str]:
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return []
    return sorted(str(key) for key in job_input)


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _request_id(job: dict[str, Any]) -> str:
    job_input = job.get("input")
    input_request_id = job_input.get("request_id") if isinstance(job_input, dict) else None
    for candidate in (job.get("id"), input_request_id, "runpod-ping"):
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return "runpod-ping"


def start_runpod_ping_worker() -> None:
    """Start the minimal worker in a real Runpod Serverless runtime."""

    _configure_ping_logging()
    LOGGER.info(
        "Starting minimal Runpod ping worker. python=%s runpod=%s",
        sys.version.split()[0],
        _package_version("runpod"),
    )
    print("PCP_RUNPOD_PING_WORKER_STARTING", flush=True)

    try:
        import runpod
    except ImportError as exc:  # pragma: no cover - deployment-only code path
        message = "Runpod serverless dependency is missing from the worker image."
        raise RuntimeError(message) from exc

    runpod.serverless.start({"handler": handle_ping_job})


if __name__ == "__main__":  # pragma: no cover - deployment-only entrypoint
    start_runpod_ping_worker()
