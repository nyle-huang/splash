from __future__ import annotations

from unittest.mock import patch

import pytest

from product_campaign_pipeline.runpod_entrypoint import (
    GENERATION_MODE,
    PING_MODE,
    main,
    resolve_worker_mode,
)


def test_resolve_worker_mode_defaults_to_generation() -> None:
    assert resolve_worker_mode(None) == GENERATION_MODE
    assert resolve_worker_mode("") == GENERATION_MODE
    assert resolve_worker_mode(" production ") == GENERATION_MODE


def test_resolve_worker_mode_accepts_ping_aliases() -> None:
    assert resolve_worker_mode("ping") == PING_MODE
    assert resolve_worker_mode("diagnostic") == PING_MODE
    assert resolve_worker_mode("minimal_ping") == PING_MODE


def test_resolve_worker_mode_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported PCP_RUNPOD_WORKER_MODE"):
        resolve_worker_mode("unknown")


def test_main_starts_ping_worker_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PCP_RUNPOD_WORKER_MODE", "ping")

    with (
        patch("product_campaign_pipeline.runpod_entrypoint._start_ping_worker") as ping_start,
        patch("product_campaign_pipeline.runpod_entrypoint._start_generation_worker") as gen_start,
    ):
        main()

    ping_start.assert_called_once_with()
    gen_start.assert_not_called()


def test_main_starts_generation_worker_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PCP_RUNPOD_WORKER_MODE", raising=False)

    with (
        patch("product_campaign_pipeline.runpod_entrypoint._start_ping_worker") as ping_start,
        patch("product_campaign_pipeline.runpod_entrypoint._start_generation_worker") as gen_start,
    ):
        main()

    gen_start.assert_called_once_with()
    ping_start.assert_not_called()
