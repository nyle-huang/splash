from __future__ import annotations

from product_campaign_pipeline.runpod_ping_worker import handle_ping_job


def test_handle_ping_job_returns_dispatch_diagnostics(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PCP_WORKER_LOG_PATH", "/tmp/pcp-runpod-test/runpod_worker.log")
    monkeypatch.setenv("HF_HOME", "/tmp/pcp-runpod-test/hf_home")
    monkeypatch.setenv("PCP_OUTPUT_ROOT", "/tmp/pcp-runpod-test/runtime_outputs")

    payload = handle_ping_job(
        {
            "id": "ping-job",
            "input": {
                "_internal_ping": True,
                "request_id": "request-from-input",
            },
        }
    )

    assert payload["status"] == "succeeded"
    assert payload["request_id"] == "ping-job"
    assert payload["worker_mode"] == "ping"
    assert payload["input_keys"] == ["_internal_ping", "request_id"]
    assert payload["versions"]["python"]
    assert "runpod" in payload["versions"]
    assert payload["paths"]["worker_log_path"] == "/tmp/pcp-runpod-test/runpod_worker.log"
    assert payload["paths"]["hf_home"] == "/tmp/pcp-runpod-test/hf_home"
    assert payload["paths"]["output_root"] == "/tmp/pcp-runpod-test/runtime_outputs"
