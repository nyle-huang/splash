"""Azure Functions entrypoint for the public demo broker."""

from __future__ import annotations

import sys
from pathlib import Path

import azure.functions as func

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from product_campaign_pipeline.demo_broker import create_demo_broker_app  # noqa: E402

broker_app = create_demo_broker_app()
app = func.AsgiFunctionApp(app=broker_app, http_auth_level=func.AuthLevel.ANONYMOUS)
