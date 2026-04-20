"""Azure Functions entrypoint for the public demo broker."""

from __future__ import annotations

import azure.functions as func

from product_campaign_pipeline.demo_broker import create_demo_broker_app

broker_app = create_demo_broker_app()
app = func.AsgiFunctionApp(app=broker_app, http_auth_level=func.AuthLevel.ANONYMOUS)
