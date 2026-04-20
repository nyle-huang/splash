#!/usr/bin/env python3
from __future__ import annotations

import argparse

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    uvicorn.run(
        "product_campaign_pipeline.service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
