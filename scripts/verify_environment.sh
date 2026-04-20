#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

log() {
  printf '[verify] %s\n' "$*"
}

if [[ -d "${VENV_DIR}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi is missing from PATH."
  exit 1
fi

log "Running nvidia-smi."
nvidia-smi

log "Running Python import and CUDA smoke tests."
python - <<'PY'
import importlib
import sys

required_modules = [
    "numpy",
    "PIL",
    "pydantic",
    "product_campaign_pipeline",
    "structlog",
    "typer",
]

for module_name in required_modules:
    importlib.import_module(module_name)
    print(f"[verify] imported {module_name}")

try:
    import torch
except ImportError as exc:
    raise SystemExit(f"[verify] torch import failed: {exc}") from exc

print(f"[verify] torch={torch.__version__}")
print(f"[verify] torch.cuda={torch.version.cuda}")

if not torch.cuda.is_available():
    raise SystemExit("[verify] torch reports CUDA unavailable.")

device_name = torch.cuda.get_device_name(0)
tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
result = (tensor * 2).cpu().tolist()

print(f"[verify] cuda_device={device_name}")
print(f"[verify] cuda_result={result}")
print(f"[verify] python={sys.version.split()[0]}")
PY

log "Environment verification complete."
