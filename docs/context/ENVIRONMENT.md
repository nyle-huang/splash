# Environment

## Machine Facts

- Platform: vast.ai VM
- Observed GPU: `NVIDIA GeForce RTX 3090 Ti`
- OS: Ubuntu 22.04.5 LTS
- Verified NVIDIA driver: `550.163.01`
- Verified CUDA runtime exposed by `nvidia-smi`: `12.4`
- Project root: `/workspace/product_campaign_pipeline`
- Raw CreativeRanking data root: `/workspace/data`
- Codex state root: `/root/.codex`
- Hugging Face token is available through `HF_TOKEN`
- Active Hugging Face cache root: `/workspace/.hf_home`

## Bootstrap Checklist

- Install or verify NVIDIA driver compatibility
- Install Python 3.12
- Create a dedicated project venv at `/workspace/product_campaign_pipeline/.venv`
- Install project dependencies from `pyproject.toml`
- Replace the default `torch` wheel with a driver-compatible `cu124` build on this host
- Verify `nvidia-smi` and torch CUDA access
- Verify Hugging Face authentication works before attempting gated model download
- Verify `from diffusers import Flux2KleinPipeline` succeeds in the project venv
- Plan for local FLUX runtime on a single 24 GB GPU with CPU offload or equivalent memory-saving configuration

## Migration Note

- The project was migrated from an older Google Compute Engine L4 VM.
- The copied project venv was not portable because its interpreter symlinks and shebangs still pointed to the old VM.
- The old Hugging Face model cache did not carry over in usable form.
- Historical experiment and review artifacts may still contain `/home/nyle_j_huang/...` paths; current execution should use `/workspace/...` paths.

## Verification Artifacts

- Driver version: `550.163.01`
- `nvidia-smi`: reports `NVIDIA GeForce RTX 3090 Ti`
- Project venv: Python `3.12.13`
- Torch smoke test: `torch=2.6.0+cu124`, `torch.cuda=12.4`, CUDA device `NVIDIA GeForce RTX 3090 Ti`
- CLI smoke test: `pcp --help` succeeds in the project venv
- Model import smoke test: `Flux2KleinPipeline`, `BlipProcessor`, `AutoModelForZeroShotObjectDetection`, `Sam2Processor`, and `Sam2Model` all import successfully
