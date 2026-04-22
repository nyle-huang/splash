"""Local FLUX.2 Klein client with lazy diffusers loading."""

from __future__ import annotations

import gc
import io
import os
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from product_campaign_pipeline.composer import FluxPrompt

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
DEFAULT_RUNPOD_CACHED_MODEL_ROOT = "/runpod-volume/huggingface-cache/hub"
MIN_REFERENCE_IMAGE_SIDE = 64


class MissingDependencyError(RuntimeError):
    """Raised when optional local generation dependencies are unavailable."""


class MissingModelAccessError(RuntimeError):
    """Raised when the local FLUX model cannot be accessed or loaded."""


class MissingCredentialsError(MissingModelAccessError):
    """Backward-compatible alias for gated model access failures."""


@dataclass(frozen=True)
class FluxGenerationRequest:
    """Prepared local generation request metadata."""

    model_id: str
    prompt: str
    prompt_payload: Mapping[str, Any] = field(default_factory=dict)
    input_images: tuple[str, ...] = ()
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    output_format: str = "png"
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    max_sequence_length: int = 256
    device: str = "cuda"
    dtype: str = "bfloat16"
    cpu_offload: bool = True
    sequential_cpu_offload: bool = False
    attention_slicing: bool = True
    output_path: str | None = None


@dataclass(frozen=True)
class FluxGenerationResult:
    """Local generation result metadata."""

    model_id: str
    output_path: str
    width: int
    height: int
    seed: int | None = None
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    elapsed_seconds: float = 0.0
    input_images: tuple[str, ...] = ()


class Flux2KleinClient:
    """Local generation boundary for `black-forest-labs/FLUX.2-klein-9B`."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: str = "bfloat16",
        cpu_offload: bool = True,
        sequential_cpu_offload: bool = False,
        attention_slicing: bool = True,
        model_load_path: str | os.PathLike[str] | None = None,
        cached_model_root: str | os.PathLike[str] | None = None,
        pipeline_factory: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.cpu_offload = bool(cpu_offload)
        self.sequential_cpu_offload = bool(sequential_cpu_offload)
        self.attention_slicing = bool(attention_slicing)
        self.model_load_path = None if model_load_path is None else os.fspath(model_load_path)
        self.cached_model_root = _resolve_cached_model_root(cached_model_root)
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any | None = None
        self._loaded_signature: tuple[str, str, str, str, bool, bool, bool] | None = None

    def build_request(
        self,
        *,
        prompt: FluxPrompt | str | Mapping[str, Any],
        input_image: str | os.PathLike[str] | None = None,
        reference_images: Sequence[str | os.PathLike[str]] = (),
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        output_format: str = "png",
        guidance_scale: float = 1.0,
        num_inference_steps: int = 4,
        max_sequence_length: int = 256,
        device: str | None = None,
        dtype: str | None = None,
        cpu_offload: bool | None = None,
        sequential_cpu_offload: bool | None = None,
        attention_slicing: bool | None = None,
        output_path: str | os.PathLike[str] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> FluxGenerationRequest:
        normalized_prompt, prompt_payload = self._normalize_prompt(prompt)
        normalized_images: list[str] = []

        if input_image is not None:
            normalized_images.append(self._normalize_image_value(input_image))
        for reference_image in reference_images:
            normalized_images.append(self._normalize_image_value(reference_image))

        payload: dict[str, Any] = dict(prompt_payload)
        if extra:
            payload.update(dict(extra))

        return FluxGenerationRequest(
            model_id=self.model_id,
            prompt=normalized_prompt,
            prompt_payload=payload,
            input_images=tuple(normalized_images),
            width=width,
            height=height,
            seed=seed,
            output_format=output_format,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            max_sequence_length=int(max_sequence_length),
            device=device or self.device,
            dtype=dtype or self.dtype,
            cpu_offload=self.cpu_offload if cpu_offload is None else bool(cpu_offload),
            sequential_cpu_offload=(
                self.sequential_cpu_offload
                if sequential_cpu_offload is None
                else bool(sequential_cpu_offload)
            ),
            attention_slicing=(
                self.attention_slicing if attention_slicing is None else bool(attention_slicing)
            ),
            output_path=None if output_path is None else os.fspath(output_path),
        )

    def generate(self, request: FluxGenerationRequest) -> FluxGenerationResult:
        pipe, torch = self._ensure_pipeline(request)
        device = self._resolve_device(torch, request.device)
        generator = None
        if request.seed is not None:
            generator_device = "cuda" if str(device).startswith("cuda") else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(int(request.seed))

        images = self._load_reference_images(request.input_images)
        call_kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "max_sequence_length": request.max_sequence_length,
            "output_type": "pil",
        }
        if request.height is not None:
            call_kwargs["height"] = int(request.height)
        if request.width is not None:
            call_kwargs["width"] = int(request.width)
        if generator is not None:
            call_kwargs["generator"] = generator
        if images:
            call_kwargs["image"] = images[0] if len(images) == 1 else images

        started_at = time.perf_counter()
        output = pipe(**call_kwargs)
        image = output.images[0]
        output_path = (
            Path(request.output_path) if request.output_path else self._default_output_path(request)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format=_normalize_output_format(request.output_format))
        elapsed_seconds = time.perf_counter() - started_at

        return FluxGenerationResult(
            model_id=request.model_id,
            output_path=str(output_path),
            width=int(image.width),
            height=int(image.height),
            seed=request.seed,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            elapsed_seconds=elapsed_seconds,
            input_images=request.input_images,
        )

    def reset_pipeline(self) -> None:
        pipe = self._pipeline
        self._pipeline = None
        self._loaded_signature = None
        if pipe is not None:
            del pipe
        gc.collect()
        try:
            torch = self._import_torch()
        except MissingDependencyError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def warmup(
        self,
        *,
        device: str | None = None,
        dtype: str | None = None,
        cpu_offload: bool | None = None,
        sequential_cpu_offload: bool | None = None,
        attention_slicing: bool | None = None,
    ) -> None:
        request = FluxGenerationRequest(
            model_id=self.model_id,
            prompt="",
            device=device or self.device,
            dtype=dtype or self.dtype,
            cpu_offload=self.cpu_offload if cpu_offload is None else bool(cpu_offload),
            sequential_cpu_offload=(
                self.sequential_cpu_offload
                if sequential_cpu_offload is None
                else bool(sequential_cpu_offload)
            ),
            attention_slicing=(
                self.attention_slicing if attention_slicing is None else bool(attention_slicing)
            ),
        )
        self._ensure_pipeline(request)

    @property
    def pipeline_loaded(self) -> bool:
        return self._pipeline is not None

    def _ensure_pipeline(self, request: FluxGenerationRequest) -> tuple[Any, Any]:
        model_load_source = self._resolve_model_load_source(request.model_id)
        signature = (
            request.model_id,
            model_load_source,
            request.device,
            request.dtype,
            request.cpu_offload,
            request.sequential_cpu_offload,
            request.attention_slicing,
        )
        if self._pipeline is not None and self._loaded_signature == signature:
            torch = self._import_torch()
            return self._pipeline, torch

        torch = self._import_torch()
        pipeline_factory = self._pipeline_factory or self._import_pipeline_factory()
        torch_dtype = self._resolve_dtype(torch, request.dtype)

        try:
            pipe = pipeline_factory.from_pretrained(
                model_load_source,
                torch_dtype=torch_dtype,
            )
        except Exception as exc:  # pragma: no cover - exercised only in live model loading
            raise MissingCredentialsError(
                "Unable to load the gated FLUX.2 Klein model locally. "
                "Confirm `hf auth whoami` succeeds on this VM and that the accepted token "
                f"has access to `{request.model_id}`. "
                f"Resolved load source: `{model_load_source}`."
            ) from exc

        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        if request.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        device = self._resolve_device(torch, request.device)
        if str(device).startswith("cuda"):
            if request.sequential_cpu_offload and hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload()
            elif request.cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)
        else:
            pipe.to(device)

        self._pipeline = pipe
        self._loaded_signature = signature
        return pipe, torch

    def _resolve_model_load_source(self, model_id: str) -> str:
        if self.model_load_path:
            path = Path(self.model_load_path).expanduser()
            if not _is_diffusers_snapshot(path):
                raise MissingModelAccessError(
                    "Configured model load path is not a diffusers snapshot with "
                    f"`model_index.json`: {path}"
                )
            return str(path.resolve())

        cached_snapshot = _find_huggingface_snapshot(
            model_id=model_id,
            cache_root=self.cached_model_root,
        )
        if cached_snapshot is not None:
            return str(cached_snapshot)
        return model_id

    def _import_torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise MissingDependencyError(
                "PyTorch is required for local FLUX.2 Klein generation."
            ) from exc
        return torch

    def _import_pipeline_factory(self) -> Any:
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise MissingDependencyError(
                "Diffusers with `Flux2KleinPipeline` support is required for local generation."
            ) from exc
        return Flux2KleinPipeline

    def _resolve_dtype(self, torch: Any, dtype_name: str) -> Any:
        normalized = dtype_name.strip().lower()
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported torch dtype for local generation: {dtype_name}")
        return mapping[normalized]

    def _resolve_device(self, torch: Any, requested_device: str) -> str:
        candidate = requested_device.strip().lower()
        if candidate == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if candidate.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for local FLUX generation, but torch.cuda.is_available() "
                "is false."
            )
        return candidate

    def _normalize_prompt(
        self,
        prompt: FluxPrompt | str | Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if isinstance(prompt, FluxPrompt):
            payload = prompt.as_dict()
            return prompt.to_bfl_prompt(), payload
        if isinstance(prompt, Mapping):
            normalized = FluxPrompt.from_mapping(prompt)
            return normalized.to_bfl_prompt(), normalized.as_dict()
        text = str(prompt)
        return text, {"prompt_text": text}

    def _normalize_image_value(self, value: str | os.PathLike[str]) -> str:
        candidate = os.fspath(value)
        parsed = urllib.parse.urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            return candidate
        path = Path(candidate).expanduser()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Reference image not found: {candidate}")
        return str(path.resolve())

    def _load_reference_images(self, values: Sequence[str]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for value in values:
            parsed = urllib.parse.urlparse(value)
            if parsed.scheme in {"http", "https"}:
                with urllib.request.urlopen(value) as response:  # pragma: no cover - networked path
                    image_bytes = response.read()
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = Image.open(value)
            images.append(self._normalize_reference_image(image.convert("RGB")))
        return images

    def _normalize_reference_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width >= MIN_REFERENCE_IMAGE_SIDE and height >= MIN_REFERENCE_IMAGE_SIDE:
            return image
        scale = max(
            MIN_REFERENCE_IMAGE_SIDE / float(max(width, 1)),
            MIN_REFERENCE_IMAGE_SIDE / float(max(height, 1)),
        )
        resized_width = max(MIN_REFERENCE_IMAGE_SIDE, int(round(width * scale)))
        resized_height = max(MIN_REFERENCE_IMAGE_SIDE, int(round(height * scale)))
        return image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

    def _default_output_path(self, request: FluxGenerationRequest) -> Path:
        prompt_source = request.prompt_payload.get("subject") or request.prompt
        slug = _slugify(str(prompt_source)) or "flux-klein"
        stem = slug[:64]
        seed_suffix = "auto" if request.seed is None else str(request.seed)
        extension = request.output_format.lower().replace("jpeg", "jpg")
        return Path.cwd() / "outputs" / "generated" / f"{stem}-{seed_suffix}.{extension}"


def _normalize_output_format(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "jpg":
        normalized = "jpeg"
    return normalized.upper()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _resolve_cached_model_root(
    cached_model_root: str | os.PathLike[str] | None,
) -> Path | None:
    if cached_model_root is not None:
        value = os.fspath(cached_model_root).strip()
    else:
        value = os.getenv("PCP_RUNPOD_CACHED_MODEL_ROOT", DEFAULT_RUNPOD_CACHED_MODEL_ROOT).strip()
    if not value:
        return None
    return Path(value).expanduser()


def _find_huggingface_snapshot(*, model_id: str, cache_root: Path | None) -> Path | None:
    if cache_root is None or not cache_root.exists() or not cache_root.is_dir():
        return None
    for model_dir in _candidate_huggingface_model_dirs(model_id=model_id, cache_root=cache_root):
        snapshot = _find_huggingface_snapshot_in_model_dir(model_dir)
        if snapshot is not None:
            return snapshot
    return None


def _candidate_huggingface_model_dirs(*, model_id: str, cache_root: Path) -> list[Path]:
    model_dir_name = "models--" + _normalize_huggingface_model_id(model_id).replace("/", "--")
    seen: set[Path] = set()
    candidates: list[Path] = []
    try:
        children = list(cache_root.iterdir())
    except OSError:
        children = []
    for path in children:
        if path in seen or not path.is_dir():
            continue
        if path.name.lower() != model_dir_name.lower():
            continue
        seen.add(path)
        candidates.append(path)
    for name in (model_dir_name, model_dir_name.lower()):
        path = cache_root / name
        if path in seen or not path.exists() or not path.is_dir():
            continue
        seen.add(path)
        candidates.append(path)
    return candidates


def _normalize_huggingface_model_id(model_id: str) -> str:
    value = model_id.strip()
    if not value:
        return value
    candidate = value if "://" in value else f"https://{value}"
    parsed = urllib.parse.urlparse(candidate)
    if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        return value.strip("/")

    parts = [part for part in parsed.path.split("/") if part]
    if parts[:1] == ["models"]:
        parts = parts[1:]
    if len(parts) < 2:
        return value.strip("/")
    repo_parts = parts[:2]
    if ":" in repo_parts[1]:
        repo_parts[1] = repo_parts[1].split(":", 1)[0]
    return "/".join(repo_parts)


def _find_huggingface_snapshot_in_model_dir(model_dir: Path) -> Path | None:
    ref_path = model_dir / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot = model_dir / "snapshots" / revision
        if _is_diffusers_snapshot(snapshot):
            return snapshot.resolve()

    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        return None
    candidates = [
        path
        for path in snapshots_dir.iterdir()
        if path.is_dir() and _is_diffusers_snapshot(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _is_diffusers_snapshot(path: Path) -> bool:
    return path.exists() and path.is_dir() and (path / "model_index.json").exists()


# Backward-compatible aliases for the original BFL-named client boundary.
BFLFluxClient = Flux2KleinClient
BFLGenerationRequest = FluxGenerationRequest
BFLSubmittedTask = FluxGenerationResult
BFLPollResult = FluxGenerationResult
