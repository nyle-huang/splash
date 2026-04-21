"""Command-line surface for planning, prompt composition, local FLUX runs, and evaluation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from product_campaign_pipeline.composer import BaselineComposer, BusinessPriorComposer, FluxPrompt, ProductBrief
from product_campaign_pipeline.evaluation import PromptRubric
from product_campaign_pipeline.flux import (
    DEFAULT_MODEL_ID,
    Flux2KleinClient,
    MissingCredentialsError,
    MissingDependencyError,
    MissingModelAccessError,
)
from product_campaign_pipeline.localization import (
    DEFAULT_CAPTION_MODEL_ID,
    DEFAULT_GROUNDING_MODEL_ID,
    DEFAULT_SAM2_MODEL_ID,
    ProductLocalizationPipeline,
    ProductPhoto,
    build_model_backed_localization_pipeline,
    save_localization_artifacts,
    select_primary_mask,
)
from product_campaign_pipeline.planner import CTRAwareRetrievalPlanner, PlannerInput, load_creative_ranking_manifest
from product_campaign_pipeline.production import (
    BusinessPriorInferenceRequest,
    run_business_prior_inference,
)


def app(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        payload = args.handler(args)
    except (MissingCredentialsError, MissingDependencyError, MissingModelAccessError) as exc:
        _print_json({"generated": False, "error": str(exc)})
        return 2
    except FileNotFoundError as exc:
        _print_json({"error": str(exc)})
        return 2
    except ValueError as exc:
        _print_json({"error": str(exc)})
        return 2
    _print_json(payload)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pcp")
    subparsers = parser.add_subparsers(dest="command")

    planner_parser = subparsers.add_parser("planner", help="Inspect CTR-aware retrieval results.")
    planner_subparsers = planner_parser.add_subparsers(dest="planner_command")
    planner_retrieve = planner_subparsers.add_parser("retrieve", help="Retrieve high-CTR creative priors.")
    _add_product_arguments(planner_retrieve)
    planner_retrieve.add_argument("--manifest", required=True, help="Path to CreativeRanking TSV manifest.")
    planner_retrieve.add_argument("--creative-metadata", help="Optional JSON file with hint and embedding metadata.")
    planner_retrieve.add_argument("--query-embedding", help="Comma-separated embedding for retrieval.")
    planner_retrieve.add_argument("--top-k", type=int, default=3)
    planner_retrieve.set_defaults(handler=_handle_planner_retrieve)

    localize_parser = subparsers.add_parser("localize", help="Localize the featured product in a source photo.")
    localize_parser.add_argument("--image", required=True, help="Path to the source product photo.")
    localize_parser.add_argument("--product-id")
    localize_parser.add_argument("--title")
    localize_parser.add_argument("--hint-phrase", action="append", default=[])
    localize_parser.add_argument("--backend", choices=("placeholder", "model"), default="placeholder")
    localize_parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL_ID)
    localize_parser.add_argument("--grounding-model", default=DEFAULT_GROUNDING_MODEL_ID)
    localize_parser.add_argument("--sam2-model", default=DEFAULT_SAM2_MODEL_ID)
    localize_parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    localize_parser.add_argument("--output-dir", help="Optional directory to save crop, mask, and overlay artifacts.")
    localize_parser.set_defaults(handler=_handle_localize_photo)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Compose prompts and optionally run local FLUX.2 Klein generation.",
    )
    generate_subparsers = generate_parser.add_subparsers(dest="generate_command")
    generate_campaign = generate_subparsers.add_parser(
        "campaign",
        help="Run baseline or business-prior prompt generation.",
    )
    _add_product_arguments(generate_campaign)
    generate_campaign.add_argument(
        "--line",
        choices=("baseline", "business-prior"),
        default="baseline",
    )
    generate_campaign.add_argument("--manifest", help="CreativeRanking TSV manifest for business-prior retrieval.")
    generate_campaign.add_argument("--creative-metadata", help="Optional JSON hints/embeddings file.")
    generate_campaign.add_argument("--query-embedding", help="Comma-separated embedding for retrieval.")
    generate_campaign.add_argument("--top-k", type=int, default=3)
    generate_campaign.add_argument("--input-image", help="Base image path or URL for local FLUX editing.")
    generate_campaign.add_argument(
        "--reference-image",
        action="append",
        default=[],
        help="Additional reference image path or URL.",
    )
    generate_campaign.add_argument("--model", default=DEFAULT_MODEL_ID)
    generate_campaign.add_argument("--run", action="store_true", help="Run local FLUX generation instead of only printing the request preview.")
    generate_campaign.add_argument("--output", help="Optional output image path for local generation.")
    generate_campaign.add_argument("--width", type=int)
    generate_campaign.add_argument("--height", type=int)
    generate_campaign.add_argument("--seed", type=int)
    generate_campaign.add_argument("--guidance-scale", type=float, default=1.0)
    generate_campaign.add_argument("--num-inference-steps", type=int, default=4)
    generate_campaign.add_argument("--max-sequence-length", type=int, default=512)
    generate_campaign.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    generate_campaign.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "bf16", "float16", "fp16", "float32", "fp32"))
    generate_campaign.add_argument("--cpu-offload", dest="cpu_offload", action="store_true", default=True)
    generate_campaign.add_argument("--no-cpu-offload", dest="cpu_offload", action="store_false")
    generate_campaign.add_argument("--sequential-cpu-offload", action="store_true")
    generate_campaign.add_argument("--attention-slicing", dest="attention_slicing", action="store_true", default=True)
    generate_campaign.add_argument("--no-attention-slicing", dest="attention_slicing", action="store_false")
    generate_campaign.set_defaults(handler=_handle_generate_campaign)

    generate_business_prior_photo = generate_subparsers.add_parser(
        "business-prior-photo",
        help="Run the full single-request business-prior image pipeline on one source photo.",
    )
    generate_business_prior_photo.add_argument("--image", required=True, help="Path to the uploaded source photo.")
    generate_business_prior_photo.add_argument("--product-title", required=True)
    generate_business_prior_photo.add_argument("--retrieval-index", required=True, help="Path to the retrieval index JSON.")
    generate_business_prior_photo.add_argument("--output-dir", required=True, help="Directory where request artifacts and final output will be written.")
    generate_business_prior_photo.add_argument("--hint-phrase", action="append", default=[])
    generate_business_prior_photo.add_argument("--request-id")
    generate_business_prior_photo.add_argument("--product-id")
    generate_business_prior_photo.add_argument("--source-page-url", default="uploaded://local")
    generate_business_prior_photo.add_argument("--source-image-url", default="uploaded://local")
    generate_business_prior_photo.add_argument("--model", default=DEFAULT_MODEL_ID)
    generate_business_prior_photo.add_argument("--width", type=int, default=512)
    generate_business_prior_photo.add_argument("--height", type=int, default=512)
    generate_business_prior_photo.add_argument("--num-inference-steps", type=int, default=4)
    generate_business_prior_photo.add_argument("--guidance-scale", type=float, default=1.0)
    generate_business_prior_photo.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    generate_business_prior_photo.add_argument("--analysis-device", default="cpu", choices=("cuda", "cpu", "auto"))
    generate_business_prior_photo.add_argument("--localization-device", default="cuda", choices=("cuda", "cpu", "auto"))
    generate_business_prior_photo.add_argument("--candidate-mode", action="append", dest="candidate_modes", default=[])
    generate_business_prior_photo.add_argument("--skip-analysis", action="store_true")
    generate_business_prior_photo.add_argument("--cpu-offload", dest="cpu_offload", action="store_true", default=True)
    generate_business_prior_photo.add_argument("--no-cpu-offload", dest="cpu_offload", action="store_false")
    generate_business_prior_photo.add_argument("--sequential-cpu-offload", action="store_true")
    generate_business_prior_photo.add_argument("--attention-slicing", dest="attention_slicing", action="store_true", default=True)
    generate_business_prior_photo.add_argument("--no-attention-slicing", dest="attention_slicing", action="store_false")
    generate_business_prior_photo.add_argument("--top-k", type=int, default=5)
    generate_business_prior_photo.add_argument("--seed", type=int)
    generate_business_prior_photo.set_defaults(handler=_handle_generate_business_prior_photo)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate prompt quality.")
    evaluate_subparsers = evaluate_parser.add_subparsers(dest="evaluate_command")
    evaluate_prompt = evaluate_subparsers.add_parser("prompt", help="Run the prompt rubric.")
    evaluate_prompt.add_argument("--prompt-file", required=True, help="Path to a prompt JSON file.")
    evaluate_prompt.add_argument("--pass-threshold", type=float, default=0.7)
    evaluate_prompt.set_defaults(handler=_handle_evaluate_prompt)

    return parser


def _add_product_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--product-name", required=True)
    parser.add_argument("--category")
    parser.add_argument("--description")
    parser.add_argument("--attribute", action="append", default=[])
    parser.add_argument("--preserve", action="append", default=[])
    parser.add_argument("--target-audience")
    parser.add_argument("--campaign-goal")
    parser.add_argument("--brand-style")


def _handle_planner_retrieve(args: argparse.Namespace) -> dict[str, Any]:
    plan = _build_retrieval_plan(args)
    return _serialize(plan)


def _handle_localize_photo(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "model":
        pipeline = build_model_backed_localization_pipeline(
            device=args.device,
            caption_model_id=args.caption_model,
            grounding_model_id=args.grounding_model,
            sam2_model_id=args.sam2_model,
        )
    else:
        pipeline = ProductLocalizationPipeline()

    photo = ProductPhoto(
        image_path=Path(args.image),
        product_id=args.product_id,
        title=args.title,
        hint_phrases=tuple(args.hint_phrase),
    )
    result = pipeline.localize(photo)
    payload: dict[str, Any] = {
        "backend": args.backend,
        "result": _serialize(result),
    }

    selected = select_primary_mask(result)
    if selected is not None:
        payload["selected_mask"] = _serialize(selected)
    if args.output_dir:
        artifacts = save_localization_artifacts(result, args.output_dir, selected_mask=selected)
        payload["artifacts"] = _serialize(artifacts)
    return payload


def _handle_generate_campaign(args: argparse.Namespace) -> dict[str, Any]:
    brief = ProductBrief(
        product_name=args.product_name,
        category=args.category,
        product_description=args.description,
        key_attributes=tuple(args.attribute),
        preservation_constraints=tuple(args.preserve),
        target_audience=args.target_audience,
        campaign_goal=args.campaign_goal,
        brand_style=args.brand_style,
    )

    retrieval_plan = None
    if args.line == "business-prior":
        if not args.manifest:
            raise ValueError("--manifest is required for the business-prior line")
        retrieval_plan = _build_retrieval_plan(args)
        prompt = BusinessPriorComposer().compose(brief, retrieval_plan)
    else:
        prompt = BaselineComposer().compose(brief)

    result: dict[str, Any] = {
        "line": args.line,
        "prompt": prompt.as_dict(),
    }
    if retrieval_plan is not None:
        result["retrieval_plan"] = _serialize(retrieval_plan)

    if args.input_image or args.reference_image or args.run:
        output_path = args.output or _default_output_path(
            product_name=args.product_name,
            line=args.line,
            seed=args.seed,
            output_format="png",
        )
        client = Flux2KleinClient(
            model_id=args.model,
            device=args.device,
            dtype=args.dtype,
            cpu_offload=args.cpu_offload,
            sequential_cpu_offload=args.sequential_cpu_offload,
            attention_slicing=args.attention_slicing,
        )
        request = client.build_request(
            prompt=prompt,
            input_image=args.input_image,
            reference_images=tuple(args.reference_image),
            width=args.width,
            height=args.height,
            seed=args.seed,
            output_path=output_path,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=args.max_sequence_length,
        )
        result["request"] = _serialize(request)
        if args.run:
            generation = client.generate(request)
            result["generated"] = True
            result["generation"] = _serialize(generation)
        else:
            result["generated"] = False
    return result


def _handle_generate_business_prior_photo(args: argparse.Namespace) -> dict[str, Any]:
    request = BusinessPriorInferenceRequest(
        image_path=args.image,
        product_title=args.product_title,
        retrieval_index_path=args.retrieval_index,
        output_dir=args.output_dir,
        hint_phrases=list(args.hint_phrase),
        request_id=args.request_id,
        product_id=args.product_id,
        source_page_url=args.source_page_url,
        source_image_url=args.source_image_url,
        model_id=args.model,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        analysis_device=args.analysis_device,
        localization_device=args.localization_device,
        candidate_modes=list(args.candidate_modes),
        skip_analysis=bool(args.skip_analysis),
        top_k=args.top_k,
        seed=args.seed,
        cpu_offload=args.cpu_offload,
        sequential_cpu_offload=args.sequential_cpu_offload,
        attention_slicing=args.attention_slicing,
    )
    result = run_business_prior_inference(request)
    return result.model_dump()


def _handle_evaluate_prompt(args: argparse.Namespace) -> dict[str, Any]:
    prompt_data = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
    prompt = FluxPrompt.from_mapping(prompt_data)
    evaluation = PromptRubric(pass_threshold=args.pass_threshold).evaluate(prompt)
    return evaluation.as_dict()


def _build_retrieval_plan(args: argparse.Namespace):
    annotations, embeddings = _load_creative_metadata(args.creative_metadata)
    items = load_creative_ranking_manifest(args.manifest, annotations=annotations, embeddings=embeddings)
    planner = CTRAwareRetrievalPlanner(items)
    planner_input = PlannerInput(
        product_name=args.product_name,
        category=args.category,
        subject_description=args.description,
        product_attributes=tuple(args.attribute),
        preservation_constraints=tuple(args.preserve),
        query_embedding=_parse_embedding(args.query_embedding),
        top_k=args.top_k,
    )
    return planner.plan(planner_input)


def _load_creative_metadata(
    metadata_path: str | None,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Sequence[float]]]:
    if not metadata_path:
        return {}, {}
    raw = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    annotations: dict[str, Mapping[str, Any]] = {}
    embeddings: dict[str, Sequence[float]] = {}

    if not isinstance(raw, Mapping):
        raise ValueError("Creative metadata JSON must be an object")

    containers: list[tuple[str, Mapping[str, Any]]] = []
    if "by_image" in raw and isinstance(raw["by_image"], Mapping):
        containers.append(("image", raw["by_image"]))
    if "by_item" in raw and isinstance(raw["by_item"], Mapping):
        containers.append(("item", raw["by_item"]))
    if not containers:
        containers.append(("flat", raw))

    for _, container in containers:
        for key, value in container.items():
            if not isinstance(value, Mapping):
                continue
            annotations[key] = value
            embedding = value.get("embedding")
            if isinstance(embedding, Sequence) and not isinstance(embedding, (str, bytes)):
                embeddings[key] = [float(component) for component in embedding]
    return annotations, embeddings


def _parse_embedding(value: str | None) -> tuple[float, ...] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    return tuple(float(part) for part in parts if part)


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_serialize(item) for item in value]
    return value


def _print_json(payload: Mapping[str, Any]) -> None:
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=True)
    sys.stdout.write("\n")


def _default_output_path(
    *,
    product_name: str,
    line: str,
    seed: int | None,
    output_format: str,
) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", product_name.lower()).strip("-") or "product"
    seed_suffix = "auto" if seed is None else str(seed)
    extension = output_format.lower().replace("jpeg", "jpg")
    return str(Path.cwd() / "outputs" / "generated" / f"{slug}-{line}-{seed_suffix}.{extension}")


if __name__ == "__main__":
    raise SystemExit(app())
