from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .models import ManifestRow, SyntheticBootstrapRecord


def _find_foreground_bounds(image: Image.Image) -> tuple[int, int, int, int] | None:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is not None:
        return bbox
    return image.getbbox()


def _solve_perspective_coefficients(
    source_points: Sequence[tuple[float, float]],
    target_points: Sequence[tuple[float, float]],
) -> tuple[float, ...]:
    matrix = []
    vector = []
    for (src_x, src_y), (dst_x, dst_y) in zip(source_points, target_points, strict=True):
        matrix.append([dst_x, dst_y, 1, 0, 0, 0, -src_x * dst_x, -src_x * dst_y])
        matrix.append([0, 0, 0, dst_x, dst_y, 1, -src_y * dst_x, -src_y * dst_y])
        vector.append(src_x)
        vector.append(src_y)

    coefficients = np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(vector))
    return tuple(float(value) for value in coefficients)


def _build_gradient_background(
    size: tuple[int, int], rng: np.random.Generator
) -> tuple[Image.Image, tuple[int, int, int]]:
    width, height = size
    start = rng.integers(170, 235, size=3)
    end = np.clip(start + rng.integers(-30, 31, size=3), 110, 245)
    axis_length = width if rng.random() >= 0.5 else height
    ramp = np.linspace(0.0, 1.0, axis_length, dtype=np.float32)

    if axis_length == width:
        weights = ramp.reshape(1, width, 1)
        gradient = ((1.0 - weights) * start + weights * end).astype(np.uint8)
        gradient = np.repeat(gradient, height, axis=0)
    else:
        weights = ramp.reshape(height, 1, 1)
        gradient = ((1.0 - weights) * start + weights * end).astype(np.uint8)
        gradient = np.repeat(gradient, width, axis=1)

    y_coords, x_coords = np.ogrid[:height, :width]
    center_x = width / 2.0
    center_y = height / 2.0
    distance = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    distance = distance / max(distance.max(), 1.0)
    vignette = np.clip(1.0 - 0.18 * distance, 0.82, 1.0)
    gradient = np.clip(gradient.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)
    background = Image.fromarray(gradient, mode="RGB")
    midpoint = tuple(int(value) for value in ((start + end) / 2.0))
    return background, midpoint


def _add_sensor_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    array = np.asarray(image, dtype=np.int16)
    noise = rng.normal(loc=0.0, scale=4.0, size=array.shape)
    noised = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noised, mode="RGB")


def _maybe_motion_blur(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    blur_amount = float(rng.uniform(0.0, 1.25))
    if blur_amount <= 0.15:
        return image

    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        return image.filter(ImageFilter.GaussianBlur(radius=blur_amount))

    kernel_size = 5 if blur_amount < 0.8 else 7
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    if rng.random() >= 0.5:
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
    else:
        kernel[:, kernel_size // 2] = 1.0 / kernel_size

    array = np.asarray(image)
    blurred = cv2.filter2D(array, -1, kernel)
    return Image.fromarray(blurred, mode="RGB")


@dataclass(slots=True)
class SyntheticBootstrapConfig:
    """Configuration for CPU-only synthetic bootstrap generation."""

    canvas_size: tuple[int, int] = (768, 768)
    seed: int | None = None
    split_name: str = "bootstrap"
    variants_per_image: int = 1


@dataclass(slots=True)
class PseudoHandheldBootstrapper:
    """Create CPU-only pseudo handheld photos from product crops."""

    canvas_size: tuple[int, int] = (768, 768)
    seed: int | None = None

    def _synthesize(
        self,
        source_path: Path,
        *,
        variant_seed: int,
    ) -> tuple[Image.Image, dict[str, float | int]]:
        rng = np.random.default_rng(variant_seed)
        source = Image.open(source_path).convert("RGBA")
        bounds = _find_foreground_bounds(source)
        if bounds is not None:
            source = source.crop(bounds)

        target_long_side = int(min(self.canvas_size) * float(rng.uniform(0.4, 0.68)))
        scale = target_long_side / max(source.size)
        resized = source.resize(
            (
                max(1, int(source.width * scale)),
                max(1, int(source.height * scale)),
            ),
            resample=Image.Resampling.LANCZOS,
        )

        padded = Image.new(
            "RGBA",
            (max(2, int(resized.width * 1.5)), max(2, int(resized.height * 1.5))),
            (0, 0, 0, 0),
        )
        padded.paste(
            resized,
            ((padded.width - resized.width) // 2, (padded.height - resized.height) // 2),
            resized,
        )

        perspective_jitter = float(rng.uniform(0.04, 0.12))
        max_dx = padded.width * perspective_jitter
        max_dy = padded.height * perspective_jitter
        src_points = (
            (0.0, 0.0),
            (float(padded.width), 0.0),
            (float(padded.width), float(padded.height)),
            (0.0, float(padded.height)),
        )
        dst_points = (
            (float(rng.uniform(0.0, max_dx)), float(rng.uniform(0.0, max_dy))),
            (float(padded.width - rng.uniform(0.0, max_dx)), float(rng.uniform(0.0, max_dy))),
            (
                float(padded.width - rng.uniform(0.0, max_dx)),
                float(padded.height - rng.uniform(0.0, max_dy)),
            ),
            (float(rng.uniform(0.0, max_dx)), float(padded.height - rng.uniform(0.0, max_dy))),
        )
        perspective = _solve_perspective_coefficients(src_points, dst_points)
        warped = padded.transform(
            padded.size,
            Image.Transform.PERSPECTIVE,
            perspective,
            resample=Image.Resampling.BICUBIC,
        )

        rotation = float(rng.uniform(-12.0, 12.0))
        rotated = warped.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)

        brightness = float(rng.uniform(0.92, 1.08))
        contrast = float(rng.uniform(0.9, 1.12))
        stylized = ImageEnhance.Brightness(rotated).enhance(brightness)
        stylized = ImageEnhance.Contrast(stylized).enhance(contrast)

        background, background_rgb = _build_gradient_background(self.canvas_size, rng)
        composed = background.copy()

        alpha = stylized.getchannel("A")
        shadow = Image.new("RGBA", stylized.size, (0, 0, 0, 0))
        shadow.putalpha(alpha)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(12.0, 18.0))))
        shadow_layer = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
        object_layer = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))

        center_x = self.canvas_size[0] // 2
        center_y = self.canvas_size[1] // 2
        offset_x = int(rng.uniform(-0.06, 0.06) * self.canvas_size[0])
        offset_y = int(rng.uniform(-0.03, 0.08) * self.canvas_size[1])
        paste_x = center_x - stylized.width // 2 + offset_x
        paste_y = center_y - stylized.height // 2 + offset_y

        shadow_layer.paste(
            shadow,
            (paste_x + int(rng.uniform(10, 18)), paste_y + int(rng.uniform(12, 22))),
            shadow,
        )
        object_layer.paste(stylized, (paste_x, paste_y), stylized)

        composed = Image.alpha_composite(composed.convert("RGBA"), shadow_layer)
        composed = Image.alpha_composite(composed, object_layer).convert("RGB")
        composed = _add_sensor_noise(composed, rng)
        composed = _maybe_motion_blur(composed, rng)
        blur_radius = float(rng.uniform(0.0, 0.9))
        if blur_radius > 0.1:
            composed = composed.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        metadata = {
            "rotation_degrees": rotation,
            "scale": scale,
            "perspective_jitter": perspective_jitter,
            "blur_radius": blur_radius,
            "brightness": brightness,
            "contrast": contrast,
            "background_r": background_rgb[0],
            "background_g": background_rgb[1],
            "background_b": background_rgb[2],
        }
        return composed, metadata

    def generate_from_manifest_rows(
        self,
        rows: Sequence[ManifestRow],
        image_root: str | Path,
        output_dir: str | Path,
        *,
        split_name: str,
        variants_per_image: int = 1,
    ) -> list[SyntheticBootstrapRecord]:
        image_root_path = Path(image_root)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        run_rng = np.random.default_rng(self.seed)
        records: list[SyntheticBootstrapRecord] = []

        for row in rows:
            source_path = image_root_path / row.image_name
            for variant_index in range(variants_per_image):
                variant_seed = int(run_rng.integers(0, 2**31 - 1))
                image, metadata = self._synthesize(source_path, variant_seed=variant_seed)
                synthetic_name = (
                    f"{Path(row.image_name).stem}_{split_name}_bootstrap_{variant_index:02d}_{variant_seed}.png"
                )
                synthetic_path = output_dir_path / synthetic_name
                image.save(synthetic_path)
                records.append(
                    SyntheticBootstrapRecord(
                        source_item_id=row.item_id,
                        source_image_name=row.image_name,
                        synthetic_image_name=synthetic_name,
                        synthetic_image_path=synthetic_path,
                        split_name=split_name,
                        variant_index=variant_index,
                        seed=variant_seed,
                        output_width=image.width,
                        output_height=image.height,
                        rotation_degrees=float(metadata["rotation_degrees"]),
                        scale=float(metadata["scale"]),
                        perspective_jitter=float(metadata["perspective_jitter"]),
                        blur_radius=float(metadata["blur_radius"]),
                        brightness=float(metadata["brightness"]),
                        contrast=float(metadata["contrast"]),
                        background_r=int(metadata["background_r"]),
                        background_g=int(metadata["background_g"]),
                        background_b=int(metadata["background_b"]),
                    )
                )

        return records

    def generate_from_image_paths(
        self,
        image_paths: Sequence[str | Path],
        output_dir: str | Path,
        *,
        split_name: str = "bootstrap",
        variants_per_image: int = 1,
    ) -> list[SyntheticBootstrapRecord]:
        rows = [
            ManifestRow(item_id=Path(path).stem, image_name=Path(path).name, ds=0, pv=0, clk=0)
            for path in image_paths
        ]
        image_root = Path(image_paths[0]).parent if image_paths else Path(output_dir)
        return self.generate_from_manifest_rows(
            rows,
            image_root=image_root,
            output_dir=output_dir,
            split_name=split_name,
            variants_per_image=variants_per_image,
        )


def generate_bootstrap_examples(
    rows: Sequence[ManifestRow],
    image_root: str | Path,
    output_dir: str | Path,
    *,
    config: SyntheticBootstrapConfig | None = None,
    split_name: str | None = None,
    variants_per_image: int | None = None,
) -> list[SyntheticBootstrapRecord]:
    """Generate pseudo handheld images from manifest rows with a lightweight config."""

    resolved = config or SyntheticBootstrapConfig()
    bootstrapper = PseudoHandheldBootstrapper(
        canvas_size=resolved.canvas_size,
        seed=resolved.seed,
    )
    return bootstrapper.generate_from_manifest_rows(
        rows,
        image_root=image_root,
        output_dir=output_dir,
        split_name=split_name or resolved.split_name,
        variants_per_image=variants_per_image or resolved.variants_per_image,
    )
