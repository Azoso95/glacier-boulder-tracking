"""Deterministic paired image augmentation for prompted tracking samples."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class ImageAugmentation:
    """Image-domain noise applied on demand by the HDF5 dataset loader."""

    enabled: bool = False
    photometric_probability: float = 0.8
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    saturation_range: tuple[float, float] = (0.8, 1.2)
    hue_range: tuple[float, float] = (-0.05, 0.05)
    gamma_range: tuple[float, float] = (0.8, 1.2)
    rotation_degrees: tuple[int, ...] = (0, 90, 180, 270)
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    gaussian_blur_probability: float = 0.3
    gaussian_blur_sigma_px: tuple[float, float] = (0.1, 1.5)
    gaussian_noise_probability: float = 0.3
    gaussian_noise_std: tuple[float, float] = (0.0, 0.03)
    composite_shadow_probability: float = 0.0
    cast_shadow_probability: float = 0.0
    cast_shadow_coverage_range: tuple[float, float] = (0.25, 0.75)
    cast_shadow_edge_softness_px: tuple[float, float] = (8.0, 40.0)
    shadow_transmission_range: tuple[float, float] = (0.10, 0.45)
    shadow_ambient_range: tuple[float, float] = (0.002, 0.012)
    shadow_red_gain_range: tuple[float, float] = (0.72, 0.90)
    shadow_green_gain_range: tuple[float, float] = (0.85, 1.00)
    shadow_blue_gain_range: tuple[float, float] = (1.00, 1.18)
    shadow_field_strength_range: tuple[float, float] = (0.00, 0.14)
    shadow_field_grid_size: int = 8

    def __post_init__(self) -> None:
        probabilities = (
            self.photometric_probability,
            self.horizontal_flip_probability,
            self.vertical_flip_probability,
            self.gaussian_blur_probability,
            self.gaussian_noise_probability,
            self.composite_shadow_probability,
            self.cast_shadow_probability,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError(
                "image augmentation probabilities must be between zero and one"
            )
        if not (
            0 <= self.cast_shadow_coverage_range[0]
            <= self.cast_shadow_coverage_range[1]
            <= 1
        ):
            raise ValueError("cast_shadow_coverage_range must be ordered within [0, 1]")
        if not (
            0 < self.cast_shadow_edge_softness_px[0]
            <= self.cast_shadow_edge_softness_px[1]
        ):
            raise ValueError("cast_shadow_edge_softness_px must be positive and ordered")
        for name, value in (
            ("brightness_range", self.brightness_range),
            ("contrast_range", self.contrast_range),
            ("saturation_range", self.saturation_range),
            ("gamma_range", self.gamma_range),
            ("shadow_transmission_range", self.shadow_transmission_range),
            ("shadow_ambient_range", self.shadow_ambient_range),
            ("shadow_red_gain_range", self.shadow_red_gain_range),
            ("shadow_green_gain_range", self.shadow_green_gain_range),
            ("shadow_blue_gain_range", self.shadow_blue_gain_range),
        ):
            if not 0 <= value[0] <= value[1] or (
                name
                in {
                    "brightness_range",
                    "contrast_range",
                    "saturation_range",
                    "gamma_range",
                }
                and value[0] == 0
            ):
                raise ValueError(f"{name} must be positive and ordered")
        if (
            not 0
            <= self.shadow_field_strength_range[0]
            <= self.shadow_field_strength_range[1]
            < 1
        ):
            raise ValueError(
                "shadow_field_strength_range must be ordered within [0, 1)"
            )
        if self.shadow_field_grid_size < 2:
            raise ValueError("shadow_field_grid_size must be at least two")
        if not -0.5 <= self.hue_range[0] <= self.hue_range[1] <= 0.5:
            raise ValueError("hue_range must be ordered within [-0.5, 0.5]")
        if not self.rotation_degrees:
            raise ValueError("rotation_degrees cannot be empty")
        if any(value % 90 != 0 for value in self.rotation_degrees):
            raise ValueError("rotation_degrees supports only multiples of 90")
        if not 0 <= self.gaussian_blur_sigma_px[0] <= self.gaussian_blur_sigma_px[1]:
            raise ValueError("gaussian_blur_sigma_px must be non-negative and ordered")
        if not 0 <= self.gaussian_noise_std[0] <= self.gaussian_noise_std[1]:
            raise ValueError("gaussian_noise_std must be non-negative and ordered")


@dataclass(frozen=True)
class GeometryChoice:
    quarter_turns: int
    horizontal_flip: bool
    vertical_flip: bool


def augment_search_episode(
    reference_image: Tensor,
    source_prompt: Tensor,
    source_point_xy: Tensor,
    target_canvas: Tensor,
    target_point_xy: Tensor,
    target_valid_mask: Tensor,
    *,
    config: ImageAugmentation,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, GeometryChoice]:
    """Augment one source and a complete target-search canvas consistently.

    Geometry is sampled once and shared by both survey dates.  Appearance is
    sampled independently by survey, but only once for the target canvas.  The
    target tiles must therefore be extracted *after* this function returns so
    overlapping tiles contain identical augmented pixels.
    """

    if not config.enabled:
        return (
            reference_image,
            source_prompt,
            source_point_xy,
            target_canvas,
            target_point_xy,
            target_valid_mask,
            GeometryChoice(0, False, False),
        )
    choice = _sample_geometry(config, generator)
    reference_shape = tuple(reference_image.shape[-2:])
    target_shape = tuple(target_canvas.shape[-2:])
    reference_image = _apply_geometry(reference_image, choice)
    source_prompt = _apply_geometry(source_prompt, choice)
    source_point_xy = _transform_point(source_point_xy, reference_shape, choice)
    target_canvas = _apply_geometry(target_canvas, choice)
    target_point_xy = _transform_point(target_point_xy, target_shape, choice)
    target_valid_mask = _apply_geometry(target_valid_mask, choice)
    reference_image = _augment_image(reference_image, config, generator)
    target_canvas = _augment_image(target_canvas, config, generator)
    return (
        reference_image,
        source_prompt,
        source_point_xy,
        target_canvas,
        target_point_xy,
        target_valid_mask,
        choice,
    )


def transform_window(
    window: tuple[int, int, int, int],
    shape_hw: tuple[int, int],
    choice: GeometryChoice,
) -> tuple[int, int, int, int]:
    """Transform a half-open ``(row, column, height, width)`` image window."""

    row, column, height, width = (int(value) for value in window)
    canvas_height, canvas_width = (int(value) for value in shape_hw)
    if (
        row < 0
        or column < 0
        or row + height > canvas_height
        or column + width > canvas_width
    ):
        raise ValueError("window lies outside the augmentation canvas")
    turns = choice.quarter_turns % 4
    if turns == 1:
        row, column, height, width = (
            canvas_width - column - width,
            row,
            width,
            height,
        )
        canvas_height, canvas_width = canvas_width, canvas_height
    elif turns == 2:
        row = canvas_height - row - height
        column = canvas_width - column - width
    elif turns == 3:
        row, column, height, width = (
            column,
            canvas_height - row - height,
            width,
            height,
        )
        canvas_height, canvas_width = canvas_width, canvas_height
    if choice.horizontal_flip:
        column = canvas_width - column - width
    if choice.vertical_flip:
        row = canvas_height - row - height
    return row, column, height, width


def augment_tracking_pair(
    reference_image: Tensor,
    source_prompt: Tensor,
    source_point_xy: Tensor,
    search_image: Tensor,
    target_heatmap: Tensor,
    target_point_xy: Tensor,
    *,
    config: ImageAugmentation,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, GeometryChoice]:
    """Augment a pair while preserving prompt/heatmap and point alignment.

    Geometry is shared between survey dates. Photometry, blur, and noise are
    sampled independently because illumination and sensor conditions can differ.
    """

    if not config.enabled:
        choice = GeometryChoice(0, False, False)
        return (
            reference_image,
            source_prompt,
            source_point_xy,
            search_image,
            target_heatmap,
            target_point_xy,
            choice,
        )
    choice = _sample_geometry(config, generator)
    _validate_rotation_shape(reference_image, choice)
    _validate_rotation_shape(search_image, choice)
    reference_shape = tuple(reference_image.shape[-2:])
    search_shape = tuple(search_image.shape[-2:])
    reference_image = _apply_geometry(reference_image, choice)
    source_prompt = _apply_geometry(source_prompt, choice)
    source_point_xy = _transform_point(source_point_xy, reference_shape, choice)
    search_image = _apply_geometry(search_image, choice)
    target_heatmap = _apply_geometry(target_heatmap, choice)
    target_point_xy = _transform_point(target_point_xy, search_shape, choice)
    reference_image = _augment_image(reference_image, config, generator)
    search_image = _augment_image(search_image, config, generator)
    return (
        reference_image,
        source_prompt,
        source_point_xy,
        search_image,
        target_heatmap,
        target_point_xy,
        choice,
    )


def _sample_geometry(
    config: ImageAugmentation, generator: torch.Generator
) -> GeometryChoice:
    index = int(torch.randint(len(config.rotation_degrees), (), generator=generator))
    turns = (config.rotation_degrees[index] // 90) % 4
    return GeometryChoice(
        quarter_turns=turns,
        horizontal_flip=_bernoulli(config.horizontal_flip_probability, generator),
        vertical_flip=_bernoulli(config.vertical_flip_probability, generator),
    )


def _validate_rotation_shape(image: Tensor, choice: GeometryChoice) -> None:
    if choice.quarter_turns % 2 and image.shape[-2] != image.shape[-1]:
        raise ValueError(
            "90/270 degree augmentation requires square reference and search crops"
        )


def _apply_geometry(tensor: Tensor, choice: GeometryChoice) -> Tensor:
    result = torch.rot90(tensor, choice.quarter_turns, dims=(-2, -1))
    if choice.horizontal_flip:
        result = torch.flip(result, dims=(-1,))
    if choice.vertical_flip:
        result = torch.flip(result, dims=(-2,))
    return result


def _transform_point(
    point_xy: Tensor, shape_hw: tuple[int, int], choice: GeometryChoice
) -> Tensor:
    point = point_xy.clone()
    if not bool(torch.isfinite(point).all()):
        return point
    x, y = point[0].clone(), point[1].clone()
    height, width = shape_hw
    if choice.quarter_turns == 1:
        x, y = y, width - 1 - x
        height, width = width, height
    elif choice.quarter_turns == 2:
        x, y = width - 1 - x, height - 1 - y
    elif choice.quarter_turns == 3:
        x, y = height - 1 - y, x
        height, width = width, height
    if choice.horizontal_flip:
        x = width - 1 - x
    if choice.vertical_flip:
        y = height - 1 - y
    return torch.stack((x, y))


def _augment_image(
    image: Tensor, config: ImageAugmentation, generator: torch.Generator
) -> Tensor:
    result = image
    if _bernoulli(config.photometric_probability, generator):
        operations = (
            lambda value: TF.adjust_brightness(
                value, _uniform(config.brightness_range, generator)
            ),
            lambda value: TF.adjust_contrast(
                value, _uniform(config.contrast_range, generator)
            ),
            lambda value: TF.adjust_saturation(
                value, _uniform(config.saturation_range, generator)
            ),
            lambda value: TF.adjust_hue(value, _uniform(config.hue_range, generator)),
            lambda value: TF.adjust_gamma(
                value, _uniform(config.gamma_range, generator)
            ),
        )
        for index in torch.randperm(len(operations), generator=generator).tolist():
            result = operations[index](result)
    if _bernoulli(config.composite_shadow_probability, generator):
        result = _composite_shadow(result, config, generator)
    if _bernoulli(config.cast_shadow_probability, generator):
        result = _cast_shadow(result, config, generator)
    if _bernoulli(config.gaussian_blur_probability, generator):
        sigma = _uniform(config.gaussian_blur_sigma_px, generator)
        if sigma > 0:
            kernel = max(3, 2 * int(round(3 * sigma)) + 1)
            result = TF.gaussian_blur(result, [kernel, kernel], [sigma, sigma])
    if _bernoulli(config.gaussian_noise_probability, generator):
        standard_deviation = _uniform(config.gaussian_noise_std, generator)
        if standard_deviation > 0:
            noise = torch.randn(
                result.shape,
                dtype=result.dtype,
                device=result.device,
                generator=generator,
            )
            result = result + standard_deviation * noise
    return result.clamp(0, 1)


def _composite_shadow(
    image: Tensor, config: ImageAugmentation, generator: torch.Generator
) -> Tensor:
    """Apply coherent, cool complete shade in linear-light RGB.

    One low-frequency irradiance field is sampled for the complete tensor. For
    search episodes this function receives the reconstructed target canvas, so
    every subsequently extracted target tile shares the same illumination.
    """

    transmission = _uniform(config.shadow_transmission_range, generator)
    ambient = _uniform(config.shadow_ambient_range, generator)
    field_strength = _uniform(config.shadow_field_strength_range, generator)
    tint = image.new_tensor(
        [
            _uniform(config.shadow_red_gain_range, generator),
            _uniform(config.shadow_green_gain_range, generator),
            _uniform(config.shadow_blue_gain_range, generator),
        ]
    )[:, None, None]

    size = config.shadow_field_grid_size
    coarse = torch.randn((1, 1, size, size), generator=generator, dtype=torch.float32)
    field = F.interpolate(
        coarse,
        size=image.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )[0, 0].to(device=image.device, dtype=image.dtype)
    field = (field - field.mean()) / field.std().clamp_min(1e-6)
    illumination = (transmission * (1 + field_strength * field)).clamp_min(0.02)

    linear = torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055).pow(2.4),
    )
    shaded = linear * illumination[None] * tint + ambient * tint
    return torch.where(
        shaded <= 0.0031308,
        12.92 * shaded,
        1.055 * shaded.clamp_min(0).pow(1 / 2.4) - 0.055,
    ).clamp(0, 1)


def _cast_shadow(
    image: Tensor, config: ImageAugmentation, generator: torch.Generator
) -> Tensor:
    """Apply a spatially coherent soft-edged cast shadow in linear light."""

    height, width = image.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, dtype=image.dtype, device=image.device),
        torch.linspace(-1, 1, width, dtype=image.dtype, device=image.device),
        indexing="ij",
    )
    angle = _uniform((0.0, 2 * math.pi), generator)
    signed = math.cos(angle) * xx + math.sin(angle) * yy
    coverage = _uniform(config.cast_shadow_coverage_range, generator)
    threshold = 1.0 - 2.0 * coverage
    softness_px = _uniform(config.cast_shadow_edge_softness_px, generator)
    softness = max(softness_px / max(height, width) * 2.0, 1e-4)
    mask = torch.sigmoid((signed - threshold) / softness)

    size = config.shadow_field_grid_size
    coarse = torch.randn((1, 1, size, size), generator=generator, dtype=torch.float32)
    irregularity = F.interpolate(
        coarse, size=(height, width), mode="bicubic", align_corners=False
    )[0, 0].to(device=image.device, dtype=image.dtype)
    irregularity = irregularity / irregularity.std().clamp_min(1e-6)
    mask = (mask + 0.08 * irregularity).clamp(0, 1)

    transmission = _uniform(config.shadow_transmission_range, generator)
    ambient = _uniform(config.shadow_ambient_range, generator)
    tint = image.new_tensor(
        [
            _uniform(config.shadow_red_gain_range, generator),
            _uniform(config.shadow_green_gain_range, generator),
            _uniform(config.shadow_blue_gain_range, generator),
        ]
    )[:, None, None]
    linear = torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055).pow(2.4),
    )
    shaded = linear * transmission * tint + ambient * tint
    mixed = linear * (1 - mask[None]) + shaded * mask[None]
    return torch.where(
        mixed <= 0.0031308,
        12.92 * mixed,
        1.055 * mixed.clamp_min(0).pow(1 / 2.4) - 0.055,
    ).clamp(0, 1)


def _uniform(interval: tuple[float, float], generator: torch.Generator) -> float:
    low, high = interval
    if low == high:
        return float(low)
    return float(torch.empty(()).uniform_(low, high, generator=generator))


def _bernoulli(probability: float, generator: torch.Generator) -> bool:
    return bool(torch.rand((), generator=generator) < probability)
