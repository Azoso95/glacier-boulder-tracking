"""Dynamic metric source prompts and target heatmaps.

Prompt rasters are deliberately derived at loading time.  The cache therefore
keeps compact annotations while training can perturb prompt quality without
rewriting the imagery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

PromptApproach = Literal["auto", "ellipse", "gaussian_point"]


@dataclass(frozen=True)
class PromptConfig:
    """Physical parameters used to render source prompts and training labels."""

    point_sigma_m: float = 0.35
    ellipse_edge_m: float = 0.10
    target_sigma_m: float = 0.35
    backend: str = "parametric"
    sam2: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if min(self.point_sigma_m, self.ellipse_edge_m, self.target_sigma_m) <= 0:
            raise ValueError("prompt distances must be positive")
        if self.backend not in {"parametric", "sam2"}:
            raise ValueError("prompt backend must be parametric or sam2")
        if self.backend == "sam2":
            from glaciertrack.data.sam2_masks import SAM2MaskConfig

            SAM2MaskConfig(**self.sam2)

    @classmethod
    def from_mapping(cls, payload: dict, *, default_sigma: float = 0.35) -> PromptConfig:
        return cls(
            point_sigma_m=float(payload.get("point_sigma_m", default_sigma)),
            ellipse_edge_m=float(payload.get("ellipse_edge_m", 0.10)),
            target_sigma_m=float(payload.get("target_sigma_m", 0.35)),
            backend=str(payload.get("backend", "parametric")),
            sam2=dict(payload.get("sam2", {})),
        )


@dataclass(frozen=True)
class PromptAugmentation:
    """Noise model for making source prompts robust to imperfect annotation."""

    enabled: bool = True
    center_jitter_m: float = 0.15
    scale_range: tuple[float, float] = (0.80, 1.25)
    orientation_jitter_deg: float = 15.0
    morphology_radius_px: tuple[int, int] = (-2, 2)
    blur_sigma_px: tuple[float, float] = (0.0, 1.25)
    point_only_probability: float = 0.15

    def __post_init__(self) -> None:
        if self.center_jitter_m < 0 or self.orientation_jitter_deg < 0:
            raise ValueError("prompt jitter cannot be negative")
        if not 0 < self.scale_range[0] <= self.scale_range[1]:
            raise ValueError("prompt scale_range must be positive and ordered")
        if self.morphology_radius_px[0] > self.morphology_radius_px[1]:
            raise ValueError("morphology_radius_px must be ordered")
        if not 0 <= self.blur_sigma_px[0] <= self.blur_sigma_px[1]:
            raise ValueError("blur_sigma_px must be non-negative and ordered")
        if not 0 <= self.point_only_probability <= 1:
            raise ValueError("point_only_probability must be between zero and one")


def render_source_prompt(
    size_hw: tuple[int, int],
    point_xy_px: Tensor | tuple[float, float],
    *,
    gsd_m_per_px: float,
    major_axis_m: float | None = None,
    minor_axis_m: float | None = None,
    orientation_deg: float | None = None,
    config: PromptConfig | None = None,
    augmentation: PromptAugmentation | None = None,
    approach: PromptApproach = "auto",
    generator: torch.Generator | None = None,
    image: Tensor | None = None,
) -> Tensor:
    """Render a soft point or ellipse prompt with shape ``1 x H x W``.

    ``major_axis_m`` and ``minor_axis_m`` are full ellipse diameters.  When either
    is unavailable, a metric Gaussian point prompt is used.  Augmentation changes
    only the source prompt and can therefore never leak the target endpoint.
    """

    if gsd_m_per_px <= 0:
        raise ValueError("gsd_m_per_px must be positive")
    height, width = _size(size_hw)
    spec = config or PromptConfig()
    aug = augmentation or PromptAugmentation(enabled=False)
    point = torch.as_tensor(point_xy_px, dtype=torch.float32).reshape(2).clone()
    if spec.backend == "sam2":
        from glaciertrack.data.sam2_masks import render_sam2_prompt

        if image is None or tuple(image.shape[-2:]) != (height, width):
            raise ValueError("SAM 2 requires the unnormalized source RGB image matching the prompt")
        return render_sam2_prompt(image, point, gsd_m_per_px, spec, aug, generator)

    if approach not in {"auto", "ellipse", "gaussian_point"}:
        raise ValueError(f"unsupported prompt approach: {approach!r}")
    ellipse_available = _positive_optional(major_axis_m) and _positive_optional(minor_axis_m)
    use_ellipse = approach != "gaussian_point" and ellipse_available
    angle = float(orientation_deg or 0.0)
    scale = 1.0
    if aug.enabled:
        point += (
            _uniform_pair(-aug.center_jitter_m, aug.center_jitter_m, generator)
            / gsd_m_per_px
        )
        scale = _uniform(aug.scale_range[0], aug.scale_range[1], generator)
        angle += _uniform(
            -aug.orientation_jitter_deg,
            aug.orientation_jitter_deg,
            generator,
        )
        if use_ellipse and _uniform(0.0, 1.0, generator) < aug.point_only_probability:
            use_ellipse = False

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    dx_m = (xx - point[0]) * gsd_m_per_px
    dy_m = (yy - point[1]) * gsd_m_per_px
    if use_ellipse:
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        along_m = cosine * dx_m + sine * dy_m
        across_m = -sine * dx_m + cosine * dy_m
        semi_major = float(major_axis_m) * scale / 2
        semi_minor = float(minor_axis_m) * scale / 2
        radius = torch.sqrt((along_m / semi_major) ** 2 + (across_m / semi_minor) ** 2)
        edge_scale = max(min(semi_major, semi_minor) / spec.ellipse_edge_m, 1.0)
        mask = torch.sigmoid((1.0 - radius) * edge_scale)
    else:
        sigma = spec.point_sigma_m * scale
        mask = torch.exp(-(dx_m.square() + dy_m.square()) / (2 * sigma**2))

    mask = mask.unsqueeze(0)
    if aug.enabled:
        radius = _randint(
            aug.morphology_radius_px[0],
            aug.morphology_radius_px[1],
            generator,
        )
        mask = _morphology(mask, radius)
        sigma = _uniform(aug.blur_sigma_px[0], aug.blur_sigma_px[1], generator)
        if sigma > 0.05:
            mask = _gaussian_blur(mask, sigma)
    return mask.clamp_(0, 1)


def render_target_heatmap(
    size_hw: tuple[int, int],
    target_xy_px: Tensor | tuple[float, float] | None,
    *,
    match_present: bool,
    gsd_m_per_px: float,
    sigma_m: float = 0.35,
) -> Tensor:
    """Render a target Gaussian with shape ``1 x H x W`` or zeros for no-match."""

    if gsd_m_per_px <= 0 or sigma_m <= 0:
        raise ValueError("GSD and target sigma must be positive")
    height, width = _size(size_hw)
    if not match_present:
        return torch.zeros((1, height, width), dtype=torch.float32)
    if target_xy_px is None:
        raise ValueError("positive targets require a target point")
    target = torch.as_tensor(target_xy_px, dtype=torch.float32).reshape(2)
    if not torch.isfinite(target).all():
        raise ValueError("positive target point must be finite")
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    dx_m = (xx - target[0]) * gsd_m_per_px
    dy_m = (yy - target[1]) * gsd_m_per_px
    return torch.exp(-(dx_m.square() + dy_m.square()) / (2 * sigma_m**2)).unsqueeze(0)


def _size(value: tuple[int, int]) -> tuple[int, int]:
    height, width = (int(value[0]), int(value[1]))
    if height <= 0 or width <= 0:
        raise ValueError("raster dimensions must be positive")
    return height, width


def _positive_optional(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0


def _uniform(low: float, high: float, generator: torch.Generator | None) -> float:
    if low == high:
        return float(low)
    return float(torch.empty(()).uniform_(low, high, generator=generator))


def _uniform_pair(low: float, high: float, generator: torch.Generator | None) -> Tensor:
    if low == high:
        return torch.full((2,), low, dtype=torch.float32)
    return torch.empty(2).uniform_(low, high, generator=generator)


def _randint(low: int, high: int, generator: torch.Generator | None) -> int:
    if low == high:
        return low
    return int(torch.randint(low, high + 1, (), generator=generator))


def _morphology(mask: Tensor, radius: int) -> Tensor:
    if radius == 0:
        return mask
    kernel = 2 * abs(radius) + 1
    batched = mask.unsqueeze(0)
    if radius > 0:
        return F.max_pool2d(batched, kernel, stride=1, padding=abs(radius)).squeeze(0)
    eroded = -F.max_pool2d(-batched, kernel, stride=1, padding=abs(radius))
    return eroded.squeeze(0)


def _gaussian_blur(mask: Tensor, sigma: float) -> Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, dtype=mask.dtype)
    kernel = torch.exp(-coordinates.square() / (2 * sigma**2))
    kernel /= kernel.sum()
    batched = mask.unsqueeze(0)
    horizontal = F.conv2d(batched, kernel.view(1, 1, 1, -1), padding=(0, radius))
    return F.conv2d(horizontal, kernel.view(1, 1, -1, 1), padding=(radius, 0)).squeeze(0)
