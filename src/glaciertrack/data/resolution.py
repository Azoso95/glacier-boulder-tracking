"""Resolution simulation that preserves tensor shape and physical footprint."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def simulate_effective_resolution(
    image: torch.Tensor,
    *,
    current_effective_gsd_m_per_px: float,
    target_effective_gsd_m_per_px: float,
    tensor_gsd_m_per_px: float,
    upsample_mode: str = "bilinear",
) -> torch.Tensor:
    """Degrade detail to a coarser GSD and return it on the original tensor grid.

    ``image`` must be a floating-point CHW or NCHW tensor. Area interpolation
    performs antialiased spatial aggregation; bilinear or bicubic interpolation
    restores only tensor compatibility and does not claim to recover detail.
    """

    if image.ndim not in {3, 4}:
        raise ValueError("image must have CHW or NCHW layout")
    if not image.is_floating_point():
        raise TypeError("image must be floating point; scale integer imagery before augmentation")
    if min(
        current_effective_gsd_m_per_px,
        target_effective_gsd_m_per_px,
        tensor_gsd_m_per_px,
    ) <= 0:
        raise ValueError("GSD values must be positive")
    if target_effective_gsd_m_per_px < current_effective_gsd_m_per_px:
        raise ValueError("target effective GSD cannot be finer than the current effective GSD")
    if upsample_mode not in {"bilinear", "bicubic"}:
        raise ValueError("upsample_mode must be 'bilinear' or 'bicubic'")
    if target_effective_gsd_m_per_px == current_effective_gsd_m_per_px:
        return image.clone()

    batched = image.unsqueeze(0) if image.ndim == 3 else image
    height, width = batched.shape[-2:]
    coarse_height = max(1, round(height * tensor_gsd_m_per_px / target_effective_gsd_m_per_px))
    coarse_width = max(1, round(width * tensor_gsd_m_per_px / target_effective_gsd_m_per_px))
    coarse = F.interpolate(batched, size=(coarse_height, coarse_width), mode="area")
    restored = F.interpolate(
        coarse,
        size=(height, width),
        mode=upsample_mode,
        align_corners=False,
    )
    return restored.squeeze(0) if image.ndim == 3 else restored
