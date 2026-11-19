"""Backbone-specific channel normalization applied on the fly."""

from __future__ import annotations

import torch
from torch import Tensor

from glaciertrack.models.backbones import PreprocessingSpec


def normalize_for_backbone(image: Tensor, spec: PreprocessingSpec) -> Tensor:
    """Convert raw integer or [0, 1] float RGB to a backbone's expected values."""

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image input must have shape Bx3xHxW")
    if image.dtype == torch.uint8:
        value = image.to(torch.float32) * spec.input_scale
    elif image.is_floating_point():
        value = image.to(torch.float32)
    else:
        raise TypeError(f"unsupported image dtype: {image.dtype}")
    mean = value.new_tensor(spec.mean).view(1, 3, 1, 1)
    std = value.new_tensor(spec.std).view(1, 3, 1, 1)
    return (value - mean) / std
