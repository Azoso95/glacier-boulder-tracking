"""Differentiable reconstruction and losses for complete tiled-search episodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional as F


class StitchedSearch(NamedTuple):
    logits: Tensor
    weight_sum: Tensor
    covered: Tensor


@dataclass(frozen=True)
class SearchLoss:
    total: Tensor
    global_localization: Tensor
    tile_presence: Tensor
    positive_localization: Tensor
    negative_suppression: Tensor
    global_ranking: Tensor
    positive_tiles: int
    negative_tiles: int


def differentiable_stitch(
    location_logits: Tensor,
    match_logits: Tensor,
    tile_windows: tuple[tuple[int, int, int, int], ...],
    canvas_shape_hw: tuple[int, int],
    *,
    stride_px: int = 1,
    match_logit_weight: float = 0.0,
    valid_mask: Tensor | None = None,
) -> StitchedSearch:
    """Blend tile logits on one canvas without detaching the computation graph."""

    if location_logits.ndim != 3:
        raise ValueError("location_logits must have shape KxHxW")
    if match_logits.reshape(-1).shape[0] != location_logits.shape[0]:
        raise ValueError("match and location tile counts differ")
    if len(tile_windows) != location_logits.shape[0]:
        raise ValueError("tile window count differs from model output")
    if stride_px <= 0:
        raise ValueError("stride_px must be positive")
    if match_logit_weight < 0:
        raise ValueError("match_logit_weight must be non-negative")
    height, width = (int(value) for value in canvas_shape_hw)
    output_shape = (math.ceil(height / stride_px), math.ceil(width / stride_px))
    device = location_logits.device
    # Stitch in fp32 even when the network uses mixed precision.  A search
    # surface averages many overlapping contributions and is also normalized
    # globally, both of which are unnecessarily fragile in fp16.
    dtype = torch.float32
    weighted_sum = torch.zeros(output_shape, device=device, dtype=dtype)
    weight_sum = torch.zeros(output_shape, device=device, dtype=dtype)
    if valid_mask is None:
        valid = torch.ones(output_shape, device=device, dtype=torch.bool)
    else:
        valid = torch.as_tensor(valid_mask, device=device, dtype=torch.bool).squeeze()
        if tuple(valid.shape) != output_shape:
            raise ValueError(f"valid mask shape {tuple(valid.shape)} differs from {output_shape}")
    for index, (row, column, tile_height, tile_width) in enumerate(tile_windows):
        if min(row, column) < 0 or row + tile_height > height or column + tile_width > width:
            raise ValueError("tile lies outside the stitch canvas")
        global_top = row // stride_px
        global_left = column // stride_px
        global_bottom = math.ceil((row + tile_height) / stride_px)
        global_right = math.ceil((column + tile_width) / stride_px)
        output_size = (global_bottom - global_top, global_right - global_left)
        logits = location_logits[index].float()
        if tuple(logits.shape) != output_size:
            logits = F.interpolate(
                logits[None, None],
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        logits = logits + float(match_logit_weight) * match_logits.reshape(-1)[index]
        weight = _blend_weight(*output_size, device=device, dtype=dtype)
        rows = slice(global_top, global_bottom)
        columns = slice(global_left, global_right)
        weight = weight * valid[rows, columns]
        # CopySlices keeps the graph linking each tile contribution to the loss.
        weighted_sum[rows, columns] = weighted_sum[rows, columns] + logits * weight
        weight_sum[rows, columns] = weight_sum[rows, columns] + weight
    covered = (weight_sum > 0) & valid
    mosaic = torch.full(output_shape, -torch.inf, device=device, dtype=dtype)
    mosaic = torch.where(
        covered,
        weighted_sum / weight_sum.clamp_min(torch.finfo(dtype).eps),
        mosaic,
    )
    return StitchedSearch(mosaic, weight_sum, covered)


def search_episode_loss(
    stitched: StitchedSearch,
    location_logits: Tensor,
    match_logits: Tensor,
    tile_windows: tuple[tuple[int, int, int, int], ...],
    target_xy_px: Tensor | tuple[float, float] | None,
    *,
    stride_px: int,
    target_sigma_px: float,
    global_weight: float = 1.0,
    presence_weight: float = 0.25,
    positive_localization_weight: float = 0.25,
    negative_suppression_weight: float = 0.10,
    ranking_weight: float = 0.25,
    ranking_margin: float = 1.0,
    ranking_exclusion_radius_px: float | None = None,
    negative_temperature: float = 0.10,
    tile_target_xy_px: Tensor | None = None,
    separate_tile_target: bool = False,
) -> SearchLoss:
    """Compute balanced tile auxiliaries plus the operational global objective."""

    if target_sigma_px <= 0 or negative_temperature <= 0:
        raise ValueError("target sigma and negative temperature must be positive")
    if ranking_exclusion_radius_px is not None and ranking_exclusion_radius_px <= 0:
        raise ValueError("ranking exclusion radius must be positive")
    if (
        min(
            global_weight,
            presence_weight,
            positive_localization_weight,
            negative_suppression_weight,
            ranking_weight,
        )
        < 0
    ):
        raise ValueError("search loss weights cannot be negative")
    zero = location_logits.sum() * 0 + match_logits.sum() * 0
    target = None
    if target_xy_px is not None:
        target = torch.as_tensor(
            target_xy_px, device=location_logits.device, dtype=torch.float32
        ).reshape(2)
        if not bool(torch.isfinite(target).all()):
            target = None
    tile_target = target
    if separate_tile_target:
        tile_target = None if tile_target_xy_px is None else torch.as_tensor(
            tile_target_xy_px, device=location_logits.device, dtype=torch.float32
        ).reshape(2)
        if tile_target is not None and not bool(torch.isfinite(tile_target).all()):
            tile_target = None
    positive_flags = []
    for row, column, height, width in tile_windows:
        positive_flags.append(
            tile_target is not None
            and column <= float(tile_target[0]) <= column + width - 1
            and row <= float(tile_target[1]) <= row + height - 1
        )
    positives = torch.as_tensor(positive_flags, device=match_logits.device, dtype=torch.bool)
    negatives = ~positives
    presence_targets = positives.to(dtype=torch.float32)
    flat_match = match_logits.float().reshape(-1)
    positive_presence = (
        F.binary_cross_entropy_with_logits(flat_match[positives], presence_targets[positives])
        if bool(positives.any())
        else zero
    )
    negative_presence = (
        F.binary_cross_entropy_with_logits(flat_match[negatives], presence_targets[negatives])
        if bool(negatives.any())
        else zero
    )
    if bool(positives.any()) and bool(negatives.any()):
        tile_presence = 0.5 * (positive_presence + negative_presence)
    else:
        tile_presence = positive_presence + negative_presence

    local_losses: list[Tensor] = []
    if tile_target is not None:
        for index in torch.nonzero(positives, as_tuple=False).flatten().tolist():
            row, column, _, _ = tile_windows[index]
            distribution = _gaussian_distribution(
                tuple(location_logits[index].shape),
                (tile_target[0] - column, tile_target[1] - row),
                sigma_px=target_sigma_px,
                device=location_logits.device,
            )
            local_losses.append(
                -(distribution * F.log_softmax(location_logits[index].float().flatten(), 0)).sum()
            )
    positive_localization = torch.stack(local_losses).mean() if local_losses else zero

    if bool(negatives.any()):
        negative_values = location_logits[negatives].float().flatten(1)
        smooth_peak = float(negative_temperature) * (
            torch.logsumexp(negative_values / float(negative_temperature), dim=1)
            - math.log(negative_values.shape[1])
        )
        negative_suppression = F.softplus(smooth_peak).mean()
    else:
        negative_suppression = zero

    global_localization = zero
    global_ranking = zero
    if target is not None:
        distribution = _global_distribution(
            stitched.covered,
            target,
            stride_px=stride_px,
            sigma_px=target_sigma_px,
        )
        covered_logits = stitched.logits.float()[stitched.covered]
        global_localization = -(
            distribution[stitched.covered] * F.log_softmax(covered_logits, dim=0)
        ).sum()
        target_support = (distribution[stitched.covered] * covered_logits).sum()
        if ranking_exclusion_radius_px is None:
            support = distribution >= distribution.amax() * 0.05
        else:
            support = _target_radius_mask(
                stitched.covered.shape,
                target,
                stride_px=stride_px,
                radius_px=float(ranking_exclusion_radius_px),
            )
        false_support = stitched.covered & ~support
        if bool(false_support.any()):
            hard_negative = stitched.logits.float()[false_support].amax()
            global_ranking = F.softplus(hard_negative - target_support + float(ranking_margin))

    total = (
        float(global_weight) * global_localization
        + float(presence_weight) * tile_presence
        + float(positive_localization_weight) * positive_localization
        + float(negative_suppression_weight) * negative_suppression
        + float(ranking_weight) * global_ranking
    )
    return SearchLoss(
        total=total,
        global_localization=global_localization,
        tile_presence=tile_presence,
        positive_localization=positive_localization,
        negative_suppression=negative_suppression,
        global_ranking=global_ranking,
        positive_tiles=int(positives.sum()),
        negative_tiles=int(negatives.sum()),
    )


def _target_radius_mask(
    shape_hw: tuple[int, int],
    target_xy_px: Tensor,
    *,
    stride_px: int,
    radius_px: float,
) -> Tensor:
    """Return coarse cells whose centers lie within a target-space radius."""

    height, width = shape_hw
    yy, xx = torch.meshgrid(
        torch.arange(height, device=target_xy_px.device, dtype=torch.float32),
        torch.arange(width, device=target_xy_px.device, dtype=torch.float32),
        indexing="ij",
    )
    offset = (stride_px - 1) / 2
    xx = xx * stride_px + offset
    yy = yy * stride_px + offset
    return (xx - target_xy_px[0]).square() + (yy - target_xy_px[1]).square() <= float(
        radius_px
    ) ** 2


def _global_distribution(
    covered: Tensor,
    target_xy_px: Tensor,
    *,
    stride_px: int,
    sigma_px: float,
) -> Tensor:
    height, width = covered.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=covered.device, dtype=torch.float32),
        torch.arange(width, device=covered.device, dtype=torch.float32),
        indexing="ij",
    )
    offset = (stride_px - 1) / 2
    xx = xx * stride_px + offset
    yy = yy * stride_px + offset
    distribution = torch.exp(
        -0.5
        * ((xx - target_xy_px[0]).square() + (yy - target_xy_px[1]).square())
        / float(sigma_px) ** 2
    )
    distribution = distribution * covered
    total = distribution.sum()
    if not bool(total > 0):
        raise ValueError("target has no support in the stitched search surface")
    return distribution / total


def _gaussian_distribution(
    shape_hw: tuple[int, int],
    target_xy_px: tuple[Tensor, Tensor],
    *,
    sigma_px: float,
    device: torch.device,
) -> Tensor:
    height, width = shape_hw
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    distribution = torch.exp(
        -0.5
        * ((xx - target_xy_px[0]).square() + (yy - target_xy_px[1]).square())
        / float(sigma_px) ** 2
    ).flatten()
    return distribution / distribution.sum().clamp_min(1e-12)


def _blend_weight(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    vertical = (
        torch.hann_window(height, periodic=False, device=device, dtype=dtype)
        if height > 1
        else torch.ones(1, device=device, dtype=dtype)
    )
    horizontal = (
        torch.hann_window(width, periodic=False, device=device, dtype=dtype)
        if width > 1
        else torch.ones(1, device=device, dtype=dtype)
    )
    return torch.outer(vertical, horizontal).clamp_min(0.05)
