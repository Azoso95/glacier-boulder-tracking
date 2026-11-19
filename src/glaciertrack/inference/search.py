"""Endpoint-blind construction of local, exhaustive, and progressive searches."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class SearchMode(str, Enum):
    """Supported operational search policies."""

    RADIUS = "radius"
    FULL_GLACIER = "full_glacier"
    PROGRESSIVE = "progressive"


@dataclass(frozen=True)
class SearchConfig:
    """Search settings on the target analysis grid."""

    mode: SearchMode | str = SearchMode.RADIUS
    tile_size_px: tuple[int, int] = (512, 512)
    tile_overlap: float = 0.50
    tile_batch_size: int = 16
    radius_m: float | None = 250.0
    maximum_radius_m: float | None = None
    ring_step_m: float = 100.0
    confidence_threshold: float = 0.80
    decision_threshold: float = 0.50
    stability_expansions: int = 1
    stability_tolerance_m: float = 1.0
    minimum_peak_margin: float = 0.10
    match_logit_weight: float = 1.0

    def __post_init__(self) -> None:
        mode = SearchMode(self.mode)
        object.__setattr__(self, "mode", mode)
        if len(self.tile_size_px) != 2 or min(self.tile_size_px) <= 0:
            raise ValueError("tile_size_px must contain two positive values")
        if not 0 <= self.tile_overlap < 1:
            raise ValueError("tile_overlap must be in [0, 1)")
        if self.tile_batch_size <= 0:
            raise ValueError("tile_batch_size must be positive")
        if mode is SearchMode.RADIUS and (self.radius_m is None or self.radius_m <= 0):
            raise ValueError("radius search requires a positive radius_m")
        if self.maximum_radius_m is not None and self.maximum_radius_m <= 0:
            raise ValueError("maximum_radius_m must be positive when provided")
        if self.ring_step_m <= 0:
            raise ValueError("ring_step_m must be positive")
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("confidence_threshold must be between zero and one")
        if not 0 < self.decision_threshold < 1:
            raise ValueError("decision_threshold must be between zero and one")
        if self.stability_expansions < 0 or self.stability_tolerance_m < 0:
            raise ValueError("stability settings cannot be negative")
        if self.minimum_peak_margin < 0:
            raise ValueError("minimum_peak_margin cannot be negative")
        if self.match_logit_weight < 0:
            raise ValueError("match_logit_weight cannot be negative")


@dataclass(frozen=True)
class SearchTile:
    """One target tile in row/column coordinates of the analysis grid."""

    tile_id: str
    row_off: int
    col_off: int
    height: int
    width: int
    minimum_distance_m: float
    center_distance_m: float
    ring_index: int


@dataclass(frozen=True)
class SearchRing:
    """A complete distance ring; early stopping is assessed only after it."""

    ring_index: int
    outer_radius_m: float | None
    tiles: tuple[SearchTile, ...]


def plan_search(
    raster_shape_hw: tuple[int, int],
    source_xy_px: tuple[float, float],
    *,
    gsd_xy_m_per_px: tuple[float, float],
    config: SearchConfig,
    aoi_mask: np.ndarray | None = None,
    aoi_mask_stride_px: int = 1,
) -> tuple[SearchRing, ...]:
    """Create an endpoint-blind search plan.

    Full-glacier and progressive modes require an explicit AOI mask.  This is a
    deliberate guard against silently treating every valid orthomosaic pixel as
    glacier terrain.
    """

    height, width = (int(value) for value in raster_shape_hw)
    if min(height, width) <= 0:
        raise ValueError("raster dimensions must be positive")
    gsd_x, gsd_y = (float(value) for value in gsd_xy_m_per_px)
    if min(gsd_x, gsd_y) <= 0:
        raise ValueError("GSD values must be positive")
    if config.mode in {SearchMode.FULL_GLACIER, SearchMode.PROGRESSIVE} and aoi_mask is None:
        raise ValueError(f"{config.mode.value} search requires an explicit glacier AOI mask")
    if aoi_mask_stride_px <= 0:
        raise ValueError("AOI mask stride must be positive")
    mask_index = (
        _MaskIndex(aoi_mask, (height, width), aoi_mask_stride_px)
        if aoi_mask is not None
        else None
    )

    tile_height, tile_width = config.tile_size_px
    rows = _offsets(height, tile_height, config.tile_overlap)
    columns = _offsets(width, tile_width, config.tile_overlap)
    provisional: list[SearchTile] = []
    for row in rows:
        actual_height = min(tile_height, height - row)
        for column in columns:
            actual_width = min(tile_width, width - column)
            if mask_index is not None and not mask_index.any(
                row, column, actual_height, actual_width
            ):
                continue
            minimum = _minimum_distance(
                source_xy_px,
                row,
                column,
                actual_height,
                actual_width,
                gsd_x,
                gsd_y,
            )
            center_x = column + (actual_width - 1) / 2
            center_y = row + (actual_height - 1) / 2
            center = math.hypot(
                (center_x - source_xy_px[0]) * gsd_x,
                (center_y - source_xy_px[1]) * gsd_y,
            )
            ring_index = int(math.floor(minimum / config.ring_step_m))
            provisional.append(
                SearchTile(
                    tile_id=f"r{row:08d}_c{column:08d}",
                    row_off=row,
                    col_off=column,
                    height=actual_height,
                    width=actual_width,
                    minimum_distance_m=minimum,
                    center_distance_m=center,
                    ring_index=ring_index,
                )
            )

    if config.mode is SearchMode.RADIUS:
        selected = [tile for tile in provisional if tile.minimum_distance_m <= config.radius_m]
        selected.sort(key=lambda tile: (tile.center_distance_m, tile.tile_id))
        return (SearchRing(0, config.radius_m, tuple(selected)),)

    if config.mode is SearchMode.PROGRESSIVE and config.maximum_radius_m is not None:
        provisional = [
            tile
            for tile in provisional
            if tile.minimum_distance_m <= config.maximum_radius_m
        ]
    provisional.sort(key=lambda tile: (tile.ring_index, tile.center_distance_m, tile.tile_id))
    if config.mode is SearchMode.FULL_GLACIER:
        return (SearchRing(0, None, tuple(provisional)),)

    groups: dict[int, list[SearchTile]] = {}
    for tile in provisional:
        groups.setdefault(tile.ring_index, []).append(tile)
    return tuple(
        SearchRing(
            ring_index=index,
            outer_radius_m=(index + 1) * config.ring_step_m,
            tiles=tuple(groups[index]),
        )
        for index in sorted(groups)
    )


def iter_tile_batches(
    ring: SearchRing,
    batch_size: int,
) -> tuple[tuple[SearchTile, ...], ...]:
    """Split compute batches without changing ring-level stopping semantics."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return tuple(
        ring.tiles[start : start + batch_size]
        for start in range(0, len(ring.tiles), batch_size)
    )


def _offsets(length: int, tile_length: int, overlap: float) -> tuple[int, ...]:
    if length <= tile_length:
        return (0,)
    step = max(1, int(round(tile_length * (1 - overlap))))
    values = list(range(0, length - tile_length + 1, step))
    final = length - tile_length
    if values[-1] != final:
        values.append(final)
    return tuple(values)


def _minimum_distance(
    point_xy: tuple[float, float],
    row: int,
    column: int,
    height: int,
    width: int,
    gsd_x: float,
    gsd_y: float,
) -> float:
    x, y = point_xy
    right = column + width - 1
    bottom = row + height - 1
    dx = max(column - x, 0.0, x - right) * gsd_x
    dy = max(row - y, 0.0, y - bottom) * gsd_y
    return math.hypot(dx, dy)


class _MaskIndex:
    def __init__(
        self,
        mask: np.ndarray,
        expected_shape: tuple[int, int],
        stride_px: int,
    ) -> None:
        array = np.asarray(mask, dtype=bool)
        expected_mask_shape = tuple(
            math.ceil(value / stride_px) for value in expected_shape
        )
        if array.shape != expected_mask_shape:
            raise ValueError(
                f"AOI mask shape {array.shape} does not match {expected_mask_shape} "
                f"for stride {stride_px}"
            )
        self.stride_px = stride_px
        self.integral = np.pad(array.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def any(self, row: int, column: int, height: int, width: int) -> bool:
        bottom = math.ceil((row + height) / self.stride_px)
        right = math.ceil((column + width) / self.stride_px)
        row = row // self.stride_px
        column = column // self.stride_px
        count = (
            self.integral[bottom, right]
            - self.integral[row, right]
            - self.integral[bottom, column]
            + self.integral[row, column]
        )
        return bool(count)
