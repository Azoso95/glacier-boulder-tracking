"""Terrain features and soft physical costs for stitched boulder searches."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from affine import Affine
from torch import Tensor, nn

from glaciertrack.data.augmentation import GeometryChoice


@dataclass(frozen=True)
class TerrainGrid:
    """Terrain features sampled on one stitched search grid."""

    direction_cosine: Tensor
    upstream_cost: Tensor
    elevation_delta_m: Tensor
    uphill_cost: Tensor
    slope_fraction: Tensor
    distance_fraction: Tensor
    confidence: Tensor
    valid: Tensor

    def feature_stack(self) -> Tensor:
        """Return bounded/standardized inputs for a learned terrain head."""

        return torch.stack(
            (
                self.direction_cosine,
                self.upstream_cost,
                torch.clamp(self.elevation_delta_m / 25.0, -2.0, 2.0),
                self.uphill_cost,
                self.slope_fraction,
                self.distance_fraction,
                self.confidence,
            ),
            dim=-1,
        )

    def combined_cost(self, direction_weight: float, elevation_weight: float) -> Tensor:
        """Return a non-negative cost with zero contribution outside valid terrain."""

        return (
            float(direction_weight) * self.upstream_cost
            + float(elevation_weight) * self.uphill_cost
        ) * self.confidence

    def transformed(self, geometry: GeometryChoice) -> TerrainGrid:
        """Apply the image episode's discrete geometry to every terrain layer.

        Terrain values are computed in world coordinates first.  The scalar
        layers are then spatially permuted exactly like the RGB target canvas,
        preserving the physical value attached to every candidate location.
        """

        def apply(value: Tensor) -> Tensor:
            result = torch.rot90(value, geometry.quarter_turns, dims=(-2, -1))
            if geometry.horizontal_flip:
                result = torch.flip(result, dims=(-1,))
            if geometry.vertical_flip:
                result = torch.flip(result, dims=(-2,))
            return result

        return TerrainGrid(
            direction_cosine=apply(self.direction_cosine),
            upstream_cost=apply(self.upstream_cost),
            elevation_delta_m=apply(self.elevation_delta_m),
            uphill_cost=apply(self.uphill_cost),
            slope_fraction=apply(self.slope_fraction),
            distance_fraction=apply(self.distance_fraction),
            confidence=apply(self.confidence),
            valid=apply(self.valid),
        )


class TerrainConditioningHead(nn.Module):
    """Small zero-initialized correction head operating on physical features."""

    def __init__(self, input_features: int = 7, hidden_features: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


class TerrainField:
    """In-memory DEM and broad flow products for one glacier."""

    def __init__(
        self,
        dem_path: str | Path,
        flow_path: str | Path,
        *,
        comparison_flow_path: str | Path | None = None,
    ) -> None:
        self.dem_path = Path(dem_path).expanduser().resolve()
        self.flow_path = Path(flow_path).expanduser().resolve()
        self.comparison_flow_path = (
            None
            if comparison_flow_path is None
            else Path(comparison_flow_path).expanduser().resolve()
        )
        self._read()

    def _read(self) -> None:
        try:
            import rasterio
        except ImportError as error:  # pragma: no cover - geo extra controls availability
            raise ImportError("terrain support requires the geo dependencies") from error

        for path in (self.dem_path, self.flow_path):
            if not path.is_file():
                raise FileNotFoundError(f"terrain raster does not exist: {path}")
        with rasterio.open(self.dem_path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"terrain DEM has no CRS: {self.dem_path}")
            self.crs = dataset.crs
            self.dem_transform = dataset.transform
            self.dem = dataset.read(1, masked=True).filled(np.nan).astype(np.float32)
        with rasterio.open(self.flow_path) as dataset:
            if dataset.crs is None or dataset.crs != self.crs:
                raise ValueError("terrain DEM and flow raster must share a CRS")
            if dataset.count < 3:
                raise ValueError("flow raster must contain slope, north, and east bands")
            self.flow_transform = dataset.transform
            values = dataset.read((1, 2, 3), masked=True).filled(np.nan).astype(np.float32)
            self.slope = values[0]
            self.flow_north = values[1]
            self.flow_east = values[2]
        self.comparison_transform: Affine | None = None
        self.comparison_north: np.ndarray | None = None
        self.comparison_east: np.ndarray | None = None
        if self.comparison_flow_path is not None:
            if not self.comparison_flow_path.is_file():
                raise FileNotFoundError(
                    f"comparison flow raster does not exist: {self.comparison_flow_path}"
                )
            with rasterio.open(self.comparison_flow_path) as dataset:
                if dataset.crs != self.crs or dataset.count < 3:
                    raise ValueError("comparison flow raster is incompatible")
                comparison = (
                    dataset.read((2, 3), masked=True).filled(np.nan).astype(np.float32)
                )
                self.comparison_transform = dataset.transform
                self.comparison_north = comparison[0]
                self.comparison_east = comparison[1]

    def grid(
        self,
        *,
        target_crs: Any,
        target_transform: Affine,
        canvas_origin_xy_px: tuple[int, int],
        shape_hw: tuple[int, int],
        stride_px: int,
        source_xy_m: tuple[float, float],
        radius_m: float,
        uphill_tolerance_m: float = 5.0,
        uphill_scale_m: float = 10.0,
    ) -> TerrainGrid:
        """Sample physical features at the cell centers of a stitched surface."""

        if stride_px <= 0 or radius_m <= 0 or uphill_scale_m <= 0:
            raise ValueError("terrain grid scales must be positive")
        if uphill_tolerance_m < 0:
            raise ValueError("uphill tolerance must be non-negative")
        height, width = shape_hw
        yy, xx = np.indices((height, width), dtype=np.float64)
        left, top = canvas_origin_xy_px
        pixel_x = left + xx * stride_px + (stride_px - 1) / 2 + 0.5
        pixel_y = top + yy * stride_px + (stride_px - 1) / 2 + 0.5
        world_x = target_transform.a * pixel_x + target_transform.b * pixel_y + target_transform.c
        world_y = target_transform.d * pixel_x + target_transform.e * pixel_y + target_transform.f
        source_x, source_y = source_xy_m
        if str(target_crs) != str(self.crs):
            from pyproj import Transformer

            transformer = Transformer.from_crs(target_crs, self.crs, always_xy=True)
            world_x, world_y = transformer.transform(world_x, world_y)
            source_x, source_y = transformer.transform(source_x, source_y)
        source_east = float(self._sample(self.flow_east, self.flow_transform, source_x, source_y))
        source_north = float(
            self._sample(self.flow_north, self.flow_transform, source_x, source_y)
        )
        source_elevation = float(self._sample(self.dem, self.dem_transform, source_x, source_y))
        elevation = self._sample(self.dem, self.dem_transform, world_x, world_y)
        slope = self._sample(self.slope, self.flow_transform, world_x, world_y)

        dx = world_x - source_x
        dy = world_y - source_y
        distance = np.hypot(dx, dy)
        flow_norm = math.hypot(source_east, source_north)
        direction_cosine = np.zeros_like(distance, dtype=np.float32)
        moving = distance > 1e-6
        if math.isfinite(flow_norm) and flow_norm > 1e-6:
            direction_cosine[moving] = (
                (dx[moving] * source_east + dy[moving] * source_north)
                / (distance[moving] * flow_norm)
            ).astype(np.float32)
        direction_cosine = np.clip(direction_cosine, -1.0, 1.0)
        elevation_delta = elevation - source_elevation
        upstream = np.maximum(-direction_cosine, 0.0)
        uphill = np.maximum(elevation_delta - float(uphill_tolerance_m), 0.0) / float(
            uphill_scale_m
        )
        uphill = np.clip(uphill, 0.0, 5.0)
        slope_fraction = np.clip(slope / 45.0, 0.0, 1.0)
        distance_fraction = np.clip(distance / float(radius_m), 0.0, 1.5)
        confidence = np.ones_like(distance, dtype=np.float32)
        confidence *= np.clip(slope / 3.0, 0.0, 1.0).astype(np.float32)
        if self.comparison_north is not None and self.comparison_east is not None:
            comparison_north = self._sample(
                self.comparison_north, self.comparison_transform, world_x, world_y
            )
            comparison_east = self._sample(
                self.comparison_east, self.comparison_transform, world_x, world_y
            )
            flow_north = self._sample(self.flow_north, self.flow_transform, world_x, world_y)
            flow_east = self._sample(self.flow_east, self.flow_transform, world_x, world_y)
            agreement = flow_north * comparison_north + flow_east * comparison_east
            confidence *= np.clip((agreement + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        valid = (
            np.isfinite(elevation)
            & np.isfinite(elevation_delta)
            & np.isfinite(slope)
            & np.isfinite(direction_cosine)
            & np.isfinite(confidence)
            & math.isfinite(source_elevation)
            & math.isfinite(flow_norm)
            & (flow_norm > 1e-6)
        )
        confidence = np.where(valid, confidence, 0.0)

        def tensor(value: np.ndarray) -> Tensor:
            return torch.from_numpy(np.nan_to_num(value, nan=0.0).astype(np.float32))

        return TerrainGrid(
            direction_cosine=tensor(direction_cosine),
            upstream_cost=tensor(upstream),
            elevation_delta_m=tensor(elevation_delta),
            uphill_cost=tensor(uphill),
            slope_fraction=tensor(slope_fraction),
            distance_fraction=tensor(distance_fraction),
            confidence=tensor(confidence),
            valid=torch.from_numpy(valid),
        )

    @staticmethod
    def _sample(
        values: np.ndarray,
        transform: Affine | None,
        x: float | np.ndarray,
        y: float | np.ndarray,
    ) -> np.ndarray:
        if transform is None:
            raise ValueError("terrain transform is unavailable")
        column, row = (~transform) * (x, y)
        column = np.asarray(column, dtype=np.float64) - 0.5
        row = np.asarray(row, dtype=np.float64) - 0.5
        left = np.floor(column).astype(np.int64)
        top = np.floor(row).astype(np.int64)
        right = left + 1
        bottom = top + 1
        valid = (
            (top >= 0)
            & (left >= 0)
            & (bottom < values.shape[0])
            & (right < values.shape[1])
        )
        output = np.full(np.broadcast(column, row).shape, np.nan, dtype=np.float32)
        if not np.any(valid):
            return output
        wx = column - left
        wy = row - top
        samples = (
            values[top[valid], left[valid]] * (1 - wx[valid]) * (1 - wy[valid])
            + values[top[valid], right[valid]] * wx[valid] * (1 - wy[valid])
            + values[bottom[valid], left[valid]] * (1 - wx[valid]) * wy[valid]
            + values[bottom[valid], right[valid]] * wx[valid] * wy[valid]
        )
        output[valid] = samples
        return output
