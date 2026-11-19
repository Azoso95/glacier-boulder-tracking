"""GeoTIFF access on a common metric analysis grid."""

from __future__ import annotations

import math
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import torch
from torch import Tensor

from glaciertrack.data.aoi import rasterize_aoi
from glaciertrack.inference.search import SearchTile


class AnalysisRaster:
    """A lazily opened GeoTIFF/COG resampled to the configured tensor GSD."""

    def __init__(
        self,
        raster_uri: str | Path,
        *,
        tensor_gsd_m_per_px: float,
        bands: tuple[int, ...] = (1, 2, 3),
        radiometric_range: tuple[float, float] | None = None,
    ) -> None:
        if tensor_gsd_m_per_px <= 0:
            raise ValueError("tensor GSD must be positive")
        self.raster_uri = str(Path(raster_uri))
        self.tensor_gsd_m_per_px = float(tensor_gsd_m_per_px)
        self.bands = tuple(int(value) for value in bands)
        self.radiometric_range = radiometric_range
        self._source: Any = None
        self._vrt: Any = None

    def __enter__(self) -> AnalysisRaster:
        try:
            import rasterio
            from rasterio.transform import from_origin
            from rasterio.vrt import WarpedVRT
        except ImportError as error:
            raise ImportError(
                "raster inference requires the geo dependencies; install with "
                "pip install 'glacier-boulder-tracker[geo]'"
            ) from error
        self._source = rasterio.open(self.raster_uri)
        if self._source.crs is None or not self._source.crs.is_projected:
            self.close()
            raise ValueError(f"target raster requires a projected CRS: {self.raster_uri}")
        if max(self.bands) > self._source.count:
            self.close()
            raise ValueError(f"requested bands exceed raster band count: {self.raster_uri}")
        bounds = self._source.bounds
        self.width = int(math.ceil((bounds.right - bounds.left) / self.tensor_gsd_m_per_px))
        self.height = int(math.ceil((bounds.top - bounds.bottom) / self.tensor_gsd_m_per_px))
        self.transform = from_origin(
            bounds.left,
            bounds.top,
            self.tensor_gsd_m_per_px,
            self.tensor_gsd_m_per_px,
        )
        native_gsd = max(
            math.hypot(self._source.transform.a, self._source.transform.d),
            math.hypot(self._source.transform.b, self._source.transform.e),
        )
        resampling = (
            rasterio.enums.Resampling.average
            if native_gsd < self.tensor_gsd_m_per_px
            else rasterio.enums.Resampling.bilinear
        )
        self._vrt = WarpedVRT(
            self._source,
            crs=self._source.crs,
            transform=self.transform,
            width=self.width,
            height=self.height,
            resampling=resampling,
        )
        self.crs = self._source.crs
        self._scale_min, self._scale_max = self._resolve_radiometric_range()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._vrt is not None:
            self._vrt.close()
            self._vrt = None
        if self._source is not None:
            self._source.close()
            self._source = None

    @property
    def shape_hw(self) -> tuple[int, int]:
        self._require_open()
        return self.height, self.width

    def read_tiles(self, tiles: tuple[SearchTile, ...] | list[SearchTile]) -> Tensor:
        """Read same-sized analysis tiles as a ``B x C x H x W`` float tensor."""

        self._require_open()
        arrays = [
            self._read_window(tile.row_off, tile.col_off, tile.height, tile.width)
            for tile in tiles
        ]
        if not arrays:
            return torch.empty((0, len(self.bands), 0, 0), dtype=torch.float32)
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1:
            raise ValueError("tiles in one model batch must have equal dimensions")
        return torch.from_numpy(np.stack(arrays)).to(torch.float32)

    def read_window(self, row: int, column: int, height: int, width: int) -> Tensor:
        """Read one in-bounds analysis-grid RGB window as ``C x H x W``."""

        self._require_open()
        return torch.from_numpy(self._read_window(row, column, height, width)).to(
            torch.float32
        )

    def read_centered_reference(
        self,
        point_xy_m: tuple[float, float],
        size_hw: tuple[int, int],
    ) -> tuple[Tensor, tuple[float, float]]:
        """Extract one centered reference patch and return its local point coordinate."""

        self._require_open()
        height, width = size_hw
        point_x, point_y = self.xy_to_pixel(*point_xy_m)
        column = int(math.floor(point_x - width / 2))
        row = int(math.floor(point_y - height / 2))
        if row < 0 or column < 0 or row + height > self.height or column + width > self.width:
            raise ValueError("reference patch crosses the source raster boundary")
        image = torch.from_numpy(self._read_window(row, column, height, width)).to(torch.float32)
        return image, (point_x - column, point_y - row)

    def xy_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        self._require_open()
        column, row = (~self.transform) * (x_m, y_m)
        return float(column), float(row)

    def pixel_to_xy(self, x_px: float, y_px: float) -> tuple[float, float]:
        self._require_open()
        x_m, y_m = self.transform * (x_px + 0.5, y_px + 0.5)
        return float(x_m), float(y_m)

    def aoi_mask(self, aoi_uri: str | Path, *, stride_px: int = 1) -> np.ndarray:
        self._require_open()
        if stride_px <= 0:
            raise ValueError("AOI mask stride must be positive")
        from affine import Affine

        return rasterize_aoi(
            aoi_uri,
            target_crs=self.crs,
            target_transform=self.transform * Affine.scale(stride_px),
            target_shape_hw=(
                math.ceil(self.height / stride_px),
                math.ceil(self.width / stride_px),
            ),
            all_touched=stride_px > 1,
        )

    def _read_window(self, row: int, column: int, height: int, width: int) -> np.ndarray:
        from rasterio.windows import Window

        if row < 0 or column < 0 or row + height > self.height or column + width > self.width:
            raise ValueError("analysis window lies outside the raster grid")
        array = self._vrt.read(
            indexes=self.bands,
            window=Window(column, row, width, height),
            out_shape=(len(self.bands), height, width),
        )
        scaled = np.clip(
            (array.astype(np.float32) - self._scale_min) / (self._scale_max - self._scale_min),
            0,
            1,
        )
        return scaled

    def _resolve_radiometric_range(self) -> tuple[float, float]:
        if self.radiometric_range is not None:
            minimum, maximum = (float(value) for value in self.radiometric_range)
        else:
            dtype = np.dtype(self._source.dtypes[self.bands[0] - 1])
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                minimum, maximum = float(info.min), float(info.max)
            elif np.issubdtype(dtype, np.floating):
                minimum, maximum = 0.0, 1.0
            else:
                raise TypeError(f"unsupported raster dtype: {dtype}")
        if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
            raise ValueError("radiometric range must contain increasing finite values")
        return minimum, maximum

    def _require_open(self) -> None:
        if self._vrt is None:
            raise RuntimeError("AnalysisRaster must be used as a context manager")
