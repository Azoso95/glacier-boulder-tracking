"""Rasterization of explicit glacier areas of interest for operational search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def rasterize_aoi(
    aoi_uri: str | Path,
    *,
    target_crs: Any,
    target_transform: Any,
    target_shape_hw: tuple[int, int],
    all_touched: bool = False,
) -> np.ndarray:
    """Read polygon AOI data, reproject it, and return a boolean target-grid mask."""

    try:
        import geopandas
        from rasterio.features import rasterize
    except ImportError as error:
        raise ImportError(
            "AOI rasterization requires the geo dependencies; install with "
            "pip install 'glacier-boulder-tracker[geo]'"
        ) from error
    path = Path(aoi_uri)
    frame = (
        geopandas.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else geopandas.read_file(path)
    )
    if frame.crs is None:
        raise ValueError(f"AOI has no CRS: {path}")
    frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty]
    if frame.empty:
        raise ValueError(f"AOI contains no usable geometry: {path}")
    if str(frame.crs) != str(target_crs):
        frame = frame.to_crs(target_crs)
    height, width = (int(value) for value in target_shape_hw)
    mask = rasterize(
        ((geometry, 1) for geometry in frame.geometry),
        out_shape=(height, width),
        transform=target_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=all_touched,
    )
    return mask.astype(bool, copy=False)


def aoi_mask_for_raster(
    aoi_uri: str | Path,
    raster_uri: str | Path,
    *,
    all_touched: bool = False,
) -> np.ndarray:
    """Rasterize an AOI on the native grid of a GeoTIFF/COG."""

    try:
        import rasterio
    except ImportError as error:
        raise ImportError(
            "AOI rasterization requires rasterio; install with "
            "pip install 'glacier-boulder-tracker[geo]'"
        ) from error
    with rasterio.open(raster_uri) as dataset:
        return rasterize_aoi(
            aoi_uri,
            target_crs=dataset.crs,
            target_transform=dataset.transform,
            target_shape_hw=(dataset.height, dataset.width),
            all_touched=all_touched,
        )
