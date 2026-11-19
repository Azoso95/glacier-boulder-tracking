"""Derive the study terrain layers from the original buffered DEM tiles."""

import argparse
from pathlib import Path

import rasterio

from dem_flow import align_to_orthophoto, clip_dem_to_orthophoto, compute_flow_direction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copernicus", required=True, type=Path)
    parser.add_argument("--srtm", required=True, type=Path)
    parser.add_argument("--reference-orthophoto", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for product, dem in (("copernicus_glo30", args.copernicus), ("srtm_gl1", args.srtm)):
        dem_path = args.output / f"{product}_dem_reproj.tif"
        flow_path = args.output / f"{product}_flowdir.tif"
        if dem_path.exists() or flow_path.exists():
            raise FileExistsError(f"Terrain outputs already exist for {product}")
        slope, south, east, transform, crs = compute_flow_direction(dem, sigma_px=10.0)
        align_to_orthophoto(
            [slope, -south, east], transform, crs, args.reference_orthophoto, flow_path
        )
        clip_dem_to_orthophoto(dem, args.reference_orthophoto, dem_path)
        with rasterio.open(flow_path, "r+") as raster:
            for band, name in enumerate(("macro_slope_degrees", "flow_north_unit",
                                         "flow_east_unit"), start=1):
                raster.set_band_description(band, name)


if __name__ == "__main__":
    main()
