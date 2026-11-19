"""Export Belvedere query and scoring tables from harmonized survey point layers."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_points(path: Path) -> gpd.GeoDataFrame:
    points = gpd.read_file(path)
    points = points.loc[points.status == "visible"].to_crs("EPSG:32632")
    if points.track_id.isna().any() or points.track_id.duplicated().any():
        raise ValueError(f"Expected unique visible track IDs in {path}")
    if not points.geom_type.eq("Point").all() or points.geometry.is_empty.any():
        raise ValueError(f"Expected nonempty point geometries in {path}")
    return points.set_index("track_id")


def query_table(source: gpd.GeoDataFrame, target: gpd.GeoDataFrame,
                source_year: int, target_year: int) -> pd.DataFrame:
    ids = source.index.intersection(target.index).sort_values()
    first, second = source.loc[ids], target.loc[ids]
    source_survey, target_survey = f"survey_{source_year}", f"survey_{target_year}"
    return pd.DataFrame({
        "query_id": [f"{track}__{source_year}_{target_year}" for track in ids],
        "track_id": ids.to_numpy(),
        "original_id": first.original_id.to_numpy(),
        "glacier_id": "belvedere",
        "survey_pair": f"{source_survey}--{target_survey}",
        "source_survey_id": source_survey,
        "target_survey_id": target_survey,
        "source_x_m": first.geometry.x.to_numpy(),
        "source_y_m": first.geometry.y.to_numpy(),
        "target_x_m": second.geometry.x.to_numpy(),
        "target_y_m": second.geometry.y.to_numpy(),
        "match_present": True,
        "source_valid": True,
        "source_crs": "EPSG:32632",
        "target_crs": "EPSG:32632",
        "effective_gsd_m_per_px": 0.2,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for year in (2015, 2020, 2023):
        parser.add_argument(f"--points-{year}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = {year: load_points(getattr(args, f"points_{year}")) for year in (2015, 2020, 2023)}
    args.output.mkdir(parents=True, exist_ok=False)
    for source, target in ((2015, 2020), (2015, 2023), (2020, 2023)):
        folder = args.output / f"{source}_{target}"
        folder.mkdir()
        table = query_table(points[source], points[target], source, target)
        table.to_parquet(folder / "evaluation_queries.parquet", index=False)
        table.drop(columns=["target_x_m", "target_y_m", "match_present"]).to_parquet(
            folder / "source_queries.parquet", index=False
        )


if __name__ == "__main__":
    main()
