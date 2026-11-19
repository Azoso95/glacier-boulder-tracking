"""Audit broad-flow and uphill-proxy violations in matched study predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

FOLDS = {0: "gorner", 1: "zmutt"}
PAIRS = ("heldout_test", "2020_2023", "2015_2020", "2015_2023")
METHODS = {
    "image": "predictions/control_image_only.json",
    "soft": "predictions/procedure_1_soft_rerank.json",
    "physics": "predictions/procedure_2_physics_loss.json",
    "zero_physics": "zero_physics_predictions.json",
    "head": "predictions/procedure_3_terrain_head.json",
}


def diagnostic(row: pd.Series, xy: list[float], field: object) -> dict[str, float]:
    source = np.asarray(
        Transformer.from_crs(row.source_crs, field.crs, always_xy=True).transform(
            float(row.source_x_m), float(row.source_y_m)
        ), dtype=float
    )
    target = np.asarray(
        Transformer.from_crs(row.target_crs, field.crs, always_xy=True).transform(
            float(xy[0]), float(xy[1])
        ), dtype=float
    )
    displacement = target - source
    east = float(field._sample(field.flow_east, field.flow_transform, *source))
    north = float(field._sample(field.flow_north, field.flow_transform, *source))
    flow = np.asarray((east, north))
    denominator = float(np.linalg.norm(displacement) * np.linalg.norm(flow))
    cosine = float(np.clip(np.dot(displacement, flow) / denominator, -1, 1)) if denominator > 1e-8 else math.nan
    source_z = float(field._sample(field.dem, field.dem_transform, *source))
    target_z = float(field._sample(field.dem, field.dem_transform, *target))
    return {"flow_cosine": cosine, "elevation_delta_m": target_z - source_z}


def summarize(rows: list[dict]) -> dict:
    cosines = np.asarray([row["flow_cosine"] for row in rows], dtype=float)
    elevation = np.asarray([row["elevation_delta_m"] for row in rows], dtype=float)
    valid = np.isfinite(cosines) & np.isfinite(elevation)
    return {
        "n": len(rows), "valid_n": int(valid.sum()),
        "upstream_n": int(np.sum(cosines[valid] < 0)),
        "uphill_over_5m_n": int(np.sum(elevation[valid] > 5)),
        "median_flow_cosine": float(np.median(cosines[valid])) if valid.any() else None,
        "median_elevation_delta_m": float(np.median(elevation[valid])) if valid.any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--auxiliary-dataset", type=Path,
                        help="Location of terrain and Belvedere query tables")
    args = parser.parse_args()
    root = args.root.resolve()
    auxiliary_dataset = (args.auxiliary_dataset or root / "dataset").resolve()
    from study_variants import evaluation_spec, paths, terrain_fields

    fields = terrain_fields(auxiliary_dataset)
    report = {}
    for fold, heldout in FOLDS.items():
        p = paths(root, fold)
        for pair in PAIRS:
            truth_path, glacier, *_ = evaluation_spec(
                auxiliary_dataset if pair != "heldout_test" else args.training_dataset.resolve(),
                p["fold"], heldout, pair
            )
            truth = pd.read_parquet(truth_path).set_index("query_id")
            folder = p["study"] / ("test" if pair == "heldout_test"
                                    else f"belvedere_{pair}")
            field = fields[glacier]
            entry = {
                "truth": summarize([
                    diagnostic(row, [row.target_x_m, row.target_y_m], field)
                    for _, row in truth.iterrows()
                ])
            }
            for method, relative in METHODS.items():
                predictions = json.loads((folder / relative).read_text(encoding="utf-8"))
                if {row["query_id"] for row in predictions} != set(truth.index):
                    raise ValueError(f"prediction IDs differ from truth: {fold}/{pair}/{method}")
                entry[method] = summarize([
                    diagnostic(truth.loc[row["query_id"]], row["prediction_xy_m"], field)
                    for row in predictions
                ])
            report[f"fold_{fold}_{pair}"] = entry
    output = root / "summary" / "physical_violation_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(output)
    for key, values in report.items():
        print(key, {name: (stats["upstream_n"], stats["uphill_over_5m_n"])
                    for name, stats in values.items()})


if __name__ == "__main__":
    main()
