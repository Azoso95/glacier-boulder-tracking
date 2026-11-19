"""Summarize frozen matcher results on held-out glaciers and Belvedere pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from survey_paths import survey_raster

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("ncc_gray", "ncc_gradient", "sift", "disk_lightglue")
PAIRS = ("gorner_test", "zmutt_test", "belvedere_2020_2023",
         "belvedere_2015_2020", "belvedere_2015_2023")
LABELS = ("Grey NCC", "Gradient NCC", "SIFT", "DISK–LightGlue")


def movement_map(root: Path, training: Path, auxiliary: Path, pair: str, method: str) -> None:
    if pair.endswith("_test"):
        glacier = pair.removesuffix("_test")
        fold = 0 if glacier == "gorner" else 1
        queries = (root / "folds" / f"fold_{fold:02d}_{glacier}"
                   / f"test__{glacier}.parquet")
        raster = survey_raster(training / "raw" / glacier / "surveys" / "survey_2005")
    else:
        years = pair.removeprefix("belvedere_")
        target = years.split("_")[1]
        queries = (auxiliary / "inference" / "belvedere_multitemporal_2015"
                   / years / "evaluation_queries.parquet")
        raster = (auxiliary / "raw" / "belvedere" / "surveys"
                  / f"survey_{target}" / "orthomosaic.tif")
    folder = root / "benchmarks" / pair / method
    truth = pd.read_parquet(queries).set_index("query_id")
    predictions = json.loads((folder / f"{method}_predictions.json").read_text(encoding="utf-8"))
    with rasterio.open(raster) as dataset:
        height = max(1, round(dataset.height * min(1, 2600 / max(dataset.width, dataset.height))))
        width = max(1, round(dataset.width * min(1, 2600 / max(dataset.width, dataset.height))))
        if dataset.count >= 3:
            image = np.moveaxis(dataset.read((1, 2, 3), out_shape=(3, height, width),
                                             resampling=Resampling.bilinear), 0, -1).astype(float)
        else:
            gray = dataset.read(1, out_shape=(height, width),
                                resampling=Resampling.bilinear).astype(float)
            image = np.repeat(gray[..., None], 3, axis=2)
        bounds = dataset.bounds
    lower, upper = np.nanpercentile(image, (2, 98))
    image = np.clip((image - lower) / max(upper - lower, 1e-9), 0, 1)
    figure, axis = plt.subplots(figsize=(12, 11), constrained_layout=True)
    axis.imshow(image, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
    predicted_points = []
    predicted_label_used = False
    for index, record in enumerate(predictions):
        row = truth.loc[record["query_id"]]
        origin = np.array((row.source_x_m, row.source_y_m), dtype=float)
        target = np.array((row.target_x_m, row.target_y_m), dtype=float)
        axis.plot((origin[0], target[0]), (origin[1], target[1]), color="#2166AC",
                  linewidth=1.2, alpha=0.8, label="Reference" if index == 0 else None)
        if record["prediction_xy_m"] is not None:
            predicted = np.asarray(record["prediction_xy_m"], dtype=float)
            predicted_points.append(predicted)
            axis.plot((origin[0], predicted[0]), (origin[1], predicted[1]),
                      color="#D62728", linewidth=1.2, alpha=0.8,
                      label="Prediction" if not predicted_label_used else None)
            predicted_label_used = True
        axis.plot(origin[0], origin[1], ".", color="#111111", markersize=2)
    x = np.r_[truth.source_x_m.to_numpy(float), truth.target_x_m.to_numpy(float)]
    y = np.r_[truth.source_y_m.to_numpy(float), truth.target_y_m.to_numpy(float)]
    if predicted_points:
        predicted_array = np.stack(predicted_points)
        x = np.r_[x, predicted_array[:, 0]]
        y = np.r_[y, predicted_array[:, 1]]
    axis.set_xlim(x.min() - 70, x.max() + 70)
    axis.set_ylim(y.min() - 70, y.max() + 70)
    axis.set_title(f"{pair.replace('_', ' ')} · {method}: red prediction, blue reference")
    axis.set(xlabel="Easting (m)", ylabel="Northing (m)")
    axis.legend(loc="best")
    figure.savefig(folder / "movement_map.png", dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--auxiliary-dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    auxiliary = args.auxiliary_dataset.resolve()
    methods = METHODS
    labels = [dict(zip(METHODS, LABELS))[method] for method in methods]
    rows = []
    for pair in PAIRS:
        for method in methods:
            path = root / "benchmarks" / pair / method / "result.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            metric = payload["metrics"][method]
            rows.append({"pair": pair, "method": method,
                         "radius_m": payload["radius_m"],
                         "analysis_gsd_m_per_px": payload["gsd_m_per_px"], **metric})
            movement_map(root, args.training_dataset.resolve(), auxiliary, pair, method)
    frame = pd.DataFrame(rows)
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "frozen_benchmarks.csv", index=False)
    (output / "frozen_benchmarks.json").write_text(
        frame.to_json(orient="records", indent=2), encoding="utf-8")
    for metric, label, filename in (
        ("success_at_2m_all_queries", "Within 2 m (%)", "frozen_success_2m.png"),
        ("coverage", "Returned endpoint (%)", "frozen_coverage.png"),
    ):
        figure, axes = plt.subplots(1, len(PAIRS), figsize=(20, 5),
                                    constrained_layout=True, sharey=True)
        for axis, pair in zip(axes, PAIRS):
            subset = frame.loc[frame.pair == pair].set_index("method")
            values = np.array([subset.loc[name, metric] for name in methods]) * 100
            palette = dict(zip(METHODS, ("#0072B2", "#56B4E9", "#999999", "#CC79A7")))
            axis.bar(np.arange(len(methods)), values, color=[palette[m] for m in methods])
            axis.set_xticks(np.arange(len(methods)), labels, rotation=45, ha="right")
            axis.set_title(pair.replace("_", " ").title() +
                           f"\nn={int(subset.iloc[0]['queries'])}", fontsize=10)
            axis.set_ylim(0, 100)
            axis.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel(label)
        figure.savefig(output / filename, dpi=170)
        plt.close(figure)
    print(frame[["pair", "method", "queries", "coverage",
                 "success_at_2m_all_queries"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
