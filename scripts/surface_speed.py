"""Calculate Belvedere surface speeds, regime agreement and endpoint-derived errors."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from paired_ablation import errors
from study_variants import BELVEDERE, FOLDS
from terrain_variants import raster_background

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGIMES = ("slow", "intermediate", "fast")
METHOD = "procedure_1_soft_rerank"
DATES = {"2015": date(2015, 10, 15), "2020": date(2020, 7, 27),
         "2023": date(2023, 7, 25)}


def speed_records(root: Path, auxiliary: Path) -> pd.DataFrame:
    frames = []
    for pair in BELVEDERE:
        source_year, target_year = pair.split("_")
        elapsed = (DATES[target_year] - DATES[source_year]).days / 365.25
        truth_file = auxiliary / "inference" / "belvedere_multitemporal_2015" / pair
        truth = pd.read_parquet(truth_file / "evaluation_queries.parquet").set_index("query_id")
        source = truth[["source_x_m", "source_y_m"]].to_numpy(float)
        target = truth[["target_x_m", "target_y_m"]].to_numpy(float)
        reference_speed = np.linalg.norm(target - source, axis=1) / elapsed
        reference_class = np.digitize(reference_speed, [7.0, 17.0])
        for fold, glacier in FOLDS.items():
            folder = root / "reports" / f"fold_{fold:02d}_{glacier}" / f"belvedere_{pair}"
            for method in (METHOD, f"{METHOD}__local7"):
                endpoint_error, predicted = errors(folder, method, truth)
                speed = np.linalg.norm(predicted - source, axis=1) / elapsed
                speed_class = np.digitize(speed, [7.0, 17.0])
                frames.append(pd.DataFrame({
                    "query_id": truth.index, "pair": pair, "fold": fold, "method": method,
                    "source_x_m": source[:, 0], "source_y_m": source[:, 1],
                    "reference_speed_m_per_year": reference_speed,
                    "predicted_speed_m_per_year": speed,
                    "absolute_speed_error_m_per_year": np.abs(speed - reference_speed),
                    "velocity_vector_error_m_per_year": endpoint_error / elapsed,
                    "endpoint_error_m": endpoint_error,
                    "correct_endpoint": endpoint_error <= 2.0,
                    "reference_regime": np.take(REGIMES, reference_class),
                    "predicted_regime": np.take(REGIMES, speed_class),
                    "regime_agrees": speed_class == reference_class,
                    "slow_as_fast": (reference_class == 0) & (speed_class == 2),
                    "fast_as_slow": (reference_class == 2) & (speed_class == 0),
                }))
    return pd.concat(frames, ignore_index=True)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    keys = ["pair", "fold", "method"]
    summary = records.groupby(keys).agg(
        queries=("query_id", "size"),
        correct_endpoints=("correct_endpoint", "sum"),
        regime_agreement=("regime_agrees", "mean"),
        slow_as_fast=("slow_as_fast", "sum"),
        fast_as_slow=("fast_as_slow", "sum"),
        median_speed_error_all_m_per_year=("absolute_speed_error_m_per_year", "median"),
    )
    correct = records.loc[records.correct_endpoint].groupby(keys).agg(
        median_speed_error_correct_m_per_year=("absolute_speed_error_m_per_year", "median"),
        max_speed_error_correct_m_per_year=("absolute_speed_error_m_per_year", "max"),
        median_velocity_vector_error_correct_m_per_year=("velocity_vector_error_m_per_year", "median"),
        max_velocity_vector_error_correct_m_per_year=("velocity_vector_error_m_per_year", "max"),
    )
    return summary.join(correct)


def plot(records: pd.DataFrame, raster: Path, output: Path) -> None:
    local = records.loc[(records.pair == "2020_2023") & records.method.str.endswith("__local7")]
    mapped = local.loc[local.fold == 1]
    image, extent, _ = raster_background(raster)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    east, north = mapped.source_x_m, mapped.source_y_m
    columns = ("reference_speed_m_per_year", "predicted_speed_m_per_year")
    for axis, column, title in zip(axes[:2], columns, ("Manual tracks", "Fold 1 · soft rerank + local7")):
        axis.imshow(image, extent=extent)
        points = axis.scatter(east, north, c=mapped[column], vmin=0, vmax=25, cmap="viridis",
                              edgecolors="white", linewidths=0.4)
        axis.set(xlim=(east.min() - 45, east.max() + 45), ylim=(north.min() - 45, north.max() + 45),
                 title=title, xlabel="Easting (m)", ylabel="Northing (m)")
        axis.ticklabel_format(style="plain", useOffset=False)
    incorrect = mapped.loc[~mapped.correct_endpoint]
    axes[1].scatter(incorrect.source_x_m, incorrect.source_y_m, marker="x", color="white",
                    linewidths=1, label="Endpoint error > 2 m")
    axes[1].legend(loc="upper left", fontsize=7)
    fig.colorbar(points, ax=list(axes[:2]), label="Surface speed (m/yr)", shrink=0.8)
    axis = axes[2]
    for low, high in ((2, 7), (17, 22)):
        axis.axvspan(low, high, color="#eeeeee")
        axis.axhspan(low, high, color="#eeeeee")
    for fold, frame in local.groupby("fold"):
        axis.scatter(frame.reference_speed_m_per_year, frame.predicted_speed_m_per_year,
                     label=f"Fold {fold}", s=16, alpha=0.7)
    upper = max(25, local.predicted_speed_m_per_year.max(), local.reference_speed_m_per_year.max())
    axis.plot([0, upper], [0, upper], color="gray", linewidth=1)
    axis.set(xlim=(0, upper), ylim=(0, upper), aspect="equal", title="Speed agreement",
             xlabel="Manual speed (m/yr)", ylabel="Predicted speed (m/yr)")
    axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--auxiliary-dataset", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    records = speed_records(args.root, args.auxiliary_dataset)
    records.to_csv(output / "surface_speed_by_query.csv", index=False)
    summarize(records).to_csv(output / "surface_speed_summary.csv")
    raster = args.auxiliary_dataset / "raw/belvedere/surveys/survey_2020/orthomosaic.tif"
    plot(records, raster, output / "belvedere_surface_speed.png")


if __name__ == "__main__":
    main()
