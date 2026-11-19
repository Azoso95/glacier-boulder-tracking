"""Frozen correspondence benchmarks for corrected two-fold tests and Belvedere."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import frozen_benchmarks as bench
from survey_paths import survey_raster


METHODS = ("ncc_gray", "ncc_gradient", "sift", "disk_lightglue")
PAIRS = ("gorner_test", "zmutt_test", "belvedere_2020_2023",
         "belvedere_2015_2020", "belvedere_2015_2023")
BELVEDERE = {
    "belvedere_2020_2023": ("2020", "2023", 75.0),
    "belvedere_2015_2020": ("2015", "2020", 125.0),
    "belvedere_2015_2023": ("2015", "2023", 200.0),
}


def specification(root: Path, training: Path, auxiliary: Path, name: str):
    if name.endswith("_test"):
        glacier = name.removesuffix("_test")
        fold = 0 if glacier == "gorner" else 1
        surveys = training / "raw" / glacier / "surveys"
        queries = (root / "folds" / f"fold_{fold:02d}_{glacier}"
                   / f"test__{glacier}.parquet")
        return bench.Pair(name, survey_raster(surveys / "survey_1999"),
                          survey_raster(surveys / "survey_2005"),
                          surveys / "survey_2005" / "aoi.gpkg",
                          queries.with_name(f"source__{glacier}.parquet"), queries, 160.0)
    source, target, radius = BELVEDERE[name]
    surveys = auxiliary / "raw" / "belvedere" / "surveys"
    queries = (auxiliary / "inference" / "belvedere_multitemporal_2015"
               / f"{source}_{target}" / "evaluation_queries.parquet")
    return bench.Pair(name, surveys / f"survey_{source}" / "orthomosaic.tif",
                      surveys / f"survey_{target}" / "orthomosaic.tif",
                      surveys / f"survey_{target}" / "coverage_aoi.gpkg",
                      queries.with_name("source_queries.parquet"), queries, radius)


def run(root: Path, training: Path, auxiliary: Path, name: str, method: str,
        device: torch.device, models: dict) -> None:
    pair = specification(root, training, auxiliary, name)
    output = root / "benchmarks" / name / method
    result = bench.infer_pair(pair, methods=(method,), models=models,
                              device=device, output=output, limit=None)
    print(json.dumps({name: result}, indent=2), flush=True)


def load_disk_lightglue(device: torch.device) -> tuple:
    from kornia.feature import DISK, LightGlueMatcher

    return (DISK.from_pretrained("depth").eval().to(device),
            LightGlueMatcher(feature_name="disk").eval().to(device))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--auxiliary-dataset", type=Path, required=True)
    parser.add_argument("--pair", choices=PAIRS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    auxiliary = args.auxiliary_dataset.resolve()
    os.environ.setdefault("TORCH_HOME", str(root / "benchmark_cache"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {"disk_lightglue": load_disk_lightglue(device)} if args.method == "disk_lightglue" else {}
    run(root, args.training_dataset.resolve(), auxiliary, args.pair, args.method, device, models)


if __name__ == "__main__":
    main()
