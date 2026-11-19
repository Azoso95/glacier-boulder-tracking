"""Reproduce the two held-out folds and the three Belvedere transfers."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

import run_benchmarks as benchmarks
import study_variants as study
from build_folds import build_folds
from glaciertrack.config import load_yaml
from glaciertrack.data.preparation import prepare_dataset
from glaciertrack.device import resolve_device
from glaciertrack.search_training import search_training_settings, train_search_from_config
from glaciertrack.training import train_from_config
from survey_paths import survey_raster

SCRIPTS = Path(__file__).resolve().parent
PREPARATION = SCRIPTS.parent / "configs" / "prepare.yaml"


def check_inputs(training: Path, auxiliary: Path, sam2: Path) -> None:
    """Check the input layout without loading endpoints or imagery into memory."""
    required = [sam2]
    for glacier, years in {
        "miage": (2022, 2024),
        "ciardoney": (2019, 2023, 2026),
        "gorner": (1999, 2005),
        "zmutt": (1999, 2005),
    }.items():
        for year in years:
            survey = Path(glacier) / "surveys" / f"survey_{year}"
            required.append(survey_raster(training / "raw" / survey))
            required.append(training / "tracks" / survey / "observations.gpkg")
        if glacier in study.FOLDS.values():
            required.append(training / "raw" / survey / "aoi.gpkg")
    for pair in study.BELVEDERE:
        pair_dir = auxiliary / "inference" / "belvedere_multitemporal_2015" / pair
        required.extend(pair_dir / filename for filename in
                        ("source_queries.parquet", "evaluation_queries.parquet"))
    for year in (2015, 2020, 2023):
        survey = auxiliary / "raw" / "belvedere" / "surveys" / f"survey_{year}"
        required.append(survey / "orthomosaic.tif")
        if year != 2015:
            required.append(survey / "coverage_aoi.gpkg")
    for glacier in ("miage", "ciardoney", "gorner", "zmutt", "belvedere"):
        terrain = auxiliary / "prepared" / "terrain" / glacier
        required.extend(terrain / filename for filename in (
            "copernicus_glo30_dem_reproj.tif", "copernicus_glo30_flowdir.tif",
            "srtm_gl1_flowdir.tif",
        ))
    missing = sorted({path for path in required if not path.is_file()})
    if missing:
        raise FileNotFoundError("\n".join(str(path) for path in missing))


def analyze(output: Path, training: Path, auxiliary: Path) -> None:
    for filename, inputs in (
        ("summary.py", ()),
        ("paired_ablation.py", ("--auxiliary-dataset", auxiliary)),
        ("terrain_audit.py", ("--training-dataset", training, "--auxiliary-dataset", auxiliary)),
        ("benchmark_summary.py", ("--training-dataset", training, "--auxiliary-dataset", auxiliary)),
        ("plot_belvedere_comparison.py", ()),
        ("surface_speed.py", ("--auxiliary-dataset", auxiliary)),
    ):
        subprocess.run(
            [sys.executable, str(SCRIPTS / filename), "--root", str(output), *map(str, inputs)],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--auxiliary-dataset", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, help="Reuse existing study manifests and patch HDF5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    training = args.training_dataset.resolve()
    auxiliary = args.auxiliary_dataset.resolve()
    sam2, output = args.sam2_checkpoint.resolve(), args.output.resolve()
    check_inputs(training, auxiliary, sam2)
    if args.check_inputs:
        print("Study input paths verified")
        return

    device = resolve_device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    prepared = args.prepared.resolve() if args.prepared else output / "prepared"
    if args.prepared is None:
        prepare_dataset(training / "raw", config=PREPARATION, output_root=prepared)
    build_folds(prepared, output / "folds", output / "runs", sam2)

    fields = study.terrain_fields(auxiliary)
    torch.set_num_threads(2)
    for fold, heldout in study.FOLDS.items():
        paths = study.paths(output, fold)
        base, search = (paths["fold"] / name for name in ("base.yaml", "search.yaml"))
        patch = output / "runs" / f"fold_{fold:02d}_{heldout}_patch" / "best.pt"
        train_from_config(base, device=str(device))
        train_search_from_config(search, initialize_from=patch, device=str(device))
        settings = search_training_settings(load_yaml(search))
        study.train_terrain(fields, output, fold, settings, device)
        study.train_terrain(fields, output, fold, settings, device, control=True)
        for pair in ("heldout_test", *study.BELVEDERE):
            for extra in ("", "zero_control"):
                study.evaluate(fields, output, training, fold, pair, settings, device,
                               extra, auxiliary)

    os.environ.setdefault("TORCH_HOME", str(output / "benchmark_cache"))
    models = {"disk_lightglue": benchmarks.load_disk_lightglue(device)}
    for pair in benchmarks.PAIRS:
        for method in benchmarks.METHODS:
            benchmarks.run(output, training, auxiliary, pair, method, device, models)
    analyze(output, training, auxiliary)


if __name__ == "__main__":
    main()
