# Glacier boulder tracking

Code for *Glacier surface velocity from supraglacial boulder tracking: can a neural network replace manual tracking on debris-covered glaciers?* The study trains two ConvNeXt–SAM2 models with Gorner and Zmutt held out in turn, tests both models on three Belvedere pairs, and compares terrain reranking, physics-loss fine-tuning, its zero-physics control, a learned terrain head, local7, and four frozen matchers.

## Setup

Use a fresh Python 3.11+ environment with CUDA-enabled PyTorch and matching torchvision. [PyTorch installation](https://pytorch.org/get-started/locally/)

```sh
python -m pip install -e .
python -m pip install "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

Download the study's SAM2 checkpoint. [SAM2.1 Hiera Small](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)

First use also fetches ImageNet and frozen-matcher weights. Exact reproduction requires the original study orthophotos, AOIs, harmonized point annotations and DEM inputs.

## Inputs

| Argument | Required layout |
| --- | --- |
| `--training-dataset` | `raw/<glacier>/surveys/survey_<year>/` with the orthophoto, `survey.yaml` and AOI. Metadata specifies `acquisition_year`, `raster` and `aoi`. The mirrored `tracks/<glacier>/surveys/survey_<year>/observations.gpkg` contains point geometry, `track_id` and `status`. Use the corrected Gorner annotations. |
| `--auxiliary-dataset` | `raw/belvedere/surveys/survey_<year>/` with `orthomosaic.tif` and `coverage_aoi.gpkg` for target years. Place the three pair folders under `inference/belvedere_multitemporal_2015/`, each with `source_queries.parquet` and `evaluation_queries.parquet`. Belvedere orthophotos are available from [Ioli et al. (2024)](https://doi.org/10.5281/zenodo.10817029). |
| Terrain | Under the auxiliary root, `prepared/terrain/<glacier>/` contains `copernicus_glo30_dem_reproj.tif`, `copernicus_glo30_flowdir.tif` and `srtm_gl1_flowdir.tif` for all five glaciers. |

`scripts/prepare_belvedere.py` builds the pair tables from the three harmonized Belvedere point layers. `scripts/prepare_terrain.py` derives the terrain products from the original buffered Copernicus GLO-30 and SRTMGL1 tiles and the reference orthophoto. Run either script with `--help` for arguments.

## Run

Run from the repository root with an empty output directory.

```sh
python scripts/run_study.py --training-dataset inputs/training --auxiliary-dataset inputs/auxiliary --sam2-checkpoint models/sam2.1_hiera_small.pt --output outputs/study
```

Add `--check-inputs` to verify paths before training, or `--prepared PATH` to use the study's existing patch cache and manifests. The runner prepares the six training survey pairs, builds both spatial holdouts, trains patch and search models, and saves predictions before scoring. The held-out radius is 160 m. Belvedere radii are 75, 125 and 200 m for 2020–2023, 2015–2020 and 2015–2023.

Resolved experimental settings are in `configs/`. Checkpoints are under `runs/` in the output directory. The `reports/`, `benchmarks/` and `summary/` directories contain endpoint predictions, coverage and error metrics, displacement maps, and Belvedere speed and flow-regime analysis.
