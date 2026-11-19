"""Run matched two-fold image/terrain ablations and Belvedere transfer tests."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from adaptive_query_panels import plot_query_panels
from glaciertrack.terrain import TerrainField
from survey_paths import survey_raster

import terrain_variants as terrain

FOLDS = {0: "gorner", 1: "zmutt"}
BELVEDERE = {
    "2020_2023": ("survey_2020", "survey_2023", 75.0),
    "2015_2020": ("survey_2015", "survey_2020", 125.0),
    "2015_2023": ("survey_2015", "survey_2023", 200.0),
}


def terrain_fields(dataset: Path) -> dict[str, TerrainField]:
    root = dataset / "prepared" / "terrain"
    return {
        glacier: TerrainField(
            root / glacier / "copernicus_glo30_dem_reproj.tif",
            root / glacier / "copernicus_glo30_flowdir.tif",
            comparison_flow_path=root / glacier / "srtm_gl1_flowdir.tif",
        )
        for glacier in ("miage", "ciardoney", "gorner", "zmutt", "belvedere")
    }


def paths(root: Path, fold: int) -> dict[str, Path]:
    name = FOLDS[fold]
    return {
        "fold": root / "folds" / f"fold_{fold:02d}_{name}",
        "study": root / "reports" / f"fold_{fold:02d}_{name}",
        "search": root / "runs" / f"fold_{fold:02d}_{name}_search" / "best.pt",
    }


def pair_settings(settings, dataset: Path, query: Path, glacier: str,
                  source: str, target: str, aoi: str, radius: float):
    survey = dataset / "raw" / glacier / "surveys"
    return replace(
        settings, train_queries=query, validation_queries=query,
        source_raster=survey_raster(survey / source),
        target_raster=survey_raster(survey / target),
        target_aoi=survey / target / aoi,
        radius_m=radius, additional_pairs=(),
    )


def source_only(source: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    terrain.source_only(pd.read_parquet(source)).to_parquet(output, index=False)
    return output


def validation_protocol(settings, fold: Path, heldout: str) -> dict:
    actual = {(settings.train_queries.resolve(), settings.validation_queries.resolve())}
    actual.update((pair["train_queries"].resolve(), pair["validation_queries"].resolve())
                  for pair in settings.additional_pairs)
    pair_files = sorted(fold.glob("train__*--*.parquet"))
    expected = {
        (path.resolve(), (fold / path.name.replace("train__", "validation__", 1)).resolve())
        for path in pair_files
    } if pair_files else {
        ((fold / f"train__{name}.parquet").resolve(),
         (fold / f"validation__{name}.parquet").resolve())
        for name in ("miage", "ciardoney", "gorner", "zmutt")
    }
    if actual != expected:
        raise ValueError("search config does not match this four-glacier fold")
    counts = {}
    for glacier in ("miage", "ciardoney", "gorner", "zmutt"):
        frames = {phase: pd.read_parquet(fold / f"{phase}__{glacier}.parquet")
                  for phase in ("train", "validation", "test")}
        ids = {phase: set(frame.query_id.astype(str)) for phase, frame in frames.items()}
        if any(ids[a] & ids[b] for a, b in (("train", "validation"), ("train", "test"),
                                             ("validation", "test"))):
            raise ValueError(f"query overlap in {glacier}")
        counts[glacier] = {phase: len(frame) for phase, frame in frames.items()}
    if counts[heldout]["train"] or not counts[heldout]["test"]:
        raise ValueError("invalid heldout assignment")
    return counts


def evaluation_spec(dataset: Path, fold: Path, heldout: str, pair: str):
    if pair == "heldout_test":
        full = fold / f"test__{heldout}.parquet"
        return full, heldout, "survey_1999", "survey_2005", "aoi.gpkg", 160.0
    source, target, radius = BELVEDERE[pair]
    full = dataset / "inference" / "belvedere_multitemporal_2015" / pair / "evaluation_queries.parquet"
    return full, "belvedere", source, target, "coverage_aoi.gpkg", radius


def train_terrain(fields, root: Path, fold: int,
                  settings, device: torch.device, control: bool = False) -> None:
    p = paths(root, fold)
    heldout = FOLDS[fold]
    counts = validation_protocol(settings, p["fold"], heldout)
    protocol = terrain.Protocol(radius_m=160.0)
    study = p["study"]
    study.mkdir(parents=True, exist_ok=True)
    terrain.dump(study / "protocol.json", {
        "heldout": heldout, "counts": counts, "protocol": asdict(protocol),
        "search_checkpoint_sha256": terrain.digest(p["search"]),
        "config_sha256": terrain.digest(p["fold"] / "search.yaml"),
    })
    caches = study / "cache"
    if not control:
        terrain.cache_surfaces(
            cache=caches / "base_train", settings=settings, phase="train",
            checkpoint=p["search"], fields=fields, protocol=protocol,
            device=device, augmentation_repeats=protocol.head_augmentation_repeats,
        )
        terrain.cache_surfaces(
            cache=caches / "base_validation", settings=settings, phase="validation",
            checkpoint=p["search"], fields=fields, protocol=protocol, device=device,
        )
        selected = terrain.tune_soft_reranker(
            caches / "base_validation", study / "soft_rerank_selection.json"
        )
        selected["selection_data"] = f"{heldout} spatial validation only"
        terrain.dump(study / "soft_rerank_selection.json", selected)
        terrain.train_physics_loss(
            settings=settings, checkpoint=p["search"], fields=fields,
            protocol=protocol, selected=selected["selected"],
            run_dir=study / "training" / "procedure_2_physics_loss", device=device,
        )
        terrain.train_head(
            caches / "base_train", caches / "base_validation",
            study / "training" / "procedure_3_terrain_head", protocol, device,
        )
        terrain.plot_training_curves(study)
    else:
        selected = json.loads((study / "soft_rerank_selection.json").read_text())
        terrain.train_physics_loss(
            settings=settings, checkpoint=p["search"], fields=fields,
            protocol=replace(protocol, physics_loss_weight=0.0),
            selected=selected["selected"],
            run_dir=study / "training" / "zero_physics_control", device=device,
        )


def evaluate(fields, root: Path, dataset: Path, fold: int, pair: str,
             settings, device: torch.device, extra: str = "",
             auxiliary_dataset: Path | None = None) -> None:
    p = paths(root, fold)
    heldout = FOLDS[fold]
    study = p["study"]
    raster_dataset = (auxiliary_dataset or dataset) if pair != "heldout_test" else dataset
    full, glacier, source, target, aoi, radius = evaluation_spec(
        raster_dataset, p["fold"], heldout, pair
    )
    pair_root = study / ("test" if pair == "heldout_test" else f"belvedere_{pair}")
    source_table = (p["fold"] / f"source__{heldout}.parquet" if pair == "heldout_test"
                    else full.with_name("source_queries.parquet"))
    query = source_only(source_table, pair_root / "source_only.parquet")
    pair_config = pair_settings(settings, raster_dataset, query, glacier, source, target, aoi, radius)
    protocol = terrain.Protocol(radius_m=radius)
    base_cache = pair_root / "cache" / "image"
    terrain.cache_surfaces(
        cache=base_cache, settings=pair_config, phase="validation",
        checkpoint=p["search"], fields=fields, protocol=protocol, device=device,
    )
    if extra == "zero_control":
        control_checkpoint = study / "training" / "zero_physics_control" / "best.pt"
        control_cache = pair_root / "cache" / "zero_physics_control"
        terrain.cache_surfaces(
            cache=control_cache, settings=pair_config, phase="validation",
            checkpoint=control_checkpoint, fields=fields, protocol=protocol, device=device,
        )
        rows = [terrain.peak(terrain.score_surface(data["logits"],
                data["features"].astype(np.float32), data["valid"]), meta)
                for meta, data in terrain.iter_cache(control_cache)]
        prediction_path = pair_root / "zero_physics_predictions.json"
        terrain.dump(prediction_path, rows)
        truth = pd.read_parquet(full)
        terrain.dump(pair_root / "zero_physics_metrics.json", {
            "metrics": terrain.metrics(rows, truth),
            "prediction_sha256": terrain.digest(prediction_path),
            "checkpoint_sha256": terrain.digest(control_checkpoint),
            "query_sha256": terrain.digest(full),
        })
        print(json.dumps({"fold": fold, "pair": pair, "zero_control": "complete"}), flush=True)
        return
    physics = study / "training" / "procedure_2_physics_loss" / "best.pt"
    head = study / "training" / "procedure_3_terrain_head" / "best_head.pt"
    selected = json.loads((study / "soft_rerank_selection.json").read_text())["selected"]
    physics_cache = pair_root / "cache" / "physics"
    terrain.cache_surfaces(
        cache=physics_cache, settings=pair_config, phase="validation",
        checkpoint=physics, fields=fields, protocol=protocol, device=device,
    )
    prediction_dir = pair_root / "predictions"
    terrain.predict_methods(
        base_cache=base_cache, physics_cache=physics_cache, selected=selected,
        head_path=head, output=prediction_dir, protocol=protocol, device=device,
    )
    surveys = raster_dataset / "raw" / glacier / "surveys"
    result = terrain.report_predictions(
        name=f"{glacier} {pair} fold {fold}", cache=base_cache,
        predictions=prediction_dir, truth_path=full, output=pair_root / "reporting",
        source_raster=survey_raster(surveys / source),
        target_raster=survey_raster(surveys / target),
        panel_renderer=partial(plot_query_panels, terrain, radius_m=radius),
    )
    terrain.dump(pair_root / "complete.json", {
        "fold": fold, "heldout": heldout, "pair": pair, "radius_m": radius,
        "metrics": result, "truth_sha256": terrain.digest(full),
        "source_only_sha256": terrain.digest(query),
    })
    print(json.dumps({"fold": fold, "pair": pair,
                      "image_success_at_2m": result["control_image_only"]["success_at_2m"]}),
          flush=True)
