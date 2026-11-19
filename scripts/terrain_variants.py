"""Terrain training, score reranking and evaluation for the study."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from affine import Affine
from PIL import Image, ImageOps
from pyproj import Transformer
from shapely.geometry import LineString

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from glaciertrack import __version__
from glaciertrack.checkpoints import load_tracker_checkpoint
from glaciertrack.data.prompts import PromptAugmentation, render_source_prompt
from glaciertrack.models.search_objective import (
    StitchedSearch,
    differentiable_stitch,
    search_episode_loss,
)
from glaciertrack.models.tracker import SearchEncoding
from glaciertrack.search_training import (
    SearchTrainingError,
    _clip_gradients,
    _EpisodeReader,
    _freeze_for_search_finetuning,
    _SinglePairReader,
)
from glaciertrack.terrain import TerrainConditioningHead, TerrainField, TerrainGrid


@dataclass(frozen=True)
class Protocol:
    radius_m: float = 75.0
    prompt_context_weight: float = 0.50
    physics_loss_weight: float = 0.25
    physics_epochs: int = 3
    physics_learning_rate: float = 2.5e-5
    head_epochs: int = 12
    head_learning_rate: float = 1e-3
    head_augmentation_repeats: int = 4
    uphill_tolerance_m: float = 5.0
    uphill_scale_m: float = 10.0
    local_neighbors: int = 7
    local_prior_strength: float = 0.25
    local_prior_scale_m: float = 20.0


METHOD_LABELS = {
    "control_image_only": "Image-only control",
    "procedure_1_soft_rerank": "1 — Fixed soft terrain reranker",
    "procedure_2_physics_loss": "2 — Physics-loss fine-tuning",
    "procedure_3_terrain_head": "3 — Learned terrain head",
}


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_stem(identifier: str) -> str:
    """Return a readable, collision-safe filename stem for a logical ID."""

    readable = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in str(identifier)
    ).strip("._")
    readable = readable[:120] or "record"
    suffix = hashlib.sha256(str(identifier).encode("utf-8")).hexdigest()[:10]
    return f"{readable}__{suffix}"


def source_only(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["query_id", "glacier_id", "source_x_m", "source_y_m"]
    if set(required).difference(frame.columns):
        raise ValueError("source query table is incomplete")
    result = frame[required].copy()
    if result.query_id.duplicated().any():
        raise ValueError("source query IDs must be unique")
    result["target_x_m"] = np.nan
    result["target_y_m"] = np.nan
    result["match_present"] = False
    return result



def hybrid_prompt(settings: Any, episode: Any) -> torch.Tensor:
    gaussian = render_source_prompt(
        settings.reference_size_hw,
        episode.source_point_px,
        gsd_m_per_px=settings.tensor_gsd_m_per_px,
        config=replace(settings.prompt_config, backend="parametric", point_sigma_m=4.75),
        augmentation=PromptAugmentation(enabled=False),
        approach="gaussian_point",
    )
    return torch.maximum(episode.source_prompt, 0.50 * gaussian)


def source_in_target(row: Any, active: _SinglePairReader) -> tuple[float, float]:
    x, y = float(row.source_x_m), float(row.source_y_m)
    if str(active.source.crs) != str(active.target.crs):
        x, y = Transformer.from_crs(
            active.source.crs, active.target.crs, always_xy=True
        ).transform(x, y)
    return float(x), float(y)


def make_terrain_grid(
    field: TerrainField,
    row: Any,
    episode: Any,
    active: _SinglePairReader,
    shape_hw: tuple[int, int],
    protocol: Protocol,
    stride: int,
) -> TerrainGrid:
    original_canvas_shape = episode.unaugmented_canvas_shape_hw or episode.canvas_shape_hw
    original_grid_shape = tuple(math.ceil(value / stride) for value in original_canvas_shape)
    grid = field.grid(
        target_crs=active.target.crs,
        target_transform=active.target.transform,
        canvas_origin_xy_px=episode.canvas_origin_xy_px,
        shape_hw=original_grid_shape,
        stride_px=stride,
        source_xy_m=source_in_target(row, active),
        radius_m=protocol.radius_m,
        uphill_tolerance_m=protocol.uphill_tolerance_m,
        uphill_scale_m=protocol.uphill_scale_m,
    )
    grid = grid.transformed(episode.augmentation_geometry)
    if tuple(grid.valid.shape) != tuple(shape_hw):
        raise SearchTrainingError(
            "transformed terrain grid shape "
            f"{tuple(grid.valid.shape)} differs from stitched heatmap {tuple(shape_hw)}"
        )
    return grid


def forward_episode(model: Any, episode: Any, settings: Any, device: torch.device) -> StitchedSearch:
    with torch.inference_mode():
        reference_features = model._encode_image(episode.reference_image[None].to(device))
        reference = model.encode_reference_features(
            reference_features,
            hybrid_prompt(settings, episode)[None].to(device),
            source_point_px=episode.source_point_px[None].to(device),
        )
        encoded = []
        output_size = None
        for start in range(0, len(episode.target_tiles), settings.encoder_batch_size):
            images = episode.target_tiles[start : start + settings.encoder_batch_size].to(device)
            part = model.encode_search(images)
            encoded.append(part.features)
            output_size = part.output_size_hw
        if output_size is None:
            raise SearchTrainingError(f"query {episode.query_id} generated no search features")
        output = model.match_encoded(reference, SearchEncoding(torch.cat(encoded), output_size))
        return differentiable_stitch(
            output.location_logits,
            output.match_logits,
            episode.tile_windows,
            episode.canvas_shape_hw,
            stride_px=settings.stitch_stride_px,
            match_logit_weight=settings.match_logit_weight,
            valid_mask=episode.valid_mask.to(device),
        )


def cache_surfaces(
    *,
    cache: Path,
    settings: Any,
    phase: str,
    checkpoint: Path,
    fields: dict[str, TerrainField],
    protocol: Protocol,
    device: torch.device,
    overwrite: bool = False,
    augmentation_repeats: int = 1,
) -> dict[str, Any]:
    if augmentation_repeats < 1:
        raise ValueError("augmentation_repeats must be at least one")
    if phase != "train" and augmentation_repeats != 1:
        raise ValueError("augmentation repeats are only valid for the training phase")
    cache.mkdir(parents=True, exist_ok=True)
    complete = cache / "complete.json"
    if complete.is_file() and not overwrite:
        return json.loads(complete.read_text(encoding="utf-8"))
    model = load_tracker_checkpoint(checkpoint, device=device).model.eval()
    records = []
    with _EpisodeReader(settings) as reader:
        work = [
            (augmentation_epoch, index, row)
            for augmentation_epoch in range(augmentation_repeats)
            for rows in [reader.rows(phase, augmentation_epoch)]
            for index, row in enumerate(rows)
        ]
        for progress, (augmentation_epoch, index, row) in enumerate(work, start=1):
            episode = reader.episode(
                row,
                phase=phase,
                epoch=augmentation_epoch,
                index=index,
            )
            active = reader.readers[reader.active]
            glacier = str(row.glacier_id)
            stitched = forward_episode(model, episode, settings, device)
            grid = make_terrain_grid(
                fields[glacier],
                row,
                episode,
                active,
                tuple(stitched.logits.shape),
                protocol,
                settings.stitch_stride_px,
            )
            source_query_id = str(row.query_id)
            key = (
                source_query_id
                if augmentation_repeats == 1
                else f"{source_query_id}__augmentation_{augmentation_epoch:02d}"
            )
            stem = artifact_stem(key)
            array_path = cache / f"{stem}.npz"
            target = (
                np.array([np.nan, np.nan], dtype=np.float32)
                if episode.target_xy_px is None
                else episode.target_xy_px.numpy().astype(np.float32)
            )
            np.savez_compressed(
                array_path,
                logits=stitched.logits.detach().cpu().numpy().astype(np.float32),
                features=grid.feature_stack().numpy().astype(np.float16),
                valid=(stitched.covered.detach().cpu() & grid.valid).numpy(),
                target_xy_px=target,
            )
            source_xy = source_in_target(row, active)
            metadata = {
                "query_id": key,
                "artifact_stem": stem,
                "source_query_id": source_query_id,
                "augmentation_epoch": augmentation_epoch,
                "augmentation_geometry": asdict(episode.augmentation_geometry),
                "glacier_id": glacier,
                "source_xy_m": list(source_xy),
                "target_xy_m": [
                    _finite_or_none(getattr(row, "target_x_m", None)),
                    _finite_or_none(getattr(row, "target_y_m", None)),
                ],
                "target_present": episode.target_xy_px is not None,
                "canvas_origin_xy_px": list(episode.canvas_origin_xy_px),
                "target_transform": list(active.target.transform)[:6],
                "target_crs": str(active.target.crs),
                "stride_px": settings.stitch_stride_px,
                "tensor_gsd_m_per_px": settings.tensor_gsd_m_per_px,
                "source_raster": str(active.settings.source_raster),
                "target_raster": str(active.settings.target_raster),
                "checkpoint": str(checkpoint),
            }
            dump(cache / f"{stem}.json", metadata)
            records.append(metadata)
            print(
                json.dumps({"cache": cache.name, "query": progress, "total": len(work)}),
                flush=True,
            )
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "phase": phase,
        "records": len(records),
        "augmentation_repeats": augmentation_repeats,
        "query_ids": [record["query_id"] for record in records],
        "items": [
            {"query_id": record["query_id"], "artifact_stem": record["artifact_stem"]}
            for record in records
        ],
    }
    dump(complete, result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def iter_cache(cache: Path) -> Iterable[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    manifest = json.loads((cache / "complete.json").read_text(encoding="utf-8"))
    items = manifest.get("items") or [
        {"query_id": query_id, "artifact_stem": query_id}
        for query_id in manifest["query_ids"]
    ]
    for item in items:
        stem = item["artifact_stem"]
        meta = json.loads((cache / f"{stem}.json").read_text(encoding="utf-8"))
        with np.load(cache / f"{stem}.npz") as data:
            yield meta, {name: data[name].copy() for name in data.files}


def physical_cost(features: np.ndarray, direction_weight: float, elevation_weight: float) -> np.ndarray:
    return (
        float(direction_weight) * features[..., 1]
        + float(elevation_weight) * features[..., 3]
    ) * features[..., 6]


def score_surface(
    logits: np.ndarray,
    features: np.ndarray,
    valid: np.ndarray,
    *,
    direction_weight: float = 0.0,
    elevation_weight: float = 0.0,
    head: TerrainConditioningHead | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    values = logits.astype(np.float32).copy()
    values -= physical_cost(features, direction_weight, elevation_weight)
    if head is not None:
        if device is None:
            raise ValueError("a terrain-head device is required")
        with torch.inference_mode():
            correction = head(torch.from_numpy(features.astype(np.float32)).to(device))
        values += correction.cpu().numpy()
    values[~valid] = -np.inf
    return values


def peak(values: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
    if not np.isfinite(values).any():
        raise ValueError(f"no finite terrain-supported candidate for {meta['query_id']}")
    y, x = np.unravel_index(np.nanargmax(values), values.shape)
    stride = int(meta["stride_px"])
    local = np.array([x, y], dtype=float) * stride + (stride - 1) / 2
    global_pixel = local + np.asarray(meta["canvas_origin_xy_px"], dtype=float)
    world = Affine(*meta["target_transform"]) * tuple(global_pixel + 0.5)
    return {
        "query_id": meta["query_id"],
        "prediction_xy_m": [float(world[0]), float(world[1])],
        "prediction_local_xy_px": local.tolist(),
        "score": float(values[y, x]),
    }


def metrics(records: list[dict[str, Any]], truth: pd.DataFrame | None = None) -> dict[str, Any]:
    if truth is None:
        errors = np.asarray([record.get("error_m", np.nan) for record in records], dtype=float)
    else:
        indexed = truth.set_index("query_id")
        errors = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(record["prediction_xy_m"], dtype=float)
                    - indexed.loc[record["query_id"], ["target_x_m", "target_y_m"]].to_numpy(
                        dtype=float
                    )
                )
                for record in records
            ]
        )
    finite = errors[np.isfinite(errors)]
    if not len(finite):
        raise ValueError("metrics require at least one finite endpoint error")
    result = {
        "queries": int(len(errors)),
        "mean_error_m": float(finite.mean()),
        "median_error_m": float(np.median(finite)),
        "p90_error_m": float(np.quantile(finite, 0.9)),
        "maximum_error_m": float(finite.max()),
        "rmse_m": float(np.sqrt(np.mean(finite**2))),
    }
    result.update({f"success_at_{value:g}m": float((errors <= value).mean()) for value in (0.5, 1, 2, 5, 10)})
    return result


def tune_soft_reranker(validation_cache: Path, output: Path) -> dict[str, Any]:
    trials = []
    weights = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    for direction_weight in weights:
        for elevation_weight in weights:
            records = []
            for meta, data in iter_cache(validation_cache):
                prediction = peak(
                    score_surface(
                        data["logits"],
                        data["features"].astype(np.float32),
                        data["valid"],
                        direction_weight=direction_weight,
                        elevation_weight=elevation_weight,
                    ),
                    meta,
                )
                target = np.asarray(data["target_xy_px"], dtype=float)
                prediction["error_m"] = float(
                    np.linalg.norm(np.asarray(prediction["prediction_local_xy_px"]) - target)
                    * meta["tensor_gsd_m_per_px"]
                )
                records.append(prediction)
            result = metrics(records)
            result.update(direction_weight=direction_weight, elevation_weight=elevation_weight)
            trials.append(result)
    selected = max(
        trials,
        key=lambda item: (
            item["success_at_2m"],
            item["success_at_5m"],
            -item["median_error_m"],
            -item["mean_error_m"],
            -(item["direction_weight"] + item["elevation_weight"]),
        ),
    )
    result = {
        "selection_data": "Miage spatial validation only",
        "selection_rule": "max success@2m, success@5m; min median, mean, total weight",
        "grid": list(weights),
        "selected": selected,
        "trials": trials,
    }
    dump(output, result)
    return result


def training_score(metric: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(metric["success_at_2m"]),
        -float(metric["median_error_m"]),
        -float(metric["mean_error_m"]),
    )


def train_physics_loss(
    *,
    settings: Any,
    checkpoint: Path,
    fields: dict[str, TerrainField],
    protocol: Protocol,
    selected: dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> Path:
    result_path = run_dir / "result.json"
    if result_path.is_file():
        return Path(json.loads(result_path.read_text(encoding="utf-8"))["best_checkpoint"])
    run_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_tracker_checkpoint(checkpoint, device=device)
    model = loaded.model
    trainable_modules = ("search_projection", "relation_encoder", "location_head", "match_head")
    _freeze_for_search_finetuning(model, trainable_modules)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=protocol.physics_learning_rate,
        weight_decay=1e-4,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    base_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    best_score = None
    best_epoch = 0
    started = time.time()
    with _EpisodeReader(settings) as reader:
        initial = run_physics_phase(
            model=model,
            reader=reader,
            settings=settings,
            fields=fields,
            protocol=protocol,
            selected=selected,
            phase="validation",
            epoch=0,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=False,
        )
        best_score = training_score(initial)
        save_model_checkpoint(best_path, base_payload, model, settings, initial, history, -1)
        print(json.dumps({"terrain_physics_epoch": 0, "validation": initial}), flush=True)
        for epoch in range(protocol.physics_epochs):
            training = run_physics_phase(
                model=model,
                reader=reader,
                settings=settings,
                fields=fields,
                protocol=protocol,
                selected=selected,
                phase="train",
                epoch=epoch,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
            )
            validation = run_physics_phase(
                model=model,
                reader=reader,
                settings=settings,
                fields=fields,
                protocol=protocol,
                selected=selected,
                phase="validation",
                epoch=epoch,
                device=device,
                optimizer=None,
                scaler=None,
                use_amp=False,
            )
            history.append({"epoch": epoch + 1, "training": training, "validation": validation})
            if training_score(validation) > best_score:
                best_score = training_score(validation)
                best_epoch = epoch + 1
                save_model_checkpoint(best_path, base_payload, model, settings, validation, history, epoch)
            save_model_checkpoint(last_path, base_payload, model, settings, validation, history, epoch)
            dump(run_dir / "history.json", {"initial_validation": initial, "epochs": history})
            print(json.dumps(history[-1]), flush=True)
    result = {
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "initial_validation": initial,
        "best_validation": max(
            [initial] + [entry["validation"] for entry in history], key=training_score
        ),
        "selected_epoch": best_epoch,
        "epochs": protocol.physics_epochs,
        "seconds": time.time() - started,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "direction_weight": selected["direction_weight"],
        "elevation_weight": selected["elevation_weight"],
        "physics_loss_weight": protocol.physics_loss_weight,
    }
    dump(result_path, result)
    return best_path


def run_physics_phase(
    *,
    model: Any,
    reader: _EpisodeReader,
    settings: Any,
    fields: dict[str, TerrainField],
    protocol: Protocol,
    selected: dict[str, Any],
    phase: str,
    epoch: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.eval()
    rows = reader.rows(phase, epoch)
    totals = {name: 0.0 for name in ("total", "image_loss", "physics_loss")}
    errors = []
    episodes = 0
    for index, row in enumerate(rows):
        episode = reader.episode(row, phase=phase, epoch=epoch, index=index)
        active = reader.readers[reader.active]
        glacier = str(row.glacier_id)
        if training:
            optimizer.zero_grad(set_to_none=True)
        projection_context = nullcontext() if training else torch.no_grad()
        with torch.no_grad():
            reference_features = model._encode_image(episode.reference_image[None].to(device))
        with projection_context:
            reference = model.encode_reference_features(
                reference_features,
                hybrid_prompt(settings, episode)[None].to(device),
                source_point_px=episode.source_point_px[None].to(device),
            )
        encoded = []
        output_size = None
        for start in range(0, len(episode.target_tiles), settings.encoder_batch_size):
            images = episode.target_tiles[start : start + settings.encoder_batch_size].to(device)
            with torch.no_grad():
                image_features = model._encode_image(images)
            with projection_context:
                part = model.encode_search_features(image_features, tuple(images.shape[-2:]))
            encoded.append(part.features)
            output_size = part.output_size_hw
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if training and use_amp
            else nullcontext()
        )
        grad_context = nullcontext() if training else torch.no_grad()
        with grad_context, context:
            output = model.match_encoded(reference, SearchEncoding(torch.cat(encoded), output_size))
            stitched = differentiable_stitch(
                output.location_logits,
                output.match_logits,
                episode.tile_windows,
                episode.canvas_shape_hw,
                stride_px=settings.stitch_stride_px,
                match_logit_weight=settings.match_logit_weight,
                valid_mask=episode.valid_mask.to(device),
            )
            image_loss = search_episode_loss(
                stitched,
                output.location_logits,
                output.match_logits,
                episode.tile_windows,
                episode.target_xy_px,
                stride_px=settings.stitch_stride_px,
                target_sigma_px=settings.target_sigma_m / settings.tensor_gsd_m_per_px,
                global_weight=0.25,
                presence_weight=0.10,
                positive_localization_weight=0.0,
                negative_suppression_weight=0.05,
                ranking_weight=1.0,
                ranking_margin=0.0,
                ranking_exclusion_radius_px=2.0 / settings.tensor_gsd_m_per_px,
                negative_temperature=0.1,
                tile_target_xy_px=episode.tile_target_xy_px,
                separate_tile_target=True,
            ).total
            grid = make_terrain_grid(
                fields[glacier],
                row,
                episode,
                active,
                tuple(stitched.logits.shape),
                protocol,
                settings.stitch_stride_px,
            )
            supported = stitched.covered & grid.valid.to(device)
            # Anchor the zero term to a finite differentiable scalar. Stitched
            # surfaces intentionally contain -inf outside covered cells, so
            # ``stitched.logits.sum() * 0`` becomes NaN for valid no-match
            # episodes (for example a true endpoint beyond the search radius).
            physics = image_loss * 0.0
            if episode.target_xy_px is not None and bool(supported.any()):
                cost = grid.combined_cost(
                    selected["direction_weight"], selected["elevation_weight"]
                ).to(device)
                probability = F.softmax(stitched.logits[supported].float(), dim=0)
                physics = (probability * cost[supported]).sum()
            total = image_loss + protocol.physics_loss_weight * physics
        if not bool(torch.isfinite(total)):
            diagnostics = {
                "query_id": episode.query_id,
                "phase": phase,
                "epoch": epoch,
                "image_loss": float(image_loss.detach()),
                "physics_loss": float(physics.detach()),
                "target_in_search": episode.target_xy_px is not None,
                "supported_cells": int(supported.sum().detach()),
                "finite_stitched_logits": int(torch.isfinite(stitched.logits).sum().detach()),
                "stitched_cells": int(stitched.logits.numel()),
                "finite_location_logits": int(
                    torch.isfinite(output.location_logits).sum().detach()
                ),
                "location_logit_cells": int(output.location_logits.numel()),
            }
            raise SearchTrainingError(
                f"non-finite terrain loss: {json.dumps(diagnostics, sort_keys=True)}"
            )
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                _clip_gradients(model, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                _clip_gradients(model, 1.0)
                optimizer.step()
        totals["total"] += float(total.detach())
        totals["image_loss"] += float(image_loss.detach())
        totals["physics_loss"] += float(physics.detach())
        episodes += 1
        if episode.target_xy_px is not None:
            flat = int(stitched.logits.detach().flatten().argmax())
            y, x = divmod(flat, stitched.logits.shape[1])
            local = np.array([x, y]) * settings.stitch_stride_px + (
                settings.stitch_stride_px - 1
            ) / 2
            errors.append(
                float(
                    np.linalg.norm(local - episode.target_xy_px.numpy())
                    * settings.tensor_gsd_m_per_px
                )
            )
        if training and (index + 1) % 20 == 0:
            print(json.dumps({"terrain_phase": phase, "epoch": epoch + 1, "query": index + 1, "total": len(rows)}), flush=True)
    array = np.asarray(errors)
    result = {name: value / episodes for name, value in totals.items()}
    result.update(
        episodes=float(episodes),
        mean_error_m=float(array.mean()),
        median_error_m=float(np.median(array)),
        p90_error_m=float(np.quantile(array, 0.9)),
        **{f"success_at_{value:g}m": float((array <= value).mean()) for value in (0.5, 1, 2, 5, 10)},
    )
    return result


def save_model_checkpoint(
    path: Path,
    base_payload: dict[str, Any],
    model: Any,
    settings: Any,
    validation: dict[str, float],
    history: list[dict[str, Any]],
    epoch: int,
) -> None:
    payload = dict(base_payload)
    payload.update(
        package_version=__version__,
        model_state={name: value.detach().cpu() for name, value in model.state_dict().items()},
        epoch=epoch,
        validation=validation,
        history=history,
    )
    config = json.loads(json.dumps(base_payload["config"]))
    config["terrain_training"] = {
        "procedure": "expected_physical_cost",
        "source": "Copernicus GLO-30",
        "comparison": "SRTM GL1 agreement confidence",
        "settings": asdict(Protocol()),
    }
    payload["config"] = config
    torch.save(payload, path)


def train_head(
    train_cache: Path,
    validation_cache: Path,
    output: Path,
    protocol: Protocol,
    device: torch.device,
) -> Path:
    result_path = output / "result.json"
    if result_path.is_file():
        return Path(json.loads(result_path.read_text(encoding="utf-8"))["best_head"])
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1200)
    head = TerrainConditioningHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=protocol.head_learning_rate, weight_decay=1e-4)
    best = None
    best_epoch = 0
    best_path = output / "best_head.pt"
    history = []
    initial = evaluate_head(validation_cache, head, device)
    torch.save({"state_dict": head.state_dict(), "epoch": -1, "validation": initial}, best_path)
    best = training_score(initial)
    train_variants: dict[str, list[tuple[dict[str, Any], dict[str, np.ndarray]]]] = defaultdict(list)
    for meta, data in iter_cache(train_cache):
        train_variants[str(meta.get("source_query_id", meta["query_id"]))].append((meta, data))
    for variants in train_variants.values():
        variants.sort(key=lambda item: int(item[0].get("augmentation_epoch", 0)))
    for epoch in range(protocol.head_epochs):
        train_items = [
            variants[epoch % len(variants)]
            for variants in train_variants.values()
        ]
        random.Random(1200 + epoch).shuffle(train_items)
        head.train()
        losses = []
        for meta, data in train_items:
            target = np.asarray(data["target_xy_px"], dtype=float)
            if not np.isfinite(target).all():
                continue
            logits = torch.from_numpy(data["logits"]).to(device)
            features = torch.from_numpy(data["features"].astype(np.float32)).to(device)
            valid = torch.from_numpy(data["valid"]).to(device)
            correction = head(features)
            adjusted = logits + correction
            loss = head_localization_loss(
                adjusted,
                valid,
                torch.as_tensor(target, dtype=torch.float32, device=device),
                stride=int(meta["stride_px"]),
                gsd=float(meta["tensor_gsd_m_per_px"]),
            )
            loss = loss + 1e-4 * correction[valid].square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_head(validation_cache, head, device)
        record = {
            "epoch": epoch + 1,
            "augmentation_cycle": epoch % protocol.head_augmentation_repeats,
            "training_trajectories": len(train_items),
            "training_loss": float(np.mean(losses)),
            "validation": validation,
        }
        history.append(record)
        if training_score(validation) > best:
            best = training_score(validation)
            best_epoch = epoch + 1
            torch.save({"state_dict": head.state_dict(), "epoch": epoch, "validation": validation}, best_path)
        dump(output / "history.json", {"initial_validation": initial, "epochs": history})
        print(json.dumps({"terrain_head": record}), flush=True)
    dump(
        result_path,
        {
            "best_head": str(best_path.resolve()),
            "initial_validation": initial,
            "best_validation": max([initial] + [item["validation"] for item in history], key=training_score),
            "selected_epoch": best_epoch,
            "epochs": protocol.head_epochs,
            "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
        },
    )
    return best_path


def head_localization_loss(
    logits: torch.Tensor,
    valid: torch.Tensor,
    target_xy: torch.Tensor,
    *,
    stride: int,
    gsd: float,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(logits.shape[0], device=logits.device, dtype=torch.float32),
        torch.arange(logits.shape[1], device=logits.device, dtype=torch.float32),
        indexing="ij",
    )
    center = (stride - 1) / 2
    x = xx * stride + center
    y = yy * stride + center
    sigma_px = 0.35 / gsd
    distribution = torch.exp(-((x - target_xy[0]).square() + (y - target_xy[1]).square()) / (2 * sigma_px**2))
    distribution = distribution * valid
    distribution = distribution / distribution.sum().clamp_min(1e-8)
    localization = -(distribution[valid] * F.log_softmax(logits[valid], dim=0)).sum()
    support = ((x - target_xy[0]).square() + (y - target_xy[1]).square()) <= (2.0 / gsd) ** 2
    false = valid & ~support
    target_score = (distribution[valid] * logits[valid]).sum()
    ranking = F.softplus(logits[false].amax() - target_score) if bool(false.any()) else localization * 0
    return localization + 0.25 * ranking


def load_head(path: Path, device: torch.device) -> TerrainConditioningHead:
    head = TerrainConditioningHead().to(device)
    head.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state_dict"])
    return head.eval()


def evaluate_head(cache: Path, head: TerrainConditioningHead, device: torch.device) -> dict[str, Any]:
    records = []
    head.eval()
    for meta, data in iter_cache(cache):
        result = peak(
            score_surface(
                data["logits"],
                data["features"].astype(np.float32),
                data["valid"],
                head=head,
                device=device,
            ),
            meta,
        )
        target = np.asarray(data["target_xy_px"], dtype=float)
        result["error_m"] = float(
            np.linalg.norm(np.asarray(result["prediction_local_xy_px"]) - target)
            * meta["tensor_gsd_m_per_px"]
        )
        records.append(result)
    return metrics(records)


def predict_methods(
    *,
    base_cache: Path,
    physics_cache: Path,
    selected: dict[str, Any],
    head_path: Path,
    output: Path,
    protocol: Protocol,
    device: torch.device,
) -> dict[str, Any]:
    completion = output / "predictions_complete.json"
    if completion.is_file():
        return json.loads(completion.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    head = load_head(head_path, device)
    physics_lookup = {meta["query_id"]: data for meta, data in iter_cache(physics_cache)}
    maps: dict[str, dict[str, np.ndarray]] = {name: {} for name in METHOD_LABELS}
    surfaces = output / "surfaces"
    surfaces.mkdir(parents=True, exist_ok=True)
    metadata = {}
    first_records = {name: [] for name in METHOD_LABELS}
    for meta, base in iter_cache(base_cache):
        query_id = meta["query_id"]
        physics = physics_lookup[query_id]
        features = base["features"].astype(np.float32)
        variants = {
            "control_image_only": score_surface(base["logits"], features, base["valid"]),
            "procedure_1_soft_rerank": score_surface(
                base["logits"],
                features,
                base["valid"],
                direction_weight=selected["direction_weight"],
                elevation_weight=selected["elevation_weight"],
            ),
            "procedure_2_physics_loss": score_surface(
                physics["logits"], physics["features"].astype(np.float32), physics["valid"]
            ),
            "procedure_3_terrain_head": score_surface(
                base["logits"], features, base["valid"], head=head, device=device
            ),
        }
        metadata[query_id] = meta
        np.savez_compressed(surfaces / f"{artifact_stem(query_id)}.npz", **variants)
        for method, values in variants.items():
            maps[method][query_id] = values
            first_records[method].append(peak(values, meta))
    all_records = dict(first_records)
    for method, records in first_records.items():
        if len(records) < 2:
            all_records[f"{method}__local7"] = records
            continue
        ids = np.asarray([record["query_id"] for record in records])
        sources = np.asarray([metadata[q]["source_xy_m"] for q in ids], dtype=float)
        endpoints = np.asarray([record["prediction_xy_m"] for record in records], dtype=float)
        adjusted_records = []
        for index, query_id in enumerate(ids):
            keep = np.arange(len(ids)) != index
            distances = np.linalg.norm(sources[keep] - sources[index], axis=1)
            order = np.argsort(distances)[: protocol.local_neighbors]
            movement = np.median((endpoints[keep] - sources[keep])[order], axis=0)
            prior_world = sources[index] + movement
            meta = metadata[query_id]
            inverse = ~Affine(*meta["target_transform"])
            prior_global = np.asarray(inverse * tuple(prior_world)) - 0.5
            prior_local = prior_global - np.asarray(meta["canvas_origin_xy_px"])
            values = maps[method][query_id].copy()
            yy, xx = np.indices(values.shape)
            stride = int(meta["stride_px"])
            x = xx * stride + (stride - 1) / 2
            y = yy * stride + (stride - 1) / 2
            gsd = float(meta["tensor_gsd_m_per_px"])
            cost = ((x - prior_local[0]) ** 2 + (y - prior_local[1]) ** 2) * gsd**2 / (
                2 * protocol.local_prior_scale_m**2
            )
            values -= protocol.local_prior_strength * cost
            maps[method][query_id] = values
            adjusted_records.append(peak(values, meta))
        all_records[f"{method}__local7"] = adjusted_records
    for method, records in all_records.items():
        dump(output / f"{method}.json", records)
    hashes = {path.name: digest(path) for path in sorted(output.glob("*.json"))}
    result = {
        "query_count": len(metadata),
        "truth_used": False,
        "methods": sorted(all_records),
        "prediction_hashes": hashes,
    }
    dump(completion, result)
    return result


def report_predictions(
    *,
    name: str,
    cache: Path,
    predictions: Path,
    truth_path: Path,
    output: Path,
    source_raster: Path,
    target_raster: Path,
    panel_renderer,
) -> dict[str, Any]:
    completion = json.loads((predictions / "predictions_complete.json").read_text(encoding="utf-8"))
    for filename, expected in completion["prediction_hashes"].items():
        if digest(predictions / filename) != expected:
            raise ValueError(f"prediction changed before truth scoring: {filename}")
    truth = pd.read_parquet(truth_path)
    if not truth.match_present.all():
        raise ValueError("localization report expects observed endpoints")
    methods = {}
    all_metrics = {}
    for method in completion["methods"]:
        records = json.loads((predictions / f"{method}.json").read_text(encoding="utf-8"))
        result = metrics(records, truth)
        all_metrics[method] = result
        methods[method] = records
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "metrics.json", all_metrics)
    plot_metric_comparison(all_metrics, output / "metrics_comparison.png", name)
    plot_metric_comparison(
        all_metrics,
        output / "metrics_comparison_local7.png",
        name,
        local7=True,
    )
    plot_cdf(methods, truth, output / "error_cdf.png", name)
    plot_cdf(methods, truth, output / "error_cdf_local7.png", name, local7=True)
    plot_vectors(methods, truth, source_raster, output, name)
    export_geopackage(methods, truth, output / "movement_vectors.gpkg")
    panel_renderer(
        cache,
        predictions,
        methods,
        truth,
        source_raster,
        target_raster,
        output / "query_panels",
        name,
    )
    return all_metrics




def plot_metric_comparison(
    values: dict[str, Any], path: Path, title: str, *, local7: bool = False
) -> None:
    names = [
        f"{name}__local7" if local7 else name
        for name in METHOD_LABELS
    ]
    labels = [METHOD_LABELS[name.removesuffix("__local7")] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    success = [100 * values[name]["success_at_2m"] for name in names]
    means = [values[name]["mean_error_m"] for name in names]
    for axis, data, ylabel in (
        (axes[0], success, "Within 2 m (%)"),
        (axes[1], means, "Mean endpoint error (m)"),
    ):
        bars = axis.bar(np.arange(len(names)), data, color=["#777777", "#e69f00", "#56b4e9", "#009e73"])
        axis.bar_label(bars, fmt="%.1f")
        axis.set_xticks(np.arange(len(names)), labels, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    suffix = " with prediction-only local-seven consensus" if local7 else ""
    fig.suptitle(f"{title}: terrain procedure comparison{suffix}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cdf(
    methods: dict[str, Any],
    truth: pd.DataFrame,
    path: Path,
    title: str,
    *,
    local7: bool = False,
) -> None:
    indexed = truth.set_index("query_id")
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for method in METHOD_LABELS:
        key = f"{method}__local7" if local7 else method
        rows = methods[key]
        errors = sorted(
            np.linalg.norm(
                np.asarray(row["prediction_xy_m"])
                - indexed.loc[row["query_id"], ["target_x_m", "target_y_m"]].to_numpy(dtype=float)
            )
            for row in rows
        )
        axis.step(
            errors,
            np.arange(1, len(errors) + 1) / len(errors),
            where="post",
            label=METHOD_LABELS[method],
        )
    suffix = " — prediction-only local-seven consensus" if local7 else ""
    axis.set(
        xlabel="Endpoint error (m)",
        ylabel="Fraction of queries",
        ylim=(0, 1),
        title=f"{title}{suffix}",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def raster_background(path: Path, max_side: int = 2200) -> tuple[np.ndarray, tuple[float, ...], str]:
    import rasterio

    with rasterio.open(path) as dataset:
        scale = min(1.0, max_side / max(dataset.width, dataset.height))
        shape = (max(1, round(dataset.height * scale)), max(1, round(dataset.width * scale)))
        rgb = dataset.read((1, 2, 3), out_shape=(3, *shape))
        alpha = dataset.dataset_mask(out_shape=shape)
        rgba = np.concatenate([np.moveaxis(rgb, 0, -1), alpha[..., None]], axis=-1)
        extent = (dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top)
        return rgba, extent, str(dataset.crs)


def plot_vectors(
    methods: dict[str, Any], truth: pd.DataFrame, raster: Path, output: Path, title: str
) -> None:
    background, extent, crs = raster_background(raster)
    indexed = truth.set_index("query_id")
    ids = truth.query_id.astype(str).to_numpy()
    origins = truth.set_index("query_id").loc[ids, ["source_x_m", "source_y_m"]].to_numpy(dtype=float)
    target_endpoints = indexed.loc[ids, ["target_x_m", "target_y_m"]].to_numpy(dtype=float)
    if len(set(truth.source_crs.astype(str))) == 1 and str(truth.source_crs.iloc[0]) != crs:
        transformer = Transformer.from_crs(truth.source_crs.iloc[0], crs, always_xy=True)
        origins = np.column_stack(transformer.transform(origins[:, 0], origins[:, 1]))
    for method, rows in methods.items():
        lookup = {row["query_id"]: row for row in rows}
        predicted = np.asarray([lookup[q]["prediction_xy_m"] for q in ids], dtype=float)
        endpoints = target_endpoints.copy()
        if str(truth.target_crs.iloc[0]) != crs:
            transformer = Transformer.from_crs(truth.target_crs.iloc[0], crs, always_xy=True)
            endpoints = np.column_stack(transformer.transform(endpoints[:, 0], endpoints[:, 1]))
            predicted = np.column_stack(transformer.transform(predicted[:, 0], predicted[:, 1]))
        errors = np.linalg.norm(predicted - endpoints, axis=1)
        fig, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
        axis.imshow(background, extent=extent)
        for points, color, label in ((endpoints, "#268cff", "Truth"), (predicted, "#ff3030", "Prediction")):
            axis.quiver(
                origins[:, 0], origins[:, 1], points[:, 0] - origins[:, 0], points[:, 1] - origins[:, 1],
                angles="xy", scale_units="xy", scale=1, color=color, width=0.003, alpha=0.85, label=label,
            )
        bounds = np.vstack((origins, endpoints, predicted))
        margin = 80
        axis.set_xlim(bounds[:, 0].min() - margin, bounds[:, 0].max() + margin)
        axis.set_ylim(bounds[:, 1].min() - margin, bounds[:, 1].max() + margin)
        axis.ticklabel_format(useOffset=False, style="plain")
        axis.legend()
        axis.set_title(f"{title} — {method}\nred predictions / blue truth; within 2 m: {(errors <= 2).sum()}/{len(errors)}")
        fig.savefig(output / f"vectors__{method}.png", dpi=180)
        plt.close(fig)


def export_geopackage(methods: dict[str, Any], truth: pd.DataFrame, path: Path) -> None:
    ids = truth.query_id.astype(str).to_numpy()
    indexed = truth.set_index("query_id").loc[ids]
    crs = str(indexed.target_crs.iloc[0])
    sources = indexed[["source_x_m", "source_y_m"]].to_numpy(dtype=float)
    if str(indexed.source_crs.iloc[0]) != crs:
        transformer = Transformer.from_crs(indexed.source_crs.iloc[0], crs, always_xy=True)
        sources = np.column_stack(transformer.transform(sources[:, 0], sources[:, 1]))
    targets = indexed[["target_x_m", "target_y_m"]].to_numpy(dtype=float)
    truth_rows = gpd.GeoDataFrame(
        {"query_id": ids},
        geometry=[LineString((a, b)) for a, b in zip(sources, targets, strict=True)],
        crs=crs,
    )
    prediction_rows = []
    for method, rows in methods.items():
        lookup = {row["query_id"]: row for row in rows}
        for query_id, source, target in zip(ids, sources, targets, strict=True):
            predicted = np.asarray(lookup[query_id]["prediction_xy_m"], dtype=float)
            prediction_rows.append(
                {
                    "query_id": query_id,
                    "method": method,
                    "error_m": float(np.linalg.norm(predicted - target)),
                    "geometry": LineString((source, predicted)),
                }
            )
    if path.exists():
        path.unlink()
    truth_rows.to_file(path, layer="truth", driver="GPKG")
    gpd.GeoDataFrame(prediction_rows, crs=crs).to_file(path, layer="predictions", driver="GPKG")




def make_contact_sheet(files: list[Path], output: Path) -> None:
    thumbnails = []
    for path in files:
        with Image.open(path) as image:
            item = image.convert("RGB")
            item.thumbnail((600, 500))
            thumbnails.append(ImageOps.pad(item, (600, 500), color="white"))
    if not thumbnails:
        return
    sheet = Image.new("RGB", (1800, math.ceil(len(thumbnails) / 3) * 500), "white")
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % 3) * 600, (index // 3) * 500))
    sheet.save(output)




def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None



def plot_training_curves(root: Path) -> None:
    physics = json.loads(
        (root / "training" / "procedure_2_physics_loss" / "history.json").read_text()
    )
    head = json.loads(
        (root / "training" / "procedure_3_terrain_head" / "history.json").read_text()
    )
    physics_result = json.loads(
        (root / "training" / "procedure_2_physics_loss" / "result.json").read_text()
    )
    head_result = json.loads(
        (root / "training" / "procedure_3_terrain_head" / "result.json").read_text()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    physics_epochs = [0] + [item["epoch"] for item in physics["epochs"]]
    physics_success = [physics["initial_validation"]["success_at_2m"]] + [
        item["validation"]["success_at_2m"] for item in physics["epochs"]
    ]
    physics_cost = [physics["initial_validation"]["physics_loss"]] + [
        item["validation"]["physics_loss"] for item in physics["epochs"]
    ]
    axes[0].plot(physics_epochs, np.asarray(physics_success) * 100, "o-", label="Validation ≤2 m (%)")
    twin = axes[0].twinx()
    twin.plot(physics_epochs, physics_cost, "s--", color="#d55e00", label="Expected terrain cost")
    axes[0].set(xlabel="Epoch", ylabel="Validation within 2 m (%)", title="Procedure 2")
    axes[0].axvline(
        physics_result["selected_epoch"],
        color="#009e73",
        linestyle="--",
        label="Selected epoch",
    )
    axes[0].legend()
    twin.set_ylabel("Expected terrain cost")
    axes[0].grid(alpha=0.2)
    head_epochs = [0] + [item["epoch"] for item in head["epochs"]]
    head_success = [head["initial_validation"]["success_at_2m"]] + [
        item["validation"]["success_at_2m"] for item in head["epochs"]
    ]
    axes[1].plot(head_epochs, np.asarray(head_success) * 100, "o-", label="Validation ≤2 m")
    axes[1].axvline(
        head_result["selected_epoch"],
        color="#009e73",
        linestyle="--",
        label="Selected epoch",
    )
    axes[1].set(xlabel="Epoch", ylabel="Validation within 2 m (%)", title="Procedure 3")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.suptitle("Terrain training diagnostics; lower loss alone is not model selection")
    fig.savefig(root / "training_curves.png", dpi=180)
    plt.close(fig)
