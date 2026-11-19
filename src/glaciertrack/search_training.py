"""Search-consistent episodic fine-tuning on complete target-tile sets."""

from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from glaciertrack import __version__
from glaciertrack.checkpoints import load_tracker_checkpoint
from glaciertrack.config import ConfigError, load_yaml
from glaciertrack.data.augmentation import (
    GeometryChoice,
    ImageAugmentation,
    augment_search_episode,
    transform_window,
)
from glaciertrack.data.prompts import (
    PromptAugmentation,
    PromptConfig,
    render_source_prompt,
)
from glaciertrack.device import resolve_device
from glaciertrack.inference.raster import AnalysisRaster
from glaciertrack.inference.search import SearchConfig, plan_search
from glaciertrack.models.search_objective import (
    SearchLoss,
    differentiable_stitch,
    search_episode_loss,
)
from glaciertrack.models.tracker import PromptedSiameseTracker, SearchEncoding


class SearchTrainingError(RuntimeError):
    """Raised when a tiled-search training episode cannot be constructed."""


@dataclass(frozen=True)
class SearchTrainingSettings:
    train_queries: Path
    validation_queries: Path
    source_raster: Path
    target_raster: Path
    target_aoi: Path
    output_dir: Path
    run_name: str
    seed: int
    epochs: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float | None
    precision: str
    tensor_gsd_m_per_px: float
    reference_size_hw: tuple[int, int]
    radius_m: float
    tile_size_hw: tuple[int, int]
    tile_overlap: float
    encoder_batch_size: int
    stitch_stride_px: int
    match_logit_weight: float
    target_sigma_m: float
    global_weight: float
    presence_weight: float
    positive_localization_weight: float
    negative_suppression_weight: float
    ranking_weight: float
    ranking_margin: float
    ranking_exclusion_radius_m: float | None
    negative_temperature: float
    trainable_modules: tuple[str, ...]
    train_query_limit: int | None
    validation_query_limit: int | None
    prompt_config: PromptConfig
    prompt_augmentation: PromptAugmentation
    train_augmentation: ImageAugmentation
    additional_pairs: tuple[dict[str, Path], ...] = ()
    training_radii_m: tuple[float, ...] = ()


@dataclass(frozen=True)
class SearchEpisode:
    query_id: str
    reference_image: Tensor
    source_prompt: Tensor
    target_tiles: Tensor
    tile_windows: tuple[tuple[int, int, int, int], ...]
    canvas_shape_hw: tuple[int, int]
    canvas_origin_xy_px: tuple[int, int]
    valid_mask: Tensor
    target_xy_px: Tensor | None
    target_present: bool
    tile_target_xy_px: Tensor | None = None
    source_point_px: Tensor | None = None
    augmentation_geometry: GeometryChoice = GeometryChoice(0, False, False)
    unaugmented_canvas_shape_hw: tuple[int, int] | None = None


def train_search_from_config(
    config: str | Path,
    *,
    initialize_from: str | Path,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Fine-tune prediction heads using differentiable operational searches."""

    config_path = Path(config).expanduser().resolve()
    payload = load_search_training_config(config_path)
    # Project configuration paths follow the existing CLI convention and are
    # resolved from the invocation directory, not from configs/experiments/.
    settings = search_training_settings(payload, Path.cwd())
    target_device = resolve_device(device or "auto")
    if output_dir is None:
        run_dir = settings.output_dir / settings.run_name
    else:
        run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_tracker_checkpoint(initialize_from, device=target_device)
    model = loaded.model
    _freeze_for_search_finetuning(model, settings.trainable_modules)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise SearchTrainingError("search fine-tuning has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    use_amp = settings.precision in {"auto", "fp16"} and target_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    initial_payload = torch.load(
        Path(initialize_from), map_location="cpu", weights_only=False
    )
    model_spec = initial_payload.get("model")
    if not isinstance(model_spec, dict):
        raise SearchTrainingError(
            "initial checkpoint does not contain a model specification"
        )
    effective_config = _effective_checkpoint_config(payload, settings)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8"
    )
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)

    history: list[dict[str, Any]] = []
    best_score: tuple[float, float] | None = None
    best_checkpoint = run_dir / "best.pt"
    last_checkpoint = run_dir / "last.pt"
    with _EpisodeReader(settings) as reader:
        initial_validation = _run_phase(
            model,
            reader,
            settings,
            phase="validation",
            epoch=0,
            device=target_device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
        )
        best_score = _selection_score(initial_validation)
        _save_checkpoint(
            best_checkpoint,
            model=model,
            model_spec=model_spec,
            config=effective_config,
            optimizer=optimizer,
            epoch=-1,
            history=history,
            validation=initial_validation,
            initialized_from=initialize_from,
        )
        print(
            json.dumps(
                {"epoch": 0, "phase": "initial_validation", **initial_validation}
            )
        )
        for epoch in range(settings.epochs):
            training = _run_phase(
                model,
                reader,
                settings,
                phase="train",
                epoch=epoch,
                device=target_device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
            )
            validation = _run_phase(
                model,
                reader,
                settings,
                phase="validation",
                epoch=epoch,
                device=target_device,
                optimizer=None,
                scaler=None,
                use_amp=use_amp,
            )
            record = {
                "epoch": epoch + 1,
                "training": training,
                "validation": validation,
            }
            history.append(record)
            (run_dir / "history.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "initial_validation": initial_validation,
                        "epochs": history,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            score = _selection_score(validation)
            if score > best_score:
                best_score = score
                _save_checkpoint(
                    best_checkpoint,
                    model=model,
                    model_spec=model_spec,
                    config=effective_config,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    validation=validation,
                    initialized_from=initialize_from,
                )
            _save_checkpoint(
                last_checkpoint,
                model=model,
                model_spec=model_spec,
                config=effective_config,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                validation=validation,
                initialized_from=initialize_from,
            )
            print(json.dumps(record, sort_keys=True))
    return {
        "schema_version": 1,
        "run_dir": str(run_dir.resolve()),
        "best_checkpoint": str(best_checkpoint.resolve()),
        "last_checkpoint": str(last_checkpoint.resolve()),
        "epochs_completed": settings.epochs,
        "initial_validation": initial_validation,
        "best_validation": _best_validation(initial_validation, history),
        "device": str(target_device),
        "precision": "fp16" if use_amp else "fp32",
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }


class _EpisodeReader:
    """Mix survey pairs while keeping just one pair's rasters open at a time."""

    def __init__(self, settings: SearchTrainingSettings):
        import pandas as pd

        self.settings = settings
        self.readers = [_SinglePairReader(settings)] + [
            _SinglePairReader(replace(settings, **pair, additional_pairs=()))
            for pair in settings.additional_pairs
        ]
        self.frames = {
            phase: pd.concat(
                [
                    reader.frames[phase].assign(episode_pair_index=index)
                    for index, reader in enumerate(self.readers)
                ],
                ignore_index=True,
            )
            for phase in ("train", "validation")
        }
        self.active = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.active is not None:
            self.readers[self.active].__exit__(*args)
            self.active = None

    def rows(self, phase, epoch):
        return _SinglePairReader.rows(self, phase, epoch)

    def episode(self, row, **kwargs):
        index = int(row.episode_pair_index)
        if self.active != index:
            self.__exit__()
            self.readers[index].__enter__()
            self.active = index
        return self.readers[index].episode(row, **kwargs)


class _SinglePairReader:
    def __init__(self, settings: SearchTrainingSettings) -> None:
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "search training requires the geo dependencies"
            ) from error
        self.settings = settings
        self.frames = {
            "train": pd.read_parquet(settings.train_queries),
            "validation": pd.read_parquet(settings.validation_queries),
        }
        required = {
            "query_id",
            "source_x_m",
            "source_y_m",
            "target_x_m",
            "target_y_m",
            "match_present",
        }
        for phase, frame in self.frames.items():
            missing = required.difference(frame.columns)
            if missing:
                raise SearchTrainingError(
                    f"{phase} query table is missing columns: {sorted(missing)}"
                )
        self.source = AnalysisRaster(
            settings.source_raster,
            tensor_gsd_m_per_px=settings.tensor_gsd_m_per_px,
        )
        self.target = AnalysisRaster(
            settings.target_raster,
            tensor_gsd_m_per_px=settings.tensor_gsd_m_per_px,
        )
        self.aoi_mask: Any = None

    def __enter__(self) -> _SinglePairReader:
        self.source.__enter__()
        try:
            self.target.__enter__()
            # Use the exact same global coarse mask as operational inference.
            # Re-rasterizing separately for each local window can change edge
            # inclusion and silently remove hard distractors.
            self.aoi_mask = self.target.aoi_mask(
                self.settings.target_aoi,
                stride_px=self.settings.stitch_stride_px,
            )
        except Exception:
            self.source.close()
            self.target.close()
            raise
        return self

    def __exit__(self, *args: Any) -> None:
        self.source.close()
        self.target.close()

    def rows(self, phase: str, epoch: int) -> list[Any]:
        frame = self.frames[phase]
        limit = (
            self.settings.train_query_limit
            if phase == "train"
            else self.settings.validation_query_limit
        )
        indices = list(range(len(frame)))
        if phase == "train":
            generator = torch.Generator().manual_seed(self.settings.seed + epoch)
            indices = torch.randperm(len(indices), generator=generator).tolist()
        if limit is not None:
            indices = indices[:limit]
        return [frame.iloc[index] for index in indices]

    def episode(self, row: Any, *, phase: str, epoch: int, index: int) -> SearchEpisode:
        from pyproj import Transformer

        settings = self.settings
        radius = settings.radius_m
        if phase == "train" and settings.training_radii_m:
            generator = torch.Generator().manual_seed(
                settings.seed + 10_000_019 * epoch + 104_729 * index
            )
            radius = settings.training_radii_m[
                int(
                    torch.randint(
                        len(settings.training_radii_m), (1,), generator=generator
                    )
                )
            ]
        source_x = float(row.source_x_m)
        source_y = float(row.source_y_m)
        if str(self.source.crs) != str(self.target.crs):
            transformer = Transformer.from_crs(
                self.source.crs, self.target.crs, always_xy=True
            )
            target_source_x, target_source_y = transformer.transform(source_x, source_y)
        else:
            target_source_x, target_source_y = source_x, source_y
        reference, source_point = self.source.read_centered_reference(
            (source_x, source_y), settings.reference_size_hw
        )
        prompt_augmentation = (
            settings.prompt_augmentation
            if phase == "train"
            else PromptAugmentation(enabled=False)
        )
        prompt_generator = torch.Generator().manual_seed(
            settings.seed + 1_000_003 * epoch + 97_409 * index
        )
        regenerate_sam2_prompt = (
            settings.prompt_config.backend == "sam2"
            and phase == "train"
            and settings.train_augmentation.enabled
        )
        prompt = (
            torch.zeros((1, *settings.reference_size_hw), dtype=reference.dtype)
            if regenerate_sam2_prompt
            else render_source_prompt(
                settings.reference_size_hw,
                source_point,
                gsd_m_per_px=settings.tensor_gsd_m_per_px,
                major_axis_m=_optional_float(row, "source_major_axis_m"),
                minor_axis_m=_optional_float(row, "source_minor_axis_m"),
                orientation_deg=_optional_float(row, "source_orientation_deg"),
                config=settings.prompt_config,
                image=reference,
                augmentation=prompt_augmentation,
                approach="gaussian_point",
                generator=prompt_generator,
            )
        )
        source_target_px = self.target.xy_to_pixel(target_source_x, target_source_y)
        search = SearchConfig(
            mode="radius",
            tile_size_px=settings.tile_size_hw,
            tile_overlap=settings.tile_overlap,
            tile_batch_size=settings.encoder_batch_size,
            radius_m=radius,
            match_logit_weight=settings.match_logit_weight,
        )
        rings = plan_search(
            self.target.shape_hw,
            source_target_px,
            gsd_xy_m_per_px=(
                settings.tensor_gsd_m_per_px,
                settings.tensor_gsd_m_per_px,
            ),
            config=search,
        )
        tiles = tuple(
            tile for tile in rings[0].tiles if self._tile_intersects_aoi(tile)
        )
        if not tiles:
            raise SearchTrainingError(f"query {row.query_id} generated no target tiles")
        if any((tile.height, tile.width) != settings.tile_size_hw for tile in tiles):
            raise SearchTrainingError(
                f"query {row.query_id} reaches a raster boundary; partial training tiles "
                "are not supported"
            )
        stride = settings.stitch_stride_px
        top = min(tile.row_off for tile in tiles)
        left = min(tile.col_off for tile in tiles)
        bottom = max(tile.row_off + tile.height for tile in tiles)
        right = max(tile.col_off + tile.width for tile in tiles)
        top = (top // stride) * stride
        left = (left // stride) * stride
        bottom = min(self.target.height, math.ceil(bottom / stride) * stride)
        right = min(self.target.width, math.ceil(right / stride) * stride)
        canvas = self.target.read_window(top, left, bottom - top, right - left)
        local_windows = tuple(
            (
                tile.row_off - top,
                tile.col_off - left,
                tile.height,
                tile.width,
            )
            for tile in tiles
        )
        mask_top = top // stride
        mask_left = left // stride
        mask_bottom = math.ceil(bottom / stride)
        mask_right = math.ceil(right / stride)
        valid = self.aoi_mask[mask_top:mask_bottom, mask_left:mask_right].copy()
        source_local = torch.tensor(
            [source_target_px[0] - left, source_target_px[1] - top],
            dtype=torch.float32,
        )
        target_present = bool(row.match_present)
        target_local = None
        if target_present:
            target_global = self.target.xy_to_pixel(
                float(row.target_x_m), float(row.target_y_m)
            )
            target_local = torch.tensor(
                [target_global[0] - left, target_global[1] - top], dtype=torch.float32
            )
        else:
            target_local = torch.tensor([math.nan, math.nan], dtype=torch.float32)
        generator = torch.Generator().manual_seed(
            settings.seed + 10_000_019 * epoch + 104_729 * index
        )
        augmentation = (
            settings.train_augmentation
            if phase == "train"
            else ImageAugmentation(enabled=False)
        )
        original_canvas_shape = tuple(canvas.shape[-2:])
        (
            reference,
            prompt,
            transformed_source,
            canvas,
            transformed_target,
            valid_tensor,
            geometry,
        ) = augment_search_episode(
            reference,
            prompt,
            torch.as_tensor(source_point, dtype=torch.float32),
            canvas,
            target_local,
            torch.from_numpy(valid)[None],
            config=augmentation,
            generator=generator,
        )
        if regenerate_sam2_prompt:
            source_prompt_generator = torch.Generator().manual_seed(
                settings.seed + 1_000_003 * epoch + 97_409 * index
            )
            prompt = render_source_prompt(
                tuple(reference.shape[-2:]),
                transformed_source,
                gsd_m_per_px=settings.tensor_gsd_m_per_px,
                major_axis_m=_optional_float(row, "source_major_axis_m"),
                minor_axis_m=_optional_float(row, "source_minor_axis_m"),
                orientation_deg=_optional_float(row, "source_orientation_deg"),
                config=settings.prompt_config,
                image=reference,
                augmentation=prompt_augmentation,
                approach="gaussian_point",
                generator=source_prompt_generator,
            )
        source_local = _transform_point(source_local, original_canvas_shape, geometry)
        local_windows = tuple(
            transform_window(window, original_canvas_shape, geometry)
            for window in local_windows
        )
        valid_tensor = valid_tensor.squeeze(0).bool()
        valid_tensor &= _radius_mask(
            valid_tensor.shape,
            source_local,
            stride_px=stride,
            radius_px=radius / settings.tensor_gsd_m_per_px,
        )
        target_value = transformed_target if target_present else None
        if target_value is not None:
            target_index = (
                int(float(target_value[1]) // stride),
                int(float(target_value[0]) // stride),
            )
            if (
                target_index[0] < 0
                or target_index[1] < 0
                or target_index[0] >= valid_tensor.shape[0]
                or target_index[1] >= valid_tensor.shape[1]
                or not bool(valid_tensor[target_index])
            ):
                # No match in the current search area. Keep the episode to learn
                # rejection, but retain tile truth if a peripheral tile sees it.
                target_value = None
                target_present = False
        target_tiles = torch.stack(
            tuple(
                canvas[:, row_off : row_off + height, col_off : col_off + width]
                for row_off, col_off, height, width in local_windows
            )
        )
        return SearchEpisode(
            query_id=str(row.query_id),
            reference_image=reference,
            source_prompt=prompt,
            source_point_px=transformed_source,
            target_tiles=target_tiles,
            tile_windows=local_windows,
            canvas_shape_hw=tuple(canvas.shape[-2:]),
            canvas_origin_xy_px=(left, top),
            valid_mask=valid_tensor,
            target_xy_px=target_value,
            target_present=target_present,
            tile_target_xy_px=transformed_target if bool(row.match_present) else None,
            augmentation_geometry=geometry,
            unaugmented_canvas_shape_hw=original_canvas_shape,
        )

    def _tile_intersects_aoi(self, tile: Any) -> bool:
        stride = self.settings.stitch_stride_px
        top = tile.row_off // stride
        left = tile.col_off // stride
        bottom = math.ceil((tile.row_off + tile.height) / stride)
        right = math.ceil((tile.col_off + tile.width) / stride)
        return bool(self.aoi_mask[top:bottom, left:right].any())


def _run_phase(
    model: PromptedSiameseTracker,
    reader: _EpisodeReader,
    settings: SearchTrainingSettings,
    *,
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
    totals = {
        "total": 0.0,
        "global_localization": 0.0,
        "tile_presence": 0.0,
        "positive_localization": 0.0,
        "negative_suppression": 0.0,
        "global_ranking": 0.0,
    }
    errors: list[float] = []
    tile_counts: list[int] = []
    skipped = 0
    for index, row in enumerate(rows):
        try:
            episode = reader.episode(row, phase=phase, epoch=epoch, index=index)
        except SearchTrainingError as error:
            if "outside the valid radius/AOI" in str(error):
                skipped += 1
                continue
            raise
        if training:
            optimizer.zero_grad(set_to_none=True)
        # ConvNeXt and the feature pyramid remain reusable constants.  An
        # explicitly trainable search projection is applied outside no_grad so
        # the descriptor space can adapt to full-search hard negatives.
        projection_trainable = "search_projection" in settings.trainable_modules
        with torch.no_grad():
            reference_features = model._encode_image(
                episode.reference_image[None].to(device)
            )
        projection_context = nullcontext() if training else torch.no_grad()
        with projection_context:
            reference = model.encode_reference_features(
                reference_features,
                episode.source_prompt[None].to(device),
                **(
                    {"source_point_px": episode.source_point_px[None].to(device)}
                    if model.matcher == "masked_xcorr_v6"
                    else {}
                ),
            )
        encoded_tiles: list[Tensor] = []
        output_size = None
        for start in range(0, len(episode.target_tiles), settings.encoder_batch_size):
            images = episode.target_tiles[
                start : start + settings.encoder_batch_size
            ].to(device)
            with torch.no_grad():
                tile_features = model._encode_image(images)
            with projection_context:
                encoded = model.encode_search_features(
                    tile_features, tuple(images.shape[-2:])
                )
            encoded_tiles.append(
                encoded.features
                if projection_trainable and training
                else encoded.features.detach()
            )
            output_size = encoded.output_size_hw
        if output_size is None:
            raise SearchTrainingError(f"query {episode.query_id} has no encoded tiles")
        # Training may use mixed precision, but deterministic validation must
        # match the operational evaluator, which runs the matcher in fp32.
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp and training
            else nullcontext()
        )
        grad_context = nullcontext() if training else torch.no_grad()
        with grad_context, context:
            output = model.match_encoded(
                reference,
                SearchEncoding(torch.cat(encoded_tiles), output_size),
            )
            stitched = differentiable_stitch(
                output.location_logits,
                output.match_logits,
                episode.tile_windows,
                episode.canvas_shape_hw,
                stride_px=settings.stitch_stride_px,
                match_logit_weight=settings.match_logit_weight,
                valid_mask=episode.valid_mask.to(device),
            )
            losses = search_episode_loss(
                stitched,
                output.location_logits,
                output.match_logits,
                episode.tile_windows,
                episode.target_xy_px,
                stride_px=settings.stitch_stride_px,
                target_sigma_px=settings.target_sigma_m / settings.tensor_gsd_m_per_px,
                global_weight=settings.global_weight,
                presence_weight=settings.presence_weight,
                positive_localization_weight=settings.positive_localization_weight,
                negative_suppression_weight=settings.negative_suppression_weight,
                ranking_weight=settings.ranking_weight,
                ranking_margin=settings.ranking_margin,
                ranking_exclusion_radius_px=(
                    None
                    if settings.ranking_exclusion_radius_m is None
                    else settings.ranking_exclusion_radius_m
                    / settings.tensor_gsd_m_per_px
                ),
                negative_temperature=settings.negative_temperature,
                tile_target_xy_px=episode.tile_target_xy_px,
                separate_tile_target=True,
            )
        if not bool(torch.isfinite(losses.total)):
            raise SearchTrainingError(f"non-finite loss for query {episode.query_id}")
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(losses.total).backward()
                scaler.unscale_(optimizer)
                _clip_gradients(model, settings.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.total.backward()
                _clip_gradients(model, settings.gradient_clip_norm)
                optimizer.step()
        _accumulate_loss(totals, losses)
        tile_counts.append(len(episode.tile_windows))
        if episode.target_xy_px is not None:
            flat = stitched.logits.detach().flatten().argmax()
            coarse_y = int(flat) // stitched.logits.shape[1]
            coarse_x = int(flat) % stitched.logits.shape[1]
            center = (settings.stitch_stride_px - 1) / 2
            predicted_x = coarse_x * settings.stitch_stride_px + center
            predicted_y = coarse_y * settings.stitch_stride_px + center
            error = (
                math.hypot(
                    predicted_x - float(episode.target_xy_px[0]),
                    predicted_y - float(episode.target_xy_px[1]),
                )
                * settings.tensor_gsd_m_per_px
            )
            errors.append(error)
    count = len(errors)
    positive_queries = sum(bool(row.match_present) for row in rows)
    episode_count = len(tile_counts)
    if episode_count == 0:
        raise SearchTrainingError(f"{phase} produced no valid episodes")
    ordered = sorted(errors)
    metrics = {name: value / episode_count for name, value in totals.items()}
    metrics.update(
        {
            "episodes": float(episode_count),
            "positive_episodes": float(count),
            "no_match_episodes": float(episode_count - count),
            "skipped_episodes": float(skipped),
            "mean_tiles": sum(tile_counts) / len(tile_counts),
            "mean_error_m": sum(errors) / count if count else math.nan,
            "median_error_m": _quantile(ordered, 0.5) if count else math.nan,
            "p90_error_m": _quantile(ordered, 0.9) if count else math.nan,
            "success_at_0_5m": sum(value <= 0.5 for value in errors)
            / max(positive_queries, 1),
            "success_at_1m": sum(value <= 1.0 for value in errors)
            / max(positive_queries, 1),
            "success_at_2m": sum(value <= 2.0 for value in errors)
            / max(positive_queries, 1),
            "success_at_5m": sum(value <= 5.0 for value in errors)
            / max(positive_queries, 1),
            "success_at_10m": sum(value <= 10.0 for value in errors)
            / max(positive_queries, 1),
        }
    )
    return metrics


def search_training_settings(
    payload: dict[str, Any], config_directory: Path | None = None
) -> SearchTrainingSettings:
    section = payload.get("search_training")
    if not isinstance(section, dict):
        raise ConfigError("search_training must be a mapping")
    base = config_directory or Path.cwd()
    pairs = section.get("pairs", [])
    pair_keys = (
        "train_queries",
        "validation_queries",
        "source_raster",
        "target_raster",
        "target_aoi",
    )
    if not isinstance(pairs, list) or any(
        not isinstance(pair, dict) or set(pair) != set(pair_keys) for pair in pairs
    ):
        raise ConfigError(
            f"search_training.pairs must list mappings containing {pair_keys}"
        )
    if pairs:
        section = {**section, **pairs[0]}
    project = payload.get("project", {})
    data = payload.get("data", {})
    sampling = payload.get("sampling", {})
    prompt = payload.get("prompt", {})
    inference = payload.get("inference", {})
    search = inference.get("search", {}) if isinstance(inference, dict) else {}
    training = payload.get("training", {})
    augmentation = training.get("image_augmentation", {}).get("train", {})
    masking = training.get("masking", {}).get("train", {}).get("augmentation", {})
    loss = section.get("loss", {})
    try:
        settings = SearchTrainingSettings(
            train_queries=_path(section["train_queries"], base),
            validation_queries=_path(section["validation_queries"], base),
            source_raster=_path(section["source_raster"], base),
            target_raster=_path(section["target_raster"], base),
            target_aoi=_path(section["target_aoi"], base),
            output_dir=_path(project.get("output_dir", "runs"), base),
            run_name=str(section.get("run_name", "search_episode")),
            seed=int(project.get("seed", 0)),
            epochs=int(section.get("epochs", 3)),
            learning_rate=float(section.get("learning_rate", 1e-5)),
            weight_decay=float(section.get("weight_decay", 1e-4)),
            gradient_clip_norm=(
                None
                if section.get("gradient_clip_norm", 1.0) is None
                else float(section.get("gradient_clip_norm", 1.0))
            ),
            precision=str(section.get("precision", training.get("precision", "auto"))),
            tensor_gsd_m_per_px=float(data.get("tensor_gsd_m_per_px", 0.1)),
            reference_size_hw=_int_pair(
                sampling.get("reference_crop_size_px", [384, 384])
            ),
            radius_m=float(section.get("radius_m", search.get("radius_m", 75.0))),
            tile_size_hw=_int_pair(
                section.get("tile_size_px", search.get("tile_size_px", [512, 512]))
            ),
            tile_overlap=float(
                section.get("tile_overlap", search.get("tile_overlap", 0.5))
            ),
            encoder_batch_size=int(
                section.get("encoder_batch_size", search.get("tile_batch_size", 8))
            ),
            stitch_stride_px=int(
                section.get("stitch_stride_px", inference.get("stitch_stride_px", 4))
            ),
            match_logit_weight=float(
                section.get(
                    "match_logit_weight", inference.get("match_logit_weight", 0.0)
                )
            ),
            target_sigma_m=float(
                section.get("target_sigma_m", prompt.get("target_sigma_m", 0.35))
            ),
            global_weight=float(loss.get("global_weight", 1.0)),
            presence_weight=float(loss.get("presence_weight", 0.25)),
            positive_localization_weight=float(
                loss.get("positive_localization_weight", 0.25)
            ),
            negative_suppression_weight=float(
                loss.get("negative_suppression_weight", 0.10)
            ),
            ranking_weight=float(loss.get("ranking_weight", 0.25)),
            ranking_margin=float(loss.get("ranking_margin", 1.0)),
            ranking_exclusion_radius_m=_optional_float_value(
                loss.get("ranking_exclusion_radius_m")
            ),
            negative_temperature=float(loss.get("negative_temperature", 0.10)),
            trainable_modules=tuple(
                str(value)
                for value in section.get(
                    "trainable_modules",
                    ["relation_encoder", "location_head", "match_head"],
                )
            ),
            train_query_limit=_optional_int(section.get("train_query_limit")),
            validation_query_limit=_optional_int(section.get("validation_query_limit")),
            prompt_config=PromptConfig.from_mapping(prompt, default_sigma=4.75),
            prompt_augmentation=PromptAugmentation(
                enabled=bool(masking.get("enabled", True)),
                center_jitter_m=float(masking.get("center_jitter_m", 0.15)),
                scale_range=_float_pair(masking.get("scale_range", [0.8, 1.25])),
                orientation_jitter_deg=float(
                    masking.get("orientation_jitter_deg", 15.0)
                ),
                morphology_radius_px=_ordered_int_pair(
                    masking.get("morphology_radius_px", [-2, 2])
                ),
                blur_sigma_px=_float_pair(masking.get("blur_sigma_px", [0.0, 1.25])),
            ),
            train_augmentation=_image_augmentation(augmentation),
            additional_pairs=tuple(
                {key: _path(pair[key], base) for key in pair_keys} for pair in pairs[1:]
            ),
            training_radii_m=tuple(
                float(value) for value in section.get("training_radii_m", [])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"invalid search_training configuration: {error}") from error
    for path in (
        settings.train_queries,
        settings.validation_queries,
        settings.source_raster,
        settings.target_raster,
        settings.target_aoi,
        *(path for pair in settings.additional_pairs for path in pair.values()),
    ):
        if not path.is_file():
            raise ConfigError(f"search-training input does not exist: {path}")
    if (
        min(
            settings.epochs,
            settings.learning_rate,
            settings.tensor_gsd_m_per_px,
            settings.radius_m,
            settings.encoder_batch_size,
            settings.stitch_stride_px,
            settings.target_sigma_m,
        )
        <= 0
    ):
        raise ConfigError("search-training sizes, rates, and epochs must be positive")
    if not 0 <= settings.tile_overlap < 1:
        raise ConfigError("search_training.tile_overlap must be in [0, 1)")
    if any(not 0 < radius <= settings.radius_m for radius in settings.training_radii_m):
        raise ConfigError("training radii must be positive and at most radius_m")
    if settings.training_radii_m and not (
        settings.presence_weight > 0 or settings.negative_suppression_weight > 0
    ):
        raise ConfigError("prefix training requires an active no-match loss")
    if settings.precision not in {"auto", "fp16", "fp32"}:
        raise ConfigError("search_training.precision must be auto, fp16, or fp32")
    return settings


def load_search_training_config(path: str | Path) -> dict[str, Any]:
    """Load a search experiment with optional recursive ``extends`` support."""

    config_path = Path(path).expanduser().resolve()
    payload = load_yaml(config_path)
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    parent_path = Path(str(parent)).expanduser()
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    inherited = load_search_training_config(parent_path)
    return _deep_merge(inherited, payload)


def _freeze_for_search_finetuning(
    model: PromptedSiameseTracker, trainable_modules: tuple[str, ...]
) -> None:
    allowed = {
        "input_adapter",
        "relation_encoder",
        "location_head",
        "match_head",
        "search_projection",
    }
    unknown = set(trainable_modules).difference(allowed)
    if unknown:
        raise SearchTrainingError(
            f"unsupported search-training modules: {sorted(unknown)}"
        )
    if not trainable_modules:
        raise SearchTrainingError(
            "search training requires at least one trainable module"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name in trainable_modules:
        module = getattr(model, name, None)
        if module is None:
            raise SearchTrainingError(f"model matcher has no {name} module")
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def _effective_checkpoint_config(
    payload: dict[str, Any], settings: SearchTrainingSettings
) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    inference = result.setdefault("inference", {})
    inference["match_logit_weight"] = settings.match_logit_weight
    search = inference.setdefault("search", {})
    search.update(
        {
            "mode": "radius",
            "radius_m": settings.radius_m,
            "tile_size_px": list(settings.tile_size_hw),
            "tile_overlap": settings.tile_overlap,
        }
    )
    inference["stitch_stride_px"] = settings.stitch_stride_px
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _save_checkpoint(
    path: Path,
    *,
    model: PromptedSiameseTracker,
    model_spec: dict[str, Any],
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    validation: dict[str, float],
    initialized_from: str | Path,
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "package_version": __version__,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "model": model_spec,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "history": history,
            "validation": validation,
            "initialized_from": str(Path(initialized_from).resolve()),
            "training_method": "differentiable_tiled_search_episode_v1",
        },
        path,
    )


def _selection_score(metrics: dict[str, float]) -> tuple[float, float]:
    return metrics["success_at_2m"], -metrics["median_error_m"]


def _best_validation(
    initial: dict[str, float], history: list[dict[str, Any]]
) -> dict[str, float]:
    values = [initial, *(record["validation"] for record in history)]
    return max(values, key=_selection_score)


def _accumulate_loss(totals: dict[str, float], losses: SearchLoss) -> None:
    for name in totals:
        totals[name] += float(getattr(losses, name).detach())


def _clip_gradients(model: PromptedSiameseTracker, maximum_norm: float | None) -> None:
    if maximum_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            maximum_norm,
        )


def _radius_mask(
    shape_hw: tuple[int, int],
    center_xy_px: Tensor,
    *,
    stride_px: int,
    radius_px: float,
) -> Tensor:
    height, width = shape_hw
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    offset = (stride_px - 1) / 2
    xx = xx * stride_px + offset
    yy = yy * stride_px + offset
    return (xx - center_xy_px[0]).square() + (
        yy - center_xy_px[1]
    ).square() <= radius_px**2


def _transform_point(
    point_xy: Tensor, shape_hw: tuple[int, int], choice: GeometryChoice
) -> Tensor:
    x, y = point_xy[0].clone(), point_xy[1].clone()
    height, width = shape_hw
    if choice.quarter_turns == 1:
        x, y = y, width - 1 - x
        height, width = width, height
    elif choice.quarter_turns == 2:
        x, y = width - 1 - x, height - 1 - y
    elif choice.quarter_turns == 3:
        x, y = height - 1 - y, x
        height, width = width, height
    if choice.horizontal_flip:
        x = width - 1 - x
    if choice.vertical_flip:
        y = height - 1 - y
    return torch.stack((x, y))


def _image_augmentation(payload: Any) -> ImageAugmentation:
    if not isinstance(payload, dict):
        payload = {}
    return ImageAugmentation(
        enabled=bool(payload.get("enabled", False)),
        photometric_probability=float(payload.get("photometric_probability", 0.8)),
        brightness_range=_float_pair(payload.get("brightness_range", [0.8, 1.2])),
        contrast_range=_float_pair(payload.get("contrast_range", [0.8, 1.2])),
        saturation_range=_float_pair(payload.get("saturation_range", [0.8, 1.2])),
        hue_range=_float_pair(payload.get("hue_range", [-0.05, 0.05])),
        gamma_range=_float_pair(payload.get("gamma_range", [0.8, 1.2])),
        rotation_degrees=tuple(
            int(value) for value in payload.get("rotation_degrees", [0, 90, 180, 270])
        ),
        horizontal_flip_probability=float(
            payload.get("horizontal_flip_probability", 0.5)
        ),
        vertical_flip_probability=float(payload.get("vertical_flip_probability", 0.5)),
        gaussian_blur_probability=float(payload.get("gaussian_blur_probability", 0.3)),
        gaussian_blur_sigma_px=_float_pair(
            payload.get("gaussian_blur_sigma_px", [0.1, 1.5])
        ),
        gaussian_noise_probability=float(
            payload.get("gaussian_noise_probability", 0.3)
        ),
        gaussian_noise_std=_float_pair(payload.get("gaussian_noise_std", [0.0, 0.03])),
        composite_shadow_probability=float(
            payload.get("composite_shadow_probability", 0.0)
        ),
        cast_shadow_probability=float(payload.get("cast_shadow_probability", 0.0)),
        cast_shadow_coverage_range=_float_pair(
            payload.get("cast_shadow_coverage_range", [0.25, 0.75])
        ),
        cast_shadow_edge_softness_px=_float_pair(
            payload.get("cast_shadow_edge_softness_px", [8.0, 40.0])
        ),
        shadow_transmission_range=_float_pair(
            payload.get("shadow_transmission_range", [0.10, 0.45])
        ),
        shadow_ambient_range=_float_pair(
            payload.get("shadow_ambient_range", [0.002, 0.012])
        ),
        shadow_red_gain_range=_float_pair(
            payload.get("shadow_red_gain_range", [0.72, 0.90])
        ),
        shadow_green_gain_range=_float_pair(
            payload.get("shadow_green_gain_range", [0.85, 1.00])
        ),
        shadow_blue_gain_range=_float_pair(
            payload.get("shadow_blue_gain_range", [1.00, 1.18])
        ),
        shadow_field_strength_range=_float_pair(
            payload.get("shadow_field_strength_range", [0.00, 0.14])
        ),
        shadow_field_grid_size=int(payload.get("shadow_field_grid_size", 8)),
    )


def _optional_float(row: Any, name: str) -> float | None:
    value = getattr(row, name, None)
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _optional_float_value(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("optional float values must be finite or null")
    return result


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError("query limits must be positive or null")
    return result


def _path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _int_pair(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("expected a two-value integer sequence")
    result = int(value[0]), int(value[1])
    if min(result) <= 0:
        raise ValueError("integer pair values must be positive")
    return result


def _float_pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("expected a two-value float sequence")
    return float(value[0]), float(value[1])


def _ordered_int_pair(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("expected a two-value integer sequence")
    result = int(value[0]), int(value[1])
    if result[0] > result[1]:
        raise ValueError("integer pair must be ordered")
    return result


def _quantile(ordered: list[float], probability: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
