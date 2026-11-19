"""Reproducible patch-level training with optional mixed precision."""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler

from glaciertrack import __version__
from glaciertrack.config import ConfigError, load_yaml
from glaciertrack.data.augmentation import ImageAugmentation
from glaciertrack.data.hdf5 import HDF5PairDataset
from glaciertrack.data.prompts import PromptApproach, PromptAugmentation, PromptConfig
from glaciertrack.device import resolve_device
from glaciertrack.models import PromptedSiameseTracker, tracker_loss


class TrainingError(RuntimeError):
    """Raised when a training run cannot be constructed safely."""


TrainingScheduler = LRScheduler | ReduceLROnPlateau


@dataclass(frozen=True)
class MaskingPhaseConfig:
    """On-demand source-prompt sampling for one dataset phase."""

    approaches: tuple[PromptApproach, ...]
    probabilities: tuple[float, ...]
    augmentation: PromptAugmentation
    resample_each_epoch: bool


@dataclass(frozen=True)
class PrecisionPolicy:
    """Resolved autocast and gradient-scaling behavior."""

    name: str
    dtype: torch.dtype | None
    gradient_scaling: bool


@dataclass(frozen=True)
class LRSchedulerSettings:
    """Learning-rate policy driven either by epoch or validation plateau."""

    name: str
    factor: float
    patience: int
    threshold: float
    cooldown: int
    minimum_lr: float


@dataclass(frozen=True)
class EarlyStoppingSettings:
    """Validation-loss stopping rule."""

    enabled: bool
    patience: int
    minimum_delta: float
    minimum_epochs: int


@dataclass(frozen=True)
class CheckpointSelectionSettings:
    """Minimum patch-classification skill required for a deployable checkpoint."""

    minimum_match_recall: float
    minimum_match_specificity: float
    minimum_match_auroc: float
    minimum_best_balanced_accuracy: float
    minimum_localization_success_at_2m: float
    match_auroc_weight: float
    localization_success_weight: float
    minimum_retrieval_top1_accuracy: float = 0.0
    retrieval_top1_weight: float = 0.0


@dataclass(frozen=True)
class TrainingSettings:
    """Validated settings required by the patch trainer."""

    train_hdf5_cache: Path
    validation_hdf5_cache: Path
    sample_manifest: Path | None
    validation_sample_manifest: Path | None
    allow_shared_cache: bool
    output_root: Path
    run_name: str | None
    device: str
    precision: str
    epochs: int
    batch_size: int
    validation_batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    match_positive_weight: float | None
    localization_weight: float
    positive_localization_weight: float
    positive_localization_margin: float
    positive_peak_ranking_weight: float
    positive_peak_ranking_margin: float
    negative_localization_weight: float
    negative_localization_margin: float
    negative_localization_temperature: float
    retrieval_ranking_weight: float
    retrieval_ranking_margin: float
    retrieval_ranking_temperature: float
    retrieval_match_logit_weight: float
    gradient_clip_norm: float | None
    lr_scheduler: LRSchedulerSettings
    early_stopping: EarlyStoppingSettings
    checkpoint_selection: CheckpointSelectionSettings
    seed: int
    backbone: str
    pretrained: bool
    pyramid_channels: int
    matcher: str
    match_channels: int
    input_representation: str
    freeze_feature_extractor: bool
    freeze_backbone: bool
    prompt: PromptConfig
    train_masking: MaskingPhaseConfig
    validation_masking: MaskingPhaseConfig
    train_image_augmentation: ImageAugmentation
    validation_image_augmentation: ImageAugmentation
    simulated_effective_gsd_m_per_px: tuple[float, ...]
    tensor_gsd_m_per_px: float
    expected_reference_size_hw: tuple[int, int] | None
    expected_search_size_hw: tuple[int, int] | None


def training_settings(payload: dict[str, Any]) -> TrainingSettings:
    """Extract trainer settings from a project configuration."""

    data = _mapping(payload, "data")
    project = _mapping(payload, "project")
    model = _mapping(payload, "model")
    training = _mapping(payload, "training")
    sampling = payload.get("sampling", {})
    if not isinstance(sampling, dict):
        raise ConfigError("sampling must be a mapping")
    prompt = _mapping(payload, "prompt")
    masking = _mapping(training, "masking")
    train_masking = _masking_phase(_mapping(masking, "train"), default_seed_phase=True)
    validation_masking = _masking_phase(
        _mapping(masking, "validation"), default_seed_phase=False
    )
    image_augmentation_raw = training.get("image_augmentation", {})
    if not isinstance(image_augmentation_raw, dict):
        raise ConfigError("training.image_augmentation must be a mapping")
    train_image_augmentation = _image_augmentation_phase(
        image_augmentation_raw.get("train", {})
    )
    validation_image_augmentation = _image_augmentation_phase(
        image_augmentation_raw.get("validation", {})
    )
    scheduler_raw = training.get("lr_scheduler", {})
    early_stopping_raw = training.get("early_stopping", {})
    checkpoint_selection_raw = training.get("checkpoint_selection", {})
    if not all(
        isinstance(value, dict)
        for value in (scheduler_raw, early_stopping_raw, checkpoint_selection_raw)
    ):
        raise ConfigError(
            "training.lr_scheduler, early_stopping, and checkpoint_selection "
            "must be mappings"
        )

    train_cache = training.get("train_hdf5_cache")
    validation_cache = training.get("validation_hdf5_cache")
    if not train_cache or not validation_cache:
        raise ConfigError(
            "training.train_hdf5_cache and training.validation_hdf5_cache are required"
        )
    try:
        settings = TrainingSettings(
            train_hdf5_cache=Path(str(train_cache)),
            validation_hdf5_cache=Path(str(validation_cache)),
            sample_manifest=(
                Path(str(training["sample_manifest"]))
                if training.get("sample_manifest")
                else None
            ),
            validation_sample_manifest=(
                Path(str(training["validation_sample_manifest"]))
                if training.get("validation_sample_manifest")
                else (
                    Path(str(training["sample_manifest"]))
                    if training.get("sample_manifest")
                    else None
                )
            ),
            allow_shared_cache=bool(training.get("allow_shared_cache", False)),
            output_root=Path(str(project.get("output_dir", "runs"))),
            run_name=_optional_string(training.get("run_name")),
            device=str(training.get("device", "auto")),
            precision=str(training.get("precision", "auto")).lower(),
            epochs=int(training.get("epochs", 100)),
            batch_size=int(training.get("batch_size", 16)),
            validation_batch_size=int(
                training.get("validation_batch_size", training.get("batch_size", 16))
            ),
            num_workers=int(training.get("num_workers", 4)),
            learning_rate=float(training.get("learning_rate", 1e-4)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            match_positive_weight=_positive_weight(
                training.get("match_positive_weight", "auto")
            ),
            localization_weight=float(training.get("localization_weight", 1.0)),
            positive_localization_weight=float(
                training.get("positive_localization_weight", 0.25)
            ),
            positive_localization_margin=float(
                training.get("positive_localization_margin", 1.0)
            ),
            positive_peak_ranking_weight=float(
                training.get("positive_peak_ranking_weight", 0.0)
            ),
            positive_peak_ranking_margin=float(
                training.get("positive_peak_ranking_margin", 1.0)
            ),
            negative_localization_weight=float(
                training.get("negative_localization_weight", 0.25)
            ),
            negative_localization_margin=float(
                training.get("negative_localization_margin", 0.0)
            ),
            negative_localization_temperature=float(
                training.get("negative_localization_temperature", 0.1)
            ),
            retrieval_ranking_weight=float(
                training.get("retrieval_ranking_weight", 0.0)
            ),
            retrieval_ranking_margin=float(
                training.get("retrieval_ranking_margin", 1.0)
            ),
            retrieval_ranking_temperature=float(
                training.get("retrieval_ranking_temperature", 0.1)
            ),
            retrieval_match_logit_weight=float(
                training.get("retrieval_match_logit_weight", 1.0)
            ),
            gradient_clip_norm=_optional_float(training.get("gradient_clip_norm", 1.0)),
            lr_scheduler=LRSchedulerSettings(
                name=str(scheduler_raw.get("name", "cosine_annealing")).lower(),
                factor=float(scheduler_raw.get("factor", 0.5)),
                patience=int(scheduler_raw.get("patience", 5)),
                threshold=float(scheduler_raw.get("threshold", 1e-3)),
                cooldown=int(scheduler_raw.get("cooldown", 0)),
                minimum_lr=float(scheduler_raw.get("minimum_lr", 1e-7)),
            ),
            early_stopping=EarlyStoppingSettings(
                enabled=bool(early_stopping_raw.get("enabled", False)),
                patience=int(early_stopping_raw.get("patience", 7)),
                minimum_delta=float(early_stopping_raw.get("minimum_delta", 1e-3)),
                minimum_epochs=int(early_stopping_raw.get("minimum_epochs", 8)),
            ),
            checkpoint_selection=CheckpointSelectionSettings(
                minimum_match_recall=float(
                    checkpoint_selection_raw.get("minimum_match_recall", 0.0)
                ),
                minimum_match_specificity=float(
                    checkpoint_selection_raw.get("minimum_match_specificity", 0.0)
                ),
                minimum_match_auroc=float(
                    checkpoint_selection_raw.get("minimum_match_auroc", 0.0)
                ),
                minimum_best_balanced_accuracy=float(
                    checkpoint_selection_raw.get("minimum_best_balanced_accuracy", 0.0)
                ),
                minimum_localization_success_at_2m=float(
                    checkpoint_selection_raw.get(
                        "minimum_localization_success_at_2m", 0.0
                    )
                ),
                match_auroc_weight=float(
                    checkpoint_selection_raw.get("match_auroc_weight", 0.5)
                ),
                localization_success_weight=float(
                    checkpoint_selection_raw.get("localization_success_weight", 0.5)
                ),
                minimum_retrieval_top1_accuracy=float(
                    checkpoint_selection_raw.get("minimum_retrieval_top1_accuracy", 0.0)
                ),
                retrieval_top1_weight=float(
                    checkpoint_selection_raw.get("retrieval_top1_weight", 0.0)
                ),
            ),
            seed=int(project.get("seed", 42)),
            backbone=str(model.get("backbone", "convnext_tiny")),
            pretrained=bool(model.get("pretrained", True)),
            pyramid_channels=int(model.get("pyramid_channels", 128)),
            matcher=str(model.get("matcher", "response_relation_v2")),
            match_channels=int(model.get("match_channels", 64)),
            input_representation=str(model.get("input_representation", "rgb")),
            freeze_feature_extractor=bool(
                training.get("freeze_feature_extractor", False)
            ),
            freeze_backbone=bool(training.get("freeze_backbone", False)),
            prompt=PromptConfig.from_mapping(prompt),
            train_masking=train_masking,
            validation_masking=validation_masking,
            train_image_augmentation=train_image_augmentation,
            validation_image_augmentation=validation_image_augmentation,
            simulated_effective_gsd_m_per_px=tuple(
                float(value)
                for value in data.get("simulated_effective_gsd_m_per_px", ())
            ),
            tensor_gsd_m_per_px=float(data.get("tensor_gsd_m_per_px", 0.1)),
            expected_reference_size_hw=_optional_pixel_size(
                sampling.get("reference_crop_size_px")
            ),
            expected_search_size_hw=_optional_pixel_size(
                sampling.get("search_crop_size_px")
            ),
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid training configuration: {error}") from error
    _validate_training_settings(settings)
    return settings


def resolve_precision(requested: str, device: torch.device) -> PrecisionPolicy:
    """Resolve ``auto``, ``fp32``, ``fp16``, or ``bf16`` for a device."""

    name = requested.strip().lower()
    if name == "auto":
        name = "fp16" if device.type == "cuda" else "fp32"
    if name == "fp32":
        return PrecisionPolicy(name, None, False)
    if name == "fp16":
        if device.type != "cuda":
            raise TrainingError(
                "fp16 mixed precision is currently supported only on CUDA"
            )
        return PrecisionPolicy(name, torch.float16, True)
    if name == "bf16":
        if device.type not in {"cpu", "cuda"}:
            raise TrainingError("bf16 mixed precision requires a CPU or CUDA device")
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise TrainingError("the selected CUDA device does not support bf16")
        return PrecisionPolicy(name, torch.bfloat16, False)
    raise TrainingError("precision must be one of: auto, fp32, fp16, bf16")


def train_from_config(
    config_path: str | Path,
    *,
    resume: str | Path | None = None,
    initialize_from: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run patch-level training and return paths plus the best validation loss."""

    if resume is not None and initialize_from is not None:
        raise TrainingError("resume and initialize_from are mutually exclusive")

    payload = load_yaml(config_path)
    settings = training_settings(payload)
    selected_device = resolve_device(device or settings.device)
    precision = resolve_precision(settings.precision, selected_device)
    _seed_everything(settings.seed, selected_device)

    train_dataset = _dataset(settings, settings.train_hdf5_cache, train=True)
    validation_dataset = _dataset(settings, settings.validation_hdf5_cache, train=False)
    _validate_dataset_dimensions(train_dataset, settings, phase="training")
    _validate_dataset_dimensions(validation_dataset, settings, phase="validation")
    _validate_dataset_pair_dimensions(train_dataset, validation_dataset)
    match_positive_weight = _resolve_match_positive_weight(settings, train_dataset)
    train_loader = _loader(
        train_dataset,
        settings.batch_size,
        shuffle=True,
        settings=settings,
        device=selected_device,
        seed=settings.seed,
    )
    validation_loader = _loader(
        validation_dataset,
        settings.validation_batch_size,
        shuffle=False,
        settings=settings,
        device=selected_device,
        seed=settings.seed + 1,
    )
    model = PromptedSiameseTracker(
        settings.backbone,
        pretrained=settings.pretrained,
        pyramid_channels=settings.pyramid_channels,
        matcher=settings.matcher,
        match_channels=settings.match_channels,
        input_representation=settings.input_representation,
    ).to(selected_device)
    initialization_checkpoint = None
    if initialize_from is not None:
        initialization_checkpoint = str(Path(initialize_from).resolve())
        _initialize_model_weights(Path(initialize_from), model, selected_device)
    if settings.freeze_feature_extractor:
        _freeze_feature_extractor(model)
    elif settings.freeze_backbone:
        _freeze_backbone(model)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = _build_lr_scheduler(optimizer, settings)
    scaler = torch.amp.GradScaler("cuda", enabled=precision.gradient_scaling)

    continuing_in_place = resume is not None and output_dir is None
    run_dir = (
        Path(resume).resolve().parent
        if continuing_in_place
        else _run_directory(output_dir or settings.output_root, settings.run_name)
    )
    run_dir.mkdir(parents=True, exist_ok=continuing_in_place)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    start_epoch = 0
    history: list[dict[str, Any]] = []
    if resume is not None:
        start_epoch, history = _restore_checkpoint(
            Path(resume),
            model,
            optimizer,
            scheduler,
            scaler,
            selected_device,
            settings.lr_scheduler.name,
        )
    if start_epoch >= settings.epochs:
        raise TrainingError(
            f"checkpoint is already at epoch {start_epoch}; configured epochs={settings.epochs}"
        )

    eligible_history = [
        row
        for row in history
        if _checkpoint_eligible(row["validation"], settings.checkpoint_selection)
    ]
    best_score = max(
        (
            _checkpoint_selection_score(
                row["validation"], settings.checkpoint_selection
            )
            for row in eligible_history
        ),
        default=-math.inf,
    )
    best_loss = next(
        (
            float(row["validation"]["total"])
            for row in reversed(eligible_history)
            if _checkpoint_selection_score(
                row["validation"], settings.checkpoint_selection
            )
            == best_score
        ),
        math.inf,
    )
    unconstrained_best_loss = min(
        (float(row["validation"]["total"]) for row in history), default=math.inf
    )
    early_best, epochs_without_improvement = _early_stopping_history_state(
        history, settings.early_stopping.minimum_delta
    )
    stopped_early = False
    for epoch in range(start_epoch, settings.epochs):
        train_dataset.set_epoch(epoch)
        validation_dataset.set_epoch(epoch)
        _set_loader_epoch(train_loader, settings.seed + epoch)
        _set_loader_epoch(validation_loader, settings.seed + 10_000 + epoch)
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=selected_device,
            precision=precision,
            match_positive_weight=match_positive_weight,
            localization_weight=settings.localization_weight,
            positive_localization_weight=settings.positive_localization_weight,
            positive_localization_margin=settings.positive_localization_margin,
            positive_peak_ranking_weight=settings.positive_peak_ranking_weight,
            positive_peak_ranking_margin=settings.positive_peak_ranking_margin,
            negative_localization_weight=settings.negative_localization_weight,
            negative_localization_margin=settings.negative_localization_margin,
            negative_localization_temperature=settings.negative_localization_temperature,
            retrieval_ranking_weight=settings.retrieval_ranking_weight,
            retrieval_ranking_margin=settings.retrieval_ranking_margin,
            retrieval_ranking_temperature=settings.retrieval_ranking_temperature,
            retrieval_match_logit_weight=settings.retrieval_match_logit_weight,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=settings.gradient_clip_norm,
            tensor_gsd_m_per_px=settings.tensor_gsd_m_per_px,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            device=selected_device,
            precision=precision,
            match_positive_weight=match_positive_weight,
            localization_weight=settings.localization_weight,
            positive_localization_weight=settings.positive_localization_weight,
            positive_localization_margin=settings.positive_localization_margin,
            positive_peak_ranking_weight=settings.positive_peak_ranking_weight,
            positive_peak_ranking_margin=settings.positive_peak_ranking_margin,
            negative_localization_weight=settings.negative_localization_weight,
            negative_localization_margin=settings.negative_localization_margin,
            negative_localization_temperature=settings.negative_localization_temperature,
            retrieval_ranking_weight=settings.retrieval_ranking_weight,
            retrieval_ranking_margin=settings.retrieval_ranking_margin,
            retrieval_ranking_temperature=settings.retrieval_ranking_temperature,
            retrieval_match_logit_weight=settings.retrieval_match_logit_weight,
            tensor_gsd_m_per_px=settings.tensor_gsd_m_per_px,
        )
        _step_scheduler(scheduler, validation_metrics["total"])
        validation_loss = float(validation_metrics["total"])
        if validation_loss < early_best - settings.early_stopping.minimum_delta:
            early_best = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint_eligible = _checkpoint_eligible(
            validation_metrics, settings.checkpoint_selection
        )
        selection_score = _checkpoint_selection_score(
            validation_metrics, settings.checkpoint_selection
        )
        record = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "match_positive_weight": match_positive_weight,
            "checkpoint_eligible": checkpoint_eligible,
            "checkpoint_selection_score": selection_score,
            "epochs_without_improvement": epochs_without_improvement,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        checkpoint = _checkpoint_payload(
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            history,
            payload,
            settings,
            precision,
        )
        _save_checkpoint(run_dir / "last.pt", checkpoint)
        if validation_metrics["total"] < unconstrained_best_loss:
            unconstrained_best_loss = validation_metrics["total"]
            _save_checkpoint(run_dir / "best_unconstrained.pt", checkpoint)
        if checkpoint_eligible and selection_score > best_score:
            best_score = selection_score
            best_loss = validation_metrics["total"]
            _save_checkpoint(run_dir / "best.pt", checkpoint)
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        stopped_early = bool(
            settings.early_stopping.enabled
            and epoch + 1 >= settings.early_stopping.minimum_epochs
            and epochs_without_improvement >= settings.early_stopping.patience
        )
        if stopped_early:
            break

    train_dataset.close()
    validation_dataset.close()
    best_checkpoint = None
    if math.isfinite(best_loss):
        best_checkpoint = str((run_dir / "best.pt").resolve())
    return {
        "run_dir": str(run_dir.resolve()),
        "best_checkpoint": best_checkpoint,
        "best_unconstrained_checkpoint": str(
            (run_dir / "best_unconstrained.pt").resolve()
        ),
        "last_checkpoint": str((run_dir / "last.pt").resolve()),
        "best_validation_loss": best_loss if math.isfinite(best_loss) else None,
        "best_checkpoint_selection_score": (
            best_score if math.isfinite(best_score) else None
        ),
        "best_unconstrained_validation_loss": unconstrained_best_loss,
        "checkpoint_selection_status": (
            "eligible_checkpoint_saved"
            if best_checkpoint is not None
            else "no_epoch_met_match_discrimination_requirements"
        ),
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "stopping_reason": (
            "validation_plateau" if stopped_early else "maximum_epochs"
        ),
        "device": str(selected_device),
        "precision": precision.name,
        "match_positive_weight": match_positive_weight,
        "initialization_checkpoint": initialization_checkpoint,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    match_positive_weight: float,
    localization_weight: float,
    positive_localization_weight: float,
    positive_localization_margin: float,
    positive_peak_ranking_weight: float,
    positive_peak_ranking_margin: float,
    negative_localization_weight: float,
    negative_localization_margin: float,
    negative_localization_temperature: float,
    retrieval_ranking_weight: float,
    retrieval_ranking_margin: float,
    retrieval_ranking_temperature: float,
    retrieval_match_logit_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float | None = None,
    tensor_gsd_m_per_px: float = 0.1,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        for module in getattr(model, "_frozen_training_modules", ()):
            module.eval()
    totals = {
        "total": 0.0,
        "match": 0.0,
        "localization": 0.0,
        "positive_localization": 0.0,
        "positive_peak_ranking": 0.0,
        "negative_localization": 0.0,
        "retrieval_ranking": 0.0,
    }
    samples = 0
    true_positive = false_positive = true_negative = false_negative = 0
    epoch_match_logits: list[np.ndarray] = []
    epoch_match_targets: list[np.ndarray] = []
    localization_errors_px: list[np.ndarray] = []
    retrieval_correct = 0
    retrieval_groups = 0
    retrieval_margins: list[float] = []
    context = nullcontext if training else torch.no_grad
    with context():
        for batch in loader:
            reference = _to_device(batch["reference_image"], device)
            prompt = _to_device(batch["source_prompt"], device)
            search = _to_device(batch["search_image"], device)
            matches = _to_device(batch["match_present"], device)
            target = _to_device(batch["target_heatmap"], device)
            if retrieval_ranking_weight > 0:
                reference, prompt = _share_group_reference(
                    reference, prompt, batch.get("group_id"), matches
                )
            active_group_ids = (
                batch.get("group_id") if retrieval_ranking_weight > 0 else None
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                if getattr(model, "matcher", None) == "masked_xcorr_v6":
                    if retrieval_ranking_weight > 0:
                        raise ValueError(
                            "masked_xcorr_v6 does not support shared-group references"
                        )
                    output = model(
                        reference,
                        prompt,
                        search,
                        source_point_px=_to_device(batch["source_point_px"], device),
                    )
                else:
                    output = model(reference, prompt, search)
                losses = tracker_loss(
                    output,
                    match_present=matches,
                    target_heatmap=target,
                    match_positive_weight=match_positive_weight,
                    localization_weight=localization_weight,
                    positive_localization_weight=positive_localization_weight,
                    positive_localization_margin=positive_localization_margin,
                    positive_peak_ranking_weight=positive_peak_ranking_weight,
                    positive_peak_ranking_margin=positive_peak_ranking_margin,
                    negative_localization_weight=negative_localization_weight,
                    negative_localization_margin=negative_localization_margin,
                    negative_localization_temperature=negative_localization_temperature,
                    group_ids=active_group_ids,
                    retrieval_ranking_weight=retrieval_ranking_weight,
                    retrieval_ranking_margin=retrieval_ranking_margin,
                    retrieval_ranking_temperature=retrieval_ranking_temperature,
                    retrieval_match_logit_weight=retrieval_match_logit_weight,
                )
            if not bool(torch.isfinite(losses.total)):
                raise TrainingError("encountered a non-finite loss")
            if training:
                if scaler is not None and precision.gradient_scaling:
                    scaler.scale(losses.total).backward()
                    scaler.unscale_(optimizer)
                    _clip_gradients(model, gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses.total.backward()
                    _clip_gradients(model, gradient_clip_norm)
                    optimizer.step()
            batch_size = int(reference.shape[0])
            samples += batch_size
            predicted_matches = output.match_logits.detach().reshape(-1) >= 0
            actual_matches = matches.detach().reshape(-1).bool()
            if active_group_ids is not None:
                group_result = _batch_retrieval_metrics(
                    output.match_logits.detach(),
                    output.location_logits.detach(),
                    actual_matches,
                    active_group_ids,
                    match_logit_weight=retrieval_match_logit_weight,
                )
                retrieval_correct += group_result[0]
                retrieval_groups += group_result[1]
                retrieval_margins.extend(group_result[2])
            epoch_match_logits.append(
                output.match_logits.detach().float().reshape(-1).cpu().numpy()
            )
            epoch_match_targets.append(actual_matches.cpu().numpy())
            if bool(actual_matches.any()):
                predicted_heatmaps = output.location_logits.detach()[actual_matches]
                target_heatmaps = target.detach()[actual_matches]
                if target_heatmaps.ndim == 4:
                    target_heatmaps = target_heatmaps[:, 0]
                width = predicted_heatmaps.shape[-1]
                predicted_flat = predicted_heatmaps.flatten(1).argmax(1)
                target_flat = target_heatmaps.flatten(1).argmax(1)
                predicted_y = torch.div(predicted_flat, width, rounding_mode="floor")
                predicted_x = predicted_flat.remainder(width)
                target_y = torch.div(target_flat, width, rounding_mode="floor")
                target_x = target_flat.remainder(width)
                errors = torch.sqrt(
                    (predicted_x - target_x).float().square()
                    + (predicted_y - target_y).float().square()
                )
                localization_errors_px.append(errors.cpu().numpy())
            true_positive += int((predicted_matches & actual_matches).sum())
            false_positive += int((predicted_matches & ~actual_matches).sum())
            true_negative += int((~predicted_matches & ~actual_matches).sum())
            false_negative += int((~predicted_matches & actual_matches).sum())
            for name in totals:
                value = getattr(losses, name)
                totals[name] += float(value.detach()) * batch_size
    if samples == 0:
        raise TrainingError("a training or validation dataset is empty")
    metrics = {name: value / samples for name, value in totals.items()}
    recall = _safe_rate(true_positive, true_positive + false_negative)
    specificity = _safe_rate(true_negative, true_negative + false_positive)
    ranking = _binary_ranking_metrics(
        np.concatenate(epoch_match_logits), np.concatenate(epoch_match_targets)
    )
    metrics.update(
        {
            "match_accuracy": (true_positive + true_negative) / samples,
            "match_precision": _safe_rate(
                true_positive, true_positive + false_positive
            ),
            "match_recall": recall,
            "match_specificity": specificity,
            "match_balanced_accuracy": (recall + specificity) / 2,
            "match_false_positive_rate": _safe_rate(
                false_positive, false_positive + true_negative
            ),
            "positive_samples": float(true_positive + false_negative),
            "negative_samples": float(true_negative + false_positive),
        }
    )
    metrics.update(ranking)
    if localization_errors_px:
        errors_px = np.concatenate(localization_errors_px).astype(np.float64)
        errors_m = errors_px * float(tensor_gsd_m_per_px)
        metrics.update(
            {
                "localization_mean_error_m": float(errors_m.mean()),
                "localization_median_error_m": float(np.median(errors_m)),
                "localization_p90_error_m": float(np.quantile(errors_m, 0.9)),
                **{
                    f"localization_success_at_{threshold:g}m": float(
                        (errors_m <= threshold).mean()
                    )
                    for threshold in (0.5, 1.0, 2.0, 5.0, 10.0)
                },
            }
        )
    if retrieval_groups:
        metrics["retrieval_top1_accuracy"] = retrieval_correct / retrieval_groups
        metrics["retrieval_mean_hard_margin"] = float(np.mean(retrieval_margins))
        metrics["retrieval_groups"] = float(retrieval_groups)
    return metrics


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _batch_retrieval_metrics(
    match_logits: Tensor,
    location_logits: Tensor,
    match_present: Tensor,
    group_ids: Any,
    *,
    match_logit_weight: float,
) -> tuple[int, int, list[float]]:
    if group_ids is None:
        return 0, 0, []
    groups = [str(value) for value in group_ids]
    if len(groups) != match_present.numel():
        raise TrainingError("group_id batch length differs from labels")
    fused = location_logits.float().flatten(1).amax(dim=1)
    fused = fused + float(match_logit_weight) * match_logits.float().reshape(-1)
    correct = 0
    count = 0
    margins: list[float] = []
    for group_id in dict.fromkeys(groups):
        group = torch.as_tensor(
            [value == group_id for value in groups],
            device=fused.device,
            dtype=torch.bool,
        )
        positives = fused[group & match_present]
        negatives = fused[group & ~match_present]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        group_labels = match_present[group]
        correct += int(group_labels[fused[group].argmax()])
        count += 1
        margins.append(float(positives.amax() - negatives.amax()))
    return correct, count, margins


def _share_group_reference(
    reference: Tensor,
    prompt: Tensor,
    group_ids: Any,
    match_present: Tensor,
) -> tuple[Tensor, Tensor]:
    if group_ids is None:
        raise TrainingError("grouped retrieval batch is missing group_id")
    groups = [str(value) for value in group_ids]
    canonical: dict[str, int] = {}
    for index, (group_id, positive) in enumerate(
        zip(groups, match_present.tolist(), strict=True)
    ):
        if positive and group_id not in canonical:
            canonical[group_id] = index
    missing = set(groups).difference(canonical)
    if missing:
        raise TrainingError(
            f"grouped retrieval batch has no positive reference for {min(missing)!r}"
        )
    indices = torch.as_tensor(
        [canonical[group_id] for group_id in groups],
        device=reference.device,
        dtype=torch.long,
    )
    return reference.index_select(0, indices), prompt.index_select(0, indices)


def _binary_ranking_metrics(
    logits: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    """Return tie-aware AUROC, average precision, and the best balanced threshold."""

    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(targets, dtype=bool).reshape(-1)
    if scores.shape != labels.shape or scores.size == 0:
        raise ValueError(
            "binary ranking inputs must be non-empty and have equal shapes"
        )
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return {
            "match_auroc": 0.0,
            "match_average_precision": 0.0,
            "match_best_balanced_accuracy": 0.5,
            "match_best_threshold_logit": 0.0,
            "match_best_threshold_probability": 0.5,
            "match_best_threshold_recall": float(positives > 0),
            "match_best_threshold_specificity": float(negatives > 0),
        }

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    auroc = (float(ranks[labels].sum()) - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )

    descending = np.argsort(-scores, kind="mergesort")
    descending_labels = labels[descending]
    cumulative_positive = np.cumsum(descending_labels)
    precision = cumulative_positive / np.arange(1, scores.size + 1)
    average_precision = float(precision[descending_labels].sum() / positives)

    best_balanced = 0.5
    best_recall = 0.0
    best_specificity = 1.0
    best_threshold = float(np.nextafter(scores.max(), math.inf))
    true_positive = false_positive = 0
    position = 0
    while position < scores.size:
        score = scores[descending[position]]
        end = position
        while end < scores.size and scores[descending[end]] == score:
            if labels[descending[end]]:
                true_positive += 1
            else:
                false_positive += 1
            end += 1
        recall = true_positive / positives
        specificity = (negatives - false_positive) / negatives
        balanced = (recall + specificity) / 2.0
        if balanced > best_balanced:
            best_balanced = balanced
            best_recall = recall
            best_specificity = specificity
            best_threshold = float(score)
        position = end
    probability = (
        1.0 / (1.0 + math.exp(-best_threshold))
        if best_threshold >= 0
        else math.exp(best_threshold) / (1.0 + math.exp(best_threshold))
    )
    return {
        "match_auroc": float(auroc),
        "match_average_precision": average_precision,
        "match_best_balanced_accuracy": float(best_balanced),
        "match_best_threshold_logit": best_threshold,
        "match_best_threshold_probability": float(probability),
        "match_best_threshold_recall": float(best_recall),
        "match_best_threshold_specificity": float(best_specificity),
    }


def _dataset(settings: TrainingSettings, path: Path, *, train: bool) -> HDF5PairDataset:
    if not path.is_file():
        raise TrainingError(f"HDF5 cache does not exist: {path}")
    masking = settings.train_masking if train else settings.validation_masking
    return HDF5PairDataset(
        path,
        prompt_config=settings.prompt,
        prompt_augmentation=masking.augmentation,
        simulated_effective_gsd_m_per_px=settings.simulated_effective_gsd_m_per_px,
        prompt_approaches=masking.approaches,
        prompt_approach_probabilities=masking.probabilities,
        image_augmentation=(
            settings.train_image_augmentation
            if train
            else settings.validation_image_augmentation
        ),
        sample_manifest=(
            settings.sample_manifest if train else settings.validation_sample_manifest
        ),
        seed=settings.seed + (0 if train else 10_000),
        resample_prompt_each_epoch=masking.resample_each_epoch,
    )


def _validate_dataset_dimensions(
    dataset: HDF5PairDataset,
    settings: TrainingSettings,
    *,
    phase: str,
) -> None:
    checks = (
        (
            "reference_crop_size_px",
            settings.expected_reference_size_hw,
            tuple(int(value) for value in dataset.reference_shape_chw[-2:]),
        ),
        (
            "search_crop_size_px",
            settings.expected_search_size_hw,
            tuple(int(value) for value in dataset.search_shape_chw[-2:]),
        ),
    )
    for name, expected, actual in checks:
        if expected is not None and actual != expected:
            raise TrainingError(
                f"{phase} HDF5 {name} mismatch: configuration requests {expected}, "
                f"but {dataset.path} contains {actual}; use the crop dimensions that "
                "created the cache or regenerate the cache"
            )


def _validate_dataset_pair_dimensions(
    train_dataset: HDF5PairDataset,
    validation_dataset: HDF5PairDataset,
) -> None:
    for name in ("reference_shape_chw", "search_shape_chw"):
        training_shape = tuple(getattr(train_dataset, name))
        validation_shape = tuple(getattr(validation_dataset, name))
        if training_shape != validation_shape:
            raise TrainingError(
                f"training and validation HDF5 {name} differ: "
                f"{training_shape} versus {validation_shape}"
            )


def _resolve_match_positive_weight(
    settings: TrainingSettings,
    dataset: HDF5PairDataset,
) -> float:
    positives = int(dataset.positive_samples)
    negatives = int(dataset.negative_samples)
    if positives <= 0 or negatives <= 0:
        raise TrainingError(
            "match/no-match training requires both positive and negative samples; "
            f"found positive={positives}, negative={negatives} in {dataset.path}"
        )
    if settings.match_positive_weight is not None:
        return float(settings.match_positive_weight)
    return float(negatives / positives)


def _loader(
    dataset: HDF5PairDataset,
    batch_size: int,
    *,
    shuffle: bool,
    settings: TrainingSettings,
    device: torch.device,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    if settings.retrieval_ranking_weight > 0:
        batch_sampler = _GroupedTrackBatchSampler(dataset, shuffle=shuffle, seed=seed)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=settings.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
        # Workers are recreated after set_epoch() so epoch-dependent prompt masks
        # are visible under both fork and spawn multiprocessing implementations.
        persistent_workers=False,
        generator=generator,
    )


def _set_loader_epoch(loader: DataLoader[dict[str, Any]], seed: int) -> None:
    if loader.generator is not None:
        loader.generator.manual_seed(seed)
    batch_sampler = loader.batch_sampler
    if isinstance(batch_sampler, _GroupedTrackBatchSampler):
        batch_sampler.set_epoch(seed)


class _GroupedTrackBatchSampler(Sampler[list[int]]):
    """Yield one complete source-track candidate set per optimization step."""

    def __init__(self, dataset: HDF5PairDataset, *, shuffle: bool, seed: int) -> None:
        if dataset.group_ids is None or dataset.match_flags is None:
            raise TrainingError(
                "retrieval ranking requires training.sample_manifest with "
                "sample_id, track_id, and match_present columns"
            )
        grouped: dict[str, list[int]] = {}
        for index, group_id in enumerate(dataset.group_ids):
            grouped.setdefault(group_id, []).append(index)
        invalid = [
            group_id
            for group_id, indices in grouped.items()
            if not any(dataset.match_flags[index] for index in indices)
            or not any(not dataset.match_flags[index] for index in indices)
        ]
        if invalid:
            raise TrainingError(
                "retrieval-ranking groups require positive and negative crops; "
                f"first invalid track={invalid[0]!r}"
            )
        self.groups = tuple(tuple(indices) for indices in grouped.values())
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Any:
        order = list(range(len(self.groups)))
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(order), generator=generator).tolist()
        for group_index in order:
            yield list(self.groups[group_index])

    def __len__(self) -> int:
        return len(self.groups)


def _masking_phase(
    payload: dict[str, Any], *, default_seed_phase: bool
) -> MaskingPhaseConfig:
    approaches_raw = payload.get(
        "approaches", {"ellipse": 0.75, "gaussian_point": 0.25}
    )
    if not isinstance(approaches_raw, dict) or not approaches_raw:
        raise ConfigError(
            "masking approaches must be a non-empty name: probability mapping"
        )
    approaches = tuple(str(name) for name in approaches_raw)
    allowed = {"auto", "ellipse", "gaussian_point"}
    if any(name not in allowed for name in approaches):
        raise ConfigError(f"masking approaches must be selected from {sorted(allowed)}")
    probabilities = tuple(float(value) for value in approaches_raw.values())
    augmentation_raw = payload.get("augmentation", {})
    if not isinstance(augmentation_raw, dict):
        raise ConfigError("masking augmentation must be a mapping")
    try:
        augmentation = PromptAugmentation(
            enabled=bool(augmentation_raw.get("enabled", True)),
            center_jitter_m=float(augmentation_raw.get("center_jitter_m", 0.15)),
            scale_range=_float_pair(augmentation_raw.get("scale_range", (0.8, 1.25))),
            orientation_jitter_deg=float(
                augmentation_raw.get("orientation_jitter_deg", 15.0)
            ),
            morphology_radius_px=_int_pair(
                augmentation_raw.get("morphology_radius_px", (-2, 2))
            ),
            blur_sigma_px=_float_pair(
                augmentation_raw.get("blur_sigma_px", (0.0, 1.25))
            ),
            point_only_probability=float(
                augmentation_raw.get("point_only_probability", 0.0)
            ),
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid masking augmentation: {error}") from error
    return MaskingPhaseConfig(
        approaches=approaches,  # type: ignore[arg-type]
        probabilities=probabilities,
        augmentation=augmentation,
        resample_each_epoch=bool(
            payload.get("resample_each_epoch", default_seed_phase)
        ),
    )


def _image_augmentation_phase(payload: Any) -> ImageAugmentation:
    if not isinstance(payload, dict):
        raise ConfigError("each training.image_augmentation phase must be a mapping")
    rotations = payload.get("rotation_degrees", (0, 90, 180, 270))
    if not isinstance(rotations, (list, tuple)):
        raise ConfigError("image augmentation rotation_degrees must be a sequence")
    try:
        return ImageAugmentation(
            enabled=bool(payload.get("enabled", False)),
            photometric_probability=float(payload.get("photometric_probability", 0.8)),
            brightness_range=_float_pair(payload.get("brightness_range", (0.8, 1.2))),
            contrast_range=_float_pair(payload.get("contrast_range", (0.8, 1.2))),
            saturation_range=_float_pair(payload.get("saturation_range", (0.8, 1.2))),
            hue_range=_float_pair(payload.get("hue_range", (-0.05, 0.05))),
            gamma_range=_float_pair(payload.get("gamma_range", (0.8, 1.2))),
            rotation_degrees=tuple(int(value) for value in rotations),
            horizontal_flip_probability=float(
                payload.get("horizontal_flip_probability", 0.5)
            ),
            vertical_flip_probability=float(
                payload.get("vertical_flip_probability", 0.5)
            ),
            gaussian_blur_probability=float(
                payload.get("gaussian_blur_probability", 0.3)
            ),
            gaussian_blur_sigma_px=_float_pair(
                payload.get("gaussian_blur_sigma_px", (0.1, 1.5))
            ),
            gaussian_noise_probability=float(
                payload.get("gaussian_noise_probability", 0.3)
            ),
            gaussian_noise_std=_float_pair(
                payload.get("gaussian_noise_std", (0.0, 0.03))
            ),
            composite_shadow_probability=float(
                payload.get("composite_shadow_probability", 0.0)
            ),
            cast_shadow_probability=float(payload.get("cast_shadow_probability", 0.0)),
            cast_shadow_coverage_range=_float_pair(
                payload.get("cast_shadow_coverage_range", (0.25, 0.75))
            ),
            cast_shadow_edge_softness_px=_float_pair(
                payload.get("cast_shadow_edge_softness_px", (8.0, 40.0))
            ),
            shadow_transmission_range=_float_pair(
                payload.get("shadow_transmission_range", (0.10, 0.45))
            ),
            shadow_ambient_range=_float_pair(
                payload.get("shadow_ambient_range", (0.002, 0.012))
            ),
            shadow_red_gain_range=_float_pair(
                payload.get("shadow_red_gain_range", (0.72, 0.90))
            ),
            shadow_green_gain_range=_float_pair(
                payload.get("shadow_green_gain_range", (0.85, 1.00))
            ),
            shadow_blue_gain_range=_float_pair(
                payload.get("shadow_blue_gain_range", (1.00, 1.18))
            ),
            shadow_field_strength_range=_float_pair(
                payload.get("shadow_field_strength_range", (0.00, 0.14))
            ),
            shadow_field_grid_size=int(payload.get("shadow_field_grid_size", 8)),
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(
            f"invalid image augmentation configuration: {error}"
        ) from error


def _validate_training_settings(settings: TrainingSettings) -> None:
    shared_cache = settings.train_hdf5_cache == settings.validation_hdf5_cache
    if shared_cache and not settings.allow_shared_cache:
        raise ConfigError(
            "training and validation caches are identical; use spatially disjoint caches or set "
            "training.allow_shared_cache=true only for a smoke test"
        )
    if min(settings.epochs, settings.batch_size, settings.validation_batch_size) <= 0:
        raise ConfigError("epochs and batch sizes must be positive")
    if settings.num_workers < 0:
        raise ConfigError("num_workers cannot be negative")
    if settings.learning_rate <= 0 or settings.weight_decay < 0:
        raise ConfigError(
            "learning_rate must be positive and weight_decay non-negative"
        )
    if (
        settings.match_positive_weight is not None
        and settings.match_positive_weight <= 0
    ):
        raise ConfigError("match_positive_weight must be positive or 'auto'")
    if (
        min(
            settings.localization_weight,
            settings.positive_localization_weight,
            settings.positive_peak_ranking_weight,
            settings.negative_localization_weight,
            settings.retrieval_ranking_weight,
        )
        < 0
    ):
        raise ConfigError("localization loss weights cannot be negative")
    if settings.negative_localization_temperature <= 0:
        raise ConfigError("negative_localization_temperature must be positive")
    if settings.retrieval_ranking_temperature <= 0:
        raise ConfigError("retrieval_ranking_temperature must be positive")
    if settings.retrieval_match_logit_weight < 0:
        raise ConfigError("retrieval_match_logit_weight cannot be negative")
    if settings.freeze_feature_extractor and settings.freeze_backbone:
        raise ConfigError(
            "choose only one of freeze_feature_extractor and freeze_backbone"
        )
    if settings.retrieval_ranking_weight > 0 and settings.sample_manifest is None:
        raise ConfigError(
            "training.sample_manifest is required when retrieval_ranking_weight > 0"
        )
    if settings.gradient_clip_norm is not None and settings.gradient_clip_norm <= 0:
        raise ConfigError("gradient_clip_norm must be positive or null")
    scheduler = settings.lr_scheduler
    if scheduler.name not in {"cosine_annealing", "reduce_on_plateau"}:
        raise ConfigError(
            "training.lr_scheduler.name must be cosine_annealing or reduce_on_plateau"
        )
    if not 0 < scheduler.factor < 1:
        raise ConfigError("training.lr_scheduler.factor must be between zero and one")
    if (
        min(
            scheduler.patience,
            scheduler.threshold,
            scheduler.cooldown,
            scheduler.minimum_lr,
        )
        < 0
    ):
        raise ConfigError("training.lr_scheduler plateau parameters cannot be negative")
    stopping = settings.early_stopping
    if stopping.patience <= 0:
        raise ConfigError("training.early_stopping.patience must be positive")
    if stopping.minimum_delta < 0 or stopping.minimum_epochs <= 0:
        raise ConfigError(
            "training.early_stopping minimum_delta must be non-negative and "
            "minimum_epochs positive"
        )
    selection = settings.checkpoint_selection
    if not all(
        0 <= value <= 1
        for value in (
            selection.minimum_match_recall,
            selection.minimum_match_specificity,
            selection.minimum_match_auroc,
            selection.minimum_best_balanced_accuracy,
            selection.minimum_localization_success_at_2m,
            selection.minimum_retrieval_top1_accuracy,
        )
    ):
        raise ConfigError("checkpoint-selection thresholds must be between 0 and 1")
    if (
        min(
            selection.match_auroc_weight,
            selection.localization_success_weight,
            selection.retrieval_top1_weight,
        )
        < 0
    ):
        raise ConfigError("checkpoint-selection weights cannot be negative")
    if (
        selection.match_auroc_weight
        + selection.localization_success_weight
        + selection.retrieval_top1_weight
        <= 0
    ):
        raise ConfigError("at least one checkpoint-selection weight must be positive")
    if settings.matcher not in {
        "cosine_v1",
        "response_relation_v2",
        "spatial_prototypes_v3",
        "xcorr_template_v4",
        "multiscale_xcorr_v5",
        "masked_xcorr_v6",
    }:
        raise ConfigError(
            "model.matcher must be cosine_v1, response_relation_v2, "
            "spatial_prototypes_v3, xcorr_template_v4, or multiscale_xcorr_v5"
        )
    if settings.match_channels <= 0:
        raise ConfigError("model.match_channels must be positive")
    if settings.input_representation not in {"rgb", "rgb_invariant_v1"}:
        raise ConfigError(
            "model.input_representation must be rgb or rgb_invariant_v1"
        )
    if not settings.simulated_effective_gsd_m_per_px:
        raise ConfigError("data.simulated_effective_gsd_m_per_px cannot be empty")
    if settings.tensor_gsd_m_per_px <= 0:
        raise ConfigError("data.tensor_gsd_m_per_px must be positive")
    for phase in (settings.train_masking, settings.validation_masking):
        if (
            any(value < 0 for value in phase.probabilities)
            or sum(phase.probabilities) <= 0
        ):
            raise ConfigError("masking probabilities must be non-negative and non-zero")


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer, settings: TrainingSettings
) -> TrainingScheduler:
    scheduler = settings.lr_scheduler
    if scheduler.name == "cosine_annealing":
        return CosineAnnealingLR(optimizer, T_max=max(settings.epochs, 1))
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler.factor,
        patience=scheduler.patience,
        threshold=scheduler.threshold,
        threshold_mode="abs",
        cooldown=scheduler.cooldown,
        min_lr=scheduler.minimum_lr,
    )


def _step_scheduler(scheduler: TrainingScheduler, validation_loss: float) -> None:
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(float(validation_loss))
    else:
        scheduler.step()


def _early_stopping_history_state(
    history: list[dict[str, Any]], minimum_delta: float
) -> tuple[float, int]:
    best = math.inf
    epochs_without_improvement = 0
    for row in history:
        value = float(row["validation"]["total"])
        if value < best - minimum_delta:
            best = value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    return best, epochs_without_improvement


def _checkpoint_eligible(
    metrics: dict[str, Any],
    settings: CheckpointSelectionSettings,
) -> bool:
    """Reject collapsed or non-discriminative validation classifiers."""

    recall = float(metrics.get("match_recall", 0.0))
    specificity = float(metrics.get("match_specificity", 0.0))
    auroc = float(metrics.get("match_auroc", 0.0))
    best_balanced = float(metrics.get("match_best_balanced_accuracy", 0.0))
    localization_success = float(metrics.get("localization_success_at_2m", 0.0))
    retrieval_top1 = float(metrics.get("retrieval_top1_accuracy", 0.0))
    return (
        recall >= settings.minimum_match_recall
        and specificity >= settings.minimum_match_specificity
        and auroc >= settings.minimum_match_auroc
        and best_balanced >= settings.minimum_best_balanced_accuracy
        and localization_success >= settings.minimum_localization_success_at_2m
        and retrieval_top1 >= settings.minimum_retrieval_top1_accuracy
    )


def _checkpoint_selection_score(
    metrics: dict[str, Any], settings: CheckpointSelectionSettings
) -> float:
    """Joint discrimination/localization score used to select deployable weights."""

    match_weight = settings.match_auroc_weight
    localization_weight = settings.localization_success_weight
    retrieval_weight = settings.retrieval_top1_weight
    denominator = match_weight + localization_weight + retrieval_weight
    return (
        match_weight * float(metrics.get("match_auroc", 0.0))
        + localization_weight * float(metrics.get("localization_success_at_2m", 0.0))
        + retrieval_weight * float(metrics.get("retrieval_top1_accuracy", 0.0))
    ) / denominator


def _autocast(device: torch.device, precision: PrecisionPolicy) -> Any:
    if precision.dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=precision.dtype)


def _to_device(value: Tensor, device: torch.device) -> Tensor:
    return value.to(device, non_blocking=device.type == "cuda")


def _clip_gradients(model: nn.Module, maximum: float | None) -> None:
    if maximum is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), maximum)


def _checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: TrainingScheduler,
    scaler: torch.amp.GradScaler,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    settings: TrainingSettings,
    precision: PrecisionPolicy,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package_version": __version__,
        "torch_version": torch.__version__,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scheduler_name": settings.lr_scheduler.name,
        "scaler_state": scaler.state_dict(),
        "history": history,
        "config": config,
        "model": {
            "backbone": settings.backbone,
            "pretrained": settings.pretrained,
            "pyramid_channels": settings.pyramid_channels,
            "matcher": settings.matcher,
            "match_channels": settings.match_channels,
            "input_representation": settings.input_representation,
        },
        "precision": asdict(precision) | {"dtype": str(precision.dtype)},
        "random_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if next(model.parameters()).device.type == "cuda"
                else None
            ),
        },
    }


def _restore_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: TrainingScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    scheduler_name: str,
) -> tuple[int, list[dict[str, Any]]]:
    if not path.is_file():
        raise TrainingError(f"resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if int(checkpoint.get("schema_version", -1)) != 1:
        raise TrainingError("unsupported checkpoint schema")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    stored_scheduler_name = checkpoint.get("scheduler_name")
    if stored_scheduler_name is not None and stored_scheduler_name != scheduler_name:
        raise TrainingError(
            "resume checkpoint scheduler differs from the configured lr_scheduler"
        )
    try:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingError(
            "resume checkpoint scheduler state is incompatible with the configured scheduler"
        ) from error
    scaler.load_state_dict(checkpoint.get("scaler_state", {}))
    random_state = checkpoint.get("random_state", {})
    if random_state:
        random.setstate(random_state["python"])
        np.random.set_state(random_state["numpy"])
        torch.set_rng_state(random_state["torch"].cpu())
        if random_state.get("cuda") is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in random_state["cuda"]]
            )
    history = list(checkpoint.get("history", []))
    return int(checkpoint["epoch"]) + 1, history


def _initialize_model_weights(
    path: Path, model: nn.Module, device: torch.device
) -> None:
    """Load model weights without restoring optimization or random state."""

    if not path.is_file():
        raise TrainingError(f"initialization checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if int(checkpoint.get("schema_version", -1)) != 1:
        raise TrainingError("unsupported initialization checkpoint schema")
    if "backbone_state" in checkpoint:
        expected_backbone = type(model.backbone).__name__
        if (
            checkpoint.get("backbone") != "convnext_tiny"
            or expected_backbone != "TorchvisionFeatureBackbone"
        ):
            raise TrainingError("dense pretraining backbone is incompatible with the tracker")
        try:
            model.backbone.load_state_dict(checkpoint["backbone_state"], strict=True)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise TrainingError(f"could not initialize pretrained backbone: {error}") from error
        return
    try:
        source_state = checkpoint["model_state"]
    except (KeyError, TypeError) as error:
        raise TrainingError(
            f"initialization checkpoint model is incompatible: {error}"
        ) from error
    try:
        model.load_state_dict(source_state, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        try:
            if (
                "input_adapter.weight" in model.state_dict()
                and "input_adapter.weight" not in source_state
            ):
                adapted = model.state_dict()
                unexpected = set(source_state).difference(adapted)
                mismatched = {
                    name
                    for name, value in source_state.items()
                    if name in adapted and tuple(value.shape) != tuple(adapted[name].shape)
                }
                if unexpected or mismatched:
                    raise ValueError(
                        f"RGB checkpoint differs beyond input adapter: "
                        f"unexpected={sorted(unexpected)}, mismatched={sorted(mismatched)}"
                    )
                adapted.update(source_state)
            else:
                adapted = _single_to_multiscale_xcorr_state(model, source_state)
            model.load_state_dict(adapted, strict=True)
        except (KeyError, RuntimeError, TypeError, ValueError) as adaptation_error:
            raise TrainingError(
                f"initialization checkpoint model is incompatible: {error}"
            ) from adaptation_error


def _single_to_multiscale_xcorr_state(
    model: nn.Module, source_state: dict[str, Tensor]
) -> dict[str, Tensor]:
    """Expand a v4 decoder so v5 initially reproduces its central scale."""

    if getattr(model, "matcher", None) != "multiscale_xcorr_v5":
        raise ValueError("target is not a multiscale XCorr model")
    target_state = model.state_dict()
    first_relation_weight = "relation_encoder.0.weight"
    source_weight = source_state[first_relation_weight]
    target_weight = target_state[first_relation_weight]
    if (
        source_weight.ndim != 4
        or source_weight.shape[1] != 1
        or target_weight.shape[1] != 3
        or source_weight.shape[0] != target_weight.shape[0]
        or source_weight.shape[2:] != target_weight.shape[2:]
    ):
        raise ValueError("source is not a compatible single-scale XCorr model")
    adapted: dict[str, Tensor] = {}
    for name, target in target_state.items():
        if name == first_relation_weight:
            expanded = torch.zeros_like(target)
            expanded[:, 1] = source_weight[:, 0]
            adapted[name] = expanded
            continue
        source = source_state[name]
        if source.shape != target.shape:
            raise ValueError(f"incompatible checkpoint tensor: {name}")
        adapted[name] = source
    return adapted


def _feature_extractor_modules(model: nn.Module) -> tuple[nn.Module, ...]:
    names = (
        "backbone",
        "pyramid",
        "reference_projection",
        "search_projection",
        "relation_encoder",
    )
    return tuple(
        module
        for name in names
        if isinstance((module := getattr(model, name, None)), nn.Module)
    )


def _freeze_feature_extractor(model: nn.Module) -> None:
    modules = _feature_extractor_modules(model)
    if not modules:
        raise TrainingError("model has no recognized feature-extractor modules")
    for module in modules:
        module.requires_grad_(False)
        module.eval()
    model._frozen_training_modules = modules  # type: ignore[attr-defined]


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze pretrained image features while retaining matcher adaptation."""

    modules = tuple(
        module
        for name in ("backbone", "pyramid")
        if isinstance((module := getattr(model, name, None)), nn.Module)
    )
    if not modules:
        raise TrainingError("model has no recognized backbone modules")
    for module in modules:
        module.requires_grad_(False)
        module.eval()
    model._frozen_training_modules = modules  # type: ignore[attr-defined]


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _run_directory(root: str | Path, name: str | None) -> Path:
    run_name = name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(root) / run_name


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"configuration requires a '{key}' mapping")
    return value


def _float_pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("expected a two-value sequence")
    return float(value[0]), float(value[1])


def _int_pair(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("expected a two-value sequence")
    return int(value[0]), int(value[1])


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _positive_weight(value: Any) -> float | None:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    return float(value)


def _optional_pixel_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("crop size must be [height, width]")
    result = int(value[0]), int(value[1])
    if min(result) <= 0:
        raise ValueError("crop dimensions must be positive")
    return result


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
