"""Portable model checkpoint loading for evaluation and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from glaciertrack.device import resolve_device
from glaciertrack.models import PromptedSiameseTracker


class CheckpointError(RuntimeError):
    """Raised when a tracker checkpoint is missing or incompatible."""


@dataclass(frozen=True)
class LoadedTracker:
    model: PromptedSiameseTracker
    config: dict[str, Any]
    device: torch.device
    epoch: int
    package_version: str | None


def load_tracker_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "auto",
) -> LoadedTracker:
    """Restore a trained tracker without downloading pretrained weights."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise CheckpointError(f"checkpoint does not exist: {checkpoint_path}")
    target_device = (
        resolve_device(device) if isinstance(device, str) else torch.device(device)
    )
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=target_device,
            weights_only=False,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointError(f"could not load checkpoint: {error}") from error
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise CheckpointError("unsupported checkpoint schema")
    model_spec = payload.get("model")
    config = payload.get("config")
    if not isinstance(model_spec, dict) or not isinstance(config, dict):
        raise CheckpointError("checkpoint is missing model or configuration metadata")
    try:
        model = PromptedSiameseTracker(
            str(model_spec["backbone"]),
            pretrained=False,
            pyramid_channels=int(model_spec["pyramid_channels"]),
            matcher=str(model_spec.get("matcher", "cosine_v1")),
            match_channels=int(model_spec.get("match_channels", 64)),
            input_representation=str(model_spec.get("input_representation", "rgb")),
        )
        model.load_state_dict(payload["model_state"], strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise CheckpointError(f"checkpoint model is incompatible: {error}") from error
    model.to(target_device).eval()
    return LoadedTracker(
        model=model,
        config=config,
        device=target_device,
        epoch=int(payload.get("epoch", -1)) + 1,
        package_version=(
            None if payload.get("package_version") is None else str(payload["package_version"])
        ),
    )
