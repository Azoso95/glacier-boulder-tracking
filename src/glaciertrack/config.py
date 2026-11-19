"""Configuration loading with validation for foundational project settings."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class ResolutionConfig:
    """Resolution settings expressed in metres per pixel."""

    tensor_gsd_m_per_px: float
    simulated_effective_gsd_m_per_px: tuple[float, ...]
    downsample_mode: str = "area"
    upsample_mode: str = "bilinear"

    def __post_init__(self) -> None:
        if self.tensor_gsd_m_per_px <= 0:
            raise ConfigError("tensor_gsd_m_per_px must be positive")
        if not self.simulated_effective_gsd_m_per_px:
            raise ConfigError("at least one effective GSD is required")
        if any(value < self.tensor_gsd_m_per_px for value in self.simulated_effective_gsd_m_per_px):
            raise ConfigError("effective GSD cannot be finer than the tensor grid")
        if self.downsample_mode != "area":
            raise ConfigError("the supported downsample mode is 'area'")
        if self.upsample_mode not in {"bilinear", "bicubic"}:
            raise ConfigError("upsample_mode must be 'bilinear' or 'bicubic'")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, recursively resolving optional ``extends`` files."""

    return _load_yaml(Path(path).expanduser().resolve(), ())


def _load_yaml(config_path: Path, ancestors: tuple[Path, ...]) -> dict[str, Any]:
    if not config_path.is_file():
        raise ConfigError(f"configuration file does not exist: {config_path}")
    if config_path in ancestors:
        chain = " -> ".join(str(value) for value in (*ancestors, config_path))
        raise ConfigError(f"configuration inheritance cycle: {chain}")
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ConfigError("configuration root must be a mapping")
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    parent_path = Path(str(parent)).expanduser()
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    inherited = _load_yaml(parent_path.resolve(), (*ancestors, config_path))
    return _deep_merge(inherited, payload)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolution_config(payload: dict[str, Any]) -> ResolutionConfig:
    """Extract and validate the resolution section of a project mapping."""

    data = payload.get("data")
    if not isinstance(data, dict):
        raise ConfigError("configuration requires a 'data' mapping")
    try:
        tensor_gsd = float(data["tensor_gsd_m_per_px"])
        effective_values = tuple(float(value) for value in data["simulated_effective_gsd_m_per_px"])
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError("invalid data resolution configuration") from error
    return ResolutionConfig(
        tensor_gsd_m_per_px=tensor_gsd,
        simulated_effective_gsd_m_per_px=effective_values,
        downsample_mode=str(data.get("downsample_mode", "area")),
        upsample_mode=str(data.get("upsample_mode", "bilinear")),
    )


def validate_project_config(payload: dict[str, Any]) -> None:
    """Validate invariants that should hold before any pipeline command runs."""

    resolution_config(payload)
    sampling = payload.get("sampling")
    if not isinstance(sampling, dict):
        raise ConfigError("configuration requires a 'sampling' mapping")
    placement = sampling.get("search_placement")
    expected_placement = (
        "controlled_composition"
        if sampling.get("sample_composition") is not None
        else "label_blind"
    )
    if placement != expected_placement:
        raise ConfigError(
            f"sampling.search_placement must be {expected_placement!r} for this sampler"
        )
    if sampling.get("keep_tracks_atomic") is not True:
        raise ConfigError("sampling.keep_tracks_atomic must be true")
    inference_search_config(payload)
    if "training" in payload:
        from glaciertrack.training import training_settings

        training_settings(payload)


def inference_search_config(payload: dict[str, Any]) -> Any:
    """Extract and validate the operational search policy without endpoint data."""

    from glaciertrack.inference.search import SearchConfig

    inference = payload.get("inference", {})
    if not isinstance(inference, dict):
        raise ConfigError("inference must be a mapping")
    search = inference.get("search", {})
    early_stop = inference.get("early_stop", {})
    if not isinstance(search, dict) or not isinstance(early_stop, dict):
        raise ConfigError("inference.search and inference.early_stop must be mappings")
    try:
        stitch_stride = int(inference.get("stitch_stride_px", 4))
        if stitch_stride <= 0:
            raise ValueError("inference.stitch_stride_px must be positive")
        return SearchConfig(
            mode=search.get("mode", "radius"),
            tile_size_px=tuple(search.get("tile_size_px", (512, 512))),
            tile_overlap=float(search.get("tile_overlap", 0.5)),
            tile_batch_size=int(search.get("tile_batch_size", 16)),
            radius_m=_optional_number(search.get("radius_m", 250.0)),
            maximum_radius_m=_optional_number(search.get("maximum_radius_m")),
            ring_step_m=float(search.get("ring_step_m", 100.0)),
            confidence_threshold=float(early_stop.get("confidence_threshold", 0.8)),
            decision_threshold=float(inference.get("decision_threshold", 0.5)),
            stability_expansions=int(early_stop.get("stability_expansions", 1)),
            stability_tolerance_m=float(early_stop.get("stability_tolerance_m", 1.0)),
            minimum_peak_margin=float(early_stop.get("minimum_peak_margin", 0.1)),
            match_logit_weight=float(inference.get("match_logit_weight", 1.0)),
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid inference search configuration: {error}") from error


def _optional_number(value: Any) -> float | None:
    return None if value is None else float(value)
