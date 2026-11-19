"""Backbone registry decoupling tracker heads from specific CNN families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class PreprocessingSpec:
    """Only channel preprocessing; tracking controls all spatial transforms."""

    input_scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    weights_id: str | None


class FeatureBackbone(nn.Module):
    """Base class for registered feature encoders."""

    out_channels: int
    output_stride: int
    feature_channels: dict[int, int]
    preprocessing: PreprocessingSpec

    def forward_multiscale(self, image: Tensor) -> dict[int, Tensor]:
        """Return feature maps keyed by output stride."""

        raise NotImplementedError


BackboneFactory = Callable[..., FeatureBackbone]
_REGISTRY: dict[str, BackboneFactory] = {}
_BUILTINS_LOADED = False


def register_backbone(name: str, factory: BackboneFactory) -> None:
    normalized = name.strip().lower()
    if not normalized or normalized in _REGISTRY:
        raise ValueError(f"invalid or duplicate backbone name: {name!r}")
    _REGISTRY[normalized] = factory


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        from glaciertrack.models.backbones import torchvision_backbones

        torchvision_backbones.register_torchvision_backbones()
        _BUILTINS_LOADED = True


def available_backbones() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_REGISTRY))


def create_backbone(name: str, **kwargs: object) -> FeatureBackbone:
    _load_builtins()
    normalized = name.strip().lower()
    try:
        factory = _REGISTRY[normalized]
    except KeyError as error:
        choices = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown backbone {name!r}; available: {choices}") from error
    return factory(**kwargs)
