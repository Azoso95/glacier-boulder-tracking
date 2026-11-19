"""Lazy registry of feature-only CNN backbones."""

from glaciertrack.models.backbones.registry import (
    FeatureBackbone,
    PreprocessingSpec,
    available_backbones,
    create_backbone,
    register_backbone,
)

__all__ = [
    "FeatureBackbone",
    "PreprocessingSpec",
    "available_backbones",
    "create_backbone",
    "register_backbone",
]
