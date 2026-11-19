"""Feature-only adapters for supported torchvision CNNs."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn
from torchvision import models

from glaciertrack.models.backbones.registry import (
    FeatureBackbone,
    PreprocessingSpec,
    register_backbone,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class TorchvisionFeatureBackbone(FeatureBackbone):
    def __init__(
        self,
        stages: tuple[nn.Module, ...],
        *,
        feature_channels: dict[int, int],
        weights_id: str | None,
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.feature_channels = dict(feature_channels)
        self.output_stride = max(feature_channels)
        self.out_channels = feature_channels[self.output_stride]
        self.preprocessing = PreprocessingSpec(
            input_scale=1.0 / 255.0,
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
            weights_id=weights_id,
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.forward_multiscale(image)[self.output_stride]

    def forward_multiscale(self, image: Tensor) -> dict[int, Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("backbone input must have shape Bx3xHxW")
        outputs: dict[int, Tensor] = {}
        value = image
        for stride, stage in zip(self.feature_channels, self.stages, strict=True):
            value = stage(value)
            outputs[stride] = value
        return outputs


def _build(name: str, *, pretrained: bool = True, **_: Any) -> FeatureBackbone:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        stages = (
            nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1),
            model.layer2,
            model.layer3,
            model.layer4,
        )
        feature_channels = {4: 64, 8: 128, 16: 256, 32: 512}
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        stages = (
            nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1),
            model.layer2,
            model.layer3,
            model.layer4,
        )
        feature_channels = {4: 256, 8: 512, 16: 1024, 32: 2048}
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        blocks = tuple(model.features.children())
        stages = (
            nn.Sequential(*blocks[0:2]),
            nn.Sequential(*blocks[2:4]),
            nn.Sequential(*blocks[4:6]),
            nn.Sequential(*blocks[6:8]),
        )
        feature_channels = {4: 96, 8: 192, 16: 384, 32: 768}
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        blocks = tuple(model.features.children())
        stages = (
            nn.Sequential(*blocks[0:3]),
            nn.Sequential(*blocks[3:4]),
            nn.Sequential(*blocks[4:6]),
            nn.Sequential(*blocks[6:9]),
        )
        feature_channels = {4: 24, 8: 40, 16: 112, 32: 1280}
    else:
        raise ValueError(name)
    weights_id = str(weights) if weights is not None else None
    return TorchvisionFeatureBackbone(
        stages,
        feature_channels=feature_channels,
        weights_id=weights_id,
    )


def register_torchvision_backbones() -> None:
    for name in ("convnext_tiny", "efficientnet_b0", "resnet18", "resnet50"):
        register_backbone(name, lambda *, _name=name, **kwargs: _build(_name, **kwargs))
