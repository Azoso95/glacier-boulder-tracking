"""Prompt-conditioned Siamese tracker with a multi-scale target feature map."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from glaciertrack.models.backbones import FeatureBackbone, create_backbone
from glaciertrack.models.outputs import TrackerOutput
from glaciertrack.models.preprocessing import normalize_for_backbone


@dataclass(frozen=True)
class ReferenceEncoding:
    """Compact boulder representation encoded once for many target tiles."""

    query: Tensor
    spatial_weights: Tensor | None = None


@dataclass(frozen=True)
class SearchEncoding:
    """Cacheable target features for reuse across multiple boulder queries."""

    features: Tensor
    output_size_hw: tuple[int, int]


class FeaturePyramid(nn.Module):
    """Top-down pyramid returning a high-resolution stride-4 feature map."""

    def __init__(self, channels_by_stride: dict[int, int], output_channels: int) -> None:
        super().__init__()
        if 4 not in channels_by_stride:
            raise ValueError("tracker backbones must expose stride-4 features")
        self.strides = tuple(sorted(channels_by_stride))
        self.lateral = nn.ModuleDict(
            {
                str(stride): nn.Conv2d(channels_by_stride[stride], output_channels, 1)
                for stride in self.strides
            }
        )
        self.smooth = nn.ModuleDict(
            {
                str(stride): nn.Conv2d(output_channels, output_channels, 3, padding=1)
                for stride in self.strides
            }
        )

    def forward(self, features: dict[int, Tensor]) -> Tensor:
        missing = set(self.strides).difference(features)
        if missing:
            raise ValueError(f"missing backbone feature strides: {sorted(missing)}")
        merged: Tensor | None = None
        for stride in reversed(self.strides):
            lateral = self.lateral[str(stride)](features[stride])
            if merged is not None:
                merged = F.interpolate(
                    merged,
                    size=lateral.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                lateral = lateral + merged
            merged = self.smooth[str(stride)](lateral)
        if merged is None:  # pragma: no cover - protected by constructor
            raise RuntimeError("feature pyramid has no levels")
        return merged


class PromptedSiameseTracker(nn.Module):
    """Shared RGB encoder, prompt pooling, correlation, and heatmap decoder."""

    def __init__(
        self,
        backbone: str | FeatureBackbone = "convnext_tiny",
        *,
        pretrained: bool = True,
        pyramid_channels: int = 128,
        matcher: str = "cosine_v1",
        match_channels: int = 64,
        input_representation: str = "rgb",
    ) -> None:
        super().__init__()
        if matcher not in {
            "cosine_v1",
            "response_relation_v2",
            "spatial_prototypes_v3",
            "xcorr_template_v4",
            "multiscale_xcorr_v5",
            "masked_xcorr_v6",
        }:
            raise ValueError(
                "matcher must be cosine_v1, response_relation_v2, "
                "spatial_prototypes_v3, xcorr_template_v4, or multiscale_xcorr_v5"
            )
        if match_channels <= 0:
            raise ValueError("match_channels must be positive")
        if input_representation not in {"rgb", "rgb_invariant_v1"}:
            raise ValueError("input_representation must be rgb or rgb_invariant_v1")
        self.matcher = matcher
        self.input_representation = input_representation
        if input_representation == "rgb_invariant_v1":
            # Preserve the pretrained RGB path at initialization, then allow the
            # model to learn how much locally normalized intensity and gradient
            # magnitude help under illumination changes.
            self.input_adapter = nn.Conv2d(5, 3, kernel_size=1, bias=False)
            with torch.no_grad():
                self.input_adapter.weight.zero_()
                for channel in range(3):
                    self.input_adapter.weight[channel, channel, 0, 0] = 1.0
        else:
            self.input_adapter = nn.Identity()
        self.backbone = (
            create_backbone(backbone, pretrained=pretrained)
            if isinstance(backbone, str)
            else backbone
        )
        self.pyramid = FeaturePyramid(self.backbone.feature_channels, pyramid_channels)
        if matcher == "cosine_v1":
            self.location_head = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 1, 3, padding=1),
            )
            self.match_head = nn.Sequential(
                nn.Linear(2, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
        else:
            relation_channels = 32
            if matcher in {"response_relation_v2", "spatial_prototypes_v3"}:
                self.reference_projection = nn.Sequential(
                    nn.Linear(pyramid_channels, match_channels),
                    nn.GELU(),
                    nn.Linear(match_channels, match_channels),
                )
            self.search_projection = nn.Conv2d(pyramid_channels, match_channels, kernel_size=1)
            relation_inputs = {
                "response_relation_v2": match_channels * 2,
                "spatial_prototypes_v3": 5,
                "xcorr_template_v4": 1,
                "multiscale_xcorr_v5": 3,
                "masked_xcorr_v6": 1,
            }[matcher]
            self.relation_encoder = nn.Sequential(
                nn.Conv2d(relation_inputs, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, relation_channels, 3, padding=1),
                nn.GELU(),
            )
            self.location_head = nn.Conv2d(relation_channels, 1, kernel_size=1)
            # Global relation features answer whether a credible match exists;
            # the response statistics expose the strength and contrast of its peak.
            self.match_head = nn.Sequential(
                nn.Linear(relation_channels * 2 + 3, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

    def encode_reference(
        self, image: Tensor, source_prompt: Tensor, *, source_point_px: Tensor | None = None,
    ) -> ReferenceEncoding:
        return self.encode_reference_features(
            self._encode_image(image), source_prompt, source_point_px=source_point_px,
        )

    def encode_reference_features(
        self, features: Tensor, source_prompt: Tensor, *, source_point_px: Tensor | None = None,
    ) -> ReferenceEncoding:
        """Encode a prompt from precomputed stride-4 pyramid features.

        Keeping this operation separate from the image encoder permits
        search-episode training to adapt the compact matcher projection while
        retaining frozen, reusable backbone features.
        """

        if source_prompt.ndim == 3:
            source_prompt = source_prompt.unsqueeze(1)
        if source_prompt.ndim != 4 or source_prompt.shape[1] != 1:
            raise ValueError("source_prompt must have shape Bx1xHxW")
        if self.matcher == "masked_xcorr_v6":
            if source_point_px is None:
                raise ValueError("masked_xcorr_v6 requires an explicit source point anchor")
            features = F.avg_pool2d(self.search_projection(features), kernel_size=2)
            weights = F.interpolate(
                source_prompt.to(features), size=features.shape[-2:], mode="area",
            ).clamp(0, 1)
            scale = features.new_tensor([
                features.shape[-1] / source_prompt.shape[-1],
                features.shape[-2] / source_prompt.shape[-2],
            ])
            anchors = (source_point_px.to(features) + 0.5) * scale - 0.5
            return self._masked_template(F.normalize(features, dim=1), weights, anchors)
        if self.matcher in {"xcorr_template_v4", "multiscale_xcorr_v5"}:
            features = F.avg_pool2d(self.search_projection(features), kernel_size=2)
            weights = F.interpolate(
                source_prompt.to(dtype=features.dtype, device=features.device),
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp_min(0)
            if self.matcher == "xcorr_template_v4":
                return ReferenceEncoding(
                    self._extract_centered_template(features, weights, size=13)
                )
            templates = tuple(
                self._extract_centered_template(features, weights, size=13, offset_scale=scale)
                for scale in (0.65, 1.0, 1.5)
            )
            return ReferenceEncoding(torch.stack(templates, dim=1))
        weights = F.interpolate(
            source_prompt.to(dtype=features.dtype, device=features.device),
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp_min(0)
        denominator = weights.sum(dim=(-2, -1)).clamp_min(1e-6)
        query = (features * weights).sum(dim=(-2, -1)) / denominator
        if self.matcher == "response_relation_v2":
            query = self.reference_projection(query)
            return ReferenceEncoding(F.normalize(query, dim=1))
        if self.matcher == "spatial_prototypes_v3":
            prototypes = self._spatial_prototypes(features, weights)
            prototypes = self.reference_projection(prototypes)
            return ReferenceEncoding(F.normalize(prototypes, dim=2))
        return ReferenceEncoding(F.normalize(query, dim=1))

    def encode_search(self, image: Tensor) -> SearchEncoding:
        return self.encode_search_features(self._encode_image(image), tuple(image.shape[-2:]))

    def encode_search_features(
        self, features: Tensor, output_size_hw: tuple[int, int]
    ) -> SearchEncoding:
        """Project precomputed stride-4 pyramid features for matching."""

        if self.matcher in {"response_relation_v2", "spatial_prototypes_v3"}:
            features = self.search_projection(features)
        elif self.matcher in {"xcorr_template_v4", "multiscale_xcorr_v5", "masked_xcorr_v6"}:
            features = F.avg_pool2d(self.search_projection(features), kernel_size=2)
        return SearchEncoding(features, output_size_hw)

    def match_encoded(
        self,
        reference: ReferenceEncoding,
        search: SearchEncoding,
    ) -> TrackerOutput:
        features = F.normalize(search.features, dim=1)
        query = reference.query
        if query.shape[0] == 1 and features.shape[0] > 1:
            query = query.expand(features.shape[0], *query.shape[1:])
        if query.shape[0] != features.shape[0]:
            raise ValueError("reference and search batch sizes are incompatible")
        if self.matcher == "cosine_v1":
            correlation = torch.einsum("bc,bchw->bhw", query, features).unsqueeze(1)
            low_resolution_logits = self.location_head(correlation)
            pooled = torch.stack(
                (
                    correlation.amax(dim=(-2, -1))[:, 0],
                    correlation.mean(dim=(-2, -1))[:, 0],
                ),
                dim=1,
            )
        elif self.matcher == "response_relation_v2":
            broadcast_query = query[:, :, None, None]
            relation = torch.cat(
                (features * broadcast_query, (features - broadcast_query).abs()), dim=1
            )
            relation_features = self.relation_encoder(relation)
            low_resolution_logits = self.location_head(relation_features)
            response = low_resolution_logits[:, 0]
            pooled = torch.cat(
                (
                    relation_features.mean(dim=(-2, -1)),
                    relation_features.amax(dim=(-2, -1)),
                    response.amax(dim=(-2, -1), keepdim=False)[:, None],
                    response.mean(dim=(-2, -1), keepdim=False)[:, None],
                    response.std(dim=(-2, -1), keepdim=False)[:, None],
                ),
                dim=1,
            )
        elif self.matcher == "spatial_prototypes_v3":
            correlations = torch.einsum("bkc,bchw->bkhw", query, features)
            relation_features = self.relation_encoder(correlations)
            low_resolution_logits = self.location_head(relation_features)
            response = low_resolution_logits[:, 0]
            pooled = torch.cat(
                (
                    relation_features.mean(dim=(-2, -1)),
                    relation_features.amax(dim=(-2, -1)),
                    response.amax(dim=(-2, -1), keepdim=False)[:, None],
                    response.mean(dim=(-2, -1), keepdim=False)[:, None],
                    response.std(dim=(-2, -1), keepdim=False)[:, None],
                ),
                dim=1,
            )
        elif self.matcher in {"xcorr_template_v4", "masked_xcorr_v6"}:
            if self.matcher == "masked_xcorr_v6":
                weights = reference.spatial_weights
                if weights is None:
                    raise ValueError("masked correlation requires reference spatial weights")
                weights = weights.expand(features.shape[0], *weights.shape[1:])
                correlation = self._weighted_cross_correlation(query, features, weights)
            else:
                correlation = self._normalized_cross_correlation(query, features)
            relation_features = self.relation_encoder(correlation)
            low_resolution_logits = self.location_head(relation_features)
            response = low_resolution_logits[:, 0]
            pooled = torch.cat(
                (
                    relation_features.mean(dim=(-2, -1)),
                    relation_features.amax(dim=(-2, -1)),
                    response.amax(dim=(-2, -1), keepdim=False)[:, None],
                    response.mean(dim=(-2, -1), keepdim=False)[:, None],
                    response.std(dim=(-2, -1), keepdim=False)[:, None],
                ),
                dim=1,
            )
        else:
            correlations = torch.cat(
                tuple(
                    self._normalized_cross_correlation(query[:, index], features)
                    for index in range(query.shape[1])
                ),
                dim=1,
            )
            relation_features = self.relation_encoder(correlations)
            low_resolution_logits = self.location_head(relation_features)
            response = low_resolution_logits[:, 0]
            pooled = torch.cat(
                (
                    relation_features.mean(dim=(-2, -1)),
                    relation_features.amax(dim=(-2, -1)),
                    response.amax(dim=(-2, -1), keepdim=False)[:, None],
                    response.mean(dim=(-2, -1), keepdim=False)[:, None],
                    response.std(dim=(-2, -1), keepdim=False)[:, None],
                ),
                dim=1,
            )
        location_logits = F.interpolate(
            low_resolution_logits,
            size=search.output_size_hw,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        return TrackerOutput(self.match_head(pooled)[:, 0], location_logits)

    def forward(
        self,
        reference_image: Tensor,
        source_prompt: Tensor,
        search_image: Tensor,
        *, source_point_px: Tensor | None = None,
    ) -> TrackerOutput:
        reference = self.encode_reference(
            reference_image, source_prompt, source_point_px=source_point_px,
        )
        search = self.encode_search(search_image)
        return self.match_encoded(reference, search)

    def _encode_image(self, image: Tensor) -> Tensor:
        represented = self._input_features(image)
        normalized = normalize_for_backbone(represented, self.backbone.preprocessing)
        return self.pyramid(self.backbone.forward_multiscale(normalized))

    def _input_features(self, image: Tensor) -> Tensor:
        if self.input_representation == "rgb":
            return image
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("tracker images must have shape Bx3xHxW")
        luminance = (
            0.2126 * image[:, 0:1]
            + 0.7152 * image[:, 1:2]
            + 0.0722 * image[:, 2:3]
        )
        local_mean = F.avg_pool2d(luminance, kernel_size=31, stride=1, padding=15)
        local_square = F.avg_pool2d(
            luminance.square(), kernel_size=31, stride=1, padding=15
        )
        local_scale = (local_square - local_mean.square()).clamp_min(1e-4).sqrt()
        local_contrast = ((luminance - local_mean) / local_scale).clamp(-3, 3)
        local_contrast = local_contrast / 6 + 0.5
        dx = F.pad(luminance[..., 1:] - luminance[..., :-1], (0, 1, 0, 0))
        dy = F.pad(luminance[..., 1:, :] - luminance[..., :-1, :], (0, 0, 0, 1))
        gradient = torch.sqrt(dx.square() + dy.square() + 1e-8)
        gradient_scale = gradient.flatten(2).mean(dim=2, keepdim=True)[..., None]
        gradient = (gradient / gradient_scale.clamp_min(1e-4) / 6).clamp(0, 1)
        features = torch.cat((image, local_contrast, gradient), dim=1)
        return self.input_adapter(features).clamp(0, 1)

    @staticmethod
    def _spatial_prototypes(features: Tensor, weights: Tensor) -> Tensor:
        """Pool center and directional descriptors around the prompt centroid."""

        batch, _, height, width = features.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=features.device, dtype=features.dtype),
            torch.arange(width, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        total = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        center_x = (weights * xx).sum(dim=(-2, -1), keepdim=True) / total
        center_y = (weights * yy).sum(dim=(-2, -1), keepdim=True) / total
        dx = xx - center_x
        dy = yy - center_y
        scale = torch.sqrt(
            (weights * (dx.square() + dy.square())).sum(dim=(-2, -1), keepdim=True) / total
        ).clamp_min(1.0)
        softness = 0.25 * scale
        regions = torch.cat(
            (
                weights,
                weights * torch.sigmoid(-dx / softness),
                weights * torch.sigmoid(dx / softness),
                weights * torch.sigmoid(-dy / softness),
                weights * torch.sigmoid(dy / softness),
            ),
            dim=1,
        )
        normalized = regions / regions.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return torch.einsum("bkhw,bchw->bkc", normalized, features)

    @staticmethod
    def _extract_centered_template(
        features: Tensor,
        weights: Tensor,
        *,
        size: int,
        offset_scale: float = 1.0,
    ) -> Tensor:
        """Sample a fixed feature template around the prompt centroid."""

        if size <= 0 or size % 2 == 0:
            raise ValueError("template size must be a positive odd integer")
        batch, _, height, width = features.shape
        total = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=features.device, dtype=features.dtype),
            torch.arange(width, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        center_x = (weights * xx).sum(dim=(-2, -1), keepdim=True) / total
        center_y = (weights * yy).sum(dim=(-2, -1), keepdim=True) / total
        if offset_scale <= 0:
            raise ValueError("template offset scale must be positive")
        offsets = torch.arange(
            -(size // 2),
            size // 2 + 1,
            device=features.device,
            dtype=features.dtype,
        ) * float(offset_scale)
        grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
        sample_x = center_x[:, 0, 0, 0, None, None] + grid_x
        sample_y = center_y[:, 0, 0, 0, None, None] + grid_y
        normalized_x = 2 * sample_x / max(width - 1, 1) - 1
        normalized_y = 2 * sample_y / max(height - 1, 1) - 1
        grid = torch.stack((normalized_x, normalized_y), dim=-1)
        if grid.shape[0] != batch:  # pragma: no cover - shape invariant
            raise RuntimeError("template grid batch mismatch")
        return F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )

    @staticmethod
    def _masked_template(features: Tensor, weights: Tensor, anchors: Tensor) -> ReferenceEncoding:
        """Cover the mask's extent but retain the annotation as the zero-offset anchor."""
        batch, _, height, width = features.shape
        if anchors.shape != (batch, 2) or not torch.isfinite(anchors).all():
            raise ValueError("source anchors must be finite Bx2 points")
        yy, xx = torch.meshgrid(
            torch.arange(height, device=features.device),
            torch.arange(width, device=features.device), indexing="ij",
        )
        distances = torch.maximum(
            (xx[None] - anchors[:, 0, None, None]).abs(),
            (yy[None] - anchors[:, 1, None, None]).abs(),
        )
        support = weights[:, 0] > 0.01
        radius = max(1, int(torch.ceil(distances.masked_fill(~support, 0).max()).item()) + 1)
        offsets = torch.arange(-radius, radius+1, device=features.device, dtype=features.dtype)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        gx = 2 * (anchors[:, 0, None, None] + dx) / max(width-1, 1) - 1
        gy = 2 * (anchors[:, 1, None, None] + dy) / max(height-1, 1) - 1
        grid = torch.stack([gx, gy], dim=-1)
        template = F.grid_sample(features, grid, align_corners=True, padding_mode="zeros")
        spatial = F.grid_sample(weights, grid, align_corners=True, padding_mode="zeros")
        spatial = spatial.masked_fill(spatial < 0.01, 0)
        return ReferenceEncoding(template, spatial)

    @staticmethod
    def _weighted_cross_correlation(template: Tensor, features: Tensor, weights: Tensor) -> Tensor:
        """Weighted NCC: ignored background affects neither numerator nor denominator."""
        with torch.autocast(device_type=features.device.type, enabled=False):
            template, features, weights = template.float(), features.float(), weights.float()
            batch, channels, height, width = features.shape
            weights = weights.expand(-1, channels, -1, -1)
            total = weights.sum((1, 2, 3), keepdim=True).clamp_min(1e-6)
            mean = (template * weights).sum((1, 2, 3), keepdim=True) / total
            centered = template - mean
            variance = (centered.square() * weights).sum((1, 2, 3), keepdim=True)
            kernel = centered * weights / variance.clamp_min(1e-6).sqrt()
            grouped = features.reshape(1, batch*channels, height, width)
            padding = (template.shape[-2] // 2, template.shape[-1] // 2)
            numerator = F.conv2d(grouped, kernel, padding=padding, groups=batch)
            sums = F.conv2d(grouped, weights, padding=padding, groups=batch)
            squares = F.conv2d(grouped.square(), weights, padding=padding, groups=batch)
            target_variance = (squares - sums.square() / total.reshape(1, batch, 1, 1))
            result = numerator / target_variance.clamp_min(1e-6).sqrt()
            return result.reshape(batch, 1, height, width).clamp(-1, 1)

    @staticmethod
    def _normalized_cross_correlation(template: Tensor, features: Tensor) -> Tensor:
        """Compare one feature template against every search location."""

        if template.ndim != 4 or features.ndim != 4:
            raise ValueError("template and search features must be four-dimensional")
        if template.shape[:2] != features.shape[:2]:
            raise ValueError("template and search channel dimensions differ")
        batch, channels, height, width = features.shape
        kernel_height, kernel_width = template.shape[-2:]
        centered = template - template.mean(dim=(1, 2, 3), keepdim=True)
        centered = (
            centered / centered.square().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6).sqrt()
        )
        grouped_features = features.reshape(1, batch * channels, height, width)
        padding = (kernel_height // 2, kernel_width // 2)
        numerator = F.conv2d(
            grouped_features,
            centered,
            padding=padding,
            groups=batch,
        )
        ones = torch.ones_like(centered)
        patch_sum = F.conv2d(
            grouped_features,
            ones,
            padding=padding,
            groups=batch,
        )
        patch_square_sum = F.conv2d(
            grouped_features.square(),
            ones,
            padding=padding,
            groups=batch,
        )
        elements = float(channels * kernel_height * kernel_width)
        patch_variance_sum = (patch_square_sum - patch_sum.square() / elements).clamp_min(1e-6)
        correlation = numerator / patch_variance_sum.sqrt()
        return correlation.reshape(batch, 1, height, width)
