"""Overlap-aware reconstruction of query-conditioned target heatmaps."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from glaciertrack.inference.search import SearchTile


@dataclass(frozen=True)
class TilePrediction:
    """Model output for one target tile."""

    tile: SearchTile
    location_logits: Tensor
    match_logit: float


@dataclass(frozen=True)
class Candidate:
    """Global candidate and the features used for confidence calibration."""

    x_px: float
    y_px: float
    peak_logit: float
    peak_margin: float
    entropy: float
    overlap_disagreement: float
    tile_match_probability: float
    boundary_distance_px: float
    searched_tiles: int
    confidence: float | None = None
    rank: int = 1

    def feature_mapping(self) -> dict[str, float]:
        return {
            "peak_logit": self.peak_logit,
            "peak_margin": self.peak_margin,
            "entropy": self.entropy,
            "overlap_disagreement": self.overlap_disagreement,
            "tile_match_probability": self.tile_match_probability,
            "boundary_distance_px": self.boundary_distance_px,
            "log_searched_tiles": math.log1p(self.searched_tiles),
            "log_candidate_rank": math.log(max(self.rank, 1)),
        }


@dataclass(frozen=True)
class LinearConfidenceCalibrator:
    """Serializable logistic calibrator fitted on held-out validation queries."""

    coefficients: Mapping[str, float]
    intercept: float = 0.0
    feature_means: Mapping[str, float] | None = None
    feature_scales: Mapping[str, float] | None = None

    def predict(self, candidate: Candidate) -> float:
        value = float(self.intercept)
        features = candidate.feature_mapping()
        unknown = set(self.coefficients).difference(features)
        if unknown:
            raise ValueError(f"unknown confidence features: {sorted(unknown)}")
        means = self.feature_means or {}
        scales = self.feature_scales or {}
        value += sum(
            float(weight)
            * (features[name] - float(means.get(name, 0.0)))
            / float(scales.get(name, 1.0))
            for name, weight in self.coefficients.items()
        )
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    def to_mapping(self) -> dict[str, object]:
        return {
            "type": "linear_logistic",
            "coefficients": dict(self.coefficients),
            "intercept": self.intercept,
            "feature_means": dict(self.feature_means or {}),
            "feature_scales": dict(self.feature_scales or {}),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> LinearConfidenceCalibrator:
        if payload.get("type") != "linear_logistic":
            raise ValueError("unsupported confidence calibrator type")
        coefficients = payload.get("coefficients")
        if not isinstance(coefficients, Mapping):
            raise ValueError("calibrator coefficients must be a mapping")
        means = payload.get("feature_means", {})
        scales = payload.get("feature_scales", {})
        if not isinstance(means, Mapping) or not isinstance(scales, Mapping):
            raise ValueError("calibrator feature statistics must be mappings")
        return cls(
            {str(key): float(value) for key, value in coefficients.items()},
            float(payload.get("intercept", 0.0)),
            {str(key): float(value) for key, value in means.items()},
            {str(key): float(value) for key, value in scales.items()},
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_mapping(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> LinearConfidenceCalibrator:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("confidence calibrator file must contain a mapping")
        return cls.from_mapping(payload)


def fit_confidence_calibrator(
    candidates: list[Candidate] | tuple[Candidate, ...],
    labels: list[bool] | tuple[bool, ...],
    *,
    l2_weight: float = 1e-3,
) -> LinearConfidenceCalibrator:
    """Fit logistic confidence on held-out validation candidates only."""

    if len(candidates) != len(labels) or not candidates:
        raise ValueError("candidates and labels must have equal non-zero length")
    if len(set(labels)) != 2:
        raise ValueError("confidence calibration requires positive and negative labels")
    names = tuple(candidates[0].feature_mapping())
    rows = [[candidate.feature_mapping()[name] for name in names] for candidate in candidates]
    features = torch.tensor(rows, dtype=torch.float64)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("confidence features must be finite")
    targets = torch.tensor(labels, dtype=torch.float64)
    means = features.mean(dim=0)
    scales = features.std(dim=0).clamp_min(1e-6)
    standardized = (features - means) / scales
    weights = torch.zeros(len(names), dtype=torch.float64, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights, intercept],
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        logits = standardized @ weights + intercept
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = loss + l2_weight * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return LinearConfidenceCalibrator(
        coefficients=dict(zip(names, weights.detach().tolist(), strict=True)),
        intercept=float(intercept.detach()),
        feature_means=dict(zip(names, means.tolist(), strict=True)),
        feature_scales=dict(zip(names, scales.tolist(), strict=True)),
    )


class HeatmapStitcher:
    """Accumulate tile logits with smooth overlap weights on a common grid."""

    def __init__(
        self,
        raster_shape_hw: tuple[int, int],
        valid_search_mask: Tensor | None = None,
        *,
        stride_px: int = 1,
        active_bounds_px: tuple[int, int, int, int] | None = None,
        match_logit_weight: float = 1.0,
    ) -> None:
        height, width = (int(value) for value in raster_shape_hw)
        if min(height, width) <= 0:
            raise ValueError("raster dimensions must be positive")
        if stride_px <= 0:
            raise ValueError("stitch stride must be positive")
        if match_logit_weight < 0:
            raise ValueError("match_logit_weight must be non-negative")
        self.raster_shape = (height, width)
        self.stride_px = int(stride_px)
        self.match_logit_weight = float(match_logit_weight)
        full_shape = (
            math.ceil(height / self.stride_px),
            math.ceil(width / self.stride_px),
        )
        if active_bounds_px is None:
            top, left, bottom, right = 0, 0, height, width
        else:
            top, left, bottom, right = (int(value) for value in active_bounds_px)
            if top < 0 or left < 0 or bottom > height or right > width:
                raise ValueError("active stitch bounds lie outside the raster")
            if bottom <= top or right <= left:
                raise ValueError("active stitch bounds must have positive area")
        self.origin = (top // self.stride_px, left // self.stride_px)
        coarse_bottom = math.ceil(bottom / self.stride_px)
        coarse_right = math.ceil(right / self.stride_px)
        self.shape = (
            coarse_bottom - self.origin[0],
            coarse_right - self.origin[1],
        )
        self.weighted_sum = torch.zeros(self.shape, dtype=torch.float32)
        self.weighted_square_sum = torch.zeros(self.shape, dtype=torch.float32)
        self.weight_sum = torch.zeros(self.shape, dtype=torch.float32)
        self.best_match_probability = torch.zeros(self.shape, dtype=torch.float32)
        self.searched_mask = torch.zeros(self.shape, dtype=torch.bool)
        if valid_search_mask is None:
            self.valid_search_mask = torch.ones(self.shape, dtype=torch.bool)
        else:
            valid = torch.as_tensor(valid_search_mask, dtype=torch.bool)
            if tuple(valid.shape) != full_shape:
                raise ValueError(
                    "valid search mask shape differs from the strided raster grid"
                )
            row, column = self.origin
            # Own the active mask. Radius search narrows it in place, and a
            # view here would corrupt the session-wide AOI shared by all later
            # queries, making predictions depend on query order.
            self.valid_search_mask = valid[
                row : row + self.shape[0], column : column + self.shape[1]
            ].clone()
        self.searched_tiles = 0

    def add(self, prediction: TilePrediction) -> None:
        tile = prediction.tile
        logits = torch.as_tensor(prediction.location_logits, dtype=torch.float32).squeeze()
        if logits.ndim != 2:
            raise ValueError("tile location logits must be two-dimensional")
        if tuple(logits.shape) != (tile.height, tile.width):
            logits = F.interpolate(
                logits[None, None],
                size=(tile.height, tile.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        global_top = tile.row_off // self.stride_px
        global_left = tile.col_off // self.stride_px
        global_bottom = math.ceil((tile.row_off + tile.height) / self.stride_px)
        global_right = math.ceil((tile.col_off + tile.width) / self.stride_px)
        output_size = (global_bottom - global_top, global_right - global_left)
        if tuple(logits.shape) != output_size:
            logits = F.interpolate(
                logits[None, None],
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        # Location cross entropy learns relative position within a positive
        # tile.  Shift the whole tile surface by the independently trained
        # match evidence so no-match tiles cannot win merely through a sharp
        # but spurious local peak.
        logits = logits + self.match_logit_weight * float(prediction.match_logit)
        weight = _blend_weight(*output_size)
        local_top = global_top - self.origin[0]
        local_left = global_left - self.origin[1]
        rows = slice(local_top, local_top + output_size[0])
        columns = slice(local_left, local_left + output_size[1])
        valid = self.valid_search_mask[rows, columns]
        weight = weight * valid
        self.weighted_sum[rows, columns] += logits * weight
        self.weighted_square_sum[rows, columns] += logits.square() * weight
        self.weight_sum[rows, columns] += weight
        probability = torch.sigmoid(torch.tensor(float(prediction.match_logit)))
        self.best_match_probability[rows, columns] = torch.maximum(
            self.best_match_probability[rows, columns], probability
        )
        self.searched_mask[rows, columns] |= valid
        self.searched_tiles += 1

    def candidate(
        self,
        *,
        calibrator: LinearConfidenceCalibrator | None = None,
        exclusion_radius_px: int = 5,
    ) -> Candidate | None:
        """Return the highest-logit candidate on the stitched search surface."""

        candidates = self.candidates(
            maximum_candidates=1,
            minimum_separation_px=exclusion_radius_px,
            calibrator=calibrator,
            margin_exclusion_radius_px=exclusion_radius_px,
        )
        return candidates[0] if candidates else None

    def candidates(
        self,
        *,
        maximum_candidates: int,
        minimum_separation_px: float,
        calibrator: LinearConfidenceCalibrator | None = None,
        margin_exclusion_radius_px: int = 5,
    ) -> tuple[Candidate, ...]:
        """Return ranked, spatially separated candidates without copying the mosaic.

        Calibration uses this to retain plausible alternative peaks instead of
        reducing every validation heatmap to a single winner.  Candidate
        separation controls sampling only; confidence features retain the same
        local margin definition used during operational inference.
        """

        if maximum_candidates <= 0:
            raise ValueError("maximum_candidates must be positive")
        if minimum_separation_px < 0:
            raise ValueError("minimum_separation_px must be non-negative")
        covered = self.weight_sum > 0
        if not bool(covered.any()):
            return ()
        logits = torch.full(self.shape, -torch.inf, dtype=torch.float32)
        logits[covered] = self.weighted_sum[covered] / self.weight_sum[covered]
        covered_logits = logits[covered]
        probabilities = torch.softmax(covered_logits, dim=0)
        entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())

        covered_count = int(covered.sum())
        # A broad peak can occupy many high-valued cells.  Retaining a generous
        # raw pool lets the greedy separation step reach distinct alternatives
        # while keeping the returned candidate table small.
        pool_size = min(covered_count, max(4096, maximum_candidates * 2048))
        values, indices = torch.topk(logits.flatten(), pool_size)
        separation = float(minimum_separation_px) / self.stride_px
        selected: list[tuple[int, int]] = []
        for flat_index in indices.tolist():
            local_y, local_x = divmod(int(flat_index), self.shape[1])
            if any(
                math.hypot(local_x - other_x, local_y - other_y) < separation
                for other_y, other_x in selected
            ):
                continue
            selected.append((local_y, local_x))
            if len(selected) >= maximum_candidates:
                break

        ranked_values = values.tolist()
        ranked_indices = indices.tolist()
        results = tuple(
            self._candidate_at(
                logits,
                local_y,
                local_x,
                rank=rank,
                entropy=entropy,
                ranked_values=ranked_values,
                ranked_indices=ranked_indices,
                margin_exclusion_radius_px=margin_exclusion_radius_px,
                calibrator=calibrator,
            )
            for rank, (local_y, local_x) in enumerate(selected, start=1)
        )
        return results

    def _candidate_at(
        self,
        logits: Tensor,
        local_y: int,
        local_x: int,
        *,
        rank: int,
        entropy: float,
        ranked_values: list[float],
        ranked_indices: list[int],
        margin_exclusion_radius_px: int,
        calibrator: LinearConfidenceCalibrator | None,
    ) -> Candidate:
        peak = float(logits[local_y, local_x])
        exclusion = max(1, math.ceil(margin_exclusion_radius_px / self.stride_px))
        second = -math.inf
        for value, flat_index in zip(ranked_values, ranked_indices, strict=True):
            other_y, other_x = divmod(int(flat_index), self.shape[1])
            if abs(other_y - local_y) > exclusion or abs(other_x - local_x) > exclusion:
                second = float(value)
                break
        # A single covered region has no competing peak and therefore supplies
        # no measurable margin evidence for confidence calibration.
        margin = peak - second if math.isfinite(second) else 0.0
        mean = self.weighted_sum[local_y, local_x] / self.weight_sum[local_y, local_x]
        mean_square = (
            self.weighted_square_sum[local_y, local_x]
            / self.weight_sum[local_y, local_x]
        )
        disagreement = float(torch.sqrt((mean_square - mean.square()).clamp_min(0)))
        coarse_y = self.origin[0] + local_y
        coarse_x = self.origin[1] + local_x
        x = min(
            coarse_x * self.stride_px + (self.stride_px - 1) / 2,
            self.raster_shape[1] - 1,
        )
        y = min(
            coarse_y * self.stride_px + (self.stride_px - 1) / 2,
            self.raster_shape[0] - 1,
        )
        boundary = float(
            min(x, y, self.raster_shape[1] - 1 - x, self.raster_shape[0] - 1 - y)
        )
        candidate = Candidate(
            x_px=float(x),
            y_px=float(y),
            peak_logit=peak,
            peak_margin=margin,
            entropy=entropy,
            overlap_disagreement=disagreement,
            tile_match_probability=float(
                self.best_match_probability[local_y, local_x]
            ),
            boundary_distance_px=boundary,
            searched_tiles=self.searched_tiles,
            rank=rank,
        )
        return (
            candidate
            if calibrator is None
            else replace(candidate, confidence=calibrator.predict(candidate))
        )

    def mosaic_logits(self) -> Tensor:
        result = torch.full(self.shape, -torch.inf, dtype=torch.float32)
        covered = self.weight_sum > 0
        result[covered] = self.weighted_sum[covered] / self.weight_sum[covered]
        return result


def _blend_weight(height: int, width: int) -> Tensor:
    vertical = torch.hann_window(height, periodic=False) if height > 1 else torch.ones(1)
    horizontal = torch.hann_window(width, periodic=False) if width > 1 else torch.ones(1)
    return torch.outer(vertical, horizontal).clamp_min_(0.05)
