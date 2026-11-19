"""Losses for explicit match/no-match and conditional localization."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from glaciertrack.models.outputs import TrackerOutput


@dataclass(frozen=True)
class TrackerLoss:
    total: Tensor
    match: Tensor
    localization: Tensor
    positive_localization: Tensor
    positive_peak_ranking: Tensor
    negative_localization: Tensor
    retrieval_ranking: Tensor


def tracker_loss(
    output: TrackerOutput,
    *,
    match_present: Tensor,
    target_heatmap: Tensor,
    match_positive_weight: float = 1.0,
    localization_weight: float = 1.0,
    positive_localization_weight: float = 0.25,
    positive_localization_margin: float = 1.0,
    positive_peak_ranking_weight: float = 0.0,
    positive_peak_ranking_margin: float = 1.0,
    negative_localization_weight: float = 0.25,
    negative_localization_margin: float = 0.0,
    negative_localization_temperature: float = 0.1,
    group_ids: Sequence[str] | None = None,
    retrieval_ranking_weight: float = 0.0,
    retrieval_ranking_margin: float = 1.0,
    retrieval_ranking_temperature: float = 0.1,
    retrieval_match_logit_weight: float = 1.0,
) -> TrackerLoss:
    """Combine match, positive localization, and negative peak suppression losses.

    Positive spatial cross entropy determines *where* the matching boulder is.
    It is invariant to adding a constant to every location logit, so negative
    crops also receive a smooth peak penalty. Positive crops receive the
    complementary target-support penalty, which resolves spatial cross
    entropy's otherwise unconstrained absolute logit level.
    """

    if negative_localization_temperature <= 0:
        raise ValueError("negative_localization_temperature must be positive")
    if retrieval_ranking_temperature <= 0:
        raise ValueError("retrieval_ranking_temperature must be positive")
    if match_positive_weight <= 0:
        raise ValueError("match_positive_weight must be positive")
    if min(
        localization_weight,
        positive_localization_weight,
        positive_peak_ranking_weight,
        negative_localization_weight,
        retrieval_ranking_weight,
    ) < 0:
        raise ValueError("localization loss weights cannot be negative")
    if retrieval_match_logit_weight < 0:
        raise ValueError("retrieval_match_logit_weight cannot be negative")

    targets = match_present.to(device=output.match_logits.device, dtype=torch.float32).reshape(-1)
    logits = output.match_logits.reshape(-1).float()
    if targets.shape != logits.shape:
        raise ValueError("match target shape differs from model output")
    positive_weight = torch.as_tensor(
        match_positive_weight, device=logits.device, dtype=logits.dtype
    )
    match = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
    )
    heatmap = target_heatmap.to(device=output.location_logits.device, dtype=torch.float32)
    if heatmap.ndim == 4 and heatmap.shape[1] == 1:
        heatmap = heatmap[:, 0]
    if heatmap.shape != output.location_logits.shape:
        raise ValueError("target heatmap shape differs from location output")
    positives = targets.bool()
    if bool(positives.any()):
        distributions = heatmap[positives].flatten(1)
        distributions = distributions / distributions.sum(1, keepdim=True).clamp_min(1e-12)
        location_logits = output.location_logits[positives].float().flatten(1)
        log_probabilities = F.log_softmax(location_logits, dim=1)
        localization = -(distributions * log_probabilities).sum(1).mean()
        target_support = (distributions * location_logits).sum(1)
        positive_localization = F.softplus(
            float(positive_localization_margin) - target_support
        ).mean()
        support_threshold = distributions.amax(dim=1, keepdim=True) * 0.05
        hard_negative_logits = location_logits.masked_fill(
            distributions > support_threshold, -torch.inf
        )
        hard_negative = hard_negative_logits.amax(dim=1)
        positive_peak_ranking = F.softplus(
            hard_negative - target_support + float(positive_peak_ranking_margin)
        ).mean()
    else:
        localization = output.location_logits.sum() * 0
        positive_localization = output.location_logits.sum() * 0
        positive_peak_ranking = output.location_logits.sum() * 0
    negatives = ~positives
    if bool(negatives.any()):
        negative_logits = output.location_logits[negatives].float().flatten(1)
        temperature = float(negative_localization_temperature)
        smooth_peak = temperature * (
            torch.logsumexp(negative_logits / temperature, dim=1)
            - math.log(negative_logits.shape[1])
        )
        negative_localization = F.softplus(
            smooth_peak - float(negative_localization_margin)
        ).mean()
    else:
        negative_localization = output.location_logits.sum() * 0
    retrieval_ranking = _grouped_retrieval_ranking_loss(
        output,
        targets=targets,
        group_ids=group_ids,
        margin=float(retrieval_ranking_margin),
        temperature=float(retrieval_ranking_temperature),
        match_logit_weight=float(retrieval_match_logit_weight),
    )
    total = (
        match
        + localization_weight * localization
        + positive_localization_weight * positive_localization
        + positive_peak_ranking_weight * positive_peak_ranking
        + negative_localization_weight * negative_localization
        + retrieval_ranking_weight * retrieval_ranking
    )
    return TrackerLoss(
        total,
        match,
        localization,
        positive_localization,
        positive_peak_ranking,
        negative_localization,
        retrieval_ranking,
    )


def _grouped_retrieval_ranking_loss(
    output: TrackerOutput,
    *,
    targets: Tensor,
    group_ids: Sequence[str] | None,
    margin: float,
    temperature: float,
    match_logit_weight: float,
) -> Tensor:
    zero = output.location_logits.sum() * 0
    if group_ids is None:
        return zero
    if len(group_ids) != targets.numel():
        raise ValueError("group_ids length differs from model output")
    spatial_peak = output.location_logits.float().flatten(1).amax(dim=1)
    fused_scores = spatial_peak + match_logit_weight * output.match_logits.float().reshape(-1)
    losses: list[Tensor] = []
    for group_id in dict.fromkeys(str(value) for value in group_ids):
        group = torch.as_tensor(
            [str(value) == group_id for value in group_ids],
            device=targets.device,
            dtype=torch.bool,
        )
        positives = fused_scores[group & targets.bool()]
        negatives = fused_scores[group & ~targets.bool()]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        # Operational retrieval takes a literal argmax over every stitched
        # candidate. Use those same extrema here; a smooth average can improve
        # while leaving the actual winning tile unchanged.
        best_positive = positives.amax()
        hard_negative = negatives.amax()
        losses.append(F.softplus(hard_negative - best_positive + margin))
    if not losses:
        return zero
    return torch.stack(losses).mean()
