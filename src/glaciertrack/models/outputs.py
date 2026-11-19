"""Stable model output shared by training and operational inference."""

from dataclasses import dataclass

import torch


@dataclass
class TrackerOutput:
    """Unnormalized match and conditional location predictions."""

    match_logits: torch.Tensor
    location_logits: torch.Tensor

    def __post_init__(self) -> None:
        if self.match_logits.ndim not in {1, 2}:
            raise ValueError("match_logits must have shape B or Bx1")
        if self.location_logits.ndim != 3:
            raise ValueError("location_logits must have shape BxHxW")
        if self.match_logits.shape[0] != self.location_logits.shape[0]:
            raise ValueError("match and location batch sizes differ")
