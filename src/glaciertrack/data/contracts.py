"""Canonical records shared by preparation, training, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class NegativeType(str, Enum):
    """Reasons why a valid source query has no target location."""

    TARGET_OUTSIDE_WINDOW = "target_outside_window"
    TARGET_UNOBSERVABLE = "target_unobservable"
    DISTRACTOR = "distractor"


class ObservationStatus(str, Enum):
    """Review state for one track in one survey."""

    VISIBLE = "visible"
    UNOBSERVABLE = "unobservable"
    OUTSIDE_COVERAGE = "outside_coverage"
    UNKNOWN = "unknown"
    INVALID_PROMPT = "invalid_prompt"


@dataclass(frozen=True)
class Survey:
    glacier_id: str
    survey_id: str
    acquisition_date: date
    raster_uri: str
    crs: str
    native_gsd_x_m: float
    native_gsd_y_m: float
    registration_rmse_m: float

    def __post_init__(self) -> None:
        if self.native_gsd_x_m <= 0 or self.native_gsd_y_m <= 0:
            raise ValueError("native survey GSD values must be positive")
        if self.registration_rmse_m < 0:
            raise ValueError("registration RMSE cannot be negative")


@dataclass(frozen=True)
class PairSample:
    """A prompted reference/search pair; point coordinates use x/y pixel order."""

    sample_id: str
    glacier_id: str
    track_id: str
    spatial_group: str
    reference_survey_id: str
    target_survey_id: str
    source_point_px: tuple[float, float]
    target_point_px: tuple[float, float] | None
    source_valid: bool
    match_present: bool
    negative_type: NegativeType | None
    tensor_gsd_m_per_px: float
    reference_native_gsd_m_per_px: float
    target_native_gsd_m_per_px: float
    reference_effective_gsd_m_per_px: float
    target_effective_gsd_m_per_px: float

    def __post_init__(self) -> None:
        gsd_values = (
            self.tensor_gsd_m_per_px,
            self.reference_native_gsd_m_per_px,
            self.target_native_gsd_m_per_px,
            self.reference_effective_gsd_m_per_px,
            self.target_effective_gsd_m_per_px,
        )
        if any(value <= 0 for value in gsd_values):
            raise ValueError("all GSD values must be positive")
        if self.reference_effective_gsd_m_per_px < self.reference_native_gsd_m_per_px:
            raise ValueError("reference effective GSD cannot be finer than native GSD")
        if self.target_effective_gsd_m_per_px < self.target_native_gsd_m_per_px:
            raise ValueError("target effective GSD cannot be finer than native GSD")
        if not self.source_valid and self.match_present:
            raise ValueError("an invalid source query cannot have a target match")
        if self.match_present:
            if self.target_point_px is None:
                raise ValueError("positive samples require a target point")
            if self.negative_type is not None:
                raise ValueError("positive samples cannot have a negative type")
        elif self.source_valid and self.negative_type is None:
            raise ValueError("valid-source no-match samples require a negative type")
