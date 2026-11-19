"""Build reproducible training samples from georeferenced surveys and observations."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from glaciertrack.config import load_yaml, validate_project_config
from glaciertrack.data.contracts import NegativeType, ObservationStatus


class PreparationError(RuntimeError):
    """Raised when source data cannot be converted without ambiguous assumptions."""


@dataclass(frozen=True)
class PreparationSettings:
    """Validated settings used by the deterministic preparation pipeline."""

    tensor_gsd_m_per_px: float
    reference_size_px: tuple[int, int]
    search_size_px: tuple[int, int]
    reference_margin_px: int
    search_jitter_m: tuple[float, float]
    samples_per_track_pair: int
    sample_composition: tuple[tuple[str, int], ...] | None
    negative_exclusion_margin_m: float
    distractor_same_spatial_group: bool
    sampling_attempts: int
    positive_target_margin_px: int
    minimum_valid_fraction: float
    bands: tuple[int, ...]
    pairing_strategy: str
    allow_reverse: bool
    minimum_interval_days: int
    maximum_interval_days: int | None
    explicit_pairs: tuple[dict[str, str], ...]
    require_spatial_group: bool
    survey_metadata_filename: str
    observation_stem: str
    aoi_stem: str
    require_aoi: bool
    compute_checksums: bool
    compression: str | None
    seed: int
    glaciers: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> PreparationSettings:
        data = _mapping(payload, "data")
        sampling = _mapping(payload, "sampling")
        preparation = payload.get("preparation", {})
        pairing = payload.get("pairing", {})
        if not isinstance(preparation, dict) or not isinstance(pairing, dict):
            raise PreparationError("preparation and pairing configuration must be mappings")
        selected = preparation.get("glaciers", [])
        if not isinstance(selected, list) or any(not isinstance(v, str) for v in selected):
            raise PreparationError("preparation.glaciers must be a list of glacier IDs")

        tensor_gsd = _positive_float(data.get("tensor_gsd_m_per_px"), "tensor GSD")
        reference_size = _pixel_size(
            sampling.get("reference_crop_size_px", [256, 256]),
            "reference_crop_size_px",
        )
        search_size = _pixel_size(
            sampling.get("search_crop_size_px", [512, 512]),
            "search_crop_size_px",
        )
        margin = int(sampling.get("reference_margin_px", 16))
        if margin < 0 or 2 * margin >= min(reference_size):
            raise PreparationError("reference_margin_px must leave a non-empty safe region")

        jitter = sampling.get("search_center_jitter_m", 0.0)
        if isinstance(jitter, (int, float)):
            jitter_xy = (float(jitter), float(jitter))
        elif isinstance(jitter, (list, tuple)) and len(jitter) == 2:
            jitter_xy = (float(jitter[0]), float(jitter[1]))
        else:
            raise PreparationError("search_center_jitter_m must be a number or [x, y]")
        if min(jitter_xy) < 0:
            raise PreparationError("search_center_jitter_m cannot be negative")

        samples_per_pair = int(sampling.get("samples_per_track_pair", 1))
        if samples_per_pair <= 0:
            raise PreparationError("samples_per_track_pair must be positive")
        composition_payload = sampling.get("sample_composition")
        sample_composition: tuple[tuple[str, int], ...] | None = None
        if composition_payload is not None:
            if not isinstance(composition_payload, dict):
                raise PreparationError("sampling.sample_composition must be a mapping")
            allowed_intents = {
                "positive",
                NegativeType.TARGET_OUTSIDE_WINDOW.value,
                NegativeType.TARGET_UNOBSERVABLE.value,
                NegativeType.DISTRACTOR.value,
            }
            unknown = set(composition_payload).difference(allowed_intents)
            if unknown:
                raise PreparationError(
                    f"unknown sample_composition entries: {sorted(unknown)}"
                )
            counts = tuple(
                (str(name), int(value)) for name, value in composition_payload.items()
            )
            if any(value < 0 for _, value in counts) or sum(value for _, value in counts) <= 0:
                raise PreparationError(
                    "sample_composition counts must be non-negative and non-zero"
                )
            sample_composition = counts
            samples_per_pair = sum(value for _, value in counts)
        exclusion_margin = float(sampling.get("negative_exclusion_margin_m", 2.0))
        if exclusion_margin < 0:
            raise PreparationError("negative_exclusion_margin_m cannot be negative")
        sampling_attempts = int(sampling.get("sampling_attempts", 64))
        if sampling_attempts <= 0:
            raise PreparationError("sampling_attempts must be positive")
        positive_margin = int(sampling.get("positive_target_margin_px", margin))
        if positive_margin < 0 or 2 * positive_margin >= min(search_size):
            raise PreparationError("positive_target_margin_px must leave a non-empty search area")
        valid_fraction = float(sampling.get("minimum_valid_fraction", 0.95))
        if not 0 <= valid_fraction <= 1:
            raise PreparationError("minimum_valid_fraction must be between zero and one")

        bands = tuple(int(value) for value in preparation.get("bands", [1, 2, 3]))
        if not bands or min(bands) < 1 or len(set(bands)) != len(bands):
            raise PreparationError("preparation.bands must contain unique one-based indexes")

        strategy = str(pairing.get("strategy", "all_chronological"))
        if strategy not in {"all_chronological", "adjacent", "explicit"}:
            raise PreparationError(
                "pairing.strategy must be all_chronological, adjacent, or explicit"
            )
        explicit = pairing.get("explicit_pairs", [])
        if not isinstance(explicit, list) or any(not isinstance(item, dict) for item in explicit):
            raise PreparationError("pairing.explicit_pairs must be a list of mappings")

        maximum = pairing.get("maximum_interval_days")
        maximum_days = None if maximum is None else int(maximum)
        minimum_days = int(pairing.get("minimum_interval_days", 0))
        if minimum_days < 0 or (maximum_days is not None and maximum_days < minimum_days):
            raise PreparationError("invalid pairing interval limits")

        compression_value = preparation.get("hdf5_compression", "gzip")
        compression = None if compression_value in {None, "none"} else str(compression_value)
        if compression not in {None, "gzip", "lzf"}:
            raise PreparationError("hdf5_compression must be gzip, lzf, or none")

        project = payload.get("project", {})
        seed = int(project.get("seed", 42)) if isinstance(project, dict) else 42
        return cls(
            tensor_gsd_m_per_px=tensor_gsd,
            reference_size_px=reference_size,
            search_size_px=search_size,
            reference_margin_px=margin,
            search_jitter_m=jitter_xy,
            samples_per_track_pair=samples_per_pair,
            sample_composition=sample_composition,
            negative_exclusion_margin_m=exclusion_margin,
            distractor_same_spatial_group=bool(
                sampling.get("distractor_same_spatial_group", True)
            ),
            sampling_attempts=sampling_attempts,
            positive_target_margin_px=positive_margin,
            minimum_valid_fraction=valid_fraction,
            bands=bands,
            pairing_strategy=strategy,
            allow_reverse=bool(pairing.get("allow_reverse", False)),
            minimum_interval_days=minimum_days,
            maximum_interval_days=maximum_days,
            explicit_pairs=tuple(explicit),
            require_spatial_group=bool(sampling.get("require_spatial_group", True)),
            survey_metadata_filename=str(
                preparation.get("survey_metadata_filename", "survey.yaml")
            ),
            observation_stem=str(preparation.get("observation_stem", "observations")),
            aoi_stem=str(preparation.get("aoi_stem", "aoi")),
            require_aoi=bool(preparation.get("require_aoi", False)),
            compute_checksums=bool(preparation.get("compute_checksums", True)),
            compression=compression,
            seed=seed,
            glaciers=tuple(selected),
        )


@dataclass(frozen=True)
class _SurveyRecord:
    glacier_id: str
    survey_id: str
    acquisition_date: date | None
    raster_uri: str
    relative_survey_dir: str
    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    count: int
    dtype: str
    nodata: float | None
    native_gsd_x_m: float
    native_gsd_y_m: float
    registration_rmse_m: float | None
    checksum: str
    radiometric_min: float
    radiometric_max: float
    aoi_uri: str | None
    aoi_crs: str | None
    aoi_checksum: str
    aoi_area_m2: float | None
    aoi_bounds: tuple[float, float, float, float] | None
    acquisition_year: int = 0

    def as_manifest_row(self) -> dict[str, Any]:
        return {
            "glacier_id": self.glacier_id,
            "survey_id": self.survey_id,
            "acquisition_date": self.acquisition_date,
            "acquisition_year": self.acquisition_year,
            "acquisition_date_precision": "day" if self.acquisition_date else "year",
            "raster_uri": self.raster_uri,
            "crs": self.crs,
            "affine_transform": list(self.transform),
            "width": self.width,
            "height": self.height,
            "band_count": self.count,
            "dtype": self.dtype,
            "nodata": self.nodata,
            "native_gsd_x_m": self.native_gsd_x_m,
            "native_gsd_y_m": self.native_gsd_y_m,
            "registration_rmse_m": self.registration_rmse_m,
            "checksum": self.checksum,
            "radiometric_min": self.radiometric_min,
            "radiometric_max": self.radiometric_max,
            "aoi_uri": self.aoi_uri,
            "aoi_crs": self.aoi_crs,
            "aoi_checksum": self.aoi_checksum,
            "aoi_area_m2": self.aoi_area_m2,
            "aoi_bounds": None if self.aoi_bounds is None else list(self.aoi_bounds),
        }


def prepare_dataset(
    raw_root: str | Path,
    *,
    config: str | Path | dict[str, Any],
    tracks_root: str | Path | None = None,
    output_root: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Discover source data and create manifests plus an HDF5 training cache.

    Annotation files mirror the raw ``glacier/surveys/survey`` hierarchy. Missing
    target observations are excluded as unknown; they are never converted to negatives.
    """

    pandas, rasterio = _require_geo_dependencies()
    payload = load_yaml(config) if not isinstance(config, dict) else config
    validate_project_config(payload)
    settings = PreparationSettings.from_mapping(payload)

    raw_path = Path(raw_root).expanduser().resolve()
    if not raw_path.is_dir():
        raise PreparationError(f"raw root does not exist or is not a directory: {raw_path}")
    tracks_path = (
        Path(tracks_root).expanduser().resolve()
        if tracks_root is not None
        else raw_path.parent.joinpath("tracks").resolve()
    )
    if not tracks_path.is_dir():
        raise PreparationError(f"tracks root does not exist or is not a directory: {tracks_path}")
    prepared_path = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else raw_path.parent.joinpath("prepared").resolve()
    )

    warnings: list[str] = []
    surveys = _discover_surveys(raw_path, settings, rasterio, warnings)
    survey_rows = [survey.as_manifest_row() for survey in surveys]
    surveys_frame = pandas.DataFrame(survey_rows)
    observations_frame = _load_all_observations(
        surveys,
        tracks_path,
        settings,
        pandas,
        rasterio,
        warnings,
    )
    observations_frame = observations_frame.sort_values(
        ["glacier_id", "survey_id", "track_id"],
        kind="stable",
    ).reset_index(drop=True)
    observations_frame = _harmonize_spatial_groups(
        observations_frame,
        settings.require_spatial_group,
    )
    pairs_frame = _build_survey_pairs(surveys_frame, settings, pandas)
    if pairs_frame.empty:
        raise PreparationError("pairing rules produced no survey pairs")

    paths = _output_paths(prepared_path)
    _check_output_paths(paths, overwrite)
    for directory in {path.parent for path in paths.values()}:
        directory.mkdir(parents=True, exist_ok=True)

    temporary = {name: _temporary_path(path) for name, path in paths.items()}
    _check_temporary_paths(temporary, overwrite)
    writer: _HDF5CacheWriter | None = None
    sample_rows: list[dict[str, Any]] = []
    counters = {
        "skipped_unknown_target": 0,
        "skipped_invalid_coverage": 0,
        "positive": 0,
        "target_outside_window": 0,
        "target_unobservable": 0,
        "distractor": 0,
        "source_invalid": 0,
    }

    try:
        writer = _HDF5CacheWriter(
            temporary["cache"],
            channels=len(settings.bands),
            reference_size=settings.reference_size_px,
            search_size=settings.search_size_px,
            tensor_gsd=settings.tensor_gsd_m_per_px,
            compression=settings.compression,
        )
        sample_rows = _extract_samples(
            surveys,
            observations_frame,
            pairs_frame,
            settings,
            writer,
            rasterio,
            counters,
        )
        writer.close()
        writer = None
        if not sample_rows:
            raise PreparationError("no valid samples were produced")

        samples_frame = pandas.DataFrame(sample_rows)
        surveys_frame.to_parquet(temporary["surveys"], index=False)
        observations_frame.to_parquet(temporary["observations"], index=False)
        pairs_frame.to_parquet(temporary["pairs"], index=False)
        samples_frame.to_parquet(temporary["samples"], index=False)
        manifest_hash = _sha256(temporary["samples"])
        with h5py.File(temporary["cache"], "r+") as handle:
            handle.attrs["manifest_sha256"] = manifest_hash

        summary = {
            "raw_root": str(raw_path),
            "tracks_root": str(tracks_path),
            "output_root": str(prepared_path),
            "glaciers": int(surveys_frame["glacier_id"].nunique()),
            "surveys": len(surveys_frame),
            "observations": len(observations_frame),
            "survey_pairs": len(pairs_frame),
            "samples": len(samples_frame),
            "sample_counts": counters,
            "sampling_intent_counts": {
                str(name): int(count)
                for name, count in samples_frame["sampling_intent"].value_counts().items()
            },
            "distractor_fallbacks": int(
                (
                    (samples_frame["sampling_intent"] == NegativeType.DISTRACTOR.value)
                    & (
                        samples_frame["negative_type"]
                        == NegativeType.TARGET_OUTSIDE_WINDOW.value
                    )
                ).sum()
            ),
            "manifest_sha256": manifest_hash,
            "warnings": warnings,
        }
        temporary["report"].write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for name in ("surveys", "observations", "pairs", "samples", "cache", "report"):
            os.replace(temporary[name], paths[name])
        return summary
    except Exception:
        if writer is not None:
            writer.close()
        for path in temporary.values():
            if path.exists():
                path.unlink()
        raise


def _require_geo_dependencies() -> tuple[Any, Any]:
    try:
        import pandas
        import pyarrow  # noqa: F401
        import rasterio
        import rasterio.enums  # noqa: F401
        import rasterio.warp  # noqa: F401
        import rasterio.windows  # noqa: F401
    except ImportError as error:
        raise PreparationError(
            "preparation requires the geo dependencies; install with "
            "pip install 'glacier-boulder-tracker[geo]'"
        ) from error
    return pandas, rasterio


def _discover_surveys(
    raw_root: Path,
    settings: PreparationSettings,
    rasterio: Any,
    warnings: list[str],
) -> list[_SurveyRecord]:
    records: list[_SurveyRecord] = []
    glacier_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir())
    if settings.glaciers:
        missing = set(settings.glaciers) - {path.name for path in glacier_dirs}
        if missing:
            raise PreparationError(f"requested glaciers not found: {sorted(missing)}")
        glacier_dirs = [path for path in glacier_dirs if path.name in settings.glaciers]
    if not glacier_dirs:
        raise PreparationError(f"no glacier directories found under {raw_root}")

    for glacier_dir in glacier_dirs:
        survey_root = glacier_dir / "surveys"
        if not survey_root.is_dir():
            survey_root = glacier_dir
        survey_dirs = sorted(path for path in survey_root.iterdir() if path.is_dir())
        for survey_dir in survey_dirs:
            metadata_path = survey_dir / settings.survey_metadata_filename
            if not metadata_path.is_file():
                raise PreparationError(f"missing survey metadata: {metadata_path}")
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise PreparationError(f"survey metadata must be a mapping: {metadata_path}")
            glacier_id = str(metadata.get("glacier_id", glacier_dir.name))
            survey_id = str(metadata.get("survey_id", survey_dir.name))
            if glacier_id != glacier_dir.name or survey_id != survey_dir.name:
                raise PreparationError(
                    f"folder and metadata identifiers disagree in {metadata_path}"
                )
            acquisition_date, acquisition_year = _survey_date(metadata, metadata_path)
            if acquisition_date is None:
                warnings.append(f"acquisition_date is year-only for {glacier_id}/{survey_id}")
            raster_path = _resolve_raster(survey_dir, metadata)
            checksum = _sha256(raster_path) if settings.compute_checksums else ""

            with rasterio.open(raster_path) as dataset:
                if dataset.crs is None:
                    raise PreparationError(f"raster has no CRS: {raster_path}")
                if not dataset.crs.is_projected:
                    raise PreparationError(
                        f"raster CRS must be projected in metres: {raster_path} ({dataset.crs})"
                    )
                linear_units = str(getattr(dataset.crs, "linear_units", "")).lower()
                if linear_units not in {"m", "meter", "meters", "metre", "metres"}:
                    raise PreparationError(
                        f"raster CRS linear unit must be metres: {raster_path} "
                        f"({linear_units or 'unknown'})"
                    )
                if max(settings.bands) > dataset.count:
                    raise PreparationError(
                        f"requested bands {settings.bands} exceed band count in {raster_path}"
                    )
                gsd_x = math.hypot(dataset.transform.a, dataset.transform.d)
                gsd_y = math.hypot(dataset.transform.b, dataset.transform.e)
                if gsd_x <= 0 or gsd_y <= 0:
                    raise PreparationError(f"invalid affine transform in {raster_path}")
                dtype = dataset.dtypes[settings.bands[0] - 1]
                selected_dtypes = {dataset.dtypes[index - 1] for index in settings.bands}
                if len(selected_dtypes) != 1:
                    raise PreparationError(
                        f"selected raster bands have mixed dtypes: {raster_path}"
                    )
                radiometric_min, radiometric_max = _radiometric_range(dtype, metadata)
                aoi_path = _resolve_aoi(survey_dir, glacier_dir, metadata, settings)
                if aoi_path is None:
                    if settings.require_aoi:
                        raise PreparationError(
                            f"no AOI polygon found for {glacier_id}/{survey_id}"
                        )
                    warnings.append(
                        f"AOI is unknown for {glacier_id}/{survey_id}; full-glacier "
                        "inference will be unavailable"
                    )
                    aoi_crs = None
                    aoi_area = None
                    aoi_bounds = None
                    aoi_checksum = ""
                else:
                    aoi_crs, aoi_area, aoi_bounds = _inspect_aoi(
                        aoi_path,
                        str(dataset.crs),
                        tuple(dataset.bounds),
                    )
                    aoi_checksum = _sha256(aoi_path) if settings.compute_checksums else ""
                transform = dataset.transform
                registration = metadata.get("registration_rmse_m")
                if registration is None:
                    warnings.append(f"registration_rmse_m is unknown for {glacier_id}/{survey_id}")
                else:
                    registration = float(registration)
                    if registration < 0:
                        raise PreparationError(
                            f"registration_rmse_m cannot be negative in {metadata_path}"
                        )
                records.append(
                    _SurveyRecord(
                        glacier_id=glacier_id,
                        survey_id=survey_id,
                        acquisition_date=acquisition_date,
                        acquisition_year=acquisition_year,
                        raster_uri=str(raster_path),
                        relative_survey_dir=str(survey_dir.relative_to(raw_root)),
                        crs=str(dataset.crs),
                        transform=(
                            transform.a,
                            transform.b,
                            transform.c,
                            transform.d,
                            transform.e,
                            transform.f,
                        ),
                        width=dataset.width,
                        height=dataset.height,
                        count=dataset.count,
                        dtype=dtype,
                        nodata=dataset.nodata,
                        native_gsd_x_m=gsd_x,
                        native_gsd_y_m=gsd_y,
                        registration_rmse_m=registration,
                        checksum=checksum,
                        radiometric_min=radiometric_min,
                        radiometric_max=radiometric_max,
                        aoi_uri=None if aoi_path is None else str(aoi_path),
                        aoi_crs=aoi_crs,
                        aoi_checksum=aoi_checksum,
                        aoi_area_m2=aoi_area,
                        aoi_bounds=aoi_bounds,
                    )
                )
    if not records:
        raise PreparationError(f"no surveys found under {raw_root}")
    keys = [(item.glacier_id, item.survey_id) for item in records]
    if len(keys) != len(set(keys)):
        raise PreparationError("duplicate glacier_id/survey_id combinations were discovered")
    return records


def _load_all_observations(
    surveys: list[_SurveyRecord],
    tracks_root: Path,
    settings: PreparationSettings,
    pandas: Any,
    rasterio: Any,
    warnings: list[str],
) -> Any:
    frames: list[Any] = []
    for survey in surveys:
        annotation_dir = tracks_root / survey.relative_survey_dir
        candidates = sorted(annotation_dir.glob(f"{settings.observation_stem}.*"))
        candidates = [
            path
            for path in candidates
            if path.suffix.lower() in {".csv", ".parquet", ".gpkg", ".geojson"}
        ]
        if len(candidates) != 1:
            raise PreparationError(
                f"expected exactly one {settings.observation_stem} annotation file in "
                f"{annotation_dir}, found {len(candidates)}"
            )
        frames.append(
            _load_observation_file(candidates[0], survey, pandas, rasterio, warnings)
        )
    records = [row for item in frames for row in item.to_dict("records")]
    frame = pandas.DataFrame.from_records(records)
    duplicates = frame.duplicated(["glacier_id", "survey_id", "track_id"], keep=False)
    if duplicates.any():
        rows = frame.loc[duplicates, ["glacier_id", "survey_id", "track_id"]]
        raise PreparationError(f"duplicate track observations found: {rows.to_dict('records')[:5]}")
    return frame


def _load_observation_file(
    path: Path,
    survey: _SurveyRecord,
    pandas: Any,
    rasterio: Any,
    warnings: list[str],
) -> Any:
    suffix = path.suffix.lower()
    vector_crs: str | None = None
    if suffix == ".csv":
        frame = pandas.read_csv(path)
    elif suffix == ".parquet":
        frame = pandas.read_parquet(path)
        if not {"x_m", "y_m"}.issubset(frame.columns) and "geometry" in frame.columns:
            geopandas = _require_geopandas()
            frame = geopandas.read_parquet(path)
            vector_crs = None if frame.crs is None else str(frame.crs)
    else:
        geopandas = _require_geopandas()
        frame = geopandas.read_file(path)
        vector_crs = None if frame.crs is None else str(frame.crs)

    required = {"track_id", "status"}
    missing = required.difference(frame.columns)
    if missing:
        raise PreparationError(f"missing columns {sorted(missing)} in {path}")
    if "glacier_id" in frame and not frame["glacier_id"].dropna().astype(str).eq(
        survey.glacier_id
    ).all():
        raise PreparationError(f"glacier_id values disagree with folder path in {path}")
    if "survey_id" in frame and not frame["survey_id"].dropna().astype(str).eq(
        survey.survey_id
    ).all():
        raise PreparationError(f"survey_id values disagree with folder path in {path}")

    if "geometry" in frame.columns and not {"x_m", "y_m"}.issubset(frame.columns):
        if vector_crs is None:
            raise PreparationError(f"geospatial annotations have no CRS: {path}")
        if vector_crs != survey.crs:
            frame = frame.to_crs(survey.crs)
        frame = frame.copy()
        frame["x_m"] = frame.geometry.x
        frame["y_m"] = frame.geometry.y
    elif not {"x_m", "y_m"}.issubset(frame.columns):
        raise PreparationError(f"annotations require x_m/y_m coordinates or point geometry: {path}")
    else:
        source_crs = _tabular_crs(frame, survey.crs, path, warnings)
        if source_crs != survey.crs:
            x_values = frame["x_m"].to_numpy(dtype=float, na_value=np.nan)
            y_values = frame["y_m"].to_numpy(dtype=float, na_value=np.nan)
            valid = np.isfinite(x_values) & np.isfinite(y_values)
            if valid.any():
                transformed_x, transformed_y = rasterio.warp.transform(
                    source_crs,
                    survey.crs,
                    x_values[valid].tolist(),
                    y_values[valid].tolist(),
                )
                x_values[valid] = transformed_x
                y_values[valid] = transformed_y
            frame = frame.copy()
            frame["x_m"] = x_values
            frame["y_m"] = y_values

    rows: list[dict[str, Any]] = []
    valid_statuses = {item.value for item in ObservationStatus}
    for index, row in frame.iterrows():
        track_id = str(row["track_id"]).strip()
        status = str(row["status"]).strip().lower()
        if not track_id or track_id.lower() == "nan":
            raise PreparationError(f"empty track_id at row {index} in {path}")
        if status not in valid_statuses:
            raise PreparationError(f"invalid status {status!r} at row {index} in {path}")
        x_value = _optional_float(row.get("x_m"))
        y_value = _optional_float(row.get("y_m"))
        point_statuses = {
            ObservationStatus.VISIBLE.value,
            ObservationStatus.INVALID_PROMPT.value,
        }
        if status in point_statuses and (x_value is None or y_value is None):
            raise PreparationError(
                f"status {status!r} requires a point at row {index} in {path}"
            )
        uncertainty = _optional_float(row.get("annotation_uncertainty_m"))
        if uncertainty is not None and uncertainty < 0:
            raise PreparationError(f"negative annotation uncertainty at row {index} in {path}")
        major_axis = _optional_float(row.get("major_axis_m"))
        minor_axis = _optional_float(row.get("minor_axis_m"))
        orientation = _optional_float(row.get("orientation_deg"))
        if (major_axis is None) != (minor_axis is None):
            raise PreparationError(
                f"major_axis_m and minor_axis_m must be supplied together at row {index} in {path}"
            )
        if major_axis is not None and (major_axis <= 0 or minor_axis <= 0):
            raise PreparationError(f"ellipse axes must be positive at row {index} in {path}")
        rows.append(
            {
                "glacier_id": survey.glacier_id,
                "survey_id": survey.survey_id,
                "track_id": track_id,
                "status": status,
                "x_m": x_value,
                "y_m": y_value,
                "crs": survey.crs,
                "spatial_group": _optional_string(row.get("spatial_group")),
                "annotation_uncertainty_m": uncertainty,
                "major_axis_m": major_axis,
                "minor_axis_m": minor_axis,
                "orientation_deg": orientation,
                "quality_flag": _optional_string(row.get("quality_flag")),
                "annotator": _optional_string(row.get("annotator")),
                "annotation_version": _optional_string(row.get("annotation_version")),
                "source_uri": str(path),
            }
        )
    return pandas.DataFrame(rows)


def _harmonize_spatial_groups(frame: Any, required: bool) -> Any:
    frame = frame.copy()
    for (glacier_id, track_id), indexes in frame.groupby(["glacier_id", "track_id"]).groups.items():
        values = {
            str(value)
            for value in frame.loc[indexes, "spatial_group"].dropna().tolist()
            if str(value)
        }
        if len(values) > 1:
            raise PreparationError(
                f"track {glacier_id}/{track_id} has conflicting spatial groups: {sorted(values)}"
            )
        if values:
            frame.loc[indexes, "spatial_group"] = next(iter(values))
        elif required:
            statuses = set(frame.loc[indexes, "status"])
            if statuses != {ObservationStatus.INVALID_PROMPT.value}:
                raise PreparationError(
                    f"track {glacier_id}/{track_id} is missing required spatial_group"
                )
            frame.loc[indexes, "spatial_group"] = f"invalid:{track_id}"
        else:
            frame.loc[indexes, "spatial_group"] = f"track:{track_id}"
    return frame


def _build_survey_pairs(frame: Any, settings: PreparationSettings, pandas: Any) -> Any:
    rows: list[dict[str, Any]] = []
    known = {
        (str(row.glacier_id), str(row.survey_id)): row
        for row in frame.itertuples(index=False)
    }
    if settings.pairing_strategy == "explicit":
        for item in settings.explicit_pairs:
            try:
                glacier_id = str(item["glacier_id"])
                reference_id = str(item["reference_survey_id"])
                target_id = str(item["target_survey_id"])
            except KeyError as error:
                raise PreparationError(f"incomplete explicit survey pair: {item}") from error
            if (glacier_id, reference_id) not in known or (glacier_id, target_id) not in known:
                raise PreparationError(f"explicit survey pair references an unknown survey: {item}")
            rows.extend(
                _pair_rows(
                    glacier_id,
                    known[(glacier_id, reference_id)],
                    known[(glacier_id, target_id)],
                    settings,
                )
            )
    else:
        for glacier_id, group in frame.groupby("glacier_id", sort=True):
            ordered = sorted(group.itertuples(), key=lambda row: _date_bounds(row)[0])
            indexes = (
                [(index, index + 1) for index in range(len(ordered) - 1)]
                if settings.pairing_strategy == "adjacent"
                else [
                    (left, right)
                    for left in range(len(ordered))
                    for right in range(left + 1, len(ordered))
                ]
            )
            for left, right in indexes:
                rows.extend(_pair_rows(str(glacier_id), ordered[left], ordered[right], settings))
    return pandas.DataFrame(rows)


def _survey_date(metadata: dict, path: Path) -> tuple[date | None, int]:
    value = metadata.get("acquisition_date")
    if value is not None:
        parsed = _parse_date(value, path)
        if metadata.get("acquisition_year", parsed.year) != parsed.year:
            raise PreparationError(f"acquisition year and date disagree in {path}")
        return parsed, parsed.year
    year = metadata.get("acquisition_year")
    if isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= 9999:
        raise PreparationError(
            f"acquisition_date or integer acquisition_year is required in {path}"
        )
    return None, year


def _date_bounds(row: Any) -> tuple[date, date]:
    if row.acquisition_date is not None:
        return row.acquisition_date, row.acquisition_date
    return date(int(row.acquisition_year), 1, 1), date(int(row.acquisition_year), 12, 31)


def _pair_rows(
    glacier_id: str,
    reference: Any,
    target: Any,
    settings: PreparationSettings,
) -> list[dict[str, Any]]:
    ref_min, ref_max = _date_bounds(reference)
    dst_min, dst_max = _date_bounds(target)
    known_dates = reference.acquisition_date is not None and target.acquisition_date is not None
    interval = (dst_min - ref_min).days if known_dates else None
    if not known_dates and not (dst_min > ref_max or ref_min > dst_max):
        raise PreparationError("year-only survey dates cannot establish within-year ordering")
    reverse = dst_max < ref_min
    if reverse and not settings.allow_reverse:
        raise PreparationError("explicit reverse survey pairs require pairing.allow_reverse=true")
    if interval is None:
        if settings.minimum_interval_days or settings.maximum_interval_days is not None:
            raise PreparationError("day-based interval filters require exact acquisition dates")
        result = [_pair_row(glacier_id, reference, target, None)]
        if settings.allow_reverse and not reverse:
            result.append(_pair_row(glacier_id, target, reference, None))
        return result
    absolute_interval = abs(interval)
    if absolute_interval < settings.minimum_interval_days:
        return []
    if (
        settings.maximum_interval_days is not None
        and absolute_interval > settings.maximum_interval_days
    ):
        return []
    result = [_pair_row(glacier_id, reference, target, interval)]
    if settings.allow_reverse and interval > 0:
        result.append(_pair_row(glacier_id, target, reference, -interval))
    return result


def _pair_row(glacier_id: str, reference: Any, target: Any, interval: int | None) -> dict[str, Any]:
    reference_id = str(reference.survey_id)
    target_id = str(target.survey_id)
    return {
        "pair_id": f"{glacier_id}:{reference_id}--{target_id}",
        "glacier_id": glacier_id,
        "reference_survey_id": reference_id,
        "target_survey_id": target_id,
        "reference_date": reference.acquisition_date,
        "target_date": target.acquisition_date,
        "interval_days": interval,
    }


def _extract_samples(
    surveys: list[_SurveyRecord],
    observations: Any,
    pairs: Any,
    settings: PreparationSettings,
    writer: _HDF5CacheWriter,
    rasterio: Any,
    counters: dict[str, int],
) -> list[dict[str, Any]]:
    survey_lookup = {(item.glacier_id, item.survey_id): item for item in surveys}
    observation_groups = {
        key: group.reset_index(drop=True)
        for key, group in observations.groupby(["glacier_id", "survey_id"], sort=False)
    }
    rng = np.random.default_rng(settings.seed)
    rows: list[dict[str, Any]] = []

    for pair in pairs.itertuples(index=False):
        key_reference = (str(pair.glacier_id), str(pair.reference_survey_id))
        key_target = (str(pair.glacier_id), str(pair.target_survey_id))
        reference_survey = survey_lookup[key_reference]
        target_survey = survey_lookup[key_target]
        empty_observations = observations.iloc[0:0]
        reference_observations = observation_groups.get(key_reference, empty_observations)
        target_observations = observation_groups.get(key_target, empty_observations)
        target_by_track = {
            str(row.track_id): row for row in target_observations.itertuples(index=False)
        }

        with (
            rasterio.open(reference_survey.raster_uri) as reference_dataset,
            rasterio.open(target_survey.raster_uri) as target_dataset,
        ):
            for source in reference_observations.itertuples(index=False):
                if source.status not in {
                    ObservationStatus.VISIBLE.value,
                    ObservationStatus.INVALID_PROMPT.value,
                }:
                    continue
                source_valid = source.status == ObservationStatus.VISIBLE.value
                target = target_by_track.get(str(source.track_id)) if source_valid else None
                if source_valid and (
                    target is None or target.status == ObservationStatus.UNKNOWN.value
                ):
                    counters["skipped_unknown_target"] += 1
                    continue
                intents = _sampling_intents(source, target, settings)
                for repetition, sampling_intent in enumerate(intents):
                    extracted = _extract_one_sample(
                        pair,
                        source,
                        target,
                        target_observations,
                        reference_survey,
                        target_survey,
                        reference_dataset,
                        target_dataset,
                        settings,
                        repetition,
                        sampling_intent,
                        rng,
                        rasterio,
                    )
                    if extracted is None:
                        counters["skipped_invalid_coverage"] += 1
                        continue
                    manifest_row, reference_image, search_image = extracted
                    writer.append(manifest_row, reference_image, search_image)
                    rows.append(manifest_row)
                    if not manifest_row["source_valid"]:
                        counters["source_invalid"] += 1
                    elif manifest_row["match_present"]:
                        counters["positive"] += 1
                    else:
                        counters[str(manifest_row["negative_type"])] += 1
    return rows


def _sampling_intents(
    source: Any,
    target: Any | None,
    settings: PreparationSettings,
) -> tuple[str | None, ...]:
    """Expand an exact per-track composition while retaining legacy sampling."""

    if settings.sample_composition is None:
        return (None,) * settings.samples_per_track_pair
    if source.status != ObservationStatus.VISIBLE.value:
        return (None,)
    counts = dict(settings.sample_composition)
    if target is not None and target.status == ObservationStatus.VISIBLE.value:
        intents: list[str] = []
        for name in (
            "positive",
            NegativeType.TARGET_OUTSIDE_WINDOW.value,
            NegativeType.DISTRACTOR.value,
        ):
            intents.extend([name] * counts.get(name, 0))
        return tuple(intents)
    if target is not None and target.status in {
        ObservationStatus.UNOBSERVABLE.value,
        ObservationStatus.OUTSIDE_COVERAGE.value,
    }:
        count = max(1, counts.get(NegativeType.TARGET_UNOBSERVABLE.value, 0))
        return (NegativeType.TARGET_UNOBSERVABLE.value,) * count
    return ()


def _planned_search_window(
    sampling_intent: str | None,
    source: Any,
    target: Any | None,
    target_observations: Any,
    target_dataset: Any,
    target_survey: _SurveyRecord,
    source_target_x: float,
    source_target_y: float,
    settings: PreparationSettings,
    rng: np.random.Generator,
    rasterio: Any,
) -> tuple[Any, float, float, str, str | None, str | None] | None:
    """Place a target crop for a requested class without ambiguous labels."""

    search_height, search_width = settings.search_size_px
    if sampling_intent is None or sampling_intent == NegativeType.TARGET_UNOBSERVABLE.value:
        jitter_x = float(
            rng.uniform(-settings.search_jitter_m[0], settings.search_jitter_m[0])
        )
        jitter_y = float(
            rng.uniform(-settings.search_jitter_m[1], settings.search_jitter_m[1])
        )
        window = _centered_window(
            target_dataset,
            source_target_x,
            source_target_y,
            jitter_x,
            jitter_y,
            search_width,
            search_height,
            settings.tensor_gsd_m_per_px,
            target_survey,
            rasterio,
        )
        outcome = (
            NegativeType.TARGET_UNOBSERVABLE.value
            if sampling_intent == NegativeType.TARGET_UNOBSERVABLE.value
            else None
        )
        return window, jitter_x, jitter_y, "source_coordinate_jitter", outcome, None

    if target is None or target.status != ObservationStatus.VISIBLE.value:
        return None
    target_x = float(target.x_m)
    target_y = float(target.y_m)
    if sampling_intent == "positive":
        margin = settings.positive_target_margin_px
        desired_x = float(rng.uniform(margin, search_width - margin))
        desired_y = float(rng.uniform(margin, search_height - margin))
        window = _point_positioned_window(
            target_dataset,
            target_x,
            target_y,
            desired_x,
            desired_y,
            search_width,
            search_height,
            settings.tensor_gsd_m_per_px,
            target_survey,
            rasterio,
        )
        return window, math.nan, math.nan, "controlled_uniform_positive", "positive", None

    if sampling_intent == NegativeType.DISTRACTOR.value:
        distractor = _distractor_window(
            source,
            target,
            target_observations,
            target_dataset,
            target_survey,
            settings,
            rng,
            rasterio,
        )
        if distractor is not None:
            window, distractor_track_id = distractor
            return (
                window,
                math.nan,
                math.nan,
                "controlled_distractor",
                NegativeType.DISTRACTOR.value,
                distractor_track_id,
            )
        # A target-excluded crop is still a certain negative for this query.
        # Record the actual outcome so unavailable distractors are never hidden.
        sampling_intent = NegativeType.TARGET_OUTSIDE_WINDOW.value

    if sampling_intent == NegativeType.TARGET_OUTSIDE_WINDOW.value:
        window = _target_excluded_window(
            target_dataset,
            target_survey,
            target_x,
            target_y,
            settings,
            rng,
            rasterio,
        )
        return (
            window,
            math.nan,
            math.nan,
            "controlled_target_excluded",
            NegativeType.TARGET_OUTSIDE_WINDOW.value,
            None,
        )
    raise PreparationError(f"unsupported sampling intent {sampling_intent!r}")


def _target_excluded_window(
    dataset: Any,
    survey: _SurveyRecord,
    target_x_m: float,
    target_y_m: float,
    settings: PreparationSettings,
    rng: np.random.Generator,
    rasterio: Any,
) -> Any:
    height, width = settings.search_size_px
    exclusion_px = settings.negative_exclusion_margin_m / settings.tensor_gsd_m_per_px
    distance = exclusion_px + float(rng.uniform(1.0, max(2.0, min(height, width) / 4)))
    safe = settings.positive_target_margin_px
    side = int(rng.integers(0, 4))
    if side == 0:
        desired_x, desired_y = -distance, float(rng.uniform(safe, height - safe))
    elif side == 1:
        desired_x, desired_y = width + distance, float(rng.uniform(safe, height - safe))
    elif side == 2:
        desired_x, desired_y = float(rng.uniform(safe, width - safe)), -distance
    else:
        desired_x, desired_y = float(rng.uniform(safe, width - safe)), height + distance
    return _point_positioned_window(
        dataset,
        target_x_m,
        target_y_m,
        desired_x,
        desired_y,
        width,
        height,
        settings.tensor_gsd_m_per_px,
        survey,
        rasterio,
    )


def _distractor_window(
    source: Any,
    target: Any,
    observations: Any,
    dataset: Any,
    survey: _SurveyRecord,
    settings: PreparationSettings,
    rng: np.random.Generator,
    rasterio: Any,
) -> tuple[Any, str] | None:
    candidates = observations[
        (observations["status"] == ObservationStatus.VISIBLE.value)
        & (observations["track_id"].astype(str) != str(source.track_id))
    ]
    if settings.distractor_same_spatial_group:
        candidates = candidates[
            candidates["spatial_group"].astype(str) == str(source.spatial_group)
        ]
    if candidates.empty:
        return None
    height, width = settings.search_size_px
    margin = settings.positive_target_margin_px
    exclusion_px = settings.negative_exclusion_margin_m / settings.tensor_gsd_m_per_px
    order = rng.permutation(len(candidates))
    for attempt in range(settings.sampling_attempts):
        candidate = candidates.iloc[int(order[attempt % len(order)])]
        desired_x = float(rng.uniform(margin, width - margin))
        desired_y = float(rng.uniform(margin, height - margin))
        window = _point_positioned_window(
            dataset,
            float(candidate.x_m),
            float(candidate.y_m),
            desired_x,
            desired_y,
            width,
            height,
            settings.tensor_gsd_m_per_px,
            survey,
            rasterio,
        )
        target_point = _point_in_output(
            dataset,
            window,
            float(target.x_m),
            float(target.y_m),
            width,
            height,
        )
        if _outside_with_margin(target_point, width, height, exclusion_px):
            return window, str(candidate.track_id)
    return None


def _outside_with_margin(
    point: tuple[float, float], width: int, height: int, margin: float
) -> bool:
    return (
        point[0] <= -margin
        or point[0] >= width + margin
        or point[1] <= -margin
        or point[1] >= height + margin
    )


def _extract_one_sample(
    pair: Any,
    source: Any,
    target: Any | None,
    target_observations: Any,
    reference_survey: _SurveyRecord,
    target_survey: _SurveyRecord,
    reference_dataset: Any,
    target_dataset: Any,
    settings: PreparationSettings,
    repetition: int,
    sampling_intent: str | None,
    rng: np.random.Generator,
    rasterio: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray] | None:
    reference_height, reference_width = settings.reference_size_px
    search_height, search_width = settings.search_size_px
    margin = settings.reference_margin_px
    reference_window = None
    reference_image = None
    reference_valid = 0.0
    for _ in range(settings.sampling_attempts):
        desired_x = float(rng.uniform(margin, reference_width - margin))
        desired_y = float(rng.uniform(margin, reference_height - margin))
        candidate_window = _point_positioned_window(
            reference_dataset,
            float(source.x_m),
            float(source.y_m),
            desired_x,
            desired_y,
            reference_width,
            reference_height,
            settings.tensor_gsd_m_per_px,
            reference_survey,
            rasterio,
        )
        candidate_image, candidate_valid = _read_crop(
            reference_dataset,
            candidate_window,
            settings.bands,
            settings.reference_size_px,
            reference_survey,
            rasterio,
        )
        if candidate_valid >= settings.minimum_valid_fraction:
            reference_window = candidate_window
            reference_image = candidate_image
            reference_valid = candidate_valid
            break
    if reference_window is None or reference_image is None:
        return None

    source_target_x, source_target_y = _transform_point(
        float(source.x_m),
        float(source.y_m),
        reference_survey.crs,
        target_survey.crs,
        rasterio,
    )
    accepted_search = None
    for _ in range(settings.sampling_attempts):
        planned = _planned_search_window(
            sampling_intent,
            source,
            target,
            target_observations,
            target_dataset,
            target_survey,
            source_target_x,
            source_target_y,
            settings,
            rng,
            rasterio,
        )
        if planned is None:
            return None
        search_window = planned[0]
        candidate_image, candidate_valid = _read_crop(
            target_dataset,
            search_window,
            settings.bands,
            settings.search_size_px,
            target_survey,
            rasterio,
        )
        if candidate_valid >= settings.minimum_valid_fraction:
            accepted_search = (*planned, candidate_image, candidate_valid)
            break
    if accepted_search is None:
        return None
    (
        search_window,
        jitter_x,
        jitter_y,
        search_placement,
        sampling_outcome,
        distractor_track_id,
        search_image,
        search_valid,
    ) = accepted_search

    source_point_px = _point_in_output(
        reference_dataset,
        reference_window,
        float(source.x_m),
        float(source.y_m),
        reference_width,
        reference_height,
    )
    source_valid = source.status == ObservationStatus.VISIBLE.value
    match_present = False
    negative_type: str | None = None
    target_point_px = (math.nan, math.nan)
    target_x_m: float | None = None
    target_y_m: float | None = None

    if sampling_intent is not None and source_valid and target is not None:
        if sampling_outcome == "positive":
            target_x_m = float(target.x_m)
            target_y_m = float(target.y_m)
            target_point_px = _point_in_output(
                target_dataset,
                search_window,
                target_x_m,
                target_y_m,
                search_width,
                search_height,
            )
            if not _inside(target_point_px, search_width, search_height):
                raise PreparationError("controlled positive sampling excluded its target")
            match_present = True
        elif sampling_outcome in {
            NegativeType.TARGET_OUTSIDE_WINDOW.value,
            NegativeType.DISTRACTOR.value,
        }:
            target_x_m = float(target.x_m)
            target_y_m = float(target.y_m)
            negative_type = sampling_outcome
        elif sampling_outcome == NegativeType.TARGET_UNOBSERVABLE.value:
            negative_type = NegativeType.TARGET_UNOBSERVABLE.value
        else:
            raise PreparationError(f"unsupported controlled sampling outcome {sampling_outcome!r}")
    elif source_valid and target is not None:
        if target.status == ObservationStatus.VISIBLE.value:
            target_x_m = float(target.x_m)
            target_y_m = float(target.y_m)
            candidate = _point_in_output(
                target_dataset,
                search_window,
                target_x_m,
                target_y_m,
                search_width,
                search_height,
            )
            if _inside(candidate, search_width, search_height):
                match_present = True
                target_point_px = candidate
            elif _contains_distractor(
                target_observations,
                str(source.track_id),
                target_dataset,
                search_window,
                search_width,
                search_height,
            ):
                negative_type = NegativeType.DISTRACTOR.value
            else:
                negative_type = NegativeType.TARGET_OUTSIDE_WINDOW.value
        elif target.status in {
            ObservationStatus.UNOBSERVABLE.value,
            ObservationStatus.OUTSIDE_COVERAGE.value,
        }:
            negative_type = NegativeType.TARGET_UNOBSERVABLE.value
        else:
            raise PreparationError(
                f"unsupported target status {target.status!r} for track {source.track_id}"
            )

    sample_key = {
        "pair_id": str(pair.pair_id),
        "track_id": str(source.track_id),
        "repetition": repetition,
        "sampling_intent": sampling_intent or "legacy_label_blind",
        "reference_window": _window_values(reference_window),
        "search_window": _window_values(search_window),
    }
    sample_id = hashlib.sha256(
        json.dumps(sample_key, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    reference_bounds = rasterio.windows.bounds(reference_window, reference_dataset.transform)
    search_bounds = rasterio.windows.bounds(search_window, target_dataset.transform)
    manifest = {
        "sample_id": sample_id,
        "pair_id": str(pair.pair_id),
        "glacier_id": str(pair.glacier_id),
        "track_id": str(source.track_id),
        "spatial_group": str(source.spatial_group),
        "reference_survey_id": str(pair.reference_survey_id),
        "target_survey_id": str(pair.target_survey_id),
        "interval_days": (
            None if pair.interval_days is None or np.isnan(pair.interval_days)
            else int(pair.interval_days)
        ),
        "repetition": repetition,
        "sampling_intent": sampling_intent or "legacy_label_blind",
        "source_valid": source_valid,
        "match_present": match_present,
        "negative_type": negative_type,
        "source_x_m": float(source.x_m),
        "source_y_m": float(source.y_m),
        "target_x_m": target_x_m,
        "target_y_m": target_y_m,
        "source_x_px": source_point_px[0],
        "source_y_px": source_point_px[1],
        "source_major_axis_m": _optional_float(source.major_axis_m),
        "source_minor_axis_m": _optional_float(source.minor_axis_m),
        "source_orientation_deg": _optional_float(source.orientation_deg),
        "target_x_px": target_point_px[0],
        "target_y_px": target_point_px[1],
        "search_jitter_x_m": jitter_x,
        "search_jitter_y_m": jitter_y,
        "search_placement": search_placement,
        "distractor_track_id": distractor_track_id,
        "reference_left_m": reference_bounds[0],
        "reference_bottom_m": reference_bounds[1],
        "reference_right_m": reference_bounds[2],
        "reference_top_m": reference_bounds[3],
        "search_left_m": search_bounds[0],
        "search_bottom_m": search_bounds[1],
        "search_right_m": search_bounds[2],
        "search_top_m": search_bounds[3],
        "reference_valid_fraction": reference_valid,
        "search_valid_fraction": search_valid,
        "tensor_gsd_m_per_px": settings.tensor_gsd_m_per_px,
        "reference_native_gsd_x_m": reference_survey.native_gsd_x_m,
        "reference_native_gsd_y_m": reference_survey.native_gsd_y_m,
        "target_native_gsd_x_m": target_survey.native_gsd_x_m,
        "target_native_gsd_y_m": target_survey.native_gsd_y_m,
        "reference_effective_gsd_m_per_px": max(
            reference_survey.native_gsd_x_m,
            reference_survey.native_gsd_y_m,
            settings.tensor_gsd_m_per_px,
        ),
        "target_effective_gsd_m_per_px": max(
            target_survey.native_gsd_x_m,
            target_survey.native_gsd_y_m,
            settings.tensor_gsd_m_per_px,
        ),
        "reference_raster_uri": reference_survey.raster_uri,
        "target_raster_uri": target_survey.raster_uri,
    }
    return manifest, reference_image, search_image


class _HDF5CacheWriter:
    def __init__(
        self,
        path: Path,
        *,
        channels: int,
        reference_size: tuple[int, int],
        search_size: tuple[int, int],
        tensor_gsd: float,
        compression: str | None,
    ) -> None:
        self.handle = h5py.File(path, "w")
        self.handle.attrs["schema_name"] = "glaciertrack-pairs"
        self.handle.attrs["schema_version"] = 1
        self.handle.attrs["manifest_sha256"] = ""
        self.handle.attrs["tensor_gsd_m_per_px"] = tensor_gsd
        self.length = 0
        self.datasets = {
            "reference_images": self._image_dataset(
                "reference_images", channels, reference_size, compression
            ),
            "search_images": self._image_dataset(
                "search_images", channels, search_size, compression
            ),
            "source_points_px": self._numeric_dataset(
                "source_points_px", (2,), np.float32, compression
            ),
            "target_points_px": self._numeric_dataset(
                "target_points_px", (2,), np.float32, compression
            ),
            "source_major_axis_m": self._numeric_dataset(
                "source_major_axis_m", (), np.float32, compression
            ),
            "source_minor_axis_m": self._numeric_dataset(
                "source_minor_axis_m", (), np.float32, compression
            ),
            "source_orientation_deg": self._numeric_dataset(
                "source_orientation_deg", (), np.float32, compression
            ),
            "reference_effective_gsd_m_per_px": self._numeric_dataset(
                "reference_effective_gsd_m_per_px", (), np.float32, compression
            ),
            "target_effective_gsd_m_per_px": self._numeric_dataset(
                "target_effective_gsd_m_per_px", (), np.float32, compression
            ),
            "source_valid": self._numeric_dataset("source_valid", (), np.bool_, compression),
            "match_present": self._numeric_dataset(
                "match_present", (), np.bool_, compression
            ),
            "sample_ids": self.handle.create_dataset(
                "sample_ids",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype("utf-8"),
                chunks=(256,),
            ),
        }

    def _image_dataset(
        self,
        name: str,
        channels: int,
        size: tuple[int, int],
        compression: str | None,
    ) -> Any:
        height, width = size
        return self.handle.create_dataset(
            name,
            shape=(0, channels, height, width),
            maxshape=(None, channels, height, width),
            dtype=np.uint8,
            chunks=(1, channels, height, width),
            compression=compression,
            shuffle=compression is not None,
        )

    def _numeric_dataset(
        self,
        name: str,
        trailing_shape: tuple[int, ...],
        dtype: Any,
        compression: str | None,
    ) -> Any:
        return self.handle.create_dataset(
            name,
            shape=(0, *trailing_shape),
            maxshape=(None, *trailing_shape),
            dtype=dtype,
            chunks=(256, *trailing_shape),
            compression=compression,
            shuffle=compression is not None,
        )

    def append(
        self,
        manifest: dict[str, Any],
        reference_image: np.ndarray,
        search_image: np.ndarray,
    ) -> None:
        next_length = self.length + 1
        for dataset in self.datasets.values():
            dataset.resize(next_length, axis=0)
        index = self.length
        self.datasets["reference_images"][index] = reference_image
        self.datasets["search_images"][index] = search_image
        self.datasets["source_points_px"][index] = (
            manifest["source_x_px"],
            manifest["source_y_px"],
        )
        self.datasets["target_points_px"][index] = (
            manifest["target_x_px"],
            manifest["target_y_px"],
        )
        self.datasets["source_major_axis_m"][index] = _nan_if_none(
            manifest["source_major_axis_m"]
        )
        self.datasets["source_minor_axis_m"][index] = _nan_if_none(
            manifest["source_minor_axis_m"]
        )
        self.datasets["source_orientation_deg"][index] = _nan_if_none(
            manifest["source_orientation_deg"]
        )
        self.datasets["reference_effective_gsd_m_per_px"][index] = manifest[
            "reference_effective_gsd_m_per_px"
        ]
        self.datasets["target_effective_gsd_m_per_px"][index] = manifest[
            "target_effective_gsd_m_per_px"
        ]
        self.datasets["source_valid"][index] = manifest["source_valid"]
        self.datasets["match_present"][index] = manifest["match_present"]
        self.datasets["sample_ids"][index] = manifest["sample_id"]
        self.length = next_length

    def close(self) -> None:
        if self.handle:
            self.handle.attrs["sample_count"] = self.length
            self.handle.close()


def _point_positioned_window(
    dataset: Any,
    x_m: float,
    y_m: float,
    desired_x: float,
    desired_y: float,
    output_width: int,
    output_height: int,
    tensor_gsd: float,
    survey: _SurveyRecord,
    rasterio: Any,
) -> Any:
    source_col, source_row = (~dataset.transform) * (x_m, y_m)
    window_width = output_width * tensor_gsd / survey.native_gsd_x_m
    window_height = output_height * tensor_gsd / survey.native_gsd_y_m
    return rasterio.windows.Window(
        source_col - desired_x * window_width / output_width,
        source_row - desired_y * window_height / output_height,
        window_width,
        window_height,
    )


def _centered_window(
    dataset: Any,
    x_m: float,
    y_m: float,
    jitter_x_m: float,
    jitter_y_m: float,
    output_width: int,
    output_height: int,
    tensor_gsd: float,
    survey: _SurveyRecord,
    rasterio: Any,
) -> Any:
    center_col, center_row = (~dataset.transform) * (x_m, y_m)
    center_col += jitter_x_m / survey.native_gsd_x_m
    center_row += jitter_y_m / survey.native_gsd_y_m
    window_width = output_width * tensor_gsd / survey.native_gsd_x_m
    window_height = output_height * tensor_gsd / survey.native_gsd_y_m
    return rasterio.windows.Window(
        center_col - window_width / 2,
        center_row - window_height / 2,
        window_width,
        window_height,
    )


def _read_crop(
    dataset: Any,
    window: Any,
    bands: tuple[int, ...],
    output_size: tuple[int, int],
    survey: _SurveyRecord,
    rasterio: Any,
) -> tuple[np.ndarray, float]:
    height, width = output_size
    downsampling = window.width > width or window.height > height
    resampling = (
        rasterio.enums.Resampling.average
        if downsampling
        else rasterio.enums.Resampling.bilinear
    )
    array = dataset.read(
        indexes=bands,
        window=window,
        out_shape=(len(bands), height, width),
        boundless=True,
        fill_value=dataset.nodata if dataset.nodata is not None else 0,
        resampling=resampling,
    )
    masks = dataset.read_masks(
        indexes=bands,
        window=window,
        out_shape=(len(bands), height, width),
        boundless=True,
        resampling=rasterio.enums.Resampling.nearest,
    )
    valid_fraction = float(np.all(masks > 0, axis=0).mean())
    scaled = np.clip(
        (array.astype(np.float32) - survey.radiometric_min)
        / (survey.radiometric_max - survey.radiometric_min),
        0,
        1,
    )
    return np.rint(scaled * 255).astype(np.uint8), valid_fraction


def _point_in_output(
    dataset: Any,
    window: Any,
    x_m: float,
    y_m: float,
    output_width: int,
    output_height: int,
) -> tuple[float, float]:
    column, row = (~dataset.transform) * (x_m, y_m)
    return (
        float((column - window.col_off) * output_width / window.width),
        float((row - window.row_off) * output_height / window.height),
    )


def _contains_distractor(
    observations: Any,
    track_id: str,
    dataset: Any,
    window: Any,
    output_width: int,
    output_height: int,
) -> bool:
    candidates = observations[
        (observations["status"] == ObservationStatus.VISIBLE.value)
        & (observations["track_id"].astype(str) != track_id)
    ]
    for row in candidates.itertuples(index=False):
        point = _point_in_output(
            dataset,
            window,
            float(row.x_m),
            float(row.y_m),
            output_width,
            output_height,
        )
        if _inside(point, output_width, output_height):
            return True
    return False


def _transform_point(
    x_m: float,
    y_m: float,
    source_crs: str,
    target_crs: str,
    rasterio: Any,
) -> tuple[float, float]:
    if source_crs == target_crs:
        return x_m, y_m
    x_values, y_values = rasterio.warp.transform(source_crs, target_crs, [x_m], [y_m])
    return float(x_values[0]), float(y_values[0])


def _inside(point: tuple[float, float], width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height


def _resolve_raster(survey_dir: Path, metadata: dict[str, Any]) -> Path:
    configured = metadata.get("raster")
    if configured is not None:
        path = survey_dir / str(configured)
        if not path.is_file():
            raise PreparationError(f"configured raster does not exist: {path}")
        return path.resolve()
    candidates = sorted([*survey_dir.glob("*.tif"), *survey_dir.glob("*.tiff")])
    if len(candidates) != 1:
        raise PreparationError(
            f"expected exactly one GeoTIFF or a survey.yaml raster entry in {survey_dir}"
        )
    return candidates[0].resolve()


def _resolve_aoi(
    survey_dir: Path,
    glacier_dir: Path,
    metadata: dict[str, Any],
    settings: PreparationSettings,
) -> Path | None:
    configured = metadata.get("aoi")
    if configured is not None:
        path = survey_dir.joinpath(str(configured)).resolve()
        if not path.is_file():
            raise PreparationError(f"configured AOI does not exist: {path}")
        if path.suffix.lower() not in {".gpkg", ".geojson", ".parquet"}:
            raise PreparationError(f"unsupported AOI format: {path}")
        return path

    for directory in (survey_dir, glacier_dir):
        candidates = sorted(
            path
            for path in directory.glob(f"{settings.aoi_stem}.*")
            if path.suffix.lower() in {".gpkg", ".geojson", ".parquet"}
        )
        if len(candidates) > 1:
            raise PreparationError(f"multiple AOI files found in {directory}: {candidates}")
        if candidates:
            return candidates[0].resolve()
    return None


def _inspect_aoi(
    path: Path,
    survey_crs: str,
    raster_bounds: tuple[float, float, float, float],
) -> tuple[str, float, tuple[float, float, float, float]]:
    geopandas = _require_geopandas()
    frame = (
        geopandas.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else geopandas.read_file(path)
    )
    if frame.empty or "geometry" not in frame:
        raise PreparationError(f"AOI contains no geometries: {path}")
    if frame.crs is None:
        raise PreparationError(f"AOI has no CRS: {path}")
    frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty:
        raise PreparationError(f"AOI contains no non-empty geometries: {path}")
    geometry_types = set(frame.geometry.geom_type.astype(str))
    if not geometry_types.issubset({"Polygon", "MultiPolygon"}):
        raise PreparationError(
            f"AOI must contain only Polygon/MultiPolygon geometry: {path} ({geometry_types})"
        )
    if not bool(frame.geometry.is_valid.all()):
        raise PreparationError(f"AOI contains invalid polygon geometry: {path}")
    if str(frame.crs) != survey_crs:
        frame = frame.to_crs(survey_crs)
    left, bottom, right, top = (float(value) for value in frame.total_bounds)
    raster_left, raster_bottom, raster_right, raster_top = raster_bounds
    if (
        right <= raster_left
        or left >= raster_right
        or top <= raster_bottom
        or bottom >= raster_top
    ):
        raise PreparationError(f"AOI does not intersect its survey raster: {path}")
    area = float(frame.geometry.area.sum())
    if not math.isfinite(area) or area <= 0:
        raise PreparationError(f"AOI has no positive area: {path}")
    return survey_crs, area, (left, bottom, right, top)


def _radiometric_range(dtype: str, metadata: dict[str, Any]) -> tuple[float, float]:
    configured = metadata.get("radiometric_range")
    if configured is not None:
        if not isinstance(configured, (list, tuple)) or len(configured) != 2:
            raise PreparationError("radiometric_range must be [minimum, maximum]")
        minimum, maximum = float(configured[0]), float(configured[1])
    else:
        numpy_dtype = np.dtype(dtype)
        if np.issubdtype(numpy_dtype, np.integer):
            limits = np.iinfo(numpy_dtype)
            minimum, maximum = float(limits.min), float(limits.max)
        elif np.issubdtype(numpy_dtype, np.floating):
            minimum, maximum = 0.0, 1.0
        else:
            raise PreparationError(f"unsupported raster dtype: {dtype}")
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
        raise PreparationError("radiometric range must contain two increasing finite values")
    return minimum, maximum


def _tabular_crs(frame: Any, survey_crs: str, path: Path, warnings: list[str]) -> str:
    if "crs" not in frame.columns:
        warnings.append(f"assuming raster CRS for tabular annotations without crs column: {path}")
        return survey_crs
    values = {str(value) for value in frame["crs"].dropna().unique() if str(value)}
    if not values:
        warnings.append(f"assuming raster CRS for annotations with empty crs column: {path}")
        return survey_crs
    if len(values) != 1:
        raise PreparationError(f"annotation file contains multiple CRS values: {path}")
    return next(iter(values))


def _require_geopandas() -> Any:
    try:
        import geopandas
    except ImportError as error:
        raise PreparationError(
            "GeoPackage/GeoJSON/GeoParquet annotations require geopandas and pyogrio"
        ) from error
    return geopandas


def _parse_date(value: Any, path: Path) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        raise PreparationError(f"acquisition_date is required in {path}")
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise PreparationError(f"acquisition_date must use YYYY-MM-DD in {path}") from error


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PreparationError(f"configuration requires a {key!r} mapping")
    return value


def _pixel_size(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        result = (int(value[0]), int(value[1]))
    else:
        raise PreparationError(f"{name} must be an integer or [height, width]")
    if min(result) <= 0:
        raise PreparationError(f"{name} values must be positive")
    return result


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PreparationError(f"{name} must be numeric") from error
    if not math.isfinite(result) or result <= 0:
        raise PreparationError(f"{name} must be positive and finite")
    return result


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nan_if_none(value: float | None) -> float:
    return math.nan if value is None else float(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return None if not result or result.lower() in {"nan", "<na>", "none"} else result


def _window_values(window: Any) -> list[float]:
    return [
        round(float(window.col_off), 8),
        round(float(window.row_off), 8),
        round(float(window.width), 8),
        round(float(window.height), 8),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_paths(root: Path) -> dict[str, Path]:
    return {
        "surveys": root / "manifests" / "surveys.parquet",
        "observations": root / "manifests" / "observations.parquet",
        "pairs": root / "manifests" / "survey_pairs.parquet",
        "samples": root / "manifests" / "samples.parquet",
        "cache": root / "cache" / "training.h5",
        "report": root / "reports" / "preparation.json",
    }


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.partial")


def _check_output_paths(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise PreparationError(
            "prepared outputs already exist; use --overwrite to replace them: "
            + ", ".join(existing)
        )


def _check_temporary_paths(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise PreparationError(
            "stale partial outputs exist; inspect or rerun with --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    for path in existing:
        path.unlink()
