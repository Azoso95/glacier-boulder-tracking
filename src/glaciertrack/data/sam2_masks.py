"""Frozen, source-only SAM 2 segmentation with content-addressed mask caching.

No destination image, endpoint or tracking error enters mask generation/selection.
SAM quality scores are heuristics, not calibrated boulder identification probabilities.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

ALGORITHM_VERSION = 1


@dataclass(frozen=True)
class SAM2MaskConfig:
    checkpoint: str
    cache_dir: str
    model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    checkpoint_sha256: str | None = None
    device: str = "auto"
    crop_sizes_m: tuple[float, ...] = (12.0, 24.0, 38.4)
    minimum_score: float = 0.5
    minimum_area_m2: float = 0.04
    maximum_area_m2: float = 200.0
    maximum_crop_fraction: float = 0.65
    edge_softness_m: float = 0.10
    fallback: str = "error"

    def __post_init__(self):
        if not self.checkpoint or not self.cache_dir:
            raise ValueError("SAM 2 checkpoint and cache_dir are required")
        if not self.crop_sizes_m or min(self.crop_sizes_m) <= 0:
            raise ValueError("SAM 2 crop sizes must be positive metres")
        if not 0 < self.minimum_area_m2 < self.maximum_area_m2:
            raise ValueError("invalid SAM 2 area limits")
        if not 0 < self.maximum_crop_fraction < 1 or not 0 <= self.minimum_score <= 1:
            raise ValueError("invalid SAM 2 quality thresholds")
        if self.edge_softness_m < 0 or self.fallback not in {"error", "gaussian"}:
            raise ValueError("invalid SAM 2 softness or fallback")


@lru_cache(maxsize=8)
def _checkpoint_digest(path: str, size: int, mtime_ns: int) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(
            hashlib, "file_digest"
        ) else _stream_digest(stream)


def _stream_digest(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def mask_identity(image: Tensor, point: Tensor, gsd: float, config: SAM2MaskConfig):
    image = image.detach().cpu()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("SAM 2 expects a 3xHxW RGB image")
    if image.dtype == torch.uint8:
        rgb = image.permute(1, 2, 0).numpy()
    else:
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("SAM 2 RGB floats must be finite and in [0, 1]")
        rgb = image.mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    path = Path(config.checkpoint).resolve()
    stat = path.stat()
    digest = _checkpoint_digest(str(path), stat.st_size, stat.st_mtime_ns)
    if config.checkpoint_sha256 and digest != config.checkpoint_sha256:
        raise ValueError("SAM 2 checkpoint checksum differs from configured provenance")
    settings = asdict(config)
    for name in ("cache_dir", "device", "checkpoint", "checkpoint_sha256"):
        settings.pop(name)
    metadata = {"algorithm_version": ALGORITHM_VERSION, "settings": settings,
                "checkpoint_sha256": digest, "point_xy_px": point.tolist(),
                "gsd_m_per_px": gsd, "image_shape": list(rgb.shape)}
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode() + rgb.tobytes()).hexdigest()
    return np.ascontiguousarray(rgb), key, metadata


@lru_cache(maxsize=2)
def _predictor(checkpoint: str, model_config: str, device: str):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as error:
        raise RuntimeError("Install the optional SAM 2 backend; see docs/sam2.md") from error
    selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model = build_sam2(model_config, checkpoint, device=selected, apply_postprocessing=False)
    return SAM2ImagePredictor(model.eval(), max_hole_area=0, max_sprinkle_area=0)


def release_sam2():
    """Release the frozen GPU segmenter after cache preparation."""
    _predictor.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def select_candidates(masks, scores, point, gsd, config):
    from scipy.ndimage import label

    x, y = np.floor(point).astype(int)
    accepted = []
    for mask, score in zip(masks, scores, strict=True):
        mask = np.asarray(mask, dtype=bool)
        if not np.isfinite(score) or float(score) < config.minimum_score or not mask[y, x]:
            continue
        components, _ = label(mask)
        component = components == components[y, x]
        area = float(component.sum()) * gsd**2
        touches = bool(component[0].any() or component[-1].any()
                       or component[:, 0].any() or component[:, -1].any())
        if (touches or not config.minimum_area_m2 <= area <= config.maximum_area_m2
                or component.mean() > config.maximum_crop_fraction):
            continue
        accepted.append((component, float(score), area))
    return accepted


def generate_mask(rgb, point, gsd, config):
    height, width = rgb.shape[:2]
    predictor = _predictor(config.checkpoint, config.model_config, config.device)
    candidates = []
    windows = set()
    with torch.inference_mode():
        for side_m in sorted(config.crop_sizes_m):
            side = max(8, int(math.ceil(side_m / gsd)))
            w, h = min(width, side), min(height, side)
            left = max(0, min(width - w, int(math.floor(float(point[0]) - w / 2))))
            top = max(0, min(height - h, int(math.floor(float(point[1]) - h / 2))))
            window = (left, top, w, h)
            if window in windows:
                continue
            windows.add(window)
            local = np.asarray(point) - [left, top]
            predictor.set_image(rgb[top:top+h, left:left+w])
            masks, scores, _ = predictor.predict(
                point_coords=local[None], point_labels=np.ones(1, dtype=np.int32),
                multimask_output=True,
            )
            for component, score, area in select_candidates(masks, scores, local, gsd, config):
                full = np.zeros((height, width), dtype=np.float32)
                full[top:top+h, left:left+w] = component
                candidates.append((score, full, area, window))
    if not candidates:
        return None, {"status": "no_valid_candidate", "candidate_count": 0}
    score, mask, area, window = max(candidates, key=lambda item: item[0])
    return mask, {"status": "sam2", "candidate_count": len(candidates),
                  "sam_quality_score": score, "area_m2": area, "crop_window": window}


def base_mask(image, point, gsd, config, fallback_sigma):
    height, width = image.shape[-2:]
    if (not torch.isfinite(point).all() or not 0 <= float(point[0]) < width
            or not 0 <= float(point[1]) < height):
        return torch.zeros((1, height, width)), {"status": "source_outside_image"}
    rgb, key, metadata = mask_identity(image, point, gsd, config)
    # A Gaussian fallback's width affects the resulting mask and therefore its identity.
    key = hashlib.sha256(f"{key}:{fallback_sigma}".encode()).hexdigest()
    cache = Path(config.cache_dir) / key[:2] / f"{key}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            mask = np.asarray(archive["mask"], dtype=np.float32)
            report = json.loads(str(archive["metadata"]))
        if mask.shape != (height, width) or not np.isfinite(mask).all():
            raise ValueError(f"invalid SAM 2 mask cache: {cache}")
        return torch.from_numpy(mask.copy())[None], report
    if torch.utils.data.get_worker_info() is not None:
        raise RuntimeError("SAM 2 cache miss in DataLoader worker; prewarm masks before training")
    mask, report = generate_mask(rgb, point.numpy(), gsd, config)
    if mask is None:
        if config.fallback == "error":
            raise ValueError(f"SAM 2 found no valid boulder mask for source point {point.tolist()}")
        yy, xx = np.mgrid[:height, :width]
        mask = np.exp(-((xx-point[0].item())**2 + (yy-point[1].item())**2)
                      * gsd**2 / (2 * fallback_sigma**2)).astype(np.float32)
        report["status"] = "gaussian_fallback"
    elif config.edge_softness_m:
        from scipy.ndimage import gaussian_filter

        mask = gaussian_filter(mask, config.edge_softness_m / gsd)
    report = {**metadata, **report, "cache_key": key, "fallback_sigma_m": fallback_sigma}
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, mask=mask, metadata=json.dumps(report))
        os.replace(temporary, cache)
    finally:
        temporary.unlink(missing_ok=True)
    return torch.from_numpy(mask.copy())[None], report


def render_sam2_prompt(image, point, gsd, spec, augmentation, generator):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import affine

    from glaciertrack.data.prompts import _gaussian_blur, _morphology, _randint, _uniform

    mask, _ = base_mask(image, point.cpu(), gsd, SAM2MaskConfig(**spec.sam2), spec.point_sigma_m)
    if augmentation.enabled and mask.any():
        original = mask.clone()
        jitter = augmentation.center_jitter_m / gsd
        mask = affine(
            mask, angle=_uniform(-augmentation.orientation_jitter_deg,
                                 augmentation.orientation_jitter_deg, generator),
            translate=[int(round(_uniform(-jitter, jitter, generator))) for _ in range(2)],
            scale=_uniform(*augmentation.scale_range, generator), shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR, center=point.tolist(),
        )
        mask = _morphology(mask, _randint(*augmentation.morphology_radius_px, generator))
        sigma = _uniform(*augmentation.blur_sigma_px, generator)
        if sigma > 0.05:
            mask = _gaussian_blur(mask, sigma)
        if mask.sum() < 1e-4:
            mask = original
    return mask.clamp(0, 1)


def warm_hdf5_masks(path, spec):
    """Populate pristine source masks in the main process, never in GPU workers."""
    import h5py

    from glaciertrack.data.hdf5 import _image_tensor

    config = SAM2MaskConfig(**spec.sam2)
    counts = {}
    with h5py.File(path, "r") as handle:
        gsd = float(handle.attrs["tensor_gsd_m_per_px"])
        for index in range(len(handle["sample_ids"])):
            _, report = base_mask(
                _image_tensor(handle["reference_images"][index]),
                torch.as_tensor(handle["source_points_px"][index], dtype=torch.float32),
                gsd, config, spec.point_sigma_m,
            )
            status = report["status"]
            counts[status] = counts.get(status, 0) + 1
            if (index + 1) % 100 == 0:
                print(f"SAM 2 masks {index+1}/{len(handle['sample_ids'])}: {counts}", flush=True)
    release_sam2()
    report = {"hdf5": str(Path(path).resolve()), "counts": counts, "config": asdict(config)}
    report_path = Path(config.cache_dir) / (
        "hdf5_" + hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:16] + ".json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
