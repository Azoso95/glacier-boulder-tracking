"""Frozen zero-shot correspondence baselines for glacier boulder queries.

No endpoints enter inference. Fixed methods/parameters are documented in the
output; endpoint labels are read only after predictions have been saved.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from affine import Affine
from glaciertrack.inference.raster import AnalysisRaster
from pyproj import Transformer
from rasterio.features import geometry_mask
from scipy import ndimage, signal
from scipy.spatial import cKDTree

GSD_M = 0.2
REFERENCE_HW = (192, 192)
TEMPLATE_M = 9.0
PROMPT_RADIUS_M = 4.75
TILE_PX = 512
VOTE_BANDWIDTH_PX = 8.0


@dataclass(frozen=True)
class Pair:
    name: str
    source: Path
    target: Path
    aoi: Path
    sources: Path
    truth: Path
    radius_m: float




def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def gray_u8(rgb_chw: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(rgb_chw, 0, -1)
    return cv2.cvtColor(np.ascontiguousarray(np.round(rgb * 255).astype(np.uint8)), cv2.COLOR_RGB2GRAY)


def search_window(target: AnalysisRaster, source_xy: tuple[float, float], radius_m: float):
    center_x, center_y = target.xy_to_pixel(*source_xy)
    radius = radius_m / GSD_M
    left = max(0, math.floor(center_x - radius))
    top = max(0, math.floor(center_y - radius))
    right = min(target.width, math.ceil(center_x + radius) + 1)
    bottom = min(target.height, math.ceil(center_y + radius) + 1)
    return left, top, right, bottom, center_x, center_y


def validity_mask(shape_hw, *, left, top, center_x, center_y, radius_m, target, shapes):
    height, width = shape_hw
    yy, xx = np.ogrid[:height, :width]
    circle = ((xx + left + 0.5 - center_x) ** 2 + (yy + top + 0.5 - center_y) ** 2) <= (radius_m / GSD_M) ** 2
    aoi = geometry_mask(
        shapes,
        out_shape=shape_hw,
        transform=target.transform * Affine.translation(left, top),
        invert=True,
        all_touched=True,
    )
    return circle & aoi


def ncc_response(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    template = template.astype(np.float32)
    centered = template - float(template.mean())
    numerator = signal.fftconvolve(image, centered[::-1, ::-1], mode="same")
    ones = np.ones(template.shape, dtype=np.float32)
    sums = signal.fftconvolve(image, ones, mode="same")
    squares = signal.fftconvolve(image * image, ones, mode="same")
    variance = np.maximum(squares - sums * sums / template.size, 1e-6)
    response = numerator / np.sqrt(variance * max(float((centered * centered).sum()), 1e-6))
    return np.clip(response, -1, 1)


def ncc_predict(reference: np.ndarray, point, canvas: np.ndarray, valid: np.ndarray, kind: str):
    ref = gray_u8(reference).astype(np.float32) / 255.0
    dst = gray_u8(canvas).astype(np.float32) / 255.0
    if kind == "gradient":
        ref = np.hypot(ndimage.sobel(ref, axis=0), ndimage.sobel(ref, axis=1))
        dst = np.hypot(ndimage.sobel(dst, axis=0), ndimage.sobel(dst, axis=1))
    size = round(TEMPLATE_M / GSD_M)
    offsets = np.arange(size, dtype=np.float32) - (size - 1) / 2
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    template = ndimage.map_coordinates(ref, [dy + point[1], dx + point[0]], order=1, mode="nearest")
    response = ncc_response(dst, template)
    response[~valid] = -np.inf
    if not np.isfinite(response).any():
        return None
    y, x = np.unravel_index(int(np.argmax(response)), response.shape)
    return {"x_px": float(x), "y_px": float(y), "score": float(response[y, x]), "candidate_count": int(valid.sum())}


def sift_matches(source_gray, target_gray, sift, matcher):
    key0, des0 = sift.detectAndCompute(source_gray, None)
    key1, des1 = sift.detectAndCompute(target_gray, None)
    if des0 is None or des1 is None or len(des1) < 2:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    pairs = matcher.knnMatch(des0, des1, k=2)
    good = [(a, b) for a, b in pairs if a.distance < 0.75 * b.distance]
    return (
        np.array([key0[a.queryIdx].pt for a, _ in good], dtype=np.float32).reshape(-1, 2),
        np.array([key1[a.trainIdx].pt for a, _ in good], dtype=np.float32).reshape(-1, 2),
        np.array([max(1e-3, 1 - a.distance / b.distance) for a, b in good], dtype=np.float32),
    )




def disk_lightglue_matches(source_rgb, target_rgb, model, device):
    from kornia.feature import laf_from_center_scale_ori

    detector, matcher = model
    source = torch.from_numpy(source_rgb.copy()).to(device=device, dtype=torch.float32)[None]
    target = torch.from_numpy(target_rgb.copy()).to(device=device, dtype=torch.float32)[None]
    with torch.inference_mode():
        feature0 = detector(source, n=512)[0]
        feature1 = detector(target, n=2048)[0]
        if feature0.n < 2 or feature1.n < 2:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        scores, indices = matcher(
            feature0.descriptors,
            feature1.descriptors,
            laf_from_center_scale_ori(feature0.keypoints[None]),
            laf_from_center_scale_ori(feature1.keypoints[None]),
            hw1=source.shape[-2:],
            hw2=target.shape[-2:],
        )
    return (
        feature0.keypoints[indices[:, 0]].detach().cpu().numpy(),
        feature1.keypoints[indices[:, 1]].detach().cpu().numpy(),
        scores[:, 0].detach().cpu().numpy(),
    )


def vote(matches: list[tuple[float, float, float]], valid: np.ndarray):
    if not matches:
        return None
    points = np.asarray([(x, y) for x, y, _ in matches], dtype=np.float32)
    weights = np.asarray([score for _, _, score in matches], dtype=np.float32)
    tree = cKDTree(points)
    local_scores = np.empty(len(points), dtype=np.float32)
    for i, neighbors in enumerate(tree.query_ball_point(points, 2.5 * VOTE_BANDWIDTH_PX)):
        delta = points[neighbors] - points[i]
        local_scores[i] = np.sum(weights[neighbors] * np.exp(-np.sum(delta * delta, axis=1) / (2 * VOTE_BANDWIDTH_PX**2)))
    winner = int(np.argmax(local_scores))
    x, y = points[winner]
    return {"x_px": float(x), "y_px": float(y), "score": float(local_scores[winner]), "candidate_count": len(matches)}


def feature_predict(reference: np.ndarray, point, canvas: np.ndarray, valid: np.ndarray, method, model, device):
    source = gray_u8(reference)
    height, width = valid.shape
    search = gray_u8(canvas)
    collected = []
    if method == "sift":
        sift = cv2.SIFT_create(nfeatures=1500)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    for top in range(0, height, TILE_PX):
        for left in range(0, width, TILE_PX):
            tile = search[top : top + TILE_PX, left : left + TILE_PX]
            if not valid[top : top + len(tile), left : left + tile.shape[1]].any():
                continue
            if tile.shape != (TILE_PX, TILE_PX):
                tile = cv2.copyMakeBorder(tile, 0, TILE_PX - tile.shape[0], 0, TILE_PX - tile.shape[1], cv2.BORDER_CONSTANT, value=0)
            if method == "sift":
                kp0, kp1, conf = sift_matches(source, tile, sift, matcher)
            else:
                rgb_tile = canvas[:, top : top + TILE_PX, left : left + TILE_PX]
                if rgb_tile.shape[1:] != (TILE_PX, TILE_PX):
                    padded = np.zeros((3, TILE_PX, TILE_PX), dtype=np.float32)
                    padded[:, : rgb_tile.shape[1], : rgb_tile.shape[2]] = rgb_tile
                    rgb_tile = padded
                kp0, kp1, conf = disk_lightglue_matches(reference, rgb_tile, model, device)
            if not len(kp0):
                continue
            delta = np.asarray(point, dtype=np.float32)[None, :] - kp0
            close = np.linalg.norm(delta, axis=1) <= PROMPT_RADIUS_M / GSD_M
            endpoints = kp1 + delta + np.asarray([left, top], dtype=np.float32)[None, :]
            for (x, y), keep, confidence in zip(endpoints, close, conf, strict=True):
                ix, iy = round(float(x)), round(float(y))
                if keep and 0 <= ix < width and 0 <= iy < height and valid[iy, ix]:
                    collected.append((float(x), float(y), float(confidence)))
    return vote(collected, valid)


def infer_pair(pair: Pair, *, methods, models, device, output: Path, limit: int | None):
    output.mkdir(parents=True, exist_ok=True)
    sources = pd.read_parquet(pair.sources, columns=["query_id", "source_x_m", "source_y_m"])
    if limit is not None:
        sources = sources.head(limit)
    records = {method: [] for method in methods}
    with AnalysisRaster(pair.source, tensor_gsd_m_per_px=GSD_M) as source, AnalysisRaster(pair.target, tensor_gsd_m_per_px=GSD_M) as target:
        shapes = list(gpd.read_file(pair.aoi).to_crs(target.crs).geometry)
        reproject = None if source.crs == target.crs else Transformer.from_crs(source.crs, target.crs, always_xy=True)
        for index, row in enumerate(sources.itertuples(index=False), start=1):
            started = time.perf_counter()
            query_xy = (float(row.source_x_m), float(row.source_y_m))
            target_center = query_xy if reproject is None else reproject.transform(*query_xy)
            reference, point = source.read_centered_reference(query_xy, REFERENCE_HW)
            left, top, right, bottom, cx, cy = search_window(target, target_center, pair.radius_m)
            canvas = target.read_window(top, left, bottom - top, right - left).numpy()
            valid = validity_mask((bottom - top, right - left), left=left, top=top, center_x=cx, center_y=cy, radius_m=pair.radius_m, target=target, shapes=shapes)
            for method in methods:
                if method.startswith("ncc_"):
                    result = ncc_predict(reference.numpy(), point, canvas, valid, method.removeprefix("ncc_"))
                else:
                    result = feature_predict(reference.numpy(), point, canvas, valid, method, models.get(method), device)
                xy = None if result is None else target.pixel_to_xy(left + result["x_px"], top + result["y_px"])
                records[method].append({
                    "query_id": str(row.query_id),
                    "prediction_xy_m": None if xy is None else list(xy),
                    "score": None if result is None else result["score"],
                    "candidate_count": 0 if result is None else result["candidate_count"],
                })
            print(json.dumps({"pair": pair.name, "query": index, "total": len(sources), "seconds": round(time.perf_counter() - started, 2)}), flush=True)
    # Commit source-only predictions before reading any target coordinates.
    hashes = {}
    for method, rows in records.items():
        path = output / f"{method}_predictions.json"
        dump(path, rows)
        hashes[method] = digest(path)
    truth = pd.read_parquet(pair.truth).set_index("query_id")
    if truth.index.has_duplicates or not set(sources.query_id).issubset(truth.index):
        raise ValueError(f"invalid scoring IDs for {pair.name}")
    if limit is None and set(sources.query_id) != set(truth.index):
        raise ValueError(f"source and reference query sets differ for {pair.name}")
    metrics = {}
    for method, rows in records.items():
        errors = []
        detail = []
        for row in rows:
            target_row = truth.loc[row["query_id"]]
            endpoint = np.array([target_row.target_x_m, target_row.target_y_m], dtype=float)
            if row["prediction_xy_m"] is None:
                error = None
            else:
                error = float(np.linalg.norm(np.asarray(row["prediction_xy_m"]) - endpoint))
                errors.append(error)
            detail.append({**row, "error_m": error})
        dump(output / f"{method}_errors.json", detail)
        count = len(rows)
        metrics[method] = {
            "queries": count,
            "coverage": len(errors) / count,
            "success_at_2m_all_queries": sum(error <= 2 for error in errors) / count,
            "success_at_5m_all_queries": sum(error <= 5 for error in errors) / count,
            "success_at_10m_all_queries": sum(error <= 10 for error in errors) / count,
            "median_error_m_accepted": None if not errors else float(np.median(errors)),
            "mean_error_m_accepted": None if not errors else float(np.mean(errors)),
            "p90_error_m_accepted": None if not errors else float(np.percentile(errors, 90)),
        }
    dump(output / "result.json", {
        "pair": pair.name,
        "query_sha256": digest(pair.truth),
        "radius_m": pair.radius_m,
        "gsd_m_per_px": GSD_M,
        "parameters": {"reference_hw": REFERENCE_HW, "template_m": TEMPLATE_M, "prompt_radius_m": PROMPT_RADIUS_M, "tile_px": TILE_PX, "vote_bandwidth_px": VOTE_BANDWIDTH_PX, "sift_ratio": 0.75, "disk_checkpoint": "official depth", "lightglue_checkpoint": "official disk"},
        "prediction_sha256": hashes,
        "metrics": metrics,
    })
    return metrics
