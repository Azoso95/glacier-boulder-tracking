"""Full-radius visual panels for the four-glacier held-out experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from pyproj import Transformer
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds

matplotlib.use("Agg")
# ruff: isort: off
import matplotlib.pyplot as plt
# ruff: isort: on


def local_background(path: Path, center: np.ndarray, half_side_m: float,
                     max_side_px: int = 2200) -> tuple[np.ndarray, tuple[float, ...]]:
    import rasterio

    x, y = map(float, center)
    bounds = (x - half_side_m, y - half_side_m,
              x + half_side_m, y + half_side_m)
    with rasterio.open(path) as dataset:
        window = from_bounds(*bounds, transform=dataset.transform)
        window = window.round_offsets().round_lengths()
        actual_bounds = window_bounds(window, dataset.transform)
        scale = min(1.0, max_side_px / max(window.width, window.height))
        shape = (max(1, round(window.height * scale)),
                 max(1, round(window.width * scale)))
        rgb = dataset.read((1, 2, 3), window=window, boundless=True,
                           out_shape=(3, *shape), resampling=Resampling.bilinear)
        alpha = dataset.dataset_mask(window=window, boundless=True,
                                     out_shape=shape, resampling=Resampling.nearest)
    rgba = np.concatenate((np.moveaxis(rgb, 0, -1), alpha[..., None]), axis=-1)
    return rgba, (actual_bounds[0], actual_bounds[2],
                  actual_bounds[1], actual_bounds[3])


def plot_query_panels(
    terrain, cache: Path, predictions: Path, methods: dict, truth,
    source_raster: Path, target_raster: Path, output: Path, title: str,
    *, radius_m: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    import rasterio

    with rasterio.open(source_raster) as raster:
        source_crs = str(raster.crs)
    indexed = truth.set_index("query_id")
    selected = list(terrain.METHOD_LABELS)
    lookup = {method: {row["query_id"]: row for row in methods[method]}
              for method in selected}
    files = []
    for index, (meta, _data) in enumerate(terrain.iter_cache(cache)):
        query_id = meta["query_id"]
        source = np.asarray(meta["source_xy_m"], dtype=float)
        if meta["target_crs"] != source_crs:
            transform = Transformer.from_crs(meta["target_crs"], source_crs,
                                             always_xy=True)
            source_display = np.asarray(transform.transform(*source))
        else:
            source_display = source
        truth_xy = indexed.loc[query_id, ["target_x_m", "target_y_m"]].to_numpy(dtype=float)
        source_bg, source_extent = local_background(source_raster, source_display, 20.0)
        target_bg, target_extent = local_background(
            target_raster, source, radius_m + 12.0,
        )
        fig, axes = plt.subplots(2, 3, figsize=(19, 12), constrained_layout=True)
        axes[0, 0].imshow(source_bg, extent=source_extent)
        axes[0, 0].set_xlim(source_display[0] - 20, source_display[0] + 20)
        axes[0, 0].set_ylim(source_display[1] - 20, source_display[1] + 20)
        axes[0, 0].plot(*source_display, "+", color="cyan", ms=14, mew=2)
        axes[0, 0].set_title("Input boulder and context")
        surface_path = predictions / "surfaces" / f"{terrain.artifact_stem(query_id)}.npz"
        with np.load(surface_path) as saved:
            displayed = {method: saved[method].copy() for method in selected}
        for axis, method in zip(axes.flat[1:1 + len(selected)], selected, strict=True):
            axis.imshow(target_bg, extent=target_extent)
            values = displayed[method]
            transform = Affine(*meta["target_transform"])
            left, top = meta["canvas_origin_xy_px"]
            stride = meta["stride_px"]
            right = left + values.shape[1] * stride
            bottom = top + values.shape[0] * stride
            x0, y0 = transform * (left, top)
            x1, y1 = transform * (right, bottom)
            finite = values[np.isfinite(values)]
            if finite.size:
                lo, hi = np.quantile(finite, (0.05, 0.995))
                overlay = np.ma.masked_where(~np.isfinite(values), values)
                axis.imshow(overlay, extent=(x0, x1, y1, y0), cmap="inferno",
                            alpha=0.46, vmin=lo, vmax=hi)
            prediction = np.asarray(lookup[method][query_id]["prediction_xy_m"])
            axis.plot(*truth_xy, "+", color="#268cff", ms=14, mew=2, label="Truth")
            axis.plot(*prediction, "x", color="#ff3030", ms=12, mew=2,
                      label="Prediction")
            extent_m = radius_m + 12.0
            axis.set_xlim(source[0] - extent_m, source[0] + extent_m)
            axis.set_ylim(source[1] - extent_m, source[1] + extent_m)
            axis.set_title(terrain.METHOD_LABELS[method])
            axis.legend(fontsize=8)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(
            0.04, 0.9, "Blue +: reference endpoint\nRed ×: predicted endpoint\n"
            "Heatmap: model score over the full search radius",
            va="top", fontsize=12,
        )
        fig.suptitle(f"{title} — {query_id}; search radius {radius_m:g} m")
        safe_id = "".join(char if char.isalnum() or char in "-_" else "_"
                          for char in query_id)
        path = output / f"{safe_id}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        files.append(path)
        print(json.dumps({"panel": index + 1, "total": len(indexed),
                          "query_id": query_id}), flush=True)
    terrain.make_contact_sheet(files[:12], output / "contact_sheet_first12.png")
    terrain.dump(output / "manifest.json", {
        "records": len(files), "radius_m": radius_m,
        "files": [str(path) for path in files],
    })
