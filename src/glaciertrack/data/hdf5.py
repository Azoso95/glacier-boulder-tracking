"""Read-only, worker-safe access to the versioned HDF5 pair cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from glaciertrack.data.augmentation import ImageAugmentation, augment_tracking_pair
from glaciertrack.data.prompts import (
    PromptApproach,
    PromptAugmentation,
    PromptConfig,
    render_source_prompt,
    render_target_heatmap,
)
from glaciertrack.data.resolution import simulate_effective_resolution

SCHEMA_NAME = "glaciertrack-pairs"
SCHEMA_VERSION = 1
REQUIRED_DATASETS = (
    "reference_images",
    "search_images",
    "source_points_px",
    "target_points_px",
    "source_valid",
    "match_present",
    "sample_ids",
)


def validate_hdf5(path: str | Path) -> dict[str, Any]:
    """Validate cache structure and return a concise serializable summary."""

    cache_path = Path(path)
    if not cache_path.is_file():
        raise OSError(f"HDF5 cache does not exist: {cache_path}")
    with h5py.File(cache_path, "r") as handle:
        if handle.attrs.get("schema_name") != SCHEMA_NAME:
            raise ValueError(f"expected schema_name={SCHEMA_NAME!r}")
        if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"expected schema_version={SCHEMA_VERSION}")
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ValueError(f"missing required HDF5 datasets: {', '.join(missing)}")
        lengths = {name: len(handle[name]) for name in REQUIRED_DATASETS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"HDF5 datasets have inconsistent lengths: {lengths}")
        reference_shape = tuple(int(value) for value in handle["reference_images"].shape[1:])
        search_shape = tuple(int(value) for value in handle["search_images"].shape[1:])
        if len(reference_shape) != 3 or len(search_shape) != 3:
            raise ValueError("reference_images and search_images must have NCHW shape")
        match_present = np.asarray(handle["match_present"], dtype=np.bool_)
        positive_samples = int(match_present.sum())
        negative_samples = int(match_present.size - positive_samples)
        return {
            "path": str(cache_path),
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "samples": next(iter(lengths.values())),
            "manifest_sha256": str(handle.attrs.get("manifest_sha256", "")),
            "tensor_gsd_m_per_px": float(handle.attrs["tensor_gsd_m_per_px"]),
            "reference_shape_chw": reference_shape,
            "search_shape_chw": search_shape,
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
        }


class HDF5PairDataset(Dataset[dict[str, Any]]):
    """Lazily opens one read-only HDF5 handle in each DataLoader worker."""

    def __init__(
        self,
        path: str | Path,
        *,
        prompt_config: PromptConfig | None = None,
        prompt_augmentation: PromptAugmentation | None = None,
        simulated_effective_gsd_m_per_px: tuple[float, ...] | None = None,
        prompt_approaches: tuple[PromptApproach, ...] = ("auto",),
        prompt_approach_probabilities: tuple[float, ...] | None = None,
        image_augmentation: ImageAugmentation | None = None,
        sample_manifest: str | Path | None = None,
        seed: int = 0,
        resample_prompt_each_epoch: bool = True,
    ) -> None:
        # Define this before validation so cleanup remains safe when
        # construction fails partway through.
        self._handle: h5py.File | None = None
        self.path = Path(path)
        summary = validate_hdf5(self.path)
        self._length = int(summary["samples"])
        self.tensor_gsd_m_per_px = float(summary["tensor_gsd_m_per_px"])
        self.reference_shape_chw = tuple(summary["reference_shape_chw"])
        self.search_shape_chw = tuple(summary["search_shape_chw"])
        self.positive_samples = int(summary["positive_samples"])
        self.negative_samples = int(summary["negative_samples"])
        self.group_ids: tuple[str, ...] | None = None
        self.group_seed_indices: tuple[int, ...] | None = None
        self.match_flags: tuple[bool, ...] | None = None
        if sample_manifest is not None:
            self.group_ids, self.match_flags = _load_sample_groups(
                self.path, Path(sample_manifest)
            )
            group_index: dict[str, int] = {}
            self.group_seed_indices = tuple(
                group_index.setdefault(group_id, len(group_index))
                for group_id in self.group_ids
            )
        self.prompt_config = prompt_config or PromptConfig()
        if self.prompt_config.backend == "sam2":
            from glaciertrack.data.sam2_masks import warm_hdf5_masks

            warm_hdf5_masks(self.path, self.prompt_config)
        self.prompt_augmentation = prompt_augmentation or PromptAugmentation(enabled=False)
        self.simulated_effective_gsd_m_per_px = simulated_effective_gsd_m_per_px
        self.prompt_approaches = prompt_approaches
        self.prompt_approach_probabilities = _validated_probabilities(
            prompt_approaches, prompt_approach_probabilities
        )
        self.image_augmentation = image_augmentation or ImageAugmentation(enabled=False)
        self.seed = int(seed)
        self.epoch = 0
        self.resample_prompt_each_epoch = bool(resample_prompt_each_epoch)
        if simulated_effective_gsd_m_per_px is not None and (
            not simulated_effective_gsd_m_per_px
            or min(simulated_effective_gsd_m_per_px) <= 0
        ):
            raise ValueError("simulated effective GSD choices must be positive")

    def _open(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        handle = self._open()
        reference = _image_tensor(handle["reference_images"][index])
        search = _image_tensor(handle["search_images"][index])
        generator = self._sample_generator(index)
        reference_current_gsd = _optional_scalar(
            handle, "reference_effective_gsd_m_per_px", index
        ) or self.tensor_gsd_m_per_px
        search_current_gsd = _optional_scalar(
            handle, "target_effective_gsd_m_per_px", index
        ) or self.tensor_gsd_m_per_px
        reference_gsd = self._sample_effective_gsd(reference_current_gsd, generator)
        search_gsd = self._sample_effective_gsd(search_current_gsd, generator)
        reference = simulate_effective_resolution(
            reference,
            current_effective_gsd_m_per_px=reference_current_gsd,
            target_effective_gsd_m_per_px=reference_gsd,
            tensor_gsd_m_per_px=self.tensor_gsd_m_per_px,
        )
        search = simulate_effective_resolution(
            search,
            current_effective_gsd_m_per_px=search_current_gsd,
            target_effective_gsd_m_per_px=search_gsd,
            tensor_gsd_m_per_px=self.tensor_gsd_m_per_px,
        )
        sample_id = handle["sample_ids"].asstr()[index]
        source_point = torch.as_tensor(
            handle["source_points_px"][index], dtype=torch.float32
        )
        target_point = torch.as_tensor(handle["target_points_px"][index], dtype=torch.float32)
        match_present = bool(handle["match_present"][index])
        major_axis = _optional_scalar(handle, "source_major_axis_m", index)
        minor_axis = _optional_scalar(handle, "source_minor_axis_m", index)
        orientation = _optional_scalar(handle, "source_orientation_deg", index)
        approach = self._sample_prompt_approach(generator)
        # Parametric prompts may be rendered before augmentation and transformed
        # exactly with the image.  SAM2 must instead see the final augmented RGB
        # values; a transformed mask inferred from the original photograph does
        # not test robustness to synthetic shade, blur, or sensor noise.
        regenerate_sam2_prompt = (
            self.prompt_config.backend == "sam2" and self.image_augmentation.enabled
        )
        source_prompt = (
            torch.zeros((1, *reference.shape[-2:]), dtype=reference.dtype)
            if regenerate_sam2_prompt
            else render_source_prompt(
                tuple(reference.shape[-2:]),
                source_point,
                gsd_m_per_px=self.tensor_gsd_m_per_px,
                major_axis_m=major_axis,
                minor_axis_m=minor_axis,
                orientation_deg=orientation,
                config=self.prompt_config,
                image=reference,
                augmentation=self.prompt_augmentation,
                approach=approach,
                generator=generator,
            )
        )
        target_heatmap = render_target_heatmap(
            tuple(search.shape[-2:]),
            target_point,
            match_present=match_present,
            gsd_m_per_px=self.tensor_gsd_m_per_px,
            sigma_m=self.prompt_config.target_sigma_m,
        )
        (
            reference,
            source_prompt,
            source_point,
            search,
            target_heatmap,
            target_point,
            geometry,
        ) = augment_tracking_pair(
            reference,
            source_prompt,
            source_point,
            search,
            target_heatmap,
            target_point,
            config=self.image_augmentation,
            generator=generator,
        )
        if regenerate_sam2_prompt:
            source_prompt = render_source_prompt(
                tuple(reference.shape[-2:]),
                source_point,
                gsd_m_per_px=self.tensor_gsd_m_per_px,
                major_axis_m=major_axis,
                minor_axis_m=minor_axis,
                orientation_deg=orientation,
                config=self.prompt_config,
                image=reference,
                augmentation=self.prompt_augmentation,
                approach=approach,
                generator=generator,
            )
        return {
            "sample_id": sample_id,
            "group_id": (
                self.group_ids[index] if self.group_ids is not None else sample_id
            ),
            "reference_image": reference,
            "search_image": search,
            "source_point_px": source_point,
            "source_prompt": source_prompt,
            "source_prompt_approach": (
                "sam2" if self.prompt_config.backend == "sam2" else approach
            ),
            "target_point_px": target_point,
            "target_heatmap": target_heatmap,
            "source_valid": bool(handle["source_valid"][index]),
            "match_present": match_present,
            "localization_valid": match_present,
            "augmentation_rotation_deg": geometry.quarter_turns * 90,
            "augmentation_horizontal_flip": geometry.horizontal_flip,
            "augmentation_vertical_flip": geometry.vertical_flip,
            "reference_effective_gsd_m_per_px": reference_gsd,
            "target_effective_gsd_m_per_px": search_gsd,
        }

    def _sample_effective_gsd(
        self, current: float, generator: torch.Generator
    ) -> float:
        if self.simulated_effective_gsd_m_per_px is None:
            return current
        choices = [
            value
            for value in self.simulated_effective_gsd_m_per_px
            if value >= current
        ]
        if not choices:
            return current
        return float(choices[int(torch.randint(len(choices), (), generator=generator))])

    def set_epoch(self, epoch: int) -> None:
        """Set the augmentation epoch before creating a DataLoader iterator."""

        self.epoch = int(epoch)

    def _sample_generator(self, index: int) -> torch.Generator:
        epoch = self.epoch if self.resample_prompt_each_epoch else 0
        seed_index = (
            self.group_seed_indices[index]
            if self.group_seed_indices is not None
            else index
        )
        sample_seed = self.seed + 1_000_003 * epoch + 97_409 * seed_index
        return torch.Generator().manual_seed(sample_seed % (2**63 - 1))

    def _sample_prompt_approach(self, generator: torch.Generator) -> PromptApproach:
        weights = torch.tensor(self.prompt_approach_probabilities, dtype=torch.float64)
        index = int(torch.multinomial(weights, 1, generator=generator))
        return self.prompt_approaches[index]

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        self.close()


def _image_tensor(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(array))
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.float32).div_(255.0)
    if not tensor.is_floating_point():
        raise TypeError(f"unsupported image dtype: {tensor.dtype}")
    return tensor.to(torch.float32)


def _optional_scalar(handle: h5py.File, name: str, index: int) -> float | None:
    if name not in handle:
        return None
    value = float(handle[name][index])
    return value if np.isfinite(value) else None


def _validated_probabilities(
    approaches: tuple[PromptApproach, ...], probabilities: tuple[float, ...] | None
) -> tuple[float, ...]:
    if not approaches:
        raise ValueError("at least one prompt approach is required")
    if any(value not in {"auto", "ellipse", "gaussian_point"} for value in approaches):
        raise ValueError(f"unsupported prompt approaches: {approaches!r}")
    if probabilities is None:
        return tuple(1 / len(approaches) for _ in approaches)
    if len(probabilities) != len(approaches):
        raise ValueError("prompt approach probabilities must match the approaches")
    if any(value < 0 for value in probabilities) or sum(probabilities) <= 0:
        raise ValueError("prompt approach probabilities must be non-negative and non-zero")
    total = float(sum(probabilities))
    return tuple(float(value) / total for value in probabilities)


def _load_sample_groups(
    cache_path: Path, manifest_path: Path
) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    if not manifest_path.is_file():
        raise OSError(f"sample manifest does not exist: {manifest_path}")
    frame = pd.read_parquet(
        manifest_path, columns=["sample_id", "track_id", "match_present"]
    )
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"sample manifest contains duplicate sample_id values: {manifest_path}")
    metadata = {
        str(row.sample_id): (str(row.track_id), bool(row.match_present))
        for row in frame.itertuples(index=False)
    }
    with h5py.File(cache_path, "r") as handle:
        sample_ids = tuple(str(value) for value in handle["sample_ids"].asstr()[:])
        cache_matches = tuple(bool(value) for value in handle["match_present"][:])
    missing = [sample_id for sample_id in sample_ids if sample_id not in metadata]
    if missing:
        raise ValueError(
            f"sample manifest is missing {len(missing)} cache samples; first={missing[0]!r}"
        )
    groups = tuple(metadata[sample_id][0] for sample_id in sample_ids)
    manifest_matches = tuple(metadata[sample_id][1] for sample_id in sample_ids)
    if manifest_matches != cache_matches:
        raise ValueError("sample manifest match_present labels disagree with HDF5 cache")
    return groups, cache_matches
