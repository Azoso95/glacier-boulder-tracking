"""Materialize fold-specific patch caches by sample ID."""

from pathlib import Path
from typing import Any

import h5py


class SplitError(RuntimeError):
    """Raised when spatial splits cannot be produced without leakage."""


def _subset_hdf5(source_path: Path, destination_path: Path, sample_ids: list[str]) -> None:
    with h5py.File(source_path, "r") as source:
        source_ids = source["sample_ids"].asstr()[:].tolist()
        index_by_id = {str(sample_id): index for index, sample_id in enumerate(source_ids)}
        missing = set(sample_ids).difference(index_by_id)
        if missing:
            raise SplitError(f"manifest samples are absent from HDF5: {sorted(missing)[:5]}")
        indices = sorted(index_by_id[sample_id] for sample_id in sample_ids)
        with h5py.File(destination_path, "w") as destination:
            for key, value in source.attrs.items():
                destination.attrs[key] = value
            destination.attrs["sample_count"] = len(indices)
            for name, dataset in source.items():
                shape = (len(indices), *dataset.shape[1:])
                options: dict[str, Any] = {}
                if dataset.chunks is not None:
                    # Preparation uses an extendable leading dimension with a
                    # 256-sample chunk. Split caches have fixed dimensions and
                    # may contain fewer than 256 samples, for which HDF5 rejects
                    # the unchanged source chunk shape.
                    options["chunks"] = tuple(
                        min(chunk, dimension)
                        for chunk, dimension in zip(
                            dataset.chunks, shape, strict=True
                        )
                    )
                if dataset.compression is not None:
                    options["compression"] = dataset.compression
                    options["compression_opts"] = dataset.compression_opts
                if dataset.shuffle:
                    options["shuffle"] = True
                target = destination.create_dataset(
                    name, shape=shape, dtype=dataset.dtype, **options
                )
                for start in range(0, len(indices), 64):
                    batch = indices[start : start + 64]
                    target[start : start + len(batch)] = dataset[batch]
