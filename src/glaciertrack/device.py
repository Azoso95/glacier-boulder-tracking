"""Centralized CPU, CUDA, and MPS device selection."""

from __future__ import annotations

from typing import Any

import torch


class DeviceError(RuntimeError):
    """Raised when an explicitly requested accelerator is unavailable."""


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, ``cuda[:index]``, or ``mps`` to a torch device."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise DeviceError(
                "CUDA was requested but is not available in this PyTorch installation"
            )
        device = torch.device(normalized)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise DeviceError(
                f"CUDA device {device.index} was requested, but only "
                f"{torch.cuda.device_count()} device(s) are visible"
            )
        return device
    if normalized == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise DeviceError("MPS was requested but is not available")
        return torch.device("mps")
    raise DeviceError(f"unsupported device request: {requested!r}")


def device_summary(requested: str = "auto") -> dict[str, Any]:
    """Return serializable runtime diagnostics without allocating accelerator memory."""

    selected = resolve_device(requested)
    summary: dict[str, Any] = {
        "requested": requested,
        "selected": str(selected),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if selected.type == "cuda":
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        summary["cuda_device_name"] = torch.cuda.get_device_name(index)
        summary["cuda_runtime"] = torch.version.cuda
    return summary
