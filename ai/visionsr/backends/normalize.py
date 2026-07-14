"""Input/output normalisation, shared by every backend.

One implementation, used by torch, ONNX Runtime and OpenVINO alike, so a model
cannot behave differently depending on which execution provider happened to be
available. Numpy rather than torch, because the ONNX and OpenVINO backends never
import torch at all.

The whole reason this exists as a declared, per-model property rather than an
assumption baked into a pipeline: getting it wrong does not raise. It returns a
plausible image with half its tonal range missing, or a segmentation mask that is
uniformly grey — the kind of bug you only find by looking.
"""

from __future__ import annotations

import numpy as np

from ..core.types import ModelSpec


def to_network(batch: np.ndarray, spec: ModelSpec) -> np.ndarray:
    """[0,1] NHWC RGB -> whatever the network was trained on."""
    mean = np.asarray(spec.input_mean, dtype=np.float32)
    std = np.asarray(spec.input_std, dtype=np.float32)

    if np.array_equal(mean, np.zeros(3)) and np.array_equal(std, np.ones(3)):
        return batch

    return (batch - mean) / std


def from_network(out: np.ndarray, spec: ModelSpec) -> np.ndarray:
    """Network output -> [0,1] NHWC.

    Rescale first and clip second, always. Clipping a [-1,1] output to [0,1] before
    undoing the normalisation deletes every value below mid-grey — which is exactly
    the bug that once turned GFPGAN's restored faces into flat pink smears.
    """
    out = out.astype(np.float32, copy=False)

    if spec.output_norm == "affine":
        mean = np.asarray(spec.input_mean, dtype=np.float32)
        std = np.asarray(spec.input_std, dtype=np.float32)
        out = out * std + mean

    elif spec.output_norm == "minmax":
        # Segmentation heads emit unbounded logits whose scale carries no meaning —
        # only the ordering does. Normalising by the map's own extremes is what turns
        # them into an alpha channel.
        low = float(out.min())
        high = float(out.max())
        out = (out - low) / (high - low) if high > low else np.zeros_like(out)

    return np.clip(out, 0.0, 1.0)
