"""Checkpoint resolution and loading.

Real-world SR checkpoints are inconsistent: some are a bare state_dict, some wrap
it in ``params``, some in ``params_ema``, some in ``state_dict``, and some prefix
every key with ``module.`` from a DataParallel run. This module normalises all of
that so architectures never see it.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from ..core.config import get_settings
from ..core.errors import WeightsCorruptError, WeightsNotFoundError
from ..core.types import ModelSpec

log = logging.getLogger(__name__)

# Checked in order; first hit wins.
_STATE_DICT_KEYS = ("params_ema", "params", "state_dict", "model", "net_g", "weight")


def resolve_weight_path(spec: ModelSpec) -> Path:
    """Locate the checkpoint for a spec, or explain how to fetch it."""
    if spec.weights is None:
        raise WeightsNotFoundError(spec.id, "<spec declares no weights>")

    path = get_settings().weight_path(spec.weights.filename)
    if not path.exists():
        raise WeightsNotFoundError(spec.id, str(path))
    return path


def onnx_graph_path(spec: ModelSpec) -> Path:
    """Where the exported ONNX graph for a torch model lives."""
    return get_settings().checkpoint_dir / "onnx" / f"{spec.id}.onnx"


def is_runnable(spec: ModelSpec) -> bool:
    """Can this model actually run, here, right now?

    Not the same question as "is its .pth on disk", which is what this used to check —
    and that mattered the moment the shipped app stopped containing PyTorch. The
    packaged build ships only ONNX graphs, so six of the eight models had no .pth beside
    them and were reported as "not downloaded" in the UI while running perfectly well.
    A status that says a working model is missing is worse than no status.

    A model is runnable if *some* backend on this machine can load it:

      - a native ONNX model needs its graph, and nothing else;
      - a torch model needs either its checkpoint (if torch is installed) or an
        exported graph (if it is not).
    """
    from importlib.util import find_spec

    from ..core.registry import ONNX_GRAPH

    if spec.weights is None:
        return True  # nothing to fetch

    checkpoint = get_settings().weight_path(spec.weights.filename)

    if spec.architecture == ONNX_GRAPH:
        return checkpoint.exists()

    if find_spec("torch") is not None and checkpoint.exists():
        return True

    return onnx_graph_path(spec).exists()


def verify_checksum(path: Path, expected_sha256: str | None) -> None:
    """Guard against a truncated or tampered download."""
    if not expected_sha256:
        return

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise WeightsCorruptError(str(path), expected_sha256, actual)


def extract_state_dict(checkpoint: Any, preferred_key: str | None = None) -> dict[str, Any]:
    """Dig the tensor dict out of whatever shape the release used."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is a {type(checkpoint).__name__}, expected a dict.")

    if preferred_key:
        if preferred_key not in checkpoint:
            raise KeyError(
                f"Checkpoint has no key {preferred_key!r}. Keys: {sorted(checkpoint)[:10]}"
            )
        state = checkpoint[preferred_key]
    else:
        state = next(
            (checkpoint[k] for k in _STATE_DICT_KEYS if k in checkpoint),
            checkpoint,  # already a bare state_dict
        )

    if not isinstance(state, dict):
        raise TypeError(f"Resolved state dict is a {type(state).__name__}, expected a dict.")

    return _strip_prefixes(state)


def _strip_prefixes(state: dict[str, Any]) -> dict[str, Any]:
    """Remove DataParallel/compile wrappers that would break key matching."""
    for prefix in ("module.", "_orig_mod."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix) :]: v for k, v in state.items()}
    return state


def load_state_dict(spec: ModelSpec) -> dict[str, Any]:
    """Full path: resolve -> verify -> torch.load -> normalise."""
    import torch

    path = resolve_weight_path(spec)
    assert spec.weights is not None  # guaranteed by resolve_weight_path
    verify_checksum(path, spec.weights.sha256)

    # weights_only=True refuses to unpickle arbitrary code — these files come off
    # the public internet, so this is not optional.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = extract_state_dict(checkpoint, spec.state_dict_key)

    log.debug("loaded %d tensors for %s from %s", len(state), spec.id, path.name)
    return state
