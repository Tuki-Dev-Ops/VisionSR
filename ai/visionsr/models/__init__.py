"""Network architectures.

Importing this package is what populates the architecture registry: each module
below carries an ``@register_architecture(...)`` decorator that only fires on
import. Adding a network means dropping a module here and listing it in
``_ARCH_MODULES`` — no engine code changes.
"""

from __future__ import annotations

import importlib
import logging
from importlib.util import find_spec

log = logging.getLogger(__name__)

# Modules whose import side effect is registering an architecture.
_ARCH_MODULES: tuple[str, ...] = (
    ".sr.rrdbnet",
    ".sr.srvgg",
    ".face.gfpgan",
)


def load_architectures() -> None:
    """Import every architecture module. Idempotent.

    An architecture is a torch ``nn.Module``, so this is a no-op where torch is not
    installed — which is the *shipped desktop runtime*, deliberately: it runs ONNX
    Runtime on DirectML, at 150MB and with no CUDA dependency, instead of freezing a
    2.5GB torch+CUDA stack into an installer.

    That build does not need architectures at all. The ONNX backend loads graphs
    directly; architectures exist to *produce* those graphs (see scripts/export_onnx.py),
    which is a development and build-time job.

    The check is on torch's presence, not a bare ``except ImportError`` — swallowing
    every import error here would hide a genuinely broken architecture module behind a
    model that mysteriously stops being registered.
    """
    if find_spec("torch") is None:
        log.debug(
            "no PyTorch — skipping architecture registration. This is expected in the "
            "packaged app, which runs pre-exported ONNX graphs."
        )
        return

    for module in _ARCH_MODULES:
        importlib.import_module(module, package=__name__)


load_architectures()
