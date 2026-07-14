"""Console setup shared by the CLI and the scripts.

Exists because of a real, reproducible Windows failure: on a console whose code
page is not UTF-8 (cp949 on a Korean install, cp1252 on a Western one, cp932 on a
Japanese one), writing a "✓" raises ``UnicodeEncodeError`` and takes the whole
command down. A progress tick is not worth crashing a batch job for.

Two defences, because either alone is insufficient:

* ``sys.stdout`` is reconfigured to UTF-8 with ``errors="replace"``, so nothing can
  raise regardless of what gets printed.
* The glyphs below degrade to ASCII when the terminal genuinely cannot render them,
  so the output stays readable rather than becoming a field of "?".
"""

from __future__ import annotations

import contextlib
import sys

from rich.console import Console


def _supports(text: str) -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # A pipe that will not take a new encoding. Rich's own fallback still
        # keeps us from crashing.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


_configure_stdout()

_UNICODE = _supports("✓✗·█░")

OK = "✓" if _UNICODE else "+"
FAIL = "✗" if _UNICODE else "x"
SKIP = "·" if _UNICODE else "-"
FULL = "█" if _UNICODE else "#"
EMPTY = "░" if _UNICODE else "."

#: Shared console. `safe_box` makes rich fall back to ASCII table borders on the
#: same terminals that cannot take the glyphs above.
console = Console(safe_box=not _UNICODE)


def bar(value: float, width: int = 20, good_is_high: bool = False) -> str:
    """A coloured 0..1 meter."""
    filled = max(0, min(width, int(value * width)))

    if good_is_high:
        colour = "green" if value > 0.6 else "yellow" if value > 0.35 else "red"
    else:
        colour = "green" if value < 0.3 else "yellow" if value <= 0.6 else "red"

    return f"[{colour}]{FULL * filled}{EMPTY * (width - filled)}[/{colour}] {value:.2f}"
