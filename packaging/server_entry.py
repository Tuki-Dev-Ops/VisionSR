"""Entry point for the frozen backend.

`uvicorn backend.app.main:app` is a fine way to start a server from a checkout, and a
bad way to start one from an executable: it re-imports the app by string, which a
frozen build resolves against its own bundled module table rather than the filesystem,
and it installs signal handlers and a reloader the desktop shell does not want.

So the frozen build runs uvicorn programmatically against an app object it already
holds. Same server, no import magic.

    visionsr-server.exe --host 127.0.0.1 --port 51234
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys


def main() -> int:
    # Windows has no fork: a frozen child re-executes the .exe, which re-runs this
    # module, which spawns another child... The guard is one line and the failure
    # without it is an exponential fork bomb of your own application.
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(prog="visionsr-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args()

    import uvicorn

    from backend.app.main import app

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        # No reloader, no worker subprocesses. The desktop shell owns this process's
        # lifetime and kills it by pid; anything that forks escapes that.
        workers=1,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
