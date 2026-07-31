"""Launch the Dead Bug coach web app.

    python scripts/run_app.py                    # http://127.0.0.1:8000
    python scripts/run_app.py --port 8080
    python scripts/run_app.py --host 0.0.0.0     # reachable from your phone

Three ways in -- camera, YouTube link, uploaded file -- all driven by the same
:class:`deadbug.live.engine.CoachEngine` that ``run_live.py`` uses, so the
browser and the CLI count the same reps on the same clip.

**Using your phone as the camera:** run with ``--host 0.0.0.0`` and open
``http://<this-machine-ip>:8000`` on the phone. Browsers only grant camera
access over HTTPS or to localhost, so a plain LAN address will be refused --
either put a TLS proxy in front, or record on the phone and use the upload tab,
which needs no permissions at all.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _lan_ip() -> str | None:
    """Best-effort local address, for the "open this on your phone" hint."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.2)
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--reload", action="store_true", help="auto-restart on code changes")
    args = ap.parse_args()

    import uvicorn

    from deadbug.webapp.server import create_app

    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"\n  Dead Bug Coach   http://{shown}:{args.port}")
    if args.host == "0.0.0.0":
        ip = _lan_ip()
        if ip:
            print(f"  on this network   http://{ip}:{args.port}")
        print("  note: the camera tab needs HTTPS or localhost -- over a plain LAN")
        print("        address the browser will refuse. Use the upload tab there.")
    print()

    uvicorn.run(create_app(args.config), host=args.host, port=args.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
