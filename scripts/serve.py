"""Run the API locally.

    python scripts/serve.py
    python scripts/serve.py --port 8080 --reload

Then open http://127.0.0.1:8000/docs for interactive documentation -- FastAPI
generates it from the schemas, so every endpoint can be tried from a browser
without writing a client first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on edit")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install fastapi 'uvicorn[standard]'")
        return 1

    from api.main import load_panel

    try:
        panel = load_panel()
    except RuntimeError as error:
        print(error)
        return 1

    print(f"dataset: {len(panel)} months, {panel.start:%Y-%m} to {panel.end:%Y-%m}")
    print(f"docs:    http://{args.host}:{args.port}/docs")

    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
