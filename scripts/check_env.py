"""Check the local environment is internally consistent before deploying.

    python scripts/check_env.py

Verifies that the running Python matches `.python-version`, that a virtual
environment is active, that every import the project needs resolves, and
prints the exact pins to put in requirements.txt.

This exists because the previous project lost a deploy to a version mismatch:
pins resolved against one Python version, deployed onto another, no prebuilt
wheel available, source build failed for want of a Fortran compiler. The check
is thirty seconds; the failure mode is an afternoon.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["numpy", "pandas", "scipy", "yfinance", "pytest"]


def main() -> int:
    problems: list[str] = []
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    full = f"{running}.{sys.version_info.micro}"

    print(f"Python running:  {full}")
    print(f"Executable:      {sys.executable}")

    in_venv = sys.prefix != sys.base_prefix
    print(f"Virtualenv:      {'active' if in_venv else 'NOT ACTIVE'}")
    if not in_venv:
        problems.append(
            "No virtualenv active. Run: source .venv/bin/activate "
            "(Windows: .venv\\Scripts\\activate)"
        )

    version_file = ROOT / ".python-version"
    if version_file.exists():
        pinned = version_file.read_text().strip()
        print(f".python-version: {pinned}")
        if not pinned.startswith(running):
            problems.append(
                f".python-version says {pinned} but you are running {full}. "
                f"Deploying on a version you never tested is how the last "
                f"project's deploy failed. Either rebuild the venv on {pinned}, "
                f"or write {full} into .python-version."
            )
    else:
        problems.append(".python-version is missing")

    print("\nPackages:")
    found: dict[str, str] = {}
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            found[name] = version
            print(f"  {name:10s} {version}")
        except ImportError:
            print(f"  {name:10s} MISSING")
            problems.append(f"{name} is not installed in this environment")

    if len(found) == len(REQUIRED):
        print("\nPins for api/requirements.txt (resolved on THIS interpreter):")
        for name in REQUIRED:
            print(f"  {name}=={found[name]}")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        return 1

    print("Environment is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
