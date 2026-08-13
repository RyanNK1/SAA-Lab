"""Verify the local environment before building or deploying.

    python scripts/check_env.py

Checks the running Python against .python-version, that a virtualenv is
active, that every required package imports, and prints exact pins for
requirements.txt resolved on this interpreter.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# What the deployed API imports. These are the pins that belong in
# api/requirements.txt, and nothing else should: installing a test client on
# the host would ship a dependency the application never uses.
RUNTIME = ["numpy", "pandas", "scipy", "fastapi", "uvicorn"]

# Needed to build the dataset and run the suite, but not to serve it.
DEVELOPMENT = ["yfinance", "pytest", "httpx2"]


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
                f"Either rebuild the venv on {pinned}, or write {full} into "
                f".python-version."
            )
    else:
        problems.append(".python-version is missing")

    def survey(names: list[str], heading: str, required: bool) -> dict[str, str]:
        print(f"\n{heading}:")
        found: dict[str, str] = {}
        for name in names:
            try:
                module = importlib.import_module(name)
                found[name] = getattr(module, "__version__", "unknown")
                print(f"  {name:10s} {found[name]}")
            except ImportError:
                print(f"  {name:10s} MISSING")
                if required:
                    problems.append(f"{name} is not installed in this environment")
        return found

    runtime = survey(RUNTIME, "Runtime packages", required=True)
    survey(DEVELOPMENT, "Development packages", required=False)

    if len(runtime) == len(RUNTIME):
        print("\nPaste these into api/requirements.txt, replacing the bare names.")
        print("They are resolved on THIS interpreter, which is the whole point:")
        print("pins from a different Python version can lack a wheel on the host,")
        print("and the fallback is a source build that needs a compiler.\n")
        for name in RUNTIME:
            # uvicorn is installed with its optional extras, and those extras
            # supply the production event loop and HTTP parser. Pinning the
            # bare name would quietly drop them.
            suffix = "[standard]" if name == "uvicorn" else ""
            print(f"  {name}{suffix}=={runtime[name]}")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. {problem}")
        return 1

    print("Environment is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
