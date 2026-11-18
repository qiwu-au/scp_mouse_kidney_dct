"""Runtime settings shared by the scripts; no packages are modified.

Each script calls ``ensure_scanpy_environment(__file__)`` when it is started directly. The call keeps the
Matplotlib and Numba caches in ``.cache/`` beside this file, unless ``MPLCONFIGDIR`` or ``NUMBA_CACHE_DIR`` is set
already. It does not select or restart a Python interpreter: start the scripts with the Python environment
described in ``README.md``.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def configure_runtime_environment() -> None:
    """Keep library caches beside the code unless they are configured already."""
    cache_root = PROJECT_ROOT / ".cache"
    for name, subdirectory in (
        ("MPLCONFIGDIR", "matplotlib"),
        ("NUMBA_CACHE_DIR", "numba"),
    ):
        if not os.environ.get(name):
            cache_directory = cache_root / subdirectory
            cache_directory.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(cache_directory)


def ensure_scanpy_environment(script_path: str | Path) -> None:
    """Called by every script before it imports the analysis packages."""
    configure_runtime_environment()
