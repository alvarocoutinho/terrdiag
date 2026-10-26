"""
terrdiag.audit.reproducibility — Run metadata emission.

`write_metadata(path)` writes a JSON file capturing:
    - terrdiag version
    - Python version
    - timestamp
    - the current RunLog as dict
    - hashes of declared input files
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from terrdiag._version import __version__
from terrdiag.audit.hashing import hash_file
from terrdiag.audit.logging import RunLog, run_log


def write_metadata(
    path: str | Path,
    inputs: dict[str, str | Path] | None = None,
    extra: dict[str, Any] | None = None,
    log: RunLog | None = None,
) -> Path:
    """Write a run metadata JSON file.

    Parameters
    ----------
    path : str | Path
        Output filepath (typically 'run_metadata.json' next to outputs).
    inputs : dict[str, str|Path] | None
        Named input files; each will be hashed and recorded.
    extra : dict | None
        Additional fields to merge (e.g. recipe info).
    log : RunLog | None
        Run log to embed. Defaults to the module-level run_log.

    Returns
    -------
    Path
        The path written.
    """
    log = log or run_log
    path = Path(path)

    payload: dict[str, Any] = {
        "terrdiag_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": {},
        "run_log": log.to_dict(),
    }

    if inputs:
        for name, fpath in inputs.items():
            fpath = Path(fpath)
            if fpath.exists():
                payload["inputs"][name] = {
                    "path": str(fpath),
                    "size_bytes": fpath.stat().st_size,
                    "sha256_16": hash_file(fpath),
                }
            else:
                payload["inputs"][name] = {"path": str(fpath), "missing": True}

    if extra:
        payload["extra"] = extra

    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
