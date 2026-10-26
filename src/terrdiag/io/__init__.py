"""
terrdiag.io — Generic panel I/O.

Public functions:
    - load_panel: read a CSV/Parquet panel, validate, return DataFrame.
    - write_panel: validate and persist a panel.
    - validate_panel: re-export from schema for convenience.
    - resolve_geo: attach X/Y coordinates to a panel.

Case-specific I/O (e.g. ANP RenovaBio parser) lives in case_studies/.
"""

from terrdiag.io.geo import resolve_geo
from terrdiag.io.panel import load_panel, validate_panel, write_panel

__all__ = [
    "load_panel",
    "resolve_geo",
    "validate_panel",
    "write_panel",
]
