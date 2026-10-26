"""
terrdiag.io.panel — Generic panel I/O.

Two public functions:
    - load_panel(path, schema, outcomes): read CSV or Parquet, validate
      against the schema, return a typed DataFrame.
    - write_panel(panel, path, schema, outcomes): validate then persist.

CSV with utf-8-sig encoding is the default format. Parquet supported via
pyarrow if available.

Reconciliation of raw certificate data (the ANP/RenovaBio specific concept)
is NOT here — it lives in case_studies/renovabio/. terrdiag only consumes
panels that already meet PanelSchema.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from terrdiag.audit.hashing import hash_dataframe
from terrdiag.audit.logging import run_log
from terrdiag.schema import PanelSchema, ValidationReport, validate_panel
from terrdiag.specs import OutcomeSpec


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        return "csv"
    if suffix in (".parquet", ".pq"):
        return "parquet"
    raise ValueError(
        f"Unsupported panel file format: {suffix!r}. Use .csv or .parquet."
    )


def load_panel(
    path: str | Path,
    schema: PanelSchema | None = None,
    outcomes: Iterable[OutcomeSpec] = (),
    audit: bool = True,
    raise_on_invalid: bool = True,
) -> pd.DataFrame:
    """Load a panel from CSV or Parquet and validate against the schema.

    Parameters
    ----------
    path : str | Path
        Path to the panel file (.csv, .tsv, .parquet, .pq).
    schema : PanelSchema | None
        Schema to validate against. Defaults to PanelSchema() (entity_id,
        cohort, X, Y, geo optional).
    outcomes : iterable of OutcomeSpec
        Outcome specs whose required columns must be present.
    audit : bool
        Emit a run-log entry. Default True.
    raise_on_invalid : bool
        If True, validation errors raise ValueError. If False, warnings
        are emitted but the DataFrame is returned anyway.

    Returns
    -------
    pd.DataFrame
        Loaded panel.

    Raises
    ------
    ValueError
        If validation fails and raise_on_invalid is True.
    """
    t0 = time.time()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Panel file not found: {path}")
    schema = schema or PanelSchema()

    fmt = _detect_format(path)
    # Force string dtype on identifier-like columns to prevent pandas from
    # coercing zero-padded CNPJs to floats or numeric cohort labels to ints.
    str_cols = {schema.entity_id_column: str, schema.cohort_column: str}
    if fmt == "csv":
        panel = pd.read_csv(
            path,
            dtype=str_cols,
            encoding="utf-8-sig",
            sep="\t" if path.suffix.lower() == ".tsv" else ",",
        )
    else:
        panel = pd.read_parquet(path)
        for col in str_cols:
            if col in panel.columns:
                panel[col] = panel[col].astype(str)

    report = validate_panel(panel, schema=schema, outcomes=list(outcomes))

    duration = time.time() - t0

    if audit:
        run_log.record(
            step="io.load_panel",
            duration_s=duration,
            params={
                "path": str(path),
                "format": fmt,
                "schema": _schema_summary(schema),
                "outcomes": [s.name for s in outcomes],
            },
            output_rows=len(panel),
            output_hash=hash_dataframe(panel),
            notes=(
                f"validation {'PASSED' if report.is_valid else 'FAILED'}; "
                f"{report.n_entities} unique entities; "
                f"{len(report.errors)} errors, {len(report.warnings)} warnings"
            ),
        )

    if raise_on_invalid:
        report.raise_if_invalid()
    elif not report.is_valid:
        import warnings as _w

        for e in report.errors:
            _w.warn(f"[load_panel] {e}", stacklevel=2)

    return panel


def write_panel(
    panel: pd.DataFrame,
    path: str | Path,
    schema: PanelSchema | None = None,
    outcomes: Iterable[OutcomeSpec] = (),
    audit: bool = True,
    validate: bool = True,
) -> Path:
    """Persist a panel to CSV or Parquet, validating before writing.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel to write.
    path : str | Path
        Output path; format inferred from suffix.
    schema, outcomes : as in `load_panel`.
    audit : bool
        Emit a run-log entry.
    validate : bool
        If True, validate before writing and raise on failure.

    Returns
    -------
    Path
        The path written.
    """
    t0 = time.time()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = schema or PanelSchema()

    if validate:
        report = validate_panel(panel, schema=schema, outcomes=list(outcomes))
        report.raise_if_invalid()

    fmt = _detect_format(path)
    if fmt == "csv":
        panel.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        panel.to_parquet(path, index=False)

    duration = time.time() - t0

    if audit:
        run_log.record(
            step="io.write_panel",
            duration_s=duration,
            params={
                "path": str(path),
                "format": fmt,
                "schema": _schema_summary(schema),
            },
            input_rows=len(panel),
            input_hash=hash_dataframe(panel),
            notes=f"wrote {path.stat().st_size} bytes",
        )

    return path


def _schema_summary(schema: PanelSchema) -> dict[str, object]:
    """Compact dict representation of a schema for audit logs."""
    return {
        "entity_id_column": schema.entity_id_column,
        "cohort_column": schema.cohort_column,
        "geo_columns": list(schema.geo_columns),
        "geo_crs_epsg": schema.geo_crs_epsg,
        "require_geo": schema.require_geo,
        "require_unique_entity": schema.require_unique_entity,
    }


# Re-export validate_panel from schema for convenience
__all__ = ["load_panel", "write_panel", "validate_panel"]
