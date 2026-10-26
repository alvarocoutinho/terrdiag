"""
terrdiag.schema — Canonical panel schema and validation.

The PanelSchema is the contract terrdiag exposes to its users: any DataFrame
that meets this schema can be analyzed by Layer 1, Layer 2, and overlay
functions. Library users are responsible for producing panels that meet
the schema; how they do it (parsing xlsx, querying a database, joining
arbitrary CSVs) is outside terrdiag's scope.

Two public objects:
    - PanelSchema: dataclass declaring which columns are required, optional
      column names, geometry expectations.
    - validate_panel(panel, schema, outcomes): returns a ValidationReport
      with errors and warnings. Use .raise_if_invalid() to fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from terrdiag.specs import OutcomeSpec


# ─── Schema ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PanelSchema:
    """Canonical wide-panel schema expected by terrdiag.

    A panel is one row per entity (mill, municipality, plant, ...) with
    longitudinal outcomes encoded as snapshot-suffixed columns.

    Attributes
    ----------
    entity_id_column : str
        Column holding the unique entity identifier (e.g. 'cnpj_clean',
        'cd_ibge', 'school_code').
    cohort_column : str
        Column labelling the cohort each entity belongs to (typically the
        label of its first observable snapshot).
    geo_columns : tuple[str, str]
        Column names for projected X and Y coordinates.
    geo_crs_epsg : int
        EPSG code of the projected CRS used in geo_columns. Default 5880
        (SIRGAS 2000 / Brazil Polyconic, meters).
    require_geo : bool
        If True, validation fails when geo_columns are missing or have nulls.
    require_unique_entity : bool
        If True, the entity_id column must have no duplicates.
    """

    entity_id_column: str = "entity_id"
    cohort_column: str = "cohort"
    geo_columns: tuple[str, str] = ("X", "Y")
    geo_crs_epsg: int = 5880
    require_geo: bool = False
    require_unique_entity: bool = True


# ─── Validation report ────────────────────────────────────────────────────


@dataclass(slots=True)
class ValidationReport:
    """Outcome of `validate_panel`. Falsy when invalid."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_entities: int = 0

    def __bool__(self) -> bool:
        return self.is_valid

    def raise_if_invalid(self) -> None:
        """Raise ValueError listing all errors, if any."""
        if not self.is_valid:
            msg = "Panel schema validation failed:\n  - " + "\n  - ".join(self.errors)
            raise ValueError(msg)

    def summary(self) -> str:
        """Human-readable summary string."""
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"PanelSchema validation: {status}",
            f"  rows: {self.n_rows}, unique entities: {self.n_entities}",
        ]
        if self.errors:
            lines.append(f"  errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


# ─── Validator ────────────────────────────────────────────────────────────


def validate_panel(
    panel: pd.DataFrame,
    schema: PanelSchema | None = None,
    outcomes: Iterable[OutcomeSpec] = (),
) -> ValidationReport:
    """Validate a DataFrame against the canonical schema.

    Parameters
    ----------
    panel : pd.DataFrame
        Wide-format panel to validate.
    schema : PanelSchema | None
        Schema to validate against. Default: `PanelSchema()` with defaults.
    outcomes : iterable of OutcomeSpec
        Outcome specifications whose required columns must exist in the panel.

    Returns
    -------
    ValidationReport
        Object with `.is_valid`, `.errors`, `.warnings`, plus summary methods.
    """
    schema = schema or PanelSchema()
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(panel, pd.DataFrame):
        return ValidationReport(
            is_valid=False,
            errors=[f"panel must be a pandas DataFrame, got {type(panel).__name__}"],
        )

    if panel.empty:
        return ValidationReport(
            is_valid=False,
            errors=["panel is empty"],
            n_rows=0,
            n_entities=0,
        )

    # ─── entity_id ───────────────────────────────────────────────────────
    eid = schema.entity_id_column
    if eid not in panel.columns:
        errors.append(f"Missing required column: {eid!r} (entity_id_column)")
    else:
        n_null = panel[eid].isna().sum()
        if n_null > 0:
            errors.append(f"entity_id column {eid!r} has {n_null} null values")
        if schema.require_unique_entity:
            n_dup = panel[eid].duplicated().sum()
            if n_dup > 0:
                errors.append(
                    f"entity_id column {eid!r} has {n_dup} duplicated values "
                    f"(require_unique_entity=True)"
                )

    # ─── cohort ──────────────────────────────────────────────────────────
    coh = schema.cohort_column
    if coh not in panel.columns:
        errors.append(f"Missing required column: {coh!r} (cohort_column)")
    else:
        if panel[coh].isna().any():
            n_null = panel[coh].isna().sum()
            warnings.append(f"cohort column {coh!r} has {n_null} null values")

    # ─── geometry ────────────────────────────────────────────────────────
    if schema.require_geo:
        for gcol in schema.geo_columns:
            if gcol not in panel.columns:
                errors.append(
                    f"Missing required geo column: {gcol!r} (require_geo=True)"
                )
            else:
                n_null = panel[gcol].isna().sum()
                if n_null > 0:
                    errors.append(
                        f"geo column {gcol!r} has {n_null} null values "
                        f"(require_geo=True)"
                    )
    else:
        for gcol in schema.geo_columns:
            if gcol in panel.columns:
                n_null = panel[gcol].isna().sum()
                if n_null > 0:
                    warnings.append(
                        f"geo column {gcol!r} has {n_null} null values"
                    )

    # ─── outcomes ────────────────────────────────────────────────────────
    for spec in outcomes:
        missing = [c for c in spec.required_columns if c not in panel.columns]
        if missing:
            errors.append(
                f"OutcomeSpec({spec.name!r}): missing required columns {missing}"
            )
        # Check delta has at least some non-null values
        if spec.delta_column in panel.columns:
            if panel[spec.delta_column].isna().all():
                warnings.append(
                    f"OutcomeSpec({spec.name!r}): delta column "
                    f"{spec.delta_column!r} is entirely null"
                )

    n_entities = (
        int(panel[eid].nunique()) if eid in panel.columns else 0
    )

    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        n_rows=len(panel),
        n_entities=n_entities,
    )
