"""
Tests for terrdiag core (schema, io, specs, audit).

These tests do not depend on any case-specific data; they exercise the
library contract using minimal in-memory DataFrames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from terrdiag import audit, io, schema, specs


# ─── Imports & version ───────────────────────────────────────────────────


def test_imports():
    import terrdiag

    assert terrdiag.__version__.startswith("0.5.0")
    assert hasattr(terrdiag, "io")
    assert hasattr(terrdiag, "schema")
    assert hasattr(terrdiag, "specs")
    assert hasattr(terrdiag, "audit")
    assert hasattr(terrdiag, "presets")


def test_public_api_surface():
    assert hasattr(io, "load_panel")
    assert hasattr(io, "write_panel")
    assert hasattr(io, "validate_panel")
    assert hasattr(io, "resolve_geo")
    assert hasattr(schema, "PanelSchema")
    assert hasattr(schema, "validate_panel")
    assert hasattr(schema, "ValidationReport")


# ─── Specs frozen + validating ───────────────────────────────────────────


def test_specs_frozen():
    from dataclasses import FrozenInstanceError

    f = specs.FeatureSpec(name="x", source_column="x")
    with pytest.raises(FrozenInstanceError):
        f.name = "y"


def test_specs_validation():
    with pytest.raises(ValueError):
        specs.LISARecipe(bandwidth_km=-1)
    with pytest.raises(ValueError):
        specs.LISARecipe(permutations=10)
    with pytest.raises(ValueError):
        specs.OutcomeSpec(name="X", delta_column="")
    with pytest.raises(ValueError):
        specs.FeatureRecipe(name="x", features=(), pca_components=1)


def test_outcome_required_columns():
    spec = specs.OutcomeSpec(
        name="X",
        snapshots={"t0": "x_t0", "t1": "x_t1"},
        delta_column="delta_x",
        first_column="x_first",
        first_origin_column="x_first_origin",
    )
    cols = set(spec.required_columns)
    assert "x_t0" in cols
    assert "x_t1" in cols
    assert "delta_x" in cols
    assert "x_first" in cols
    assert "x_first_origin" in cols


# ─── Schema validation ───────────────────────────────────────────────────


def _make_minimal_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["A", "B", "C"],
            "cohort": ["2022", "2022", "2023"],
            "X": [1.0, 2.0, 3.0],
            "Y": [10.0, 20.0, 30.0],
            "metric_t0": [1.0, 2.0, np.nan],
            "metric_t1": [1.1, 2.1, 3.1],
            "delta_metric": [0.1, 0.1, np.nan],
            "metric_first": [1.0, 2.0, 3.1],
            "metric_first_origin": ["t0", "t0", "t1"],
        }
    )


def test_validate_panel_valid():
    panel = _make_minimal_panel()
    out_spec = specs.OutcomeSpec(
        name="metric",
        snapshots={"t0": "metric_t0", "t1": "metric_t1"},
        delta_column="delta_metric",
        first_column="metric_first",
        first_origin_column="metric_first_origin",
    )
    report = schema.validate_panel(panel, outcomes=[out_spec])
    assert report.is_valid, report.summary()
    assert report.n_rows == 3
    assert report.n_entities == 3


def test_validate_panel_missing_entity_id():
    panel = _make_minimal_panel().drop(columns="entity_id")
    report = schema.validate_panel(panel)
    assert not report.is_valid
    assert any("entity_id" in e for e in report.errors)


def test_validate_panel_duplicate_entity_id():
    panel = _make_minimal_panel()
    panel.loc[0, "entity_id"] = "B"  # duplicate
    report = schema.validate_panel(panel)
    assert not report.is_valid
    assert any("duplicated" in e for e in report.errors)


def test_validate_panel_missing_outcome_cols():
    panel = _make_minimal_panel().drop(columns="delta_metric")
    out_spec = specs.OutcomeSpec(
        name="metric",
        snapshots={"t0": "metric_t0", "t1": "metric_t1"},
        delta_column="delta_metric",
    )
    report = schema.validate_panel(panel, outcomes=[out_spec])
    assert not report.is_valid
    assert any("delta_metric" in e for e in report.errors)


def test_validate_panel_require_geo():
    panel = _make_minimal_panel()
    panel.loc[0, "X"] = np.nan
    sch = schema.PanelSchema(require_geo=True)
    report = schema.validate_panel(panel, schema=sch)
    assert not report.is_valid


def test_report_raises():
    panel = _make_minimal_panel().drop(columns="entity_id")
    report = schema.validate_panel(panel)
    with pytest.raises(ValueError, match="entity_id"):
        report.raise_if_invalid()


# ─── load_panel / write_panel roundtrip ──────────────────────────────────


def test_load_write_roundtrip_csv(tmp_path: Path):
    panel = _make_minimal_panel()
    path = tmp_path / "panel.csv"
    io.write_panel(panel, path)
    loaded = io.load_panel(path)
    assert len(loaded) == len(panel)
    assert set(loaded.columns) == set(panel.columns)


def test_load_panel_validates_outcomes(tmp_path: Path):
    panel = _make_minimal_panel().drop(columns="metric_t0")
    path = tmp_path / "panel.csv"
    panel.to_csv(path, index=False, encoding="utf-8-sig")  # write without validation

    out_spec = specs.OutcomeSpec(
        name="metric",
        snapshots={"t0": "metric_t0", "t1": "metric_t1"},
        delta_column="delta_metric",
    )
    with pytest.raises(ValueError, match="metric_t0"):
        io.load_panel(path, outcomes=[out_spec])


def test_load_panel_with_custom_schema(tmp_path: Path):
    panel = pd.DataFrame(
        {
            "cnpj_clean": ["00000000000001", "00000000000002"],
            "cohort": ["2022", "2022"],
            "X": [1.0, 2.0],
            "Y": [10.0, 20.0],
        }
    )
    path = tmp_path / "panel.csv"
    panel.to_csv(path, index=False)

    sch = schema.PanelSchema(entity_id_column="cnpj_clean")
    loaded = io.load_panel(path, schema=sch)
    # CNPJ must remain string (leading zeros preserved)
    assert loaded["cnpj_clean"].iloc[0] == "00000000000001"


# ─── resolve_geo ─────────────────────────────────────────────────────────


def test_resolve_geo_basic(tmp_path: Path):
    panel = pd.DataFrame(
        {
            "entity_id": ["A", "B", "C"],
            "cohort": ["x", "x", "x"],
        }
    )
    pts = pd.DataFrame(
        {
            "entity_id": ["A", "B"],
            "X": [10.0, 20.0],
            "Y": [100.0, 200.0],
        }
    )
    pts_path = tmp_path / "pts.csv"
    pts.to_csv(pts_path, index=False)

    out = io.resolve_geo(panel, points_source=pts_path)
    assert out.loc[out["entity_id"] == "A", "X"].iloc[0] == 10.0
    assert pd.isna(out.loc[out["entity_id"] == "C", "X"].iloc[0])


def test_resolve_geo_missing_entity_id(tmp_path: Path):
    panel = pd.DataFrame({"id": ["A"], "cohort": ["x"]})
    pts_path = tmp_path / "pts.csv"
    pd.DataFrame({"entity_id": ["A"], "X": [1.0], "Y": [2.0]}).to_csv(pts_path, index=False)

    with pytest.raises(KeyError, match="entity_id"):
        io.resolve_geo(panel, points_source=pts_path)


# ─── audit log + hashing ─────────────────────────────────────────────────


def test_run_log_records_steps(tmp_path: Path):
    audit.run_log.reset()
    panel = _make_minimal_panel()
    p = tmp_path / "x.csv"
    io.write_panel(panel, p)
    io.load_panel(p)

    steps = [e.step for e in audit.run_log.entries]
    assert "io.write_panel" in steps
    assert "io.load_panel" in steps


def test_hash_deterministic():
    a = _make_minimal_panel()
    b = _make_minimal_panel().sample(frac=1, random_state=99)  # shuffle rows
    assert audit.hash_dataframe(a) == audit.hash_dataframe(b)


def test_hash_sensitive_to_columns():
    a = _make_minimal_panel()
    b = a.copy()
    b["new_col"] = 1
    assert audit.hash_dataframe(a) != audit.hash_dataframe(b)
