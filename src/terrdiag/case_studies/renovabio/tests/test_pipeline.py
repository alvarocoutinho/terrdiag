"""
Tests for the RenovaBio case study pipeline.

These tests require ANP xlsx fixtures + gdf_pts under fixtures/ (or set
TERRDIAG_FIXTURES env var).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from terrdiag.case_studies.renovabio import anp_parser
from terrdiag.case_studies.renovabio.make_panel import RENOVABIO_SCHEMA, run
from terrdiag.case_studies.renovabio.presets import (
    CENTRO_SUL_UFS,
    E1GC_CANA1G_ROUTES,
    RENOVABIO_OUTCOMES,
    RENOVABIO_SNAPSHOTS,
    VALID_BRAZILIAN_UFS,
)
from terrdiag.case_studies.renovabio.reconcile import (
    reconcile_and_filter,
    to_wide_with_deltas,
)
from terrdiag import io


FIXTURE_DIR = Path(os.environ.get("TERRDIAG_FIXTURES", "fixtures"))
GDF_PATH = FIXTURE_DIR / "gdf_pts_vol_corrected_cnpj.csv"


def _have_fixtures() -> bool:
    if not FIXTURE_DIR.exists():
        return False
    for snap in RENOVABIO_SNAPSHOTS:
        if not (FIXTURE_DIR / snap.path).exists():
            return False
    return GDF_PATH.exists()


needs_fixtures = pytest.mark.skipif(
    not _have_fixtures(),
    reason=f"Fixture files missing under {FIXTURE_DIR}. Set TERRDIAG_FIXTURES to override.",
)


# ─── SnapshotSource ──────────────────────────────────────────────────────


def test_snapshot_source_validation():
    with pytest.raises(ValueError):
        anp_parser.SnapshotSource(path="x", date="invalid", label="t0")
    with pytest.raises(ValueError):
        anp_parser.SnapshotSource(path="x", date="2022-01-01", label="bad")


# ─── Full pipeline canonical numbers ─────────────────────────────────────


@needs_fixtures
def test_make_panel_n230(tmp_path: Path):
    """The canonical reconciled panel must yield 230 mills."""
    out_path = tmp_path / "panel.csv"
    wide = run(
        snapshots_dir=FIXTURE_DIR,
        gdf_pts_path=GDF_PATH,
        output_path=out_path,
    )
    assert wide["cnpj_clean"].nunique() == 230


@needs_fixtures
def test_canonical_uf_distribution():
    raw = anp_parser.load_anp_raw(
        snapshots=RENOVABIO_SNAPSHOTS,
        valid_ufs=VALID_BRAZILIAN_UFS,
        base_path=FIXTURE_DIR,
    )
    long = reconcile_and_filter(
        raw,
        geography=CENTRO_SUL_UFS,
        routes=E1GC_CANA1G_ROUTES,
        option="reconciled_v2",
    )
    one_per_mill = long.drop_duplicates("cnpj_clean")
    uf = one_per_mill["uf"].value_counts().to_dict()
    assert uf["SP"] == 123
    assert uf["GO"] == 33
    assert uf["MG"] == 29
    assert uf["PR"] == 19
    assert uf["MS"] == 17
    assert uf["MT"] == 9


@needs_fixtures
def test_strict_smaller_than_reconciled():
    raw = anp_parser.load_anp_raw(
        snapshots=RENOVABIO_SNAPSHOTS,
        valid_ufs=VALID_BRAZILIAN_UFS,
        base_path=FIXTURE_DIR,
    )
    strict = reconcile_and_filter(
        raw, geography=CENTRO_SUL_UFS, routes=E1GC_CANA1G_ROUTES, option="strict"
    )
    reconciled = reconcile_and_filter(
        raw,
        geography=CENTRO_SUL_UFS,
        routes=E1GC_CANA1G_ROUTES,
        option="reconciled_v2",
    )
    assert reconciled["cnpj_clean"].nunique() > strict["cnpj_clean"].nunique()
    assert reconciled["cnpj_clean"].nunique() == 230


@needs_fixtures
def test_wide_panel_matches_schema(tmp_path: Path):
    out_path = tmp_path / "panel.csv"
    wide = run(
        snapshots_dir=FIXTURE_DIR,
        gdf_pts_path=GDF_PATH,
        output_path=out_path,
    )
    report = io.validate_panel(
        wide, schema=RENOVABIO_SCHEMA, outcomes=RENOVABIO_OUTCOMES
    )
    assert report.is_valid, report.summary()


@needs_fixtures
def test_load_panel_after_make_panel(tmp_path: Path):
    """After make_panel.run, the CSV must be loadable by io.load_panel."""
    out_path = tmp_path / "panel.csv"
    run(snapshots_dir=FIXTURE_DIR, gdf_pts_path=GDF_PATH, output_path=out_path)

    panel = io.load_panel(
        out_path, schema=RENOVABIO_SCHEMA, outcomes=RENOVABIO_OUTCOMES
    )
    assert panel["cnpj_clean"].nunique() == 230
    # Round-trip preserves the leading-zero CNPJs
    assert panel["cnpj_clean"].str.len().eq(14).all()


# ─── Bug-regression tests ────────────────────────────────────────────────


@needs_fixtures
def test_bug3_no_embedded_newlines():
    raw = anp_parser.load_anp_raw(
        snapshots=RENOVABIO_SNAPSHOTS,
        valid_ufs=VALID_BRAZILIAN_UFS,
        base_path=FIXTURE_DIR,
    )
    long = reconcile_and_filter(
        raw,
        geography=CENTRO_SUL_UFS,
        routes=E1GC_CANA1G_ROUTES,
        option="reconciled_v2",
    )
    for c in long.columns:
        if long[c].dtype == object:
            bad = (
                long[c]
                .astype(str)
                .str.contains(r"[\r\n\t]", regex=True, na=False)
                .sum()
            )
            assert bad == 0, f"Column {c!r} has embedded whitespace in {bad} cells"


@needs_fixtures
def test_bug2_uf_coverage():
    raw = anp_parser.load_anp_raw(
        snapshots=RENOVABIO_SNAPSHOTS,
        valid_ufs=VALID_BRAZILIAN_UFS,
        base_path=FIXTURE_DIR,
    )
    long = reconcile_and_filter(
        raw,
        geography=CENTRO_SUL_UFS,
        routes=E1GC_CANA1G_ROUTES,
        option="reconciled_v2",
    )
    assert long["uf"].notna().all()
