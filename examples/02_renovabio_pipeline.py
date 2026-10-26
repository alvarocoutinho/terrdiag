"""
02_renovabio_pipeline.py — End-to-end RenovaBio case study pipeline.

Shows the full split between case_studies (panel construction) and
terrdiag (analysis-ready panel consumption).

Usage:
    python examples/02_renovabio_pipeline.py [fixtures_dir]

Default fixtures_dir is 'fixtures' relative to current directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

from terrdiag import audit, io

from terrdiag.case_studies.renovabio import make_panel
from terrdiag.case_studies.renovabio.make_panel import RENOVABIO_SCHEMA
from terrdiag.case_studies.renovabio.presets import RENOVABIO_OUTCOMES


def main(fixtures_dir: str = "fixtures") -> int:
    fixtures = Path(fixtures_dir)
    if not fixtures.exists():
        print(f"ERROR: fixtures directory {fixtures} not found.", file=sys.stderr)
        return 1

    print(f"╭─ RenovaBio end-to-end pipeline")
    print(f"│  fixtures: {fixtures.resolve()}")
    print(f"╰──")
    print()

    # === Phase 1: case study — build the canonical panel from raw xlsx
    print("PHASE 1 — terrdiag.case_studies.renovabio.make_panel.run")
    print("─" * 60)
    output_csv = Path("painel_canonico.csv")
    wide = make_panel.run(
        snapshots_dir=fixtures,
        gdf_pts_path=fixtures / "gdf_pts_vol_corrected_cnpj.csv",
        output_path=output_csv,
    )
    print(
        f"\n  → {wide['cnpj_clean'].nunique()} mills, {len(wide)} rows, "
        f"{wide.shape[1]} columns"
    )
    print(f"  → output: {output_csv}")
    print(f"  → metadata: {output_csv.with_suffix('.metadata.json')}")

    # === Phase 2: library user — load the canonical panel and use it
    print()
    print("PHASE 2 — terrdiag.io.load_panel (downstream analysis)")
    print("─" * 60)
    panel = io.load_panel(
        output_csv, schema=RENOVABIO_SCHEMA, outcomes=RENOVABIO_OUTCOMES
    )
    print(
        f"\n  → loaded: {panel['cnpj_clean'].nunique()} mills, "
        f"{len(panel)} rows"
    )
    print(f"  → UF SP={int((panel['uf']=='SP').sum())}, "
          f"GO={int((panel['uf']=='GO').sum())}, "
          f"MG={int((panel['uf']=='MG').sum())}")
    print(f"  → cohort 2022={int((panel['cohort']=='2022').sum())}, "
          f"2023={int((panel['cohort']=='2023').sum())}")
    print(f"  → ΔNEEA range: {panel['delta_neea'].min():.2f} … "
          f"{panel['delta_neea'].max():.2f}")
    print(f"  → Δvol  range: {panel['delta_vol_pct'].min():.2f} … "
          f"{panel['delta_vol_pct'].max():.2f}")

    print()
    print(f"Audit log ({len(audit.run_log.entries)} steps):")
    for e in audit.run_log.entries:
        print(f"  [{e.duration_s:.2f}s] {e.step} — {e.notes}")

    # Canonical invariants
    print()
    print("Canonical invariants:")
    ok = True
    checks = [
        ("n_mills == 230", panel["cnpj_clean"].nunique() == 230),
        ("UF SP == 123", int((panel["uf"] == "SP").sum()) == 123),
        ("UF GO == 33", int((panel["uf"] == "GO").sum()) == 33),
        ("cohort 2022 == 214", int((panel["cohort"] == "2022").sum()) == 214),
        ("cohort 2023 == 16", int((panel["cohort"] == "2023").sum()) == 16),
        ("Geo coverage ≥ 228", int(panel["X"].notna().sum()) >= 228),
    ]
    for label, passed in checks:
        mark = "✓" if passed else "✗"
        print(f"  {mark}  {label}")
        ok = ok and passed

    return 0 if ok else 2


if __name__ == "__main__":
    fix = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
    sys.exit(main(fix))
