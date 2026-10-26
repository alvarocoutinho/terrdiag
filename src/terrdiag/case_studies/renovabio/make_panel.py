"""
terrdiag.case_studies.renovabio.make_panel — End-to-end panel construction.

Run this once to produce the canonical wide panel for terrdiag analysis.
The output is a CSV at the canonical schema; subsequent terrdiag.io.load_panel
calls consume it directly.

Usage (as a function):

    from terrdiag.case_studies.renovabio import make_panel

    panel = make_panel.run(
        snapshots_dir="/path/to/anp/xlsx",
        gdf_pts_path="/path/to/gdf_pts.csv",
        output_path="painel_canonico.csv",
    )

Usage (as a script):

    python -m terrdiag.case_studies.renovabio.make_panel \\
        --snapshots-dir ./anp \\
        --gdf-pts ./gdf_pts.csv \\
        --output ./painel_canonico.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from terrdiag import audit, io
from terrdiag.schema import PanelSchema

from terrdiag.case_studies.renovabio.anp_parser import load_anp_raw
from terrdiag.case_studies.renovabio.presets import (
    CENTRO_SUL_UFS,
    E1GC_CANA1G_ROUTES,
    RENOVABIO_OUTCOMES,
    RENOVABIO_SNAPSHOTS,
    VALID_BRAZILIAN_UFS,
)
from terrdiag.case_studies.renovabio.reconcile import reconcile_and_filter, to_wide_with_deltas


# The RenovaBio panel uses 'cnpj_clean' as entity_id
RENOVABIO_SCHEMA = PanelSchema(
    entity_id_column="cnpj_clean",
    cohort_column="cohort",
    geo_columns=("X", "Y"),
    geo_crs_epsg=5880,
    require_geo=False,
    require_unique_entity=True,
)


def run(
    snapshots_dir: str | Path,
    gdf_pts_path: str | Path,
    output_path: str | Path,
    snapshots=RENOVABIO_SNAPSHOTS,
    metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build the canonical RenovaBio panel end-to-end.

    Parameters
    ----------
    snapshots_dir : str | Path
        Directory containing the three ANP xlsx files.
    gdf_pts_path : str | Path
        Path to the points source for georeferencing (CSV or shapefile).
    output_path : str | Path
        Where to write the resulting canonical wide panel (CSV).
    snapshots : iterable of SnapshotSource
        Defaults to RENOVABIO_SNAPSHOTS.
    metadata_path : str | Path | None
        If provided, also writes the run_metadata.json there.

    Returns
    -------
    pd.DataFrame
        The wide canonical panel (also written to output_path).
    """
    audit.run_log.reset()

    # 1. Parse raw ANP snapshots
    raw = load_anp_raw(
        snapshots=snapshots,
        valid_ufs=VALID_BRAZILIAN_UFS,
        base_path=snapshots_dir,
    )

    # 2. Filter + reconcile (Opção 2)
    long = reconcile_and_filter(
        raw,
        geography=CENTRO_SUL_UFS,
        routes=E1GC_CANA1G_ROUTES,
        biofuel="etanol hidratado",
        option="reconciled_v2",
    )

    # 3. Pivot to wide with cohort-aware deltas
    wide = to_wide_with_deltas(
        long,
        outcomes=RENOVABIO_OUTCOMES,
        entity_id_column="cnpj_clean",
    )

    # 4. Attach X, Y from gdf_pts
    wide = io.resolve_geo(wide, points_source=gdf_pts_path, schema=RENOVABIO_SCHEMA)

    # 5. Validate against the schema
    report = io.validate_panel(wide, schema=RENOVABIO_SCHEMA, outcomes=RENOVABIO_OUTCOMES)
    report.raise_if_invalid()

    # 6. Write to disk
    output_path = Path(output_path)
    io.write_panel(wide, output_path, schema=RENOVABIO_SCHEMA, outcomes=RENOVABIO_OUTCOMES)

    # 7. Write metadata next to the output
    if metadata_path is None:
        metadata_path = output_path.with_suffix(".metadata.json")
    audit.write_metadata(
        metadata_path,
        inputs={
            **{snap.label: Path(snapshots_dir) / snap.path for snap in snapshots},
            "gdf_pts": Path(gdf_pts_path),
        },
        extra={
            "case_study": "renovabio",
            "outcomes": [s.name for s in RENOVABIO_OUTCOMES],
            "n_mills": int(wide["cnpj_clean"].nunique()),
            "n_observations": int(len(wide)),
        },
    )

    return wide


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="renovabio.make_panel",
        description="Build the canonical RenovaBio panel from ANP snapshots.",
    )
    p.add_argument("--snapshots-dir", required=True, help="Dir with ANP xlsx files")
    p.add_argument("--gdf-pts", required=True, help="Points source for georef")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument(
        "--metadata", default=None, help="Optional explicit metadata JSON path"
    )
    args = p.parse_args(argv)

    wide = run(
        snapshots_dir=args.snapshots_dir,
        gdf_pts_path=args.gdf_pts,
        output_path=args.output,
        metadata_path=args.metadata,
    )

    print(f"\nPainel canônico: {len(wide)} mills, {wide.shape[1]} columns")
    print(f"  output:   {args.output}")
    print(f"  metadata: {args.metadata or Path(args.output).with_suffix('.metadata.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
