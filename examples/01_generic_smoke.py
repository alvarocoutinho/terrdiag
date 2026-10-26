"""
01_generic_smoke.py — Generic terrdiag user scenario.

Demonstrates how a researcher in a different policy domain would use
terrdiag: they bring their own pre-built panel CSV, declare their schema
and OutcomeSpecs, and call the library.

No ANP, no RenovaBio specifics. The terrdiag library does not know nor
care where the panel came from.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from terrdiag import audit, io, schema, specs


def _build_synthetic_panel(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Pretend this came from any source: education, health, environment."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "entity_id": [f"E{i:04d}" for i in range(n)],
            "cohort": rng.choice(["2020", "2021"], size=n, p=[0.85, 0.15]),
            "region": rng.choice(["N", "S", "E", "W"], size=n),
            "score_t0": rng.normal(70, 10, n).round(2),
            "score_t1": rng.normal(72, 10, n).round(2),
            "score_t2": rng.normal(74, 10, n).round(2),
        }
    )
    df["score_first"] = df["score_t0"]
    df["score_first_origin"] = "t0"
    df.loc[df["cohort"] == "2021", "score_first"] = df.loc[df["cohort"] == "2021", "score_t1"]
    df.loc[df["cohort"] == "2021", "score_first_origin"] = "t1"
    df["delta_score"] = df["score_t2"] - df["score_first"]
    return df


def main() -> int:
    print("╭─ Generic terrdiag user — synthetic policy panel")
    print("╰──")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # 1. User has their data — for the demo, generate a fake one
        panel = _build_synthetic_panel()
        panel_path = td / "my_panel.csv"
        panel.to_csv(panel_path, index=False)
        print(f"[setup] wrote synthetic panel: {panel_path.name} ({len(panel)} rows)")

        # 2. User declares their schema and outcomes
        my_schema = schema.PanelSchema(
            entity_id_column="entity_id",
            cohort_column="cohort",
        )
        my_outcome = specs.OutcomeSpec(
            name="score",
            snapshots={"t0": "score_t0", "t1": "score_t1", "t2": "score_t2"},
            delta_column="delta_score",
            first_column="score_first",
            first_origin_column="score_first_origin",
            units="points",
            direction="higher_better",
            description="Synthetic policy score (test).",
        )

        # 3. User loads via terrdiag.io
        print("[1/2] io.load_panel ...")
        loaded = io.load_panel(panel_path, schema=my_schema, outcomes=[my_outcome])
        print(
            f"      → {len(loaded)} rows, {loaded[my_schema.entity_id_column].nunique()} entities"
        )
        print(f"      → delta_score range: {loaded['delta_score'].min():.2f} … "
              f"{loaded['delta_score'].max():.2f}")

        # 4. Validation works as advertised
        print("[2/2] io.validate_panel ...")
        report = io.validate_panel(loaded, schema=my_schema, outcomes=[my_outcome])
        print(f"      → is_valid={report.is_valid}, "
              f"errors={len(report.errors)}, warnings={len(report.warnings)}")

        # 5. Audit log
        print()
        print("Audit log:")
        for e in audit.run_log.entries:
            print(f"  [{e.duration_s:.3f}s] {e.step} — {e.notes}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
