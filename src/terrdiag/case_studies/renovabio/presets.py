"""
terrdiag.case_studies.renovabio.presets — RenovaBio-specific configuration.

These constants describe the ANP/RenovaBio case study and do not belong
inside the generic terrdiag library.
"""

from __future__ import annotations

from terrdiag.specs import OutcomeSpec

from terrdiag.case_studies.renovabio.anp_parser import SnapshotSource

# ─── Snapshots ──────────────────────────────────────────────────────────────
RENOVABIO_SNAPSHOTS: tuple[SnapshotSource, ...] = (
    SnapshotSource(
        path="certificados-aprovados.xlsx",
        date="2022-02-22",
        label="t0_2022",
    ),
    SnapshotSource(
        path="certificados-aprovados-producao_2025.xlsx",
        date="2023-10-09",
        label="t1_2023",
    ),
    SnapshotSource(
        path="certificados-aprovados-producao_2026.xlsx",
        date="2026-04-17",
        label="t2_2026",
    ),
)

# ─── Geography filters ──────────────────────────────────────────────────────
CENTRO_SUL_UFS: tuple[str, ...] = ("SP", "GO", "MG", "PR", "MS", "MT")

VALID_BRAZILIAN_UFS: frozenset[str] = frozenset(
    {
        "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
        "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
        "RS", "RO", "RR", "SC", "SP", "SE", "TO",
    }
)

# ─── Route filters ──────────────────────────────────────────────────────────
E1GC_CANA1G_ROUTES: tuple[str, ...] = (
    "Etanol combustível de primeira geração – cana-de-açúcar",
    "Etanol combustível de primeira geração – cana-de-açúcar e milho em usina integrada",
    "Etanol combustível de primeira geração – cana-de-açúcar ",
    "Etanol combustível de primeira geração – cana- de açúcar",
)

# ─── Outcomes for terrdiag Layer 2 ──────────────────────────────────────────
NEEA_SPEC = OutcomeSpec(
    name="NEEA",
    snapshots={"t0": "neea_t0_2022", "t1": "neea_t1_2023", "t2": "neea_t2_2026"},
    delta_column="delta_neea",
    first_column="neea_first",
    first_origin_column="neea_first_origin",
    units="gCO2eq/MJ",
    direction="higher_better",
    description=(
        "Nota de Eficiência Energético-Ambiental (g CO2-eq/MJ avoided). "
        "Mill-level outcome certified by ANP; higher means more avoided emissions."
    ),
)

VOL_SPEC = OutcomeSpec(
    name="vol_pct",
    snapshots={
        "t0": "vol_pct_t0_2022",
        "t1": "vol_pct_t1_2023",
        "t2": "vol_pct_t2_2026",
    },
    delta_column="delta_vol_pct",
    first_column="vol_pct_first",
    first_origin_column="vol_pct_first_origin",
    units="%",
    direction="higher_better",
    description=(
        "Volumetric eligibility share of certified output. "
        "Lower values indicate operational/declarative issues."
    ),
)

RENOVABIO_OUTCOMES: tuple[OutcomeSpec, ...] = (NEEA_SPEC, VOL_SPEC)
