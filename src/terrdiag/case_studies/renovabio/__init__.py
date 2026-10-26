"""
terrdiag.case_studies.renovabio — RenovaBio panel construction & feature engineering.

Public API:

  Panel construction (from raw ANP):
    - SnapshotSource, load_anp_raw      (anp_parser)
    - reconcile_and_filter,
      to_wide_with_deltas               (reconcile)
    - make_panel.run                    (end-to-end orchestrator)

  Layer 1 preparation:
    - attach_cd_ibge                    (bridge)
    - attach_uai, apply_recipe,
      build_muni_panel                  (muni_features)
    - IEEE_CALLABLE_RECIPE,
      IEEE_CLUSTER_FEATURES             (recipe)

  Presets:
    - RENOVABIO_SNAPSHOTS, NEEA_SPEC, VOL_SPEC, RENOVABIO_OUTCOMES,
      CENTRO_SUL_UFS, E1GC_CANA1G_ROUTES, VALID_BRAZILIAN_UFS
"""

from terrdiag.case_studies.renovabio.anp_parser import (
    SnapshotSource,
    load_anp_raw,
)
from terrdiag.case_studies.renovabio.bridge import attach_cd_ibge
from terrdiag.case_studies.renovabio.muni_features import (
    apply_recipe,
    attach_uai,
    build_muni_panel,
)
from terrdiag.case_studies.renovabio.reconcile import (
    reconcile_and_filter,
    to_wide_with_deltas,
)
from terrdiag.case_studies.renovabio.recipe import (
    IEEE_CALLABLE_RECIPE,
    IEEE_CLUSTER_FEATURES,
)

__all__ = [
    "SnapshotSource",
    "load_anp_raw",
    "attach_cd_ibge",
    "attach_uai",
    "apply_recipe",
    "build_muni_panel",
    "reconcile_and_filter",
    "to_wide_with_deltas",
    "IEEE_CALLABLE_RECIPE",
    "IEEE_CLUSTER_FEATURES",
]
