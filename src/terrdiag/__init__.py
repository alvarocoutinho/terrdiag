"""
terrdiag — Two-Layer Territorial Diagnostic Framework

Public API (Fase 1):

  Schema and validation:
    - PanelSchema, validate_panel    (terrdiag.schema)

  Generic I/O:
    - load_panel, write_panel, resolve_geo, validate_panel  (terrdiag.io)

  Configuration:
    - FeatureSpec, FeatureRecipe, OutcomeSpec, LISARecipe   (terrdiag.specs)
    - DEFAULT_LISA                                          (terrdiag.presets)

  Audit and reproducibility:
    - run_log, hash_dataframe, hash_file, write_metadata    (terrdiag.audit)

  Reference case studies (NOT the generic library API):
    - renovabio: Brazilian biofuel CBIO program             (terrdiag.case_studies.renovabio)

Higher analysis layers (1, 2, overlay, plotting) ship in subsequent releases.
"""

from terrdiag._version import __version__, __version_info__
from terrdiag import audit, io, layer1, layer2, overlay, presets, schema, specs

__all__ = [
    "__version__",
    "__version_info__",
    "audit",
    "io",
    "layer1",
    "layer2",
    "overlay",
    "presets",
    "schema",
    "specs",
]
