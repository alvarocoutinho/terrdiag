"""
terrdiag.presets — Generic defaults.

Only library-level defaults live here. Case-specific presets (snapshot
descriptors, geography filters, outcome specs for known programs) belong
to case_studies/<case>/presets.py.
"""

from __future__ import annotations

from terrdiag.specs import LISARecipe

# Default LISA recipe — used when the user doesn't specify one
DEFAULT_LISA = LISARecipe(
    kernel="gaussian",
    bandwidth_km=100.0,
    sensitivity_bandwidths_km=(50.0, 100.0, 150.0),
    permutations=999,
    random_state=42,
    significance_level=0.05,
)

__all__ = ["DEFAULT_LISA"]
