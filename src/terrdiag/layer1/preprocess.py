"""
terrdiag.layer1.preprocess — Apply declared transforms and clips.

The IEEE_CALLABLE_RECIPE (case study) already materializes columns with
transform and clip applied INSIDE the compute callable. This module exists
for the alternate path where:

    - features are pre-existing columns in the panel (not compute callables),
    - and the user wants the FeatureSpec.transform and FeatureSpec.clip
      to be applied declaratively before clustering.

It is safe to call this even if features were already transformed —
'identity' transform is a no-op and clip is idempotent on already-clipped
values. The two-step pattern (apply_recipe → preprocess) is the explicit
way to layer additional transforms on top of recipe outputs.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd

from terrdiag.audit.hashing import hash_dataframe
from terrdiag.audit.logging import run_log
from terrdiag.specs import FeatureSpec


def _apply_transform(s: pd.Series, kind: str) -> pd.Series:
    """Apply a single transform to a series."""
    if kind == "identity":
        return s
    if kind == "log1p":
        return pd.Series(np.log1p(s.clip(lower=0)), index=s.index)
    if kind == "zscore":
        mu, sd = s.mean(), s.std(ddof=0)
        if sd == 0 or pd.isna(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mu) / sd
    if kind == "percentile_rank":
        return s.rank(pct=True)
    raise ValueError(f"Unknown transform: {kind!r}")


def preprocess(
    panel: pd.DataFrame,
    features: Iterable[FeatureSpec],
    audit: bool = True,
) -> pd.DataFrame:
    """Apply each FeatureSpec's transform and clip to its source column.

    The transformed value overwrites the source column (in a copy of the
    input panel). If a feature has transform='identity' and clip=None,
    the column is unchanged.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel; must contain every spec.source_column.
    features : iterable of FeatureSpec
    audit : bool

    Returns
    -------
    pd.DataFrame
        Copy of panel with features transformed/clipped in place.
    """
    t0 = time.time()
    features = list(features)
    out = panel.copy()
    actions: list[str] = []

    for spec in features:
        col = spec.source_column
        if col not in out.columns:
            raise KeyError(f"FeatureSpec({spec.name}): source_column {col!r} not in panel")
        s = out[col]
        if spec.transform != "identity":
            s = _apply_transform(s, spec.transform)
            actions.append(f"{spec.name}:transform={spec.transform}")
        if spec.clip is not None:
            lo, hi = spec.clip
            s = s.clip(lower=lo, upper=hi)
            actions.append(f"{spec.name}:clip=({lo},{hi})")
        out[col] = s

    duration = time.time() - t0
    if audit:
        run_log.record(
            step="layer1.preprocess",
            duration_s=duration,
            params={"n_features": len(features), "n_actions": len(actions)},
            input_rows=len(panel),
            output_rows=len(out),
            output_hash=hash_dataframe(out),
            notes=f"{len(actions)} transform/clip operations applied",
        )

    return out
