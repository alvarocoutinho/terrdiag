"""
terrdiag.layer2.inspect — Pure inspection functions for Layer 2.

These return DataFrames that the analyst can print, save, or pass to plotting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from terrdiag.layer2.lisa import LISAResult
    from terrdiag.layer2.moran import MoranResult


# ─── LISA ────────────────────────────────────────────────────────────────


def lisa_classification_counts(result: "LISAResult") -> pd.DataFrame:
    """Counts of HH/LL/HL/LH/ns with shares."""
    counts = pd.Series(result.classifications).value_counts()
    # Ensure all 5 classes appear (zero where absent)
    for cls in ["HH", "LL", "HL", "LH", "ns"]:
        if cls not in counts.index:
            counts[cls] = 0
    counts = counts.reindex(["HH", "LL", "HL", "LH", "ns"])
    df = counts.to_frame("count")
    df["share"] = df["count"] / df["count"].sum()
    return df


def lisa_top_contributors(
    result: "LISAResult",
    panel: pd.DataFrame,
    n: int = 10,
    classification: str | None = None,
) -> pd.DataFrame:
    """Top n entities by absolute local I.

    If `classification` is set ('HH'/'LL'/'HL'/'LH'), filter to that class first.
    Returns identifier columns from panel + LISA fields.
    """
    if len(panel) != len(result.local_I):
        raise ValueError("panel length mismatch with LISA result")
    df = result.to_dataframe(panel)
    if classification is not None:
        df = df[df["classification"] == classification]
    df = df.sort_values("local_I", key=lambda s: s.abs(), ascending=False).head(n)
    return df


def lisa_by_group(
    result: "LISAResult",
    panel: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Crosstab of LISA classification × a categorical column (e.g. uf)."""
    if group_column not in panel.columns:
        raise KeyError(f"{group_column!r} not in panel")
    if len(panel) != len(result.local_I):
        raise ValueError("panel length mismatch")
    ct = pd.crosstab(panel[group_column], result.classifications)
    # Order classes
    for c in ["HH", "LL", "HL", "LH", "ns"]:
        if c not in ct.columns:
            ct[c] = 0
    return ct[["HH", "LL", "HL", "LH", "ns"]]


def lisa_significance_diagnostics(result: "LISAResult") -> pd.DataFrame:
    """p-value distribution diagnostics at multiple alpha thresholds.

    Useful for choosing an alpha and assessing multiple-comparisons exposure.
    """
    alphas = [0.001, 0.01, 0.05, 0.10]
    rows = []
    for a in alphas:
        n_sig = int((result.p_values < a).sum())
        n_sig_fdr = int((result.p_values_fdr < a).sum())
        rows.append({
            "alpha": a,
            "n_significant_raw": n_sig,
            "n_significant_fdr": n_sig_fdr,
            "share_significant_raw": n_sig / len(result.p_values),
            "share_significant_fdr": n_sig_fdr / len(result.p_values),
        })
    return pd.DataFrame(rows)


# ─── Moran ────────────────────────────────────────────────────────────────


def moran_permutation_diagnostics(result: "MoranResult") -> pd.DataFrame:
    """Permutation distribution summary."""
    perm = result.permutation_distribution
    quants = np.percentile(perm, [1, 5, 25, 50, 75, 95, 99])
    return pd.DataFrame({
        "statistic": ["min", "p01", "p05", "p25", "p50 (median)", "p75", "p95", "p99",
                       "max", "mean", "std", "observed I"],
        "value": [perm.min(), *quants, perm.max(), perm.mean(), perm.std(), result.I],
    })
