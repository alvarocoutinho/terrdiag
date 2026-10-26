"""
terrdiag.overlay.inspect — Inspection helpers for diagnostic cells.

Returns DataFrames suitable for printing, CSV export, or paper tables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from terrdiag.overlay.diagnostic import OverlayResult


def cell_summary(overlay: "OverlayResult") -> pd.DataFrame:
    """One row per non-empty (cluster, LISA class) cell with statistics.

    Returns columns: cluster, lisa_class, n_mills, mean_outcome, median_outcome.
    """
    rows = []
    for cluster in overlay.mill_table["cluster"].unique():
        for cls in ["HH", "LL", "HL", "LH", "ns"]:
            mask = (
                (overlay.mill_table["cluster"] == cluster)
                & (overlay.mill_table["lisa_class"] == cls)
            )
            sub = overlay.mill_table[mask]
            if len(sub) == 0:
                continue
            cluster_label = (
                overlay.cluster_names.get(cluster, str(cluster))
                if overlay.cluster_names else str(cluster)
            )
            rows.append({
                "cluster_id": int(cluster),
                "cluster_name": cluster_label,
                "lisa_class": cls,
                "n_mills": len(sub),
                "share_of_cluster": len(sub) / len(overlay.mill_table[overlay.mill_table["cluster"] == cluster]),
                "share_of_lisa_class": len(sub) / max(1, len(overlay.mill_table[overlay.mill_table["lisa_class"] == cls])),
                "mean_outcome": float(sub["outcome_value"].mean()),
                "median_outcome": float(sub["outcome_value"].median()),
                "mean_local_I": float(sub["local_I"].mean()),
                "median_p": float(sub["p_value"].median()),
            })
    return pd.DataFrame(rows)


def cell_representatives(
    overlay: "OverlayResult",
    cluster: int | str,
    lisa_class: str,
    n: int = 5,
    sort_by: str = "local_I",
) -> pd.DataFrame:
    """Return top-n mills in a given (cluster, LISA class) cell.

    Useful for narrating exemplary cases per cell.

    Parameters
    ----------
    overlay : OverlayResult
    cluster : int or str
        Cluster id or semantic name (if cluster_names was set).
    lisa_class : str
        One of 'HH', 'LL', 'HL', 'LH', 'ns'.
    n : int
    sort_by : str
        Column to sort by (default 'local_I' — descending).
    """
    sub = overlay.cell(cluster, lisa_class)
    if sort_by in sub.columns:
        sub = sub.sort_values(sort_by, key=lambda s: s.abs(), ascending=False)
    return sub.head(n)


def cluster_lisa_marginals(overlay: "OverlayResult") -> dict:
    """Return marginal distributions of cluster and LISA class.

    Useful for sanity checks: marginals should match the original Layer 1
    and Layer 2 distributions (after dropping orphan mills).
    """
    return {
        "cluster_marginal": overlay.mill_table["cluster"].value_counts().sort_index().to_dict(),
        "lisa_marginal": overlay.mill_table["lisa_class"].value_counts().to_dict(),
    }


def association_test(overlay: "OverlayResult") -> dict:
    """Chi-squared test of independence between cluster and LISA class.

    The null is independence. A significant result means cluster membership
    is associated with LISA classification — i.e. the territorial typology
    *does* stratify the local spatial regime. This is informative for the
    paper: do clusters reflect spatial dynamics, or are they truly orthogonal?
    """
    try:
        from scipy.stats import chi2_contingency
    except ImportError as e:
        raise ImportError("association_test requires scipy") from e

    ct = overlay.cell_counts
    chi2, p, dof, expected = chi2_contingency(ct.values)
    # Cramér's V — effect size
    n = ct.values.sum()
    k = min(ct.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0
    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p),
        "degrees_of_freedom": int(dof),
        "cramers_v": float(cramer_v),
        "n": int(n),
        "interpretation": (
            "ASSOCIATED" if p < 0.05 else "INDEPENDENT"
        ) + f" at α=0.05",
    }
