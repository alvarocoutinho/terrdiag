"""
terrdiag.layer1.independence — Post-hoc validation: cluster vs outcome.

Tests whether Layer 1 territorial typology is INDEPENDENT of an outcome
variable (NEEA, vol_pct, etc.). The expected null result — outcome
distribution does NOT differ significantly across clusters — validates
that the typology is purely structural, not contaminated by outcome leakage.

Critical design point: the outcome variable is passed as a SEPARATE
argument, not as part of any FeatureSpec. There is no possible code path
where this function uses the same data that fed `cluster()`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from terrdiag.audit.logging import run_log


@dataclass(slots=True)
class IndependenceTest:
    """Result of one Kruskal-Wallis test."""

    outcome_name: str
    statistic: float
    p_value: float
    n_clusters: int
    n_per_cluster: dict[int, int]
    cluster_medians: dict[int, float]
    cluster_means: dict[int, float]
    interpretation: str

    def summary(self) -> str:
        return (
            f"Kruskal-Wallis: {self.outcome_name} × cluster — "
            f"H={self.statistic:.3f}, p={self.p_value:.4f} "
            f"({self.interpretation})"
        )

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "outcome_name": self.outcome_name,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "n_clusters": int(self.n_clusters),
            "n_per_cluster": {int(k): int(v) for k, v in self.n_per_cluster.items()},
            "cluster_medians": {int(k): float(v) for k, v in self.cluster_medians.items()},
            "cluster_means": {int(k): float(v) for k, v in self.cluster_means.items()},
            "interpretation": self.interpretation,
        }

    def plot(self, panel: "pd.DataFrame", labels, **kwargs):
        """Boxplot of outcome by cluster (panel + labels required because
        the test only carries summary statistics, not the raw values)."""
        from terrdiag.layer1.plotting import plot_outcome_by_cluster
        return plot_outcome_by_cluster(self, panel, labels, **kwargs)


def kruskal_wallis(
    panel: pd.DataFrame,
    labels: np.ndarray | pd.Series,
    outcome_column: str,
    alpha: float = 0.05,
    audit: bool = True,
) -> IndependenceTest:
    """Kruskal-Wallis H test of outcome distribution across cluster labels.

    Parameters
    ----------
    panel : pd.DataFrame
        Entity-level panel that includes `outcome_column`.
    labels : np.ndarray | pd.Series
        Cluster labels from `terrdiag.layer1.cluster`. Length must match panel.
    outcome_column : str
        Column with the outcome values to test (e.g. 'delta_neea').
        This column must NOT be a feature used in clustering.
    alpha : float
        Significance level for the interpretation field. Default 0.05.
    audit : bool

    Returns
    -------
    IndependenceTest

    Notes
    -----
    For Layer 1 to be valid as a structural typology, the expected result
    is p > alpha — clusters should NOT differ significantly on the outcome.
    A significant result (p < alpha) suggests either:
      (a) the typology spuriously stratifies the outcome, OR
      (b) the territorial structure is genuinely associated with outcome
          differences (interpretively meaningful).
    Either case warrants investigation; this function only reports.
    """
    try:
        from scipy.stats import kruskal
    except ImportError as e:
        raise ImportError(
            "terrdiag.layer1.independence requires scipy. "
            "Install with: pip install 'terrdiag[layer1]'"
        ) from e

    t0 = time.time()

    if outcome_column not in panel.columns:
        raise KeyError(f"outcome_column {outcome_column!r} not in panel")
    if len(labels) != len(panel):
        raise ValueError(
            f"len(labels)={len(labels)} != len(panel)={len(panel)}"
        )

    labels = np.asarray(labels)
    y = panel[outcome_column].to_numpy(dtype=float)
    mask = ~np.isnan(y)
    y = y[mask]
    lab = labels[mask]

    unique_labels = sorted(set(lab))
    groups = [y[lab == lbl] for lbl in unique_labels]
    if any(len(g) < 5 for g in groups):
        import warnings as _w

        _w.warn(
            f"Some clusters have <5 valid observations for {outcome_column}; "
            f"Kruskal-Wallis may be unreliable.",
            stacklevel=2,
        )

    H, p = kruskal(*groups)

    medians = {int(lbl): float(np.median(g)) for lbl, g in zip(unique_labels, groups)}
    means = {int(lbl): float(np.mean(g)) for lbl, g in zip(unique_labels, groups)}
    counts = {int(lbl): int(len(g)) for lbl, g in zip(unique_labels, groups)}

    if p < alpha:
        interp = f"DEPENDENT at α={alpha} — outcome differs across clusters"
    else:
        interp = f"INDEPENDENT at α={alpha} — no significant difference"

    result = IndependenceTest(
        outcome_name=outcome_column,
        statistic=float(H),
        p_value=float(p),
        n_clusters=len(unique_labels),
        n_per_cluster=counts,
        cluster_medians=medians,
        cluster_means=means,
        interpretation=interp,
    )

    duration = time.time() - t0
    if audit:
        run_log.record(
            step="layer1.kruskal_wallis",
            duration_s=duration,
            params={
                "outcome_column": outcome_column,
                "alpha": alpha,
                "n_clusters": result.n_clusters,
            },
            input_rows=len(panel),
            output_rows=1,
            output_hash=None,
            notes=result.summary(),
        )

    return result
