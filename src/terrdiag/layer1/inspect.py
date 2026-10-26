"""
terrdiag.layer1.inspect — Pure inspection functions returning DataFrames.

These functions do not import matplotlib; they produce tabular artifacts
that the user can print, save to CSV, or pass to plotting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from terrdiag.layer1.cluster import ClusterResult


def cluster_profile(
    result: "ClusterResult",
    panel: pd.DataFrame,
    statistic: str = "mean",
) -> pd.DataFrame:
    """Per-cluster summary statistics for every feature used in clustering.

    Parameters
    ----------
    result : ClusterResult
    panel : pd.DataFrame
        The panel passed to `cluster()` — must contain the source columns.
    statistic : {'mean', 'median', 'both'}
        Summary statistic per cluster.

    Returns
    -------
    pd.DataFrame
        Index = feature_name, columns = cluster integers (and 'overall'),
        cells = the requested statistic. With statistic='both', columns
        are MultiIndex (statistic, cluster).
    """
    if len(panel) != len(result.labels):
        raise ValueError(
            f"panel has {len(panel)} rows, labels has {len(result.labels)}"
        )

    df = panel[list(result.feature_source_columns)].copy()
    df.columns = list(result.feature_names)
    df["_cluster"] = result.labels

    if statistic == "mean":
        prof = df.groupby("_cluster").mean().T
        prof["overall"] = df.drop(columns="_cluster").mean()
    elif statistic == "median":
        prof = df.groupby("_cluster").median().T
        prof["overall"] = df.drop(columns="_cluster").median()
    elif statistic == "both":
        means = df.groupby("_cluster").mean().T
        meds = df.groupby("_cluster").median().T
        prof = pd.concat(
            {"mean": means, "median": meds}, axis=1,
        )
    else:
        raise ValueError(f"statistic must be 'mean', 'median', or 'both', got {statistic!r}")

    prof.index.name = "feature"
    return prof


def feature_loadings(result: "ClusterResult") -> pd.DataFrame:
    """PCA loadings as a DataFrame.

    Returns
    -------
    pd.DataFrame
        Index = feature_name, columns = ['PC1', 'PC2', ...].
        Each cell = loading of feature on that component (higher absolute
        value = feature drives that component more).
    """
    n_comp = result.pca_loadings.shape[1]
    columns = [f"PC{i + 1}" for i in range(n_comp)]
    df = pd.DataFrame(
        result.pca_loadings, index=list(result.feature_names), columns=columns
    )
    df.index.name = "feature"
    return df


def cluster_crosstab(
    result: "ClusterResult",
    panel: pd.DataFrame,
    group_column: str,
    normalize: bool = False,
) -> pd.DataFrame:
    """Crosstab of cluster × group_column (e.g. 'uf', 'rota_tag').

    Parameters
    ----------
    result : ClusterResult
    panel : pd.DataFrame
        Must contain `group_column`.
    group_column : str
    normalize : bool
        If True, normalize within rows (group_column) so each row sums to 1.

    Returns
    -------
    pd.DataFrame
        rows = group values, columns = cluster integers.
    """
    if group_column not in panel.columns:
        raise KeyError(f"{group_column!r} not in panel")
    if len(panel) != len(result.labels):
        raise ValueError("panel/labels length mismatch")

    ct = pd.crosstab(panel[group_column], result.labels)
    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0)
    return ct


def centroid_distances(result: "ClusterResult") -> pd.DataFrame:
    """Pairwise Euclidean distances between centroids in PCA space."""
    k = len(result.centroids)
    d = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            d[i, j] = np.linalg.norm(result.centroids[i] - result.centroids[j])
    df = pd.DataFrame(d, index=range(k), columns=range(k))
    df.index.name = "cluster"
    df.columns.name = "cluster"
    return df


def silhouette_per_cluster(
    result: "ClusterResult",
) -> pd.DataFrame:
    """Per-cluster silhouette statistics.

    Returns mean and quartiles of the silhouette coefficient for each cluster.
    """
    try:
        from sklearn.metrics import silhouette_samples
    except ImportError as e:
        raise ImportError(
            "silhouette_per_cluster requires scikit-learn"
        ) from e

    sil = silhouette_samples(result.pca_components, result.labels)
    df = pd.DataFrame({"silhouette": sil, "cluster": result.labels})
    out = df.groupby("cluster")["silhouette"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )
    return out


def naming_suggestion(result: "ClusterResult", panel: pd.DataFrame) -> pd.DataFrame:
    """Tentative semantic naming based on cluster profile.

    Heuristic: ranks each cluster on a small set of axes
    (development, intensity, frontier-ness) using the cluster profile,
    and proposes a name. Purely advisory — the analyst should validate
    against centroids and crosstabs.

    Returns
    -------
    pd.DataFrame
        cluster | log_pib_pc | natveg_share | log_trator_per_est | suggested_name
    """
    prof = cluster_profile(result, panel, statistic="mean")
    # Pull the axes we care about; missing axes default to 0
    rows = []
    for c in sorted(prof.columns.drop("overall", errors="ignore")):
        row = {
            "cluster": int(c),
            "log_pib_pc": float(prof.at["log_pib_pc", c]) if "log_pib_pc" in prof.index else float("nan"),
            "natveg_share": float(prof.at["natveg_share", c]) if "natveg_share" in prof.index else float("nan"),
            "log_trator_per_est": float(prof.at["log_trator_per_est", c]) if "log_trator_per_est" in prof.index else float("nan"),
            "share_est_at": float(prof.at["share_est_at", c]) if "share_est_at" in prof.index else float("nan"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)

    # Heuristic naming:
    # - C2_extensive_vulnerable_frontier: highest natveg_share AND lowest log_pib_pc
    # - C0_agroindustrial_intensive: highest log_trator_per_est OR share_est_at
    # - C1_developed_diversified: highest log_pib_pc with mid natveg
    df["suggested_name"] = ""
    if df["natveg_share"].notna().any():
        idx_frontier = df["natveg_share"].idxmax()
        df.loc[idx_frontier, "suggested_name"] = "C2_extensive_vulnerable_frontier"
    remaining = df[df["suggested_name"] == ""]
    if len(remaining) and remaining["log_trator_per_est"].notna().any():
        idx_intensive = remaining["log_trator_per_est"].idxmax()
        df.loc[idx_intensive, "suggested_name"] = "C0_agroindustrial_intensive"
    remaining = df[df["suggested_name"] == ""]
    if len(remaining):
        df.loc[remaining.index[0], "suggested_name"] = "C1_developed_diversified"

    return df
