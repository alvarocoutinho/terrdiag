"""
terrdiag.layer1.plotting — matplotlib figures for Layer 1 diagnostics.

All functions return a matplotlib Figure for the user to display, save,
or customize.

Color palette (cluster-stable): we use a fixed Set2 ordering so that
cluster 0/1/2 always get the same color across all plots in one session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from terrdiag.layer1.cluster import ClusterResult
    from terrdiag.layer1.independence import IndependenceTest


# Colorblind-aware palette, stable ordering by cluster integer
_CLUSTER_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def _cluster_color(k: int) -> str:
    return _CLUSTER_COLORS[k % len(_CLUSTER_COLORS)]


def _figure(figsize: tuple[float, float] = (8, 6)):
    """Lazy import of matplotlib + create a figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "terrdiag.layer1.plotting requires matplotlib. "
            "Install with: pip install matplotlib (or 'pip install terrdiag')."
        ) from e
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ─── Plots ────────────────────────────────────────────────────────────────


def plot_pca_scatter(
    result: "ClusterResult",
    figsize: tuple[float, float] = (9, 7),
    show_centroids: bool = True,
    show_legend: bool = True,
    title: str | None = None,
):
    """PCA scatter colored by cluster, with optional centroids overlay."""
    fig, ax = _figure(figsize)
    Xp = result.pca_components
    labels = result.labels
    ev = result.explained_variance_ratio

    for k in sorted(set(labels)):
        mask = labels == k
        ax.scatter(
            Xp[mask, 0], Xp[mask, 1],
            c=_cluster_color(k), s=42, alpha=0.75,
            edgecolors="white", linewidth=0.6,
            label=f"Cluster {k} (n={int(mask.sum())})",
        )

    if show_centroids:
        ax.scatter(
            result.centroids[:, 0], result.centroids[:, 1],
            c="black", s=200, marker="X",
            edgecolors="white", linewidth=2,
            label="Centroids", zorder=10,
        )

    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
    ax.set_title(title or f"PCA — clusters in 2D ({ev.sum():.1%} total var)")
    ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.5)
    if show_legend:
        ax.legend(loc="best", framealpha=0.85)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_centroids(
    result: "ClusterResult",
    figsize: tuple[float, float] = (7, 6),
):
    """Centroid coordinates in PCA space, annotated."""
    fig, ax = _figure(figsize)
    C = result.centroids
    ev = result.explained_variance_ratio
    for k, (x, y) in enumerate(C):
        ax.scatter(x, y, c=_cluster_color(k), s=400, marker="X",
                   edgecolors="black", linewidth=1.5)
        ax.annotate(
            f"C{k}\n({x:+.2f}, {y:+.2f})",
            xy=(x, y), xytext=(8, 8), textcoords="offset points",
            fontsize=10, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9),
        )
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
    ax.set_title("Cluster centroids in PCA space")
    ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_feature_loadings(
    result: "ClusterResult",
    top_n: int | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Heatmap of PCA loadings per feature (sorted by PC1 absolute value)."""
    import matplotlib.pyplot as plt

    from terrdiag.layer1.inspect import feature_loadings as _fl

    df = _fl(result)
    df = df.loc[df["PC1"].abs().sort_values(ascending=False).index]
    if top_n:
        df = df.head(top_n)

    if figsize is None:
        figsize = (5 + 0.3 * df.shape[1], 0.32 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=figsize)

    vmax = float(np.abs(df.values).max())
    im = ax.imshow(df.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns)
    for i in range(len(df)):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > vmax * 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="loading")
    ax.set_title("PCA feature loadings (sorted by |PC1|)")
    fig.tight_layout()
    return fig


def plot_cluster_profile(
    result: "ClusterResult",
    panel: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
    standardize: bool = True,
):
    """Heatmap of mean feature values by cluster.

    By default values are standardized (z-score across clusters) so each
    feature's row spans the same range — making patterns visually comparable.
    Set standardize=False for raw means.
    """
    import matplotlib.pyplot as plt

    from terrdiag.layer1.inspect import cluster_profile as _cp

    prof = _cp(result, panel, statistic="mean").drop(columns="overall", errors="ignore")
    if standardize:
        # z-score per feature (row)
        prof = prof.sub(prof.mean(axis=1), axis=0).div(
            prof.std(axis=1).replace(0, np.nan), axis=0
        )

    if figsize is None:
        figsize = (1.5 + 0.6 * prof.shape[1], 0.3 * len(prof) + 1.5)
    fig, ax = plt.subplots(figsize=figsize)
    vmax = float(np.abs(prof.values).max()) if standardize else None
    cmap = "RdBu_r" if standardize else "viridis"
    im = ax.imshow(
        prof.values, cmap=cmap, aspect="auto",
        vmin=-vmax if vmax else None, vmax=vmax,
    )
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels(prof.index)
    ax.set_xticks(range(prof.shape[1]))
    ax.set_xticklabels([f"C{c}" for c in prof.columns])
    for i in range(len(prof)):
        for j in range(prof.shape[1]):
            v = prof.values[i, j]
            color = "white" if (standardize and abs(v) > (vmax or 0) * 0.55) else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="z-score" if standardize else "mean")
    ax.set_title(f"Cluster feature profile ({'standardized' if standardize else 'mean'})")
    fig.tight_layout()
    return fig


def plot_ari_bootstrap(
    result: "ClusterResult",
    figsize: tuple[float, float] = (8, 5),
):
    """Histogram of bootstrap ARI values with mean and CI95 markers."""
    fig, ax = _figure(figsize)
    ari = result.ari_bootstrap
    ax.hist(ari, bins=30, color="#999999", edgecolor="white", alpha=0.85)
    mean = float(ari.mean())
    lo = float(np.percentile(ari, 2.5))
    hi = float(np.percentile(ari, 97.5))
    ax.axvline(mean, color="#d95f02", lw=2.2, label=f"mean = {mean:.3f}")
    ax.axvline(lo, color="#1b9e77", lw=1.5, linestyle="--", label=f"CI95 lo = {lo:.3f}")
    ax.axvline(hi, color="#1b9e77", lw=1.5, linestyle="--", label=f"CI95 hi = {hi:.3f}")
    ax.set_xlabel("Adjusted Rand Index")
    ax.set_ylabel("count")
    ax.set_title(f"Bootstrap stability (n={len(ari)} resamples)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_silhouette_profile(
    result: "ClusterResult",
    figsize: tuple[float, float] = (8, 5),
):
    """Horizontal silhouette profile per cluster (the classic sklearn-style plot)."""
    try:
        from sklearn.metrics import silhouette_samples
    except ImportError as e:
        raise ImportError("requires scikit-learn") from e

    fig, ax = _figure(figsize)
    sil = silhouette_samples(result.pca_components, result.labels)
    y_lower = 10
    for k in sorted(set(result.labels)):
        sk = np.sort(sil[result.labels == k])
        y_upper = y_lower + len(sk)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, sk,
            facecolor=_cluster_color(k), edgecolor=_cluster_color(k), alpha=0.85,
        )
        ax.text(-0.04, (y_lower + y_upper) / 2, f"C{k}", va="center", fontsize=10)
        y_lower = y_upper + 10
    ax.axvline(result.silhouette, color="#d95f02", lw=2,
               label=f"mean = {result.silhouette:.3f}")
    ax.set_xlabel("silhouette coefficient")
    ax.set_yticks([])
    ax.set_title("Silhouette profile by cluster")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_cluster_by_group(
    result: "ClusterResult",
    panel: pd.DataFrame,
    group_column: str = "uf",
    figsize: tuple[float, float] | None = None,
    normalize: bool = False,
):
    """Stacked bar chart of cluster composition within each group value."""
    import matplotlib.pyplot as plt

    from terrdiag.layer1.inspect import cluster_crosstab

    ct = cluster_crosstab(result, panel, group_column, normalize=normalize)
    if figsize is None:
        figsize = (1.2 + 0.6 * len(ct), 4.5)
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(ct))
    for k in ct.columns:
        ax.bar(
            ct.index.astype(str), ct[k].values, bottom=bottom,
            color=_cluster_color(k), label=f"C{k}", edgecolor="white", linewidth=1,
        )
        bottom += ct[k].values
    ylabel = "share" if normalize else "n municipalities"
    ax.set_ylabel(ylabel)
    ax.set_xlabel(group_column)
    ax.set_title(f"Cluster composition by {group_column}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_xy_map(
    result: "ClusterResult",
    panel: pd.DataFrame,
    x_column: str = "X",
    y_column: str = "Y",
    figsize: tuple[float, float] = (9, 8),
    point_size: int = 38,
):
    """Spatial scatter (X, Y) colored by cluster.

    For Brazilian projected CRS (EPSG:5880) the units are meters.
    Does not draw a basemap; for a styled cartographic map, pair with
    geopandas/contextily outside terrdiag.
    """
    fig, ax = _figure(figsize)
    if x_column not in panel.columns or y_column not in panel.columns:
        raise KeyError(f"panel must contain {x_column!r} and {y_column!r}")

    for k in sorted(set(result.labels)):
        mask = result.labels == k
        if mask.sum() == 0:
            continue
        ax.scatter(
            panel.iloc[mask][x_column], panel.iloc[mask][y_column],
            c=_cluster_color(k), s=point_size, alpha=0.75,
            edgecolors="white", linewidth=0.5,
            label=f"C{k} (n={int(mask.sum())})",
        )
    ax.set_xlabel(f"{x_column} (m, EPSG:5880)")
    ax.set_ylabel(f"{y_column} (m, EPSG:5880)")
    ax.set_title("Spatial distribution of clusters")
    ax.set_aspect("equal")
    ax.legend(loc="best", framealpha=0.85)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_outcome_by_cluster(
    test: "IndependenceTest",
    panel: pd.DataFrame,
    labels: np.ndarray,
    figsize: tuple[float, float] = (8, 5),
):
    """Boxplot of outcome by cluster, with Kruskal-Wallis annotation."""
    fig, ax = _figure(figsize)
    if test.outcome_name not in panel.columns:
        raise KeyError(f"{test.outcome_name!r} not in panel")
    if len(panel) != len(labels):
        raise ValueError("panel/labels length mismatch")

    data, positions, colors = [], [], []
    for k in sorted(set(labels)):
        mask = (labels == k) & panel[test.outcome_name].notna()
        if mask.sum() == 0:
            continue
        data.append(panel.loc[mask, test.outcome_name].to_numpy())
        positions.append(k)
        colors.append(_cluster_color(k))

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"C{k}\n(n={int((labels == k).sum())})" for k in positions])
    ax.set_ylabel(test.outcome_name)
    ax.set_title(
        f"{test.outcome_name} by cluster — Kruskal-Wallis "
        f"H={test.statistic:.2f}, p={test.p_value:.4f}"
    )
    ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
