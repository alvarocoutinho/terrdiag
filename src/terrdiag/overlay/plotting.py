"""
terrdiag.overlay.plotting — Figures for diagnostic cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from terrdiag.layer2.plotting import LISA_COLORS

if TYPE_CHECKING:
    from terrdiag.overlay.diagnostic import OverlayResult


# Cluster colors — stable, mirror layer1 palette
_CLUSTER_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def _figure(figsize):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ─── Heatmap ──────────────────────────────────────────────────────────────


def plot_overlay_heatmap(
    overlay: "OverlayResult",
    figsize: tuple = (10, 5),
    normalize: str | None = None,
    annotate: bool = True,
):
    """Heatmap of cell counts (cluster × LISA class)."""
    fig, ax = _figure(figsize)
    ct = overlay.counts(normalize=normalize)

    cmap = "Blues" if normalize is None else "RdBu_r"
    im = ax.imshow(ct.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns)
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels([str(x) for x in ct.index])
    ax.set_xlabel("LISA classification")
    ax.set_ylabel("Cluster")

    if annotate:
        vmax = ct.values.max()
        for i in range(len(ct.index)):
            for j in range(len(ct.columns)):
                v = ct.values[i, j]
                if normalize is None:
                    txt = f"{int(v)}" if v == int(v) else f"{v:.0f}"
                else:
                    txt = f"{v:.2f}"
                color = "white" if v > vmax * 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)

    title = f"Diagnostic cells — {overlay.outcome_name}"
    if normalize:
        title += f" (normalized by {normalize})"
    if overlay.bandwidth_metadata:
        title += f"\n[weights: {overlay.bandwidth_metadata}]"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=("share" if normalize else "count"))
    fig.tight_layout()
    return fig


# ─── Sankey-like (using matplotlib) ───────────────────────────────────────


def plot_overlay_sankey(
    overlay: "OverlayResult",
    figsize: tuple = (12, 7),
    show_ns: bool = True,
):
    """Flow diagram from clusters (left) to LISA classes (right).

    Uses a simple matplotlib implementation (no plotly/sankey deps).
    Width of each ribbon is proportional to mill count.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    fig, ax = plt.subplots(figsize=figsize)
    ct = overlay.cell_counts.copy()
    if not show_ns:
        ct = ct.drop(columns="ns", errors="ignore")

    clusters = list(ct.index)
    lisas = list(ct.columns)
    total = ct.values.sum()
    if total == 0:
        ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
        return fig

    # Layout
    pad = 0.02
    bar_w = 0.06
    x_left, x_right = 0.10, 0.90

    cluster_totals = ct.sum(axis=1)
    lisa_totals = ct.sum(axis=0)

    # Cluster y positions
    cluster_y = {}
    y = 0.0
    cluster_height = {}
    for c in clusters:
        h = float(cluster_totals[c]) / total * (1.0 - len(clusters) * pad)
        cluster_y[c] = (y, y + h)
        cluster_height[c] = h
        y += h + pad

    lisa_y = {}
    y = 0.0
    lisa_height = {}
    for l in lisas:
        h = float(lisa_totals[l]) / total * (1.0 - len(lisas) * pad)
        lisa_y[l] = (y, y + h)
        lisa_height[l] = h
        y += h + pad

    # Draw nodes
    for c in clusters:
        y0, y1 = cluster_y[c]
        rect = Rectangle((x_left - bar_w, y0), bar_w, y1 - y0,
                          color=_CLUSTER_COLORS[clusters.index(c) % len(_CLUSTER_COLORS)],
                          alpha=0.85)
        ax.add_patch(rect)
        ax.text(x_left - bar_w - 0.005, (y0 + y1) / 2, str(c),
                 ha="right", va="center", fontsize=10, weight="bold")
    for l in lisas:
        y0, y1 = lisa_y[l]
        rect = Rectangle((x_right, y0), bar_w, y1 - y0,
                          color=LISA_COLORS.get(l, "#888"), alpha=0.85)
        ax.add_patch(rect)
        ax.text(x_right + bar_w + 0.005, (y0 + y1) / 2,
                 f"{l} ({int(lisa_totals[l])})",
                 ha="left", va="center", fontsize=10, weight="bold")

    # Draw ribbons
    # Within each source/target node we'll progressively offset
    cluster_offset = {c: 0.0 for c in clusters}
    lisa_offset = {l: 0.0 for l in lisas}
    for c in clusters:
        for l in lisas:
            n = ct.loc[c, l]
            if n == 0:
                continue
            h_src = n / total * (1.0 - len(clusters) * pad)
            h_tgt = n / total * (1.0 - len(lisas) * pad)
            y_src_a = cluster_y[c][0] + cluster_offset[c]
            y_src_b = y_src_a + h_src
            y_tgt_a = lisa_y[l][0] + lisa_offset[l]
            y_tgt_b = y_tgt_a + h_tgt
            cluster_offset[c] += h_src
            lisa_offset[l] += h_tgt

            # Cubic bezier
            verts = [
                (x_left, y_src_a),
                ((x_left + x_right) / 2, y_src_a),
                ((x_left + x_right) / 2, y_tgt_a),
                (x_right, y_tgt_a),
                (x_right, y_tgt_b),
                ((x_left + x_right) / 2, y_tgt_b),
                ((x_left + x_right) / 2, y_src_b),
                (x_left, y_src_b),
                (x_left, y_src_a),
            ]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
            p = PathPatch(Path(verts, codes), facecolor=LISA_COLORS.get(l, "#888"),
                           alpha=0.45, edgecolor="none")
            ax.add_patch(p)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.invert_yaxis()
    ax.axis("off")
    title = f"Mill flow — Layer 1 cluster → Layer 2 LISA class ({overlay.outcome_name})"
    if overlay.bandwidth_metadata:
        title += f"\n[weights: {overlay.bandwidth_metadata}]"
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ─── Spatial map ──────────────────────────────────────────────────────────


def plot_overlay_map(
    overlay: "OverlayResult",
    x_column: str = "X",
    y_column: str = "Y",
    figsize: tuple = (12, 10),
    show_ns: bool = False,
    color_by: str = "lisa",
):
    """Spatial map of mills colored by overlay cell.

    color_by:
        - 'lisa'    : color by LISA class (default), shape/border by cluster
        - 'cluster' : color by cluster, shape/border by LISA class
    """
    import matplotlib.pyplot as plt

    if x_column not in overlay.mill_table.columns or y_column not in overlay.mill_table.columns:
        raise KeyError(
            f"mill_table missing {x_column!r} or {y_column!r}; "
            f"available: {list(overlay.mill_table.columns)[:10]}..."
        )

    fig, ax = plt.subplots(figsize=figsize)
    df = overlay.mill_table
    clusters = sorted(df["cluster"].unique())
    cluster_markers = ["o", "s", "^", "D", "v", "P"]

    for ci, cluster in enumerate(clusters):
        marker = cluster_markers[ci % len(cluster_markers)]
        cluster_label = (
            overlay.cluster_names.get(cluster, f"C{cluster}")
            if overlay.cluster_names else f"C{cluster}"
        )
        for cls in ["ns", "LH", "HL", "LL", "HH"]:
            if cls == "ns" and not show_ns:
                continue
            sub = df[(df["cluster"] == cluster) & (df["lisa_class"] == cls)]
            if len(sub) == 0:
                continue
            color = LISA_COLORS.get(cls, "#888")
            size = 55 if cls != "ns" else 28
            alpha = 0.85 if cls != "ns" else 0.4
            ax.scatter(
                sub[x_column], sub[y_column],
                marker=marker, c=color, s=size, alpha=alpha,
                edgecolors="black" if cls != "ns" else "white",
                linewidth=0.6 if cls != "ns" else 0.3,
                label=f"{cluster_label} × {cls} (n={len(sub)})",
            )

    ax.set_xlabel(f"{x_column}")
    ax.set_ylabel(f"{y_column}")
    ax.set_title(
        f"Mill-level diagnostic cells — {overlay.outcome_name}\n"
        f"[weights: {overlay.bandwidth_metadata}]"
    )
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8, framealpha=0.85, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ─── Spatial map with municipal geometries ────────────────────────────────


def plot_overlay_map_geo(
    overlay: "OverlayResult",
    geo_df,
    join_column: str = "cd_ibge",
    geo_join_column: str | None = None,
    x_column: str = "X",
    y_column: str = "Y",
    figsize: tuple = (12, 11),
    show_ns: bool = False,
    color_municipalities: str = "cluster",
    state_boundaries: bool = True,
    background_color: str = "#f8f8f8",
    crs: str | None = None,
):
    """Spatial map with municipal geometries as background and mills overlaid.

    Plots each municipality (polygon) colored by Layer 1 cluster (or neutral),
    then overlays mill points using cluster-as-shape and LISA-as-color.

    Parameters
    ----------
    overlay : OverlayResult
    geo_df : GeoDataFrame or path
        Geopandas GeoDataFrame containing municipal geometries, OR a path
        (str or Path) to a .geojson / .shp / .gpkg file.
    join_column : str
        Column in overlay.mill_table that holds the municipal IBGE code.
        Default 'cd_ibge'.
    geo_join_column : str | None
        Column in geo_df with the same IBGE code. If None, falls back to
        `join_column`. Useful when shapefiles use different naming
        conventions (e.g. 'id', 'cd_geocmu', 'CD_MUN').
    x_column, y_column : str
        Mill X, Y coordinates in overlay.mill_table.
    figsize : tuple
    show_ns : bool
        Show non-significant mills (default False).
    color_municipalities : {'cluster', 'none', 'mill_density'}
        How to color municipal polygons:
          - 'cluster'      : by Layer 1 cluster (default; needs cluster info)
          - 'none'         : neutral fill, no encoding
          - 'mill_density' : by number of mills per municipality
    state_boundaries : bool
        If True, dissolve municipal geometries by 'uf' and draw thicker UF
        boundaries on top (needs 'uf' column in geo_df).
    background_color : str
        Background fill color (for sea/no-data areas).
    crs : str | None
        Optional CRS string to reproject geo_df (e.g. 'EPSG:5880' to match
        the projected mill X, Y in meters). If None, geo_df's CRS is used
        unchanged.

    Returns
    -------
    matplotlib.Figure
    """
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_overlay_map_geo requires geopandas. "
            "Install with: pip install 'terrdiag[geo]'"
        ) from e

    # Load geo_df if it was passed as a path
    if isinstance(geo_df, (str, bytes)) or hasattr(geo_df, "__fspath__"):
        geo_df = gpd.read_file(geo_df)
    elif not hasattr(geo_df, "geometry"):
        raise TypeError(
            "geo_df must be a GeoDataFrame or a path to a geo file; got "
            f"{type(geo_df).__name__}"
        )

    geo_col = geo_join_column or join_column
    if geo_col not in geo_df.columns:
        raise KeyError(
            f"geo_df missing join column {geo_col!r}; available: "
            f"{list(geo_df.columns)[:8]}"
        )
    if join_column not in overlay.mill_table.columns:
        raise KeyError(
            f"overlay.mill_table missing {join_column!r}; available: "
            f"{list(overlay.mill_table.columns)[:8]}"
        )

    # Normalize IBGE codes to 7-digit strings (CSV / pandas can lose leading
    # zeros and / or read them as integers)
    geo_df = geo_df.copy()
    geo_df[geo_col] = geo_df[geo_col].astype(str).str.zfill(7)
    mills = overlay.mill_table.copy()
    mills[join_column] = mills[join_column].astype(str).str.zfill(7)

    # Reproject geo_df if a target CRS is given
    if crs is not None:
        if geo_df.crs is None:
            geo_df = geo_df.set_crs(crs)
        else:
            geo_df = geo_df.to_crs(crs)

    # Compute cluster per municipality from the overlay's mill table.
    muni_cluster = (
        mills.dropna(subset=["cluster"])
        .groupby(join_column)["cluster"]
        .agg(lambda s: int(s.mode().iloc[0]))
        .reset_index()
        .rename(columns={join_column: geo_col})
    )
    geo_df = geo_df.merge(muni_cluster, on=geo_col, how="left")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(background_color)

    # ── Layer A: municipal polygons ──
    if color_municipalities == "cluster":
        # Cluster colors aligned with layer1 palette
        cluster_palette = {
            0: "#1b9e77", 1: "#d95f02", 2: "#7570b3",
            3: "#e7298a", 4: "#66a61e", 5: "#e6ab02",
        }
        # Plot non-cluster munis first (light grey)
        no_cluster = geo_df[geo_df["cluster"].isna()]
        if len(no_cluster):
            no_cluster.plot(
                ax=ax, color="#eaeaea", edgecolor="white", linewidth=0.3,
            )
        # Plot each cluster
        for c, color in cluster_palette.items():
            sub = geo_df[geo_df["cluster"] == c]
            if len(sub) == 0:
                continue
            label = (
                overlay.cluster_names.get(c, f"C{c}")
                if overlay.cluster_names else f"C{c}"
            )
            sub.plot(
                ax=ax, color=color, edgecolor="white", linewidth=0.4,
                alpha=0.55, label=f"muni: {label}",
            )
    elif color_municipalities == "mill_density":
        # Number of mills per muni
        density = mills.groupby(join_column).size().rename("n_mills").reset_index()
        density = density.rename(columns={join_column: geo_col})
        geo_df = geo_df.merge(density, on=geo_col, how="left").fillna({"n_mills": 0})
        geo_df.plot(
            ax=ax, column="n_mills", cmap="YlOrBr",
            edgecolor="white", linewidth=0.3,
            legend=True, legend_kwds={"label": "mills per municipality"},
        )
    elif color_municipalities == "none":
        geo_df.plot(ax=ax, color="#e0e0e0", edgecolor="white", linewidth=0.3)
    else:
        raise ValueError(
            f"color_municipalities must be 'cluster', 'mill_density', or 'none'; "
            f"got {color_municipalities!r}"
        )

    # ── Layer B: optional UF boundaries ──
    if state_boundaries and "uf" in geo_df.columns:
        try:
            uf_dissolve = geo_df.dissolve(by="uf")
            uf_dissolve.boundary.plot(
                ax=ax, color="black", linewidth=1.2, alpha=0.7,
            )
        except Exception:
            pass  # silently skip if dissolve fails

    # ── Layer C: mill points ──
    clusters = sorted(mills["cluster"].dropna().unique())
    cluster_markers = ["o", "s", "^", "D", "v", "P"]

    legend_handles = []
    for ci, cluster in enumerate(clusters):
        marker = cluster_markers[ci % len(cluster_markers)]
        cluster_label = (
            overlay.cluster_names.get(int(cluster), f"C{int(cluster)}")
            if overlay.cluster_names else f"C{int(cluster)}"
        )
        for cls in ["ns", "LH", "HL", "LL", "HH"]:
            if cls == "ns" and not show_ns:
                continue
            sub = mills[(mills["cluster"] == cluster) & (mills["lisa_class"] == cls)]
            if len(sub) == 0:
                continue
            color = LISA_COLORS.get(cls, "#888")
            size = 80 if cls != "ns" else 36
            alpha = 0.95 if cls != "ns" else 0.55
            sc = ax.scatter(
                sub[x_column], sub[y_column],
                marker=marker, c=color, s=size, alpha=alpha,
                edgecolors="black" if cls != "ns" else "white",
                linewidth=0.9 if cls != "ns" else 0.4,
                label=f"{cluster_label} × {cls} (n={len(sub)})",
                zorder=10,
            )
            legend_handles.append(sc)

    ax.set_xlabel(f"{x_column}")
    ax.set_ylabel(f"{y_column}")
    ax.set_title(
        f"Mill-level diagnostic cells over municipal geometry — {overlay.outcome_name}\n"
        f"[weights: {overlay.bandwidth_metadata}]"
    )
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=7, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.20)
    fig.tight_layout()
    return fig
