"""
terrdiag.overlay.diagnostic — Cross Layer 1 (municipal cluster) with Layer 2 (mill LISA).

The two-layer framework's payoff: each mill belongs simultaneously to a
territorial typology (Layer 1, derived from municipal socioeconomic features)
AND a local spatial regime (Layer 2, derived from neighbor dynamics of the
outcome variable). The Cartesian product (cluster × LISA class) defines the
diagnostic cells of the framework.

Cells with high count represent typical configurations; low-count cells
indicate exceptions worth narrating. The cell SP_intensive×LL (developed
densely-milled territory in cold spot) is conceptually different from
GO_frontier×HH (frontier territory in hot spot), even though both are
"interesting" — their interpretations are inverse.

The diagnostic_cells function aligns mills with their municipalities and
produces:
  - 3 × 5 count matrix (cluster × LISA class)
  - mill-level table with both classifications attached
  - inspection helpers and plots
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from terrdiag.audit.hashing import hash_dataframe
from terrdiag.audit.logging import run_log


# ─── Result container ──────────────────────────────────────────────────────


@dataclass(slots=True)
class OverlayResult:
    """Result of cross-tabulating Layer 1 cluster × Layer 2 LISA class.

    Attributes
    ----------
    cell_counts : pd.DataFrame
        Rows = cluster IDs, columns = LISA classes (HH/LL/HL/LH/ns).
        Cells = count of mills in that combination.
    mill_table : pd.DataFrame
        Long format: one row per mill, with cluster, LISA class, outcome.
        Includes any identifier columns from the original panel.
    cluster_names : dict[int, str] | None
        Optional semantic naming (e.g. {0: 'C0_intensive', ...}).
    outcome_name : str
        Name of the Layer 2 outcome (e.g. 'delta_neea').
    bandwidth_metadata : dict
        Method/bandwidth used in Layer 2 for traceability.
    n_mills : int
    n_clusters : int

    Methods
    -------
    summary()             one-line text summary
    counts(normalize=...) cell counts (raw or normalized)
    cell(name, lisa)      mills in a specific cell
    plot(kind=...)        figures
    report()              JSON-serializable dict
    """

    cell_counts: pd.DataFrame
    mill_table: pd.DataFrame
    cluster_names: dict | None
    outcome_name: str
    bandwidth_metadata: dict
    n_mills: int
    n_clusters: int

    def summary(self) -> str:
        ct = self.cell_counts
        sig_cells = (ct.drop(columns=["ns"], errors="ignore")).sum().sum()
        return (
            f"OverlayResult: {self.n_mills} mills × {self.n_clusters} clusters × "
            f"5 LISA classes, outcome={self.outcome_name}, "
            f"{sig_cells} mills in significant LISA classes "
            f"({sig_cells / self.n_mills:.1%}). "
            f"Weights: {self.bandwidth_metadata}"
        )

    def counts(self, normalize: str | None = None) -> pd.DataFrame:
        """Cell counts. normalize: None | 'cluster' (row) | 'lisa' (col) | 'all'."""
        ct = self.cell_counts.copy()
        if normalize == "cluster":
            return ct.div(ct.sum(axis=1), axis=0)
        if normalize == "lisa":
            return ct.div(ct.sum(axis=0), axis=1)
        if normalize == "all":
            return ct / ct.values.sum()
        return ct

    def cell(self, cluster: int | str, lisa_class: str) -> pd.DataFrame:
        """Return mills belonging to a specific (cluster, LISA class) cell."""
        df = self.mill_table
        if isinstance(cluster, str) and self.cluster_names is not None:
            inv = {v: k for k, v in self.cluster_names.items()}
            if cluster in inv:
                cluster = inv[cluster]
        return df[(df["cluster"] == cluster) & (df["lisa_class"] == lisa_class)]

    def to_dict(self) -> dict:
        return {
            "n_mills": int(self.n_mills),
            "n_clusters": int(self.n_clusters),
            "outcome_name": self.outcome_name,
            "bandwidth_metadata": self.bandwidth_metadata,
            "cell_counts": self.cell_counts.to_dict(),
            "cluster_names": self.cluster_names,
        }

    def plot(self, kind: str = "heatmap", **kwargs):
        """Diagnostic figures.

        Available kinds:
          - 'heatmap' : matrix of cell counts (rows × cols, colored by count)
          - 'sankey'  : flow diagram from clusters to LISA classes
          - 'map'     : spatial scatter colored by cell identity (no basemap)
          - 'map_geo' : same as 'map' with municipal polygons as background;
                        requires geopandas and a geo_df=... argument
        """
        from terrdiag.overlay import plotting
        dispatch = {
            "heatmap": plotting.plot_overlay_heatmap,
            "sankey": plotting.plot_overlay_sankey,
            "map": plotting.plot_overlay_map,
            "map_geo": plotting.plot_overlay_map_geo,
        }
        if kind not in dispatch:
            raise ValueError(
                f"Unknown plot kind: {kind!r}. Available: {sorted(dispatch.keys())}"
            )
        return dispatch[kind](self, **kwargs)


# ─── Core function ────────────────────────────────────────────────────────


def diagnostic_cells(
    layer1_result,
    layer2_result,
    layer1_panel: pd.DataFrame,
    layer2_panel: pd.DataFrame,
    join_column: str = "cd_ibge",
    cluster_names: dict | None = None,
    bandwidth_metadata: dict | None = None,
    audit: bool = True,
) -> OverlayResult:
    """Cross-tabulate Layer 1 cluster with Layer 2 LISA classification.

    Parameters
    ----------
    layer1_result : terrdiag.layer1.ClusterResult
        Output of `layer1.cluster()`. Provides one cluster label per municipality.
    layer2_result : terrdiag.layer2.LISAResult
        Output of `layer2.local_moran()`. Provides one classification per mill.
    layer1_panel : pd.DataFrame
        Municipal panel that fed `layer1.cluster()`. Must contain `join_column`.
        Length must match layer1_result.labels.
    layer2_panel : pd.DataFrame
        Mill panel that fed `layer2.local_moran()`. Must contain `join_column`
        plus identifier columns (cnpj_clean, emissor, uf are conventional).
        Length must match layer2_result.classifications.
    join_column : str
        Column linking mills to their municipality. Default 'cd_ibge'.
    cluster_names : dict[int, str] | None
        Optional semantic naming for clusters, e.g.
        {0: 'C0_agroindustrial_intensive', 1: 'C1_developed_diversified',
         2: 'C2_extensive_vulnerable_frontier'}
    bandwidth_metadata : dict | None
        Optional metadata stored in result for traceability
        (e.g. {'method': 'knn', 'k': 6}).
    audit : bool

    Returns
    -------
    OverlayResult
    """
    t0 = time.time()

    # Validate
    if len(layer1_panel) != len(layer1_result.labels):
        raise ValueError(
            f"layer1_panel ({len(layer1_panel)}) and layer1_result.labels "
            f"({len(layer1_result.labels)}) have mismatched lengths"
        )
    if len(layer2_panel) != len(layer2_result.classifications):
        raise ValueError(
            f"layer2_panel ({len(layer2_panel)}) and layer2_result.classifications "
            f"({len(layer2_result.classifications)}) have mismatched lengths"
        )
    if join_column not in layer1_panel.columns:
        raise KeyError(f"layer1_panel missing {join_column!r}")
    if join_column not in layer2_panel.columns:
        raise KeyError(f"layer2_panel missing {join_column!r}")

    # Build Layer 1 muni → cluster lookup
    muni_clusters = pd.DataFrame({
        join_column: layer1_panel[join_column].to_numpy(),
        "cluster": layer1_result.labels,
    })

    # Build Layer 2 mill-level frame with LISA class
    mills = layer2_panel.copy()
    mills["lisa_class"] = layer2_result.classifications
    mills["local_I"] = layer2_result.local_I
    mills["p_value"] = layer2_result.p_values
    mills["outcome_value"] = layer2_result.z_attribute  # standardized

    # Merge cluster onto mills
    mills = mills.merge(muni_clusters, on=join_column, how="left")
    n_orphan = mills["cluster"].isna().sum()
    if n_orphan > 0:
        import warnings as _w
        _w.warn(
            f"{n_orphan} mills could not be matched to a Layer 1 cluster "
            f"(mill's {join_column} not in Layer 1 panel). They will be dropped.",
            stacklevel=2,
        )
        mills = mills.dropna(subset=["cluster"]).reset_index(drop=True)
    mills["cluster"] = mills["cluster"].astype(int)

    # 3 × 5 cell count matrix
    cell_counts = pd.crosstab(mills["cluster"], mills["lisa_class"])
    # Ensure all 5 LISA classes present
    for cls in ["HH", "LL", "HL", "LH", "ns"]:
        if cls not in cell_counts.columns:
            cell_counts[cls] = 0
    cell_counts = cell_counts[["HH", "LL", "HL", "LH", "ns"]]
    cell_counts.index.name = "cluster"

    # Index labeling with semantic names if provided
    if cluster_names is not None:
        cell_counts = cell_counts.rename(index=cluster_names)

    result = OverlayResult(
        cell_counts=cell_counts,
        mill_table=mills,
        cluster_names=cluster_names,
        outcome_name=layer2_result.attribute_name,
        bandwidth_metadata=bandwidth_metadata or {},
        n_mills=len(mills),
        n_clusters=len(set(mills["cluster"])),
    )

    duration = time.time() - t0
    if audit:
        run_log.record(
            step="overlay.diagnostic_cells",
            duration_s=duration,
            params={
                "join_column": join_column,
                "outcome_name": result.outcome_name,
                "n_orphan_mills": int(n_orphan),
                "bandwidth_metadata": bandwidth_metadata,
            },
            input_rows=len(layer2_panel),
            output_rows=len(mills),
            output_hash=hash_dataframe(cell_counts),
            notes=result.summary(),
        )

    return result
