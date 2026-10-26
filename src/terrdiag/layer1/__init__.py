"""
terrdiag.layer1 — Entity-level territorial typology via K-means + PCA.

Public API:

  Pipeline:
    - preprocess: apply transform + clip declared in FeatureSpecs
    - impute:     fill missing values (default: median by UF)
    - cluster:    StandardScaler → PCA → KMeans + bootstrap ARI
                  WITH GUARD-RAIL against outcome variables (NEEA, vol_pct).
    - kruskal_wallis: post-hoc independence test (cluster × outcome)

  Inspection (pure functions, return DataFrames):
    - cluster_profile     mean/median by cluster per feature
    - feature_loadings    PCA loadings per feature
    - cluster_crosstab    cluster × group (e.g. UF) crosstab
    - centroid_distances  pairwise distances between centroids
    - silhouette_per_cluster
    - naming_suggestion   heuristic semantic naming for clusters

  Plotting (matplotlib figures):
    - plot_pca_scatter, plot_centroids, plot_feature_loadings,
      plot_cluster_profile, plot_ari_bootstrap, plot_silhouette_profile,
      plot_cluster_by_group, plot_xy_map, plot_outcome_by_cluster

Both ClusterResult and IndependenceTest expose helper methods:
    result.summary()                — one-line text summary
    result.profile(panel)           — DataFrame
    result.loadings()               — DataFrame
    result.plot(kind='pca'|...)     — Figure
    result.report(panel)            — dict (JSON-serializable)

    test.summary() / test.to_dict() / test.plot(panel, labels)

The guard-rail in `cluster` is non-negotiable: any feature whose name or
source_column matches /neea|vol_pct|vol_eleg|cbio/i raises ValueError.
"""

from terrdiag.layer1.cluster import ClusterResult, cluster
from terrdiag.layer1.impute import impute
from terrdiag.layer1.independence import IndependenceTest, kruskal_wallis
from terrdiag.layer1.preprocess import preprocess
from terrdiag.layer1 import inspect, plotting
from terrdiag.layer1.inspect import (
    cluster_crosstab,
    cluster_profile,
    centroid_distances,
    feature_loadings,
    naming_suggestion,
    silhouette_per_cluster,
)
from terrdiag.layer1.plotting import (
    plot_ari_bootstrap,
    plot_centroids,
    plot_cluster_by_group,
    plot_cluster_profile,
    plot_feature_loadings,
    plot_outcome_by_cluster,
    plot_pca_scatter,
    plot_silhouette_profile,
    plot_xy_map,
)

__all__ = [
    # core
    "ClusterResult",
    "IndependenceTest",
    "cluster",
    "impute",
    "kruskal_wallis",
    "preprocess",
    # submodules
    "inspect",
    "plotting",
    # inspect functions
    "cluster_profile",
    "cluster_crosstab",
    "centroid_distances",
    "feature_loadings",
    "naming_suggestion",
    "silhouette_per_cluster",
    # plotting functions
    "plot_pca_scatter",
    "plot_centroids",
    "plot_feature_loadings",
    "plot_cluster_profile",
    "plot_ari_bootstrap",
    "plot_silhouette_profile",
    "plot_cluster_by_group",
    "plot_xy_map",
    "plot_outcome_by_cluster",
]
