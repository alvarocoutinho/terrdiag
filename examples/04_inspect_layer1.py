"""
04_inspect_layer1.py — Diagnostic inspection of a Layer 1 result.

Designed to be run end-to-end in Colab after the cluster pipeline.
Demonstrates the new inspection API:

    - Tabular: result.profile(), result.loadings(), result.report()
    - Figures: result.plot(kind=...), helper plotting functions
    - Cluster naming via heuristic

Usage:
    python examples/04_inspect_layer1.py \\
        --panel /path/to/painel_canonico_wide.csv \\
        --crosswalk /path/to/crosswalk_centrosul.csv \\
        --raw /path/to/base_psm_integrada_raw.csv \\
        --mill-confounder /path/to/mill_confounder_base.csv \\
        --output-dir ./figs
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from terrdiag import io, layer1
from terrdiag.specs import FeatureRecipe
from terrdiag.case_studies.renovabio import (
    IEEE_CALLABLE_RECIPE,
    IEEE_CLUSTER_FEATURES,
    apply_recipe,
    attach_cd_ibge,
    attach_uai,
    build_muni_panel,
)
from terrdiag.case_studies.renovabio.make_panel import RENOVABIO_SCHEMA
from terrdiag.case_studies.renovabio.presets import RENOVABIO_OUTCOMES


def main(
    panel_path: str,
    crosswalk_path: str,
    raw_path: str,
    mill_confounder_path: str,
    output_dir: str,
) -> int:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ─── Pipeline ──────────────────────────────────────────────────────
    panel = io.load_panel(panel_path, schema=RENOVABIO_SCHEMA, outcomes=RENOVABIO_OUTCOMES)
    panel = attach_cd_ibge(panel, crosswalk_path=crosswalk_path, overrides_path="default")
    raw = pd.read_csv(raw_path, low_memory=False, dtype={"0_cd_ibge": str})
    raw = attach_uai(raw, mill_confounder_path=mill_confounder_path)
    raw = apply_recipe(raw, IEEE_CALLABLE_RECIPE)
    feat_names = [s.name for s in IEEE_CALLABLE_RECIPE]
    muni = build_muni_panel(panel, raw, feature_columns=feat_names)
    muni = layer1.preprocess(muni, IEEE_CLUSTER_FEATURES)
    muni = layer1.impute(muni, columns=feat_names, group_column="uf")
    recipe = FeatureRecipe(
        name="IEEE_v1", features=IEEE_CLUSTER_FEATURES,
        pca_components=2, k_clusters=3, random_state=42, n_bootstrap=200,
    )
    result = layer1.cluster(muni, recipe)
    muni["cluster"] = result.labels

    print("═" * 70)
    print(result.summary())
    print("═" * 70)

    # ─── Tabular ──────────────────────────────────────────────────────
    # Profile (mean per cluster, top 8 by inter-cluster range)
    prof = result.profile(muni)
    prof["range"] = prof.drop(columns="overall").max(axis=1) - prof.drop(columns="overall").min(axis=1)
    print("\n[CLUSTER PROFILE] top 8 by inter-cluster range:")
    print(prof.sort_values("range", ascending=False).head(8).drop(columns="range").round(3).to_string())

    # Loadings
    loadings = result.loadings()
    print("\n[PCA LOADINGS] sorted by |PC1|:")
    print(loadings.loc[loadings["PC1"].abs().sort_values(ascending=False).index].head(6).round(3).to_string())

    # Crosstab
    print("\n[CLUSTER × UF]")
    print(layer1.cluster_crosstab(result, muni, "uf").to_string())

    # Naming
    print("\n[NAMING SUGGESTION] (heuristic — verify against centroids)")
    print(layer1.naming_suggestion(result, muni).round(3).to_string(index=False))

    # ─── Figures ──────────────────────────────────────────────────────
    figs_saved: list[str] = []

    def _save(name: str, fig):
        path = out / f"{name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        figs_saved.append(str(path))

    _save("01_pca", result.plot(kind="pca"))
    _save("02_centroids", result.plot(kind="centroids"))
    _save("03_loadings", result.plot(kind="loadings"))
    _save("04_profile", result.plot(kind="profile", panel=muni))
    _save("05_ari", result.plot(kind="ari"))
    _save("06_silhouette", result.plot(kind="silhouette"))
    _save("07_by_uf", layer1.plot_cluster_by_group(result, muni, group_column="uf"))

    # Spatial mill map: requires aligning cluster labels with mill-level X/Y panel
    mill = panel.merge(muni[["cd_ibge", "cluster"]], on="cd_ibge", how="left")
    mill_geo = mill[mill["cluster"].notna() & mill["X"].notna()].copy()
    mill_labels = mill_geo["cluster"].astype(int).to_numpy()
    result_mill = dataclasses.replace(result, labels=mill_labels)
    _save("08_map_mills", layer1.plot_xy_map(result_mill, mill_geo))

    # Independence test + boxplot
    test = layer1.kruskal_wallis(mill_geo, mill_labels, "delta_neea")
    print(f"\n[INDEPENDENCE TEST]")
    print(f"  {test.summary()}")
    _save("09_kw_neea", test.plot(mill_geo, mill_labels))

    # ─── Report ───────────────────────────────────────────────────────
    rep = result.report(muni)
    rep["independence_neea"] = test.to_dict()
    rep["cluster_uf_crosstab"] = layer1.cluster_crosstab(result, muni, "uf").to_dict()
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(f"\nReport → {out / 'report.json'}")

    print(f"\nFigures saved ({len(figs_saved)}):")
    for f in figs_saved:
        print(f"  {f}")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", required=True)
    p.add_argument("--crosswalk", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--mill-confounder", required=True)
    p.add_argument("--output-dir", default="figs_layer1")
    args = p.parse_args()
    sys.exit(main(
        args.panel, args.crosswalk, args.raw, args.mill_confounder, args.output_dir
    ))
