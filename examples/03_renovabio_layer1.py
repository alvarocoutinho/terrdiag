"""
03_renovabio_layer1.py — RenovaBio: panel → cd_ibge → muni features → Layer 1 → independence.

Demonstrates the full RenovaBio pipeline from the canonical panel
through K-means territorial typology with the NEEA guard-rail and
post-hoc Kruskal-Wallis independence test.

Usage:
    python examples/03_renovabio_layer1.py \\
        --panel /path/to/painel_canonico_wide.csv \\
        --crosswalk /path/to/crosswalk_centrosul.csv \\
        --raw /path/to/base_psm_integrada_raw.csv \\
        --mill-confounder /path/to/mill_confounder_base.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from terrdiag import audit, io, layer1
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
) -> int:
    audit.run_log.reset()
    print("═" * 70)
    print("RenovaBio Layer 1 pipeline")
    print("═" * 70)

    # 1. Load canonical panel
    print("\n[1] Load panel ......................................")
    panel = io.load_panel(panel_path, schema=RENOVABIO_SCHEMA, outcomes=RENOVABIO_OUTCOMES)
    print(f"    → {panel['cnpj_clean'].nunique()} mills, {len(panel)} obs")

    # 2. Bridge cd_ibge
    print("\n[2] Attach cd_ibge .................................")
    panel = attach_cd_ibge(panel, crosswalk_path=crosswalk_path, overrides_path="default")
    print(f"    → resolved: {panel['cd_ibge'].notna().sum()}/{len(panel)}")
    print(f"    → unique municipalities: {panel['cd_ibge'].nunique()}")

    # 3. Raw + UAI
    print("\n[3] Load raw + attach UAI ..........................")
    raw = pd.read_csv(raw_path, low_memory=False, dtype={"0_cd_ibge": str})
    raw = attach_uai(raw, mill_confounder_path=mill_confounder_path)
    print(f"    → raw munis: {len(raw)}, UAI coverage: {raw['uai_housing'].notna().sum()}")

    # 4. Apply recipe
    print("\n[4] Apply IEEE recipe (22 features) ................")
    raw = apply_recipe(raw, IEEE_CALLABLE_RECIPE)
    feat_names = [s.name for s in IEEE_CALLABLE_RECIPE]

    # 5. Build muni panel
    print("\n[5] Build municipal panel ..........................")
    muni = build_muni_panel(panel, raw, feature_columns=feat_names)
    print(f"    → {len(muni)} municipalities × {muni.shape[1]} cols")
    miss_per_feat = muni[feat_names].isna().mean().sort_values(ascending=False)
    if (miss_per_feat > 0).any():
        print(f"    → top features by missingness:")
        for n, v in miss_per_feat.head(5).items():
            print(f"        {n}: {v:.1%}")

    # 6. Preprocess + impute
    print("\n[6] Preprocess + impute (median by uf) .............")
    muni = layer1.preprocess(muni, IEEE_CLUSTER_FEATURES)
    muni = layer1.impute(muni, columns=feat_names, group_column="uf")
    assert muni[feat_names].isna().sum().sum() == 0, "missing values remain"

    # 7. Cluster (canonical params)
    print("\n[7] Cluster (K=3, PCA=2, random_state=42) ..........")
    recipe = FeatureRecipe(
        name="IEEE_v1",
        features=IEEE_CLUSTER_FEATURES,
        impute_group_column="uf",
        scaler="standard",
        pca_components=2,
        k_clusters=3,
        random_state=42,
        n_bootstrap=200,
    )
    result = layer1.cluster(muni, recipe)
    print(f"    → {result.summary()}")

    # 8. Independence test (post-hoc; NEEA passed SEPARATELY)
    print("\n[8] Kruskal-Wallis: cluster × ΔNEEA (post-hoc) .....")
    muni["cluster"] = result.labels
    mill = panel.merge(muni[["cd_ibge", "cluster"]], on="cd_ibge", how="left")
    test = layer1.kruskal_wallis(mill, mill["cluster"].to_numpy(), "delta_neea")
    print(f"    → {test.summary()}")
    print(f"    → n per cluster: {test.n_per_cluster}")
    print(f"    → medians ΔNEEA: " + ", ".join(
        f"C{k}={v:+.2f}" for k, v in test.cluster_medians.items()))

    # 9. Cluster × UF distribution (sanity)
    print("\n[9] Cluster × UF distribution ......................")
    ct = pd.crosstab(muni["uf"], muni["cluster"])
    print(ct.to_string())

    # 10. Audit log
    print("\n[10] Audit log ......................................")
    for e in audit.run_log.entries:
        print(f"     [{e.duration_s:.2f}s] {e.step}")

    print("\n✓ pipeline complete")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", required=True, help="painel_canonico_wide.csv")
    p.add_argument("--crosswalk", required=True, help="crosswalk_centrosul.csv")
    p.add_argument("--raw", required=True, help="base_psm_integrada_raw.csv")
    p.add_argument("--mill-confounder", required=True, help="mill_confounder_base.csv (for UAI)")
    args = p.parse_args()
    sys.exit(main(args.panel, args.crosswalk, args.raw, args.mill_confounder))
