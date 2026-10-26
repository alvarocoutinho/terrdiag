"""
terrdiag.case_studies.renovabio.muni_features — Municipal feature engineering.

Three steps from mill-level panel → analysis-ready municipal panel:

  1. attach_uai(raw, mill_confounder_path)
       Anexa as 5 colunas UAI ao raw frame, vindas do confounder base
       (que tem UAI já calculado por município).

  2. apply_recipe(raw_with_uai, recipe=IEEE_CALLABLE_RECIPE)
       Materializa as 22 features na DataFrame, aplicando os callables
       de cada CallableFeatureSpec (com transform e clip).

  3. build_muni_panel(panel_mill, raw_with_features, cd_ibge_col='cd_ibge')
       Filtra `raw_with_features` para os municípios presentes no
       painel mill-level, e adiciona `n_usinas` por município.

The output is a municipal-level DataFrame with one row per cd_ibge,
ready for terrdiag.layer1 (clustering on 22 socio-environmental features).

NEEA and vol(%) do NOT enter the municipal frame for clustering —
they are outcome variables analyzed at mill-level by Layer 2 (LISA).
The municipal panel produced here is intentionally outcome-agnostic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from terrdiag.audit.hashing import hash_dataframe
from terrdiag.audit.logging import run_log

from terrdiag.case_studies.renovabio.recipe import (
    IEEE_CALLABLE_RECIPE,
    CallableFeatureSpec,
)


# UAI columns expected (after attach_uai) — used to validate input
UAI_COLUMNS = ("uai_housing", "uai_environ", "uai_food", "uai_mobility", "uai_climate")


def attach_uai(
    raw: pd.DataFrame,
    mill_confounder_path: str | Path,
    cd_ibge_col_raw: str = "0_cd_ibge",
    cd_ibge_col_confounder: str = "cd_ibge",
    audit: bool = True,
) -> pd.DataFrame:
    """Attach UAI columns to the raw integrated database.

    The base_psm_integrada_raw.csv does not carry the UAI block. UAI is
    sourced from the legacy mill_confounder_base.csv (one row per
    municipality with `uai_housing`, `uai_environ`, etc.).

    Parameters
    ----------
    raw : pd.DataFrame
        The integrated raw database (5570 Brazilian municipalities × 165 cols).
    mill_confounder_path : str | Path
        Path to mill_confounder_base.csv with UAI columns.
    cd_ibge_col_raw : str
        Name of the cd_ibge column in `raw` (default '0_cd_ibge').
    cd_ibge_col_confounder : str
        Name of the cd_ibge column in the confounder file (default 'cd_ibge').
    audit : bool

    Returns
    -------
    pd.DataFrame
        `raw` with the 5 UAI columns appended (NaN where mun is missing
        from the confounder base — to be imputed later).
    """
    t0 = time.time()

    mc = pd.read_csv(mill_confounder_path, dtype={cd_ibge_col_confounder: str})
    uai_present = [c for c in UAI_COLUMNS if c in mc.columns]
    if not uai_present:
        raise ValueError(
            f"mill_confounder file {mill_confounder_path} has no UAI columns; "
            f"expected any of {UAI_COLUMNS}, found {sorted(mc.columns)}"
        )

    uai_subset = mc[[cd_ibge_col_confounder] + uai_present].drop_duplicates(
        cd_ibge_col_confounder
    )
    out = raw.copy()
    out[cd_ibge_col_raw] = out[cd_ibge_col_raw].astype(str)
    out = out.merge(
        uai_subset.rename(columns={cd_ibge_col_confounder: cd_ibge_col_raw}),
        on=cd_ibge_col_raw,
        how="left",
    )

    n_with_uai = int(out["uai_housing"].notna().sum())

    duration = time.time() - t0
    if audit:
        run_log.record(
            step="renovabio.attach_uai",
            duration_s=duration,
            params={
                "mill_confounder_path": str(mill_confounder_path),
                "uai_columns_attached": uai_present,
            },
            input_rows=len(raw),
            output_rows=len(out),
            output_hash=hash_dataframe(out.head(2000)),
            notes=f"{n_with_uai}/{len(out)} rows with UAI values",
        )

    return out


def apply_recipe(
    df: pd.DataFrame,
    recipe: Iterable[CallableFeatureSpec] = IEEE_CALLABLE_RECIPE,
    audit: bool = True,
) -> pd.DataFrame:
    """Materialize each feature in `recipe` as a column on `df`.

    For each CallableFeatureSpec:
      1. Call spec.compute(df) → Series of pre-transform values
      2. Apply transform if specified (log1p)
      3. Apply clip if specified
      4. Store as df[spec.name]

    Parameters
    ----------
    df : pd.DataFrame
        Input with all columns the callables expect (raw + UAI for IEEE recipe).
    recipe : iterable of CallableFeatureSpec
    audit : bool

    Returns
    -------
    pd.DataFrame
        `df` with one new column per feature in the recipe.
    """
    t0 = time.time()
    out = df.copy()
    recipe = list(recipe)
    materialized: list[str] = []
    failed: list[tuple[str, str]] = []

    for spec in recipe:
        try:
            values = spec.compute(out)
            if spec.transform == "log1p":
                values = pd.Series(np.log1p(values.clip(lower=0)), index=values.index)
            if spec.clip is not None:
                lo, hi = spec.clip
                values = values.clip(lower=lo, upper=hi)
            out[spec.name] = values
            materialized.append(spec.name)
        except KeyError as e:
            failed.append((spec.name, f"missing column: {e}"))
        except Exception as e:
            failed.append((spec.name, repr(e)))

    duration = time.time() - t0
    if audit:
        run_log.record(
            step="renovabio.apply_recipe",
            duration_s=duration,
            params={
                "recipe_size": len(recipe),
                "materialized": len(materialized),
                "failed": [f[0] for f in failed],
            },
            input_rows=len(df),
            output_rows=len(out),
            output_hash=hash_dataframe(out.head(2000)),
            notes=f"materialized {len(materialized)}/{len(recipe)} features",
        )

    if failed:
        import warnings as _w

        msg = "apply_recipe failures:\n  " + "\n  ".join(
            f"{name}: {err}" for name, err in failed
        )
        _w.warn(msg, stacklevel=2)

    return out


def build_muni_panel(
    panel_mill: pd.DataFrame,
    raw_with_features: pd.DataFrame,
    feature_columns: Iterable[str],
    cd_ibge_col_mill: str = "cd_ibge",
    cd_ibge_col_raw: str = "0_cd_ibge",
    audit: bool = True,
) -> pd.DataFrame:
    """Filter raw to municipalities present in the mill panel + add n_usinas.

    Parameters
    ----------
    panel_mill : pd.DataFrame
        Mill-level panel (after attach_cd_ibge), one row per mill.
    raw_with_features : pd.DataFrame
        Raw integrated database after `apply_recipe`. Includes all 5570
        municipalities; this function filters to those in the mill panel.
    feature_columns : iterable of str
        Columns to retain in the output (typically the 22 feature names).
    cd_ibge_col_mill : str
        cd_ibge column in panel_mill.
    cd_ibge_col_raw : str
        cd_ibge column in raw_with_features.
    audit : bool

    Returns
    -------
    pd.DataFrame
        One row per municipality with [cd_ibge, n_usinas, nm_mun, uf, *features].
    """
    t0 = time.time()
    feature_columns = list(feature_columns)

    # Count mills per municipality
    n_per_mun = (
        panel_mill.groupby(cd_ibge_col_mill).size().rename("n_usinas").reset_index()
    )
    n_per_mun[cd_ibge_col_mill] = n_per_mun[cd_ibge_col_mill].astype(str)

    # Filter raw to municipalities in panel + select relevant columns
    raw_with_features = raw_with_features.copy()
    raw_with_features[cd_ibge_col_raw] = raw_with_features[cd_ibge_col_raw].astype(str)

    # Identifier columns to preserve
    id_cols = []
    if "0_nm_mun" in raw_with_features.columns:
        id_cols.append("0_nm_mun")
    if "0_sg_uf" in raw_with_features.columns:
        id_cols.append("0_sg_uf")
    elif "0_sigla_uf" in raw_with_features.columns:
        id_cols.append("0_sigla_uf")

    keep_cols = [cd_ibge_col_raw] + id_cols + feature_columns
    keep_cols = [c for c in keep_cols if c in raw_with_features.columns]

    muni = raw_with_features[keep_cols].drop_duplicates(cd_ibge_col_raw)
    muni = muni.merge(
        n_per_mun.rename(columns={cd_ibge_col_mill: cd_ibge_col_raw}),
        on=cd_ibge_col_raw,
        how="inner",  # only keep municipalities present in panel
    )

    # Rename columns for downstream
    rename_map = {cd_ibge_col_raw: "cd_ibge"}
    if "0_nm_mun" in muni.columns:
        rename_map["0_nm_mun"] = "nm_mun"
    if "0_sg_uf" in muni.columns:
        rename_map["0_sg_uf"] = "uf"
    if "0_sigla_uf" in muni.columns:
        rename_map["0_sigla_uf"] = "uf"
    muni = muni.rename(columns=rename_map)

    # Final column order
    final = ["cd_ibge", "nm_mun", "uf", "n_usinas"] + feature_columns
    final = [c for c in final if c in muni.columns]
    out = muni[final].sort_values("cd_ibge").reset_index(drop=True)

    duration = time.time() - t0
    if audit:
        run_log.record(
            step="renovabio.build_muni_panel",
            duration_s=duration,
            params={
                "n_features": len(feature_columns),
                "cd_ibge_col_mill": cd_ibge_col_mill,
                "cd_ibge_col_raw": cd_ibge_col_raw,
            },
            input_rows=len(panel_mill),
            output_rows=len(out),
            output_hash=hash_dataframe(out),
            notes=f"{len(out)} municipalities (vs {panel_mill[cd_ibge_col_mill].nunique()} in mill panel)",
        )

    return out
