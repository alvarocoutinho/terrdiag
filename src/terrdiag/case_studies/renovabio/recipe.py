"""
terrdiag.case_studies.renovabio.recipe — Feature engineering recipe.

Two artifacts:

  IEEE_CALLABLE_RECIPE
        22 CallableFeatureSpec entries. Each spec encodes how to compute
        one feature from the raw integrated database (base_psm_integrada_raw.csv).
        Uses helpers `safe_div`, `safe_log1p`, `to_num` ported from
        terrdiag_alpha v0.4 (frozen logic).

  IEEE_CLUSTER_FEATURES
        22 terrdiag.specs.FeatureSpec entries (frozen dataclasses).
        Each points to the SAME 22 columns after `apply_recipe` materializes
        them on a DataFrame. This is the spec consumed by Layer 1 (transform,
        clip, missing threshold).

The split exists because v0.4's compute callables are RenovaBio-specific
(they reference '5_num_est_total', '14_area_total', etc.) and don't belong
in the generic library. The downstream FeatureSpec entries are generic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from terrdiag.specs import FeatureSpec


# ─── Numeric helpers (ported verbatim from terrdiag_alpha v0.4) ────────────


def to_num(s) -> pd.Series:
    """Coerce to numeric, handling Brazilian comma decimals."""
    if hasattr(s, "str"):
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def safe_div(num, den) -> pd.Series:
    """Element-wise division returning NaN where denominator is ≤ 0."""
    n = to_num(num)
    d = to_num(den)
    return pd.Series(np.where(d > 0, n / d, np.nan), index=n.index)


def safe_log1p(s) -> pd.Series:
    """log1p with negative-value protection."""
    v = to_num(s).clip(lower=0)
    return pd.Series(np.log1p(v), index=v.index)


# ─── CallableFeatureSpec (case-specific, has compute callable) ─────────────


@dataclass(frozen=True)
class CallableFeatureSpec:
    """A feature with a compute() callable. RenovaBio-specific.

    For the generic Layer 1 FeatureSpec (which only references an existing
    column), see terrdiag.specs.FeatureSpec.
    """

    name: str
    label: str
    dimension: str
    compute: Callable[[pd.DataFrame], pd.Series]
    transform: Optional[str] = None
    clip: Optional[tuple[float, float]] = None
    description: Optional[str] = None


# ─── IEEE_CALLABLE_RECIPE — 22 features, ported from terrdiag_alpha v0.4 ───


IEEE_CALLABLE_RECIPE: list[CallableFeatureSpec] = [
    # ─── A. Economy and scale ──────────────────────────────────────────────
    CallableFeatureSpec(
        name="log_pib_pc", label="GDP pc (log)", dimension="Economy",
        compute=lambda df: to_num(df["1_pib_percap"]),
        transform="log1p",
        description="Log of GDP per capita (R$/year)",
    ),
    CallableFeatureSpec(
        name="log_densidade_pop", label="Pop Density (log)", dimension="Economy",
        compute=lambda df: safe_div(df["2_pop_2017_ibge"], df["14_area_total"]),
        transform="log1p",
        description="Log of population density (people/km²)",
    ),
    CallableFeatureSpec(
        name="share_vadc_ind", label="Industry/GVA", dimension="Economy",
        compute=lambda df: safe_div(df["1_vadc_ind"], df["1_vadc_bruto"]),
        description="Industrial share of gross value added",
    ),
    CallableFeatureSpec(
        name="share_vadc_serv", label="Services/GVA", dimension="Economy",
        compute=lambda df: safe_div(df["1_vadc_serv"], df["1_vadc_bruto"]),
        description="Services share of gross value added",
    ),
    # ─── B. Land structure and labor ───────────────────────────────────────
    CallableFeatureSpec(
        name="log_pess_ocup_per_est", label="Workers/Farm (log)", dimension="Land",
        compute=lambda df: safe_div(df["9_num_pess_ocup_total"], df["5_num_est_total"]),
        transform="log1p",
        description="Log of average workers per agricultural establishment",
    ),
    CallableFeatureSpec(
        name="share_est_af", label="Family Farms (%)", dimension="Land",
        compute=lambda df: safe_div(df["5_num_est_af"], df["5_num_est_total"]),
        clip=(0.0, 1.0),
        description="Share of establishments classified as family farming",
    ),
    CallableFeatureSpec(
        name="share_area_af", label="Family Farm Area (%)", dimension="Land",
        compute=lambda df: safe_div(df["6_area_lav_af"], df["6_area_lav_total"]),
        clip=(0.0, 1.0),
        description="Share of crop area under family farming",
    ),
    CallableFeatureSpec(
        name="natveg_share", label="Native Vegetation (%)", dimension="Land",
        compute=lambda df: safe_div(df["14_vegetacao_natural"], df["14_area_total"]),
        clip=(0.0, 1.0),
        description="Share of municipal area under native vegetation",
    ),
    # ─── C. Technological capacity ─────────────────────────────────────────
    CallableFeatureSpec(
        name="log_trator_per_est", label="Tractors/Farm (log)", dimension="Technology",
        compute=lambda df: safe_div(df["11_num_trator_total"], df["5_num_est_total"]),
        transform="log1p",
        description="Log of tractor density (tractors per establishment)",
    ),
    CallableFeatureSpec(
        name="share_est_at", label="Tech Assistance (%)", dimension="Technology",
        compute=lambda df: safe_div(df["10_num_est_receb_at"], df["5_num_est_total"]),
        clip=(0.0, 1.0),
        description="Share of establishments receiving technical assistance",
    ),
    CallableFeatureSpec(
        name="share_est_fin_total", label="Rural Credit (%)", dimension="Technology",
        compute=lambda df: safe_div(df["13_num_est_fin_total"], df["5_num_est_total"]),
        clip=(0.0, 1.0),
        description="Share of establishments accessing rural credit",
    ),
    CallableFeatureSpec(
        name="share_est_energia", label="Energy Access (%)", dimension="Technology",
        compute=lambda df: to_num(
            df["7_est_com_energiapct"] if "7_est_com_energiapct" in df.columns
            else df["7_est_com_energia%"]
        ),
        clip=(0.0, 1.0),
        description="Share of establishments with electrical energy access",
    ),
    # ─── D. Social conditions ──────────────────────────────────────────────
    CallableFeatureSpec(
        name="ivs_infra", label="Infra Vulnerability", dimension="Social",
        compute=lambda df: to_num(df["17_ivs_infraestrutura_urbana"]),
        description="IVS sub-index: urban infrastructure vulnerability",
    ),
    CallableFeatureSpec(
        name="ivs_renda_t", label="Income Vulnerability", dimension="Social",
        compute=lambda df: to_num(df["17_ivs_renda_e_trabalho"]),
        description="IVS sub-index: income and labor vulnerability",
    ),
    CallableFeatureSpec(
        name="idhm_educ", label="HDI Education", dimension="Social",
        compute=lambda df: to_num(df["17_idhm_educ"]),
        description="HDI education sub-index",
    ),
    CallableFeatureSpec(
        name="gini", label="Gini", dimension="Social",
        compute=lambda df: to_num(df["17_i_gini"]),
        description="Gini index of income inequality",
    ),
    CallableFeatureSpec(
        name="share_pop_agua", label="Water Access (%)", dimension="Social",
        compute=lambda df: safe_div(df["16_populacao_atendida_agua"], df["2_pop_2017_ibge"]),
        clip=(0.0, 1.0),
        description="Share of total population with treated water access",
    ),
    # ─── E. Municipal governance (UAI) — sourced separately ────────────────
    # The base_psm_integrada_raw.csv does NOT carry UAI columns.
    # apply_recipe expects the UAI columns ('uai_housing', etc.) to be
    # already attached to the input frame (e.g. via attach_uai from
    # mill_confounder_base.csv).
    CallableFeatureSpec(
        name="uai_housing", label="UAI Housing", dimension="Governance",
        compute=lambda df: to_num(df["uai_housing"]),
        description="Urban Adaptation Index — housing dimension",
    ),
    CallableFeatureSpec(
        name="uai_environ", label="UAI Environment", dimension="Governance",
        compute=lambda df: to_num(df["uai_environ"]),
        description="Urban Adaptation Index — environmental management",
    ),
    CallableFeatureSpec(
        name="uai_food", label="UAI Food Security", dimension="Governance",
        compute=lambda df: to_num(df["uai_food"]),
        description="Urban Adaptation Index — urban food system",
    ),
    CallableFeatureSpec(
        name="uai_mobility", label="UAI Mobility", dimension="Governance",
        compute=lambda df: to_num(df["uai_mobility"]),
        description="Urban Adaptation Index — urban mobility",
    ),
    CallableFeatureSpec(
        name="uai_climate", label="UAI Climate", dimension="Governance",
        compute=lambda df: to_num(df["uai_climate"]),
        description="Urban Adaptation Index — climate impact response",
    ),
]


# ─── IEEE_CLUSTER_FEATURES — generic FeatureSpec for Layer 1 ───────────────
# Built from IEEE_CALLABLE_RECIPE: each entry points to the column name
# materialized by apply_recipe (== spec.name).

IEEE_CLUSTER_FEATURES: tuple[FeatureSpec, ...] = tuple(
    FeatureSpec(
        name=spec.name,
        source_column=spec.name,  # apply_recipe writes feature value under spec.name
        transform="identity",     # transform already applied by apply_recipe
        clip=spec.clip,
        missing_threshold=0.20,
        description=spec.description or "",
        dimension=spec.dimension,
    )
    for spec in IEEE_CALLABLE_RECIPE
)
