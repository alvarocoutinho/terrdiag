"""
terrdiag.io.geo — Geographic resolution.

`resolve_geo` attaches X/Y coordinates to a panel from a points source.
The points source can be:

- a CSV with columns including the entity_id column and X, Y
- a CSV where the entity_id is embedded in a popup-style 'cnpj' field
  (legacy gdf_pts format; auto-detected)
- a shapefile / GeoPackage / GeoJSON (requires geopandas extra)

Geopandas is optional. Default path works on plain DataFrames.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from terrdiag.audit.hashing import hash_dataframe
from terrdiag.audit.logging import run_log
from terrdiag.schema import PanelSchema

if TYPE_CHECKING:
    import geopandas as gpd  # noqa: F401


DEFAULT_CRS_EPSG = 5880  # SIRGAS 2000 / Brazil Polyconic

_POPUP_CNPJ_PATTERN = re.compile(r"cnpj:\s*(\d+)", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\D")


def _try_import_geopandas():
    """Best-effort import of geopandas. Returns module or None."""
    try:
        import geopandas as gpd

        return gpd
    except ImportError:
        return None


def _extract_id_from_popup(s: object) -> str | None:
    """Extract digits-only id from a popup-style cnpj field."""
    if pd.isna(s):
        return None
    m = _POPUP_CNPJ_PATTERN.search(str(s))
    if not m:
        return None
    return m.group(1).zfill(14)


def _load_points(path: str | Path, entity_id_column: str) -> pd.DataFrame:
    """Read a points source into a DataFrame with [entity_id, X, Y]."""
    path = Path(path)
    gpd = _try_import_geopandas()

    if path.suffix.lower() in (".shp", ".gpkg", ".geojson"):
        if gpd is None:
            raise ImportError(
                f"Reading {path.suffix} requires geopandas. "
                f"Install with: pip install 'terrdiag[geo]'"
            )
        gdf = gpd.read_file(path)
        if gdf.crs is None or gdf.crs.to_epsg() != DEFAULT_CRS_EPSG:
            gdf = gdf.to_crs(epsg=DEFAULT_CRS_EPSG)
        df = pd.DataFrame(
            {
                entity_id_column: gdf.get(entity_id_column, gdf.get("cnpj")),
                "X": gdf.geometry.x,
                "Y": gdf.geometry.y,
            }
        )
    elif path.suffix.lower() in (".csv", ".tsv"):
        df = pd.read_csv(path)
        if entity_id_column not in df.columns:
            if "cnpj" in df.columns:
                extracted = df["cnpj"].apply(_extract_id_from_popup)
                if extracted.notna().sum() / max(len(extracted), 1) > 0.5:
                    df[entity_id_column] = extracted
                else:
                    df[entity_id_column] = (
                        df["cnpj"]
                        .astype(str)
                        .str.replace(_DIGITS_RE, "", regex=True)
                        .str.zfill(14)
                    )
            else:
                raise ValueError(
                    f"Points CSV must contain {entity_id_column!r} or a 'cnpj' field; "
                    f"found columns: {sorted(df.columns)}"
                )
        if "X" not in df.columns or "Y" not in df.columns:
            raise ValueError(
                f"Points CSV must contain 'X' and 'Y' columns; "
                f"found: {sorted(df.columns)}"
            )
    else:
        raise ValueError(
            f"Unsupported points source format: {path.suffix}. "
            f"Use .csv, .shp, .gpkg, or .geojson."
        )

    df[entity_id_column] = df[entity_id_column].astype(str)
    return (
        df[[entity_id_column, "X", "Y"]]
        .dropna(subset=[entity_id_column])
        .drop_duplicates(entity_id_column)
    )


def resolve_geo(
    panel: pd.DataFrame,
    points_source: str | Path,
    schema: PanelSchema | None = None,
    audit: bool = True,
) -> pd.DataFrame:
    """Attach X, Y coordinates to a panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with an entity_id column.
    points_source : str | Path
        Path to a points source (CSV, .shp, .gpkg, .geojson).
    schema : PanelSchema | None
        Schema declaring `entity_id_column` and CRS. Default PanelSchema().
    audit : bool
        Emit a run-log entry.

    Returns
    -------
    pd.DataFrame
        Panel with X, Y columns added (or NaN where unmatched).
    """
    t0 = time.time()
    schema = schema or PanelSchema()
    eid = schema.entity_id_column

    if eid not in panel.columns:
        raise KeyError(
            f"Panel does not contain entity_id column {eid!r}; "
            f"adjust the schema or rename your column."
        )

    pts = _load_points(points_source, entity_id_column=eid)
    pts_map = pts.set_index(eid)[["X", "Y"]].to_dict("index")

    out = panel.copy()
    out["X"] = out[eid].map(lambda c: pts_map.get(str(c), {}).get("X"))
    out["Y"] = out[eid].map(lambda c: pts_map.get(str(c), {}).get("Y"))

    n_unique_in_panel = out[eid].nunique()
    missing_ids = out[out["X"].isna()][eid].drop_duplicates().tolist()
    n_with_geo = n_unique_in_panel - len(missing_ids)

    duration = time.time() - t0

    if audit:
        run_log.record(
            step="io.resolve_geo",
            duration_s=duration,
            params={
                "points_source": str(points_source),
                "entity_id_column": eid,
                "crs_epsg": schema.geo_crs_epsg,
            },
            input_rows=len(panel),
            output_rows=len(out),
            output_hash=hash_dataframe(out),
            notes=(
                f"{n_with_geo}/{n_unique_in_panel} unique entities georef'd; "
                f"{len(missing_ids)} unmatched"
            ),
        )

    return out
