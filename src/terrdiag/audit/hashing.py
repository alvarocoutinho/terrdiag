"""
terrdiag.audit.hashing — Deterministic hashing of DataFrames and files.

Two functions:
    - hash_dataframe(df): content hash of a DataFrame, ignoring index order
      but capturing column order and dtypes.
    - hash_file(path): SHA-256 hash of file bytes (for inputs).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def hash_dataframe(df: pd.DataFrame, sample_rows: int | None = None) -> str:
    """Return a short (16-char) SHA-256 hash of a DataFrame's content.

    The hash is invariant to row order (rows are sorted by all columns before
    hashing) but sensitive to column order and dtype changes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to hash.
    sample_rows : int | None
        If set, hash only the first `sample_rows` rows after sorting. Useful
        for sanity checks on very large frames. None = full content.

    Returns
    -------
    str
        16-character lowercase hex digest.

    Notes
    -----
    Sorting before hashing makes the hash invariant to operations that
    permute rows but preserve content (e.g. groupby + concat). NaN values
    are normalized to a sentinel string.
    """
    if df is None or df.empty:
        return "empty" + "0" * 11

    # Normalize: sort by all columns to make hash order-invariant
    try:
        sorted_df = df.sort_values(by=list(df.columns), kind="stable").reset_index(drop=True)
    except (TypeError, ValueError):
        # Mixed dtypes that can't be sorted — fall back to insertion order
        sorted_df = df.reset_index(drop=True)

    if sample_rows is not None:
        sorted_df = sorted_df.head(sample_rows)

    # Stable string representation: columns + dtypes + rows
    parts: list[str] = []
    parts.append("|".join(map(str, sorted_df.columns)))
    parts.append("|".join(map(str, sorted_df.dtypes.astype(str))))
    # Convert to CSV-ish for stability across pandas versions
    parts.append(sorted_df.to_csv(index=False, lineterminator="\n", na_rep="<NA>"))

    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def hash_file(path: str | Path, chunk_size: int = 65536) -> str:
    """Return SHA-256 (16-char) of file bytes.

    Parameters
    ----------
    path : str | Path
        Filesystem path.
    chunk_size : int
        Bytes per read chunk. Default 64KB.

    Returns
    -------
    str
        16-character lowercase hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]
