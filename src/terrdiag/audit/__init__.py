"""
terrdiag.audit — Reproducibility utilities.

Every public pipeline function emits an audit entry capturing:
  - timestamp
  - function name and parameters
  - input hash(es)
  - output hash and shape
  - duration

The user does not need to call these manually; they fire automatically
when the audit parameter is True (default) in pipeline functions.
"""

from terrdiag.audit.hashing import hash_dataframe, hash_file
from terrdiag.audit.logging import RunLog, run_log
from terrdiag.audit.reproducibility import write_metadata

__all__ = [
    "RunLog",
    "hash_dataframe",
    "hash_file",
    "run_log",
    "write_metadata",
]
