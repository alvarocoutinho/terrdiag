"""
terrdiag.audit.logging — Lightweight pipeline run log.

A `RunLog` collects per-step entries in memory and can be exported
to disk as text or JSON. Functions call `run_log.record(...)` to add
an entry; this is a process-wide singleton-ish object accessed via
the module-level `run_log` instance.

Design rationale: a thin layer to avoid the user having to import Python's
logging module and configure handlers. The cost is one global object,
which is acceptable for a research library where each notebook session
is one logical "run".
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LogEntry:
    """One pipeline step record."""

    step: str
    timestamp: str
    duration_s: float
    params: dict[str, Any]
    input_hash: str | None = None
    input_rows: int | None = None
    output_hash: str | None = None
    output_rows: int | None = None
    notes: str = ""


@dataclass(slots=True)
class RunLog:
    """In-memory log of pipeline steps.

    Use the module-level instance `run_log` for the typical case;
    instantiate your own `RunLog()` only if you need parallel runs
    in the same process.
    """

    entries: list[LogEntry] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def record(
        self,
        step: str,
        duration_s: float,
        params: dict[str, Any] | None = None,
        input_hash: str | None = None,
        input_rows: int | None = None,
        output_hash: str | None = None,
        output_rows: int | None = None,
        notes: str = "",
    ) -> None:
        """Add an entry for a completed pipeline step."""
        self.entries.append(
            LogEntry(
                step=step,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                duration_s=round(duration_s, 4),
                params=params or {},
                input_hash=input_hash,
                input_rows=input_rows,
                output_hash=output_hash,
                output_rows=output_rows,
                notes=notes,
            )
        )

    def reset(self) -> None:
        """Clear all entries and reset start time."""
        self.entries.clear()
        self.started_at = time.time()

    def to_text(self) -> str:
        """Render as a human-readable text log."""
        if not self.entries:
            return "(empty run log)"
        lines = [f"# terrdiag run log — {len(self.entries)} entries", ""]
        for i, e in enumerate(self.entries, 1):
            lines.append(f"[{i:>2}] {e.timestamp}  {e.step}  ({e.duration_s}s)")
            if e.input_rows is not None:
                lines.append(f"     in:  rows={e.input_rows}  hash={e.input_hash}")
            if e.output_rows is not None:
                lines.append(f"     out: rows={e.output_rows}  hash={e.output_hash}")
            if e.params:
                trimmed = {
                    k: (v if not isinstance(v, str) or len(v) < 80 else v[:77] + "...")
                    for k, v in e.params.items()
                }
                lines.append(f"     params: {trimmed}")
            if e.notes:
                lines.append(f"     notes: {e.notes}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serializable representation."""
        return {
            "started_at": self.started_at,
            "n_entries": len(self.entries),
            "entries": [asdict(e) for e in self.entries],
        }

    def save(self, path: str | Path, format: str = "text") -> None:
        """Persist log to disk.

        Parameters
        ----------
        path : str | Path
            Output filepath.
        format : {'text', 'json'}
            Output format.
        """
        path = Path(path)
        if format == "text":
            path.write_text(self.to_text(), encoding="utf-8")
        elif format == "json":
            path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        else:
            raise ValueError(f"format must be 'text' or 'json', got {format!r}")

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"<RunLog: {len(self.entries)} entries>"


# Module-level singleton — the common case
run_log = RunLog()
