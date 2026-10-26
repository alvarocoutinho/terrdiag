"""
terrdiag.case_studies — Reference data-preparation pipelines for known policy programs.

These subpackages are NOT part of the generic terrdiag library API. They live
inside the package so they can be imported after a normal `pip install terrdiag`,
but they encode case-specific logic (file formats, geography filters, reconciliation
rules). For the canonical analysis API, use `terrdiag.io`, `terrdiag.schema`, etc.

Available case studies:
    - renovabio: Brazilian biofuel CBIO program (ANP certification snapshots)

Usage:
    from terrdiag.case_studies.renovabio import make_panel
    wide = make_panel.run(...)
"""

__all__: list[str] = ["renovabio"]
